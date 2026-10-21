"""训练复现、配对公共材料和原始记录；不生成任何密码学秘密。"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch

from utils.sampling import balanced_partition, label_shard_partition, labels_of


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def append_record(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def derived_seed(seed, round_id, client_id):
    message = 'TRSFL-training:{}/{}/{}'.format(seed, round_id, client_id).encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:4], 'big')


def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args):
    if not args.allow_nondeterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.device == 'auto':
        name = 'cuda:{}'.format(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    else:
        name = args.device
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError("本实验设备仅支持 cpu 或 cuda:N")
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("明确请求 CUDA，但服务器 CUDA 不可用")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError("CUDA 设备编号无效")
        device = torch.device('cuda', index)
    # 所有对照组采用相同确定性规则；不支持的运算应显式报错。
    torch.use_deterministic_algorithms(not args.allow_nondeterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = not args.allow_nondeterministic
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def dataset_fingerprint(dataset):
    digest = hashlib.sha256(labels_of(dataset).tobytes())
    data = getattr(dataset, 'data', None)
    if data is not None:
        if hasattr(data, 'cpu'):
            data = data.cpu().numpy()
        digest.update(np.ascontiguousarray(data).tobytes())
    elif hasattr(dataset, 'image_bytes'):
        for image in dataset.image_bytes:
            digest.update(len(image).to_bytes(8, 'big'))
            digest.update(image)
    else:
        raise ValueError("无法建立数据内容指纹")
    return digest.hexdigest()


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_partition(partition, size, num_users):
    if set(partition) != set(range(num_users)):
        raise ValueError("划分缺少客户端")
    indices = [index for client in range(num_users) for index in partition[client]]
    if (len({len(partition[c]) for c in partition}) != 1 or len(indices) != size
            or any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(size))):
        raise ValueError("划分必须等量、不重叠并完整覆盖训练集")


def prepare_pairing(directory, dataset, initial_state, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / 'pairing.json'
    if metadata.exists():
        if json.loads(metadata.read_text(encoding='utf-8')) != config:
            raise ValueError("配对配置已存在且不同；请使用新的结果目录，禁止覆盖旧实验")
    else:
        save_json(metadata, config)
    split_path = directory / 'split.json'
    if split_path.exists():
        saved = json.loads(split_path.read_text(encoding='utf-8'))
        partition = {int(key): value for key, value in saved['clients'].items()}
    else:
        partitioner = balanced_partition if config['iid'] else label_shard_partition
        partition = partitioner(dataset, config['num_users'], config['seed'])
        labels = labels_of(dataset)
        histograms = {str(client): np.bincount(labels[indices], minlength=10).tolist()
                      for client, indices in partition.items()}
        save_json(split_path, dict(clients=partition, label_histograms=histograms,
                                   method='balanced-iid' if config['iid'] else 'label-sorted-shards-2'))
    validate_partition(partition, len(dataset), config['num_users'])
    initial_path = directory / 'initial_model.pt'
    if initial_path.exists():
        stored = torch.load(initial_path, map_location='cpu', weights_only=True)
        if (set(stored) != set(initial_state)
                or any(not torch.equal(stored[key], initial_state[key]) for key in stored)):
            raise ValueError("现有初始模型与本次种子生成结果不同；检查运行环境与配对配置")
        initial_state = stored
    else:
        torch.save(initial_state, initial_path)
    return partition, initial_state


def environment_record(device):
    versions = {}
    # 官方发行名与 import 名不同；保留旧发行名记录以便识别历史环境。
    for package in ('torch', 'torchvision', 'numpy', 'matplotlib', 'charm-crypto-framework', 'charm-crypto',
                    'pycryptodome', 'sympy', 'pyarrow', 'Pillow'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), packages=versions, device=str(device),
                torch_cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                cudnn_version=torch.backends.cudnn.version(),
                gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
