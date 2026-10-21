"""只并行本地 SGD；全部客户端完成后由主进程执行协议聚合。"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import torch
import torch.multiprocessing as multiprocessing

from models.Nets import CNNMnist
from models.Update import LocalUpdate
from utils.datasets import load_datasets
from utils.experiment import derived_seed, resolve_device, set_training_seed


_worker_args = None
_worker_train = None


def initialize_worker(args):
    global _worker_args, _worker_train
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.device = resolve_device(args)
    # LegacyMNIST 是局部类，不能跨 spawn 边界传递；各进程独立加载一次。
    train, _, _ = load_datasets(args)
    _worker_args, _worker_train = args, train


def train_client(client_id, local_seed, state, indices):
    model = CNNMnist(_worker_args).to(_worker_args.device)
    model.load_state_dict(state, strict=True)
    # 构造模型会消耗随机数，必须在构造后恢复本客户端的训练随机安排。
    set_training_seed(local_seed)
    local = LocalUpdate(_worker_args, _worker_train, indices, seed=local_seed)
    weights, loss = local.train(model)
    return client_id, weights, loss


class ClientTrainingError(RuntimeError):
    def __init__(self, client_id, reason):
        self.client_id = client_id
        super().__init__('Client {} local training failed: {}'.format(client_id, reason))


def collect_results(futures, count, on_complete=None):
    if len(futures) != count or set(futures.values()) != set(range(count)):
        raise ValueError('Exactly one training task per client is required')
    states, losses = [None] * count, [None] * count
    try:
        for future in as_completed(futures):
            client_id = futures[future]
            try:
                returned_id, state, loss = future.result()
                if returned_id != client_id:
                    raise ValueError('Worker returned a different client ID')
                states[client_id], losses[client_id] = state, loss
            except Exception as error:
                raise ClientTrainingError(client_id, str(error)) from error
            if on_complete is not None:
                on_complete(client_id)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    return states, losses


class ClientTrainer:
    def __init__(self, args):
        if not 1 <= args.client_workers <= min(10, args.num_users):
            raise ValueError('client_workers must be between 1 and min(10, num_users)')
        self.args = args
        self.executor = ProcessPoolExecutor(
            max_workers=args.client_workers,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(args,))

    def train_round(self, state, partition, seed, round_id):
        if set(partition) != set(range(self.args.num_users)):
            raise ValueError('Training partition must contain every client')
        if any(tensor.device.type != 'cpu' for tensor in state.values()):
            raise ValueError('Only CPU model states may cross the process boundary')
        futures = {}
        try:
            for client_id in range(self.args.num_users):
                future = self.executor.submit(
                    train_client, client_id, derived_seed(seed, round_id, client_id),
                    state, partition[client_id])
                futures[future] = client_id
        except Exception as error:
            for future in futures:
                future.cancel()
            raise ClientTrainingError(client_id, str(error)) from error

        def progress(client_id):
            print('[{}] round {}/{} client {}/{} DONE'.format(
                self.args.aggregation, round_id, self.args.epochs,
                client_id + 1, self.args.num_users), flush=True)

        return collect_results(futures, self.args.num_users, progress)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
