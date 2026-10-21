#!/usr/bin/env python
"""固定 20 客户端/2 ES 的浮点明文与 TRS-FL 两组训练对照。"""
import copy
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from models.Nets import CNNMnist
from models.Update import LocalUpdate
from models.secure_aggregation import ModelAggregator, load_protocol
from models.test import test_img, prediction_agreement
from utils.datasets import load_datasets
from utils.experiment import (append_record, dataset_fingerprint, derived_seed,
                              environment_record, prepare_pairing, resolve_device,
                              save_json, set_training_seed, state_fingerprint)
from utils.options import args_parser


def validate_args(args):
    if args.num_users != 20 or args.num_es != 2 or args.frac != 1.0:
        raise ValueError("本次实验固定 20 客户端、2 ES、每组 10 人、全员参与")
    if not 1 <= args.threshold <= args.num_users // args.num_es:
        raise ValueError("组内门限必须在 1 至 10 之间，默认 3")
    if any(value <= 0 for value in (args.epochs, args.local_ep, args.local_bs, args.bs)):
        raise ValueError("轮数、epoch 数和 batch size 必须为正数")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.momentum) or not 0 <= args.momentum < 1:
        raise ValueError("学习率必须为正数，SGD 动量必须属于 [0,1)")
    if args.dataset != 'mnist' or args.model != 'cnn' or args.iid is not True:
        raise ValueError("本次实验仅支持 MNIST、CNN 和 IID 划分")
    if args.num_classes != 10:
        raise ValueError("MNIST 正式任务必须使用 10 类")
    expected_channels = 1
    if args.num_channels is None:
        args.num_channels = expected_channels
    if args.num_channels != expected_channels:
        raise ValueError("输入通道数与数据集不匹配")
    if args.target_accuracy is not None and not 0 <= args.target_accuracy <= 100:
        raise ValueError("收敛目标应为 [0,100] 内的百分比")
    # 重复次数与种子职责分开：重复之间改变训练安排，每次内部两组共享安排。
    # 检查完整种子范围，避免最后一次重复越界；协议随机性不使用这些种子。
    if args.repeats <= 0:
        raise ValueError("独立重复次数必须为正整数")
    if args.seed < 0 or args.seed + args.repeats > 2**32:
        raise ValueError("每次重复的训练种子必须位于 [0,2**32)")
    seeds = list(range(args.seed, args.seed + args.repeats))
    return seeds


def source_fingerprints(args):
    root = Path(__file__).resolve().parent
    files = ['main_fed.py', 'models/Nets.py', 'models/Update.py', 'models/Fed.py',
             'models/secure_aggregation.py', 'models/test.py', 'utils/sampling.py',
             'utils/experiment.py', 'utils/datasets.py', 'utils/options.py']
    result = {file: hashlib.sha256((root / file).read_bytes()).hexdigest() for file in files}
    result['protocol'] = hashlib.sha256(Path(args.protocol_path).read_bytes()).hexdigest()
    return result


def run_mode(args, seed, template, initial_state, partition, train, test, directory, pair_config):
    # 禁止覆盖已存在的模式目录；重跑使用新 result_dir，避免混入旧曲线。
    directory.mkdir(parents=True, exist_ok=False)
    round_id, client_id, phase = 0, None, 'initialization'
    records = []
    config = dict(pair_config, aggregation=args.aggregation, arguments={
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()}, environment=environment_record(args.device))
    save_json(directory / 'config.json', config)
    save_json(directory / 'status.json', dict(status='running', committed_round=0))
    try:
        aggregator = ModelAggregator(initial_state, args)
        save_json(directory / 'model_layout.json', aggregator.binding)
        net = copy.deepcopy(template).to(args.device)
        net.load_state_dict(initial_state, strict=True)
        # checkpoint 只保存已经完成全部检查的全局模型；初始模型表示提交轮次 0。
        torch.save(initial_state, directory / 'last_good_model.pt')
        # 默认逐轮、逐客户端显示进度，flush 保证管道日志即时可见。
        for round_id in range(1, args.epochs + 1):
            phase, states, local_losses = 'local_training', [], []
            for client_id in range(args.num_users):
                print('[{}] round {}/{} client {}/{}'.format(
                    args.aggregation, round_id, args.epochs, client_id + 1, args.num_users), flush=True)
                local_seed = derived_seed(seed, round_id, client_id)
                set_training_seed(local_seed)
                local = LocalUpdate(args, train, partition[client_id], seed=local_seed)
                weights, local_loss = local.train(copy.deepcopy(net).to(args.device))
                states.append(weights)
                local_losses.append(local_loss)
            client_id, phase = None, 'aggregation'
            result = aggregator.aggregate(states, round_id)
            del states
            candidate = copy.deepcopy(template).to(args.device)
            candidate.load_state_dict(result.state, strict=True)
            diagnostics = result.diagnostics
            phase = 'prediction_consistency'
            if args.aggregation == 'trsfl':
                reference = copy.deepcopy(template).to(args.device)
                reference.load_state_dict(result.reference_state, strict=True)
                agreement = prediction_agreement(candidate, reference, test, args)
                diagnostics['prediction_agreement'] = agreement
                if agreement != 1.0:
                    raise RuntimeError("协议模型与同批量化明文模型的预测不一致；本轮不提交")
                del reference
                diagnostics['consistency_passed'] = True
            else:
                # 明文模式没有协议验签，不能把未执行的检查标成通过。
                diagnostics['prediction_agreement'] = None
                diagnostics['consistency_passed'] = None
            phase = 'evaluation'
            accuracy, test_loss = test_img(candidate, test, args)
            row = dict(status='ok', dataset=args.dataset, model=args.model,
                       iid=args.iid, mode=args.aggregation, seed=seed, round=round_id,
                       train_loss=float(np.mean(local_losses)), test_accuracy=accuracy,
                       test_loss=test_loss, **diagnostics)
            # 到这里才提交下一轮模型；不会对恢复模型再执行一次额外梯度更新。
            phase = 'commit'
            temporary = directory / 'last_good_model.pt.tmp'
            torch.save(result.state, temporary)
            temporary.replace(directory / 'last_good_model.pt')
            if args.save_round_models:
                torch.save(result.state, directory / ('round_{:03d}.pt'.format(round_id)))
            net.load_state_dict(result.state, strict=True)
            append_record(directory / 'metrics.jsonl', row)
            records.append(row)
            save_json(directory / 'status.json', dict(status='running', committed_round=round_id))
            print('[{}] round {}/{} test_accuracy={:.3f}% test_loss={:.6f}'.format(
                args.aggregation, round_id, args.epochs, accuracy, test_loss), flush=True)
            del candidate, result
        target_round = next((row['round'] for row in records
                             if args.target_accuracy is not None and row['test_accuracy'] >= args.target_accuracy), None)
        save_json(directory / 'summary.json', dict(
            status='complete', seed=seed, mode=args.aggregation,
            final_accuracy=records[-1]['test_accuracy'], final_loss=records[-1]['test_loss'],
            target_accuracy=args.target_accuracy, target_round=target_round,
            target_status='not_configured' if args.target_accuracy is None else
                          ('reached' if target_round is not None else 'not_reached'),
            consistency_checked_rounds=sum(row['consistency_passed'] is True for row in records)))
        save_json(directory / 'status.json', dict(status='complete', committed_round=args.epochs))
    except Exception as error:
        failure = dict(status='failed', seed=seed, mode=args.aggregation, round=round_id,
                       client_id=client_id, phase=phase, error_type=type(error).__name__,
                       reason=str(error), committed_round=records[-1]['round'] if records else 0)
        append_record(directory / 'metrics.jsonl', failure)
        save_json(directory / 'failure.json', dict(failure, traceback=traceback.format_exc()))
        save_json(directory / 'status.json', failure)
        raise


def main():
    args = args_parser()
    seeds = validate_args(args)
    args.device = resolve_device(args)
    # 提前检查导入，使缺失协议依赖不会在数小时明文训练之后才暴露。
    modes = ['float', 'trsfl'] if args.aggregation == 'all' else [args.aggregation]
    if 'trsfl' in modes:
        load_protocol(args.protocol_path)
    train, test, preprocessing = load_datasets(args)
    expected_shape = (1, 28, 28)
    if tuple(train[0][0].shape) != expected_shape or tuple(test[0][0].shape) != expected_shape:
        raise ValueError("实际图片形状与选定数据集不匹配")
    fingerprints = dict(train=dataset_fingerprint(train), test=dataset_fingerprint(test))
    task = 'mnist_cnn_iid'
    task_dir = Path(args.result_dir).resolve() / task
    sources = source_fingerprints(args)
    runtime = environment_record(args.device)
    # 分别启动两组时也核对训练相关软件条件；不让环境变化混入配对比较。
    training_runtime = dict(python=runtime['python'], platform=runtime['platform'],
                            packages={name: runtime['packages'][name] for name in
                                      ('torch', 'torchvision', 'numpy', 'Pillow')},
                            torch_cuda_build=runtime['torch_cuda_build'],
                            cudnn_version=runtime['cudnn_version'], gpu_name=runtime['gpu_name'])
    for seed in seeds:
        set_training_seed(derived_seed(seed, 0, 0))
        template = CNNMnist(args)
        initial = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
        # 配对信息包含所有影响训练及编码的配置，跨命令复用时逐项比对。
        pair_config = dict(dataset=args.dataset, model=args.model, iid=args.iid, seed=seed,
                           num_users=args.num_users, num_es=args.num_es,
                           group_size=args.num_users // args.num_es, threshold=args.threshold,
                           epochs=args.epochs, local_ep=args.local_ep, local_bs=args.local_bs,
                           test_bs=args.bs, lr=args.lr, momentum=args.momentum,
                           weight_decay=0, lr_schedule='constant', preprocessing=preprocessing,
                           encoding_scale=args.encoding_scale, max_abs=args.max_abs,
                           modulus=str(args.modulus), curve_id=args.curve_id,
                           coefficient_bits=args.coefficient_bits, max_message_age=args.max_message_age,
                           data_format=args.data_format, fingerprints=fingerprints,
                           device=str(args.device), deterministic=not args.allow_nondeterministic,
                           target_accuracy=args.target_accuracy, architecture=str(template),
                           initial_model_sha256=state_fingerprint(initial), source_sha256=sources,
                           training_runtime=training_runtime,
                           seed_schedule='sha256(TRSFL-training:seed/round/client) first 4 bytes')
        seed_dir = task_dir / ('seed_{}'.format(seed))
        partition, initial = prepare_pairing(seed_dir / 'paired', train, initial, pair_config)
        pair_config['partition_sha256'] = hashlib.sha256(
            json.dumps(partition, sort_keys=True).encode()).hexdigest()
        for mode in modes:
            mode_args = copy.copy(args)
            mode_args.aggregation, mode_args.seed = mode, seed
            run_mode(mode_args, seed, template, initial, partition, train, test,
                     seed_dir / mode, pair_config)
    # 汇总只读取完成且配置匹配的结果；原始每轮记录始终保留。
    from summarize_results import summarize_task
    summarize_task(task_dir)


if __name__ == '__main__':
    main()
