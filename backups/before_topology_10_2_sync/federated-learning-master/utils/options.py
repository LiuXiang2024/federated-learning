#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 服务器部署目标与依赖确认项见 SERVER_RUN.md。

import argparse
from pathlib import Path

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1.0, help="本实验固定全员参与")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=None, help="自动按数据集确定为 1 或 3")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    # 接入实验新增参数：默认一次运行三组，数据/输出位置均可由服务器命令指定。
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--aggregation', choices=['float', 'quantized', 'trsfl', 'all'], default='all')
    parser.add_argument('--num_es', type=int, default=4)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--encoding_scale', type=int, default=1000000)
    parser.add_argument('--max_abs', type=float, default=100.0)
    parser.add_argument('--modulus', type=lambda value: int(value, 0),
                        default=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141)
    parser.add_argument('--curve_id', type=int, default=714)
    parser.add_argument('--coefficient_bits', type=int, default=128)
    parser.add_argument('--max_message_age', type=int, default=3600, help='签密时间戳最大有效秒数')
    parser.add_argument('--data_dir', type=Path, default=root / 'data')
    parser.add_argument('--data_format', choices=['torchvision', 'parquet'], default='torchvision')
    parser.add_argument('--download', action='store_true', help='明确启用 torchvision 数据下载')
    parser.add_argument('--result_dir', type=Path, default=root / 'results')
    parser.add_argument('--protocol_path', type=Path, default=root.parent / 'TRS-FL.py')
    parser.add_argument('--device', default='auto', help='auto、cpu 或 cuda:N；显式 CUDA 不静默降级')
    parser.add_argument('--seeds', type=int, nargs='+', help='配对训练种子；省略时仅使用 --seed')
    parser.add_argument('--target_accuracy', type=float, default=None, help='预先固定的收敛准确率，单位百分比')
    parser.add_argument('--allow_nondeterministic', action='store_true', help='允许非确定性计算，需在报告中注明')
    parser.add_argument('--save_round_models', action='store_true', help='另存每轮聚合模型用于核查')
    args = parser.parse_args()
    # 共享解析器仍供旧 main_nn.py 使用，因此在这里落实默认输入通道。
    if args.num_channels is None:
        args.num_channels = 1 if args.dataset == 'mnist' else 3
    return args
