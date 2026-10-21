"""随机打乱后等分的 IID 划分；覆盖全部样本，不丢弃余数。"""
import numpy as np


def labels_of(dataset):
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "train_labels", None)
    if targets is None:
        raise ValueError("数据集缺少 targets 标签")
    if hasattr(targets, "cpu"):
        targets = targets.cpu().numpy()
    labels = np.asarray(targets, dtype=np.int64)
    if labels.shape != (len(dataset),):
        raise ValueError("样本和标签长度不一致")
    return labels


def balanced_partition(dataset, num_users, seed):
    count = len(dataset)
    if num_users <= 0 or count < num_users or count % num_users:
        raise ValueError("样本数必须能被客户端数整除；不静默丢样本")
    rng = np.random.default_rng(seed)
    pieces = np.split(rng.permutation(count), num_users)
    return {client: sorted(piece.tolist()) for client, piece in enumerate(pieces)}


# 保留旧训练代码使用的函数名；新入口显式提供种子并保存划分。
def mnist_iid(dataset, num_users, seed=1):
    return balanced_partition(dataset, num_users, seed)
