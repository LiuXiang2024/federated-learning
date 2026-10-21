"""平衡 IID / 标签排序双分片划分；覆盖全部样本，不丢弃余数。"""
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


def balanced_partition(dataset, num_users, iid, seed):
    count = len(dataset)
    divisor = num_users if iid else 2 * num_users
    if num_users <= 0 or count < divisor or count % divisor:
        raise ValueError("样本数必须能被客户端数（Non-IID 为两倍客户端数）整除；不静默丢样本")
    rng = np.random.default_rng(seed)
    if iid:
        pieces = np.split(rng.permutation(count), num_users)
        return {client: sorted(piece.tolist()) for client, piece in enumerate(pieces)}
    # 稳定标签排序后切成 2n 个等长分片，每人两个，保留相同客户端样本数。
    order = np.argsort(labels_of(dataset), kind="stable")
    shards = np.split(order, 2 * num_users)
    assignments = rng.permutation(2 * num_users).reshape(num_users, 2)
    return {client: sorted(np.concatenate([shards[index] for index in pair]).tolist())
            for client, pair in enumerate(assignments)}


# 保留旧训练代码使用的函数名；新入口显式提供种子并保存划分。
def mnist_iid(dataset, num_users, seed=1):
    return balanced_partition(dataset, num_users, True, seed)


def mnist_noniid(dataset, num_users, seed=1):
    return balanced_partition(dataset, num_users, False, seed)
