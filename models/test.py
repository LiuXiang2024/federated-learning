"""测试集只用于逐轮评价，不参与本地训练和参数更新。"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def test_img(net_g, datatest, args):
    was_training = net_g.training
    net_g.eval()
    loss_sum, correct = 0.0, 0
    if len(datatest) == 0:
        raise ValueError("评价集为空")
    try:
        for data, target in DataLoader(datatest, batch_size=args.bs, shuffle=False, num_workers=0):
            # 使用主入口确定的同一个设备，禁止评价阶段另行强制切换 CUDA。
            data, target = data.to(args.device), target.to(args.device)
            logits = net_g(data)
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("评价模型产生非有限输出")
            loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
            correct += logits.argmax(dim=1).eq(target).sum().item()
    finally:
        net_g.train(was_training)
    return float(100.0 * correct / len(datatest)), float(loss_sum / len(datatest))


@torch.no_grad()
def prediction_agreement(model, reference_model, dataset, args):
    """两份恢复模型在同一完整测试集上的预测类别一致率，返回 [0,1]。"""
    model.eval()
    reference_model.eval()
    if len(dataset) == 0:
        raise ValueError("预测一致性评价集为空")
    equal = 0
    for data, _ in DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=0):
        data = data.to(args.device)
        output, reference = model(data), reference_model(data)
        if not torch.isfinite(output).all().item() or not torch.isfinite(reference).all().item():
            raise RuntimeError("预测一致性检查出现非有限输出")
        equal += output.argmax(1).eq(reference.argmax(1)).sum().item()
    return float(equal / len(dataset))
