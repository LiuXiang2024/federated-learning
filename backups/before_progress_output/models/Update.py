"""沿用原有 CNN/MLP 的本地 SGD；协议模块不参与模型训练。"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = sorted(int(index) for index in idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate:
    def __init__(self, args, dataset=None, idxs=None, seed=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # 独立控制 minibatch 打乱；不受此前协议调用次数或评价次数影响。
        generator = torch.Generator()
        generator.manual_seed(args.seed if seed is None else seed)
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=True,
            generator=generator, num_workers=0)
        if len(self.ldr_train.dataset) == 0:
            raise ValueError("客户端训练数据为空")

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        loss_sum, samples_seen = 0.0, 0
        for epoch in range(self.args.local_ep):
            for batch_index, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                loss = self.loss_func(net(images), labels)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("本地训练出现非有限损失")
                loss.backward()
                optimizer.step()
                # 对全部本地 epoch 的样本损失加权，避免最后一个小批次权重偏大。
                loss_sum += loss.item() * labels.size(0)
                samples_seen += labels.size(0)
                if self.args.verbose and batch_index % 10 == 0:
                    print("Local epoch {}, batch {}, loss {:.6f}".format(epoch, batch_index, loss.item()))
        # 返回完整参数，转到 CPU 后立即复制，避免 100 份模型占用 GPU 显存。
        state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        return state, float(loss_sum / samples_seen)
