"""可配置的服务器数据入口：torchvision 原始格式或已有 Parquet 文件。"""
from io import BytesIO
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class ParquetImages(Dataset):
    def __init__(self, directory, split, transform, grayscale):
        # 可选依赖只在明确选择 parquet 时导入；不下载、不自动切换数据源。
        import pyarrow.parquet as pq
        self.transform, self.grayscale = transform, grayscale
        files = sorted(Path(directory).rglob(split + '*.parquet'))
        if not files:
            raise FileNotFoundError("未找到 Parquet 分片: " + str(directory) + "/" + split)
        self.image_bytes, labels = [], []
        for path in files:
            table = pq.read_table(path, columns=['image', 'label'])
            for image in table.column('image').to_pylist():
                if not isinstance(image, dict) or not isinstance(image.get('bytes'), bytes):
                    raise ValueError("Parquet image 必须包含内嵌 bytes，不能依赖外部图片路径")
                self.image_bytes.append(image['bytes'])
            labels.extend(table.column('label').to_pylist())
        self.targets = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(BytesIO(self.image_bytes[index])) as source:
            image = source.convert('L' if self.grayscale else 'RGB')
        return self.transform(image), int(self.targets[index])


def load_datasets(args):
    root = Path(args.data_dir).resolve()
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        name, factory = 'mnist', datasets.MNIST
    else:
        raise ValueError("仅支持 mnist")
    if args.data_format == 'parquet':
        if args.download:
            raise ValueError("--download 只适用于 torchvision；Parquet 需提前准备")
        directory = root / 'mnist'
        train = ParquetImages(directory, 'train', transform, name == 'mnist')
        test = ParquetImages(directory, 'test', transform, name == 'mnist')
    else:
        directory = root / name
        # 当前仓库自带的 MNIST 位于 mnist/raw；新版 torchvision 通常要求
        # mnist/MNIST/raw。显式识别这一旧布局，不复制、不下载或修改原始文件。
        if name == 'mnist' and (directory / 'raw').is_dir() and not (directory / 'MNIST' / 'raw').is_dir():
            class LegacyMNIST(datasets.MNIST):
                @property
                def raw_folder(self):
                    return str(Path(self.root) / 'raw')

                @property
                def processed_folder(self):
                    return str(Path(self.root) / 'processed')
            factory = LegacyMNIST
        train = factory(str(directory), train=True, download=args.download, transform=transform)
        test = factory(str(directory), train=False, download=args.download, transform=transform)
    return train, test, repr(transform)
