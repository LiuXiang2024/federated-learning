# MNIST 数据目录

当前训练仅使用 MNIST。--data_dir 指向本目录或服务器数据根目录，加载器识别 mnist/raw 和 mnist/MNIST/raw 两种布局；可选 Parquet 放在 mnist 中，使用 --data_format parquet。

默认不下载，只有显式 --download 才允许 torchvision 下载。官方训练集 60000 个样本在 20 客户端配置下每人 3000 个；Non-IID 使用 40 个标签排序分片，每人两个。测试集不参与本地 SGD。

本地已有 cifar 文件夹仅是留存数据，不参与当前实验，本次未删除。
