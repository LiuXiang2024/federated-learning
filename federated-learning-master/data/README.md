# MNIST 数据目录

当前训练仅使用 MNIST。--data_dir 指向本目录或服务器数据根目录，加载器识别 mnist/raw 和 mnist/MNIST/raw 两种布局；可选 Parquet 放在 mnist 中，使用 --data_format parquet。

默认不下载，只有显式 --download 才允许 torchvision 下载。官方训练集 60000 个样本仅按 IID 随机打乱后等分，20 个客户端每人 3000 个样本，保存划分索引与标签直方图供两组复用。测试集不参与本地 SGD。

本地已有 cifar 文件夹仅是留存数据，不参与当前实验，本次未删除。
