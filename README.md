# TRS-FL 联邦学习实验

本项目仅保留并行训练版本，评价 TRS-FL 接入后对聚合结果、准确率、损失和收敛过程的影响。

- [启动命令](启动命令.md)：服务器单轮测试、IID/Non-IID 100 轮训练、本地检查与结果重绘。
- [实验设计与运行流程](docs/实验设计与运行流程.md)：配对规则、并行调度、ES/AS 分层聚合、编码与指标口径。

## 1. 当前配置

| 项目 | 默认设置 |
|---|---|
| 数据集、模型 | MNIST、CNN，21,840 个参数 |
| 数据分布 | IID；`--noniid` 切换标签偏斜 |
| 客户端、并发数 | 50 人全员参与，最多 10 个本地训练进程 |
| ES 分组、重构门限 | 2 组，每组 25 人，门限 3 |
| 全局、本地轮数 | 每组 100 轮，每客户端每轮 3 个 epoch |
| 训练、测试 batch size | 32、128 |
| 优化器 | SGD，学习率 0.01，动量 0.5，无权重衰减或学习率调度 |
| 对照组 | float 与 trsfl，默认依次运行 |
| 重复与种子 | `--repeats 1 --seed 1` |
| 定点编码 | 尺度 1,000,000，参数绝对值界 100，超界拒绝 |
| 协议参数 | secp256k1 / Charm 714，批验系数 128 位范围，时间窗口 3,600 秒 |

IID 与 Non-IID 均完整使用 60,000 张训练图片，每人 1,200 张，完整 10,000 张测试图片用于评价。同一分布下 float 与 trsfl 共享初始模型、数据划分和训练随机安排。`--aggregation all --epochs 100` 是两组各运行 100 轮。

Non-IID 按标签排序后分成 100 片，每人随机分配 2 片。真实类别数量不同，个别分片可能跨类别边界；两个分片不表示严格两个类别。`--iid` 和 `--noniid` 互斥，省略时采用 IID。

## 2. 文件与数据

```text
F:\FL\
  README.md                     使用总览
  启动命令.md                   可复制命令
  main_fed.py                   并行训练入口
  TRS-FL.py                     签密协议
  models/、utils/               模型、聚合、数据与进程池
  data/                         MNIST 与留存的 CIFAR 数据
  results/                      新实验与历史结果
  docs/                         实验流程与整理核验记录
  test_parallel_training.py     调度和小样本训练检查
  test_noniid.py                Non-IID 划分和配对检查
  server_checks.py              编码与协议套件
  summarize_results.py          结果汇总及绘图
  requirements*.txt、LICENSE    依赖与许可证
```

默认数据、结果、协议路径均相对于代码根目录，移动整个项目后无需修改源码。原始 MNIST 支持 `data/mnist/raw` 和 `data/mnist/MNIST/raw` 两种布局；默认不下载，只有显式 `--download` 才启用 torchvision 下载。CIFAR 作为已有数据保留，不参与当前训练。

未附带数据文件时，按[启动命令中的数据准备步骤](启动命令.md#2-准备-mnist-数据)下载 MNIST。该步骤只下载/加载数据并核对样本数，不启动训练或签密协议。代码通过 torchvision 的 MNIST 下载入口获取并检查原始文件，预期训练集 60,000 张、测试集 10,000 张。

`--data_format parquet` 可读取 `data/mnist/` 下的 `train*.parquet` 与 `test*.parquet`，要求 `image.bytes` 和 `label` 列；不与 `--download` 混用，也不自动替代原始格式。

## 3. 环境与运行

`requirements.txt` 列出直接依赖，不是已验证的版本锁定文件。部署时先选用与驱动、CUDA 和 Python 匹配的 PyTorch/torchvision，再安装项目依赖。Charm 使用 `charm-crypto-framework==0.62`，需可用的 EC 后端及曲线 714；不要用旧的 `charm-crypto` 发行包替代。构建环境通常需要编译工具、GMP 与 OpenSSL 开发库。Parquet 可选依赖见 `requirements-parquet.txt`。

当前仅完成本地目录整理与代码更新；服务器旧目录未自动同步。[启动命令](启动命令.md)按代码平铺部署到 `/root/autodl-tmp/Parallel Computing` 编写，并显式复用服务器已有数据位置。运行前先同步代码，再进行单轮检查。

10 个工作进程共享一张 GPU，各有 CUDA 上下文和工作区；实际显存峰值和速度需在目标 GPU 测量。进程池按空闲情况派发任务，收齐 50 份结果后执行聚合。ES 分组与同时训练的 10 个任务没有固定对应关系。

新实验必须使用新的结果目录，已有模式目录拒绝覆盖。前台终端训练的持续运行取决于终端和服务是否仍在运行；关闭服务器或终止对应终端会影响任务。当前没有自动断点续训入口。

## 4. 结果文件与图片

自定义 `--result_dir` 是下面的 `结果根目录`；启动命令同时通过 `tee` 把日志保存在本次运行目录。

```text
结果根目录/
  mnist_cnn_iid/                 或 mnist_cnn_noniid/
    seed_1/
      paired/                   初始模型、索引、标签直方图与配对配置
      float/、trsfl/
        config.json、model_layout.json
        metrics.jsonl、status.json
        last_good_model.pt、summary.json
        failure.json            仅普通异常被捕获时生成
    summary/
      final_by_seed.csv、paired_differences.csv、consistency.csv
      statistics.json
      test_accuracy.png、test_loss.png
```

默认只保留最近提交模型，`--save_round_models` 可保存每轮模型。`summary.json` 表示本组全部轮次完成；`status.json` 可能因人工中断而保留旧状态，判断是否仍运行须结合实际进程。

图片采用白底、蓝色圆点 FedAvg、橙色方点 TRS-FL，不显示顶部标题。主图覆盖全部轮次，放大窗口取最后 20 轮，每条曲线最多标记 10 个点；100 轮时标记 82、84、…、100 轮，折线保留 81–100 轮真实数据。单轮不显示放大窗口。范围由两组数据及标准差带共同确定，未进行平滑。

重复数为 1 时不绘制标准差带，样本标准差记录为 null；多次重复绘制均值和 ±1 个样本标准差带。`summarize_results.py` 中 `ZOOM_ROUNDS` 控制局部窗口长度。IID、Non-IID 和不同代码/配置的结果分别汇总。

历史 `results/trsfl_parallel10_100rounds_GW1kv1/` 是 20 客户端的 IID 结果。它可用于查看绘图样式，不能作为当前 50 客户端或 Non-IID 实验结果。

## 5. 已验证范围

2026-10-21 本地 22 项回归检查通过：3 项 Non-IID、8 项并行、11 项编码检查。其中包含 50 客户端两轮小样本 CPU 并行/串行参照一致性、数据划分与配对复用，以及移动后默认路径检查。

这些检查不等于 50 客户端正式训练、完整签密协议套件、10 进程 CUDA 显存峰值或加速比已验证。现有历史 20 客户端结果与当前配置分开解释；指标定义和结论边界见[实验设计与运行流程](docs/实验设计与运行流程.md)。

## 6. GitHub 发布与隐私

公开版本默认包含训练源码、检查脚本、依赖、许可证及说明文档。MNIST、CIFAR 是公开数据，不应当作个人凭据；如需随仓库分发，仍应核对原数据的再分发条款并注明来源。当前保留 `data/` 忽略规则是为了减少仓库体积，同时提供独立下载方式，并非认定这些数据含个人隐私。项目代码的 LICENSE 不替代数据集的许可。

实验结果、本地迁移记录、凭据文件和私钥继续排除。公开版本可以不包含 `data/`、`results/` 和 `docs/目录整理记录.json`；这些路径由本地准备数据或运行实验后提供。GitHub 普通 Git 存储阻止超过 100 MiB 的单个文件，较大的数据或压缩包需另行选择分发方式，见 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

忽略规则不会自动取消已跟踪文件，也不会清除 Git 历史。当前本地 `.git` 的旧提交保留了历史部署文档中的服务器连接信息，并包含提交者姓名、邮箱等记录；按项目所有者的选择保留现有历史，不另建源码快照或清理历史。后续提交仍应检查新增内容中的凭据和私钥，提交时使用准备公开的署名与邮箱。当前尚未执行上传。
