# TRS-FL 联邦学习实验

本项目将完整 TRS-FL 接入真实 MNIST 本地 SGD 训练，仅使用 CNN 与 IID 划分，比较浮点 FedAvg（float）和完整协议（trsfl）。量化明文计算仅保留为同批内部检查，不再设置 quantized 独立训练组；不使用 CIFAR。

## 当前默认配置

| 项目 | 值 |
|---|---|
| 客户端 / ES / 每组人数 | 20 / 2 / 10 |
| 秘密重构门限 / 参与比例 | 3 / 100% |
| 全局轮数 / 本地 epoch / batch size | 100 / 3 / 32 |
| 学习率 / SGD 动量 | 0.01 / 0.5 |
| 独立重复次数 / 起始训练种子 | 1 / 1 |
| 模型 / 数据分布 | 固定 CNN / IID |
| 编码尺度 / 参数绝对值界 | 1000000 / 100 |

`--repeats` 控制独立重复次数；`--seed` 控制起始训练种子。默认每组运行 1 次，使用训练种子 1；两组共享初始权重、划分和本地随机安排。协议随机性独立生成。`--seeds` 已删除，不能使用旧命令中的该参数。

默认两组各 100 轮，共 200 个全局轮次。没有配置 `--target_accuracy` 时只保存完整曲线，不计算达到特定准确率的轮次；不会自动提前停止。

## 依赖与部署

主依赖见 requirements.txt，可选 Parquet 依赖见 requirements-parquet.txt。需要 PyTorch、Torchvision、NumPy、Matplotlib、Pillow，以及 Charm EC、PyCryptodome、SymPy、PyParsing。Charm 发行包名为 charm-crypto-framework，代码导入名为 charm；当前协议使用 ECGroup(714)，不使用 pairing 后端。

先确认服务器操作系统/架构、Python、GPU 驱动、CUDA、Torch/Torchvision 兼容组合及 Charm EC 编译条件，再在隔离环境安装依赖。requirements.txt 不是完整锁定环境，不能据此假定任意版本组合均运行通过。不要用缺失的历史部署记录作为当前环境证明。

上传时保持父目录的 TRS-FL.py 和本目录相对位置一致。数据支持 data_dir/mnist/raw 或 data_dir/mnist/MNIST/raw；不默认下载，只有显式 --download 才允许 torchvision 下载。已有本地 CIFAR 数据不参与本实验。

## 服务器启动示例

以下假定代码已部署到 /FL/trsfl，环境位于 /FL/envs/trsfl，MNIST 位于 /FL/data；这些是命令示例，按实际路径替换，程序本身不依赖该目录。当前本地更新没有自动上传服务器。

```bash
cd /FL/trsfl/federated-learning-master
source /FL/envs/trsfl/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p /FL/runs
RUN_DIR=$(mktemp -d /FL/runs/mnist_cnn_iid_XXXXXX)
echo "结果目录：$RUN_DIR"
set -o pipefail
python -u main_fed.py \
  --dataset mnist --model cnn --iid \
  --num_users 20 --num_es 2 --threshold 3 --frac 1.0 \
  --epochs 100 --local_ep 3 --local_bs 32 --lr 0.01 --momentum 0.5 \
  --aggregation all --repeats 1 --seed 1 \
  --encoding_scale 1000000 --max_abs 100 \
  --device cuda:0 --data_dir /FL/data --data_format torchvision \
  --protocol_path /FL/trsfl/TRS-FL.py \
  --result_dir "$RUN_DIR/results" 2>&1 | tee "$RUN_DIR/train.log"
```

模型和数据划分固定为 CNN 与 IID，省略 --model cnn --iid 也采用相同配置；--device cpu 使用 CPU。每个任务使用新结果目录，禁止覆盖已有模式目录。可在 tmux 中启动后保留终端会话；本示例不会自动创建 tmux。

训练日志仅显示当前对照组、全局轮次和客户端编号，以及每轮结束后的测试准确率与测试损失。不显示本地 epoch、batch 损失或阶段提示；已移除 --verbose 参数。异常仍正常报告。

## 组秘密与 AS 验证

每轮组秘密由 AS 为两个 ES 组分别生成并保存在本地，份额分发供客户端重构。客户端签密上传不包含组秘密、秘密份额、轮密钥或私钥；AS 验签和去掩码直接使用自己保存的秘密，不要求客户端传回秘密。

上传仍包含动态假名、临时公钥、完整密文、签名与认证信息。AS 利用公开上下文与自身私钥派生客户端轮密钥，并检查固定批次；“组秘密不上传”不等于当前代码可以省略全部逐用户信息。当前角色均在单进程中模拟，实际顺序是先训练，再由 AS 准备本轮秘密和客户端轮状态，最后签密上传。

## 验证步骤

在目标服务器先确认依赖导入、GPU 可见性和 MNIST 可加载，再按以下顺序验证：

```bash
python server_checks.py --suite codec
python server_checks.py --suite protocol
```

验证用例使用可核算的小状态检查编码、20 人聚合、验证失败拒绝、重复配置和边界；正式训练入口只使用真实 SGD 输出。通过后先将正式命令的 --epochs 100 改为 --epochs 1，保持 --repeats 1，使用全新的结果目录验证两组单轮，再进入正式实验。小规模验证仍保留 20 客户端、2 ES，不用随机向量代替模型训练。

## 输出文件

- train.log：控制台进度与异常。
- results/mnist_cnn_iid/seed_1：本次运行记录，包含 paired、float、trsfl。
- 每组 metrics.jsonl：逐轮训练平均损失、测试准确率/损失及聚合诊断。
- 每组 config.json、model_layout.json、status.json：配置、完整模型布局和状态。
- 每组 last_good_model.pt：最后一次通过检查并提交的全局模型。--save_round_models 可另外保存每轮模型。
- 每组 summary.json：该次该组最终指标；异常另写 failure.json。
- 任务 summary/：final_by_seed.csv、paired_differences.csv、consistency.csv、statistics.json，以及 test_accuracy.png、test_loss.png。统计基于已完成且配置一致的两组结果。默认 n=1，不绘制标准差阴影，样本标准差记录为 null；配对差值只包含 trsfl−float。

两组结束后自动汇总，也可以手动执行 `python summarize_results.py <任务目录>`。最终结果不覆盖为最佳轮结果，未达到收敛目标与未配置目标分别记录。

## 验证状态

2026-10-21 本地已通过 16 个 Python 文件的语法检查，以及 `python -B server_checks.py --suite codec` 的全部 11 项检查，覆盖 CNN 完整模型布局、默认 CNN/IID 配置、IID 划分、编码边界和同批参照。已尝试 protocol 套件，但因缺少 Charm（`ModuleNotFoundError: No module named 'charm'`）在初始化时终止，未运行其中的协议用例；本机也未检测到 PyCryptodome 的 `Crypto` 模块。组秘密不上传且 AS 使用本地秘密的路径已做静态核对，新增的客户端秘密清除回归用例仍待协议环境验证。

未安装额外依赖，未运行模型训练或正式实验，未同步服务器。这些检查不代表完整密码学协议或 100 轮训练已经验证。

## 原项目与引用

本项目基于 Shaoxiong Ji 的 PyTorch Federated Learning 实现，保留 LICENSE。原仓库的历史结果不属于本次 TRS-FL 实验结果。

Shaoxiong Ji. A PyTorch Implementation of Federated Learning. Zenodo, 2018. DOI: 10.5281/zenodo.4321561。

McMahan et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS, 2017。
