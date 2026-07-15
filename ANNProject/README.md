# ANNProject

这是一个 ANN 学习、训练、评估和可视化项目。当前阶段已先按 `docs/` 指引完成管理型整理，核心目标是让后续代码维护、测试和学习更清楚。

## 目录结构

```text
ANNProject/
├─ src/ann_project/
│  ├─ models/              # 模型定义与当前核心 BP/MLP 实现
│  ├─ evaluation/          # 评估与统计分析工具
│  ├─ training/            # 训练计时等训练辅助工具
│  └─ visualization/       # 可视化工具
├─ scripts/                # 可运行训练入口
├─ notebooks/              # 学习脚本和 Notebook
├─ data/
│  ├─ raw/                 # 原始数据
│  ├─ interim/             # 中间数据
│  └─ processed/           # 可直接训练的数据
├─ outputs/
│  ├─ figures/             # 图像输出
│  ├─ metrics/             # 指标、训练结果 CSV
│  └─ checkpoints/         # 模型权重
├─ BPNetwork.py            # 旧导入兼容入口
└─ Analysis_Tool/          # 旧工具导入兼容入口
```

## 当前重要文件

- `src/ann_project/models/bp_network.py`：当前 ANN、数据预处理、训练器等主要代码。
- `scripts/train_2026_01_15.py`：基于 `data/raw/ANN测试原数据2026.1.15.xlsx` 的训练入口。
- `notebooks/神经网络搭建.py`、`notebooks/神经网络初始化.py`：学习和演示脚本。
- `outputs/metrics/`：历史训练结果和指标 CSV。
- `outputs/figures/`：历史训练图像。

## 环境准备

建议先创建虚拟环境，再安装依赖：

```bash
cd ANNProject
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon Mac 上如需安装 PyTorch，可优先参考 PyTorch 官方 macOS 安装方式。当前代码会先尝试 CUDA，后续建议改成优先支持 `mps`。

## 运行入口

```bash
cd ANNProject
python3 scripts/train_2026_01_15.py
```

## 测试

```bash
cd ANNProject
pytest
```

当前测试先做包结构导入检查。如果本地还没有安装 `torch`，测试会自动跳过。

如果从旧 Notebook 或旧脚本中仍然使用 `from BPNetwork import ...` 或 `from Analysis_Tool...`，目前保留了兼容入口，便于逐步迁移。

## 后续维护建议

1. 先安装依赖并跑通 `scripts/train_2026_01_15.py`。
2. 给 `src/ann_project/models/bp_network.py` 拆分职责：模型、数据、训练器、配置。
3. 增加 `tests/`，至少覆盖数据形状、模型前向传播和一小批训练。
4. 重要实验写入 `docs/experiments/`，记录数据版本、参数和指标。
