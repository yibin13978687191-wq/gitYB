# ANN Project Management Plan

## 1. 项目定位

这个仓库目前是一个 Python ANN 学习和实验项目，主要目标是：

- 学习人工神经网络的基本搭建、训练、评估和可视化流程。
- 保留可复现的实验记录。
- 用 Git 管理每一次代码修改，方便回退、比较和继续开发。

当前阶段先做“管理重构”，不急着重写模型逻辑。

## 2. 当前目录问题

当前 `ANNProject/` 中同时存在源码、练习脚本、数据文件、结果 CSV、图片产物、IDE 配置和 Python 缓存。主要问题是：

- 源码和训练结果混在同一层，后续提交记录会变得混乱。
- `PracticeData/` 同时承担数据、练习代码和 Notebook 的职责。
- `.idea/`、`.DS_Store`、`__pycache__/`、`.pyc` 不应该进入 Git。
- 生成图片、训练结果、模型权重应该和源码分开管理。
- 缺少依赖声明、运行说明、实验记录模板和测试目录。

## 3. 推荐目标结构

后续建议逐步整理成以下结构：

```text
repo-root/
├─ src/
│  └─ ann_project/
│     ├─ models/
│     ├─ data/
│     ├─ training/
│     ├─ evaluation/
│     ├─ visualization/
│     └─ utils/
├─ scripts/
├─ notebooks/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ outputs/
│  ├─ figures/
│  ├─ metrics/
│  └─ checkpoints/
├─ tests/
├─ docs/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
└─ .gitignore
```

## 4. 当前文件迁移建议

建议分阶段迁移，避免一次性移动太多文件导致运行路径失效。

| 当前位置 | 建议位置 | 说明 |
| --- | --- | --- |
| `ANNProject/BPNetwork.py` | `src/ann_project/models/` 和 `src/ann_project/training/` | 后续拆分模型定义、数据处理、训练器 |
| `ANNProject/Analysis_Tool/` | `src/ann_project/evaluation/`、`src/ann_project/visualization/` | 拆分评估、计时、可视化工具 |
| `ANNProject/DataTest_2026.1.15.py` | `scripts/` | 作为可运行训练入口 |
| `ANNProject/PracticeData/*.csv` | `data/raw/` 或 `data/processed/` | 区分原始数据和处理后数据 |
| `ANNProject/PracticeData/*.xlsx` | `data/raw/` | 原始 Excel 数据 |
| `ANNProject/PracticeData/*.ipynb` | `notebooks/` | 探索实验和学习笔记 |
| `ANNProject/PracticeData/神经网络*.py` | `notebooks/` 或 `scripts/learning/` | 学习脚本，不放核心源码 |
| `ANNProject/visualization*.png` | `outputs/figures/` | 生成图片默认不作为核心代码提交 |
| `ANNProject/*results*.csv` | `outputs/metrics/` | 训练结果默认作为实验产物 |

## 5. Git 管理方案

### 分支规则

- `main`：稳定可运行版本。
- `feature/...`：新增功能，例如 `feature/add-dataset-loader`。
- `fix/...`：修复问题，例如 `fix/loss-logging`。
- `refactor/...`：重构代码，例如 `refactor/split-bpnetwork`。
- `experiment/...`：调参或实验，例如 `experiment/hidden-layers-64-32`。

### 提交规则

推荐提交信息格式：

```text
type: short description
```

常用类型：

- `feat`：新增功能
- `fix`：修复 bug
- `refactor`：重构，不改变功能
- `docs`：文档更新
- `test`：测试相关
- `chore`：项目配置、清理缓存等
- `exp`：实验记录

示例：

```text
docs: add ANN project management plan
refactor: split model trainer from BPNetwork
fix: correct metric calculation
exp: record baseline training on 2026-01-15 dataset
```

### 推荐日常流程

```bash
git status
git switch -c feature/your-task-name
git add .
git commit -m "feat: describe your change"
git push -u origin feature/your-task-name
```

合并到 `main` 前先确认：

- 代码能运行。
- 重要结果已经记录。
- 不必要的缓存和产物没有被提交。
- README 或文档已同步更新。

## 6. 数据和产物管理

建议规则：

- 小型原始数据可以提交到 Git，前提是没有隐私和版权风险。
- 频繁生成的训练结果、图片、模型权重默认不提交。
- 如果模型权重或大数据必须长期保存，后续考虑 Git LFS。
- 每次实验只提交摘要记录，不提交大量中间文件。

## 7. 推荐重构路线

### 阶段 1：管理骨架

- 增加 `.gitignore`。
- 增加项目管理文档。
- 增加 ANN 使用指南。
- 梳理哪些文件应该进入 Git。

### 阶段 2：环境复现

- 创建 `requirements.txt`。
- 记录 Python 版本。
- 确认 PyTorch 在 Apple Silicon 上的安装方式和运行设备。

### 阶段 3：目录迁移

- 新建 `src/ann_project/`。
- 先迁移工具类，再迁移训练入口。
- 每移动一小步就运行一次脚本。

### 阶段 4：模型拆分

- 模型定义放 `models/`。
- 数据读取和预处理放 `data/`。
- 训练循环放 `training/`。
- 指标和图表放 `evaluation/`、`visualization/`。

### 阶段 5：测试和实验记录

- 增加基础测试，至少验证数据形状、模型前向传播、训练器能跑通一小批数据。
- 建立 `docs/experiments/` 记录关键实验。

