# ANN Development Guide

## 1. 正确理解 ANN 项目

ANN 项目通常不只是一个模型文件，而是一条完整流程：

```text
数据 -> 预处理 -> 模型定义 -> 训练 -> 验证 -> 评估 -> 可视化 -> 实验记录
```

如果只改模型层数或学习率，但没有记录数据、参数和指标，后面很难判断一次修改到底是进步还是偶然。

## 2. 推荐工作流

每次开发或实验按这个顺序走：

1. 明确目标：例如提高准确率、降低损失、优化训练速度、清理代码结构。
2. 新建 Git 分支：每个目标一个分支。
3. 固定数据版本：确认本次训练用哪个数据文件。
4. 固定参数：记录学习率、batch size、epoch、隐藏层结构、随机种子。
5. 运行训练：保存关键指标，不保存所有临时文件。
6. 评估结果：用同一套指标比较不同实验。
7. 提交代码和实验摘要：让 Git 记录这次改变。

## 3. ANN 代码应该怎么拆

推荐把代码拆成这些职责：

| 模块 | 负责内容 |
| --- | --- |
| `models/` | 网络结构，例如 MLP、BPNetwork |
| `data/` | 数据读取、清洗、标准化、训练/验证集划分 |
| `training/` | 训练循环、优化器、损失函数、早停 |
| `evaluation/` | 准确率、精确率、召回率、F1、混淆矩阵等 |
| `visualization/` | 损失曲线、指标曲线、预测结果图 |
| `scripts/` | 可直接运行的训练或评估入口 |
| `notebooks/` | 探索、学习、临时验证 |

一个健康的 ANN 项目里，模型文件不应该同时负责所有事情。模型只关心输入如何变成输出；训练器才关心 epoch、loss、optimizer 和日志。

## 4. 数据管理方式

建议把数据分三层：

- `data/raw/`：原始数据，不手动修改。
- `data/interim/`：临时处理后的数据。
- `data/processed/`：可直接训练的数据。

命名建议：

```text
data/raw/ann_test_2026_01_15.xlsx
data/processed/ann_test_2026_01_15_features.csv
data/processed/ann_test_2026_01_15_labels.csv
```

尽量避免在核心脚本里写死中文文件名、日期文件名和绝对路径。后续可以通过配置文件传入数据路径。

## 5. 训练参数管理

初期可以用脚本顶部的字典管理参数：

```python
CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "hidden_layers": [64, 32],
    "random_seed": 42,
}
```

后续项目变大后，再迁移到 `configs/*.yaml`。

关键点是：每次实验必须知道“当时用的参数是什么”。

## 6. 评价指标

如果是分类任务，常用指标：

- accuracy
- precision
- recall
- f1-score
- confusion matrix

如果是回归任务，常用指标：

- MAE
- MSE
- RMSE
- R2

不要只看训练集 loss。更重要的是验证集或测试集指标，因为 ANN 很容易过拟合。

## 7. PyTorch 训练的基本结构

推荐训练循环保持清晰：

```python
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # evaluate validation metrics here
        pass
```

这段结构很重要：

- `model.train()` 开启训练模式。
- `optimizer.zero_grad()` 清空上一轮梯度。
- `loss.backward()` 反向传播。
- `optimizer.step()` 更新参数。
- `model.eval()` 切换评估模式。
- `torch.no_grad()` 避免评估阶段记录梯度。

## 8. Apple Silicon 上的设备选择

你的机器是 Apple Silicon Mac。PyTorch 通常可以优先尝试 `mps`：

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

训练时也要把数据移动到同一个设备：

```python
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)
```

如果遇到某些算子不支持 `mps`，可以临时退回 `cpu`，先保证结果正确。

## 9. 实验记录模板

建议后续在 `docs/experiments/` 中为重要实验记录一份 Markdown：

```markdown
# Experiment: baseline-ann-2026-01-15

Date: 2026-07-13
Git commit:
Dataset:
Task type:

## Parameters

- Learning rate:
- Batch size:
- Epochs:
- Hidden layers:
- Random seed:

## Results

- Train loss:
- Validation loss:
- Accuracy/F1/RMSE:

## Notes

- What changed:
- What improved:
- What to try next:
```

实验记录不需要写成论文，重点是让未来的你能复现和比较。

## 10. 学习路线

建议按这个顺序理解当前项目：

1. 先理解数据：输入特征是什么，目标标签是什么。
2. 再理解模型：输入维度、隐藏层、激活函数、输出维度。
3. 再理解训练：损失函数、优化器、学习率、epoch。
4. 再理解评估：当前到底在用什么指标判断模型好坏。
5. 最后做重构：把已经理解的部分拆到清晰目录中。

这个顺序会比直接改网络层数更稳，因为 ANN 的效果经常取决于数据处理和评价方式，而不只是模型结构。

