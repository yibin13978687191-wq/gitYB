# Evolution 评估与训练模块说明

## 目标

本目录用于记录当前项目中“评估器重构后的结构说明、模块职责与调用方式”，方便后续开发、调试和 API 参考。

## 设计思路

原先的评估器逻辑过于集中，分类与回归任务、指标计算、报告生成以及缓存管理都堆叠在同一层。为提升可读性与可复用性，已经按以下思路进行拆分：

1. 配置层：统一管理评估参数
2. 指标层：分别负责回归和分类指标计算
3. 报告层：负责输出结果为 dict / DataFrame / 文本
4. 引擎层：串联配置、指标与报告，提供统一调用入口

## 主要模块

### 1. 配置模块
- 文件：[ANNProject/src/evaluation/config.py](../src/evaluation/config.py)
- 作用：管理统一评估配置，例如 `task_type`、`confidence_level`、`output_format`。
- 核心类：`EvaluationConfig`

### 2. 指标模块
- 文件：[ANNProject/src/evaluation/metrics.py](../src/evaluation/metrics.py)
- 作用：让回归和分类任务的指标计算彼此独立，便于扩展。
- 核心类：
  - `RegressionMetrics`
  - `ClassificationMetrics`

### 3. 报告模块
- 文件：[ANNProject/src/evaluation/reporter.py](../src/evaluation/reporter.py)
- 作用：把评估结果格式化输出，支持 dict、DataFrame、纯文本三种形式。
- 核心类：`EvaluationReporter`

### 4. 引擎模块
- 文件：[ANNProject/src/evaluation/engine.py](../src/evaluation/engine.py)
- 作用：提供统一的评估入口，记录预测结果、计算统计、生成报告。
- 核心类：`EvaluationEngine`

### 5. 兼容适配层
- 文件：[ANNProject/src/evaluation/evaluation_framework.py](../src/evaluation/evaluation_framework.py)
- 作用：对外保留旧接口，兼容当前训练器和模型类的调用方式。
- 核心类：
  - `NeuralNetworkAnalyzer`
  - `AnalysisMixin`

## 推荐使用方式

### 回归任务

```python
from ann_project.evaluation.evaluation_framework import NeuralNetworkAnalyzer

analyzer = NeuralNetworkAnalyzer(task_type="regression")
```

### 分类任务

```python
from ann_project.evaluation.evaluation_framework import NeuralNetworkAnalyzer

analyzer = NeuralNetworkAnalyzer(task_type="classification")
```

### 统一配置方式

```python
from ann_project.evaluation.config import EvaluationConfig

config = EvaluationConfig(task_type="classification", confidence_level=0.95, output_format="dict")
```

## 文件职责摘要

- `config.py`：定义统一配置类
- `metrics.py`：单独编写回归/分类指标逻辑
- `reporter.py`：统一输出格式
- `engine.py`：统一调度评估流程
- `evaluation_framework.py`：兼容旧版调用方式，方便接入训练流程

## 维护建议

- 新增任务类型时，优先在 `metrics.py` 中新增对应指标计算器
- 新增输出格式时，优先在 `reporter.py` 中扩展
- 若需要把训练流程进一步解耦，可在后续把 `EvaluationEngine` 作为训练回调接入
