"""Evaluation and analysis tools.

提供可复用的评估引擎、指标计算、报告生成，与模型架构解耦。

模块分工:
  config.py      EvaluationConfig            — 统一配置
  metrics.py     RegressionMetrics           — 回归指标计算（无状态）
                 ClassificationMetrics       — 分类指标计算（无状态）
  tracker.py     EvaluationTracker           — batch 缓存 + 历史管理
  engine.py      EvaluationEngine            — 编排层（便捷方法 + 完整调用链）
  reporter.py    EvaluationReporter          — 格式化输出（无状态）
"""

from .config import EvaluationConfig
from .engine import EvaluationEngine
from .metrics import ClassificationMetrics, RegressionMetrics
from .reporter import EvaluationReporter
from .tracker import EvaluationTracker

__all__ = [
    "ClassificationMetrics",
    "EvaluationConfig",
    "EvaluationEngine",
    "EvaluationReporter",
    "EvaluationTracker",
    "RegressionMetrics",
]
