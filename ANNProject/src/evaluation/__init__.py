"""Evaluation and analysis tools.

提供可复用的评估引擎、指标计算、报告生成，与模型架构解耦。
CNN 或其他新架构可直接使用本模块进行评估，无需继承混入类。
"""

from .config import EvaluationConfig
from .engine import EvaluationEngine
from .metrics import ClassificationMetrics, RegressionMetrics
from .reporter import EvaluationReporter

__all__ = [
    "ClassificationMetrics",
    "EvaluationConfig",
    "EvaluationEngine",
    "EvaluationReporter",
    "RegressionMetrics",
]
