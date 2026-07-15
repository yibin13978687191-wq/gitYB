"""Configuration helpers for evaluation and reporting.

摘要:
- 定义统一配置类 EvaluationConfig，屏蔽分类/回归任务的差异
- 统一处理 task_type、confidence_level、output_format 等参数
- 适用于后续评估器、报告生成器和训练回调的统一入口
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class EvaluationConfig:
    """Unified configuration for both classification and regression evaluation."""

    task_type: str = "regression"
    confidence_level: float = 0.95
    output_format: str = "dict"
    label_mapping: Optional[Dict[Any, int]] = None

    def __post_init__(self) -> None:
        self.task_type = str(self.task_type).strip().lower()
        if self.task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in the open interval (0, 1)")
        self.output_format = self._normalize_output_format(self.output_format)

    @staticmethod
    def _normalize_output_format(output_format: Optional[str]) -> str:
        if output_format is None:
            return "dict"
        normalized = str(output_format).strip().lower()
        aliases = {
            "dict": "dict",
            "dictionary": "dict",
            "json": "dict",
            "dataframe": "dataframe",
            "pandas": "dataframe",
            "pd.dataframe": "dataframe",
            "pd_dataframe": "dataframe",
            "str": "str",
            "text": "str",
            "string": "str",
        }
        return aliases.get(normalized, normalized)
