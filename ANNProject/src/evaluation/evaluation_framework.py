"""Backward-compatible evaluation framework adapter.

摘要:
- 保留旧版 NeuralNetworkAnalyzer 与 AnalysisMixin 的调用接口
- 通过新的配置、指标、报告与引擎模块提供底层能力
- 适用于当前训练器和模型类无需大改的场景
"""

import warnings
from typing import Dict, Optional, Union

import pandas as pd
import torch

from .config import EvaluationConfig
from .engine import EvaluationEngine

warnings.filterwarnings("ignore")


class NeuralNetworkAnalyzer(EvaluationEngine):
    """Backward-compatible analyzer adapter over the new evaluation engine."""

    def __init__(self, task_type: str = "regression", confidence_level: float = 0.95):
        config = EvaluationConfig(task_type=task_type, confidence_level=confidence_level)
        super().__init__(config)
        self.task_type = task_type

    def compute_regression_statistics(self, y_true, y_pred, mode: str = "train", epoch: int = 0):
        return self.regression_metrics.compute(y_true, y_pred, mode=mode, epoch=epoch)

    def compute_classification_statistics(self, y_true, y_pred, epoch: int = 0):
        return self.classification_metrics.compute(y_true, y_pred, epoch=epoch)

    def compute_epoch_statistics(self, epoch: int = 0, split: str = "val") -> Dict:
        return super().compute_epoch_statistics(epoch=epoch, split=split)

    def get_performance_summary(self) -> Dict:
        history = self.regression_history if self.task_type == "regression" else self.classification_history
        return self.reporter.to_dict(history)

    def generate_report(self, output_format: str = "dict") -> Union[Dict, str, pd.DataFrame]:
        return super().generate_report(output_format)


class AnalysisMixin:
    """Compatibility mixin that delegates to the new evaluation engine."""

    def __init__(self, task_type: Optional[str] = None, *args, **kwargs):
        self._analysis_enabled = kwargs.pop("enable_analysis", True)
        task_type = task_type or getattr(self, "task_type", "regression")
        if task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")

        if self._analysis_enabled:
            self.analyzer = NeuralNetworkAnalyzer(task_type=task_type)
        else:
            self.analyzer = None

    def enable_analysis(self):
        self._analysis_enabled = True
        if not hasattr(self, "analyzer") or self.analyzer is None:
            task_type = getattr(self, "task_type", "regression")
            self.analyzer = NeuralNetworkAnalyzer(task_type=task_type)

    def disable_analysis(self):
        self._analysis_enabled = False

    def record_predictions(self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = "train"):
        if self._analysis_enabled and hasattr(self, "analyzer") and self.analyzer is not None:
            self.analyzer.record_batch_predictions(y_true, y_pred, split)

    def get_statistics(self, epoch: int = 0, split: str = "val"):
        if self._analysis_enabled and hasattr(self, "analyzer") and self.analyzer is not None:
            return self.analyzer.compute_epoch_statistics(epoch=epoch, split=split)
        return {}

    def get_analysis_report(self, output_format: str = "dict"):
        if self._analysis_enabled and hasattr(self, "analyzer") and self.analyzer is not None:
            return self.analyzer.generate_report(output_format)
        return {}

    def get_performance_summary(self) -> dict:
        if self._analysis_enabled and hasattr(self, "analyzer") and self.analyzer is not None:
            return self.analyzer.get_performance_summary()
        return {}

    def save_analysis(self, filepath: str):
        if self._analysis_enabled and hasattr(self, "analyzer") and self.analyzer is not None:
            import pickle
            with open(filepath, "wb") as handle:
                pickle.dump({
                    "task_type": self.analyzer.task_type,
                    "regression_history": self.analyzer.regression_history,
                    "classification_history": self.analyzer.classification_history,
                }, handle)

    def load_analysis(self, filepath: str):
        if self._analysis_enabled and hasattr(self, "analyzer") and self.analyzer is not None:
            import pickle
            with open(filepath, "rb") as handle:
                data = pickle.load(handle)
            self.analyzer.task_type = data.get("task_type", self.analyzer.task_type)
            self.analyzer.regression_history = data.get("regression_history", [])
            self.analyzer.classification_history = data.get("classification_history", [])
