"""向后兼容的评估框架适配层。

职责:
- NeuralNetworkAnalyzer — 旧版分析器的兼容包装，继承新的 EvaluationEngine
- AnalysisMixin — 供模型类混入使用的 mixin

迁移方向:
    新代码建议直接使用 EvaluationEngine + BaseNeuralNetwork 的组合方式。
    这个模块是为重构过渡期保留的兼容层，后续可逐步弃用。
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch

from .config import EvaluationConfig
from .engine import EvaluationEngine


class NeuralNetworkAnalyzer(EvaluationEngine):
    """旧版分析器的兼容包装，继承 EvaluationEngine 并提供便捷方法。

    新代码应直接使用 EvaluationEngine，无需经过此包装。

    用法:
        analyzer = NeuralNetworkAnalyzer(task_type="regression")
        analyzer.record_batch_predictions(y_true, y_pred, split="train")
        stats = analyzer.compute_epoch_statistics(epoch=1, split="train")
        report = analyzer.generate_report(output_format="dict")
    """

    def __init__(self, task_type: str = "regression", confidence_level: float = 0.95):
        """初始化分析器。

        Args:
            task_type: 任务类型（"regression" 或 "classification"）。
            confidence_level: 置信区间的置信水平（默认 0.95 = 95%）。
        """
        config = EvaluationConfig(
            task_type=task_type, confidence_level=confidence_level
        )
        super().__init__(config)

    def compute_regression_statistics(
        self, y_true, y_pred, mode: str = "train", epoch: int = 0
    ) -> Dict[str, Any]:
        """便捷方法：直接计算回归指标（不积累历史）。"""
        return self.regression_metrics.compute(y_true, y_pred, mode=mode, epoch=epoch)

    def compute_classification_statistics(
        self, y_true, y_pred, epoch: int = 0
    ) -> Dict[str, Any]:
        """便捷方法：直接计算分类指标（不积累历史）。"""
        return self.classification_metrics.compute(y_true, y_pred, epoch=epoch)

    def compute_epoch_statistics(
        self, epoch: int = 0, split: str = "val"
    ) -> Dict[str, Any]:
        """累加指定 split 所有缓存的 batch 预测，计算 epoch 统计。"""
        return super().compute_epoch_statistics(epoch=epoch, split=split)

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取全周期性能汇总（.to_dict 格式）。"""
        history = (
            self.regression_history
            if self.config.task_type == "regression"
            else self.classification_history
        )
        # 新 Reporter 为无状态，需显式传入 task_type
        return self.reporter.to_dict(history, task_type=self.config.task_type)

    def generate_report(
        self, output_format: str = "dict"
    ) -> Union[Dict[str, Any], str, pd.DataFrame]:
        """生成评估报告，支持多种输出格式。"""
        return super().generate_report(output_format)


class AnalysisMixin:
    """供模型类混入使用的评估 mixin。

    在新架构中，BaseNeuralNetwork 已通过组合方式实现等同功能。
    此 mixin 仅为向后兼容保留。
    """

    def __init__(self, task_type: Optional[str] = None, *args, **kwargs):
        self._analysis_enabled = kwargs.pop("enable_analysis", True)

        # 确定任务类型：优先显式传入，其次从已有属性读取，最后默认 "regression"
        task_type = task_type or getattr(self, "task_type", "regression")
        if task_type not in {"regression", "classification"}:
            raise ValueError(
                f"task_type 必须是 'regression' 或 'classification'，收到: {task_type}"
            )

        self.task_type = task_type

        if self._analysis_enabled:
            self.analyzer = NeuralNetworkAnalyzer(task_type=task_type)
        else:
            self.analyzer = None

    # ── 内部属性：统一的 "分析器是否可用" 判断 ──

    @property
    def _analyzer_ready(self) -> bool:
        return self._analysis_enabled and hasattr(self, "analyzer") and self.analyzer is not None

    # ── 开关控制 ──

    def enable_analysis(self) -> None:
        self._analysis_enabled = True
        if not self._analyzer_ready:
            task_type = getattr(self, "task_type", "regression")
            self.analyzer = NeuralNetworkAnalyzer(task_type=task_type)

    def disable_analysis(self) -> None:
        self._analysis_enabled = False

    # ── 数据记录与统计 ──

    def record_predictions(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = "train"
    ) -> None:
        if self._analyzer_ready:
            self.analyzer.record_batch_predictions(y_true, y_pred, split)

    def get_statistics(self, epoch: int = 0, split: str = "val") -> Dict[str, Any]:
        if self._analyzer_ready:
            return self.analyzer.compute_epoch_statistics(epoch=epoch, split=split)
        return {}

    def get_analysis_report(self, output_format: str = "dict") -> Any:
        if self._analyzer_ready:
            return self.analyzer.generate_report(output_format)
        return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        if self._analyzer_ready:
            return self.analyzer.get_performance_summary()
        return {}

    # ── 状态持久化 ──

    def save_analysis(self, filepath: str) -> None:
        if not self._analyzer_ready:
            return
        with open(filepath, "wb") as handle:
            pickle.dump(
                {
                    "task_type": self.analyzer.config.task_type,
                    "regression_history": self.analyzer.regression_history,
                    "classification_history": self.analyzer.classification_history,
                },
                handle,
            )

    def load_analysis(self, filepath: str) -> None:
        if not self._analyzer_ready:
            return
        with open(filepath, "rb") as handle:
            data = pickle.load(handle)
        # 通过 tracker 直接覆写历史（向后兼容）
        tracker = self.analyzer.tracker
        tracker._regression_history = data.get("regression_history", [])
        tracker._classification_history = data.get("classification_history", [])
        task_type = data.get("task_type")
        if task_type:
            self.analyzer.config.task_type = task_type
