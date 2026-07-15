"""High-level evaluation engine orchestrating config, metrics, and reporting.

摘要:
- EvaluationEngine 是统一评估入口，负责收集预测结果、调用指标计算、生成报告
- 支持分类任务和回归任务共用一套调用链
- 后续可以继续扩展为训练回调、批量评估、实验结果汇总等能力
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import EvaluationConfig
from .metrics import ClassificationMetrics, RegressionMetrics
from .reporter import EvaluationReporter


class EvaluationEngine:
    """Unified evaluation engine for regression and classification tasks."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.task_type = self.config.task_type
        self.regression_history: List[Dict] = []
        self.classification_history: List[Dict] = []
        self.predictions_cache = {
            "train": {"y_true": [], "y_pred": []},
            "val": {"y_true": [], "y_pred": []},
            "test": {"y_true": [], "y_pred": []},
        }
        self.batch_statistics = defaultdict(list)
        self.regression_metrics = RegressionMetrics(self.config)
        self.classification_metrics = ClassificationMetrics(self.config)
        self.reporter = EvaluationReporter(self.task_type)

    def clear_cache(self) -> None:
        for split in self.predictions_cache:
            self.predictions_cache[split]["y_true"].clear()
            self.predictions_cache[split]["y_pred"].clear()
        self.batch_statistics.clear()

    def record_batch_predictions(self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = "train") -> None:
        assert split in {"train", "val", "test"}, "split must be 'train', 'val' or 'test'"
        y_true_np = y_true.detach().cpu().numpy().reshape(-1)
        if self.task_type == "classification":
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_np = torch.argmax(y_pred, dim=1).detach().cpu().numpy().reshape(-1)
            else:
                y_pred_np = (torch.sigmoid(y_pred) > 0.5).long().detach().cpu().numpy().reshape(-1)
        else:
            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        self.predictions_cache[split]["y_true"].append(y_true_np)
        self.predictions_cache[split]["y_pred"].append(y_pred_np)

    def compute_epoch_statistics(self, epoch: int = 0, split: str = "val") -> Dict:
        if not self.predictions_cache[split]["y_true"]:
            return {}
        y_true_all = np.concatenate(self.predictions_cache[split]["y_true"])
        y_pred_all = np.concatenate(self.predictions_cache[split]["y_pred"])
        if self.task_type == "regression":
            result = self.regression_metrics.compute(y_true_all, y_pred_all, mode=split, epoch=epoch)
            self.regression_history.append(result)
            self.reporter.task_type = "regression"
            return {"task_type": "regression", **result}
        result = self.classification_metrics.compute(y_true_all, y_pred_all, epoch=epoch)
        self.classification_history.append(result)
        self.reporter.task_type = "classification"
        return {"task_type": "classification", **result}

    def generate_report(self, output_format: Optional[str] = None) -> object:
        history = self.regression_history if self.task_type == "regression" else self.classification_history
        output_format = self.config._normalize_output_format(output_format or self.config.output_format)

        if output_format == "dict":
            return self.reporter.to_dict(history)
        if output_format == "dataframe":
            return self.reporter.to_dataframe(history)
        return self.reporter.to_text(history)
