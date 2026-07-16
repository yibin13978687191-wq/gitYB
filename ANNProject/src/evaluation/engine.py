"""评估引擎：编排 Tracker、Metrics、Reporter 完成端到端评估。

职责:
- EvaluationEngine 是评估的编排层，不负责缓存也不负责格式化
- 缓存管理委托给 EvaluationTracker
- 指标计算委托给 RegressionMetrics / ClassificationMetrics
- 报告输出委托给 EvaluationReporter
- 引擎只负责"在正确的时间调用正确的组件"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .config import EvaluationConfig
from .metrics import ClassificationMetrics, RegressionMetrics
from .reporter import EvaluationReporter
from .tracker import EvaluationTracker


class EvaluationEngine:
    """评估编排器：连接 Tracker、Metrics、Reporter 的胶水层。

    用法:
        engine = EvaluationEngine(config)
        # 训练循环中记录 batch
        engine.record_batch_predictions(y_true, y_pred, split="train")
        # epoch 结尾计算统计
        stats = engine.compute_epoch_statistics(epoch=1, split="train")
        # 训练结束后生成报告
        report = engine.generate_report(output_format="dict")
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        # ── 配置 ──
        self.config = config or EvaluationConfig()

        # ── 子组件（各司其职） ──
        self.tracker = EvaluationTracker()                    # 缓存 + 历史
        self.regression_metrics = RegressionMetrics(self.config)   # 回归计算
        self.classification_metrics = ClassificationMetrics(self.config)  # 分类计算
        self.reporter = EvaluationReporter()                  # 格式化（无状态）

    # ── 快捷属性 ──

    @property
    def task_type(self) -> str:
        """当前引擎配置的任务类型。"""
        return self.config.task_type

    @task_type.setter
    def task_type(self, value: str) -> None:
        """兼容旧代码中直接修改 task_type 的写法。"""
        self.config.task_type = value

    # ═════════════════════════════════════════════════════
    # Batch 级别：记录预测
    # ═════════════════════════════════════════════════════

    def record_batch_predictions(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = "train"
    ) -> None:
        """记录一个 batch 的预测结果，存入 tracker 缓存。

        自动将 torch.Tensor 转为 numpy 并展平为一维。
        分类任务会将概率 / logits 转为硬标签。
        """
        # ── Tensor → numpy ──
        y_true_np = y_true.detach().cpu().numpy().reshape(-1)

        if self.config.task_type == "classification":
            # 分类：将 logits 或概率转为硬标签
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred_np = torch.argmax(y_pred, dim=1).detach().cpu().numpy().reshape(-1)
            else:
                y_pred_np = (torch.sigmoid(y_pred) > 0.5).long().detach().cpu().numpy().reshape(-1)
        else:
            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)

        self.tracker.record_batch(y_true_np, y_pred_np, split)

    # ═════════════════════════════════════════════════════
    # Epoch 级别：合并缓存 → 计算 → 追加历史
    # ═════════════════════════════════════════════════════

    def compute_epoch_statistics(
        self, epoch: int = 0, split: str = "val"
    ) -> Dict[str, Any]:
        """合并指定 split 的所有 batch 预测，计算 epoch 统计并追加历史。

        流程:
          1. Tracker 合并缓存、调用 metrics_func 计算、清空缓存
          2. 根据 task_type 选择 RegressionMetrics 或 ClassificationMetrics
          3. 将结果追加到对应历史
          4. 返回带 task_type 前缀的指标字典

        Returns:
            带 task_type 字段的指标字典。缓存为空时返回 {}。
        """
        # 根据任务类型构造 metrics_func 闭包
        if self.config.task_type == "regression":
            def metrics_func(y_true, y_pred):
                return self.regression_metrics.compute(
                    y_true, y_pred, mode=split, epoch=epoch
                )
            history_key = "regression"
        else:
            def metrics_func(y_true, y_pred):
                return self.classification_metrics.compute(
                    y_true, y_pred, epoch=epoch
                )
            history_key = "classification"

        # 委托 Tracker 完成合并→计算→清空
        result = self.tracker.compute_epoch(epoch, split, metrics_func)
        if not result:
            return {}

        # 追加到对应历史
        self.tracker.append_result(result, history_key)

        # 返回带任务类型前缀的结果
        return {"task_type": self.config.task_type, **result}

    # ═════════════════════════════════════════════════════
    # 报告生成
    # ═════════════════════════════════════════════════════

    def generate_report(
        self, output_format: Optional[str] = None
    ) -> Union[Dict[str, Any], str, pd.DataFrame]:
        """生成评估报告。

        Args:
            output_format: "dict" / "dataframe" / "str" 或别名。
                           None 时使用 config.output_format。

        Returns:
            对应格式的评估结果。
        """
        fmt = self.config._normalize_output_format(
            output_format or self.config.output_format
        )
        history = self.tracker.get_history(self.config.task_type)
        return self.reporter.generate(history, self.config.task_type, fmt)

    # ═════════════════════════════════════════════════════
    # 历史与缓存管理
    # ═════════════════════════════════════════════════════

    @property
    def regression_history(self) -> List[Dict[str, Any]]:
        """回归评估历史（只读）。"""
        return self.tracker.get_history("regression")

    @property
    def classification_history(self) -> List[Dict[str, Any]]:
        """分类评估历史（只读）。"""
        return self.tracker.get_history("classification")

    def clear_cache(self) -> None:
        """清空所有缓存（保留历史）。"""
        self.tracker.clear()
