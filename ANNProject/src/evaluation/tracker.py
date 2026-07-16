"""按 split 积累 batch 预测、生成 epoch 统计、管理历史记录。

职责:
- EvaluationTracker 负责三件事，且只负责这三件事:
  1. 按 train/val/test split 缓存 batch 级别的预测结果
  2. 在 epoch 结束时将所有缓存的 batch 合并、交由 Metrics 计算、清空缓存
  3. 按回归/分类分别积累历史记录，供 Reporter 格式化输出

与 Engine 的分工:
  Tracker 只管"存数据、取数据、存历史"，
  不管"用什么 Metrics 计算、计算参数是什么"——那是 Engine 的职责。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np


class EvaluationTracker:
    """按 split 缓存 batch 预测结果，按 epoch 生成统计，积累历史。

    用法:
        tracker = EvaluationTracker()

        # 训练循环中，每 batch 记录一次
        tracker.record_batch(y_true, y_pred, split="train")

        # epoch 结束时，合并缓存并计算
        result = tracker.compute_epoch(
            epoch=1, split="train",
            metrics_func=lambda y, p: {"mse": ((y-p)**2).mean()}
        )
        # 此时缓存已清空，result 被追加到回归历史中

        # 获取历史
        history = tracker.get_history("regression")
    """

    VALID_SPLITS = frozenset({"train", "val", "test"})

    def __init__(self):
        # ── 按 split 缓存的 batch 预测，每个 split 一对列表 ──
        self._predictions_cache: Dict[str, Dict[str, List[np.ndarray]]] = {
            split: {"y_true": [], "y_pred": []} for split in self.VALID_SPLITS
        }
        # ── 按任务类型积累的 epoch 历史 ──
        self._regression_history: List[Dict[str, Any]] = []
        self._classification_history: List[Dict[str, Any]] = []

    # ═══════════════════════════════════════════════════════════
    # 1. batch 级别：记录预测结果
    # ═══════════════════════════════════════════════════════════

    def record_batch(
        self, y_true: np.ndarray, y_pred: np.ndarray, split: str = "train"
    ) -> None:
        """缓存一个 batch 的预测结果。

        Args:
            y_true: 真实值，一维 numpy 数组。
            y_pred: 预测值，一维 numpy 数组。
            split: 数据集划分，限 "train" / "val" / "test"。
        """
        assert split in self.VALID_SPLITS, f"split 必须是 {self.VALID_SPLITS}，收到: {split}"
        self._predictions_cache[split]["y_true"].append(y_true)
        self._predictions_cache[split]["y_pred"].append(y_pred)

    # ═══════════════════════════════════════════════════════════
    # 2. epoch 级别：合并缓存 → 计算 → 清空 → 追加历史
    # ═══════════════════════════════════════════════════════════

    def compute_epoch(
        self,
        epoch: int,
        split: str,
        metrics_func: Callable[[np.ndarray, np.ndarray], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """合并指定 split 的所有缓存 batch，调用 metrics_func 计算，返回结果。

        流程:
          1. 若缓存为空，返回空字典
          2. 合并该 split 所有 batch 的 y_true / y_pred
          3. 调用 metrics_func(y_true_all, y_pred_all) 获取指标字典
          4. 清空该 split 的缓存
          5. 返回指标字典（不自动追加历史，由调用方决定）

        Args:
            epoch: 当前 epoch 编号。
            split: 数据集划分。
            metrics_func: 接收 (y_true, y_pred) 返回指标字典的可调用对象。
        """
        cache = self._predictions_cache[split]
        if not cache["y_true"]:
            return {}

        # 合并所有 batch 的预测结果
        y_true_all = np.concatenate(cache["y_true"])
        y_pred_all = np.concatenate(cache["y_pred"])

        # 清空缓存（必须先清空，即使后续计算失败也不污染下一 epoch）
        cache["y_true"].clear()
        cache["y_pred"].clear()

        # 调用外部指标函数
        return metrics_func(y_true_all, y_pred_all)

    # ═══════════════════════════════════════════════════════════
    # 3. 历史管理：追加 / 查询 / 清空
    # ═══════════════════════════════════════════════════════════

    def append_result(
        self, result: Dict[str, Any], task_type: str
    ) -> None:
        """将一个 epoch 的计算结果追加到对应任务类型的历史中。

        Args:
            result: compute_epoch 返回的指标字典。
            task_type: "regression" 或 "classification"。
        """
        if task_type == "regression":
            self._regression_history.append(result)
        elif task_type == "classification":
            self._classification_history.append(result)
        else:
            raise ValueError(f"task_type 必须是 'regression' 或 'classification'，收到: {task_type}")

    def get_history(self, task_type: str) -> List[Dict[str, Any]]:
        """获取指定任务类型的完整 epoch 历史。"""
        if task_type == "regression":
            return self._regression_history
        elif task_type == "classification":
            return self._classification_history
        raise ValueError(f"task_type 必须是 'regression' 或 'classification'，收到: {task_type}")

    def clear(self) -> None:
        """重置所有缓存和历史（慎用）。"""
        for split in self.VALID_SPLITS:
            self._predictions_cache[split]["y_true"].clear()
            self._predictions_cache[split]["y_pred"].clear()
        self._regression_history.clear()
        self._classification_history.clear()
