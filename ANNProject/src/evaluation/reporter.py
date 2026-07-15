"""Formatting helpers for evaluation results.

摘要:
- EvaluationReporter 负责把评估结果格式化为 dict / DataFrame / text
- 便于训练过程中的日志打印、CSV 导出或接口回传
- 让报告输出和指标计算职责分离，降低模块耦合
"""

from __future__ import annotations

from typing import Dict, List, Union

import pandas as pd


class EvaluationReporter:
    """Render evaluation results into dict, text, or DataFrame."""

    def __init__(self, task_type: str):
        self.task_type = task_type

    def to_dict(self, history: List[Dict]) -> Dict:
        if not history:
            return {}
        latest = history[-1]
        if self.task_type == "regression":
            return {
                "task_type": "regression",
                "total_epochs": len(history),
                "latest_epoch": latest.get("epoch", 0),
                "latest_mse": latest.get("mse", 0.0),
                "latest_rmse": latest.get("rmse", 0.0),
                "latest_mae": latest.get("mae", 0.0),
                "latest_r2": latest.get("r2", 0.0),
                "latest_confidence_level": latest.get("confidence_level", 0.0),
                "best_mse": min(item.get("mse", 0.0) for item in history),
                "best_r2": max(item.get("r2", 0.0) for item in history),
                "mse_trend": [item.get("mse", 0.0) for item in history],
                "r2_trend": [item.get("r2", 0.0) for item in history],
            }
        return {
            "task_type": "classification",
            "total_epochs": len(history),
            "latest_epoch": latest.get("epoch", 0),
            "latest_accuracy": latest.get("accuracy", 0.0),
            "latest_f1_macro": latest.get("f1_macro", 0.0),
            "latest_f1_weighted": latest.get("f1_weighted", 0.0),
            "best_accuracy": max(item.get("accuracy", 0.0) for item in history),
            "best_f1_macro": max(item.get("f1_macro", 0.0) for item in history),
            "accuracy_trend": [item.get("accuracy", 0.0) for item in history],
            "f1_macro_trend": [item.get("f1_macro", 0.0) for item in history],
        }

    def to_dataframe(self, history: List[Dict]) -> pd.DataFrame:
        if not history:
            return pd.DataFrame()
        return pd.DataFrame(history)

    def to_text(self, history: List[Dict]) -> str:
        if not history:
            return "No evaluation history available."
        latest = history[-1]
        if self.task_type == "regression":
            return (
                f"Task type: regression\n"
                f"Latest epoch: {latest.get('epoch', 0)}\n"
                f"MSE: {latest.get('mse', 0.0):.6f}\n"
                f"RMSE: {latest.get('rmse', 0.0):.6f}\n"
                f"R^2: {latest.get('r2', 0.0):.6f}"
            )
        return (
            f"Task type: classification\n"
            f"Latest epoch: {latest.get('epoch', 0)}\n"
            f"Accuracy: {latest.get('accuracy', 0.0):.6f}\n"
            f"F1 macro: {latest.get('f1_macro', 0.0):.6f}"
        )
