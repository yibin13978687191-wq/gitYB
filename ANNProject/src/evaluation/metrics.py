"""Task-specific metric computation for regression and classification.

摘要:
- RegressionMetrics 专注回归指标（MSE、RMSE、MAE、R2 等）
- ClassificationMetrics 专注分类指标（accuracy、precision、recall、F1、confusion matrix）
- 两类计算器彼此独立，避免把分类/回归逻辑塞进同一个函数
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .config import EvaluationConfig


class RegressionMetrics:
    """Compute regression metrics from arrays."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.z_score = stats.norm.ppf((1 + config.confidence_level) / 2)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, mode: str = "train", epoch: int = 0) -> Dict[str, object]:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        residuals = y_true - y_pred
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        explained_variance = float(explained_variance_score(y_true, y_pred))

        residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        n_samples = len(residuals)
        se = residual_std / np.sqrt(n_samples) if n_samples > 0 else 0.0
        ci_lower = mse - self.z_score * se
        ci_upper = mse + self.z_score * se
        ci_width = ci_upper - ci_lower
        confidence_level_score = 1 / (1 + ci_width / mse) if mse > 0 else 1.0

        if n_samples > 1 and residual_std > 0:
            _, p_value = stats.ttest_1samp(residuals, 0)
            is_significant = bool(p_value < 0.05)
        else:
            p_value = 1.0
            is_significant = False

        prediction_interval_width = 2 * self.z_score * residual_std

        return {
            "epoch": int(epoch),
            "mode": mode,
            "mse": float(mse),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "explained_variance": explained_variance,
            "confidence_level": float(np.clip(confidence_level_score, 0, 1)),
            "significance_p_value": float(p_value),
            "is_significant": is_significant,
            "prediction_interval_width": float(prediction_interval_width),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "sample_size": int(n_samples),
        }


class ClassificationMetrics:
    """Compute classification metrics from arrays."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int = 0) -> Dict[str, object]:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same number of elements")

        if len(y_true) == 0:
            return {
                "epoch": int(epoch),
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "precision_weighted": 0.0,
                "recall_weighted": 0.0,
                "f1_weighted": 0.0,
                "confusion_matrix": np.zeros((0, 0), dtype=int),
                "class_accuracy": {},
                "sample_size": 0,
                "num_classes": 0,
            }

        all_labels = np.unique(np.concatenate([y_true, y_pred]))
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        y_true_enc = np.array([label_to_idx[label] for label in y_true], dtype=int)
        y_pred_enc = np.array([label_to_idx[label] for label in y_pred], dtype=int)

        accuracy = float(accuracy_score(y_true_enc, y_pred_enc))
        precision_macro = float(precision_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))
        recall_macro = float(recall_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))
        precision_weighted = float(precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0))
        recall_weighted = float(recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true_enc, y_pred_enc, labels=np.arange(len(all_labels)))

        class_accuracy = {}
        for class_idx in range(len(all_labels)):
            class_mask = y_true_enc == class_idx
            if np.sum(class_mask) > 0:
                class_correct = np.sum((y_true_enc == class_idx) & (y_pred_enc == class_idx))
                class_accuracy[class_idx] = float(class_correct / np.sum(class_mask))
            else:
                class_accuracy[class_idx] = 0.0

        return {
            "epoch": int(epoch),
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "class_accuracy": class_accuracy,
            "sample_size": int(len(y_true)),
            "num_classes": int(len(all_labels)),
        }
