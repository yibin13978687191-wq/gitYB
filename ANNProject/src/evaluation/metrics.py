"""Task-specific metric computation for regression and classification.

摘要:
- RegressionMetrics 专注回归指标（MSE、RMSE、MAE、R2 等）
- ClassificationMetrics 专注分类指标（accuracy、precision、recall、F1、confusion matrix）
- 两类计算器彼此独立，避免把分类/回归逻辑塞进同一个函数
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

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

        n_samples = len(residuals)
        residual_mean = float(np.mean(residuals))
        # 样本标准差，ddof=1 用贝塞尔校正获得无偏估计
        residual_std = float(np.std(residuals, ddof=1)) if n_samples > 1 else 0.0
        # 残差均值的标准误（Standard Error of the Mean Residual），
        # 描述 residual_mean 作为真实偏置估计的不确定性
        se_mean = residual_std / np.sqrt(n_samples) if n_samples > 0 else 0.0
        # 残差均值的置信区间：residual_mean ± z * SE_Mean
        # 这与 ttest_1samp 的零假设检验在统计意义上一致
        bias_ci_lower = residual_mean - self.z_score * se_mean
        bias_ci_upper = residual_mean + self.z_score * se_mean
        bias_ci_width = bias_ci_upper - bias_ci_lower
        # 残差均值的相对精度：CI 宽度越小、偏置越接近 0，该值越接近 1
        # 具体公式：1 - (bias_ci_width / (abs(residual_mean) + bias_ci_width + 1e-8))
        # 比旧版 1/(1+CI/MSE) 更灵数稳定，且不受 MSE 趋零的影响
        denom = abs(residual_mean) + bias_ci_width + 1e-8
        mean_residual_precision = float(1.0 - bias_ci_width / denom)

        # 对残差做单样本 t 检验：零假设 H0 为 mean(residuals) == 0（即无系统性偏置）
        if n_samples > 1 and residual_std > 0:
            _, p_value = stats.ttest_1samp(residuals, 0)
            bias_significant = bool(p_value < 0.05)
        else:
            p_value = 1.0
            bias_significant = False

        prediction_interval_width = 2 * self.z_score * residual_std

        # R² 在极端情况下可为非常大的负数（模型比直接用均值更差），
        # 保留原始值的同时加一个 clip 后的版本
        r2_clipped = max(r2, -1.0)
        if r2 < -1.0:
            logger.warning(f"R² = {r2:.4f} is unusually low (clipped to -1.0), epoch={epoch}")

        return {
            # --- 基本信息 ---
            "epoch": int(epoch),
            "mode": mode,
            "sample_size": int(n_samples),

            # --- 核心误差指标 ---
            "mse": float(mse),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "r2_clipped": r2_clipped,
            "explained_variance": explained_variance,

            # --- 偏置分析（基于残差的均值） ---
            "mean_residual": residual_mean,
            "mean_residual_ci": (float(bias_ci_lower), float(bias_ci_upper)),
            "mean_residual_precision": mean_residual_precision,
            "bias_p_value": float(p_value),
            "bias_significant": bias_significant,

            # --- 预测区间 ---
            "prediction_interval_width": float(prediction_interval_width),
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
