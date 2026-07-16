"""Task-specific metric computation for regression and classification.

摘要:
- RegressionMetrics 专注回归指标（MSE、RMSE、MAE、R2 等）
- ClassificationMetrics 专注分类指标（accuracy、precision、recall、F1、confusion matrix）
- 两类计算器彼此独立，避免把分类/回归逻辑塞进同一个函数
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import scipy.stats as stats
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
    """回归指标计算器：汇总模型的误差、偏置分析和预测区间。

    用法:
        config = EvaluationConfig(confidence_level=0.95)
        rm = RegressionMetrics(config)
        stats = rm.compute(y_true, y_pred, mode="val", epoch=10)

    返回的字典包含三个层次:
        - 核心误差: MSE, RMSE, MAE, R², explained_variance
        - 偏置分析: 残差均值、t 检验、置信区间
        - 预测区间: prediction_interval_width
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        # 根据置信水平计算 z 分数（用于置信区间）
        # 例: confidence_level=0.95 → z_score≈1.96
        self.z_score = stats.norm.ppf((1 + config.confidence_level) / 2)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, mode: str = "train", epoch: int = 0) -> Dict[str, object]:
        # ── 输入标准化：确保一维数组 ──
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        # ── 残差与核心误差指标 ──
        # residuals = 真实值 - 预测值，正值表示低估、负值表示高估
        residuals = y_true - y_pred

        # MSE: 均方误差，对较大误差有更高的惩罚权重
        mse = mean_squared_error(y_true, y_pred)
        # RMSE: 均方根误差，单位与原始目标值一致
        rmse = float(np.sqrt(mse))
        # MAE: 平均绝对误差，对所有误差赋予相同权重
        mae = float(mean_absolute_error(y_true, y_pred))
        # R²: 决定系数，衡量模型解释的方差比例（1.0=完美，≤0=比均值还差）
        r2 = float(r2_score(y_true, y_pred))
        # 解释方差: 模型捕捉到的目标方差比例，与 R² 相近但不完全相同
        explained_variance = float(explained_variance_score(y_true, y_pred))

        # ── 偏置分析（基于残差的分布特征） ──
        n_samples = len(residuals)
        # 残差均值：若显著非零，说明模型存在系统性偏置（bias）
        residual_mean = float(np.mean(residuals))
        # 样本标准差（ddof=1 贝塞尔校正），用于推断总体标准差
        residual_std = float(np.std(residuals, ddof=1)) if n_samples > 1 else 0.0
        # 残差均值的标准误（Standard Error of the Mean, SEM）：
        # 描述 residual_mean 作为总体真实偏置估计的不确定性
        se_mean = residual_std / np.sqrt(n_samples) if n_samples > 0 else 0.0
        # 残差均值的置信区间：residual_mean ± z * SEM
        # 如果区间不包含 0，则模型存在统计上显著的偏置
        bias_ci_lower = residual_mean - self.z_score * se_mean
        bias_ci_upper = residual_mean + self.z_score * se_mean
        bias_ci_width = bias_ci_upper - bias_ci_lower
        # 偏置精度评分：综合 CI 宽度和偏置幅度，值越接近 1 表示偏置估计越可靠
        # 公式: 1 - CI_width / (|mean_residual| + CI_width + ε)
        denom = abs(residual_mean) + bias_ci_width + 1e-8
        mean_residual_precision = float(1.0 - bias_ci_width / denom)

        # 对残差做单样本 t 检验：H0 = "残差均值为 0"（即无系统性偏置）
        if n_samples > 1 and residual_std > 0:
            _, p_value = stats.ttest_1samp(residuals, 0)
            # p < 0.05 时拒绝 H0，认为存在显著偏置
            bias_significant = bool(p_value < 0.05)
        else:
            # 样本量不足或残差无波动时，无法拒绝 H0
            p_value = 1.0
            bias_significant = False

        # ── 预测区间宽度（模型在单次预测上的不确定性） ──
        # 公式: 2 * z * σ_residual，表示约 95% 的预测误差落在此区间内
        prediction_interval_width = 2 * self.z_score * residual_std

        # ── R² 裁剪（防止下游处理遇到极端负值） ──
        # R² 可为任意负数（模型劣于恒用均值），裁剪至 -1.0 后仍可用作比较
        r2_clipped = max(r2, -1.0)
        if r2 < -1.0:
            logger.warning(f"R² = {r2:.4f} is unusually low (clipped to -1.0), epoch={epoch}")

        return {
            # ── 基本信息 ──
            "epoch": int(epoch),
            "mode": mode,
            "sample_size": int(n_samples),

            # ── 核心误差指标 ──
            "mse": float(mse),                # 均方误差
            "rmse": rmse,                     # 均方根误差
            "mae": mae,                       # 平均绝对误差
            "r2": r2,                         # 原始决定系数（可能为负）
            "r2_clipped": r2_clipped,         # 裁剪后决定系数（≥ -1.0）
            "explained_variance": explained_variance,  # 解释方差

            # ── 偏置分析（基于残差的均值） ──
            "mean_residual": residual_mean,           # 残差均值（>0 = 低估偏置）
            "mean_residual_ci": (float(bias_ci_lower), float(bias_ci_upper)),  # 置信区间
            "mean_residual_precision": mean_residual_precision,  # 偏置精度评分 [0,1]
            "bias_p_value": float(p_value),            # t 检验 p 值
            "bias_significant": bias_significant,      # p < 0.05 时 True

            # ── 预测区间 ──
            "prediction_interval_width": float(prediction_interval_width),  # 单次预测不确定性
        }


class ClassificationMetrics:
    """分类指标计算器：汇总准确率、精确率、召回率、F1 和混淆矩阵。

    用法:
        config = EvaluationConfig(task_type="classification")
        cm = ClassificationMetrics(config)
        stats = cm.compute(y_true, y_pred, epoch=5)

    返回的指标包含 macro 和 weighted 两种平均方式:
        - macro: 对每个类等权平均（适合类别均衡场景）
        - weighted: 按各类样本量加权平均（适合类别不均衡场景）
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int = 0) -> Dict[str, object]:
        # ── 输入校验与标准化 ──
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true 和 y_pred 的样本数必须一致")

        # ── 空数据保护：直接返回零值填充的字典 ──
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

        # ── 标签编码：将原始标签（可能是字符串/非连续整数）映射为 [0, 1, ..., K-1] ──
        # 合并所有真实和预测标签以获取完整的类别空间
        all_labels = np.unique(np.concatenate([y_true, y_pred]))
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        y_true_enc = np.array([label_to_idx[label] for label in y_true], dtype=int)
        y_pred_enc = np.array([label_to_idx[label] for label in y_pred], dtype=int)

        # ── 全局分类指标 ──
        # accuracy: 整体准确率 = 正确预测数 / 总样本数
        accuracy = float(accuracy_score(y_true_enc, y_pred_enc))

        # macro 平均：先计算每个类各自的指标，再取算术平均
        # 小类和大类有相同权重，适合类别均衡场景
        precision_macro = float(precision_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))
        recall_macro = float(recall_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0))

        # weighted 平均：按每个类的样本量加权平均
        # 大类贡献更多，适合类别不均衡场景
        precision_weighted = float(precision_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0))
        recall_weighted = float(recall_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true_enc, y_pred_enc, average="weighted", zero_division=0))

        # 混淆矩阵：cm[i][j] = 真实类别 i 中被预测为类别 j 的样本数
        cm = confusion_matrix(y_true_enc, y_pred_enc, labels=np.arange(len(all_labels)))

        # ── 逐类准确率（用于识别哪些类别容易分错） ──
        class_accuracy = {}
        for class_idx in range(len(all_labels)):
            class_mask = y_true_enc == class_idx
            if np.sum(class_mask) > 0:
                class_correct = np.sum((y_true_enc == class_idx) & (y_pred_enc == class_idx))
                class_accuracy[class_idx] = float(class_correct / np.sum(class_mask))
            else:
                # 训练集中无此类的样本时，准确率记为 0
                class_accuracy[class_idx] = 0.0

        return {
            # ── 基本信息 ──
            "epoch": int(epoch),
            "sample_size": int(len(y_true)),
            "num_classes": int(len(all_labels)),

            # ── 整体准确率 ──
            "accuracy": accuracy,                          # 总体正确预测比例

            # ── Macro 平均（各类等权） ──
            "precision_macro": precision_macro,            # 宏平均精确率
            "recall_macro": recall_macro,                  # 宏平均召回率
            "f1_macro": f1_macro,                          # 宏平均 F1

            # ── Weighted 平均（按样本量加权） ──
            "precision_weighted": precision_weighted,      # 加权平均精确率
            "recall_weighted": recall_weighted,            # 加权平均召回率
            "f1_weighted": f1_weighted,                    # 加权平均 F1

            # ── 详细诊断信息 ──
            "confusion_matrix": cm,                        # 混淆矩阵 K×K
            "class_accuracy": class_accuracy,              # 每个类的独立准确率
        }
