"""评估结果的格式化工具。

职责:
- EvaluationReporter 把评估指标（dict 列表）转为不同格式的输出
- 支持 dict（程序内使用）、DataFrame（CSV/分析）、text（终端打印）
- 与指标计算分离，regression_metrics / classification_metrics 只负责算，本模块只负责排
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


class EvaluationReporter:
    """评估结果渲染器：将历史指标列表转为 dict、DataFrame 或纯文本。

    用法:
        reporter = EvaluationReporter(task_type="regression")
        summary = reporter.to_dict(history)
        df = reporter.to_dataframe(history)
        print(reporter.to_text(history))
    """

    def __init__(self, task_type: str):
        # 任务类型："regression" 或 "classification"，决定输出字段
        self.task_type = task_type

    # ─────────────────────────────────────────────────
    # 输出格式：dict（适合程序内聚合/接口回传）
    # ─────────────────────────────────────────────────

    def to_dict(self, history: List[Dict]) -> Dict[str, Any]:
        """将评估历史压缩为单层汇总字典，包含最新值和趋势。

        Args:
            history: 每 epoch 一条的指标字典列表。

        Returns:
            汇总字典，含 task_type、latest_*、best_*、*_trend 等信息。
            空历史返回空字典。
        """
        if not history:
            return {}
        latest = history[-1]

        if self.task_type == "regression":
            return {
                # ── 任务信息 ──
                "task_type": "regression",
                "total_epochs": len(history),
                "latest_epoch": latest.get("epoch", 0),
                "n_samples": latest.get("sample_size", 0),

                # ── 最新误差指标 ──
                "latest_mse": latest.get("mse", 0.0),
                "latest_rmse": latest.get("rmse", 0.0),
                "latest_mae": latest.get("mae", 0.0),
                "latest_r2": latest.get("r2", 0.0),
                "latest_r2_clipped": latest.get("r2_clipped", -1.0),
                "latest_explained_variance": latest.get("explained_variance", 0.0),

                # ── 最新偏置分析 ──
                "latest_mean_residual": latest.get("mean_residual", 0.0),
                "latest_bias_p_value": latest.get("bias_p_value", 1.0),
                "latest_bias_significant": latest.get("bias_significant", False),
                "latest_prediction_interval_width": latest.get("prediction_interval_width", 0.0),

                # ── 全周期最佳值 ──
                "best_mse": min(item.get("mse", float("inf")) for item in history if item.get("mse") is not None),
                "best_r2": max(item.get("r2", float("-inf")) for item in history if item.get("r2") is not None),

                # ── 趋势序列（用于绘图或分析收敛性） ──
                "mse_trend": [item.get("mse", 0.0) for item in history],
                "r2_trend": [item.get("r2", 0.0) for item in history],
                "mae_trend": [item.get("mae", 0.0) for item in history],
            }

        return {
            # ── 任务信息 ──
            "task_type": "classification",
            "total_epochs": len(history),
            "latest_epoch": latest.get("epoch", 0),
            "n_samples": latest.get("sample_size", 0),
            "n_classes": latest.get("num_classes", 0),

            # ── 最新分类指标 ──
            "latest_accuracy": latest.get("accuracy", 0.0),
            "latest_precision_macro": latest.get("precision_macro", 0.0),
            "latest_recall_macro": latest.get("recall_macro", 0.0),
            "latest_f1_macro": latest.get("f1_macro", 0.0),
            "latest_precision_weighted": latest.get("precision_weighted", 0.0),
            "latest_recall_weighted": latest.get("recall_weighted", 0.0),
            "latest_f1_weighted": latest.get("f1_weighted", 0.0),

            # ── 全周期最佳值 ──
            "best_accuracy": max(
                item.get("accuracy", 0.0) for item in history if item.get("accuracy") is not None
            ),
            "best_f1_macro": max(
                item.get("f1_macro", 0.0) for item in history if item.get("f1_macro") is not None
            ),

            # ── 趋势序列 ──
            "accuracy_trend": [item.get("accuracy", 0.0) for item in history],
            "f1_macro_trend": [item.get("f1_macro", 0.0) for item in history],
        }

    # ─────────────────────────────────────────────────
    # 输出格式：DataFrame（适合 CSV 导出或 pandas 分析）
    # ─────────────────────────────────────────────────

    def to_dataframe(self, history: List[Dict]) -> pd.DataFrame:
        """将评估历史转为 DataFrame，每行一个 epoch。

        自动处理特殊类型:
        - tuple 列（如 mean_residual_ci）拆为两列 *_lower / *_upper
        - 嵌套 dict 列（如 class_accuracy）转 JSON 字符串
        - 空历史返回空 DataFrame
        """
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)

        # 展开 tuple 类型列（如 mean_residual_ci = (lower, upper)）
        # 避免 DataFrame 列类型退化为 object
        tuple_cols = [col for col in df.columns if df[col].dtype == object and isinstance(df[col].iloc[0], tuple)]
        for col in tuple_cols:
            df[f"{col}_lower"] = df[col].str[0]
            df[f"{col}_upper"] = df[col].str[1]
            df.drop(columns=[col], inplace=True)

        # 展开嵌套 dict 列（如 class_accuracy = {0: 0.9, 1: 0.8}）
        # 转 JSON 字符串保存，避免列数爆炸
        dict_cols = [col for col in df.columns if df[col].dtype == object and isinstance(df[col].iloc[0], dict)]
        for col in dict_cols:
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)

        return df

    # ─────────────────────────────────────────────────
    # 输出格式：纯文本（适合终端打印 / 日志）
    # ─────────────────────────────────────────────────

    def to_text(self, history: List[Dict]) -> str:
        """将评估历史格式化为可读文本。

        返回多行字符串，空历史则返回提示信息。
        """
        if not history:
            return "暂无可用的评估历史。"

        latest = history[-1]

        if self.task_type == "regression":
            # 最新 epoch 的核心回归指标
            lines = [
                "═══ 回归评估结果 ═══",
                f"  Epoch:     {latest.get('epoch', 0)}",
                f"  样本量:     {latest.get('sample_size', 0)}",
                f"  MSE:       {latest.get('mse', 0.0):.6f}",
                f"  RMSE:      {latest.get('rmse', 0.0):.6f}",
                f"  MAE:       {latest.get('mae', 0.0):.6f}",
                f"  R²:        {latest.get('r2', 0.0):.6f}",
                f"  解释方差:   {latest.get('explained_variance', 0.0):.6f}",
            ]
            # 偏置分析（仅在有显著偏置时附加警告）
            bias_sig = latest.get("bias_significant", False)
            if bias_sig:
                p = latest.get("bias_p_value", 1.0)
                lines.append(f"  ⚠ 存在显著偏置 (p={p:.4f})")
            return "\n".join(lines)

        # ── 分类结果 ──
        lines = [
            "═══ 分类评估结果 ═══",
            f"  Epoch:       {latest.get('epoch', 0)}",
            f"  样本量:       {latest.get('sample_size', 0)}",
            f"  类别数:       {latest.get('num_classes', 0)}",
            f"  Accuracy:    {latest.get('accuracy', 0.0):.4f}",
            f"  Precision:   {latest.get('precision_macro', 0.0):.4f} (macro)",
            f"  Recall:      {latest.get('recall_macro', 0.0):.4f} (macro)",
            f"  F1-Score:    {latest.get('f1_macro', 0.0):.4f} (macro)",
            f"  F1-Score:    {latest.get('f1_weighted', 0.0):.4f} (weighted)",
        ]
        return "\n".join(lines)
