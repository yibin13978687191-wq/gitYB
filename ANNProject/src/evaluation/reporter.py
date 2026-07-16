"""评估结果的格式化工具。

职责:
- EvaluationReporter 把指标历史（dict 列表）转为不同格式的输出
- 无状态设计：不从构造函数接收 task_type，所有格式方法自动检测
- 支持 dict（程序内聚合）、DataFrame（CSV 导出）、text（终端打印）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class EvaluationReporter:
    """评估结果渲染器：将历史指标列表转为 dict、DataFrame 或纯文本。

    所有方法都是无状态的——输入 history，输出格式化结果。
    task_type 会自动从 history 数据中检测，无需手动指定。

    用法:
        reporter = EvaluationReporter()
        summary = reporter.to_dict(history)
        df = reporter.to_dataframe(history)
        print(reporter.to_text(history))
    """

    # ── 任务类型检测 ───────────────────────────────────

    @staticmethod
    def detect_task_type(history: List[Dict]) -> str:
        """根据 history 中第一条数据的字段自动判断任务类型。

        检测逻辑（按优先级）:
          - 包含 "accuracy" 或 "precision_macro" → "classification"
          - 否则默认 → "regression"
        """
        if not history:
            return "regression"
        keys = set(history[0].keys())
        if "accuracy" in keys or "precision_macro" in keys:
            return "classification"
        return "regression"

    # ─────────────────────────────────────────────────
    # 输出格式：dict（适合程序内聚合/接口回传）
    # ─────────────────────────────────────────────────

    def to_dict(
        self, history: List[Dict], task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """将评估历史压缩为单层汇总字典，包含最新值、最佳值和趋势。

        Args:
            history: 每 epoch 一条的指标字典列表。
            task_type: 可选，不传则自动检测。

        Returns:
            汇总字典。空历史返回空字典。
        """
        if not history:
            return {}
        task_type = task_type or self.detect_task_type(history)
        latest = history[-1]

        if task_type == "regression":
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
                "latest_prediction_interval_width": latest.get(
                    "prediction_interval_width", 0.0
                ),

                # ── 全周期最佳值（None 安全） ──
                "best_mse": min(
                    (item["mse"] for item in history if item.get("mse") is not None),
                    default=float("nan"),
                ),
                "best_r2": max(
                    (item["r2"] for item in history if item.get("r2") is not None),
                    default=float("nan"),
                ),

                # ── 趋势序列（用于绘图或分析收敛性） ──
                "mse_trend": [item.get("mse", 0.0) for item in history],
                "r2_trend": [item.get("r2", 0.0) for item in history],
                "mae_trend": [item.get("mae", 0.0) for item in history],
            }

        # ── 分类 ──
        return {
            "task_type": "classification",
            "total_epochs": len(history),
            "latest_epoch": latest.get("epoch", 0),
            "n_samples": latest.get("sample_size", 0),
            "n_classes": latest.get("num_classes", 0),

            "latest_accuracy": latest.get("accuracy", 0.0),
            "latest_precision_macro": latest.get("precision_macro", 0.0),
            "latest_recall_macro": latest.get("recall_macro", 0.0),
            "latest_f1_macro": latest.get("f1_macro", 0.0),
            "latest_precision_weighted": latest.get("precision_weighted", 0.0),
            "latest_recall_weighted": latest.get("recall_weighted", 0.0),
            "latest_f1_weighted": latest.get("f1_weighted", 0.0),

            "best_accuracy": max(
                (item["accuracy"] for item in history if item.get("accuracy") is not None),
                default=float("nan"),
            ),
            "best_f1_macro": max(
                (item["f1_macro"] for item in history if item.get("f1_macro") is not None),
                default=float("nan"),
            ),

            "accuracy_trend": [item.get("accuracy", 0.0) for item in history],
            "f1_macro_trend": [item.get("f1_macro", 0.0) for item in history],
        }

    # ─────────────────────────────────────────────────
    # 输出格式：DataFrame（适合 CSV 导出或 pandas 分析）
    # ─────────────────────────────────────────────────

    def to_dataframe(self, history: List[Dict]) -> pd.DataFrame:
        """将评估历史转为 DataFrame，每行一个 epoch。

        自动处理特殊类型:
        - tuple 列（如 mean_residual_ci）拆为 *_lower / *_upper
        - 嵌套 dict 列（如 class_accuracy）转字符串
        - 空历史返回空 DataFrame
        """
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)

        # 展开 tuple 类型列，避免列类型退化为 object
        for col in list(df.columns):
            if df[col].dtype == object and isinstance(df[col].iloc[0], tuple):
                df[f"{col}_lower"] = df[col].str[0]
                df[f"{col}_upper"] = df[col].str[1]
                df.drop(columns=[col], inplace=True)

        # 展开嵌套 dict 列，转字符串避免列爆炸
        for col in list(df.columns):
            if df[col].dtype == object and isinstance(df[col].iloc[0], dict):
                df[col] = df[col].apply(str)

        return df

    # ─────────────────────────────────────────────────
    # 输出格式：纯文本（适合终端打印 / 日志）
    # ─────────────────────────────────────────────────

    def to_text(
        self, history: List[Dict], task_type: Optional[str] = None
    ) -> str:
        """将评估历史格式化为可读文本。

        Args:
            history: 每 epoch 一条的指标字典列表。
            task_type: 可选，不传则自动检测。

        Returns:
            多行字符串。空历史返回提示信息。
        """
        if not history:
            return "暂无可用的评估历史。"
        task_type = task_type or self.detect_task_type(history)
        latest = history[-1]

        if task_type == "regression":
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
            if latest.get("bias_significant", False):
                p = latest.get("bias_p_value", 1.0)
                lines.append(f"  ⚠ 存在显著偏置 (p={p:.4f})")
            return "\n".join(lines)

        # ── 分类 ──
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

    # ─────────────────────────────────────────────────
    # 统一入口
    # ─────────────────────────────────────────────────

    def generate(
        self,
        history: List[Dict],
        task_type: Optional[str] = None,
        output_format: str = "dict",
    ) -> Union[Dict[str, Any], str, pd.DataFrame]:
        """统一的报告生成入口，按 output_format 分发。

        Args:
            history: 每 epoch 一条的指标字典列表。
            task_type: 可选，不传则自动检测。
            output_format: "dict" / "dataframe" / "str" 之一。

        Returns:
            对应格式的结果。空历史时 dict 返回空字典、str 返回提示。
        """
        output_format = output_format.strip().lower()
        if output_format in ("dataframe", "pandas"):
            return self.to_dataframe(history)
        if output_format in ("str", "text", "string"):
            return self.to_text(history, task_type)
        return self.to_dict(history, task_type)
