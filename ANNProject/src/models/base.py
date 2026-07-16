"""Abstract base class for all neural network models.

所有神经网络模型（BP、CNN 等）都应继承 BaseNeuralNetwork，
确保训练器、评估器等模块可以统一调用标准的模型接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BaseNeuralNetwork(nn.Module, ABC):
    """Abstract base class defining the model interface for the project.

    子类必须实现:
    - forward(x) — 前向传播
    - build_layers() — 构造网络层（推荐）

    子类可覆写的钩子方法（训练器会调用这些）:
    - record_predictions() — 记录每 batch 的预测结果
    - get_statistics() — 获取当前 epoch 的统计数据
    - record_training_step() — 记录训练步的中间状态
    - get_analysis_report() — 获取完整分析报告
    - plot_results() — 绘制训练结果图
    - plot_feature_importance() — 绘制特征重要性
    """

    def __init__(
        self,
        task_type: str = "regression",
        enable_analysis: bool = False,
        enable_visualization: bool = False,
    ):
        nn.Module.__init__(self)
        # 校验任务类型，避免后续错误扩散
        if task_type not in {"regression", "classification"}:
            raise ValueError(f"task_type 必须是 'regression' 或 'classification', 收到: {task_type}")
        self.task_type = task_type

        # 可选分析/可视化开关（由外部训练器或评估引擎注入）
        self._analysis_enabled = enable_analysis
        self._visualization_enabled = enable_visualization

    # ─── 子类必须实现的抽象方法 ────────────────────────────

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    # ─── 公共工具方法 ─────────────────────────────────────

    def get_parameter_num(self) -> int:
        """返回模型的总参数量。"""
        return sum(p.numel() for p in self.parameters())

    def print_network_structure(self) -> None:
        """打印网络的层级结构，子类可覆写以提供更详细输出。"""
        print(self)

    def print_parameter_shapes(self) -> None:
        """打印所有参数张量的形状。"""
        print("\n=== 模型参数形状 ===")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")

    # ─── 可选钩子（训练器/评估器会调用，子类可选择性覆写） ──

    def record_predictions(self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = "train") -> None:
        """记录一个 batch 的预测结果，供评估引擎汇总。
        默认空操作；接入评估引擎时子类可覆写，或由 Trainer 通过组合方式注入。
        """

    def get_statistics(self, epoch: int = 0, split: str = "val") -> Dict[str, Any]:
        """获取当前 epoch 的统计指标（MSE、R²、准确率等）。
        默认返回空字典；接入评估引擎时覆写或通过组合注入。
        """
        return {}

    def record_training_step(self, training_data: Optional[Dict[str, Any]]) -> None:
        """记录一次训练步的中间数据（损失、预测等）。
        默认空操作；接入可视化模块时覆写或通过组合注入。
        """

    def get_analysis_report(self, output_format: str = "dict") -> Any:
        """获取整个训练过程的完整分析报告。
        默认返回空字典；接入评估引擎时覆写或通过组合注入。
        """
        return {}

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """绘制训练结果图。
        默认空操作；接入可视化模块时覆写或通过组合注入。
        """

    def plot_feature_importance(
        self, feature_names: Optional[list] = None, save_path: Optional[str] = None
    ) -> None:
        """绘制特征重要性。
        默认空操作；接入可视化模块时覆写或通过组合注入。
        """

    def save_visualization_results(self, base_path: str) -> None:
        """保存所有可视化结果。
        默认空操作；接入可视化模块时覆写或通过组合注入。
        """

    def get_metrics_data(self, metric_type: str) -> Dict[str, Any]:
        """获取指定类型的指标数据。
        默认返回空字典；接入可视化模块时覆写或通过组合注入。
        """
        return {}

    def reset_visualizer(self) -> None:
        """重置可视化器状态。
        默认空操作；接入可视化模块时覆写或通过组合注入。
        """


# ─── 保持向后兼容：保留原有重导出 ──────────────────────────

from ann_project.core.device import DeviceManager  # noqa: E402, F401
from ann_project.core.seed import set_seed  # noqa: E402, F401
from ann_project.data.preprocessor import DataPreprocessor  # noqa: E402, F401
from ann_project.training.trainer import ModelTrainer, TrainingConfigurator  # noqa: E402, F401

__all__ = [
    "BaseNeuralNetwork",
    "DeviceManager",
    "set_seed",
    "DataPreprocessor",
    "ModelTrainer",
    "TrainingConfigurator",
]
