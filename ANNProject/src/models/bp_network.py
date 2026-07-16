"""Back-propagation ANN model implementation.

ANN 继承自 BaseNeuralNetwork，专注于网络架构定义（层结构、前向传播），
评估和可视化通过组合方式接入，而非混入继承。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ann_project.models.base import BaseNeuralNetwork

# ─── 项目根目录 ─────────────────────────────────────────
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ANN(BaseNeuralNetwork):
    """全连接前馈神经网络（BP / MLP），支持回归与分类。

    通过组合方式接入评估引擎（EvaluationEngine）和
    可视化模块（NeuralNetworkVisualizer），而非通过多重继承混入。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        output_dim: int,
        task_type: str = "regression",
        enable_analysis: bool = True,
        enable_visualization: bool = True,
        dropout_rate: float = 0.3,
        activation: str = "relu",
        use_batchnorm: bool = True,
    ):
        # ── 基类初始化（仅继承 nn.Module，不含混入类） ──
        super().__init__(task_type=task_type)

        # ── 架构参数 ──
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims) if not isinstance(hidden_dims, tuple) else hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.activation_name = activation

        # ── 激活函数（静态方法，无需作为实例方法） ──
        self.activation_func = self._get_activation(activation)

        # ── 构建网络层 ──
        self._build_layers()

        # ── 可选的评估和可视化组件（组合方式注入） ──
        self._analyzer = None
        self._visualizer = None
        if enable_analysis:
            self._init_analyzer(task_type)
        if enable_visualization:
            self._init_visualizer(task_type)

    # ════════════════════════════════════════════════════════
    # 网络层构建
    # ════════════════════════════════════════════════════════

    def _build_layers(self) -> None:
        """构造网络的所有层：输入层 → 隐藏层 → Dropout → 输出层。

        注意：Dropout 不放入隐藏层的 ModuleList，而是作为独立属性，
        避免 forward 中意外对其进行线性变换。
        """
        # 输入层：input_dim → hidden_dims[0]
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dims[0])

        # 隐藏层序列：hidden_dims[i] → hidden_dims[i+1]
        hidden_layers = []
        batch_norms = [] if self.use_batchnorm else None
        for i in range(len(self.hidden_dims) - 1):
            hidden_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            if self.use_batchnorm:
                # 在激活函数之前做 BN，作用于第 i 个隐藏层的输出
                batch_norms.append(nn.BatchNorm1d(self.hidden_dims[i]))
        self.hidden_layers = nn.ModuleList(hidden_layers)

        # 最后一个隐藏层后的 BatchNorm（在 Dropout 之前）
        if self.use_batchnorm:
            batch_norms.append(nn.BatchNorm1d(self.hidden_dims[-1]))
            self.batch_norms = nn.ModuleList(batch_norms)

        # Dropout——独立属性，不在 hidden_layers 中
        self.dropout = nn.Dropout(self.dropout_rate)

        # 输出层：hidden_dims[-1] → output_dim
        self.output_layer = nn.Linear(self.hidden_dims[-1], self.output_dim)

    # ════════════════════════════════════════════════════════
    # 前向传播
    # ════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """标准前向传播：输入 → 激活 → 隐藏层 × N → Dropout → 输出。"""
        # 输入层 + 激活
        x = self.input_layer(x)
        if self.use_batchnorm and len(self.batch_norms) > 0:
            x = self.batch_norms[0](x)
        x = self.activation_func(x)

        # 隐藏层 + 激活（batch_norms[0] 已用于输入层，从 [1:] 开始）
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_batchnorm and (i + 1) < len(self.batch_norms):
                # batch_norms[0]=输入层BN, batch_norms[i+1]=第i个隐藏层输出BN
                x = self.batch_norms[i + 1](x)
            x = self.activation_func(x)

        # Dropout（在输出层之前，仅在训练时生效）
        x = self.dropout(x)

        # 输出层（无激活，由损失函数处理）
        x = self.output_layer(x)
        return x

    # ════════════════════════════════════════════════════════
    # 静态工具方法
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """将字符串名称映射为 PyTorch 激活函数模块。"""
        activation_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
        }
        if activation not in activation_map:
            raise ValueError(f"不支持的激活函数: {activation}，可选: {list(activation_map.keys())}")
        return activation_map[activation]

    # ════════════════════════════════════════════════════════
    # 可选组件初始化（组合方式）
    # ════════════════════════════════════════════════════════

    def _init_analyzer(self, task_type: str) -> None:
        """初始化评估分析引擎。"""
        from ann_project.evaluation import EvaluationConfig, EvaluationEngine
        config = EvaluationConfig(task_type=task_type)
        self._analyzer = EvaluationEngine(config=config)

    def _init_visualizer(self, task_type: str) -> None:
        """初始化可视化引擎。"""
        from ann_project.visualization.data_visualization import NeuralNetworkVisualizer
        self._visualizer = NeuralNetworkVisualizer(task_type=task_type)
        # 同时初始化 TrainingMetrics 中的 task_type
        self._visualizer.task_type = task_type

    # ════════════════════════════════════════════════════════
    # 覆写 BaseNeuralNetwork 的钩子 → 委托给组合组件
    # ════════════════════════════════════════════════════════

    def record_predictions(self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = "train") -> None:
        if self._analyzer is not None:
            self._analyzer.log_batch(y_true, y_pred, split)

    def get_statistics(self, epoch: int = 0, split: str = "val") -> Dict[str, Any]:
        if self._analyzer is not None:
            return self._analyzer.eval_epoch(epoch=epoch, split=split)
        return {}

    def get_analysis_report(self, output_format: str = "dict") -> Any:
        if self._analyzer is not None:
            return self._analyzer.report(output_format)
        return {}

    def record_training_step(self, training_data: Optional[Dict[str, Any]]) -> None:
        if training_data is None or self._visualizer is None:
            return
        training_dict = self._visualizer.metrics.get_metrics("training")
        for metric_name, metric_value in training_data.items():
            if metric_name == "epoch_avg_loss":
                training_dict["epoch_avg_loss"] = metric_value
            else:
                training_dict.setdefault(metric_name, []).append(metric_value)

    def record_regression_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        if metrics_dict is None or self._visualizer is None:
            return
        regression_metrics = self._visualizer.metrics.get_metrics("regression")
        for metric_name, metric_value in metrics_dict.items():
            if metric_name in ("mse", "mae", "r2", "rmse") and isinstance(metric_value, (int, float)):
                regression_metrics[metric_name].append(metric_value)
            else:
                regression_metrics[metric_name] = metric_value

    def plot_results(self, save_path: Optional[str] = None) -> None:
        if self._visualizer is not None:
            self._visualizer.plot_training_history(save_path)

    def plot_feature_importance(
        self, feature_names: Optional[List[str]] = None, save_path: Optional[str] = None
    ) -> None:
        if self._visualizer is not None:
            self._visualizer.plot_feature_importance(self, feature_names, save_path)

    def save_visualization_results(self, base_path: str) -> None:
        if self._visualizer is not None:
            self._visualizer.save_all_plots(base_path)

    def get_metrics_data(self, metric_type: str) -> Dict[str, Any]:
        if self._visualizer is not None:
            return self._visualizer.metrics.get_metrics(metric_type)
        return {}

    def reset_visualizer(self) -> None:
        if self._visualizer is not None:
            self._visualizer.metrics.reset()

    # ════════════════════════════════════════════════════════
    # 保留兼容接口（打印调试信息）
    # ════════════════════════════════════════════════════════

    def print_network_structure(self) -> None:
        """打印网络的层结构详情。"""
        print(f"ANN(input_dim={self.input_dim}, hidden={list(self.hidden_dims)}, "
              f"output_dim={self.output_dim}, dropout={self.dropout_rate}, "
              f"batchnorm={self.use_batchnorm}, activation={self.activation_name})")
        print(f"  输入层: {self.input_layer}")
        for i, layer in enumerate(self.hidden_layers):
            print(f"  隐藏层[{i}]: {layer}")
        print(f"  Dropout: {self.dropout}")
        print(f"  输出层: {self.output_layer}")

    def get_activation_func(self) -> nn.Module:
        """返回当前使用的激活函数模块。"""
        return self.activation_func


# ════════════════════════════════════════════════════════════
# 独立训练/保存函数（保留向后兼容）
# ════════════════════════════════════════════════════════════

def save_model(model: torch.nn.Module, save_path: str, train_dict: dict) -> None:
    """保存模型及训练历史。"""
    from ann_project.training.trainer import ModelManager
    file_manager = ModelManager(model)
    file_manager.save_model(model, save_path, additional_info={"training_history": train_dict})


def train_model(
    model: torch.nn.Module,
    train_dataloader,
    val_dataloader,
    device,
    epochs: int = 200,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    print_interval: int = 50,
) -> tuple:
    """便捷训练函数：配置 → 训练 → 返回 (trainer, training_dict)。"""
    from ann_project.training.trainer import TrainingConfigurator, ModelTrainer
    training_config = TrainingConfigurator(model, device, learning_rate=learning_rate, weight_decay=weight_decay)
    criterion, optimizer, scheduler, model = training_config()
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, device, epochs=epochs)
    training_dict, validation_dict = trainer.train(train_dataloader, val_dataloader, print_interval=print_interval)
    return trainer, training_dict, validation_dict


if __name__ == "__main__":
    # 简单的即席测试入口
    file_path = PROJECT_ROOT / "outputs/metrics/Results2.csv"
    data = pd.read_csv(file_path, encoding="utf-8", header=0, index_col=0)
    data.drop(columns=["Label", "FeretX", "FeretY", "FeretAngle", "MinFeret"], inplace=True, errors="ignore")
    x = data.iloc[:, 0:4]
    y = data.iloc[:, 4:]

    from ann_project.data import DataPreprocessor
    from ann_project.core import DeviceManager, set_seed

    data_preprocessor = DataPreprocessor(x, y, test_size=0.2)
    train_dataloader, val_dataloader = data_preprocessor.create_dataloaders(batch_size=10, shuffle=True)
    input_dim = data_preprocessor.get_input_dim()
    output_dim = data_preprocessor.get_output_dim()

    set_seed(42)
    device_manager = DeviceManager(device_id=None)
    device = device_manager.get_device()
    model = ANN(
        input_dim,
        (12, 16, 28),
        output_dim,
        activation="relu",
        use_batchnorm=True,
        task_type="regression",
    )
    trainer, train_dict, val_dict = train_model(model, train_dataloader, val_dataloader, device)
    trainer.visualize_training_results(save_path=PROJECT_ROOT / "outputs/figures/visualization.png")
    print(trainer.get_training_results(output_format="str"))
