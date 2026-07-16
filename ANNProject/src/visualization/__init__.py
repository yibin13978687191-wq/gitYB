"""Visualization utilities.

提供可复用的可视化组件（MetricsContainer、NeuralNetworkVisualizer），
与模型架构解耦。CNN 或其他新架构可直接使用。
"""

from .data_visualization import NeuralNetworkVisualizer, TrainingMetrics, VisualMixin

__all__ = [
    "NeuralNetworkVisualizer",
    "TrainingMetrics",
    "VisualMixin",
]
