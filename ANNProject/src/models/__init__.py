"""Model definitions.

所有神经网络模型通过此处统一导出。
BaseNeuralNetwork 是项目内所有网络模型的抽象基类，
新的 CNN、RNN 等架构只需实现 BaseNeuralNetwork 接口即可
无缝接入现有的训练器、评估器管线。
"""

from .base import BaseNeuralNetwork

from .bp_network import ANN, save_model, train_model

__all__ = [
    "ANN",
    "BaseNeuralNetwork",
    "save_model",
    "train_model",
]
