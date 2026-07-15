"""Back-propagation ANN model definitions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from ann_project.core import DeviceManager, set_seed
from ann_project.data import DataPreprocessor
from ann_project.evaluation.evaluation_framework import AnalysisMixin
from ann_project.training.trainer import ModelManager, ModelTrainer, TrainingConfigurator
from ann_project.visualization.data_visualization import VisualMixin

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ANN(nn.Module, AnalysisMixin, VisualMixin):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        task_type: str,
        enable_analysis: bool = True,
        enable_visualization: bool = True,
        dropout_rate=0.3,
        activation="relu",
        use_batchnorm=True,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.task_type = task_type

        assert task_type in ["regression", "classification"], \
            "task_type 必须是 'regression' 或 'classification'"

        AnalysisMixin.__init__(self, task_type=task_type, enable_analysis=enable_analysis)
        VisualMixin.__init__(self)
        nn.Module.__init__(self)

        if enable_visualization:
            super().enable_visualization()

        self.layers = nn.ModuleDict({
            "input": nn.ModuleList(),
            "hidden": nn.ModuleList(),
            "output": nn.ModuleList(),
        })
        if self.use_batchnorm:
            self.layers["batch_norms"] = nn.ModuleList()

        self.activation_func = self._get_activation(activation)
        self.create_layer()

    @staticmethod
    def _get_activation(activation) -> nn.Module:
        if activation == "relu":
            return nn.ReLU()
        if activation == "leaky_relu":
            return nn.LeakyReLU()
        if activation == "elu":
            return nn.ELU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "sigmoid":
            return nn.Sigmoid()
        if activation == "tanh":
            return nn.Tanh()
        raise ValueError(f"Invalid activation function: {activation}")

    def create_layer(self):
        self.layers["input"].append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i < len(self.hidden_dims) - 1:
                self.layers["hidden"].append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
                if "batch_norms" in self.layers.keys():
                    self.layers["batch_norms"].append(nn.BatchNorm1d(self.hidden_dims[i]))

        self.layers["hidden"].append(nn.Dropout(self.dropout_rate))
        if "batch_norms" in self.layers.keys():
            self.layers["batch_norms"].append(nn.BatchNorm1d(self.hidden_dims[-1]))
        self.layers["output"].append(nn.Linear(self.hidden_dims[-1], self.output_dim))

    def print_network_structure(self):
        print("Network Structure:")
        for layer_key, module in self.layers.items():
            print(f"{layer_key}:{len(module)}")

    def print_parameter_size(self):
        for name, param in self.named_parameters():
            print(param.size())

    def get_parameter_num(self):
        return sum(p.numel() for p in self.parameters())

    def get_activation_func(self):
        return self.activation_func

    def get_layers_moduleslist(self, dim_type: str):
        dim_list = ["input", "hidden", "output", "batch_norms"]
        if dim_type not in dim_list:
            raise ValueError("Invalid dimension type. Please choose from 'input', 'hidden', 'output', or 'batch_norms'.")
        if dim_type in self.layers.keys():
            return self.layers[dim_type]
        raise ValueError("Invalid dimension type. Please choose from 'input', 'hidden', 'output'.")

    def print_model_parameters_shape(self):
        print("\n=== 模型参数 ===")
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")

    def print_model_parameters(self):
        print("\n=== 模型参数 ===")
        for name, param in self.named_parameters():
            print(f"{name}: {param}")

    def forward(self, x):
        input_layer = self.get_layers_moduleslist("input")
        hidden_layer = self.get_layers_moduleslist("hidden")
        output_layer = self.get_layers_moduleslist("output")
        batch_norms = self.get_layers_moduleslist("batch_norms") if self.use_batchnorm else None

        def batch_norm(tensor, i=0):
            for index, batch in enumerate(batch_norms):
                if index == i:
                    return batch(tensor)
            raise ValueError("Invalid index")

        def input_fc(tensor):
            for linear in input_layer:
                tensor = linear(tensor)
                if self.use_batchnorm:
                    tensor = batch_norm(tensor, 0)
            return self.activation_func(tensor)

        def hidden_fc(tensor):
            for index, linear in enumerate(hidden_layer):
                tensor = linear(tensor)
                if index == len(hidden_layer) - 1:
                    break
                if self.use_batchnorm:
                    tensor = batch_norm(tensor, index + 1)
                tensor = self.activation_func(tensor)
            return tensor

        def output_fc(tensor):
            for linear in output_layer:
                tensor = linear(tensor)
            return tensor

        x = input_fc(x)
        x = hidden_fc(x)
        x = output_fc(x)
        return x


def save_model(model: torch.nn.Module, save_path: str, train_dict: dict):
    file_manager = ModelManager(model)
    file_manager.save_model(
        model,
        save_path,
        additional_info={"training_history": train_dict},
    )


def train_model(
    model: torch.nn.Module,
    train_dataloader,
    val_dataloader,
    device,
    epochs=200,
    learning_rate=0.001,
    weight_decay=0.0001,
    print_interval=50,
):
    training_config = TrainingConfigurator(model, device, learning_rate=learning_rate, weight_decay=weight_decay)
    criterion, optimizer, scheduler, model = training_config()
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, device, epochs=epochs)
    train_dict = trainer.train(train_dataloader, val_dataloader, print_interval=print_interval)
    return trainer, train_dict


if __name__ == "__main__":
    file_path = PROJECT_ROOT / "outputs/metrics/Results2.csv"
    data = pd.read_csv(file_path, encoding="utf-8", header=0, index_col=0)
    data.drop(columns=["Label", "FeretX", "FeretY", "FeretAngle", "MinFeret"], inplace=True, errors="ignore")
    x = data.iloc[:, 0:4]
    y = data.iloc[:, 4:]

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
    trainer, train_dict = train_model(model, train_dataloader, val_dataloader, device)
    trainer.visualize_training_results(save_path=PROJECT_ROOT / "outputs/figures/visualization.png")
    print(trainer.get_training_results(output_format="str"))
