"""Reusable training orchestration."""

from __future__ import annotations

import importlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ann_project.training.training_timer import TrainingTimer, TrainingTimeMetrics


class TrainingConfigurator:
    """Build optimizer, criterion, and scheduler for a model."""

    def __init__(
        self,
        model: nn.Module,
        device,
        learning_rate=0.001,
        weight_decay=0.0001,
        criterion: nn.Module | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.criterion = criterion
        self.scheduler = scheduler

    def __call__(self):
        return self.setup_training_components(self.model, self.learning_rate, self.weight_decay)

    def setup_training_components(self, model, learning_rate=0.001, weight_decay=0.0001):
        criterion = self.criterion or nn.SmoothL1Loss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        scheduler = self.scheduler or optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        model = model.to(self.device)
        return criterion, optimizer, scheduler, model


class ModelTrainer(TrainingTimer):
    """Reusable epoch training loop."""

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device, epochs: int,
                 project_name="NeuralNetworkTraining"):
        super().__init__(epochs)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_total_history = []
        self.val_total_history = []
        self.time_metrics_history = []
        self.learning_rate_history = []
        self.epochs = epochs
        self.last_learning_rate = self.scheduler.get_last_lr()[0] if self.scheduler is not None else None

    def train_epoch(self, train_loader, epoch_index: int):
        self.model.train()
        self.start_epoch()
        epoch_train_history = {
            "train_loss": [],
            "total_sample": [],
            "batch_num": [],
            "y_pred": [],
            "y_true": [],
            "epoch_avg_loss": 0.0,
        }

        for batch_idx, (data, target) in enumerate(train_loader):
            self.start_data_loading()
            data, target = data.to(self.device), target.to(self.device)
            self.record_data_loading()

            self.start_forward()
            y_pred = self.model(data)
            self.end_forward()

            self.start_backward()
            self.optimizer.zero_grad()
            loss = self.criterion(y_pred, target)
            loss.backward()
            self.end_backward()

            self.start_optimization()
            self.optimizer.step()
            self.end_optimization()

            epoch_train_history["y_pred"].append(y_pred.cpu().detach().numpy())
            epoch_train_history["y_true"].append(target.cpu().detach().numpy())
            epoch_train_history["train_loss"].append(loss.item())
            epoch_train_history["total_sample"].append(target.size(0))
            epoch_train_history["batch_num"].append(batch_idx + 1)
            self.model.record_predictions(target, y_pred)

        epoch_train_history["epoch_avg_loss"] = sum(epoch_train_history["train_loss"]) / len(train_loader)
        epoch_train_history["y_pred"] = np.concatenate(epoch_train_history["y_pred"], axis=0)
        epoch_train_history["y_true"] = np.concatenate(epoch_train_history["y_true"], axis=0)
        time_metrics = self.end_epoch(epoch_index + 1)
        return epoch_train_history, time_metrics

    def validate(self, val_loader, epoch_index: int):
        self.model.eval()
        epoch_val_history = {
            "val_loss": [],
            "total_sample": [],
            "batch_num": [],
            "y_pred": [],
            "y_true": [],
            "epoch_avg_loss": 0.0,
        }
        self.start_validation()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                y_pred = self.model(data)
                loss = self.criterion(y_pred, target)
                epoch_val_history["y_pred"].append(y_pred.cpu().detach().numpy())
                epoch_val_history["y_true"].append(target.cpu().detach().numpy())
                epoch_val_history["val_loss"].append(loss.item())
                epoch_val_history["total_sample"].append(target.size(0))
                epoch_val_history["batch_num"].append(batch_idx + 1)
                self.model.record_predictions(target, y_pred, split="val")

        self.end_validation()
        epoch_val_history["epoch_avg_loss"] = sum(epoch_val_history["val_loss"]) / len(val_loader)
        epoch_val_history["y_pred"] = np.concatenate(epoch_val_history["y_pred"], axis=0)
        epoch_val_history["y_true"] = np.concatenate(epoch_val_history["y_true"], axis=0)
        time_metrics = self.end_epoch(epoch_index + 1)
        return epoch_val_history, time_metrics

    def train(self, train_loader, val_loader=None, print_interval: int = 5,
              print_interval_time_report: bool = False,
              print_time_summary_report: bool = False,
              print_analysis_report: bool = False):
        epochs = self.epochs
        print(f"\n=== 开始训练，共{epochs}个epoch ===")

        for epoch in range(epochs):
            train_data_history, train_time_metrics = self.train_epoch(train_loader, epoch)
            train_statistics_data = self.model.get_statistics(epoch, split="train")
            self.train_total_history.append(train_data_history)
            self.time_metrics_history.append(train_time_metrics)

            if val_loader is not None:
                val_data_history, val_time_metrics = self.validate(val_loader, epoch)
                self.val_total_history.append(val_data_history)
            else:
                val_data_history, val_time_metrics = None, None

            val_statistics_data = self.model.get_statistics(epoch, split="val")
            self.last_learning_rate = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            self.learning_rate_history.append(self.last_learning_rate)
            self.model.record_training_step(train_data_history)
            self.model.record_training_step(val_data_history)
            self.model.record_regression_metrics(train_statistics_data)
            self.model.record_regression_metrics(val_statistics_data)

            if print_analysis_report:
                if (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == epochs - 1:
                    self.print_epoch_time_summary(epoch, train_data_history["epoch_avg_loss"], train_time_metrics)
                    if print_interval_time_report:
                        self.print_detailed_time_report(train_time_metrics)

        print("\n训练完成!")
        if print_time_summary_report:
            self.print_summary()
        performance_report = self.generate_performance_report()

        analysis_report = self.model.get_analysis_report(output_format="dict") if print_analysis_report else None
        training_dict = {
            "train_total_history": self.train_total_history,
            "statistical_analysis_report": analysis_report,
            "time_metrics_history": self.time_metrics_history,
            "performance_report": performance_report,
            "total_training_time": self.get_total_time(),
        }
        validation_dict = {
            "val_total_history": self.val_total_history,
            "time_metrics_history": self.time_metrics_history,
            "performance_report": performance_report,
            "total_training_time": self.get_total_time(),
        }
        return training_dict, validation_dict

    def print_epoch_time_summary(self, epoch: int, train_loss: float, time_metrics: TrainingTimeMetrics):
        print(f"\n=== 训练第{epoch + 1}个epoch ===")
        print(f"\n训练损失: {train_loss:.6f}, 学习率: {self.optimizer.param_groups[0]['lr']:.5f} ")
        print(f"本epoch用时: {time_metrics.epoch_time_formatted}")
        print(f"累计用时: {self._format_time(time_metrics.total_time_so_far)}")
        print(f"预计剩余时间: {time_metrics.estimated_time_remaining}")
        print(f"内存使用: {time_metrics.memory_usage_mb:.1f} MB")
        if time_metrics.gpu_memory_mb is not None:
            print(f"GPU内存使用: {time_metrics.gpu_memory_mb:.1f} MB")

    def print_detailed_time_report(self, time_metrics: TrainingTimeMetrics):
        print(f"\n{'─' * 40}")
        print("详细时间分析:")
        print(f"{'─' * 40}")

        components = [
            ("数据加载", time_metrics.data_loading_time),
            ("前向传播", time_metrics.forward_time),
            ("反向传播", time_metrics.backward_time),
            ("优化器更新", time_metrics.optimization_time),
            ("验证", time_metrics.validation_time),
        ]
        for name, time_taken in components:
            percentage = (time_taken / time_metrics.epoch_time * 100) if time_metrics.epoch_time > 0 else 0
            print(f"  {name:12s}: {self._format_time(time_taken):15s} ({percentage:5.1f}%)")

        other_time = time_metrics.epoch_time - sum([t for _, t in components])
        other_percentage = (other_time / time_metrics.epoch_time * 100) if time_metrics.epoch_time > 0 else 0
        print(f"  其他        : {self._format_time(other_time):15s} ({other_percentage:5.1f}%)")
        print(f"{'─' * 40}")

    def generate_performance_report(self):
        if not self.time_metrics_history:
            return {}
        epoch_times = [m.epoch_time for m in self.time_metrics_history]
        return {
            "total_epochs": len(self.time_metrics_history),
            "total_training_time_formatted": TrainingTimer._format_time(self.get_total_time()),
            "average_epoch_time": self._format_time(np.mean(epoch_times).astype(float)),
            "fastest_epoch": self._format_time(np.min(epoch_times).astype(float)),
            "slowest_epoch": self._format_time(np.max(epoch_times).astype(float)),
            "epoch_time_std": self._format_time(np.std(epoch_times).astype(float)),
            "time_efficiency": self._format_time(len(self.time_metrics_history) / self.get_total_time() if self.get_total_time() > 0 else 0),
            "memory_usage_avg": np.mean([m.memory_usage_mb for m in self.time_metrics_history]),
            "gpu_memory_avg": np.mean([m.gpu_memory_mb for m in self.time_metrics_history if m.gpu_memory_mb is not None])
            if any(m.gpu_memory_mb is not None for m in self.time_metrics_history) else None,
        }

    def get_training_results(self, output_format: str = "dict", statistic_analysis: bool = True, time_analysis: bool = True):
        if statistic_analysis:
            statistics_report = self.model.get_analysis_report(output_format)
        else:
            statistics_report = None
        if time_analysis:
            self.print_summary()
        return statistics_report

    def get_y_pred(self) -> np.ndarray:
        y_pred = self.model.get_metrics_data("training")
        return y_pred["y_pred"]

    def get_last_y_pred(self) -> np.ndarray:
        y_pred = self.model.get_metrics_data("training")
        return y_pred["y_pred"][-1]

    def get_validation_y(self):
        y_pred = self.model.get_metrics_data("validation")
        return y_pred["y"]

    def visualize_training_results(self, save_path: str):
        self.model.plot_results(save_path)
        self.model.plot_feature_importance(save_path=save_path)

    def export_training_results_to_csv(self, save_path: str):
        self.model.save_visualization_results(save_path)
        print(f"训练结果已导出到: {save_path}")


class ModelManager:
    """Model persistence helpers."""

    def __init__(self, saved_model):
        self.model = saved_model

    @staticmethod
    def save_model(_model, filepath, additional_info=None):
        save_dict = {
            "model_state_dict": _model.state_dict(),
            "model_class": _model.__class__.__name__,
            "model_module": _model.__class__.__module__,
            "model_config": {
                "input_dim": _model.input_dim,
                "hidden_dims": _model.hidden_dims,
                "output_dim": _model.output_dim,
                "dropout_rate": _model.dropout_rate,
                "use_batchnorm": _model.use_batchnorm,
                "task_type": getattr(_model, "task_type", None),
            },
            "additional_info": additional_info,
        }
        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath}")
        print(f"模型大小: {os.path.getsize(filepath) / 1024:.1f} KB" if os.path.exists(filepath) else "")

    @staticmethod
    def load_model(filepath, device="cpu", model_cls=None):
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint["model_config"]
        if model_cls is None:
            model_module = checkpoint.get("model_module")
            model_class_name = checkpoint.get("model_class")
            if model_module and model_class_name:
                module = importlib.import_module(model_module)
                model_cls = getattr(module, model_class_name)
        if model_cls is None:
            raise ValueError("model_cls is required when checkpoint does not include a resolvable model class")
        model_kwargs = {
            "input_dim": config["input_dim"],
            "hidden_dims": config["hidden_dims"],
            "output_dim": config["output_dim"],
            "dropout_rate": config["dropout_rate"],
            "use_batchnorm": config["use_batchnorm"],
        }
        if config.get("task_type") is not None:
            model_kwargs["task_type"] = config["task_type"]
        else:
            model_kwargs["task_type"] = checkpoint.get("task_type", "regression")
        try:
            loaded_model = model_cls(**model_kwargs)
        except TypeError:
            model_kwargs.pop("task_type", None)
            loaded_model = model_cls(**model_kwargs)
        loaded_model.load_state_dict(checkpoint["model_state_dict"])
        loaded_model = loaded_model.to(device)
        print(f"模型已从 {filepath} 加载")
        print(f"模型配置: {config}")
        return loaded_model, checkpoint.get("additional_info", {})
