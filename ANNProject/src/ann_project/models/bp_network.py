"""
BP神经网络完整实现 - 科研导向
作者：深度学习研究助手
功能：使用PyTorch实现多层感知机（MLP）进行多分类任务
"""
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from PIL.ImageCms import Flags
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pandas as pd
from torchsummary import summary
import inspect
# 设置随机种子以确保可重复性 - 科研必备！
from scipy import stats
import seaborn as sns
from typing import Dict, Tuple, List,Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import psutil
from ann_project.evaluation.evaluation_framework import *
from ann_project.visualization.data_visualization import VisualMixin
from ann_project.training.training_timer import *

PROJECT_ROOT = Path(__file__).resolve().parents[3]

class Inspector :
    @staticmethod
    def is_called_from_function(target_function_name):
        """
        检查当前是否在指定名称的函数中被调用
        Args:
        target_function_name: 目标函数名称
        Returns:
        bool: 如果在目标函数中则返回True，否则返回False
        """
    # 获取调用栈
        stack = inspect.stack()
    # 遍历调用栈，检查是否包含目标函数
        for frame_info in stack:
            if frame_info.function == target_function_name:
                return True
        return False
class DeviceManager:
    """设备管理器类"""
    def __init__(self, device_id=None):
        """
        初始化设备管理器

        Args:
            device_id: 指定GPU ID，如果为None则自动选择
        """
        self.device_id = device_id
        if self.device_id == -1:
            self.device = torch.device('cpu')
        else:
            self.device = self._get_device()
        self._setup_device()

    def _get_device(self):
        """获取最佳设备"""
        if torch.cuda.is_available():
            if self.device_id is not None:
                # 检查指定设备是否有效
                if self.device_id < torch.cuda.device_count():
                    device = torch.device(f'cuda:{self.device_id}')
                    print(f"使用指定GPU: {self.device_id} ({torch.cuda.get_device_name(self.device_id)})")
                else:
                    print(f"警告: GPU {self.device_id} 不存在，使用GPU 0")
                    device = torch.device('cuda:0')
            else:
                # 自动选择设备（基于内存使用）
                device = self._select_best_gpu()
        else:
            device = torch.device('cpu')
            print("警告: CUDA不可用，使用CPU")

        return device

    def _select_best_gpu(self):
        """自动选择最佳GPU（基于空闲内存）"""
        best_gpu = 0
        max_free_memory = 0

        for i in range(torch.cuda.device_count()):
            # 获取设备属性
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)

            # 计算可用内存
            free_memory = total_memory - allocated - cached

            print(f"  GPU {i}: {props.name}")
            print(f"    总内存: {total_memory / 1e9:.2f} GB")
            print(f"    已分配: {allocated / 1e9:.2f} GB")
            print(f"    缓存: {cached / 1e9:.2f} GB")
            print(f"    可用: {free_memory / 1e9:.2f} GB")

            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = i

        print(f"\n自动选择GPU {best_gpu}，可用内存: {max_free_memory / 1e9:.2f} GB")
        return torch.device(f'cuda:{best_gpu}')

    def _setup_device(self):
        """设置设备"""
        # 设置当前设备
        if 'cuda' in str(self.device):
            torch.cuda.set_device(self.device)

            # 清空缓存
            torch.cuda.empty_cache()

            # 设置随机种子（保证可复现性）
            torch.cuda.manual_seed_all(42)

    def get_device(self):
        """获取设备"""
        return self.device

    def to_device(self, data):
        """将数据移动到设备"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif hasattr(data, 'to'):
            return data.to(self.device, non_blocking=True)  # 异步传输
        else:
            return data

    def memory_info(self):
        """获取内存信息"""
        if 'cuda' in str(self.device):
            device_id = self.device.index if self.device.index else 0

            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            max_allocated = torch.cuda.max_memory_allocated(device_id)

            return {
                'allocated_gb': allocated / 1e9,
                'reserved_gb': reserved / 1e9,
                'max_allocated_gb': max_allocated / 1e9,
                'device': torch.cuda.get_device_name(device_id)
            }
        return {'device': 'cpu'}
class DataPreprocessor:
    def __init__(self,x_dataframe,y_dataframe, test_size=0.2):
        self.x_tensor = torch.tensor(x_dataframe.values, dtype=torch.float32)
        self.y_tensor = torch.tensor(y_dataframe.values, dtype=torch.float32)
        self.test_size = test_size
        if self.test_size == 0:
            self.x_train=self.x_tensor
            self.y_train=self.y_tensor
        else:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_tensor, self.y_tensor, test_size=test_size, random_state=42
        )
    def create_dataloaders(self,batch_size=8, shuffle=True):
        if self.test_size == 0:
            _train_dataset = TensorDataset(self.x_train, self.y_train)
            _train_dataloader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)
            return _train_dataloader
        else:
            _train_dataset = TensorDataset(self.x_train, self.y_train)
            _val_dataset = TensorDataset(self.x_val, self.y_val)
            _train_dataloader = DataLoader(_train_dataset, batch_size=batch_size, shuffle=True)
            _val_dataloader = DataLoader(_val_dataset , batch_size=batch_size, shuffle=True)
            return _train_dataloader, _val_dataloader
    def get_input_dim(self):
        return self.x_tensor.shape[1]
    def get_output_dim(self):
        return self.y_tensor.shape[1]

class ANN(nn.Module,AnalysisMixin,VisualMixin):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 task_type: str ,
                 enable_analysis:bool=True,
                 enable_visualization:bool=True,
                 dropout_rate=0.3,
                 activation='relu', use_batchnorm=True):
        """
        初始化神经网络结构
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，如[128, 64]表示两个隐藏层
            output_dim: 输出维度（分类数）
            dropout_rate: Dropout比率，用于防止过拟合
            activation: 激活函数类型
            use_batchnorm: 是否使用批标准化
        """
        AnalysisMixin.__init__(self)
        VisualMixin.__init__(self)
        nn.Module.__init__(self)  # 调用nn.Module父类初始化
        # 接受网络参数
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.task_type = task_type


        assert task_type in ['regression', 'classification'], \
            "task_type 必须是 'regression' 或 'classification'"

        # 如果启用了分析功能，则启用指定任务类型的分析混入功能
        if enable_analysis:
            super().enable_analysis()
        if enable_visualization:
            super().enable_visualization()

        # 构建网络层
        self.layers = nn.ModuleDict({'input':nn.ModuleList(),
                                     'hidden':nn.ModuleList(),
                                     'output':nn.ModuleList()})
        # 如果启用了批标准化功能，则创建批标准化层
        if self.use_batchnorm:
            self.layers['batch_norms'] = nn.ModuleList()
        # 选择激活函数
        self.activation_func = self._get_activation(activation)
        # 创建网络层
        self.create_layer()

    @staticmethod
    def _get_activation(activation)->nn.Module:
        """
        获取激活函数
        参数:
            activation: 激活函数类型
        返回:
            激活函数对象
        """
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Invalid activation function: {activation}")
    def create_layer(self):
        # 当维度类型为输入层时，向网络层列表中添加一个线性变换层
        self.layers['input'].append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        # 遍历隐藏层维度列表，构建神经网络的隐藏层结构
        # 对于每个隐藏层维度，创建对应的线性变换层和批归一化层（如果启用）
        for i, hidden_dim in enumerate(self.hidden_dims):
            # 当当前层不是最后一层时，创建输入输出维度相同的线性层
            if i < len(self.hidden_dims)-1:
                self.layers['hidden'].append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                # 如果启用了批归一化，则为当前层添加批归一化操作
            # 当当前层是最后一层时，创建连接到输出维度的线性层
                if "batch_norms" in self.layers.keys():
                    self.layers['batch_norms'].append(nn.BatchNorm1d(self.hidden_dims[i]))

        self.layers['hidden'].append(nn.Dropout(self.dropout_rate))
        if "batch_norms" in self.layers.keys():
            self.layers['batch_norms'].append(nn.BatchNorm1d(self.hidden_dims[-1]))
        # 添加最终的输出层，将最后一个隐藏层映射到输出维度
        self.layers['output'].append(nn.Linear(self.hidden_dims[-1], self.output_dim))
    def print_network_structure(self):
        # 打印网络中各层的信息，用于调试和模型结构查看
        print("Network Structure:")
        for layer_key, module in self.layers.items():
            print(f"{layer_key}:{len(module)}")
    def print_parameter_size(self):
        for name, param in self.named_parameters ():
            print(param.size())
    def get_parameter_num(self):
        return sum(p.numel() for p in self.parameters())
    def get_activation_func(self):
        return self.activation_func
    def get_layers_moduleslist(self,dim_type:str):
        dim_list = ['input', 'hidden', 'output', 'batch_norms']
        if dim_type not in dim_list:
            raise ValueError("Invalid dimension type. Please choose from 'input', 'hidden', 'output', or 'batch_norms'.")
        else:
            if dim_type in self.layers.keys():
                return self.layers[dim_type]
            else:
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
        input_layer = self.get_layers_moduleslist('input')
        hidden_layer = self.get_layers_moduleslist('hidden')
        output_layer = self.get_layers_moduleslist('output')
        batch_norms = self.get_layers_moduleslist('batch_norms') if self.use_batchnorm else None
        def batch_norm(x, i=0):
            for index, batch in enumerate(batch_norms):
                if index == i and i==0:
                    return batch(x)

                elif index == i and i>0:
                    return batch(x)
                else:
                    continue
            raise ValueError("Invalid index")

        def input_fc(x):
            for index, linear in enumerate(input_layer):
                x = linear(x)
                if self.use_batchnorm:
                    x =batch_norm(x,0)
            return self.activation_func(x)
        def hidden_fc(x):
            for index, linear in enumerate(hidden_layer):
                x = linear(x)
                if index == len(hidden_layer)-1:
                    break
                if self.use_batchnorm:
                    x = batch_norm(x,index+1)
                x = self.activation_func(x)
            return x

            # Dropout防止过拟合
        def output_fc(x):
            for index, linear in enumerate(output_layer):
                x = linear(x)
            return x
        x = input_fc(x)
        x = hidden_fc(x)
        x = output_fc(x)
        return x
class TrainingConfigurator:
    """
    训练配置类
    科研中需要系统调整训练参数
    """
    def __init__(self,
                 model: nn.Module,
                 device,
                 learning_rate=0.001,
                 weight_decay=0.0001,
                 ):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
    def __call__(self):
        """
        调用该方法进行训练配置
        返回:
            配置好的损失函数、优化器、学习率调度器
        """
        return self.setup_training_components(self.model, self.learning_rate, self.weight_decay)
    def setup_training_components(self, _model, learning_rate=0.001, weight_decay=0.0001):
        """
        设置训练所需的所有组件
        返回:
            配置好的损失函数、优化器、学习率调度器
        """
        criterion = nn.SmoothL1Loss()
        print(f"损失函数: {criterion.__class__.__name__}")
        # 2. 优化器 - Adam优化器（自适应学习率，通常效果较好）
        optimizer = optim.Adam(
            _model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)  # 一阶和二阶矩估计的衰减率
        )
        print(f"优化器: {optimizer.__class__.__name__} (lr={learning_rate})")

        # 3. 学习率调度器 - 训练过程中动态调整学习率
        # StepLR: 每step_size个epoch将学习率乘以gamma
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,  # 每30个epoch调整一次
            gamma=0.5  # 学习率减半
        )
        print(f"学习率调度器: StepLR (step_size=30, gamma=0.5)")

        _model = _model.to(self.device)
        print(f"训练设备: {self.device}")

        return criterion, optimizer, scheduler, _model
class ModelTrainer(TrainingTimer):
    """
    模型训练器
    封装完整的训练流程
    """
    def __init__(self, model:nn.Module,
                 criterion:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 scheduler:torch.optim.lr_scheduler._LRScheduler,
                 device:torch.device,
                 epochs:int,
                 project_name="NeuralNetworkTraining"):
        super().__init__(epochs)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        # 记录训练历史
        self.train_total_history = []
        self.val_total_history = []
        self.time_metrics_history = []
        self.learning_rate_history = []
        self.epochs = epochs
        self.last_learning_rate=self.scheduler.get_last_lr()[0] if self.scheduler is not None else None

    def train_epoch(self, train_loader, epoch_index: int):
        """训练一个epoch,返回损失和时间指标"""
        self.model.train()  # 设置为训练模式（启用Dropout和BatchNorm训练模式）
        self.start_epoch()
        epoch_train_history = {
            'train_loss': [],
            'total_sample': [],
            'batch_num': [],
            'y_pred': np.ndarray,
            'y_true': np.ndarray,
            'epoch_avg_loss':0.0,
        }


        for batch_idx, (data, target) in enumerate(train_loader):
            # 启动数据加载计时器
            self.start_data_loading()
            # 将数据移动到设备（GPU/CPU）
            data, target = data.to(self.device), target.to(self.device)

            # 记录数据加载时间的计时器方法调用
            # 通常在数据加载完成后调用，以便进行性能分析和监控
            self.record_data_loading()
            # 前向传播
            self.start_forward()
            y_pred = self.model(data)
            self.end_forward()

            self.start_backward()
            # 梯度清零 - 非常重要！防止梯度累积
            self.optimizer.zero_grad()
            # 计算损失
            loss = self.criterion(y_pred, target)

            # todo:可视化模型记录批次损失

            # 反向传播 - PyTorch自动计算梯度
            loss.sum().backward()

            self.end_backward()

            self.start_optimization()
            # 参数更新
            self.optimizer.step()
            self.end_optimization()
            # todo:统计信息
            epoch_train_history['y_pred'] = y_pred.cpu().detach().numpy()
            epoch_train_history['y_true'] = target.cpu().detach().numpy()

            epoch_train_history['train_loss'].append(loss.item())
            epoch_train_history['total_sample'].append(target.size(0))
            epoch_train_history['batch_num'].append(batch_idx+1)
            # 给统计模块框架记录结果
            self.model.record_predictions(target, y_pred)

            # 计算平均损失和准确率
        epoch_train_history['epoch_avg_loss']=(sum(epoch_train_history['train_loss']) / len(train_loader))
        time_metrics = self.end_epoch(epoch_index + 1)
        # todo 记录训练步骤

        return epoch_train_history,time_metrics
    def validate(self, val_loader, epoch_index: int):
        """验证模型"""
        self.model.eval()
        # 记录验证损失
        epoch_val_history = {
            'val_loss': [],
            'total_sample': [],
            'batch_num': [],
            'y_pred': np.ndarray,
            'y_true': np.ndarray,
            'epoch_avg_loss': 0.0,
        }
        # 开始验证计时
        self.start_validation()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # 将数据移动到设备（GPU/CPU）
                data, target = data.to(self.device), target.to(self.device)
                # 记录数据加载时间的计时器方法调用
                # 通常在数据加载完成后调用，以便进行性能分析和监控
                y_pred = self.model(data)
                # 计算损失
                loss = self.criterion(y_pred, target)
                # 记录统计信息
                epoch_val_history['y_pred']=y_pred.cpu().detach().numpy()
                epoch_val_history['y_true']=target.cpu().detach().numpy()
                epoch_val_history['val_loss'].append(loss.item())
                epoch_val_history['total_sample'].append(target.size(0))
                epoch_val_history['batch_num'].append(batch_idx+1)
                # 给统计模块框架记录结果
                self.model.record_predictions(target, y_pred, split="val")

        self.end_validation()

            # 计算平均损失和准确率
        epoch_val_history['epoch_avg_loss']=sum(epoch_val_history['val_loss']) / len(val_loader)
        # 获取更新后的时间指标（包含验证时间）
        time_metrics = self.end_epoch(epoch_index + 1)
        return epoch_val_history, time_metrics

    def train(self, train_loader:DataLoader,
              val_loader:DataLoader=None,
              print_interval: int = 5,

              print_interval_time_report: bool = False,
              print_time_summary_report: bool = False,
              print_analysis_report: bool = False
              ) ->Dict:
        """
        完整训练过程

        参数:
            train_loader: 训练数据加载器
            epochs: 训练轮数
            print_interval: 每隔多少epoch打印详细时间报告
            save_checkpoints: 是否保存检查点
        """

        epochs=self.epochs
        print(f"\n=== 开始训练，共{epochs}个epoch ===")

        for epoch in range(epochs):
            # 训练一个epoch
            train_data_history,train_time_metrics= self.train_epoch(train_loader, epoch)

            # 记录统计信息
            train_statistics_data=self.model.get_statistics(epoch,split="train")
            # 记录每一轮的训练数据
            self.train_total_history.append(train_data_history)
            self.time_metrics_history.append(train_time_metrics)
            # 验证一个epoch
            if val_loader is not None:
                val_data_history, val_time_metrics = self.validate(val_loader, epoch)
                self.val_total_history.append(val_data_history)
            else:
                val_data_history, val_time_metrics = None, None
            # 记录统计信息
            val_statistics_data=self.model.get_statistics(epoch,split="val")

            # 学习率调整
            self.last_learning_rate = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            self.learning_rate_history.append(self.last_learning_rate)
            # 记录训练和验证步骤到可视化模型
            self.model.record_training_step(train_data_history)
            self.model.record_training_step(val_data_history)
            # 记录指标统计信息步骤到可视化模型
            self.model.record_regression_metrics(train_statistics_data)
            self.model.record_regression_metrics(val_statistics_data)

            # 打印进度
            if print_analysis_report:
                if (epoch + 1) % print_interval == 0 or epoch == 0 or epoch == epochs - 1:
                    self.print_epoch_time_summary(epoch, train_data_history['epoch_avg_loss'], train_time_metrics)
                    if print_interval_time_report:
                        self.print_detailed_time_report(train_time_metrics)


        print(f"\n训练完成!")
        if print_time_summary_report:
            self.print_summary()
        performance_report = self.generate_performance_report()

        if print_analysis_report:
            analysis_report=self.model.get_analysis_report(output_format='dict')
        else:
            analysis_report=None

        training_dict={
            'train_total_history': self.train_total_history,
            'statistical_analysis_report':analysis_report,
            'time_metrics_history': self.time_metrics_history,
            'performance_report': performance_report,
            'total_training_time': self.get_total_time()
        }
        validation_dict={
            'val_total_history': self.val_total_history,
            'time_metrics_history': self.time_metrics_history,
            'performance_report': performance_report,
            'total_training_time': self.get_total_time()
        }
        return training_dict, validation_dict
    def print_epoch_time_summary(self, epoch: int, train_loss: float,
                                 time_metrics: TrainingTimeMetrics):
        """打印epoch总结"""
        print(f"\n=== 训练第{epoch + 1}个epoch ===")
        print(f"\n训练损失: {train_loss:.6f}, 学习率: {self.optimizer.param_groups[0]['lr']:.5f} ")
        print(f"本epoch用时: {time_metrics.epoch_time_formatted}")
        print(f"累计用时: {self._format_time(time_metrics.total_time_so_far)}")
        print(f"预计剩余时间: {time_metrics.estimated_time_remaining}")
        print(f"内存使用: {time_metrics.memory_usage_mb:.1f} MB")
        if time_metrics.gpu_memory_mb is not None:
            print(f"GPU内存使用: {time_metrics.gpu_memory_mb:.1f} MB")

    def print_detailed_time_report(self, time_metrics: TrainingTimeMetrics):
        """打印详细时间报告"""
        print(f"\n{'─' * 40}")
        print("详细时间分析:")
        print(f"{'─' * 40}")

        components = [
            ('数据加载', time_metrics.data_loading_time),
            ('前向传播', time_metrics.forward_time),
            ('反向传播', time_metrics.backward_time),
            ('优化器更新', time_metrics.optimization_time),
            ('验证', time_metrics.validation_time)
        ]

        for name, time_taken in components:
            percentage = (time_taken / time_metrics.epoch_time * 100) if time_metrics.epoch_time > 0 else 0
            print(f"  {name:12s}: {self._format_time(time_taken):15s} ({percentage:5.1f}%)")

        other_time = time_metrics.epoch_time - sum([t for _, t in components])
        other_percentage = (other_time / time_metrics.epoch_time * 100) if time_metrics.epoch_time > 0 else 0
        print(f"  其他        : {self._format_time(other_time):15s} ({other_percentage:5.1f}%)")
        print(f"{'─' * 40}")

    def generate_performance_report(self) -> Dict:
        """生成性能报告"""
        if not self.time_metrics_history:
            return {}

        epoch_times = [m.epoch_time for m in self.time_metrics_history]

        report = {
            'total_epochs': len(self.time_metrics_history),
            'total_training_time_formatted': TrainingTimer._format_time(self.get_total_time()),
            'average_epoch_time': self._format_time(np.mean(epoch_times).astype(float)),
            'fastest_epoch': self._format_time(np.min(epoch_times).astype(float)),
            'slowest_epoch': self._format_time(np.max(epoch_times).astype(float)),
            'epoch_time_std': self._format_time(np.std(epoch_times).astype(float)),
            'time_efficiency': self._format_time(len(
                self.time_metrics_history) / self.get_total_time() if self.get_total_time() > 0 else 0),
            'memory_usage_avg': np.mean([m.memory_usage_mb for m in self.time_metrics_history]),
            'gpu_memory_avg': np.mean(
                [m.gpu_memory_mb for m in self.time_metrics_history if m.gpu_memory_mb is not None]) if any(
                m.gpu_memory_mb is not None for m in self.time_metrics_history) else None
        }

        return report

    def get_training_results(self, output_format: str= 'dict',
                             statistic_analysis: bool = True,
                             time_analysis: bool = True):

        """
        显示训练结果的详细信息

        参数:
            training_results: train()方法返回的字典
            detailed: 是否显示详细信息
        """

        if statistic_analysis:
            statistics_report = self.model.get_analysis_report(output_format)

        else:
            statistics_report = None
        if time_analysis:
            self.print_summary()

        return statistics_report
    def get_y_pred(self)-> np.ndarray:
        """
        预测标签
        """
        y_pred = self.model.get_metrics_data("training")
        return y_pred["y_pred"]
    def get_last_y_pred(self)-> np.ndarray:
        """
        预测标签
        """
        y_pred = self.model.get_metrics_data("training")
        return y_pred["y_pred"][-1]
    def get_validation_y(self):
        """
        获取验证集预测值
        """
        y_pred = self.model.get_metrics_data("validation")
        return y_pred["y"]
    def visualize_training_results(self, save_path:str):
        """
        可视化训练结果
        """
        self.model.plot_results(save_path)
        self.model.plot_feature_importance(save_path=save_path)
    def export_training_results_to_csv(self,save_path:str):
        """
    将训练全过程的详细数据导出为CSV文件

    """
        self.model.save_visualization_results(save_path)
        '''
        import csv
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 写入训练损失历史
            writer.writerow(['Epoch', 'Train_Loss','y_pred'])
            for key, values in self.model.get_metrics_data("training").items():
                writer.writerow()

            # 写入性能统计数据
            writer.writerow([])  # 空行分隔
            writer.writerow(['Performance_Metrics', 'Value'])
            performance_report = training_results.get('performance_report', {})
            for metric, value in performance_report.items():
                writer.writerow([metric, value])
        '''
        print(f"训练结果已导出到: {save_path}")
class ReportGenerator(ModelTrainer):
    """
    报告生成器
    生成训练报告和可视化结果
    """
    def __init__(self, model):
        super().__init__()
        
        
class ModelManager:
    """
    模型管理器
    处理模型的保存、加载和推理
    """
    def __init__(self, saved_model):
        self.model = saved_model
    @staticmethod
    def save_model(_model, filepath, additional_info=None):
        """
        保存模型

        参数:
            model: 要保存的模型
            filepath: 保存路径
            additional_info: 额外的信息（如训练历史、超参数等）
        """
        save_dict = {
            'model_state_dict': _model.state_dict(),
            'model_class': _model.__class__.__name__,
            'model_config': {
                'input_dim': _model.input_dim,
                'hidden_dims': _model.hidden_dims,
                'output_dim': _model.output_dim,
                'dropout_rate': _model.dropout_rate,
                'use_batchnorm': _model.use_batchnorm
            },
            'additional_info': additional_info
        }

        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath}")
        print(f"模型大小: {os.path.getsize(filepath) / 1024:.1f} KB" if os.path.exists(filepath) else "")

    @staticmethod
    def load_model(filepath, device='cpu'):
        """
        加载模型

        参数:
            filepath: 模型文件路径
            device: 加载设备

        返回:
            加载的模型
        """
        # 从指定文件路径加载PyTorch模型检查点到目标设备
        # 该操作将把保存的模型状态字典、优化器状态等信息加载到指定的计算设备上
        checkpoint = torch.load(filepath, map_location=device)

        # 重新创建模型
        config = checkpoint['model_config']
        loaded_model = ANN(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            output_dim=config['output_dim'],
            task_type=checkpoint.get('task_type', 'regression'),
            dropout_rate=config['dropout_rate'],
            use_batchnorm=config['use_batchnorm']
        )

        # 加载模型参数
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model = loaded_model.to(device)

        print(f"模型已从 {filepath} 加载")
        print(f"模型配置: {config}")
        return loaded_model, checkpoint.get('additional_info', {})
    # 设置随机数种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
# 保存模型
def save_model(model: torch.nn.Module, save_path: str, _train_dict: dict):
    file_manager = ModelManager(model)
    os.makedirs('saved_models', exist_ok=True)
    file_manager.save_model(model,
                            save_path,
                            additional_info={
                                'training_history': _train_dict }
                            )
def train_model(model: torch.nn.Module):
    print("\n=== 训练配置 ===")
    # 学习率和权重衰减设置
    training_config = TrainingConfigurator(model,device, learning_rate=0.001, weight_decay=0.0001)
    criterion, optimizer, scheduler, model = training_config()
    #初始化模型训练器对象
    trainer = ModelTrainer(model,criterion,optimizer,scheduler,device,epochs=200)
    train_dict=trainer.train(train_dataloader,val_dataloader, print_interval=50)
    return trainer,train_dict
    # 导出结果
    # save_model(model, 'saved_models/ANN.pth', train_dict)

if __name__ == '__main__':

       # 加载CSV数据文件并进行预处理
    file_path = PROJECT_ROOT / "outputs/metrics/Results2.csv"
    data = pd.read_csv(file_path,encoding='utf-8',header=0,index_col=0)
    # 删除不需要的列：标签和费雷特相关特征
    data.drop(columns=['Label','FeretX','FeretY','FeretAngle',"MinFeret"],inplace=True,errors='ignore')
    # 分离特征变量X（前4列）和目标变量Y（第5列及以后）
    x = data.iloc[:, 0:4]
    y = data.iloc[:, 4:]
    # 创建数据预处理器实例，划分训练集、验证集和测试集
    data_preprocessor = DataPreprocessor(x, y, test_size=0.2)
    # 创建训练和验证数据加载器
    train_dataloader, val_dataloader = data_preprocessor.create_dataloaders(batch_size=10, shuffle=True)
    # 获取输入维度和输出维度
    input_dim = data_preprocessor.get_input_dim()
    output_dim = data_preprocessor.get_output_dim()
    # 设置随机数种子
    set_seed(42)
    isp = Inspector()
    # 初始化设备管理器
    device_manager = DeviceManager(device_id=None)
    device_id=device_manager.device_id
    # 获取设备
    device = device_manager.get_device()
    # todo:定义训练模型
    model = ANN(input_dim,
                (12, 16, 28),
                output_dim, activation='relu',
                use_batchnorm=True,
                task_type='regression')
    # todo：训练数据集
    trainer, train_dict=train_model(model)
    # todo 可视化训练结果,并保存结果
    trainer.visualize_training_results(save_path=PROJECT_ROOT / "outputs/figures/visualization.png")

    print(trainer.get_training_results(output_format='str'))
    # print(trainer.get_y_pred())
    # trainer.export_training_results_to_csv('D:\\Python\\ANNProject\\visualization')
