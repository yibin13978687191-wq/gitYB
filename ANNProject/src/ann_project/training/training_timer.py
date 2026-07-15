import os
import time
import numpy as np
from typing import Dict, Tuple, List,Optional
from dataclasses import dataclass, field
import psutil
import torch

@dataclass
class TrainingTimeMetrics:
    """训练时间指标数据类"""
    epoch: int
    epoch_time: float  # 本epoch用时(秒)
    epoch_time_formatted: str  # 格式化时间
    data_loading_time: float  # 数据加载时间
    forward_time: float  # 前向传播时间
    backward_time: float  # 反向传播时间
    optimization_time: float  # 优化器更新时间
    validation_time: float  # 验证时间
    total_time_so_far: float  # 累计总时间
    estimated_time_remaining: Optional[str]  # 预计剩余时间
    memory_usage_mb: float  # 内存使用(MB)
    gpu_memory_mb: Optional[float]  # GPU内存使用(MB)
    @classmethod
    def _get_time_attribute(cls,attribute_name: str):
        time_in_seconds = getattr(cls, attribute_name)
        return TrainingTimer._format_time(time_in_seconds)
class TrainingTimer:
    """
    训练时间统计器
    精确统计每个epoch和总训练时间
    """

    def __init__(self, total_epochs: int):
        """
        初始化时间统计器

        Args:
            total_epochs: 总训练epoch数
        """
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_start_time = None
        self._data_loading_start = None
        self.history: List[TrainingTimeMetrics] = []

        # 时间细分统计
        self.time_components = {
            'data_loading': 0.0,
            'forward': 0.0,
            'backward': 0.0,
            'optimization': 0.0,
            'validation': 0.0,
            'other': 0.0
        }

    def start_epoch(self):
        """开始一个新epoch的计时"""
        self.epoch_start_time = time.time()
        self.time_components = {k: 0.0 for k in self.time_components}

    def record_data_loading(self):
        """记录数据加载时间"""
        if hasattr(self, '_data_loading_start'):
            self.time_components['data_loading'] += time.time() - self._data_loading_start

    def start_data_loading(self):
        """开始数据加载计时"""
        self._data_loading_start = time.time()

    def start_forward(self):
        """开始前向传播计时"""
        self._forward_start = time.time()

    def end_forward(self):
        """结束前向传播计时"""
        self.time_components['forward'] += time.time() - self._forward_start

    def start_backward(self):
        """开始反向传播计时"""
        self._backward_start = time.time()

    def end_backward(self):
        """结束反向传播计时"""
        self.time_components['backward'] += time.time() - self._backward_start

    def start_optimization(self):
        """开始优化器更新计时"""
        self._optimization_start = time.time()

    def end_optimization(self):
        """结束优化器更新计时"""
        self.time_components['optimization'] += time.time() - self._optimization_start

    def start_validation(self):
        """开始验证计时"""
        self._validation_start = time.time()

    def end_validation(self):
        """结束验证计时"""
        self.time_components['validation'] += time.time() - self._validation_start

    def end_epoch(self, epoch: int) -> TrainingTimeMetrics:
        """
        结束当前epoch的计时

        Returns:
            TrainingTimeMetrics: 当前epoch的时间指标
        """
        epoch_time = time.time() - self.epoch_start_time

        # 计算其他时间
        recorded_time = sum(self.time_components.values())
        self.time_components['other'] = max(0, epoch_time - recorded_time)

        # 如果是第一个epoch，设置总开始时间
        if self.start_time is None:
            self.start_time = time.time() - epoch_time

        # 计算累计总时间
        total_time_so_far = time.time() - self.start_time

        # 估计剩余时间
        if epoch > 1:
            avg_epoch_time = total_time_so_far / epoch
            remaining_epochs = self.total_epochs - epoch
            estimated_remaining = avg_epoch_time * remaining_epochs
            estimated_str = self._format_time(estimated_remaining)
        else:
            estimated_str = "计算中..."

        # 获取内存使用情况
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # 获取GPU内存使用（如果可用）
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        # 创建时间指标对象
        metrics = TrainingTimeMetrics(
            epoch=epoch,
            epoch_time=epoch_time,
            epoch_time_formatted=self._format_time(epoch_time),
            data_loading_time=self.time_components['data_loading'],
            forward_time=self.time_components['forward'],
            backward_time=self.time_components['backward'],
            optimization_time=self.time_components['optimization'],
            validation_time=self.time_components['validation'],
            total_time_so_far=total_time_so_far,
            estimated_time_remaining=estimated_str,
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory
        )

        self.history.append(metrics)
        return metrics

    def get_total_time(self) -> float:
        """获取总训练时间"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_average_epoch_time(self) -> float:
        """获取平均每个epoch的时间"""
        if not self.history:
            return 0.0
        return np.mean([m.epoch_time for m in self.history])

    def print_summary(self):
        """打印训练时间总结"""
        if not self.history:
            print("暂无训练时间数据")
            return

        total_time = self.get_total_time()
        avg_epoch_time = self.get_average_epoch_time()

        print("\n" + "=" * 60)
        print("训练时间统计总结")
        print("=" * 60)
        print(f"总训练时间: {self._format_time(total_time)}")
        print(f"总epoch数: {len(self.history)}")
        print(f"平均每个epoch时间: {self._format_time(avg_epoch_time)}")

        if len(self.history) > 1:
            print(f"\n最快epoch: Epoch {np.argmin([m.epoch_time for m in self.history]) + 1} "
                  f"({self._format_time(min([m.epoch_time for m in self.history]))})")
            print(f"最慢epoch: Epoch {np.argmax([m.epoch_time for m in self.history]) + 1} "
                  f"({self._format_time(max([m.epoch_time for m in self.history]))})")

        # 时间分布
        if len(self.history) > 0:
            last_epoch = self.history[-1]
            print(f"\n最后一个epoch时间分布:")
            print(f"  数据加载: {self._format_time(last_epoch.data_loading_time)} "
                  f"({last_epoch.data_loading_time / last_epoch.epoch_time * 100:.1f}%)")
            print(f"  前向传播: {self._format_time(last_epoch.forward_time)} "
                  f"({last_epoch.forward_time / last_epoch.epoch_time * 100:.1f}%)")
            print(f"  反向传播: {self._format_time(last_epoch.backward_time)} "
                  f"({last_epoch.backward_time / last_epoch.epoch_time * 100:.1f}%)")
            print(f"  优化器更新: {self._format_time(last_epoch.optimization_time)} "
                  f"({last_epoch.optimization_time / last_epoch.epoch_time * 100:.1f}%)")
            print(f"  验证: {self._format_time(last_epoch.validation_time)} "
                  f"({last_epoch.validation_time / last_epoch.epoch_time * 100:.1f}%)")
            other_time = last_epoch.epoch_time - sum([
                last_epoch.data_loading_time,
                last_epoch.forward_time,
                last_epoch.backward_time,
                last_epoch.optimization_time,
                last_epoch.validation_time,
            ])
            other_percentage = 100 - sum([
                last_epoch.data_loading_time / last_epoch.epoch_time * 100,
                last_epoch.forward_time / last_epoch.epoch_time * 100,
                last_epoch.backward_time / last_epoch.epoch_time * 100,
                last_epoch.optimization_time / last_epoch.epoch_time * 100,
                last_epoch.validation_time / last_epoch.epoch_time * 100,
            ])
            print(f"  其他: {self._format_time(other_time)} ({other_percentage:.1f}%)")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.3f}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds %= 60
            return f"{int(minutes)}分{seconds:.0f}秒"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds %= 60
            return f"{int(hours)}时{int(minutes)}分{seconds:.0f}秒"
