"""向后兼容的评估框架适配层。

职责:
- NeuralNetworkAnalyzer — 旧版分析器的兼容包装，内部委托给新的 EvaluationEngine
- AnalysisMixin — 供模型类混入使用的 mixin，将评估方法注入模型

迁移方向:
    这个模块是为渐进式重构保留的过渡层。
    新代码建议直接使用 EvaluationEngine + BaseNeuralNetwork 的组合方式，
    而不是通过混入类注入评估能力。
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch

from .config import EvaluationConfig
from .engine import EvaluationEngine


class NeuralNetworkAnalyzer(EvaluationEngine):
    """旧版分析器的兼容包装，继承 EvaluationEngine 并提供便捷方法。

    这是向前兼容的过渡接口。新代码应直接使用 EvaluationEngine。

    用法:
        analyzer = NeuralNetworkAnalyzer(task_type="regression")
        analyzer.record_batch_predictions(y_true, y_pred, split="train")
        stats = analyzer.compute_epoch_statistics(epoch=1, split="train")
        report = analyzer.generate_report(output_format="dict")
    """

    def __init__(self, task_type: str = "regression", confidence_level: float = 0.95):
        """初始化分析器。

        Args:
            task_type: 任务类型（"regression" 或 "classification"）。
            confidence_level: 置信区间的置信水平（默认 0.95 = 95%）。
        """
        config = EvaluationConfig(task_type=task_type, confidence_level=confidence_level)
        # 委托 Engine 基类初始化，由其创建 RegressionMetrics/ClassificationMetrics 等子组件
        super().__init__(config)
        # 显示存储 task_type，方便外部直接读取
        self.task_type = task_type

    def compute_regression_statistics(
        self, y_true, y_pred, mode: str = "train", epoch: int = 0
    ) -> Dict[str, Any]:
        """便捷方法：直接计算回归指标（不积累历史）。

        适用于只需单次评估、不需要跟踪历史的场景。
        """
        return self.regression_metrics.compute(y_true, y_pred, mode=mode, epoch=epoch)

    def compute_classification_statistics(
        self, y_true, y_pred, epoch: int = 0
    ) -> Dict[str, Any]:
        """便捷方法：直接计算分类指标（不积累历史）。"""
        return self.classification_metrics.compute(y_true, y_pred, epoch=epoch)

    def compute_epoch_statistics(self, epoch: int = 0, split: str = "val") -> Dict[str, Any]:
        """累加该 split 所有缓存的 batch 预测，计算一个 epoch 的汇总统计。

        每次调用后自动清空缓存，避免跨 epoch 累积。
        """
        return super().compute_epoch_statistics(epoch=epoch, split=split)

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取全周期性能汇总（.to_dict 格式）。"""
        history = (
            self.regression_history
            if self.task_type == "regression"
            else self.classification_history
        )
        return self.reporter.to_dict(history)

    def generate_report(
        self, output_format: str = "dict"
    ) -> Union[Dict[str, Any], str, pd.DataFrame]:
        """生成评估报告，支持多种输出格式。"""
        return super().generate_report(output_format)


class AnalysisMixin:
    """供模型类混入使用的评估 mixin。

    在新架构中，BaseNeuralNetwork 已通过组合方式实现等同功能。
    此 mixin 仅为向后兼容保留，新的模型类应直接使用
    EvaluationEngine 并与训练器组合。

    用法（旧式混入）:
        class MyModel(nn.Module, AnalysisMixin):
            def __init__(self):
                nn.Module.__init__(self)
                AnalysisMixin.__init__(self, task_type="regression")
    """

    def __init__(self, task_type: Optional[str] = None, *args, **kwargs):
        """初始化分析能力。

        Args:
            task_type: 任务类型，未指定时从 self.task_type 读取，默认 "regression"。
            enable_analysis: 通过 kwargs 传入，控制分析器是否激活（默认 True）。
        """
        self._analysis_enabled = kwargs.pop("enable_analysis", True)

        # 确定任务类型：优先显式传入，其次从已有属性读取，最后默认 "regression"
        task_type = task_type or getattr(self, "task_type", "regression")
        if task_type not in {"regression", "classification"}:
            raise ValueError(
                f"task_type 必须是 'regression' 或 'classification'，收到: {task_type}"
            )

        # 记录任务类型，使 enable_analysis() 等后续方法可用
        self.task_type = task_type

        # 按需初始化分析器
        if self._analysis_enabled:
            self.analyzer = NeuralNetworkAnalyzer(task_type=task_type)
        else:
            self.analyzer = None

    # ── 内部属性：统一的 "分析器是否可用" 判断 ──

    @property
    def _analyzer_ready(self) -> bool:
        """分析器可用检查：已启用 且 初始化了 analyzer 对象。"""
        return self._analysis_enabled and hasattr(self, "analyzer") and self.analyzer is not None

    # ── 开关控制 ──

    def enable_analysis(self) -> None:
        """启用评估分析。若尚未初始化分析器则自动创建。"""
        self._analysis_enabled = True
        if not self._analyzer_ready:
            task_type = getattr(self, "task_type", "regression")
            self.analyzer = NeuralNetworkAnalyzer(task_type=task_type)

    def disable_analysis(self) -> None:
        """禁用评估分析。"""
        self._analysis_enabled = False

    # ── 数据记录与统计 ──

    def record_predictions(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = "train"
    ) -> None:
        """记录一个 batch 的预测结果到分析器缓存中。"""
        if self._analyzer_ready:
            self.analyzer.record_batch_predictions(y_true, y_pred, split)

    def get_statistics(self, epoch: int = 0, split: str = "val") -> Dict[str, Any]:
        """获取当前 epoch 的汇总统计（清空缓存）。"""
        if self._analyzer_ready:
            return self.analyzer.compute_epoch_statistics(epoch=epoch, split=split)
        return {}

    def get_analysis_report(self, output_format: str = "dict") -> Any:
        """获取完整评估报告。"""
        if self._analyzer_ready:
            return self.analyzer.generate_report(output_format)
        return {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取全周期性能汇总。"""
        if self._analyzer_ready:
            return self.analyzer.get_performance_summary()
        return {}

    # ── 状态持久化（使用 pickle，仅限同一 Python 版本间使用） ──

    def save_analysis(self, filepath: str) -> None:
        """将评估历史保存到文件。

        注意: 使用 pickle 序列化，仅限相同 Python 版本间加载。
        跨版本或长期归档建议使用 .generate_report('dataframe') 导出 CSV。
        """
        if not self._analyzer_ready:
            return
        with open(filepath, "wb") as handle:
            pickle.dump(
                {
                    "task_type": self.analyzer.task_type,
                    "regression_history": self.analyzer.regression_history,
                    "classification_history": self.analyzer.classification_history,
                },
                handle,
            )

    def load_analysis(self, filepath: str) -> None:
        """从 pickle 文件加载评估历史，恢复到分析器中。

        Args:
            filepath: pickle 文件路径。

        注意:
           - 加载仅填充历史数据，不会重新计算指标。
           - 分析器的 task_type 会被文件中的值覆盖。
        """
        if not self._analyzer_ready:
            return
        with open(filepath, "rb") as handle:
            data = pickle.load(handle)
        self.analyzer.task_type = data.get("task_type", self.analyzer.task_type)
        self.analyzer.regression_history = data.get("regression_history", [])
        self.analyzer.classification_history = data.get("classification_history", [])
