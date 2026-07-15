import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, mean_squared_error,
                             mean_absolute_error, r2_score, explained_variance_score,roc_curve, auc,root_mean_squared_error)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RegressionStatistics:
    """回归问题统计信息"""
    epoch: int
    mode: str
    mse: float
    rmse: float
    mae: float
    r2: float
    explained_variance: float
    confidence_level: float  # 损失置信度 (0-1)
    significance_p_value: float  # 统计显著性 p值
    is_significant: bool  # 是否统计显著 (p < 0.05)
    prediction_interval_width: float  # 预测区间宽度
    confidence_interval: Tuple[float, float]  # 置信区间
    sample_size: int
    @classmethod
    def get_statistics(cls) -> Dict[str, float]:

        return {
            'epoch':cls.epoch,
            'mode':cls.mode,
            'mse':cls.mse,
            'rmse':cls.rmse,
            'mae':cls.mae,
            'r2':cls.r2,
            'explained_variance':cls.explained_variance,
            'confidence_level':cls.confidence_level,
            'significance_p_value':cls.significance_p_value,
            'prediction_interval_width':cls.prediction_interval_width,
            'confidence_interval': cls.confidence_interval,
            'sample_size': cls.sample_size,
        }

@dataclass
class ClassificationStatistics:
    """分类问题统计信息"""

    '''
    epoch (int): 当前训练轮次编号
    accuracy (float): 整体准确率，所有正确预测样本占总样本的比例
    precision_macro (float): 宏平均精确率，各类别精确率的算术平均值
    recall_macro (float): 宏平均召回率，各类别召回率的算术平均值
    f1_macro (float): 宏平均F1分数，基于宏平均精确率和召回率计算得出
    precision_weighted (float): 加权平均精确率，根据各类别样本数量加权计算的精确率
    recall_weighted (float): 加权平均召回率，根据各类别样本数量加权计算的召回率
    f1_weighted (float): 加权平均F1分数，基于加权平均精确率和召回率计算得出
    confusion_matrix (np.ndarray): 混淆矩阵，显示真实标签与预测标签的对应关系
    class_accuracy (Dict[int, float]): 每个类别的准确率，以类别索引为键，准确率为值的字典
    sample_size (int): 样本总数，用于评估的数据样本数量
    num_classes (int): 类别总数，数据集中不同类别的数量
    '''
    epoch: int  #
    accuracy: float #准确率
    precision_macro: float #宏平均精确率
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    confusion_matrix: np.ndarray
    class_accuracy: Dict[int, float]  # 每个类别的准确率
    sample_size: int
    num_classes: int
@dataclass
class TrainingMetrics:
    """训练指标数据容器类"""
    # 训练过程指标
    learning_rates: List[float]
    batch_sizes: List[int]
    train_losses: List[float]
    val_losses: List[float]
    train_accuracies: List[float]
    val_accuracies: List[float]
    # 预测结果
    y_pred: List[np.ndarray]
    target: List[np.ndarray]

        # 在TrainingMetrics.reset()方法中
    def reset(self):
        """重置所有指标"""
        # 当前实现会重新创建列表，可考虑复用已有内存
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], list):
                self.__dict__[key].clear()  # 使用clear()而非重新赋值


class NeuralNetworkAnalyzer:
    """
    神经网络分析与统计模块
    支持回归和分类问题的统计分析
    """

    def __init__(self, task_type: str = 'regression', confidence_level: float = 0.95):
        """
        初始化分析器

        Args:
            task_type: 任务类型 'regression' 或 'classification'
            confidence_level: 置信水平 (0-1)
        """
        assert task_type in ['regression', 'classification'], \
            "task_type 必须是 'regression' 或 'classification'"

        # 初始化任务类型和置信水平
        self.task_type = task_type
        self.confidence_level = confidence_level
        # 根据置信水平计算对应的Z分数（标准正态分布的分位数）
        self.z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # 存储历史统计信息
        self.regression_history: List[RegressionStatistics] = []
        self.classification_history: List[ClassificationStatistics] = []

        # 缓存预测结果用于后续分析
        self.predictions_cache = {
            'train': {'y_true': [], 'y_pred': []},
            'val': {'y_true': [], 'y_pred': []},
            'test': {'y_true': [], 'y_pred': []}
        }

        # 批处理统计
        self.batch_statistics = defaultdict(list)

    def clear_cache(self):
        """清空缓存"""
        for split in self.predictions_cache:
            self.predictions_cache[split]['y_true'].clear()
            self.predictions_cache[split]['y_pred'].clear()
        self.batch_statistics.clear()

    def record_batch_predictions(self,
                                 y_true: torch.Tensor,
                                 y_pred: torch.Tensor,
                                 split: str = 'train'):
        """
        记录批处理的预测结果

        Args:
            y_true: 真实标签
            y_pred: 预测值/概率
            split: 数据分割 ('train', 'val', 'test')
        """
        assert split in ['train', 'val', 'test'], "split 必须是 'train', 'val' 或 'test'"

        # 转换为numpy数组
        y_true_np = y_true.detach().cpu().numpy()

        if self.task_type == 'classification':
            # 对于分类问题，获取预测类别
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # 多分类：取最大概率的类别
                y_pred_np = torch.argmax(y_pred, dim=1).detach().cpu().numpy()
            else:
                # 二分类：阈值0.5
                y_pred_np = (torch.sigmoid(y_pred) > 0.5).long().detach().cpu().numpy()
        else:
            # 回归问题：直接使用预测值
            y_pred_np = y_pred.detach().cpu().numpy()

        # 展平数组
        y_true_np = y_true_np.reshape(-1)
        y_pred_np = y_pred_np.reshape(-1)

        # 存储到缓存
        self.predictions_cache[split]['y_true'].append(y_true_np)
        self.predictions_cache[split]['y_pred'].append(y_pred_np)

    def compute_regression_statistics(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      mode: str = 'train',
                                      epoch: int = 0) -> RegressionStatistics:
        """
        计算回归问题的统计信息

        Returns:
            RegressionStatistics: 回归统计信息
        """
        # 确保是一维数组
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        # 基础指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        explained_variance = explained_variance_score(y_true, y_pred)

        # 计算残差
        residuals = y_true - y_pred

        # 1. 损失置信度计算
        # 基于残差的分布计算置信度
        residual_std = np.std(residuals, ddof=1)
        n_samples = len(residuals)

        # 标准误差
        se = residual_std / np.sqrt(n_samples)

        # 置信区间
        ci_lower = mse - self.z_score * se
        ci_upper = mse + self.z_score * se
        ci_width = ci_upper - ci_lower

        # 置信度评分 (区间越窄置信度越高)
        confidence_level_score = 1 / (1 + ci_width / mse) if mse > 0 else 1.0

        # 2. 统计显著性检验
        # t检验：检验残差均值是否显著不为0
        if n_samples > 1 and residual_std > 0:
            t_statistic, p_value = stats.ttest_1samp(residuals, 0)
            is_significant = p_value < 0.05
        else:
            t_statistic, p_value = 0.0, 1.0
            is_significant = False

        # 3. 预测区间宽度
        # 基于残差标准差计算预测区间
        prediction_interval_width = 2 * self.z_score * residual_std

        return RegressionStatistics(
            epoch=epoch,
            mode=mode,
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2=r2,
            explained_variance=explained_variance,
            confidence_level=float(np.clip(confidence_level_score, 0, 1)),
            significance_p_value=float(p_value),
            is_significant=is_significant,
            prediction_interval_width=float(prediction_interval_width),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            sample_size=n_samples
        )
    @staticmethod
    def compute_classification_statistics(
                                          y_true: np.ndarray,
                                          y_pred: np.ndarray,
                                          epoch: int = 0) -> ClassificationStatistics:
        """
        计算分类问题的统计信息

        Returns:
            ClassificationStatistics: 分类统计信息
        """
        # 确保是一维数组
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        n_samples = len(y_true)
        num_classes = len(np.unique(y_true))

        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)

        # 多类别指标（宏平均和加权平均）
        try:
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except Exception as e:
            # 如果计算失败，使用默认值
            warnings.warn(f"计算分类指标时出错: {e}")
            precision_macro = recall_macro = f1_macro = 0.0
            precision_weighted = recall_weighted = f1_weighted = 0.0

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 每个类别的准确率
        class_accuracy = {}
        for class_idx in range(num_classes):
            class_mask = y_true == class_idx
            if np.sum(class_mask) > 0:
                class_correct = np.sum((y_true == class_idx) & (y_pred == class_idx))
                class_accuracy[class_idx] = class_correct / np.sum(class_mask)
            else:
                class_accuracy[class_idx] = 0.0

        return ClassificationStatistics(
            epoch=epoch,
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
            confusion_matrix=cm,
            class_accuracy=class_accuracy,
            sample_size=n_samples,
            num_classes=num_classes
        )

    def compute_epoch_statistics(self, epoch: int = 0, split: str = 'val') -> Dict:
        """
        计算一个epoch的完整统计信息

        Args:
            epoch: 当前epoch编号
            split: 使用哪个数据分割的预测结果

        Returns:
            Dict: 包含所有统计信息的字典
        """
        # 合并所有批次的预测结果
        y_true_all = np.concatenate(self.predictions_cache[split]['y_true'])
        y_pred_all = np.concatenate(self.predictions_cache[split]['y_pred'])

        if len(y_true_all) == 0:
            warnings.warn(f"没有 {split} 数据的预测结果")
            return {}

        if self.task_type == 'regression':
            #返回回归统计信息的数据类
            stats_obj = self.compute_regression_statistics(y_true_all, y_pred_all, split, epoch)

            self.regression_history.append(stats_obj)

            # 转换为字典返回
            result = {
                'task_type': 'regression',
                'epoch': stats_obj.epoch,
                'mode': split,
                'mse': stats_obj.mse,
                'rmse': stats_obj.rmse,
                'mae': stats_obj.mae,
                'r2': stats_obj.r2,
                'explained_variance': stats_obj.explained_variance,
                'confidence_level': stats_obj.confidence_level,
                'significance_p_value': stats_obj.significance_p_value,
                'is_significant': stats_obj.is_significant,
                'prediction_interval_width': stats_obj.prediction_interval_width,
                'confidence_interval': stats_obj.confidence_interval,
                'sample_size': stats_obj.sample_size
            }
        else:  # classification
            stats_obj = self.compute_classification_statistics(y_true_all, y_pred_all, epoch)
            self.classification_history.append(stats_obj)

            result = {
                'task_type': 'classification',
                'epoch': stats_obj.epoch,
                'accuracy': stats_obj.accuracy,
                'precision_macro': stats_obj.precision_macro,
                'recall_macro': stats_obj.recall_macro,
                'f1_macro': stats_obj.f1_macro,
                'precision_weighted': stats_obj.precision_weighted,
                'recall_weighted': stats_obj.recall_weighted,
                'f1_weighted': stats_obj.f1_weighted,
                'confusion_matrix': stats_obj.confusion_matrix.tolist(),
                'class_accuracy': stats_obj.class_accuracy,
                'sample_size': stats_obj.sample_size,
                'num_classes': stats_obj.num_classes
            }

        # 清空当前split的缓存，为下一个epoch做准备
        self.predictions_cache[split]['y_true'].clear()
        self.predictions_cache[split]['y_pred'].clear()

        return result

    def compute_cross_validation_statistics(self, k_folds: int = 5) -> Dict:
        """
        计算交叉验证统计信息（如果数据已分割）

        Note: 需要在训练前记录不同fold的预测结果

        Returns:
            Dict: 交叉验证统计信息
        """
        # 这里是一个框架，实际实现需要根据具体的数据分割方式调整
        results = {
            'mean_performance': {},
            'std_performance': {},
            'fold_results': []
        }

        # 示例：假设有k_folds个fold的数据
        for fold in range(k_folds):
            # 获取该fold的预测结果
            # 这里需要根据实际的数据分割方式实现
            fold_key = f'fold_{fold}'

            if fold_key in self.batch_statistics:
                fold_results = self.batch_statistics[fold_key]
                if fold_results:
                    results['fold_results'].append(fold_results[-1])  # 取最后一个epoch的结果

        # 计算均值和标准差
        if results['fold_results']:
            if self.task_type == 'regression':
                mse_values = [r['mse'] for r in results['fold_results']]
                results['mean_performance']['mse'] = np.mean(mse_values)
                results['std_performance']['mse'] = np.std(mse_values)
            else:
                accuracy_values = [r['accuracy'] for r in results['fold_results']]
                results['mean_performance']['accuracy'] = np.mean(accuracy_values)
                results['std_performance']['accuracy'] = np.std(accuracy_values)

        return results

    def get_performance_summary(self) -> Dict:
        """
        获取性能总结

        Returns:
            Dict: 包含所有历史统计的总结
        """
        if self.task_type == 'regression':
            if not self.regression_history:
                return {}

            cls_latest_history = self.regression_history[-1]
            cls_history = self.regression_history

            summary = {
                'task_type': 'regression',
                'total_epochs': len(cls_history),
                'latest_epoch': cls_latest_history.epoch,
                'latest_mse': cls_latest_history.mse,
                'latest_rmse': cls_latest_history.rmse,
                'latest_mae': cls_latest_history.mae,
                'latest_r2': cls_latest_history.r2,
                'latest_confidence_level': cls_latest_history.confidence_level,
                'best_mse': min([s.mse for s in cls_history]),
                'best_mse_epoch': [s.epoch for s in cls_history][np.argmin([s.mse for s in cls_history])],
                'best_r2': max([s.r2 for s in cls_history]),
                'best_r2_epoch': [s.epoch for s in cls_history][np.argmax([s.r2 for s in cls_history])],
                'mse_trend': [s.mse for s in cls_history],
                'r2_trend': [s.r2 for s in cls_history],
                'confidence_trend': [s.confidence_level for s in cls_history],
                'convergence_analysis': self._analyze_convergence(cls_history)
            }
        else:
            if not self.classification_history:
                return {}

            cls_latest_history = self.classification_history[-1]
            cls_history = self.classification_history

            summary = {
                'task_type': 'classification',
                'total_epochs': len(cls_history),
                'latest_epoch': cls_latest_history.epoch,
                'latest_accuracy': cls_latest_history.accuracy,
                'latest_f1_macro': cls_latest_history.f1_macro,
                'latest_f1_weighted': cls_latest_history.f1_weighted,
                'best_accuracy': max([s.accuracy for s in cls_history]),
                'best_accuracy_epoch': [s.epoch for s in cls_history][np.argmax([s.accuracy for s in cls_history])],
                'best_f1_macro': max([s.f1_macro for s in cls_history]),
                'best_f1_macro_epoch': [s.epoch for s in cls_history][np.argmax([s.f1_macro for s in cls_history])],
                'accuracy_trend': [s.accuracy for s in cls_history],
                'f1_macro_trend': [s.f1_macro for s in cls_history],
                'class_accuracy_distribution': self._analyze_class_distribution(cls_history),
                'confusion_matrix_analysis': self._analyze_confusion_matrices(cls_history)
            }

        return  summary
    @staticmethod
    def _analyze_convergence(history: List[RegressionStatistics]) -> Dict:
        """分析回归模型的收敛性"""
        if len(history) < 3:
            return {'status': '数据不足', 'converged': False}

        mse_values = [s.mse for s in history]
        recent_window = min(5, len(mse_values))

        # 计算最近几个epoch的MSE变化
        recent_mse = mse_values[-recent_window:]
        mse_change = np.std(recent_mse) / np.mean(recent_mse) if np.mean(recent_mse) > 0 else 0
        converged = mse_change < 0.01  # 变化小于1%认为收敛

        return {
            'status': '已收敛' if converged else '未收敛',
            'converged': converged,
            'recent_mse_std': np.std(recent_mse),
            'recent_mse_cv': mse_change,  # 变异系数
            'total_mse_reduction': (mse_values[0] - mse_values[-1]) / mse_values[0] if mse_values[0] > 0 else 0
        }

    def _analyze_class_distribution(self, history: List[ClassificationStatistics]) -> Dict:
        """分析分类准确度的类别分布"""
        if not history:
            return {}

        latest = history[-1]

        return {
            'class_accuracy': latest.class_accuracy,
            'accuracy_std': np.std(list(latest.class_accuracy.values())) if latest.class_accuracy else 0,
            'min_class_accuracy': min(latest.class_accuracy.values()) if latest.class_accuracy else 0,
            'max_class_accuracy': max(latest.class_accuracy.values()) if latest.class_accuracy else 0,
            'imbalance_ratio': self._calculate_imbalance_ratio(latest.confusion_matrix)
        }

    @staticmethod
    def _analyze_confusion_matrices(history: List[ClassificationStatistics]) -> Dict:
        """分析混淆矩阵"""
        if not history:
            return {}

        latest = history[-1]
        cm = latest.confusion_matrix

        # 计算常见指标
        n_classes = cm.shape[0]
        per_class_metrics = {}

        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            per_class_metrics[i] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'support': cm[i, :].sum()
            }

        return {
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'overall_accuracy': np.trace(cm) / cm.sum() if cm.sum() > 0 else 0
        }

    @staticmethod
    def _calculate_imbalance_ratio(matrix: np.ndarray) -> float:
        """计算类别不平衡比例"""
        if matrix.sum() == 0:
            return 0

        class_sizes = matrix.sum(axis=1)
        max_size = class_sizes.max()
        min_size = class_sizes.min()

        return max_size / min_size if min_size > 0 else float('inf')

    def save_statistics(self, filepath: str):
        """保存统计信息到文件"""
        import pickle

        data_to_save = {
            'task_type': self.task_type,
            'regression_history': self.regression_history,
            'classification_history': self.classification_history,
            'predictions_cache': self.predictions_cache
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)

    def load_statistics(self, filepath: str):
        """从文件加载统计信息"""
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.task_type = data['task_type']
        self.regression_history = data['regression_history']
        self.classification_history = data['classification_history']
        self.predictions_cache = data['predictions_cache']

    def generate_report(self, output_format: str = 'dict') -> Union[Dict, str, pd.DataFrame]:
        """
        生成分析报告

        Args:
            output_format: 'dict', 'str', 或 'dataframe'

        Returns:
            指定格式的报告
        """
        summary = self.get_performance_summary()

        if output_format == ('dict'or'Dict'):
            return summary
        elif output_format == ('dataframe' or'pd.Dataframe') :
            if self.task_type == 'regression':
                data = []
                for satistic in self.regression_history:
                    data.append({
                        'epoch': satistic.epoch,
                        'mode':satistic.mode,
                        'mse': satistic.mse,
                        'rmse': satistic.rmse,
                        'mae': satistic.mae,
                        'r2': satistic.r2,
                        'confidence_level': satistic.confidence_level,
                        'significance_p_value': satistic.significance_p_value,
                        'is_significant': satistic.is_significant,
                        'sample_size': satistic.sample_size
                    })
                return pd.DataFrame(data)
            else:
                data = []
                for satistic in self.classification_history:
                    data.append({
                        'epoch': satistic.epoch,
                        'accuracy': satistic.accuracy,
                        'precision_macro': satistic.precision_macro,
                        'recall_macro': satistic.recall_macro,
                        'f1_macro': satistic.f1_macro,
                        'precision_weighted': satistic.precision_weighted,
                        'recall_weighted': satistic.recall_weighted,
                        'f1_weighted': satistic.f1_weighted,
                        'sample_size': satistic.sample_size,
                        'num_classes': satistic.num_classes
                    })
                return pd.DataFrame(data)
        else:  # str format
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("神经网络分析报告")
            report_lines.append("=" * 60)

            if self.task_type == 'regression':
                report_lines.append(f"任务类型: 回归分析")
                if self.regression_history:
                    latest = self.regression_history[-1]
                    report_lines.append(f"\n最新Epoch ({latest.epoch}) 统计:")
                    report_lines.append(f"  数据集: {latest.mode}")
                    report_lines.append(f"  MSE: {latest.mse:.6f}")
                    report_lines.append(f"  RMSE: {latest.rmse:.6f}")
                    report_lines.append(f"  MAE: {latest.mae:.6f}")
                    report_lines.append(f"  R²: {latest.r2:.4f}")
                    report_lines.append(f"  解释方差: {latest.explained_variance:.4f}")
                    report_lines.append(f"  损失置信度: {latest.confidence_level:.4f}")
                    report_lines.append(f"  统计显著性 p值: {latest.significance_p_value:.6f}")
                    report_lines.append(f"  是否显著: {'是' if latest.is_significant else '否'}")
                    report_lines.append(f"  预测区间宽度: {latest.prediction_interval_width:.4f}")
                    report_lines.append(
                        f"  置信区间: [{latest.confidence_interval[0]:.4f}, {latest.confidence_interval[1]:.4f}]")
                    report_lines.append(f"  样本数: {latest.sample_size}")

                    # 添加收敛分析
                    conv = self._analyze_convergence(self.regression_history)
                    report_lines.append(f"\n收敛分析:")
                    report_lines.append(f"  状态: {conv['status']}")
                    report_lines.append(f"  MSE总减少: {conv['total_mse_reduction'] * 100:.1f}%")
            else:
                report_lines.append(f"任务类型: 分类分析")
                if self.classification_history:
                    latest = self.classification_history[-1]
                    report_lines.append(f"\n最新Epoch ({latest.epoch}) 统计:")
                    report_lines.append(f"  准确率: {latest.accuracy:.4f}")
                    report_lines.append(f"  宏平均精确率: {latest.precision_macro:.4f}")
                    report_lines.append(f"  宏平均召回率: {latest.recall_macro:.4f}")
                    report_lines.append(f"  宏平均F1: {latest.f1_macro:.4f}")
                    report_lines.append(f"  加权平均精确率: {latest.precision_weighted:.4f}")
                    report_lines.append(f"  加权平均召回率: {latest.recall_weighted:.4f}")
                    report_lines.append(f"  加权平均F1: {latest.f1_weighted:.4f}")
                    report_lines.append(f"  类别数: {latest.num_classes}")
                    report_lines.append(f"  样本数: {latest.sample_size}")

                    # 添加类别准确率
                    report_lines.append(f"\n各类别准确率:")
                    for class_idx, acc in latest.class_accuracy.items():
                        report_lines.append(f"  类别 {class_idx}: {acc:.4f}")

            report_lines.append("\n" + "=" * 60)

            return "\n".join(report_lines)



class AnalysisMixin:
    """
    分析功能混合类
    可以与其他神经网络类混合使用
    """
    def __init__(self, task_type: str = 'regression', *args, **kwargs):
        self._analysis_initialized = True
        if self._analysis_initialized:
            self.analyzer = NeuralNetworkAnalyzer(task_type=task_type)
            print(f"数据分析模块已初始化 - 问题类型: {task_type}")
        else:
            print("数据分析模块已禁用")
    def enable_analysis(self):
        """启用分析功能"""
        self._analysis_initialized = True
    def disable_analysis(self):
        """禁用分析功能"""
        self._analysis_initialized = False

    def record_predictions(self, y_true: torch.Tensor, y_pred: torch.Tensor, split: str = 'train'):
        """记录预测结果"""
        if hasattr(self, '_analysis_initialized') and self._analysis_initialized:
            self.analyzer.record_batch_predictions(y_true, y_pred, split)

    def get_statistics(self, epoch: int = 0, split: str = 'val'):
        """计算统计信息"""
        if hasattr(self, '_analysis_initialized') and self._analysis_initialized:
            return self.analyzer.compute_epoch_statistics(epoch, split)
        return {}

    def get_analysis_report(self, output_format: str = 'dict'):
        """获取分析报告output_format: str in ['dict','str','pd.DataFrame'    """

        if hasattr(self, '_analysis_initialized') and self._analysis_initialized:
            return self.analyzer.generate_report(output_format)
        return {}
    def get_performance_summary(self)->dict:
        """获取性能摘要"""
        if hasattr(self, '_analysis_initialized') and self._analysis_initialized:
            return self.analyzer.get_performance_summary()
        return {}
    def save_analysis(self, filepath: str):
        """保存分析数据"""
        if hasattr(self, '_analysis_initialized') and self._analysis_initialized:
            self.analyzer.save_statistics(filepath)

    def load_analysis(self, filepath: str):
        """加载分析数据"""
        if hasattr(self, '_analysis_initialized') and self._analysis_initialized:
            self.analyzer.load_statistics(filepath)
