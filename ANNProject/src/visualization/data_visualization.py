import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, confusion_matrix,mean_squared_error, mean_absolute_error, r2_score)
import warnings

warnings.filterwarnings('ignore')


# ==================== 1. 数据容器类 ====================

@dataclass
class TrainingMetrics:
    """训练指标数据容器类"""
    # 训练过程指标
    training_metrics={
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'total_sample': [],
        'batch_num': [],
        'y_pred': [],
        'y_true': [],
        'y_pred_val': [],
        'y_true_val': [],
        'learning_rates': [],
        'batch_sizes': [],
        'epoch_avg_loss': 0.0,
    }


    # 分类问题指标
    classification_metrics = {
        "epoch": 0,  # 当前epoch
        "precision": [],
        "accuracy": 0.0,  # 准确率
        "precision_macro": 0.0,  # 宏平均精确率
        "recall_macro": 0.0,# 宏平均召回率
        "f1_macro": 0.0,  # 宏平均F1,
        "precision_weighted": 0.0,  # 宏平均精确率,
        "recall_weighted": 0.0,# 召回率
        "f1_weighted": 0.0,# F1分数
        "confusion_matrix": np.array([]),  # 混淆矩阵,
        "class_accuracy": {},  # 每个类别的准确率
        "sample_size": 0,
        "num_classes": 0,
    }

    # 回归问题指标
    regression_metrics = {
        "epoch": 0,
        "mse": [],
        "rmse": [],
        "mae": [],
        "r2": [],
        "explained_variance": 0.0,
        "confidence_level": 0.0,
        "significance_p_value": 0.0,
        "is_significant": False,
        "prediction_interval_width": 0.0,
        "prediction_interval": np.array([]),
        "sample_size": 0,
    }
    metrics = [training_metrics, classification_metrics, regression_metrics]
    # 附加信息
    @classmethod
    def get_metrics(cls, metric_type: str)-> Dict:
        """
        param in metric_type: 'classification','regression','training'
        """
        if metric_type == 'classification':
            return cls.classification_metrics
        elif metric_type == 'regression':
            return cls.regression_metrics
        elif metric_type == 'training':
            return cls.training_metrics
        else:
            raise ValueError("Invalid type")

    @classmethod
    def update_metrics(cls, update_dict: Dict, metric_type:str, **kwargs):
        """更新指标"""

        if metric_type == 'classification':
            if update_dict is not None:
                cls.classification_metrics.update(update_dict)
            cls.classification_metrics.update(**kwargs)
        elif metric_type == 'regression':
            if update_dict is not None:
                cls.regression_metrics.update(update_dict)
            cls.regression_metrics.update(**kwargs)
        elif metric_type == 'training':
            if update_dict is not None:
                cls.training_metrics.update(update_dict)
            cls.training_metrics.update(**kwargs)

    def reset(self):
        """重置所有指标"""
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], list):
                self.__dict__[key] = []
    @classmethod
    def to_dataframe(cls,metric_type: str) :
        """转换为DataFrame"""
        if metric_type == 'training':
            # 从training_dict中选择指定的键值对
            training_dict = cls.get_metrics('training')
            keys_to_select = ['train_loss', 'y_pred', 'y_true']
            subset_dict = {k: training_dict[k] for k in keys_to_select if k in training_dict}

            return pd.DataFrame(subset_dict)
        elif metric_type == 'classification':
            return pd.DataFrame(cls.classification_metrics)
        elif metric_type == 'regression':
            return pd.DataFrame(cls.regression_metrics)
        else:
            raise ValueError("Invalid type")


# ==================== 2. 可视化模块主体 ====================

class NeuralNetworkVisualizer:
    """神经网络可视化器"""

    def __init__(self, task_type: str = 'classification',
                 figsize: Tuple[int, int] = (15, 10)):
        """
        初始化可视化器

        Args:
            task_type: 问题类型 ('classification' 或 'regression')
            figsize: 图像尺寸
        """
        self.task_type = task_type
        self.figsize = figsize
        self.metrics = TrainingMetrics()
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def _record_classification_metrics(self, predictions: np.ndarray,
                                       ground_truth: np.ndarray):
        """记录分类问题指标"""
        try:
            # 确保是类别标签
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                pred_labels = np.argmax(predictions, axis=1)
                if ground_truth.ndim > 1 and ground_truth.shape[1] > 1:
                    true_labels = np.argmax(ground_truth, axis=1)
                else:
                    true_labels = ground_truth
            else:
                pred_labels = (predictions > 0.5).astype(int)
                true_labels = ground_truth

            # 计算指标
            self.metrics.update_metrics(metric_type='classification',
                                        precision=precision_score(true_labels, pred_labels,
                                                                  average='weighted', zero_division=0))
            self.metrics.update_metrics(metric_type='classification',
                                        recall=recall_score(true_labels, pred_labels,
                                                    average='weighted', zero_division=0))
            self.metrics.update_metrics(metric_type='classification',
                                        f1_scores=f1_score(true_labels, pred_labels,
                                                   average='weighted', zero_division=0))

            # 记录混淆矩阵
            cm = confusion_matrix(true_labels, pred_labels)
            self.metrics.update_metrics(metric_type='classification', confusion_matrix=cm)

        except Exception as e:
            print(f"记录分类指标时出错: {e}")
        '''
    def _record_regression_metrics(self, predictions: np.ndarray,
                                   ground_truth: np.ndarray):
        """记录回归问题指标"""
        try:
            self.metrics.update_metrics(metric_type='regression', mse=mean_squared_error(ground_truth, predictions))
            self.metrics.update_metrics(metric_type='regression', mae=mean_absolute_error(ground_truth, predictions))
            self.metrics.update_metrics(metric_type='regression', r2=r2_score(ground_truth, predictions))
        except Exception as e:
            print(f"记录回归指标时出错: {e}")
        '''
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        fig, axes = plt.subplots(3, 3, figsize=self.figsize)
        axes = axes.flatten()
        epochs = range(len(self.metrics.get_metrics('training')['train_loss']))
        train_loss = self.metrics.get_metrics('training')['train_loss']
        val_loss = self.metrics.get_metrics('training')['val_loss']
        # 1. 损失曲线
        if train_loss is not None:
            axes[0].plot(epochs, train_loss,
                         color=self.colors[0], linewidth=2)
        if val_loss:
            axes[0].plot(epochs,val_loss, label='验证损失',
                         color=self.colors[1], linewidth=2, linestyle='--')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练和验证损失')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 准确率曲线（如果存在）
        if ('train_acc' in self.metrics.get_metrics('training').keys()
                and self.metrics.get_metrics('training')['train_acc'].__len__() > 0):
            axes[1].plot(epochs, self.metrics.get_metrics('training')['train_acc'], label='训练准确率',
                         color=self.colors[2], linewidth=2)
        else:
            axes[1].axis('off')
        if ('val_acc' in self.metrics.get_metrics('training').keys()
                and self.metrics.get_metrics('training')['val_acc'].__len__() > 0):
            axes[1].plot(epochs, self.metrics.get_metrics('training')['val_acc'], label='验证准确率',
                         color=self.colors[3], linewidth=2, linestyle='--')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('准确率')
            axes[1].set_title('训练和验证准确率')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        else:
            axes[1].axis('off')


        # 3. 问题特定指标
        if self.task_type == 'classification' and (self.metrics.get_metrics('classification')is not None):
            epochs = range(len(self.metrics.get_metrics('classification')["f1_weighted"]))
            axes[2].plot(epochs, self.metrics.get_metrics('classification')["precision_weighted"], label='精确率',
                         color=self.colors[4], linewidth=2)
            axes[2].plot(epochs, self.metrics.get_metrics('classification')["recall_weighted"], label='召回率',
                         color=self.colors[5], linewidth=2)
            axes[2].plot(epochs, self.metrics.get_metrics('classification')["f1_weighted"], label='F1分数',
                         color=self.colors[6], linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('分数')
            axes[2].set_title('分类指标')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        elif self.task_type == 'regression' and (self.metrics.get_metrics("regression")is not None):
            epochs = range(len(self.metrics.get_metrics("regression")["mse"]))
            mse = self.metrics.get_metrics("regression")["mse"]
            mae = self.metrics.get_metrics("regression")["mae"]
            r2 = self.metrics.get_metrics("regression")["r2"]
            axes[2].plot(epochs, mse, label='MSE变化曲线',
                         color=self.colors[7], linewidth=2)
            axes[3].plot(epochs, mae, label='MAE变化曲线',
                         color=self.colors[8], linewidth=2)
            axes[4].plot(epochs, r2, label='R²变化曲线',
                         color=self.colors[9], linewidth=2)
            axes[2].set_title('回归指标')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('MSE')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            axes[3].set_title('回归指标')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('MAE')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            axes[4].set_title('回归指标')
            axes[4].set_xlabel('Epoch')
            axes[4].set_ylabel('R²')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)

        else:
            for ax in axes[2:]:
                ax.axis('off')


        # 4. 最终评估图
        if ((self.metrics.get_metrics('training')['y_pred'] is not None)
                and (self.metrics.get_metrics('training')['y_true'] is not None)):
            self._plot_final_evaluation(axes[6])
        else:
            axes[6].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.show()

    def _plot_final_evaluation(self, ax):
        """绘制最终评估图"""
        if (not self.metrics.get_metrics('training') ['y_pred']
                or not self.metrics.get_metrics('training')['y_true']):
            return

        # 使用最后一次的预测结果
        preds = self.metrics.get_metrics('training')['y_pred'][-1].flatten()
        truths = self.metrics.get_metrics('training')['y_true'][-1].flatten()

        if self.task_type == 'classification':
            # 绘制混淆矩阵
            if preds.ndim > 1 and preds.shape[1] > 1:
                pred_labels = np.argmax(preds, axis=1)
                true_labels = np.argmax(truths, axis=1) if truths.ndim > 1 else truths
            else:
                pred_labels = (preds > 0.5).astype(int)
                true_labels = truths.astype(int)

            cm = confusion_matrix(true_labels, pred_labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_title('混淆矩阵')

        elif self.task_type == 'regression':
            # 绘制预测 vs 真实值散点图
            ax.scatter(truths, preds, alpha=0.6, color=self.colors[0])

            # 添加理想线
            min_val = min(truths.min(), preds.min())
            max_val = max(truths.max(), preds.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='理想线')

            # 计算回归线
            z = np.polyfit(truths, preds, 1)
            p = np.poly1d(z)
            ax.plot(truths, p(truths), 'g-', linewidth=2, label='回归线')

            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title('预测 vs 真实值')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def plot_feature_importance(self, model: nn.Module,
                                feature_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
        """绘制特征重要性（适用于第一层权重）"""
        if not hasattr(model, 'layers') and not hasattr(model, 'linear1'):
            print("无法提取模型权重")
            return

        # 尝试获取第一层权重
        weights = None
        if hasattr(model, 'linear1'):
            weights = model.linear1.weight.detach().cpu().numpy()
        elif hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
            for layer in model.layers:
                if isinstance(layer, nn.Linear):
                    weights = layer.weight.detach().cpu().numpy()
                    break

        if weights is None:
            print("未找到线性层")
            return

        # 计算特征重要性（权重绝对值平均）
        importance = np.abs(weights).mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))

        if feature_names and len(feature_names) == len(importance):
            indices = np.arange(len(importance))
            ax.barh(indices, importance, color=self.colors[:len(importance)])
            ax.set_yticks(indices)
            ax.set_yticklabels(feature_names)
        else:
            indices = np.arange(len(importance))
            ax.bar(indices, importance, color=self.colors[:len(importance)])
            ax.set_xlabel('特征索引')

        ax.set_ylabel('重要性（权重绝对值平均）')
        ax.set_title('特征重要性分析')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_report(self, save_path: Optional[str] = None) -> Dict:
        """生成评估报告"""
        report = {
            'task_type': self.task_type,
            'total_epochs': len(self.metrics.get_metrics('training')['train_loss']),
            'final_train_loss': self.metrics.get_metrics('training')['train_loss'][-1]
            if self.metrics.get_metrics('training')['train_loss'] else None,
            'final_val_loss': self.metrics.get_metrics('training')['val_loss'][-1]
            if self.metrics.get_metrics('training')['val_loss'] else None,
        }

        if self.task_type == 'classification':
            if self.metrics.get_metrics('classification')['f1_scores']:
                report.update({
                    'final_precision': self.metrics.get_metrics('classification')['precision'][-1],
                    'final_recall': self.metrics.get_metrics('classification')['recall'][-1],
                    'final_f1_score': self.metrics.get_metrics('classification')['f1_scores'][-1],
                    'confusion_matrix': self.metrics.get_metrics('classification')['confusion_matrix']

                })


        elif self.task_type == 'regression':
            if self.metrics.get_metrics('regression')['r2']:
                report.update({
                    'final_mse': self.metrics.get_metrics('regression')['mse'][-1],
                    'final_mae': self.metrics.get_metrics('regression')['mae'][-1],
                    'final_r2': self.metrics.get_metrics('regression')['r2'][-1],
                    'final_rmse': self.metrics.get_metrics('regression')['rmse'][-1],

                })

        # 保存报告
        if save_path:
            df_report = pd.DataFrame([report])
            df_report.to_csv(save_path, index=False)
            print(f"报告已保存至: {save_path}")

        return report

    def save_all_plots(self, base_path: str):
        """保存所有图表"""
        import os
        os.makedirs(base_path, exist_ok=True)

        # 保存训练历史
        self.plot_training_history(os.path.join(base_path, 'training_history.png'))

        # 保存指标数据
        df_training = self.metrics.to_dataframe("training")
        df_training.to_csv(os.path.join(base_path, 'training_metrics.csv'), index=False)
        if self.task_type == 'classification':
            df_classification = self.metrics.to_dataframe("classification")
            df_classification.to_csv(os.path.join(base_path, 'classification_metrics.csv'), index=False)
        else:
            df_regression = self.metrics.to_dataframe("regression")
            df_regression.to_csv(os.path.join(base_path, 'regression_metrics.csv'), index=False)


        self.generate_report(os.path.join(base_path, 'evaluation_report.csv'))

        print(f"所有结果已保存至: {base_path}")


# ==================== 3. 嵌入接口（混合类） ====================

class VisualMixin:
    """可视化混合类，可以被神经网络模型继承"""

    def __init__(self, task_type: str = 'regression',*args, **kwargs):
        """
        初始化可视化器

        Args:
            task_type: 问题类型
            **kwargs: 传递给Visualizer的额外参数
        """
        super().__init__(*args, **kwargs)
        self.task_type =task_type
        self._is_visualization_enabled = True
        if task_type not in ['regression', 'classification']:
            raise ValueError("任务类型必须是'regression'或'classification'")
        if self._is_visualization_enabled:
            self.visualizer = NeuralNetworkVisualizer(task_type=task_type)
            print(f"可视化器已初始化 - 问题类型: {task_type}")
        else:
            print("可视化已禁用")
    def enable_visualization(self):
        """启用可视化"""
        self._is_visualization_enabled = True

    def disable_visualization(self):
        """禁用可视化"""
        self._is_visualization_enabled = False

    def record_training_step(self, training_step, **kwargs):
        """记录训练步骤"""
        if hasattr(self, '_is_visualization_enabled') and self._is_visualization_enabled:
            training_dict = self.visualizer.metrics.get_metrics('training')
            for metric_name, metric_value in training_step.items():
                if metric_name=='epoch_avg_loss' :
                    training_dict['epoch_avg_loss']=metric_value
                else:
                    training_dict[metric_name].append(metric_value)

    def record_regression_metrics(self, metrics_dict):
        """记录回归指标"""
        if hasattr(self, '_is_visualization_enabled') and self._is_visualization_enabled:
            regression_metrics = self.visualizer.metrics.get_metrics('regression')
            for metric_name, metric_value in metrics_dict.items():
                if metric_name in ['mse', 'mae', 'r2', 'rmse'] and isinstance(metric_value, float):
                    regression_metrics[metric_name].append(metric_value)
                else:
                    regression_metrics[metric_name]=metric_value


    def get_metrics_data(self, metric_type: str) -> Dict:
        """获取指标数据"""
        if hasattr(self, 'visualizer'):
            return self.visualizer.metrics.get_metrics(metric_type)
        else:
            print("请先初始化可视化器")
            return {}
    def plot_results(self, save_path: Optional[str] = None):
        """绘制结果"""
        if hasattr(self, 'visualizer'):
            self.visualizer.plot_training_history(save_path)
        else:
            print("请先初始化可视化器")

    def plot_feature_importance(self, feature_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
        """绘制特征重要性"""
        if hasattr(self, 'visualizer'):
            self.visualizer.plot_feature_importance(self,feature_names, save_path)
        else:
            print("请先初始化可视化器")

    def save_visualization_results(self, base_path: str):
        """保存可视化结果"""
        if hasattr(self, 'visualizer'):
            self.visualizer.save_all_plots(base_path)
        else:
            print("请先初始化可视化器")

    def get_metrics_report(self) -> Dict:
        """获取指标报告"""
        if hasattr(self, 'visualizer'):
            return self.visualizer.generate_report()
        else:
            print("请先初始化可视化器")
            return {}

    def reset_visualizer(self):
        """重置可视化器"""
        if hasattr(self, 'visualizer'):
            self.visualizer.metrics.reset()
            print("可视化器已重置")


# ==================== 示例神经网络类 ====================



# ==================== 使用示例 ====================




    # # 可视化结果
    # cls_model.plot_results("classification_results.png")
    #
    # # 特征重要性分析
    # feature_names = [f'Feature_{i}' for i in range(10)]
    # cls_model.plot_feature_importance(feature_names, "feature_importance.png")
    #
    # # 保存所有结果
    # cls_model.save_visualization_results("./classification_visualization")
    #
    # # 获取报告
    # report = cls_model.get_metrics_report()
    # print("\n分类模型评估报告:")
    # for key, value in report.items():
    #     print(f"  {key}: {value}")
    #
    # # 3. 回归模型示例
    #
    #
    # # 可视化结果
    # reg_model.plot_results("regression_results.png")
    #
    # # 保存所有结果
    # reg_model.save_visualization_results("./regression_visualization")
    #
    # print("\n回归模型评估报告:")
    # reg_report = reg_model.get_metrics_report()
    # for key, value in reg_report.items():
    #     print(f"  {key}: {value}")
    #
    # print("\n所有示例完成!")




