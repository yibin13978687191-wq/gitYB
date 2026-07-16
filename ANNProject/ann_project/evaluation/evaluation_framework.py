"""此模块已移除——请使用 ann_project.evaluation.Engine 替代。

从 ann_project.evaluation 导入:
  - EvaluationConfig, EvaluationEngine, EvaluationTracker
  - RegressionMetrics, ClassificationMetrics
  - EvaluationReporter

如需要旧接口 NeuralNetworkAnalyzer / AnalysisMixin 的功能:
  - EvaluationEngine 已包含 compute_regression_statistics() /
    compute_classification_statistics() / generate_report() 等方法
  - 使用 BaseNeuralNetwork + EvaluationEngine 组合替代 AnalysisMixin
"""

raise ImportError(
    "evaluation_framework.py 已移除。请从 ann_project.evaluation 直接导入:\n"
    "  from ann_project.evaluation import EvaluationConfig, EvaluationEngine\n"
    "  engine = EvaluationEngine(config=EvaluationConfig(task_type='regression'))\n"
    "  engine.record_batch_predictions(y_true, y_pred, split='train')\n"
    "  stats = engine.compute_epoch_statistics(epoch=1, split='train')"
)
