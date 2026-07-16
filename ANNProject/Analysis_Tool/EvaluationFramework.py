"""评估工具包装器。

导入 ann_project.evaluation 中的新接口。
旧版 evaluation_framework 已移除，请勿使用。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ann_project.evaluation import (  # noqa: F401,F403
    EvaluationConfig,
    EvaluationEngine,
    EvaluationReporter,
    EvaluationTracker,
    RegressionMetrics,
    ClassificationMetrics,
)
