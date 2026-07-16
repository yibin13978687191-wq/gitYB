"""Smoke tests for module imports and basic interfaces."""

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_ann_project_model_imports():
    from ann_project.models import ANN, DataPreprocessor, DeviceManager, ModelTrainer

    assert ANN is not None
    assert DataPreprocessor is not None
    assert DeviceManager is not None
    assert ModelTrainer is not None


def test_evaluation_engine_generate_report_case_insensitive():
    """EvaluationEngine.generate_report 应接受大小写不敏感的 output_format。"""
    from ann_project.evaluation import EvaluationConfig, EvaluationEngine

    engine = EvaluationEngine(config=EvaluationConfig(task_type="regression"))
    result = engine.generate_report(output_format="Dict")

    assert isinstance(result, dict)


def test_evaluation_engine_round_trip():
    """EvaluationEngine 的完整调用链：record_batch → compute_epoch → generate_report。"""
    import numpy as np
    from ann_project.evaluation import EvaluationConfig, EvaluationEngine

    engine = EvaluationEngine(config=EvaluationConfig(task_type="regression"))

    # 模拟 2 个 batch 的回归数据
    engine.record_batch_predictions(
        np.array([1.0, 2.0]), np.array([1.1, 2.2]), split="train"
    )
    engine.record_batch_predictions(
        np.array([3.0, 4.0]), np.array([2.9, 4.1]), split="train"
    )

    stats = engine.compute_epoch_statistics(epoch=0, split="train")
    assert "task_type" in stats
    assert "mse" in stats
    assert stats["sample_size"] == 4

    # 直接评估（不经过缓存）
    direct = engine.compute_regression_statistics(
        np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.0, 3.0])
    )
    assert "mse" in direct

    # 报告输出
    report = engine.generate_report(output_format="dict")
    assert "total_epochs" in report
    assert report["latest_mse"] > 0
