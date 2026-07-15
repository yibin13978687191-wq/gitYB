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

