"""Compatibility wrapper for training timer tools."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ann_project.training.training_timer import *  # noqa: F401,F403

