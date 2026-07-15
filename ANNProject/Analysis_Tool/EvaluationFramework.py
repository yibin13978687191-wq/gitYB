"""Compatibility wrapper for evaluation tools."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ann_project.evaluation.evaluation_framework import *  # noqa: F401,F403

