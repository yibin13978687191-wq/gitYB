"""Compatibility wrapper for the reorganized ANN package.

New code should import from ``ann_project.models``. This file keeps older
notebooks and scripts that import ``BPNetwork`` working during the transition.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ann_project.models.bp_network import *  # noqa: F401,F403

