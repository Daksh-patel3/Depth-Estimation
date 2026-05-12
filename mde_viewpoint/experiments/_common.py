"""Shared utilities for experiment scripts (config loading, seeding)."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load and return a YAML config file as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_project_root_on_path() -> None:
    """Make the project root importable when scripts are run as files.

    Allows ``python experiments/run_zero_shot_eval.py`` to resolve
    ``from mde_viewpoint...`` style imports.
    """
    here = Path(__file__).resolve()
    # experiments/_common.py is two levels under the project root
    # (project_root / mde_viewpoint / experiments / _common.py).
    project_root = here.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
