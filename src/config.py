"""Shared config loading and helpers.

Every script reads hyperparameters/paths/seeds/triggers from configs/default.yaml
via this module. No magic numbers live in code (SPEC Section 6).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML config into a plain dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' to mps -> cuda -> cpu based on what is available."""
    import torch

    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility (SPEC Section 6)."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except Exception:
        pass


def abspath(rel: str) -> Path:
    """Resolve a config-relative path against the repo root, making parents."""
    p = REPO_ROOT / rel
    return p
