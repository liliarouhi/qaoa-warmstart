"""Shared helpers for experiment scripts: config loading and reproducible random streams."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # run from a fresh clone without installing


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["db_path"] = str((ROOT / cfg["db"]).resolve())
    return cfg


def rng_for(seed: int, *keys) -> np.random.Generator:
    """Independent generator per (seed, keys): adding a family or size never changes the others."""
    words = [seed] + [int.from_bytes(str(k).encode(), "little") % (2**32) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(words))
