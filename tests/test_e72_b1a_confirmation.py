from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "ops" / "exp_scaling" / "aggregate_e72_b1a_confirmation.py"
SPEC = importlib.util.spec_from_file_location("aggregate_e72_b1a_confirmation", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_registered_verdict_boundaries():
    assert MODULE.verdict([-0.149, 0.149]) == "E"
    assert MODULE.verdict([-0.30, -0.01]) == "W"
    assert MODULE.verdict([0.151, 0.30]) == "B"
    assert MODULE.verdict([-0.20, 0.10]) == "U"
    assert MODULE.verdict([-0.10, 0.20]) == "U"


def test_paired_bootstrap_is_deterministic():
    left = MODULE.percentile_interval(
        [-0.1, 0.0, 0.1, 0.05, -0.05], np.random.default_rng(123)
    )
    right = MODULE.percentile_interval(
        [-0.1, 0.0, 0.1, 0.05, -0.05], np.random.default_rng(123)
    )
    assert left == right
    assert left[0] <= 0 <= left[1]


def test_confirmation_uses_only_fresh_seeds():
    assert MODULE.SEEDS == (48, 49, 50, 51, 52)
    assert set(MODULE.SEEDS).isdisjoint({43, 44, 45, 46, 47})
    assert MODULE.BOOTSTRAP_RESAMPLES == 10_000
    assert MODULE.MARGIN == 0.15
