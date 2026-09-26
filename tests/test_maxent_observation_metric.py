"""The dual controller must observe the estimator its target was measured from.

Three B2b cohorts were discarded to one mistake wearing three disguises: a
target measured as `train/entropy` handed to a controller regulating a
different quantity. Sequence entropy was ~250x the target, the conditional
content-token mean ~10-30x. Each time the coefficient pinned at a bound and the
policy never reached the entropy the arm claimed to hold.

These tests pin the mapping so a fourth disguise cannot appear silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oat_drgrpo.maxent_controllers import MaxEntDualController  # noqa: E402


def _controller(**overrides):
    kwargs = {
        "base_alpha": 0.0001,
        "min_alpha": 0.0001,
        "max_alpha": 0.5,
        "target_ratio": 0.8,
        "warmup_steps": 64,
        "alpha_lr": 0.01,
        "configured_target_entropy": 0.616,
        "entropy_units": "masked_mean_token_nats_v1",
        "observation_metric_key": "entropy",
    }
    kwargs.update(overrides)
    return MaxEntDualController(**kwargs)


def test_masked_mean_units_observe_train_entropy():
    """`train/entropy` is logged as `entropy`; the units must accept that key."""
    controller = _controller()
    assert controller.observation_metric_key == "entropy"
    assert controller.entropy_units == "masked_mean_token_nats_v1"


def test_units_and_observation_must_agree():
    """The pairing is what broke; a mismatch must raise rather than regulate
    the wrong quantity."""
    with pytest.raises(ValueError):
        _controller(observation_metric_key="maxent_conditional_token_entropy")
    with pytest.raises(ValueError):
        _controller(entropy_units="sequence_nats_v1")


def test_absolute_target_is_used_verbatim():
    """A positive configured target bypasses the ratio path, so the number the
    launcher injects is the number regulated."""
    controller = _controller(configured_target_entropy=0.1843)
    assert controller.target_entropy == pytest.approx(0.1843)
