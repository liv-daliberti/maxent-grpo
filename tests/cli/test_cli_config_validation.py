"""Unit tests for the Hydra config validation helpers."""

from __future__ import annotations

import pytest

from maxent_grpo.cli.config_validation import (
    validate_training_config,
)
from maxent_grpo.config import GRPOConfig


def test_validate_training_config_rejects_conflicting_maxent_knobs():
    cfg = GRPOConfig()
    cfg.train_grpo_objective = True
    cfg.maxent_tau = 0.1
    with pytest.raises(ValueError) as exc:
        validate_training_config(cfg, command="train-baseline")
    assert "maxent_tau" in str(exc.value)


def test_validate_training_config_allows_maxent_when_flagged():
    cfg = GRPOConfig()
    cfg.train_grpo_objective = False
    cfg.maxent_tau = 0.2
    validate_training_config(cfg, command="train-maxent")


def test_validate_training_config_requires_flag_when_missing():
    overrides = {"maxent_tau": 0.25}
    with pytest.raises(ValueError) as exc:
        validate_training_config(overrides, command="train-baseline")
    assert "train-baseline" in str(exc.value)


def test_validate_training_config_respects_command_defaults():
    overrides = {"maxent_tau": 0.3}
    validate_training_config(overrides, command="train-maxent")
