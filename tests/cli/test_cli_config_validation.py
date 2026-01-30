"""Unit tests for the Hydra config validation helpers."""

from __future__ import annotations

import pytest

from maxent_grpo.cli.config_validation import (
    validate_generation_config,
    validate_inference_config,
    validate_training_config,
)
from maxent_grpo.cli.hydra_cli import InferenceCommand
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


def test_validate_training_config_rejects_info_seed_on_baseline():
    cfg = GRPOConfig()
    cfg.info_seed_enabled = True
    with pytest.raises(ValueError) as exc:
        validate_training_config(cfg, command="train-baseline")
    assert "InfoSeed" in str(exc.value)


def test_validate_training_config_requires_info_seed_for_infoseed_command():
    cfg = GRPOConfig()
    cfg.info_seed_enabled = False
    with pytest.raises(ValueError) as exc:
        validate_training_config(cfg, command="train-infoseed")
    assert "info_seed_enabled" in str(exc.value)


def test_validate_training_config_flags_info_seed_overrides_without_flag():
    cfg = GRPOConfig()
    cfg.info_seed_lambda = 0.5
    with pytest.raises(ValueError) as exc:
        validate_training_config(cfg, command="train-maxent")
    assert "info_seed_lambda" in str(exc.value)


def test_validate_training_config_allows_info_seed_when_enabled():
    cfg = GRPOConfig()
    cfg.info_seed_enabled = True
    cfg.info_seed_lambda = 0.25
    validate_training_config(cfg, command="train-infoseed")


def test_validate_generation_config_requires_minimal_fields():
    with pytest.raises(ValueError) as exc:
        validate_generation_config({}, command="generate")
    assert "hf_dataset" in str(exc.value)


def test_validate_generation_config_accepts_positive_values():
    payload = {
        "hf_dataset": "demo",
        "model": "m",
        "vllm_server_url": "http://localhost",
        "num_generations": 1,
        "max_new_tokens": 16,
    }
    validate_generation_config(payload, command="generate")


def test_validate_inference_config_requires_models():
    inf_cmd = InferenceCommand(models=[])
    with pytest.raises(ValueError) as exc:
        validate_inference_config(inf_cmd, command="inference")
    assert "models" in str(exc.value)


def test_validate_inference_config_checks_dataset_names():
    inf_cmd = InferenceCommand(models=[{"model_name_or_path": "demo"}], dataset="unknown")
    with pytest.raises(ValueError) as exc:
        validate_inference_config(inf_cmd, command="inference")
    assert "Unknown inference dataset" in str(exc.value)
