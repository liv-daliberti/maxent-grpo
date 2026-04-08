"""Unit tests for the Hydra config validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from maxent_grpo.cli.config_validation import (
    validate_training_config,
)
from maxent_grpo.config import GRPOConfig


def test_validate_training_config_rejects_conflicting_maxent_knobs():
    cfg = GRPOConfig()
    cfg.objective = "grpo"
    cfg.train_grpo_objective = True
    cfg.maxent_tau = 0.1
    with pytest.raises(ValueError) as exc:
        validate_training_config(cfg, command="train-baseline")
    assert "maxent_tau" in str(exc.value)


def test_validate_training_config_allows_maxent_when_flagged():
    cfg = GRPOConfig()
    cfg.objective = "maxent_listwise"
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


def test_validate_training_config_accepts_explicit_listwise_variant() -> None:
    overrides = {
        "objective": "maxent_listwise",
        "maxent_tau": 0.3,
        "num_generations": 2,
        "per_device_train_batch_size": 2,
    }
    validate_training_config(overrides, command="train-maxent")


def test_validate_training_config_rejects_unknown_objective_variant() -> None:
    overrides = {
        "objective": "not-real",
    }
    with pytest.raises(ValueError, match="objective"):
        validate_training_config(overrides, command="train-maxent")


def test_validate_training_config_allows_shared_grpo_scoring_knobs() -> None:
    overrides = {
        "objective": "grpo",
        "maxent_reference_logprobs_source": "model",
        "maxent_logprob_chunk_size": 8,
        "maxent_backward_chunk_size": 4,
        "maxent_allow_empty_weight_fallback": True,
        "maxent_alpha": 0.0,
    }
    validate_training_config(overrides, command="train-baseline")


def test_validate_training_config_allows_grpo_entropy_bonus_only_knobs() -> None:
    overrides = {
        "objective": "grpo_entropy_bonus",
        "policy_entropy_bonus_coef": 0.1,
        "maxent_policy_entropy": True,
        "maxent_policy_entropy_mode": "sample",
    }
    validate_training_config(overrides, command="train-maxent")


def test_validate_training_config_rejects_listwise_knobs_for_grpo_entropy_bonus() -> None:
    overrides = {
        "objective": "grpo_entropy_bonus",
        "policy_entropy_bonus_coef": 0.1,
        "maxent_tau": 0.3,
    }
    with pytest.raises(ValueError) as exc:
        validate_training_config(overrides, command="train-maxent")
    assert "maxent_tau" in str(exc.value) or "maxent_objective_variant" in str(exc.value)


def test_validate_training_config_rejects_sample_entropy_mode_for_entropy_loss() -> None:
    overrides = {
        "objective": "maxent_entropy",
        "maxent_alpha": 0.2,
        "maxent_policy_entropy_mode": "sample",
    }
    with pytest.raises(ValueError, match="requires maxent_policy_entropy_mode='exact'"):
        validate_training_config(overrides, command="train-maxent")


@pytest.mark.parametrize(
    "overrides",
    [
        {"objective": "maxent_entropy", "maxent_reward_signal_gate": True},
        {"objective": "maxent_entropy", "maxent_bonus_positive_only": True},
        {"objective": "maxent_entropy", "maxent_cusp_gate": True},
    ],
)
def test_validate_training_config_rejects_unsupported_reward_shaping_knobs(
    overrides,
) -> None:
    with pytest.raises(ValueError, match="Removed training keys are no longer supported"):
        validate_training_config(overrides, command="train-maxent")


def test_validate_training_config_rejects_listwise_without_whole_prompt_groups() -> None:
    overrides = {
        "objective": "maxent_listwise",
        "maxent_tau": 0.3,
        "num_generations": 4,
        "per_device_train_batch_size": 2,
    }
    with pytest.raises(ValueError, match="whole prompt groups"):
        validate_training_config(overrides, command="train-maxent")


def test_validate_training_config_rejects_listwise_without_positive_tau() -> None:
    overrides = {
        "objective": "maxent_listwise",
        "maxent_tau": 0.0,
    }
    with pytest.raises(ValueError, match="maxent_tau > 0"):
        validate_training_config(overrides, command="train-maxent")


def test_validate_training_config_rejects_listwise_alpha() -> None:
    overrides = {
        "objective": "maxent_listwise",
        "maxent_tau": 0.3,
        "maxent_alpha": 0.2,
    }
    with pytest.raises(ValueError, match="does not use maxent_alpha"):
        validate_training_config(overrides, command="train-maxent")


def test_validate_training_config_accepts_grpo_parity_preset_payloads() -> None:
    repo = Path(__file__).resolve().parents[2]
    preset_paths = [
        repo / "configs/recipes/hydra/grpo_custom_math.yaml",
        repo / "configs/recipes/hydra/grpo_custom_code_mbpp.yaml",
        repo / "configs/recipes/hydra/seed_grpo_math.yaml",
    ]
    for preset_path in preset_paths:
        with preset_path.open("r", encoding="utf-8") as handle:
            preset = yaml.safe_load(handle)
        assert isinstance(preset, dict)
        block = preset["maxent"]
        with (repo / block["recipe"]).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        assert isinstance(payload, dict)
        payload.update(block.get("training", {}))
        validate_training_config(payload, command="train-maxent", source=str(preset_path))


def test_validate_training_config_rejects_seed_grpo_on_maxent_objective() -> None:
    overrides = {
        "objective": "maxent_entropy",
        "maxent_alpha": 0.2,
        "seed_grpo_enabled": True,
    }
    with pytest.raises(ValueError, match="seed_grpo_enabled requires objective=grpo"):
        validate_training_config(overrides, command="train-maxent")


def test_validate_training_config_accepts_baseline_preset_payloads() -> None:
    repo = Path(__file__).resolve().parents[2]
    preset_paths = [
        repo / "configs/recipes/hydra/baseline_math.yaml",
        repo / "configs/recipes/hydra/baseline_code_mbpp.yaml",
    ]
    for preset_path in preset_paths:
        with preset_path.open("r", encoding="utf-8") as handle:
            preset = yaml.safe_load(handle)
        assert isinstance(preset, dict)
        block = preset["baseline"]
        with (repo / block["recipe"]).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        assert isinstance(payload, dict)
        payload.update(block.get("training", {}))
        validate_training_config(
            payload, command="train-baseline", source=str(preset_path)
        )


def test_validate_training_config_accepts_listwise_preset_payloads() -> None:
    repo = Path(__file__).resolve().parents[2]
    preset_paths = [
        repo / "configs/recipes/hydra/maxent_listwise_math.yaml",
        repo / "configs/recipes/hydra/maxent_listwise_code_mbpp.yaml",
    ]
    for preset_path in preset_paths:
        with preset_path.open("r", encoding="utf-8") as handle:
            preset = yaml.safe_load(handle)
        assert isinstance(preset, dict)
        block = preset["maxent"]
        with (repo / block["recipe"]).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        assert isinstance(payload, dict)
        payload.update(block.get("training", {}))
        validate_training_config(payload, command="train-maxent", source=str(preset_path))
