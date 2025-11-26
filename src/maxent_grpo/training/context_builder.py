"""Helpers to map GRPOConfig into training runtime settings for InfoSeed runs."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from maxent_grpo.config import GRPOConfig
from .types import (
    EvaluationSettings,
    GenerationSettings,
    ScoringSettings,
)
from .runtime import SeedAugmentationConfig


def _build_seed_aug(cfg: GRPOConfig) -> Optional[SeedAugmentationConfig]:
    """Return SeedAugmentationConfig when InfoSeed is enabled."""

    if not getattr(cfg, "info_seed_enabled", False):
        return None
    num_seeds = max(int(getattr(cfg, "info_seed_num_seeds", 0)), 0)
    if num_seeds <= 0:
        return None
    return SeedAugmentationConfig(
        enabled=True,
        num_seeds=num_seeds,
        completions_per_seed=1,
        template=getattr(cfg, "info_seed_prompt_template", "\n[seed={seed}]"),
        include_original=True,
    )


def apply_info_seed_to_generation(
    gen_settings: GenerationSettings, cfg: GRPOConfig
) -> GenerationSettings:
    """Attach seed augmentation to GenerationSettings."""

    seed_aug = _build_seed_aug(cfg)
    if seed_aug is None:
        return gen_settings
    return replace(gen_settings, seed_augmentation=seed_aug)


def apply_info_seed_to_scoring(
    scoring: ScoringSettings, cfg: GRPOConfig
) -> ScoringSettings:
    """Attach InfoSeed loss knobs to ScoringSettings."""

    return replace(
        scoring,
        info_seed_lambda=float(getattr(cfg, "info_seed_lambda", 0.0)),
        info_seed_temperature=float(getattr(cfg, "info_seed_temperature", 0.1)),
        info_seed_loss_type=str(getattr(cfg, "info_seed_loss_type", "infonce")),
        info_seed_pooling=str(getattr(cfg, "info_seed_pooling", "mean")),
        info_seed_alpha_entropy=float(getattr(cfg, "info_seed_alpha_entropy", 0.0)),
    )


def apply_info_seed_to_evaluation(
    eval_settings: EvaluationSettings, cfg: GRPOConfig
) -> EvaluationSettings:
    """Populate EvaluationSettings.seed_eval using GRPOConfig fields when present."""

    if not getattr(cfg, "info_seed_enabled", False):
        return eval_settings
    num_seeds = max(int(getattr(cfg, "info_seed_num_seeds", 0)), 0)
    if num_seeds <= 0:
        return eval_settings
    seed_eval_cfg = {
        "enabled": True,
        "num_seeds": num_seeds,
        "samples_per_seed": 1,
        "template": getattr(cfg, "info_seed_prompt_template", "\n[seed={seed}]"),
        "pooling": str(getattr(cfg, "info_seed_pooling", "mean")),
    }
    return replace(eval_settings, seed_eval=seed_eval_cfg)


def apply_info_seed(
    generation: GenerationSettings,
    scoring: ScoringSettings,
    evaluation: EvaluationSettings,
    cfg: GRPOConfig,
) -> tuple[GenerationSettings, ScoringSettings, EvaluationSettings]:
    """Convenience helper to update all settings with InfoSeed config."""

    generation = apply_info_seed_to_generation(generation, cfg)
    scoring = apply_info_seed_to_scoring(scoring, cfg)
    evaluation = apply_info_seed_to_evaluation(evaluation, cfg)
    return generation, scoring, evaluation


__all__ = [
    "apply_info_seed",
    "apply_info_seed_to_generation",
    "apply_info_seed_to_scoring",
    "apply_info_seed_to_evaluation",
]
