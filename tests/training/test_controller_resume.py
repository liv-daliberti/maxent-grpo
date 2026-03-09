"""Controller-state resume and pipeline routing tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from maxent_grpo.config import GRPOConfig
from maxent_grpo.training.state import load_controller_state_chain
from maxent_grpo.training.weighting.logic import (
    CONTROLLER_STATE_FILENAME,
    save_controller_state,
)
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightingSettings,
)


def _context_stub(training_args: GRPOConfig) -> SimpleNamespace:
    """Construct a minimal context-like object for controller state loading."""

    weighting = WeightingSettings(
        tau=0.0,
        beta=0.0,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=True),
        q_distribution=QDistributionSettings(temperature=1.0, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=None,
            learning_rate=0.0,
            minimum_value=0.0,
            maximum_value=0.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.0, horizon=0, step_size=0.0),
        train_grpo_objective=bool(getattr(training_args, "train_grpo_objective", True)),
    )
    controller = SimpleNamespace(
        state_path=str(Path(training_args.output_dir) / CONTROLLER_STATE_FILENAME),
        resume_from=getattr(training_args, "controller_resume_from", None),
        overwrite_existing=bool(getattr(training_args, "overwrite_output_dir", False)),
    )
    accelerator = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        num_processes=1,
        wait_for_everyone=lambda: None,
        broadcast_object_list=lambda payload, src=0: payload,
    )
    runtime = SimpleNamespace(accelerator=accelerator, device="cpu")
    return SimpleNamespace(controller=controller, runtime=runtime, scoring=SimpleNamespace(weighting=weighting))


@pytest.fixture
def controller_state(tmp_path: Path) -> tuple[Path, float, float]:
    """Persist a controller snapshot with non-default tau/beta/meta fields."""

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    weighting = WeightingSettings(
        tau=0.33,
        beta=0.07,
        normalization=WeightNormalizationSettings(denom=1.0, len_norm_ref=True),
        q_distribution=QDistributionSettings(temperature=0.5, epsilon=1e-6),
        tau_schedule=TauSchedule(
            target_entropy=1.2,
            learning_rate=0.05,
            minimum_value=0.05,
            maximum_value=2.0,
            warmup_steps=0,
        ),
        kl_controller=KlControllerSettings(target=0.1, horizon=8, step_size=0.2),
        train_grpo_objective=False,
    )
    weighting.denom = weighting.tau + weighting.beta
    setattr(weighting, "_tau_log", math.log(weighting.tau))
    setattr(weighting, "_tau_entropy_ema", 0.95)
    weighting.controller_meta.enabled = True
    weighting.controller_meta.learning_rate = 0.1
    weighting.controller_meta.optimizer = "sgd"
    weighting.controller_meta.last_tau_grad = 0.02
    weighting.controller_meta.last_beta_grad = -0.03
    setattr(weighting, "_meta_last_tau_grad", 0.02)
    setattr(weighting, "_meta_last_beta_grad", -0.03)
    state_path = resume_dir / CONTROLLER_STATE_FILENAME
    save_controller_state(str(state_path), weighting)
    return resume_dir, weighting.tau, weighting.beta


def test_controller_resume_loads_saved_meta_fields(
    tmp_path: Path,
    controller_state: tuple[Path, float, float],
) -> None:
    """load_controller_state_chain should restore tau/beta/meta snapshots."""

    resume_dir, expected_tau, expected_beta = controller_state
    training_args = GRPOConfig()
    training_args.output_dir = str(tmp_path / "output")
    training_args.controller_resume_from = str(resume_dir)
    training_args.train_grpo_objective = False
    ctx = _context_stub(training_args)

    load_controller_state_chain(ctx.controller, ctx.runtime.accelerator, ctx.scoring.weighting)

    weighting = ctx.scoring.weighting
    assert weighting.tau == pytest.approx(expected_tau)
    assert weighting.beta == pytest.approx(expected_beta)
    assert weighting.controller_meta.enabled is True
    assert weighting.controller_meta.learning_rate == pytest.approx(0.1)
    assert getattr(weighting, "_meta_last_tau_grad", None) == pytest.approx(0.02)


def test_controller_resume_handles_missing_fields(tmp_path: Path) -> None:
    """Resume should backfill controller internals when state JSON is partial."""

    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    state_path = resume_dir / CONTROLLER_STATE_FILENAME
    state_path.write_text(json.dumps({"tau": 0.25, "beta": 0.15}), encoding="utf-8")

    training_args = GRPOConfig()
    training_args.output_dir = str(tmp_path / "out")
    training_args.controller_resume_from = str(resume_dir)
    training_args.train_grpo_objective = False
    ctx = _context_stub(training_args)

    load_controller_state_chain(ctx.controller, ctx.runtime.accelerator, ctx.scoring.weighting)

    weighting = ctx.scoring.weighting
    assert weighting.tau == pytest.approx(0.25)
    assert weighting.beta == pytest.approx(0.15)
    assert getattr(weighting, "_tau_log") == pytest.approx(math.log(max(weighting.tau, 1e-8)))
    assert math.isfinite(getattr(weighting, "_tau_entropy_ema"))
