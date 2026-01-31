"""Cross-mode controller resume tests for MaxEnt/InfoSeed pipelines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maxent_grpo import grpo as grpo_cli
from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.pipelines.training import infoseed as infoseed_pipeline
from maxent_grpo.pipelines.training import maxent as maxent_pipeline
from maxent_grpo.training.state import load_controller_state_chain
from maxent_grpo.training.weighting.logic import (
    CONTROLLER_STATE_FILENAME,
    build_weighting_settings,
    save_controller_state,
)
from maxent_grpo.training.weighting.types import (
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightNormalizationSettings,
    WeightingSettings,
)


@pytest.fixture
def controller_state(tmp_path):
    """Persist a controller snapshot with non-default tau/beta for resume tests."""

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


def _builder_stub(training_args: GRPOConfig) -> SimpleNamespace:
    """Construct a minimal TrainingLoopContext-like stub."""

    weighting = build_weighting_settings(training_args)
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
    logging_handles = SimpleNamespace(wandb_run=None)
    optimization = SimpleNamespace(
        handles=SimpleNamespace(
            optimizer=SimpleNamespace(
                zero_grad=lambda *_, **__: None  # pragma: no cover - stubbed path
            )
        ),
        schedule=SimpleNamespace(num_epochs=0, grad_accum_steps=1, total_training_steps=0),
    )
    return SimpleNamespace(
        controller=controller,
        runtime=runtime,
        scoring=SimpleNamespace(weighting=weighting),
        logging=logging_handles,
        optimization=optimization,
        generation=None,
        evaluation=None,
        reward=None,
        eval_reward=None,
        settings=None,
    )


@pytest.mark.parametrize("pipeline_name", ["maxent", "infoseed"])
def test_custom_loop_pipelines_resume_controller_state(
    monkeypatch, tmp_path, controller_state, pipeline_name
):
    """Ensure both MaxEnt and InfoSeed pipelines reload controller snapshots."""

    resume_dir, expected_tau, expected_beta = controller_state
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = GRPOConfig()
    training_args.output_dir = str(tmp_path / "output")
    training_args.controller_resume_from = str(resume_dir)
    training_args.train_grpo_objective = False
    if pipeline_name == "infoseed":
        training_args.info_seed_enabled = True
    model_args = SimpleNamespace()
    loaded = {}

    def _fake_builder(script, cfg, model, **kwargs):
        if kwargs.get("force_grpo_objective") is not None:
            cfg.train_grpo_objective = kwargs["force_grpo_objective"]
        return _builder_stub(cfg)

    def _fake_run_loop(ctx):
        load_controller_state_chain(ctx.controller, ctx.runtime.accelerator, ctx.scoring.weighting)
        loaded["tau"] = ctx.scoring.weighting.tau
        loaded["beta"] = ctx.scoring.weighting.beta
        loaded["meta_enabled"] = ctx.scoring.weighting.controller_meta.enabled
        loaded["meta_lr"] = ctx.scoring.weighting.controller_meta.learning_rate
        loaded["meta_tau_grad"] = getattr(ctx.scoring.weighting, "_meta_last_tau_grad", None)

    if pipeline_name == "maxent":
        monkeypatch.setattr(maxent_pipeline, "build_training_loop_context", _fake_builder)
        monkeypatch.setattr(maxent_pipeline, "run_training_loop", _fake_run_loop)
        maxent_pipeline.run_maxent_training(script_args, training_args, model_args)
    else:
        monkeypatch.setattr(infoseed_pipeline, "build_training_loop_context", _fake_builder)
        monkeypatch.setattr(infoseed_pipeline, "run_training_loop", _fake_run_loop)
        infoseed_pipeline.run_infoseed_training(script_args, training_args, model_args)

    assert loaded["tau"] == pytest.approx(expected_tau)
    assert loaded["beta"] == pytest.approx(expected_beta)
    assert loaded["meta_enabled"] is True
    assert loaded["meta_lr"] == pytest.approx(0.1)
    assert loaded["meta_tau_grad"] == pytest.approx(0.02)


def test_controller_resume_handles_missing_fields(tmp_path):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    state_path = resume_dir / CONTROLLER_STATE_FILENAME
    state_path.write_text(json.dumps({"tau": 0.25, "beta": 0.15}), encoding="utf-8")
    training_args = GRPOConfig()
    training_args.output_dir = str(tmp_path / "out")
    training_args.controller_resume_from = str(resume_dir)
    training_args.train_grpo_objective = False
    ctx = _builder_stub(training_args)
    load_controller_state_chain(ctx.controller, ctx.runtime.accelerator, ctx.scoring.weighting)
    weighting = ctx.scoring.weighting
    assert weighting.tau == pytest.approx(0.25)
    assert weighting.beta == pytest.approx(0.15)
    assert getattr(weighting, "_tau_log") == pytest.approx(math.log(max(weighting.tau, 1e-8)))
    assert math.isfinite(getattr(weighting, "_tau_entropy_ema"))


def test_baseline_grpo_resume_switches_to_meta(monkeypatch, tmp_path):
    script_args = GRPOScriptArguments(dataset_name="dummy")
    base_training = GRPOConfig()
    base_training.output_dir = str(tmp_path / "baseline_out")
    base_training.train_grpo_objective = True
    base_ctx = _builder_stub(base_training)
    save_controller_state(base_ctx.controller.state_path, base_ctx.scoring.weighting)

    meta_training = GRPOConfig()
    meta_training.output_dir = str(tmp_path / "meta_out")
    meta_training.controller_resume_from = base_training.output_dir
    meta_training.train_grpo_objective = True
    meta_training.controller_meta_enabled = True
    model_args = SimpleNamespace()

    loaded: dict[str, Any] = {}

    def _fake_builder(script, cfg, model, **kwargs):
        loaded["force_grpo"] = kwargs.get("force_grpo_objective")
        return _builder_stub(cfg)

    def _fake_run_loop(ctx):
        load_controller_state_chain(ctx.controller, ctx.runtime.accelerator, ctx.scoring.weighting)
        loaded["tau"] = ctx.scoring.weighting.tau
        loaded["beta"] = ctx.scoring.weighting.beta
        save_controller_state(ctx.controller.state_path, ctx.scoring.weighting)

    monkeypatch.setattr(
        maxent_pipeline,
        "build_training_loop_context",
        _fake_builder,
    )
    monkeypatch.setattr(maxent_pipeline, "run_training_loop", _fake_run_loop)

    def _fail_baseline(*_args, **_kwargs):  # pragma: no cover - sanity guard
        raise AssertionError("baseline pipeline should be bypassed when meta is enabled")

    monkeypatch.setattr(
        "maxent_grpo.pipelines.training.baseline.run_baseline_training",
        _fail_baseline,
    )

    grpo_cli.main(script_args, meta_training, model_args)
    assert loaded["force_grpo"] is True
    assert loaded["tau"] == pytest.approx(base_ctx.scoring.weighting.tau)
    assert loaded["beta"] == pytest.approx(base_ctx.scoring.weighting.beta)
    saved_state = Path(meta_training.output_dir) / CONTROLLER_STATE_FILENAME
    assert saved_state.is_file()
