"""Tests for the shared training-loop builder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import logging

import pytest

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.pipelines.training import loop_common


@pytest.fixture(autouse=True)
def _patch_loop_common(monkeypatch):
    monkeypatch.setattr(
        loop_common,
        "get_model",
        lambda *_a, **_k: SimpleNamespace(config=SimpleNamespace()),
    )
    monkeypatch.setattr(
        loop_common,
        "get_tokenizer",
        lambda *_a, **_k: SimpleNamespace(pad_token_id=0, eos_token_id=0),
    )
    monkeypatch.setattr(
        loop_common,
        "load_datasets",
        lambda *_a, **_k: (["row"], []),
    )
    monkeypatch.setattr(
        loop_common,
        "load_reward_functions",
        lambda *_a, **_k: (["reward"], [1.0]),
    )
    monkeypatch.setattr(
        loop_common,
        "load_eval_reward_functions",
        lambda *_a, **_k: ([], []),
    )
    monkeypatch.setattr(
        loop_common,
        "require_accelerator",
        lambda *_a, **_k: SimpleNamespace(
            device="cpu", is_main_process=True, num_processes=1, process_index=0
        ),
    )
    monkeypatch.setattr(
        loop_common,
        "require_dataloader",
        lambda *_a, **_k: lambda dataset, batch_size, shuffle=False, **_kwargs: dataset,
    )
    monkeypatch.setattr(
        loop_common,
        "build_training_state",
        lambda *_a, **_k: SimpleNamespace(step_logger=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(
        "maxent_grpo.training.optim.build_optimization_handles",
        lambda _model, _training_args: SimpleNamespace(
            optimizer=SimpleNamespace(step=lambda: None, zero_grad=lambda **_: None),
            lr_scheduler=None,
            base_optimizer=SimpleNamespace(),
            learning_rate=float(getattr(_training_args, "learning_rate", 0.0)),
        ),
    )


def _script_and_training_args() -> tuple[GRPOScriptArguments, GRPOConfig, SimpleNamespace]:
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = GRPOConfig()
    training_args.output_dir = "var/data/out"
    training_args.controller_resume_from = "resume_dir"
    training_args.learning_rate = 1e-4
    training_args.init_kl_coeff = 0.07
    training_args.kl_target = 0.1
    training_args.kl_horizon = 5
    training_args.kl_ctl_step_size = 0.5
    training_args.maxent_tau = 0.42
    training_args.maxent_tau_min = 0.1
    training_args.maxent_tau_max = 0.8
    training_args.maxent_tau_lr = 0.2
    training_args.maxent_tau_warmup_steps = 3
    model_args = SimpleNamespace(model_name_or_path="dummy")
    return script_args, training_args, model_args


def test_build_loop_context_copies_weighting_settings():
    script_args, training_args, model_args = _script_and_training_args()
    ctx = loop_common.build_training_loop_context(
        script_args,
        training_args,
        model_args,
        deps_namespace="test",
        apply_info_seed_cfg=False,
        force_grpo_objective=None,
    )
    weighting = ctx.scoring.weighting
    assert weighting.tau == pytest.approx(training_args.maxent_tau)
    assert weighting.tau_min == pytest.approx(training_args.maxent_tau_min)
    assert weighting.tau_max == pytest.approx(training_args.maxent_tau_max)
    assert weighting.tau_lr == pytest.approx(training_args.maxent_tau_lr)
    assert weighting.tau_warmup_steps == training_args.maxent_tau_warmup_steps
    assert weighting.beta == pytest.approx(training_args.init_kl_coeff)
    assert ctx.optimization.handles.learning_rate == pytest.approx(
        training_args.learning_rate
    )


@pytest.mark.parametrize("apply_cfg", [True, False])
@pytest.mark.parametrize("info_seed_enabled", [True, False])
def test_build_loop_context_infoseed_toggles(apply_cfg, info_seed_enabled):
    script_args, training_args, model_args = _script_and_training_args()
    training_args.info_seed_enabled = info_seed_enabled
    training_args.info_seed_num_seeds = 3
    training_args.info_seed_lambda = 0.25
    training_args.info_seed_temperature = 0.3
    training_args.info_seed_loss_type = "ce"
    training_args.info_seed_pooling = "sum"
    training_args.info_seed_alpha_entropy = 0.7
    ctx = loop_common.build_training_loop_context(
        script_args,
        training_args,
        model_args,
        deps_namespace="test",
        apply_info_seed_cfg=apply_cfg,
        force_grpo_objective=None,
    )
    if apply_cfg and info_seed_enabled:
        assert ctx.generation.seed_augmentation is not None
        assert (
            ctx.generation.seed_augmentation.num_seeds
            == training_args.info_seed_num_seeds
        )
        assert ctx.scoring.info_seed_lambda == pytest.approx(
            training_args.info_seed_lambda
        )
        assert ctx.scoring.info_seed_loss_type == training_args.info_seed_loss_type
        assert ctx.scoring.info_seed_pooling == training_args.info_seed_pooling
        assert (
            ctx.evaluation.seed_eval is not None
            and ctx.evaluation.seed_eval["enabled"] is True
        )
    else:
        assert ctx.generation.seed_augmentation is None
        assert ctx.scoring.info_seed_lambda == 0.0
        assert ctx.scoring.info_seed_loss_type == "infonce"
        assert ctx.evaluation.seed_eval is None


@pytest.mark.parametrize("force_grpo", [None, True])
def test_build_loop_context_sets_controller_and_optimizer(force_grpo):
    script_args, training_args, model_args = _script_and_training_args()
    training_args.output_dir = str(Path("var") / "data" / "run")
    training_args.train_grpo_objective = False
    ctx = loop_common.build_training_loop_context(
        script_args,
        training_args,
        model_args,
        deps_namespace="custom",
        apply_info_seed_cfg=False,
        force_grpo_objective=force_grpo,
    )
    expected_state_path = str(
        Path(training_args.output_dir) / "controller_state.json"
    )
    assert ctx.controller.state_path == expected_state_path
    assert ctx.controller.resume_from == training_args.controller_resume_from
    assert ctx.controller.overwrite_existing is False
    assert ctx.optimization.handles.learning_rate == pytest.approx(
        training_args.learning_rate
    )
    if force_grpo is None:
        assert training_args.train_grpo_objective is False
    else:
        assert training_args.train_grpo_objective is True


def test_build_generation_settings_seeds_metadata():
    _script_args, training_args, _ = _script_and_training_args()
    training_args.dataset_name = "hf/train"
    training_args.hub_model_id = "org/model"
    gen_settings = loop_common.build_generation_settings(training_args)
    stats = gen_settings.generation_stats
    assert stats["dataset_name"] == "hf/train"
    assert stats["model_id"] == "org/model"


def test_build_generation_settings_clamps_backoff(caplog):
    _script_args, training_args, _ = _script_and_training_args()
    training_args.vllm_retry_sleep = -5
    training_args.vllm_backoff = -2
    training_args.vllm_backoff_multiplier = 0.2
    training_args.vllm_max_retries = -1
    with caplog.at_level(logging.WARNING):
        gen_settings = loop_common.build_generation_settings(training_args)
    vllm_cfg = gen_settings.vllm
    assert vllm_cfg.retry_sleep == 0.0
    assert vllm_cfg.backoff == 0.0


def test_build_loop_context_applies_chunking_knobs():
    script_args, training_args, model_args = _script_and_training_args()
    training_args.maxent_logprob_chunk_size = 256
    training_args.max_prompt_length = 1024
    training_args.max_completion_length = 2048
    ctx = loop_common.build_training_loop_context(
        script_args,
        training_args,
        model_args,
        deps_namespace="test",
        apply_info_seed_cfg=False,
        force_grpo_objective=None,
    )
    batching = ctx.scoring.batching
    assert batching.logprob_chunk_size == training_args.maxent_logprob_chunk_size
    assert batching.score_slice == training_args.maxent_logprob_chunk_size
    gen = ctx.generation
    assert gen.max_prompt_len == training_args.max_prompt_length
    assert gen.max_completion_len == training_args.max_completion_length
