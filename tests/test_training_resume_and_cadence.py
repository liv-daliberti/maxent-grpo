"""Unit tests for custom-loop resume, cadence, warmup, and checkpoint helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.pipelines.training import loop_common
from maxent_grpo.training.metrics import _should_log
from maxent_grpo.training.state import (
    _write_trainer_state_json,
    build_checkpoint_saver,
    load_trainer_state_metadata,
    resolve_resume_checkpoint,
)


@pytest.fixture(autouse=True)
def _patch_loop_common(monkeypatch):
    """Lightweight stubs to avoid heavy imports during loop construction."""

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
    monkeypatch.setattr(loop_common, "load_datasets", lambda *_a, **_k: (["row"], []))
    monkeypatch.setattr(
        loop_common, "load_reward_functions", lambda *_a, **_k: (["reward"], [1.0])
    )
    monkeypatch.setattr(
        loop_common, "load_eval_reward_functions", lambda *_a, **_k: ([], [])
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


def test_resolve_resume_checkpoint_prefers_init(tmp_path):
    ckpt_init = tmp_path / "checkpoint-10"
    ckpt_init.mkdir()
    training_args = SimpleNamespace(
        init_from_checkpoint=str(ckpt_init),
        resume_from_checkpoint="missing",
        output_dir=str(tmp_path),
    )
    resolved, requested = resolve_resume_checkpoint(training_args)
    assert resolved == str(ckpt_init)
    assert requested is True


def test_resolve_resume_checkpoint_falls_back_to_latest(tmp_path):
    ckpt_old = tmp_path / "checkpoint-5"
    ckpt_old.mkdir()
    ckpt_new = tmp_path / "checkpoint-20"
    ckpt_new.mkdir()
    training_args = SimpleNamespace(
        init_from_checkpoint=None,
        resume_from_checkpoint=True,
        output_dir=str(tmp_path),
    )
    resolved, requested = resolve_resume_checkpoint(training_args)
    assert resolved == str(ckpt_new)
    assert requested is True


def test_load_trainer_state_metadata_reads_trainer_state(tmp_path):
    ckpt = tmp_path / "checkpoint-7"
    ckpt.mkdir()
    state_file = ckpt / "trainer_state.json"
    state_file.write_text(
        json.dumps(
            {
                "global_step": 12,
                "num_input_tokens_seen": 33.0,
                "best_global_step": 11,
            }
        ),
        encoding="utf-8",
    )
    meta = load_trainer_state_metadata(str(ckpt))
    assert meta["global_step"] == 12
    assert meta["num_input_tokens_seen"] == 33.0
    assert meta["best_global_step"] == 11


def test_write_trainer_state_json_merges_metadata(tmp_path):
    ckpt_dir = tmp_path / "checkpoint-30"
    ckpt_dir.mkdir()
    base_state = {
        "best_model_checkpoint": "checkpoint-10",
        "best_metric": 0.1,
        "best_global_step": 10,
        "log_history": [{"step": 1}],
    }
    accel = SimpleNamespace(
        is_local_process_zero=False, is_world_process_zero=True
    )
    training_args = SimpleNamespace(
        max_steps=200,
        num_train_epochs=3,
        save_steps=5,
        logging_steps=2,
    )
    _write_trainer_state_json(
        str(ckpt_dir),
        training_args,
        global_step=30,
        num_input_tokens_seen=99.5,
        base_state=base_state,
        accelerator=accel,
    )
    payload = json.loads((ckpt_dir / "trainer_state.json").read_text())
    assert payload["global_step"] == 30
    assert payload["num_input_tokens_seen"] == 99.5
    assert payload["best_global_step"] == 10
    assert payload["best_model_checkpoint"] == "checkpoint-10"
    assert payload["is_local_process_zero"] is False
    assert payload["is_world_process_zero"] is True


def test_build_checkpoint_saver_persists_and_pushes(monkeypatch, tmp_path, training_stubs):
    saved = {"model": [], "tokenizer": [], "optimizer": [], "push": None}

    class _Model:
        def save_pretrained(self, path, **kwargs):
            saved["model"].append((path, kwargs))
            Path(path, "weights.bin").write_text("model", encoding="utf-8")

    class _Tokenizer:
        def save_pretrained(self, path):
            saved["tokenizer"].append(path)
            Path(path, "tokenizer").write_text("tok", encoding="utf-8")

    class _Optimizer:
        def state_dict(self):
            return {"state": 1}

    class _Accel:
        def __init__(self):
            self.waits = 0
            self.saved_state = None
            self.is_main_process = True

        def wait_for_everyone(self):
            self.waits += 1

        def save_state(self, path):
            self.saved_state = path

        def unwrap_model(self, model):
            return model

    def _fake_push(args, extra_ignore_patterns=None, include_checkpoints=None):
        saved["push"] = args.output_dir

    monkeypatch.setattr("maxent_grpo.core.hub.push_to_hub_revision", _fake_push)
    monkeypatch.setattr(
        "torch.save",
        lambda obj, path: Path(path).write_text("opt", encoding="utf-8"),
    )

    training_args = SimpleNamespace(
        output_dir=str(tmp_path),
        save_strategy="steps",
        save_steps=1,
        push_to_hub=True,
        hub_strategy="every_save",
        __dict__={},
    )
    accel = _Accel()
    runtime_handles = SimpleNamespace(accelerator=accel, model=_Model())
    optim_handles = SimpleNamespace(optimizer=_Optimizer())
    tokenizer = _Tokenizer()
    state_ref = {"state": SimpleNamespace(global_step=7, num_input_tokens_seen=12.5)}
    saver = build_checkpoint_saver(
        training_args,
        runtime_handles,
        optim_handles,
        tokenizer,
        state_ref=state_ref,
        base_trainer_state={"best_global_step": 5},
    )
    saver("checkpoint-7")
    ckpt_dir = Path(tmp_path) / "checkpoint-7"
    assert ckpt_dir.exists()
    assert saved["model"] and saved["tokenizer"]
    trainer_state = json.loads((ckpt_dir / "trainer_state.json").read_text())
    assert trainer_state["global_step"] == 7
    assert trainer_state["num_input_tokens_seen"] == 12.5
    assert trainer_state["best_global_step"] == 5
    assert saved["push"] == str(ckpt_dir)
    assert (ckpt_dir / "optimizer.pt").read_text() == "opt"


def test_should_log_respects_strategy_and_first_step(caplog):
    ctx = SimpleNamespace(
        training_args=SimpleNamespace(
            logging_strategy="steps", logging_steps=2, logging_first_step=False
        )
    )
    assert _should_log(ctx, 0) is False
    assert _should_log(ctx, 1) is False
    assert _should_log(ctx, 2) is True
    caplog.set_level("WARNING")
    ctx.training_args.logging_strategy = "epoch"
    assert _should_log(ctx, 10) is False
    assert "epoch is not supported" in caplog.text


def test_build_evaluation_settings_steps_and_disabled(caplog):
    cfg = GRPOConfig()
    cfg.do_eval = True
    cfg.eval_steps = 5
    cfg.evaluation_strategy = "steps"
    eval_settings = loop_common.build_evaluation_settings(cfg)
    assert eval_settings.enabled is True
    assert eval_settings.every_n_steps == 5
    cfg.eval_steps = None
    eval_settings = loop_common.build_evaluation_settings(cfg)
    assert eval_settings.enabled is False
    cfg.evaluation_strategy = "epoch"
    with caplog.at_level("WARNING"):
        eval_settings = loop_common.build_evaluation_settings(cfg)
    assert eval_settings.enabled is False
    assert "epoch is not supported" in caplog.text


def test_warmup_steps_prefer_ratio_when_zero(monkeypatch):
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = GRPOConfig()
    training_args.output_dir = "var/data/out"
    training_args.max_steps = 100
    training_args.warmup_steps = 0
    training_args.warmup_ratio = 0.1
    _model_args = SimpleNamespace(model_name_or_path="dummy")
    ctx = loop_common.build_training_loop_context(
        script_args,
        training_args,
        _model_args,
        deps_namespace="test",
        apply_info_seed_cfg=False,
        force_grpo_objective=None,
    )
    assert ctx.optimization.schedule.warmup_steps == 10
    training_args.warmup_steps = 5
    ctx = loop_common.build_training_loop_context(
        script_args,
        training_args,
        _model_args,
        deps_namespace="test",
        apply_info_seed_cfg=False,
        force_grpo_objective=None,
    )
    assert ctx.optimization.schedule.warmup_steps == 5


def test_build_scoring_settings_enables_clip_objective():
    cfg = GRPOConfig()
    cfg.clip_range = 0.2
    cfg.maxent_use_clip_objective = False
    cfg.maxent_clip_objective_coef = 1.0
    cfg.maxent_clip_adv_baseline = 0.1
    scoring = loop_common.build_scoring_settings(cfg)
    assert scoring.clipping.clip_range == pytest.approx(0.2)
    assert scoring.clipping.use_clip_objective is True


def test_build_scoring_settings_disables_tail_tokens_without_len_norm():
    cfg = GRPOConfig()
    cfg.maxent_length_normalize_ref = False
    cfg.maxent_score_tail_tokens = 256

    scoring = loop_common.build_scoring_settings(cfg)

    assert scoring.batching.score_tail_tokens is None
