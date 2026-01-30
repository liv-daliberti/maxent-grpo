"""CPU-only integration tests for the custom MaxEnt/InfoSeed pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.pipelines.training.infoseed import run_infoseed_training
from maxent_grpo.pipelines.training.maxent import run_maxent_training
from maxent_grpo.training.weighting.loss import SequenceScores
from tests.helpers.run_setup_stubs import install_training_stubs, FakeAccelerator


@pytest.fixture
def cpu_pipeline_env(monkeypatch):
    """Install lightweight stubs and patch heavy training dependencies."""

    install_training_stubs(monkeypatch)
    import torch  # pylint: disable=import-error

    from maxent_grpo.pipelines.training import loop_common
    from maxent_grpo.training import loop as loop_mod
    from maxent_grpo.training import pipeline as pipeline_mod
    from maxent_grpo.training.weighting import loss as loss_mod

    state = {"proc_index": 0, "num_procs": 1}

    class _TestAccelerator(FakeAccelerator):
        def __init__(self):
            super().__init__()
            self.process_index = state["proc_index"]
            self.num_processes = state["num_procs"]
            self.is_main_process = self.process_index == 0
            self.state = SimpleNamespace(
                deepspeed_plugin=SimpleNamespace(zero_stage=0),
                distributed_type=None,
            )

        def broadcast_object_list(self, payload, src=0):
            return payload

        def gather_object(self, obj):
            return [obj]

        def optimizer_step(self, optimizer):
            optimizer.step()

    class _Loader:
        def __init__(self, dataset, batch_size, shuffle=False, **_kwargs):  # noqa: ARG002
            self._dataset = list(dataset)

        def __iter__(self):
            for row in self._dataset:
                yield row

    monkeypatch.setattr(loop_common, "require_accelerator", lambda *_a, **_k: _TestAccelerator)
    monkeypatch.setattr(loop_common, "require_dataloader", lambda *_a, **_k: _Loader)
    monkeypatch.setattr(
        loop_common,
        "get_model",
        lambda *_a, **_k: SimpleNamespace(parameters=lambda: [], config=SimpleNamespace()),
    )
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, texts, **_kwargs):
            batch = len(texts)
            return {
                "input_ids": torch.zeros((batch, 1), dtype=torch.int64),
                "attention_mask": torch.ones((batch, 1), dtype=torch.int64),
            }

    monkeypatch.setattr(loop_common, "get_tokenizer", lambda *_a, **_k: _Tokenizer())
    monkeypatch.setattr(
        loop_common,
        "load_datasets",
        lambda *_a, **_k: ([{"prompt": ["p"], "answer": ["a"]}], []),
    )
    monkeypatch.setattr(loop_common, "load_reward_functions", lambda *_a, **_k: ([], []))
    monkeypatch.setattr(loop_common, "load_eval_reward_functions", lambda *_a, **_k: ([], []))

    def _fake_prepare_training_batch(ctx, generator, batch):  # noqa: ARG001
        reward_comp = SimpleNamespace(
            total_utils=[1.0],
            advantage_samples=[0.1],
            per_reward_values={"reward": [0.1]},
            q_grouped=[[1.0]],
            advantage=SimpleNamespace(grouped=[[0.0]]),
            pairs=SimpleNamespace(prompts=["prompt"], completions=["completion"]),
        )
        weight_stats = SimpleNamespace(
            weights_grouped=[[1.0]],
            flat_weights=[1.0],
            weight_entropy=0.0,
            weight_entropy_min=0.0,
            weight_entropy_max=0.0,
            advantage_entropy=[0.0],
        )
        ref_stats = SimpleNamespace(
            ref_logp_sum=[0.0],
            ref_logp_sum_raw=[0.0],
            ref_tok_counts=[1],
            ref_logp_mean=0.0,
            avg_completion_tokens=1.0,
        )
        length_stats = SimpleNamespace(
            min_length=1.0,
            mean_length=1.0,
            max_length=1.0,
            clipped_ratio=0.0,
            min_terminated=1.0,
            mean_terminated=1.0,
            max_terminated=1.0,
        )
        batch_stats = pipeline_mod._BatchStats(  # type: ignore[attr-defined]
            score_batch=SimpleNamespace(),
            ref_stats=ref_stats,
            weight_stats=weight_stats,
            length_stats=length_stats,
            num_completion_tokens=1.0,
            prompt_token_count=1.0,
        )
        scores = SequenceScores(
            cur_logp_sum=torch.tensor([0.0]),
            behavior_logp_sum=torch.tensor([0.0]),
            log_ratio_train=torch.tensor([0.0]),
            denom_tok_tensor=torch.tensor([1.0]),
        )
        return pipeline_mod.PreparedBatch(
            grouped_completions=[["completion"]],
            reward_comp=reward_comp,
            batch_stats=batch_stats,
            total_input_tokens=2.0,
            scores=scores,
            seed_metrics={"seed": 1.0},
        )

    monkeypatch.setattr(pipeline_mod, "prepare_training_batch", _fake_prepare_training_batch)

    def _fake_build_loss_inputs(*_args, **_kwargs):
        return (SimpleNamespace(), SimpleNamespace())

    loss_outputs = SimpleNamespace(
        total_loss_scalar=0.1,
        policy_loss_scalar=0.05,
        kl_loss_scalar=0.01,
        weighted_kl_loss_scalar=0.01,
        clip_loss_scalar=0.0,
        scalars=SimpleNamespace(kl_loss=0.01),
    )
    diagnostics = SimpleNamespace(
        clip_ratio=0.0,
        clip_ratio_low_mean=0.0,
        clip_ratio_low_min=0.0,
        clip_ratio_high_mean=0.0,
        clip_ratio_high_max=0.0,
        clip_ratio_region_mean=0.0,
        kl_value=0.01,
        kl_per_token_by_len_bucket={},
        kl_token_count_by_len_bucket={},
    )

    monkeypatch.setattr(loss_mod, "build_loss_inputs", _fake_build_loss_inputs)
    monkeypatch.setattr(
        loss_mod,
        "evaluate_losses",
        lambda *_a, **_k: (loss_outputs, diagnostics),
    )

    class _GeneratorStub:
        def __init__(self, ctx):
            self.ctx = ctx

        def generate(self, *_a, **_k):
            return ([["completion"]], None)

    monkeypatch.setattr(loop_mod, "CompletionGenerator", lambda ctx: _GeneratorStub(ctx))
    monkeypatch.setattr(loop_mod, "_maybe_patch_zero_no_sync", lambda *_a, **_k: False)

    return SimpleNamespace(state=state)


def _write_controller_state(path: Path, tau: float = 0.4, beta: float = 0.3) -> None:
    path.write_text(
        json.dumps({"tau": tau, "beta": beta, "tau_log": 0.0}),
        encoding="utf-8",
    )


def _training_args(tmp_path: Path, **overrides) -> GRPOConfig:  # noqa: ARG001
    defaults = dict(
        num_train_epochs=1,
        num_generations=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_ratio=0.0,
        learning_rate=1e-4,
        max_prompt_length=16,
        max_completion_length=32,
        do_eval=False,
        save_strategy="steps",
        save_steps=1,
        max_steps=1,
        init_kl_coeff=0.1,
        maxent_tau=0.5,
        maxent_q_temperature=1.0,
        maxent_q_epsilon=1e-6,
    )
    cfg = GRPOConfig(**defaults)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_run_maxent_training_cpu_writes_controller(tmp_path, cpu_pipeline_env):
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    _write_controller_state(resume_dir / "controller_state.json")
    output_dir = tmp_path / "maxent"
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = _training_args(
        tmp_path,
        train_grpo_objective=False,
        controller_resume_from=str(resume_dir),
        output_dir=str(output_dir),
    )
    model_args = SimpleNamespace()
    run_maxent_training(script_args, training_args, model_args)
    state_path = output_dir / "controller_state.json"
    assert state_path.is_file()
    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    assert float(loaded["tau"]) >= 0.0
    assert float(loaded["beta"]) >= 0.0
    assert "meta" not in loaded or isinstance(loaded["meta"], dict)


def test_run_maxent_training_cpu_meta_controller(tmp_path, cpu_pipeline_env):
    resume_dir = tmp_path / "resume_meta"
    resume_dir.mkdir()
    initial_tau = 0.4
    initial_beta = 0.3
    _write_controller_state(resume_dir / "controller_state.json", tau=initial_tau, beta=initial_beta)
    output_dir = tmp_path / "maxent_meta"
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = _training_args(
        tmp_path,
        train_grpo_objective=False,
        controller_resume_from=str(resume_dir),
        output_dir=str(output_dir),
    )
    training_args.controller_meta_enabled = True
    training_args.controller_meta_method = "analytic"
    training_args.controller_meta_lr = 0.05
    training_args.controller_meta_update_interval = 1
    training_args.maxent_target_weight_entropy = 0.8
    training_args.kl_target = 0.05
    model_args = SimpleNamespace()
    run_maxent_training(script_args, training_args, model_args)
    state_path = output_dir / "controller_state.json"
    assert state_path.is_file()
    loaded = json.loads(state_path.read_text(encoding="utf-8"))
    meta_payload = loaded.get("meta", {}).get("controller", {})
    assert meta_payload.get("enabled") is True
    assert meta_payload.get("learning_rate") == pytest.approx(0.05)
    assert float(loaded["tau"]) != pytest.approx(initial_tau)
    assert float(loaded["beta"]) != pytest.approx(initial_beta)


def test_run_maxent_training_preflights_hub(tmp_path, cpu_pipeline_env, monkeypatch):
    output_dir = tmp_path / "maxent_hub"
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = _training_args(
        tmp_path,
        train_grpo_objective=False,
        output_dir=str(output_dir),
    )
    training_args.push_to_hub = True
    training_args.hub_model_id = "tester/maxent-smoke"
    training_args.hub_model_revision = "dryrun"
    calls: list[str] = []

    def _fake_create_repo(repo_id, **_kwargs):
        calls.append(f"repo:{repo_id}")
        return f"https://hub/{repo_id}"

    def _fake_create_branch(repo_id, branch, **_kwargs):
        calls.append(f"branch:{repo_id}:{branch}")

    monkeypatch.setattr("maxent_grpo.core.hub.create_repo", _fake_create_repo)
    monkeypatch.setattr("maxent_grpo.core.hub.create_branch", _fake_create_branch)
    monkeypatch.setattr("maxent_grpo.core.hub.list_repo_commits", lambda *_a, **_k: [])
    run_maxent_training(script_args, training_args, SimpleNamespace())
    assert "repo:tester/maxent-smoke" in calls
    assert "branch:tester/maxent-smoke:dryrun" in calls


def test_run_infoseed_training_cpu_meta_controller_updates(tmp_path, cpu_pipeline_env):
    resume_dir = tmp_path / "resume_infoseed"
    resume_dir.mkdir()
    initial_tau = 0.25
    initial_beta = 0.15
    _write_controller_state(resume_dir / "controller_state.json", tau=initial_tau, beta=initial_beta)
    output_dir = tmp_path / "infoseed_meta"
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = _training_args(
        tmp_path,
        controller_resume_from=str(resume_dir),
        output_dir=str(output_dir),
        info_seed_enabled=True,
        info_seed_num_seeds=1,
        info_seed_lambda=0.1,
        info_seed_temperature=0.5,
    )
    training_args.controller_meta_enabled = True
    training_args.controller_meta_method = "analytic"
    training_args.controller_meta_lr = 0.05
    training_args.controller_meta_update_interval = 1
    training_args.maxent_target_weight_entropy = 0.7
    training_args.kl_target = 0.05
    model_args = SimpleNamespace()
    run_infoseed_training(script_args, training_args, model_args)
    state_path = output_dir / "controller_state.json"
    assert state_path.is_file()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert float(saved["tau"]) != pytest.approx(initial_tau)
    assert float(saved["beta"]) != pytest.approx(initial_beta)
    meta_payload = saved.get("meta", {}).get("controller", {})
    assert meta_payload.get("enabled") is True
    assert meta_payload.get("learning_rate") == pytest.approx(0.05)


def test_run_infoseed_training_cpu_handles_non_main_rank(tmp_path, cpu_pipeline_env):
    cpu_pipeline_env.state["proc_index"] = 1
    cpu_pipeline_env.state["num_procs"] = 2
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    _write_controller_state(resume_dir / "controller_state.json", tau=0.2, beta=0.1)
    output_dir = tmp_path / "infoseed"
    script_args = GRPOScriptArguments(dataset_name="dummy")
    training_args = _training_args(
        tmp_path,
        controller_resume_from=str(resume_dir),
        output_dir=str(output_dir),
        info_seed_enabled=True,
        info_seed_num_seeds=1,
        info_seed_lambda=0.1,
        info_seed_temperature=0.5,
    )
    training_args.controller_meta_enabled = True
    training_args.controller_meta_method = "analytic"
    training_args.controller_meta_lr = 0.02
    training_args.controller_meta_update_interval = 1
    model_args = SimpleNamespace()
    run_infoseed_training(script_args, training_args, model_args)
    state_path = output_dir / "controller_state.json"
    # Non-main ranks should not write controller snapshots but must complete.
    assert not state_path.exists()
