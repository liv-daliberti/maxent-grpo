from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from exp_scaling import check_e14_preflight as gate


STAMP = "e14_preflight_test"


def _metrics_rows() -> list[dict[str, float]]:
    initial = {
        "eval/multi_answer/eval_count": 96.0,
        "eval/multi_answer/response_tok_len": 3.0,
        "misc/lr": 0.0,
        "misc/prompt_consumed": 0.0,
        "misc/prompt_dataset_len": 192.0,
        "misc/query_step": 0.0,
        "trainer/global_step": 0.0,
        "trainer/policy_sgd_step": 0.0,
        "trainer/step": 0.0,
    }
    update = {
        "actor/canonical_action_count": 3.0,
        "actor/canonical_action_support_size": 3.0,
        "actor/canonical_finish_length_count": 16.0,
        "actor/canonical_finish_unexpected_count": 0.0,
        "actor/canonical_graph_actions": 1.0,
        "actor/canonical_invalid_count": 0.0,
        "actor/canonical_sampler_learner": 1.0,
        "actor/canonical_sampler_fixed_shape": 1.0,
        "actor/canonical_behavior_q_norm_error_max": 2e-8,
        "actor/canonical_behavior_q_row_count": 48.0,
        "actor/canonical_behavior_q_support_max": 3.0,
        "actor/canonical_behavior_q_support_min": 3.0,
        "actor/no_eos_count": 0.0,
        "actor/num_data": 16.0,
        "actor/response_tok_len": 3.0,
        "actor/sampling_max_tokens": 3.0,
        "actor/sampling_temperature": 1.0,
        "misc/lr": 0.0,
        "misc/prompt_consumed": 16.0,
        "misc/query_step": 16.0,
        "train/canonical_action_count": 3.0,
        "train/canonical_action_vocab_size": 3.0,
        "train/canonical_behavior_denominator_actor": 1.0,
        "train/canonical_behavior_kl_actor_learner_max": 0.003,
        "train/canonical_behavior_kl_actor_learner_mean": 0.001,
        "train/canonical_behavior_kl_learner_actor_max": 0.004,
        "train/canonical_behavior_kl_learner_actor_mean": 0.002,
        "train/canonical_behavior_prefix_ess_fraction_min": 0.97,
        "train/canonical_behavior_q_norm_error_max": 2e-8,
        "train/canonical_behavior_q_row_count": 48.0,
        "train/canonical_behavior_q_support_max": 3.0,
        "train/canonical_behavior_q_support_min": 3.0,
        "train/canonical_behavior_ratio_max": 1.10,
        "train/canonical_behavior_ratio_min": 0.91,
        "train/canonical_behavior_sequence_ess_fraction": 0.96,
        "train/canonical_behavior_tv_max": 0.03,
        "train/canonical_behavior_tv_mean": 0.01,
        "train/canonical_sampled_prefix_entropy_ratio": 0.5,
        "train/canonical_sampled_prefix_entropy_sum": 0.5 * gate.math.log(27),
        "train/canonical_token_entropy_mean": 0.5,
        "train/entropy": 0.5,
        "train/learning_round": 1.0,
        "train/pg_loss": 0.1,
        "train/policy_grad_norm": 0.2,
        "train/pg_loss_nan": 0.0,
        "train/pg_loss_inf": 0.0,
        "trainer/global_step": 1.0,
        "trainer/policy_sgd_step": 1.0,
        "trainer/step": 1.0,
    }
    terminal = dict(update)
    terminal.update(
        {
            "eval/multi_answer/eval_count": 96.0,
            "eval/multi_answer/response_tok_len": 3.0,
            "trainer/step": 2.0,
        }
    )
    return [initial, update, terminal]


def _config_log() -> str:
    return "\n".join(
        f"│   {key!r}: {value!r},"
        for key, value in gate.EXPECTED_CONFIG.items()
    ).replace("'inf'", "inf")


def test_config_parser_ignores_nested_library_configuration_fields():
    log = _config_log() + "\nDeepSpeed config {'seed': 1234, 'zero': 2}\n"

    parsed = gate._parse_config(log)

    assert parsed["seed"] == "9005"


def _fixture(tmp_path: Path, *, logical_repo_root: Path | None = None) -> dict:
    logical_repo_root = logical_repo_root or (tmp_path / "logical_repo")
    source_root = tmp_path / "snapshot" / "src"
    package = source_root / "oat_drgrpo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "objective.py").write_text("OBJECTIVE = 'canonical'\n", encoding="utf-8")

    run_dir = tmp_path / "var" / "data" / f"run_{STAMP}_grpo_s9005"
    debug = run_dir / "debug_1"
    debug.mkdir(parents=True)
    metrics_path = debug / "train_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row) + "\n" for row in _metrics_rows()),
        encoding="utf-8",
    )

    artifacts = tmp_path / "var" / "artifacts"
    artifacts.mkdir(parents=True)
    identity_path = artifacts / f"{STAMP}_e14_identity.tsv"
    source_hash = gate.source_tree_hash(
        source_root, logical_repo_root=logical_repo_root
    )
    identity_rows = {
        "phase": "preflight",
        "stamp": STAMP,
        "source_hash": source_hash,
        "dataset_identity": json.dumps(
            gate.EXPECTED_DATASET_IDENTITY, sort_keys=True
        ),
        "runtime_identity": json.dumps(
            gate.EXPECTED_RUNTIME_IDENTITY, sort_keys=True
        ),
        "target_optimizer_updates": "1",
        "trajectory_query_budget": "1",
        "group_size": "16",
    }
    identity_path.write_text(
        "key\tvalue\n"
        + "".join(f"{key}\t{value}\n" for key, value in identity_rows.items()),
        encoding="utf-8",
    )

    stdout_path = artifacts / "job.out"
    stdout_path.write_text(
        "\n".join(
            (
                f"[slurm] oat_zero_source_root={source_root.resolve()}",
                "[slurm] job_id=12345",
                "[slurm] vllm_use_v1=0",
                "[train] canonical_graph_actions=1 action_count=3 "
                "learner_sampling=1 fixed_shape_sampling=1 "
                "max_entropy=3.295836866004329 "
                "vllm_use_v1=0",
                f"[experiment] save_path={run_dir.resolve()}",
                gate.EXPECTED_RUNTIME_IDENTITY["tokenizer_revision"],
                _config_log(),
                "canonical prompt materialization verified: rows=192 "
                "template=qwen_graph_digits dataset_map_cache=disabled",
                "canonical graph actor configured: token_ids=(16, 17, 18) "
                "action_count=3 tokenizer=x vllm_engine=v0",
                "canonical graph policy: action_token_ids=(16, 17, 18) "
                "horizon=3 tokenizer=x",
                "canonical learner sampler finished data_len=16 seed=9006 "
                "normalization_error_max=2e-08 fixed_shape=1",
                "post-learning done step=1",
                "eval/log done step=1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    stderr_path = artifacts / "job.err"
    stderr_path.write_text("", encoding="utf-8")
    return {
        "run_dir": run_dir,
        "identity_path": identity_path,
        "source_root": source_root,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "metrics_path": metrics_path,
        "logical_repo_root": logical_repo_root,
    }


def _check(paths: dict, **overrides):
    kwargs = {
        "run_dir": paths["run_dir"],
        "identity_path": paths["identity_path"],
        "source_root": paths["source_root"],
        "stdout_path": paths["stdout_path"],
        "stderr_path": paths["stderr_path"],
        "slurm_state": "COMPLETED",
        "slurm_exit_code": "0:0",
        "job_id": "12345",
        "logical_repo_root": paths["logical_repo_root"],
    }
    kwargs.update(overrides)
    return gate.check_preflight(**kwargs)


def _rewrite_metrics(paths: dict, rows: list[dict]) -> None:
    paths["metrics_path"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_passing_probe_emits_a_bound_machine_readable_approval(tmp_path):
    paths = _fixture(tmp_path)

    payload = _check(paths)
    approval = tmp_path / "approval.json"
    gate.write_approval(approval, payload)

    persisted = json.loads(approval.read_text(encoding="utf-8"))
    assert persisted["approved"] is True
    assert persisted["checks"]["exactly_one_optimizer_update"] is True
    assert persisted["metrics_summary"]["oat_steps"] == [0, 1, 2]
    assert persisted["checks"]["behavior_policy_overlap"] is True
    assert persisted["checks"]["learner_side_behavior_sampling"] is True
    assert persisted["checks"]["fixed_shape_behavior_sampling"] is True
    assert (
        persisted["log_summary"]["canonical_training_sampler"]
        == "learner_hf_fixed_shape"
    )
    assert persisted["metrics_summary"]["behavior_ratio_min"] == pytest.approx(
        0.91
    )
    assert persisted["metrics_summary"]["behavior_ratio_max"] == pytest.approx(
        1.10
    )
    assert persisted["evidence"]["metrics"]["sha256"] == gate._sha256_file(
        paths["metrics_path"]
    )


def _approved_fixture(tmp_path: Path) -> tuple[dict, Path]:
    paths = _fixture(tmp_path)
    approval = tmp_path / "approval.json"
    gate.write_approval(approval, _check(paths))
    return paths, approval


def test_c0_verifier_replays_evidence_and_binds_source_hash(tmp_path):
    paths, approval = _approved_fixture(tmp_path)
    identity, _, _ = gate.read_identity(paths["identity_path"])

    summary = gate.verify_approval_for_source(
        approval,
        expected_source_hash=identity["source_hash"],
        logical_repo_root=paths["logical_repo_root"],
    )

    assert summary["job_id"] == "12345"
    assert summary["preflight_stamp"] == STAMP
    assert summary["source_hash"] == identity["source_hash"]
    assert summary["evidence_files_verified"] == 5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("approved", False, "approved=true"),
        ("gate", "manual_override", "gate identifier"),
        ("protocol", "E13", "wrong protocol"),
        ("slurm", {"state": "FAILED", "exit_code": "1:0"}, "Slurm"),
    ),
)
def test_c0_verifier_rejects_forged_gate_metadata(tmp_path, field, value, message):
    paths, approval = _approved_fixture(tmp_path)
    identity, _, _ = gate.read_identity(paths["identity_path"])
    payload = json.loads(approval.read_text(encoding="utf-8"))
    payload[field] = value
    approval.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(gate.GateError, match=message):
        gate.verify_approval_for_source(
            approval,
            expected_source_hash=identity["source_hash"],
            logical_repo_root=paths["logical_repo_root"],
        )


def test_c0_verifier_rejects_current_python_source_drift(tmp_path):
    paths, approval = _approved_fixture(tmp_path)

    with pytest.raises(gate.GateError, match="C0 Python source differs"):
        gate.verify_approval_for_source(
            approval,
            expected_source_hash="0" * 64,
            logical_repo_root=paths["logical_repo_root"],
        )


def test_c0_verifier_rejects_mutated_metrics_evidence(tmp_path):
    paths, approval = _approved_fixture(tmp_path)
    identity, _, _ = gate.read_identity(paths["identity_path"])
    with paths["metrics_path"].open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(gate.GateError, match="evidence 'metrics' changed"):
        gate.verify_approval_for_source(
            approval,
            expected_source_hash=identity["source_hash"],
            logical_repo_root=paths["logical_repo_root"],
        )


def test_c0_verifier_rehashes_entire_source_snapshot(tmp_path):
    paths, approval = _approved_fixture(tmp_path)
    identity, _, _ = gate.read_identity(paths["identity_path"])
    # This file is not the separately hashed source_snapshot_marker evidence.
    # Rehashing the complete tree must nevertheless catch its mutation.
    (paths["source_root"] / "oat_drgrpo" / "objective.py").write_text(
        "OBJECTIVE = 'changed'\n", encoding="utf-8"
    )

    with pytest.raises(gate.GateError, match="source snapshot changed"):
        gate.verify_approval_for_source(
            approval,
            expected_source_hash=identity["source_hash"],
            logical_repo_root=paths["logical_repo_root"],
        )


def test_c0_launcher_has_no_manual_boolean_bypass():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "exp_scaling"
        / "launch_e14_canonical_smoke.sh"
    ).read_text(encoding="utf-8")

    assert "${OAT_ZERO_E14_PREFLIGHT_APPROVED:-0}" not in launcher
    assert "OAT_ZERO_E14_PREFLIGHT_APPROVAL" in launcher
    assert "verify_e14_preflight_approval.py" in launcher
    assert "--expected-source-hash \"$source_hash\"" in launcher
    assert "export OAT_ZERO_AUTO_RESUME=0" in launcher
    assert "export OAT_ZERO_WATCHDOG_REQUEUE=0" in launcher


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda rows: rows[:2], "initial, update, and terminal"),
        (
            lambda rows: [
                rows[0],
                rows[1],
                {**rows[2], "trainer/global_step": 2.0},
            ],
            "exactly one learning update",
        ),
        (
            lambda rows: [
                rows[0],
                {**rows[1], "actor/response_tok_len": 2.0},
                rows[2],
            ],
            "actor/response_tok_len",
        ),
        (
            lambda rows: [
                rows[0],
                {**rows[1], "train/canonical_behavior_ratio_max": 1.20001},
                rows[2],
            ],
            "full behavior-policy ratio range",
        ),
        (
            lambda rows: [
                rows[0],
                {**rows[1], "train/maxent_alpha_used": 0.0},
                rows[2],
            ],
            "forbidden treatment telemetry",
        ),
    ),
)
def test_metrics_fail_closed_on_incomplete_or_contaminated_probe(
    tmp_path, mutation, message
):
    paths = _fixture(tmp_path)
    _rewrite_metrics(paths, mutation(_metrics_rows()))

    with pytest.raises(gate.GateError, match=message):
        _check(paths)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "actor/canonical_behavior_q_row_count",
            47.0,
            "canonical_behavior_q_row_count",
        ),
        (
            "actor/canonical_behavior_q_support_min",
            2.0,
            "canonical_behavior_q_support_min",
        ),
        (
            "actor/canonical_behavior_q_norm_error_max",
            1.0001e-6,
            "actor behavior-q normalization error",
        ),
        (
            "actor/canonical_sampler_learner",
            0.0,
            "canonical_sampler_learner",
        ),
        (
            "actor/canonical_sampler_fixed_shape",
            0.0,
            "canonical_sampler_fixed_shape",
        ),
        (
            "train/canonical_behavior_q_norm_error_max",
            1.0001e-6,
            "learner-received behavior-q normalization error",
        ),
        (
            "train/canonical_behavior_denominator_actor",
            0.0,
            "canonical_behavior_denominator_actor",
        ),
        (
            "train/canonical_behavior_ratio_min",
            0.7999,
            "full behavior-policy ratio range",
        ),
        (
            "train/canonical_behavior_tv_max",
            0.05001,
            "behavior-policy TV",
        ),
        (
            "train/canonical_behavior_kl_actor_learner_max",
            0.01001,
            r"KL\(actor\|\|learner\)",
        ),
        (
            "train/canonical_behavior_kl_learner_actor_max",
            0.01001,
            r"KL\(learner\|\|actor\)",
        ),
        (
            "train/canonical_behavior_sequence_ess_fraction",
            0.8999,
            "sequence ESS fraction",
        ),
        (
            "train/canonical_behavior_prefix_ess_fraction_min",
            0.8999,
            "prefix ESS fraction",
        ),
    ),
)
def test_behavior_policy_overlap_gate_enforces_full_distribution_contract(
    tmp_path, field, value, message
):
    paths = _fixture(tmp_path)
    rows = _metrics_rows()
    rows[1][field] = value
    _rewrite_metrics(paths, rows)

    with pytest.raises(gate.GateError, match=message):
        _check(paths)


def test_learner_sampler_metric_fails_closed_when_missing(tmp_path):
    paths = _fixture(tmp_path)
    rows = _metrics_rows()
    rows[1].pop("actor/canonical_sampler_learner")
    _rewrite_metrics(paths, rows)

    with pytest.raises(gate.GateError, match="canonical_sampler_learner"):
        _check(paths)


def test_source_snapshot_is_rehashed_instead_of_trusting_tsv(tmp_path):
    paths = _fixture(tmp_path)
    (paths["source_root"] / "oat_drgrpo" / "__init__.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    with pytest.raises(gate.GateError, match="source snapshot"):
        _check(paths)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            "rows=192 template=qwen_graph_digits dataset_map_cache=disabled",
            "rows=128 template=qwen_graph_digits dataset_map_cache=enabled",
            "canonical_prompt_materialization",
        ),
        ("'maxent_alpha': 0.0", "'maxent_alpha': 0.01", "maxent_alpha"),
        (
            "'canonical_graph_learner_sampling': True",
            "'canonical_graph_learner_sampling': False",
            "canonical_graph_learner_sampling",
        ),
        (
            "'canonical_graph_fixed_shape_sampling': True",
            "'canonical_graph_fixed_shape_sampling': False",
            "canonical_graph_fixed_shape_sampling",
        ),
        (
            "learner_sampling=1 fixed_shape_sampling=1",
            "learner_sampling=1 fixed_shape_sampling=0",
            "learner_side_training_sampler",
        ),
        (
            "normalization_error_max=2e-08 fixed_shape=1",
            "normalization_error_max=2e-08 fixed_shape=0",
            "learner_side_sampling_completed",
        ),
        ("post-learning done step=1", "post-learning start step=1", "post_learning"),
    ),
)
def test_runtime_log_must_prove_prompt_and_zero_treatment_contract(
    tmp_path, old, new, message
):
    paths = _fixture(tmp_path)
    text = paths["stdout_path"].read_text(encoding="utf-8")
    paths["stdout_path"].write_text(text.replace(old, new), encoding="utf-8")

    with pytest.raises(gate.GateError, match=message):
        _check(paths)


def test_terminal_slurm_success_is_mandatory(tmp_path):
    paths = _fixture(tmp_path)

    with pytest.raises(gate.GateError, match="not a clean terminal success"):
        _check(paths, slurm_state="FAILED", slurm_exit_code="1:0")


def test_identity_requires_exact_frozen_runtime(tmp_path):
    paths = _fixture(tmp_path)
    text = paths["identity_path"].read_text(encoding="utf-8")
    paths["identity_path"].write_text(
        text.replace('"vllm_version": "0.8.4"', '"vllm_version": "0.8.5"'),
        encoding="utf-8",
    )

    with pytest.raises(gate.GateError, match="runtime identity"):
        _check(paths)


def test_cli_removes_stale_approval_before_a_failed_recheck(tmp_path, monkeypatch):
    repo_root = Path(gate.__file__).resolve().parents[2]
    paths = _fixture(tmp_path, logical_repo_root=repo_root)
    approval = tmp_path / "approval.json"
    approval.write_text('{"approved": true}\n', encoding="utf-8")
    rows = _metrics_rows()
    rows[1]["actor/canonical_invalid_count"] = 1.0
    _rewrite_metrics(paths, rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_e14_preflight.py",
            "--run-dir",
            str(paths["run_dir"]),
            "--identity",
            str(paths["identity_path"]),
            "--source-root",
            str(paths["source_root"]),
            "--stdout-log",
            str(paths["stdout_path"]),
            "--stderr-log",
            str(paths["stderr_path"]),
            "--slurm-state",
            "COMPLETED",
            "--slurm-exit-code",
            "0:0",
            "--job-id",
            "12345",
            "--approval-out",
            str(approval),
        ],
    )

    with pytest.raises(gate.GateError, match="canonical_invalid_count"):
        gate.main()
    assert not approval.exists()


def test_slurm_status_parser_binds_json_to_job_id(tmp_path):
    status = tmp_path / "status.json"
    status.write_text(
        json.dumps(
            {"job_id": "12345", "state": "COMPLETED", "exit_code": "0:0"}
        ),
        encoding="utf-8",
    )

    assert gate.parse_slurm_status(status, expected_job_id="12345") == (
        "COMPLETED",
        "0:0",
    )
    with pytest.raises(gate.GateError, match="wrong job ID"):
        gate.parse_slurm_status(status, expected_job_id="99999")
