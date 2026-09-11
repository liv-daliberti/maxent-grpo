from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from exp_scaling import check_e15_treatments as gate
from exp_scaling.e14_archival import (
    E15_SCHEMA,
    EXPECTED_CHECKPOINT_STEPS,
    REMOVED_CHECKPOINT_STEPS,
    replay_archive_authorization_if_needed,
    write_archive_receipt,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_payload(
    *, c0: Path, outcome: Path, stamp: str = "e15_m075_test"
) -> dict[str, str]:
    return {
        "phase": "m075",
        "stamp": stamp,
        "source_hash": "a" * 64,
        "dataset_identity": json.dumps(
            gate.EXPECTED_DATASET_IDENTITY, sort_keys=True, separators=(",", ":")
        ),
        "runtime_identity": json.dumps(
            gate.EXPECTED_RUNTIME_IDENTITY, sort_keys=True, separators=(",", ":")
        ),
        "target_optimizer_updates": "128",
        "trajectory_query_budget": "2032",
        "group_size": "16",
        "protocol": "E15",
        "protocol_arm": "M075",
        "arm": "maxent",
        "maxent_alpha": "0.075",
        "c0_approval": str(c0.resolve()),
        "c0_approval_sha256": _hash(c0),
        "e14_outcome": str(outcome.resolve()),
        "e14_outcome_sha256": _hash(outcome),
    }


def _write_identity(path: Path, values: dict[str, str]) -> None:
    lines = ["key\tvalue", *(f"{key}\t{value}" for key, value in values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_e15_identity_binds_protocol_c0_and_e14_outcome(tmp_path):
    c0 = tmp_path / "c0.json"
    outcome = tmp_path / "e14.json"
    c0.write_text("{}\n", encoding="utf-8")
    outcome.write_text("{}\n", encoding="utf-8")
    identity = tmp_path / "e15_m075_test_e15_identity.tsv"
    values = _identity_payload(c0=c0, outcome=outcome)
    _write_identity(identity, values)

    observed, dataset, runtime = gate.read_treatment_identity(
        identity, arm="M075", c0_approval=c0, e14_outcome=outcome
    )
    assert observed["protocol"] == "E15"
    assert dataset == gate.EXPECTED_DATASET_IDENTITY
    assert runtime == gate.EXPECTED_RUNTIME_IDENTITY

    values["e14_outcome_sha256"] = "0" * 64
    _write_identity(identity, values)
    with pytest.raises(gate.GateError, match="outcome hash changed"):
        gate.read_treatment_identity(
            identity, arm="M075", c0_approval=c0, e14_outcome=outcome
        )


def test_e15_uses_unchanged_safety_diversity_thresholds_and_selection():
    c0 = {
        "p_valid_mean": 0.30,
        "exact_action_entropy_mean": 1.0,
        "n_eff_valid_mean": 2.0,
    }
    passing = gate.classify_treatment(
        arm="M075",
        c0_endpoint=c0,
        endpoint={
            "p_valid_mean": 0.24,
            "exact_action_entropy_mean": 1.0 + math.log(1.25),
            "n_eff_valid_mean": 2.5,
        },
        final_reward=0.01,
    )
    assert passing["viable"] is True

    selected, _ = gate.choose_arm(
        [
            {**passing, "arm": "M075", "n_eff_valid_mean": 2.50},
            {**passing, "arm": "M10", "n_eff_valid_mean": 2.62},
        ]
    )
    assert selected == "M075"
    selected, _ = gate.choose_arm(
        [
            {**passing, "arm": "M075", "n_eff_valid_mean": 2.50},
            {**passing, "arm": "M10", "n_eff_valid_mean": 2.70},
        ]
    )
    assert selected == "M10"


def test_e15_nonpositive_reward_is_runtime_valid_but_scientifically_unsafe(
    tmp_path, monkeypatch
):
    run = tmp_path / "run"
    debug = run / "debug_1"
    saved = debug / "saved_models"
    for step in (32, 64, 96, 128, 129):
        (saved / f"step_{step:05d}").mkdir(parents=True)
    eval_results = debug / "eval_results"
    eval_results.mkdir()
    for step in (0, 32, 64, 96, 128, 129):
        (eval_results / f"{step}_multi_answer.json").write_text(
            "{}\n", encoding="utf-8"
        )
    metrics = debug / "train_metrics.jsonl"
    metrics.write_text("{}\n", encoding="utf-8")

    rows = []
    for step in range(130):
        global_step = min(step, 128)
        row = {
            "trainer/step": step,
            "trainer/global_step": global_step,
            "trainer/policy_sgd_step": global_step,
        }
        if step == 0:
            row.update(
                {
                    "misc/prompt_dataset_len": 192,
                    "misc/query_step": 0,
                    "misc/prompt_consumed": 0,
                }
            )
        if step in (0, 32, 64, 96, 128, 129):
            row.update(
                {
                    "eval/multi_answer/eval_count": 96,
                    "eval/multi_answer/response_tok_len": 3,
                }
            )
        if step > 0:
            for key in gate.e14.MAXENT_BASE_KEYS:
                row[key] = 0.0
        if step == 129:
            row["misc/query_step"] = 2048
            row["misc/prompt_consumed"] = 2048
        rows.append(row)

    diagnostic = {
        "reward": 0.0,
        "entropy": 1.0,
        "entropy_loss": -0.01,
        "prefix_ratio_max": 1.0,
        "ratio_min": 1.0,
        "ratio_max": 1.0,
    }
    monkeypatch.setattr(gate.e14, "_read_metrics", lambda _: rows)
    monkeypatch.setattr(gate.e14, "_inspect_update", lambda *_, **__: diagnostic)
    monkeypatch.setattr(
        gate.e14.c0,
        "_weight_identity",
        lambda _: ("a" * 64, [{"name": "model.safetensors", "sha256": "b" * 64}]),
    )

    with pytest.raises(gate.GateError, match="reward is not positive"):
        gate.e14.inspect_treatment_metrics(run, arm="M05", alpha=0.05)

    summary, _, _ = gate.e14.inspect_treatment_metrics(
        run,
        arm="M075",
        alpha=0.075,
        require_positive_final_reward=False,
    )
    assert summary["final_32_mean_rollout_reward"] == 0.0
    classification = gate.classify_treatment(
        arm="M075",
        c0_endpoint={
            "p_valid_mean": 0.30,
            "exact_action_entropy_mean": 1.0,
            "n_eff_valid_mean": 2.0,
        },
        endpoint={
            "p_valid_mean": 0.30,
            "exact_action_entropy_mean": 1.5,
            "n_eff_valid_mean": 3.0,
        },
        final_reward=summary["final_32_mean_rollout_reward"],
    )
    assert classification["runtime_valid"] is True
    assert classification["behaviorally_safe"] is False
    assert classification["viable"] is False
    assert classification["failures"] == [
        "final-32 mean rollout reward is not positive"
    ]


def test_e15_metric_inspection_defers_reward_gate(monkeypatch, tmp_path):
    sentinel = ({"final_32_mean_rollout_reward": 0.0}, tmp_path, tmp_path)
    observed = {}

    def inspect(run_dir, **kwargs):
        observed["run_dir"] = run_dir
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(gate.e14, "inspect_treatment_metrics", inspect)
    assert gate._inspect_treatment_metrics(
        tmp_path,
        arm="M075",
        alpha=0.075,
        archived_removed_steps=None,
    ) == sentinel
    assert observed == {
        "run_dir": tmp_path,
        "arm": "M075",
        "alpha": 0.075,
        "archived_removed_steps": None,
        "require_positive_final_reward": False,
    }


def test_e14_no_dose_outcome_is_replayed_not_trusted(tmp_path, monkeypatch):
    c0 = tmp_path / "c0.json"
    m01 = tmp_path / "m01.json"
    m05 = tmp_path / "m05.json"
    for path in (c0, m01, m05):
        path.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema": gate.e14.COMPARISON_SCHEMA,
        "protocol": "E14",
        "status": "no_viable_dose",
        "selected_arm": None,
        "selection_rationale": "none",
        "single_seed_engineering_calibration": True,
        "does_not_authorize_scale_or_domain_expansion": True,
        "thresholds": {},
        "arms": [],
        "evidence": {
            "c0_approval": {"path": str(c0.resolve()), "sha256": _hash(c0)},
            "m01_result": {"path": str(m01.resolve()), "sha256": _hash(m01)},
            "m05_result": {"path": str(m05.resolve()), "sha256": _hash(m05)},
        },
        "compared_at_utc": "2026-07-18T00:00:00+00:00",
    }
    outcome = tmp_path / "outcome.json"
    outcome.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    replayed = {**payload, "compared_at_utc": "2026-07-18T01:00:00+00:00"}
    monkeypatch.setattr(gate.e14, "compare_result_payloads", lambda **_: replayed)
    assert gate.replay_e14_outcome(
        outcome, c0_approval_path=c0
    )["fully_replayed"] is True

    payload["arms"] = [{"arm": "M05", "viable": True}]
    outcome.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="does not match replayed"):
        gate.replay_e14_outcome(outcome, c0_approval_path=c0)


def _e15_validation_artifact(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    debug = run / "debug_1"
    saved = debug / "saved_models"
    for tag in EXPECTED_CHECKPOINT_STEPS:
        checkpoint = saved / tag
        checkpoint.mkdir(parents=True)
        content = b"endpoint" if tag in {"step_00128", "step_00129"} else tag.encode()
        (checkpoint / "model.safetensors").write_bytes(content)
        (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    metrics = debug / "train_metrics.jsonl"
    metrics.write_text("{}\n", encoding="utf-8")
    identity = tmp_path / "identity.tsv"
    identity.write_text("identity\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    evidence_paths = {
        "identity": identity,
        "metrics": metrics,
        "endpoint_audit": audit,
        "checkpoint_config": saved / "step_00128" / "config.json",
    }
    weight = saved / "step_00128" / "model.safetensors"
    digest = hashlib.sha256()
    digest.update(weight.name.encode("utf-8"))
    digest.update(bytes.fromhex(_hash(weight)))
    artifact = tmp_path / "m075_result.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": gate.RESULT_SCHEMA,
                "gate": gate.GATE_NAME,
                "protocol": "E15",
                "arm": "M075",
                "runtime_valid": True,
                "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                "job_id": "12345",
                "run_dir": str(run.resolve()),
                "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
                "identity": {"stamp": "e15_test", "source_hash": "a" * 64},
                "checks": {"exact_step_00128_audit": True},
                "metrics_summary": {
                    "step_128_weights_manifest_sha256": digest.hexdigest(),
                    "step_129_byte_identical_alias": True,
                },
                "evidence": {
                    label: {"path": str(path.resolve()), "sha256": _hash(path)}
                    for label, path in evidence_paths.items()
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact, saved


def test_e15_archival_receipt_replays_exact_retained_layout(tmp_path):
    artifact, saved = _e15_validation_artifact(tmp_path)
    receipt = write_archive_receipt(artifact)
    assert json.loads(receipt.read_text(encoding="utf-8"))["schema"] == E15_SCHEMA
    for tag in REMOVED_CHECKPOINT_STEPS:
        shutil.rmtree(saved / tag)
    authorization = replay_archive_authorization_if_needed(artifact)
    assert authorization is not None
    assert authorization["removed_steps"] == REMOVED_CHECKPOINT_STEPS
