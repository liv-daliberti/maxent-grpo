from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "ops/exp_scaling/audit_e111_scheduler_amendments.py"
LEDGER = ROOT / "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e111_scheduler_amendments_are_exact_and_outcome_blind():
    audit = _load("e111_scheduler_amendments")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    report, violations = audit.validate(ledger, ledger["runs"])
    assert violations == []
    assert report["scheduler_only"] is True
    assert report["environment_changed"] is False
    assert report["treatment_environment_changed"] is False
    assert report["storage_only_recovery_changed"] is True
    assert report["checkout_wrapper_amendment_effective_for_submitted_jobs"] is False
    assert report["runtime_ops_amendment_effective_on_restart"] is True
    assert report["mechanism_gate_qwen3_checkpoint_interval"] == 2
    assert report["mechanism_gate_qwen3_checkpoint_start"] == 2
    assert report["mechanism_gate_qwen3_hardware"] == "a6000"
    assert report["mechanism_gate_qwen3_time_limit"] == "00:45:00"
    assert report["mechanism_gate_small_scale_node"] == "node[202-204,403]"
    assert report["mechanism_gate_small_scale_time_limit"] == "00:45:00"
    assert report["small_scale_node026_health_widening_applied"] is True
    assert report["e112_paired_hardware_changed"] is False
    assert report["checkpoint_validator_installed"] is True
    assert report["exact_timeout_requeue_applied"] is True
    assert report["outcomes_inspected"] is False


def test_e111_scheduler_amendment_rejects_environment_drift(tmp_path, monkeypatch):
    audit = _load("e111_scheduler_amendments_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.loads(audit.PARTITION_RECORD.read_text(encoding="utf-8"))
    payload["environment_changed"] = True
    tampered = tmp_path / "partition.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "PARTITION_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert "E111 partition amendment has invalid environment_changed" in violations


def test_e111_durability_amendment_rejects_recovery_drift(tmp_path, monkeypatch):
    audit = _load("e111_durability_amendment_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.loads(audit.DURABILITY_RECORD.read_text(encoding="utf-8"))
    payload["recovery_environment_changed"] = False
    tampered = tmp_path / "durability.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "DURABILITY_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert (
        "E111 3B durability amendment has invalid recovery_environment_changed"
        in violations
    )


def test_e111_runtime_ops_amendment_rejects_treatment_drift(
    tmp_path, monkeypatch
):
    audit = _load("e111_runtime_ops_amendment_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.loads(audit.RUNTIME_OPS_RECORD.read_text(encoding="utf-8"))
    payload["treatment_changed"] = True
    tampered = tmp_path / "runtime_ops.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "RUNTIME_OPS_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert (
        "E111 3B runtime-ops durability amendment has invalid treatment_changed"
        in violations
    )


def test_e111_l40_amendment_rejects_e112_hardware_drift(tmp_path, monkeypatch):
    audit = _load("e111_l40_amendment_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.loads(audit.L40_RECORD.read_text(encoding="utf-8"))
    payload["e112_paired_hardware_changed"] = True
    tampered = tmp_path / "l40.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "L40_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert (
        "E111 3B L40 placement amendment has invalid e112_paired_hardware_changed"
        in violations
    )


def test_e111_a6000_return_rejects_nonzero_l40_steps(tmp_path, monkeypatch):
    audit = _load("e111_a6000_return_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.loads(audit.RETURN_A6000_RECORD.read_text(encoding="utf-8"))
    payload["l40_training_steps"] = 1
    tampered = tmp_path / "return_a6000.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "RETURN_A6000_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert (
        "E111 3B A6000 return amendment has invalid l40_training_steps"
        in violations
    )


def test_e111_small_scale_backfill_rejects_environment_drift(
    tmp_path, monkeypatch
):
    audit = _load("e111_small_scale_backfill_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    original = audit.SMALL_BACKFILL_RECORD
    payload = json.loads(original.read_text(encoding="utf-8"))
    payload["environment_changed"] = True
    tampered = tmp_path / "small_scale.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "SMALL_BACKFILL_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert (
        "E111 small-scale node026 amendment has invalid environment_changed"
        in violations
    )

    monkeypatch.setattr(audit, "SMALL_BACKFILL_RECORD", original)
    payload = json.loads(audit.SMALL_HEALTH_RECORD.read_text(encoding="utf-8"))
    payload["treatment_changed"] = True
    tampered = tmp_path / "small_scale_health.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "SMALL_HEALTH_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert (
        "E111 small-scale node026 health widening has invalid treatment_changed"
        in violations
    )


def test_e111_two_step_amendments_reject_interval_or_threshold_drift(
    tmp_path, monkeypatch
):
    audit = _load("e111_two_step_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    original_two_step = audit.TWO_STEP_RECORD

    payload = json.loads(original_two_step.read_text(encoding="utf-8"))
    payload["new_interval"] = 4
    tampered = tmp_path / "two_step.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "TWO_STEP_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert "E111 3B two-step amendment has invalid new_interval" in violations

    monkeypatch.setattr(audit, "TWO_STEP_RECORD", original_two_step)
    payload = json.loads(audit.TWO_STEP_START_RECORD.read_text(encoding="utf-8"))
    payload["new_resume_from"] = 4
    tampered = tmp_path / "two_step_start.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "TWO_STEP_START_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert "E111 3B two-step start amendment has invalid new_resume_from" in violations


def test_e111_timeout_requeue_rejects_replacement_job_drift(tmp_path, monkeypatch):
    audit = _load("e111_timeout_requeue_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.loads(audit.TIMEOUT_REQUEUE_RECORD.read_text(encoding="utf-8"))
    payload["replacement_jobs_submitted"] = True
    tampered = tmp_path / "timeout_requeue.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "TIMEOUT_REQUEUE_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert (
        "E111 timeout requeue recovery has invalid replacement_jobs_submitted"
        in violations
    )


def test_e111_checkpoint_validator_rejects_optimizer_update_drift(tmp_path, monkeypatch):
    audit = _load("e111_checkpoint_validator_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.loads(audit.CHECKPOINT_VALIDATOR_RECORD.read_text(encoding="utf-8"))
    payload["optimizer_update_changed"] = True
    tampered = tmp_path / "checkpoint_validator.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "CHECKPOINT_VALIDATOR_RECORD", tampered)
    _, violations = audit.validate(ledger, ledger["runs"])
    assert (
        "E111 checkpoint ZIP-validation amendment has invalid optimizer_update_changed"
        in violations
    )
