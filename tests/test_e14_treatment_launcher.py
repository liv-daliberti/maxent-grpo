from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from exp_scaling import verify_e14_c0_approval as gate


def _approval_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    evidence_root = tmp_path / "evidence"
    source_marker = evidence_root / "snapshot" / "src" / "oat_drgrpo" / "__init__.py"
    source_marker.parent.mkdir(parents=True)
    source_marker.write_text("# frozen\n", encoding="utf-8")
    files = {
        "identity": evidence_root / "identity.tsv",
        "preflight_approval": evidence_root / "preflight.json",
        "metrics": evidence_root / "metrics.jsonl",
        "source_snapshot_marker": source_marker,
        "stdout": evidence_root / "stdout.log",
        "stderr": evidence_root / "stderr.log",
        "endpoint_audit": evidence_root / "endpoint.json",
        "checkpoint_config": evidence_root / "config.json",
    }
    for label, path in files.items():
        if label != "source_snapshot_marker":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{label}\n", encoding="utf-8")
    evidence = {
        label: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for label, path in files.items()
    }
    source_hash = "a" * 64
    payload = {
        "approved": True,
        "gate": gate.GATE_NAME,
        "protocol": "E14",
        "arm": "C0",
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": "12345",
        "run_dir": str((tmp_path / "run").resolve()),
        "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
        "identity": {
            "stamp": "e14_c0_test",
            "source_hash": source_hash,
            "dataset": gate.EXPECTED_DATASET_IDENTITY,
            "runtime": gate.EXPECTED_RUNTIME_IDENTITY,
        },
        "checks": {key: True for key in gate.REQUIRED_C0_CHECKS},
        "endpoint_summary": {
            "p_valid_mean": 0.25,
            "exact_action_entropy_mean": 1.5,
            "n_eff_valid_mean": 2.0,
        },
        "evidence": evidence,
    }
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    def replay(**_kwargs):
        result = dict(payload)
        result["approved_at_utc"] = datetime.now(timezone.utc).isoformat()
        return result

    monkeypatch.setattr(gate, "check_c0", replay)
    monkeypatch.setattr(
        gate, "replay_archive_authorization_if_needed", lambda _path: None
    )
    return approval, source_hash, files


def test_c0_approval_replays_before_treatment(tmp_path, monkeypatch):
    approval, source_hash, _ = _approval_fixture(tmp_path, monkeypatch)

    summary = gate.verify_c0_approval_for_source(
        approval,
        expected_source_hash=source_hash,
        logical_repo_root=tmp_path,
    )

    assert summary["full_c0_gate_replayed"] is True
    assert summary["endpoint_p_valid_mean"] == 0.25
    assert summary["evidence_files_verified"] == 8


def test_c0_approval_fails_when_evidence_changes(tmp_path, monkeypatch):
    approval, source_hash, files = _approval_fixture(tmp_path, monkeypatch)
    files["endpoint_audit"].write_text("changed\n", encoding="utf-8")

    with pytest.raises(gate.GateError, match="changed after approval"):
        gate.verify_c0_approval_for_source(
            approval,
            expected_source_hash=source_hash,
            logical_repo_root=tmp_path,
        )


def test_c0_approval_fails_on_treatment_source_drift(tmp_path, monkeypatch):
    approval, _, _ = _approval_fixture(tmp_path, monkeypatch)

    with pytest.raises(gate.GateError, match="treatment Python source differs"):
        gate.verify_c0_approval_for_source(
            approval,
            expected_source_hash="b" * 64,
            logical_repo_root=tmp_path,
        )


def test_treatment_launcher_contract_is_fixed_and_fail_closed():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "exp_scaling"
        / "launch_e14_canonical_smoke.sh"
    ).read_text(encoding="utf-8")

    assert "m01|m01-config)" in launcher
    assert "m05|m05-config)" in launcher
    assert "MAXENT_ALPHA=0.01" in launcher
    assert "MAXENT_ALPHA=0.05" in launcher
    assert 'REQUIRE_C0_APPROVAL=1' in launcher
    assert 'OAT_ZERO_E14_C0_APPROVAL' in launcher
    assert "verify_e14_c0_approval.py" in launcher
    assert '--expected-source-hash "$source_hash"' in launcher
    assert 'export OAT_ZERO_ONLY_ARMS="$MANIFEST_ARM"' in launcher
    assert 'export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0' in launcher
    assert 'export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0' in launcher
    assert 'export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0' in launcher
    assert 'export OAT_ZERO_TRAIN_SEEDS=9005' in launcher
    assert 'TARGET_UPDATES=128' in launcher
    assert 'LEARNING_RATE=0.0000002' in launcher
