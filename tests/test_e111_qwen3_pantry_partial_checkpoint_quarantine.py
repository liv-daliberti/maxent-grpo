from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "ops/exp_scaling/audit_e111_qwen3_pantry_partial_checkpoint_quarantine.py"
GATE = ROOT / "ops/exp_scaling/audit_e111_verified_support_discovery_mechanism_gate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"ok")


def test_pantry_quarantine_validator_accepts_exact_finalized_recovery(tmp_path, monkeypatch):
    audit = _load(AUDITOR, "e111_pantry_recovery_pass")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen before recovery\n", encoding="utf-8")
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    model = quarantine / "mp_rank_00_model_states.pt"
    optimizer = quarantine / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    _zip(model)
    optimizer.write_bytes(b"truncated")
    stdout = tmp_path / "pantry.out"
    stdout.write_text("post-requeue clean log\n", encoding="utf-8")
    record = tmp_path / "record.json"
    counts = {"tracebacks": 1, "partial_checkpoint": 1}
    payload = {
        "schema": "e111_qwen3_pantry_partial_checkpoint_quarantine_v1",
        "protocol": str(protocol),
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "job_id": 30674762,
        "latest_marker": "step_00002",
        "checkpoint_storage_only": True,
        "quarantined_bytes_recoverable": True,
        "bytes_deleted": False,
        "job_signaled_for_quarantine": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "verified_after_requeue": True,
        "stdout_reinitialized_on_requeue": True,
        "pre_recovery_failure_counts": counts,
        "post_recovery_failure_counts": {"tracebacks": 0, "partial_checkpoint": 0},
        "post_recovery_valid_checkpoints": [{"path": "step_00004"}],
        "quarantined_checkpoint_evidence": {
            model.name: {"path": str(model), "size": model.stat().st_size},
            optimizer.name: {"path": str(optimizer), "size": optimizer.stat().st_size},
        },
    }
    record.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "RECORD", record)
    monkeypatch.setattr(audit, "CORRUPTION", audit.CORRUPTION)
    log_dir = tmp_path / "var/artifacts/logs"
    log_dir.mkdir(parents=True)
    (log_dir / "e111-q3-pantry-30674762.out").write_text(
        stdout.read_text(encoding="utf-8"), encoding="utf-8"
    )
    report, violations = audit.validate()
    assert violations == []
    assert report["passed"] is True
    assert report["allowed_traceback_occurrences"] == 0


def test_terminal_gate_allows_only_exact_recorded_partial_checkpoint_traceback(tmp_path):
    gate = _load(GATE, "e111_pantry_gate_exact")
    log = tmp_path / "pantry.out"
    log.write_text(
        "Traceback (most recent call last):\n"
        + gate.RECOVERED_PARTIAL_CHECKPOINT_FAILURE
        + "\n",
        encoding="utf-8",
    )
    kwargs = {
        "job_id": 30674762,
        "recovered_occurrences": {},
        "partial_checkpoint_occurrences": {"30674762": 1},
    }
    assert gate.unrecovered_stdout_failures(log, **kwargs) == []
    log.write_text(log.read_text() + "Traceback (most recent call last):\nboom\n")
    markers = gate.unrecovered_stdout_failures(log, **kwargs)
    assert "Traceback (most recent call last)" in markers
