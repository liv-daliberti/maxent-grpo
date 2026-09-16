from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "ops/exp_scaling/audit_e111_verified_support_discovery_mechanism_gate.py"
CONTRACT = ROOT / "tests/test_e111_verified_support_discovery_mechanism_gate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e111_live_auditor_reports_partial_progress_without_crashing(tmp_path):
    audit = _load(AUDITOR, "e111_partial_audit")
    contract = _load(CONTRACT, "e111_contract_fixture")
    row = contract._mechanism_row()
    row["misc/global_step"] = 1
    row["train/canonical_replay_retention_safe_balance"] = 0.0
    attempt = tmp_path / "debug_job1"
    attempt.mkdir()
    import json
    (attempt / "train_metrics.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    report, violations = audit.parse_run(tmp_path)
    assert report["last_step"] == 1
    assert report["semantic_advantage_min"] == -0.02
    assert report["semantic_advantage_max"] == 0.04
    assert report["semantic_both_sign_updates"] == 1
    assert "only 1/64 optimizer steps" in violations
    assert (
        "retention-safe balance was disabled on a replay update"
        not in violations
    )

    cleanup = tmp_path / "cleanup.out"
    cleanup.write_text(
        "Traceback (most recent call last):\n"
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'/tmp/od2961'\n"
        "Traceback (most recent call last):\n"
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'/tmp/test_plasma-abc123'\n",
        encoding="utf-8",
    )
    assert audit.unrecovered_stdout_failures(
        cleanup,
        job_id=30674758,
        recovered_occurrences={},
        partial_checkpoint_occurrences={},
    ) == []
    cleanup.write_text(
        cleanup.read_text(encoding="utf-8")
        + "Traceback (most recent call last):\nRuntimeError: real failure\n",
        encoding="utf-8",
    )
    assert "Traceback (most recent call last)" in audit.unrecovered_stdout_failures(
        cleanup,
        job_id=30674758,
        recovered_occurrences={},
        partial_checkpoint_occurrences={},
    )

    receipt = tmp_path / "TRAINING_COMPLETE.json"
    receipt.write_text(
        '{"schema":"oat_zero_training_complete_v1","terminal_step":64}\n',
        encoding="utf-8",
    )
    assert audit.completed_by_receipt(tmp_path, {"last_step": 64}) is True
    assert audit.completed_by_receipt(tmp_path, {"last_step": 63}) is False
    receipt.write_text(
        '{"schema":"wrong","terminal_step":64}\n',
        encoding="utf-8",
    )
    assert audit.completed_by_receipt(tmp_path, {"last_step": 64}) is False
