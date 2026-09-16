from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "ops/exp_scaling/audit_e111_proposal_retention_resume_recovery.py"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e111_proposal_retention_resume_recovery_is_exact_and_outcome_blind():
    audit = _load("e111_resume_recovery")
    report, violations = audit.validate()
    assert violations == []
    assert report["passed"] is True
    assert report["checkpoint_deserialization_only"] is True
    assert report["optimizer_update_changed"] is False
    assert report["treatment_changed"] is False
    assert report["outcomes_inspected"] is False
    assert report["pointmaze"] == "excluded"
    assert set(report["trigger_job_ids"]) >= {30674729, 30674733, 30674754}


def test_e111_proposal_retention_resume_recovery_rejects_treatment_tamper(
    tmp_path,
):
    audit = _load("e111_resume_recovery_tamper")
    payload = json.loads(audit.RECORD.read_text(encoding="utf-8"))
    payload["optimizer_update_changed"] = True
    tampered = tmp_path / "recovery.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    audit.RECORD = tampered
    report, violations = audit.validate()
    assert report["passed"] is False
    assert "E111 resume recovery has invalid optimizer_update_changed" in violations
