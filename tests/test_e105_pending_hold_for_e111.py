from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "var/artifacts/e105_pending_hold_for_e111.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field(record: str, name: str) -> str:
    marker = f"{name}="
    assert marker in record
    return record.split(marker, 1)[1].split(" ", 1)[0]


def _submit_line(record: str) -> str:
    return record.split("SubmitLine=", 1)[1].split(" WorkDir=", 1)[0]


def test_e105_hold_is_exact_reversible_and_nonterminating():
    payload = json.loads(RECORD.read_text(encoding="utf-8"))
    assert payload["schema"] == "e105_pending_hold_for_e111_v1"
    assert payload["applied"] is True
    assert payload["reversible"] is True
    assert payload["jobs_signaled"] is False
    assert payload["jobs_canceled"] is False
    assert payload["endpoint_outcomes_inspected"] is False
    assert payload["pointmaze"] == "excluded"
    assert payload["held_job_ids"] == payload["pending_job_ids"]
    assert len(payload["held_job_ids"]) == 51
    assert len(payload["running_job_ids"]) == 21
    assert not set(payload["held_job_ids"]) & set(payload["running_job_ids"])

    for path_key, digest_key in (
        ("protocol", "protocol_sha256"),
        ("ledger", "ledger_sha256"),
        ("supersession", "supersession_sha256"),
    ):
        path = Path(payload[path_key])
        assert path.is_file()
        assert payload[digest_key] == _digest(path)

    expected = {str(job_id) for job_id in payload["held_job_ids"]}
    before = payload["before_scheduler_records"]
    after = payload["after_scheduler_records"]
    assert set(before) == expected
    assert set(after) == expected
    for job_id in expected:
        assert _field(before[job_id], "JobState") == "PENDING"
        assert _field(after[job_id], "JobState") == "PENDING"
        assert _field(after[job_id], "Reason") in {"JobHeldUser", "JobHeldAdmin"}
        assert _submit_line(before[job_id]) == _submit_line(after[job_id])


def test_e105_hold_does_not_claim_e112_release_eligibility():
    payload = json.loads(RECORD.read_text(encoding="utf-8"))
    assert payload["applied"] is True
    assert payload["held_job_ids"]
    assert all(
        _field(record, "JobState") == "PENDING"
        for record in payload["after_scheduler_records"].values()
    )
