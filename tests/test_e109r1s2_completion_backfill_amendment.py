from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import apply_e109r1s2_completion_backfill_amendment as amendment  # noqa: E402


def test_amendment_targets_only_the_two_registered_continuations():
    assert amendment.JOB_IDS == (30874012, 30874013)
    payload = json.loads(
        amendment.CONTINUATION_ARTIFACT.read_text(encoding="utf-8")
    )
    records = amendment.records_by_job(payload)
    assert {job_id: records[job_id]["seed"] for job_id in records} == {
        30874012: 73,
        30874013: 74,
    }


def test_amendment_changes_only_backfill_surface():
    assert amendment.OLD_NODE_LIST == "node[104,205-207,805]"
    assert amendment.NEW_NODE_LIST == "node[103-104,205-207,805]"
    assert amendment.OLD_TIME_LIMIT == "3-00:00:00"
    assert amendment.NEW_TIME_LIMIT == "12:00:00"


def test_protocol_preserves_science_and_terminal_gate():
    protocol = amendment.PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "pending at zero runtime",
        "before reading either E109 or",
        "E112 evaluation endpoint",
        "job IDs",
        "scientific environment",
        "3,072 optimizer steps",
        "13/15 to 15/15",
    ):
        assert required in protocol
