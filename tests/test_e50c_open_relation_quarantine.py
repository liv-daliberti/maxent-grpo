from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_quarantine_is_fail_closed_and_preserves_evidence() -> None:
    script = (
        ROOT
        / "ops/math_strategy_calibration/"
        "quarantine_e50c_open_relation_result.py"
    ).read_text(encoding="utf-8")
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e50c_open_relation_quarantine_20260726.md"
    ).read_text(encoding="utf-8")
    for fragment in (
        '"pass": False',
        '"selected_source_indices": []',
        '"wrong_route_success_count": 0',
        "false_new_failure_sha256",
        "progress_log_sha256",
        "os.replace",
    ):
        assert fragment in script
    assert "ACTION SPECIFIED BEFORE TERMINATION" in protocol
    assert "authorize no training" in protocol
