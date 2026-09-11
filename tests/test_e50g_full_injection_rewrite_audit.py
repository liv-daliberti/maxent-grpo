from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_rewrite_audit_is_frozen_and_non_authorizing() -> None:
    source = (
        ROOT
        / "ops/math_strategy_calibration/"
        "audit_e50g_full_injection_rewrites.py"
    ).read_text(encoding="utf-8")
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e50g_full_injection_rewrite_audit_20260726.md"
    ).read_text(encoding="utf-8")
    assert "EXPECTED =" in source
    assert "KINDS = (" in source
    assert '"zero_false_new"' in source
    assert '"zero_signature_changes"' in source
    assert "RETROSPECTIVE FROZEN-CODE AUDIT" in protocol
    assert "cannot by itself authorize training" in protocol
