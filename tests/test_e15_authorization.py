from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from exp_scaling import verify_e15_authorization as gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    c0 = tmp_path / "c0.json"
    m01 = tmp_path / "m01.json"
    m05 = tmp_path / "m05.json"
    c0.write_text("{}\n", encoding="utf-8")
    m01.write_text("{}\n", encoding="utf-8")
    m05.write_text("{}\n", encoding="utf-8")
    arms = [
        {
            "arm": "M01",
            "runtime_valid": True,
            "behaviorally_safe": True,
            "diversity_effective": False,
            "viable": False,
            "failures": [
                "exact action-entropy gain < log(1.25)",
                "exact valid-mode effective support gain < 25%",
            ],
            "exact_action_entropy_gain_vs_c0": 0.08,
            "n_eff_valid_ratio_vs_c0": 1.05,
            "p_valid_retention_vs_c0": 0.98,
        },
        {
            "arm": "M05",
            "runtime_valid": True,
            "behaviorally_safe": True,
            "diversity_effective": False,
            "viable": False,
            "failures": ["exact valid-mode effective support gain < 25%"],
            "exact_action_entropy_gain_vs_c0": 0.47,
            "n_eff_valid_ratio_vs_c0": 1.19,
            "p_valid_retention_vs_c0": 0.94,
        },
    ]
    outcome = {
        "schema": gate.COMPARISON_SCHEMA,
        "protocol": "E14",
        "status": "no_viable_dose",
        "selected_arm": None,
        "selection_rationale": "no runtime-valid arm passed both safety and diversity gates",
        "single_seed_engineering_calibration": True,
        "does_not_authorize_scale_or_domain_expansion": True,
        "thresholds": dict(gate.EXPECTED_THRESHOLDS),
        "arms": arms,
        "evidence": {
            "c0_approval": {"path": str(c0.resolve()), "sha256": _sha(c0)},
            "m01_result": {"path": str(m01.resolve()), "sha256": _sha(m01)},
            "m05_result": {"path": str(m05.resolve()), "sha256": _sha(m05)},
        },
        "compared_at_utc": "2026-07-18T00:00:00+00:00",
    }
    outcome_path = tmp_path / "outcome.json"
    outcome_path.write_text(
        json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gate, "EXPECTED_C0_APPROVAL_SHA256", _sha(c0))
    monkeypatch.setattr(gate, "EXPECTED_E14_OUTCOME_SHA256", _sha(outcome_path))
    monkeypatch.setattr(
        gate,
        "verify_c0_approval_for_source",
        lambda *_args, **_kwargs: {"full_c0_gate_replayed": True},
    )

    def replay(**_kwargs):
        payload = dict(outcome)
        payload["compared_at_utc"] = "2026-07-18T00:00:01+00:00"
        return payload

    monkeypatch.setattr(gate, "compare_result_payloads", replay)
    return c0, m01, m05, outcome_path


def test_e15_authorization_replays_c0_and_e14_comparison(tmp_path, monkeypatch):
    c0, _, _, outcome = _authorization_fixture(tmp_path, monkeypatch)

    summary = gate.verify_e15_authorization(
        c0_approval_path=c0,
        e14_outcome_path=outcome,
        expected_source_hash=gate.EXPECTED_SOURCE_HASH,
        logical_repo_root=tmp_path,
    )

    assert summary["authorized_protocol"] == "E15"
    assert summary["authorized_arms"] == ["M075", "M10"]
    assert summary["c0_gate_replayed"] is True
    assert summary["e14_comparison_replayed"] is True
    assert math.isclose(summary["m05_n_eff_valid_ratio_vs_c0"], 1.19)


def test_e15_authorization_rejects_changed_comparison(tmp_path, monkeypatch):
    c0, _, _, outcome = _authorization_fixture(tmp_path, monkeypatch)
    outcome.write_text("{}\n", encoding="utf-8")

    with pytest.raises(gate.GateError, match="different E14 comparison"):
        gate.verify_e15_authorization(
            c0_approval_path=c0,
            e14_outcome_path=outcome,
            expected_source_hash=gate.EXPECTED_SOURCE_HASH,
            logical_repo_root=tmp_path,
        )


def test_e15_authorization_rejects_changed_evidence(tmp_path, monkeypatch):
    c0, _, m05, outcome = _authorization_fixture(tmp_path, monkeypatch)
    m05.write_text("changed\n", encoding="utf-8")

    with pytest.raises(gate.GateError, match="evidence 'm05_result' changed"):
        gate.verify_e15_authorization(
            c0_approval_path=c0,
            e14_outcome_path=outcome,
            expected_source_hash=gate.EXPECTED_SOURCE_HASH,
            logical_repo_root=tmp_path,
        )


def test_e15_launcher_contract_is_fixed_and_fail_closed():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "exp_scaling"
        / "launch_e15_canonical_dose.sh"
    ).read_text(encoding="utf-8")

    assert "m075|m075-config)" in launcher
    assert "m10|m10-config)" in launcher
    assert "MAXENT_ALPHA=0.075" in launcher
    assert "MAXENT_ALPHA=0.10" in launcher
    assert "OAT_ZERO_E15_C0_APPROVAL" in launcher
    assert "OAT_ZERO_E15_E14_OUTCOME" in launcher
    assert "verify_e15_authorization.py" in launcher
    assert "source_tree_hash" in launcher
    assert 'export OAT_ZERO_TRAIN_SEEDS=9005' in launcher
    assert 'export OAT_ZERO_ONLY_ARMS=maxent' in launcher
    assert 'export OAT_ZERO_MAXENT_CONTROL_RATIO=0' in launcher
    assert 'export OAT_ZERO_MAXENT_DUAL_RATIO=0' in launcher
    assert 'export OAT_ZERO_MAXENT_LENGTH_TARGET=0' in launcher
    assert 'TARGET_UPDATES=128' in launcher
    assert 'MAX_QUERIES" != "2032"' in launcher
    assert 'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1' in launcher
    assert 'OAT_ZERO_AUTO_RESUME=0' in launcher
    assert 'OAT_ZERO_WATCHDOG_REQUEUE=0' in launcher
