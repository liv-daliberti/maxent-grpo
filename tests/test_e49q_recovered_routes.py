from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "certify_e49q_recovered_routes.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49q_visible_trace_recovery_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49q_recovered_routes.sh"
)
SLURM = ROOT / "ops/slurm/e49q_recovered_routes_node915.slurm"
CALIBRATION = (
    ROOT
    / "var/artifacts/e49j_mathir_signature_veto_v1/"
    "calibration_report.json"
)
EVIDENCE = [
    ROOT / "var/artifacts/e49h_curated_distinct_routes_toy_v1",
    ROOT / "var/artifacts/e49k_curated_eval_buffer_v1",
    ROOT / "var/artifacts/e49n_curated_eval_recovery_v1",
]


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49q_recovered_routes_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_requires_complete_traces_and_one_exact_pass_per_route():
    module = _load()
    passing = {
        "completed_invalid": False,
        "pass": True,
    }
    failing = {
        "completed_invalid": False,
        "pass": False,
    }
    record = {
        "pass": False,
        "sound_audits": {
            "S1": [passing, failing],
            "S2": [failing, passing],
        },
    }
    assert module._eligible(record)
    record["sound_audits"]["S2"] = [failing, failing]
    assert not module._eligible(record)
    record["sound_audits"]["S2"] = [
        passing,
        {"completed_invalid": True, "pass": False},
    ]
    assert not module._eligible(record)


def test_existing_failed_cohorts_offer_identity_bound_recovery_support():
    module = _load()
    candidates = module._validated_candidates(EVIDENCE, CALIBRATION)
    eligible = [
        candidate for candidate in candidates
        if module._eligible(candidate["record"])
    ]
    assert len(candidates) == 38
    assert len(eligible) == 9
    assert len(
        {
            candidate["record"]["row_id"]
            for candidate in eligible
            if candidate["record"]["split"] == "eval"
        }
    ) == 7


def test_preflight_is_read_only_and_reports_all_traces():
    module = _load()
    args = argparse.Namespace(
        evidence=EVIDENCE,
        e49j_calibration=CALIBRATION,
    )
    candidates = module._validated_candidates(
        args.evidence,
        args.e49j_calibration,
    )
    assert all(
        all(
            len(candidate["record"]["sound_audits"][strategy_id]) == 2
            for strategy_id in ("S1", "S2")
        )
        for candidate in candidates
        if module._eligible(candidate["record"])
    )


def test_launcher_freezes_exact_recovery_cohort_and_unanimous_gate():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    slurm = SLURM.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49Q 72B REQUEST" in protocol
    assert "four-way unanimity" in protocol
    assert "eligible_candidate_count" in launcher
    assert 'record.get("eligible_candidate_count") != 14' in launcher
    assert '"requests_at_freeze": 0' in launcher
    assert "--workers 4" in slurm
    assert "--nodelist=node915" in slurm
