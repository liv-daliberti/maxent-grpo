from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "calibrate_e49g_hardened_pair_veto.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49g_hardened_pair_veto_calibration_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49g_hardened_pair_veto.sh"
)
SLURM = ROOT / "ops/slurm/e49g_hardened_pair_veto_node915.slurm"


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49g_hardened_pair_veto_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assessment(**updates):
    payload = {
        "route_a_sound_and_self_contained": True,
        "route_b_sound_and_self_contained": True,
        "route_a_minimal_core": "factor both integers",
        "route_b_minimal_core": "invoke the gcd-lcm product theorem",
        "same_decisive_core": False,
        "routine_reduction_exists": False,
        "routine_reduction_witness": "No routine reduction exists.",
        "different_labels_or_granularity_only": False,
        "route_a_exclusive_necessary_fact": "the prime exponent table",
        "route_b_exclusive_necessary_fact": "the gcd-lcm product identity",
        "relation": "distinct",
        "rationale": "The necessary mathematical facts differ.",
    }
    payload.update(updates)
    return payload


def test_strict_distinct_requires_every_conservative_condition():
    module = _load()
    assert module._strict_distinct(_assessment()) is True
    for updates in (
        {"route_a_sound_and_self_contained": False},
        {"same_decisive_core": True},
        {"routine_reduction_exists": True},
        {"different_labels_or_granularity_only": True},
        {"relation": "ambiguous"},
        {
            "route_b_exclusive_necessary_fact": (
                "the prime exponent table"
            )
        },
    ):
        assert module._strict_distinct(_assessment(**updates)) is False


def test_protocol_freezes_four_way_unanimous_veto_before_requests():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49G 72B REQUEST" in text
    assert "four-way unanimity" in text
    assert "zero false-new predictions" in text
    assert "at least three of the four" in text


def test_launcher_and_slurm_freeze_blinded_cohort_before_requests():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    slurm = SLURM.read_text(encoding="utf-8")
    assert "requests_at_freeze" in launcher
    assert "assessment_count" in launcher
    assert "116" in launcher
    assert "--nodelist=node915" in slurm
    assert "E49G frozen binding failed" in slurm
    assert "--workers 4" in slurm
