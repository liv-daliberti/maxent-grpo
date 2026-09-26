from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "calibrate_e49i_algorithm_signature_veto.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49i_algorithm_signature_veto_calibration_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49i_algorithm_signature_veto.sh"
)
SLURM = (
    ROOT
    / "ops/slurm/"
    "e49i_algorithm_signature_veto_node915.slurm"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49i_algorithm_signature_test",
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
        "route_a_operator_signature": [
            "invoke gcd-lcm product theorem",
            "multiply inputs",
        ],
        "route_b_operator_signature": [
            "prime factorize inputs",
            "construct gcd and lcm",
            "multiply",
        ],
        "same_primitive_signature": False,
        "routine_signature_translation_exists": False,
        "routine_signature_translation_witness": "",
        "different_labels_or_granularity_only": False,
        "route_a_exclusive_operator": "invoke gcd-lcm theorem",
        "route_b_exclusive_operator": "construct prime exponent table",
        "relation": "distinct",
        "rationale": "Both routes execute a necessary exclusive operator.",
    }
    payload.update(updates)
    return payload


def test_strict_vote_requires_sound_exclusive_algorithm_signatures():
    module = _load()
    assert module._strict_distinct(_assessment()) is True
    for update in (
        {"route_a_sound_and_self_contained": False},
        {"same_primitive_signature": True},
        {"routine_signature_translation_exists": True},
        {"different_labels_or_granularity_only": True},
        {"route_a_exclusive_operator": ""},
        {"relation": "ambiguous"},
    ):
        assert module._strict_distinct(_assessment(**update)) is False


def test_protocol_freezes_algorithm_boundary_and_zero_false_new_gate():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49I 72B REQUEST" in text
    assert "minimal **executed algorithm signature**" in text
    assert "at least three of four" in text
    assert "zero false-new predictions" in text
    assert "a unit detour is" in text


def test_launcher_binds_failed_e49g_and_fresh_request_count():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    slurm = SLURM.read_text(encoding="utf-8")
    assert "E49I predecessor/cohort evidence changed" in launcher
    assert '"requests_at_freeze": 0' in launcher
    assert '"decision_threshold": "at_least_3_of_4"' in launcher
    assert "--nodelist=node915" in slurm
    assert "E49I frozen binding failed" in slurm
    assert "--workers 4" in slurm
