from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "calibrate_e49j_mathir_signature_veto.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49j_restricted_mathir_signature_calibration_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49j_mathir_signature_veto.sh"
)
SLURM = ROOT / "ops/slurm/e49j_mathir_signature_veto_node915.slurm"


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49j_mathir_signature_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assessment(left, right, **updates):
    payload = {
        "route_a_sound_and_self_contained": True,
        "route_b_sound_and_self_contained": True,
        "route_a_signature_complete": True,
        "route_b_signature_complete": True,
        "route_a_mathir_signature": left,
        "route_b_mathir_signature": right,
        "rationale": "The restricted signatures were parsed literally.",
    }
    payload.update(updates)
    return payload


def test_generic_bookkeeping_is_removed_before_exact_comparison():
    module = _load()
    left = [
        "READ_GIVENS",
        "INVERSE_VARIATION_INVARIANT",
        "SOLVE_EQUATION",
    ]
    right = [
        "INVERSE_VARIATION_INVARIANT",
        "ARITHMETIC",
    ]
    assert module._canonical_signature(left) == (
        "INVERSE_VARIATION_INVARIANT",
    )
    assert module._distinct_vote(_assessment(left, right)) is False


def test_restricted_primitive_difference_votes_distinct_only_when_sound():
    module = _load()
    left = ["GCD_LCM_PRODUCT_THEOREM", "ARITHMETIC"]
    right = ["PRIME_FACTOR_GCD_LCM", "ARITHMETIC"]
    assert module._distinct_vote(_assessment(left, right)) is True
    assert (
        module._distinct_vote(
            _assessment(
                left,
                right,
                route_b_sound_and_self_contained=False,
            )
        )
        is False
    )
    assert (
        module._distinct_vote(
            _assessment(left, right, route_a_signature_complete=False)
        )
        is False
    )


def test_protocol_freezes_enum_canonicalization_and_unanimity():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "FROZEN BEFORE ANY E49J 72B REQUEST" in text
    assert "restricted executable MathIR" in text
    assert "deterministic local comparator" in text
    assert "four-way unanimity" in text
    assert "zero false-new" in text


def test_calibration_anchors_are_explicit_aliases_and_distinctions():
    module = _load()
    prompt = module._prompt(
        {
            "problem": "p",
            "reference_answer": "a",
            "route_a": {"plan": "x"},
            "route_b": {"plan": "y"},
        },
        role="mathir_canonical_parser",
        swapped=False,
    )
    assert "INVERSE_VARIATION_INVARIANT" in prompt
    assert "INDEPENDENT_CHOICE_PRODUCT" in prompt
    assert "RADIX_GROUP_MAP" in prompt
    assert "RADIX_REPEATED_DIVISION" in prompt
    assert "GCD_LCM_PRODUCT_THEOREM" in prompt
    assert "PRIME_FACTOR_GCD_LCM" in prompt


def test_launcher_binds_failed_free_text_predecessor_before_requests():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    slurm = SLURM.read_text(encoding="utf-8")
    assert "E49J predecessor/cohort evidence changed" in launcher
    assert '"requests_at_freeze": 0' in launcher
    assert '"decision_threshold": "four_way_unanimity"' in launcher
    assert "--nodelist=node915" in slurm
    assert "E49J frozen binding failed" in slurm
    assert "--workers 4" in slurm
