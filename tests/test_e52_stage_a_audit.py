from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "ops/exp_scaling/audit_e52_stage_a.py"
SPEC = importlib.util.spec_from_file_location("audit_e52_stage_a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _run(
    *,
    distinct: float,
    pass8: float,
    status: str = "complete",
):
    return {
        "status": status,
        "evaluations": [
            {
                "step": step,
                "distinct8": distinct,
                "pass8": pass8,
                "mean8": 0.3,
            }
            for step in range(1, 17)
        ],
    }


def _seeds(
    *,
    hybrid_distinct: float,
    hybrid_pass8: float,
    status: str = "complete",
):
    return {
        seed: {
            "runs": {
                "grpo": _run(distinct=0.4, pass8=0.3, status=status),
                "maxent_inverse": _run(
                    distinct=0.6,
                    pass8=0.4,
                    status=status,
                ),
                "maxent_inverse_canonical": _run(
                    distinct=hybrid_distinct,
                    pass8=hybrid_pass8,
                    status=status,
                ),
            }
        }
        for seed in MODULE.SEEDS
    }


def test_seed_mean_gate_requires_target_free_stable_multiplicity():
    result = MODULE.seed_mean_behavioral_gate(
        _seeds(hybrid_distinct=0.8, hybrid_pass8=0.4)
    )

    assert result["status"] == "pass"
    assert result["hybrid_wins"] == 8
    assert result["hybrid_excess_wins"] == 8
    assert result["hybrid_positive_multiplicity_boundaries"] == 8


def test_seed_mean_gate_rejects_three_seed_single_mode_plateau():
    result = MODULE.seed_mean_behavioral_gate(
        _seeds(hybrid_distinct=0.5, hybrid_pass8=0.5)
    )

    assert result["status"] == "fail"
    assert result["checks"]["higher_mean_distinct8"] is True
    assert result["checks"]["positive_multiplicity_at_least_six"] is False


def test_seed_mean_gate_is_provisional_until_every_seed_is_terminal():
    result = MODULE.seed_mean_behavioral_gate(
        _seeds(
            hybrid_distinct=0.8,
            hybrid_pass8=0.4,
            status="running",
        )
    )

    assert result["status"] == "pending"
    assert result["provisional_last_eight"]["status"] == "pass"
