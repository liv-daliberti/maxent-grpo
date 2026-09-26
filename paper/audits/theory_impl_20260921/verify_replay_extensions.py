"""Numerical regression check for the discrete bank-contraction proposition.

Run from any directory. This check supplements, and does not replace, the proof.
"""
from pathlib import Path
import json
import numpy as np


def main():
    seed = 20260921
    rng = np.random.default_rng(seed)
    cases = 20_000
    max_relative_excess = 0.0
    max_order_violation = 0.0
    for _ in range(cases):
        size = int(rng.integers(4, 25))
        correct = int(rng.integers(2, size))
        k = int(rng.integers(2, correct + 1))
        z = rng.normal(0, 2, size)
        p = np.exp(z - z.max())
        p /= p.sum()
        bank = rng.choice(correct, k, replace=False)
        mass = p[bank].sum()
        correctness = p[:correct].sum()
        coefficient = float(rng.uniform(0.1, 16))
        margin = float(rng.uniform(0.1, 3))
        dose = coefficient * (1 - correctness) + margin
        gamma = float(rng.uniform(0, 2))
        step_size = gamma / (margin * mass)
        target = np.zeros(size)
        target[bank] = 1 / k
        fresh = -correctness * p
        fresh[:correct] += p[:correct]
        next_z = z + step_size * (coefficient * fresh + dose * (target - p))
        span = float(np.ptp(z[bank]))
        next_span = float(np.ptp(next_z[bank]))
        lhs = np.expm1(next_span)
        rhs = (1 + np.expm1(-gamma) / k) * np.expm1(span)
        max_relative_excess = max(
            max_relative_excess, float((lhs - rhs) / max(1, rhs))
        )
        order = np.argsort(z[bank])
        max_order_violation = max(
            max_order_violation, float(-np.min(np.diff(next_z[bank][order])))
        )
    result = {
        "cases": cases,
        "seed": seed,
        "gamma_interval": [0, 2],
        "max_relative_contraction_excess": max_relative_excess,
        "max_bank_order_violation": max_order_violation,
        "passed": max_relative_excess < 1e-10 and max_order_violation < 1e-10,
    }
    output = Path(__file__).with_name("replay_discrete_checks.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["passed"]


if __name__ == "__main__":
    main()
