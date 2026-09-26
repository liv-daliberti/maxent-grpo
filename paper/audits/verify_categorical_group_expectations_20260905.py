#!/usr/bin/env python3
"""Independent ordered-group enumeration for the categorical mean-flow lemmas.

Uses only the standard library; never imports the training implementation.
This supplementary check is finite arithmetic, not the analytic proof.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def group_weight(kind: str, successes: int, width: int) -> float:
    if kind == "drgrpo":
        return 1.0
    if kind == "grpo_population_std":
        return 1.0 / math.sqrt(successes * (width - successes) / width**2)
    if kind == "grpo_sample_std_eps":
        return 1.0 / (
            math.sqrt(successes * (width - successes) / (width * (width - 1)))
            + 1e-8
        )
    raise ValueError(kind)


def check_group_expectation(
    probabilities: tuple[float, ...], rewards: tuple[int, ...], width: int, kind: str
) -> dict:
    """Average the literal sampled estimator over all ordered category groups."""
    correct_mass = sum(p * r for p, r in zip(probabilities, rewards))
    gradient = [p * (r - correct_mass) for p, r in zip(probabilities, rewards)]
    mean = [0.0] * len(probabilities)
    for categories in itertools.product(range(len(probabilities)), repeat=width):
        group_probability = math.prod(probabilities[a] for a in categories)
        successes = sum(rewards[a] for a in categories)
        if successes in (0, width):
            continue
        for category in categories:
            if kind == "maxrl":
                advantage = width * rewards[category] / successes - 1.0
            else:
                advantage = group_weight(kind, successes, width) * (
                    rewards[category] - successes / width
                )
            for coordinate, probability in enumerate(probabilities):
                score = float(category == coordinate) - probability
                mean[coordinate] += group_probability * advantage * score / width

    if kind == "maxrl":
        coefficient = (1.0 - (1.0 - correct_mass) ** (width - 1)) / correct_mass
    else:
        numerator = sum(
            math.comb(width, k)
            * correct_mass**k
            * (1.0 - correct_mass) ** (width - k)
            * group_weight(kind, k, width)
            * k
            * (width - k)
            for k in range(1, width)
        )
        coefficient = numerator / (width**2 * correct_mass * (1.0 - correct_mass))
    claimed_mean = [coefficient * value for value in gradient]
    error = max(abs(actual - expected) for actual, expected in zip(mean, claimed_mean))
    if error > 1e-12:
        raise AssertionError((probabilities, rewards, width, kind, mean, claimed_mean))
    return {
        "probabilities": probabilities,
        "rewards": rewards,
        "group_width": width,
        "variant": kind,
        "ordered_groups": len(probabilities) ** width,
        "correct_mass": correct_mass,
        "coefficient": coefficient,
        "enumerated_mean": mean,
        "claimed_mean": claimed_mean,
        "maximum_absolute_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = (
        ((0.2, 0.3, 0.5), (1, 1, 0)),
        ((0.04, 0.11, 0.85), (1, 1, 0)),
        ((0.4, 0.5, 0.1), (1, 1, 0)),
        ((0.1, 0.15, 0.25, 0.5), (1, 1, 0, 0)),
    )
    variants = ("drgrpo", "grpo_population_std", "grpo_sample_std_eps", "maxrl")
    checks = [
        check_group_expectation(probabilities, rewards, width, kind)
        for probabilities, rewards in cases
        for width in range(2, 7)
        for kind in variants
    ]
    report = {
        "description": "Independent exhaustive ordered-group mean-update enumeration",
        "check_count": len(checks),
        "passed": True,
        "tolerance": 1e-12,
        "maximum_absolute_error": max(row["maximum_absolute_error"] for row in checks),
        "checks": checks,
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
        print(
            f"Passed {len(checks)} checks; maximum absolute error "
            f"{report['maximum_absolute_error']:.6g}; wrote {args.output}"
        )
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
