#!/usr/bin/env python3
"""Deterministic score-space toy for open-bank MaxEnt replay.

The toy deliberately excludes PPO and language-model parameter sharing. It
isolates the derivative applied to one prompt-local bank, which is the claim we
need before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Sequence


def softmax(scores: Sequence[float]) -> list[float]:
    offset = max(scores)
    weights = [math.exp(value - offset) for value in scores]
    total = sum(weights)
    return [value / total for value in weights]


def split_score_gradients(
    scores: Sequence[float],
    *,
    mass_weight: float,
    balance_weight: float,
) -> dict[str, list[float]]:
    """Exact derivatives of mass + reverse-KL balance with respect to scores."""

    count = len(scores)
    if count == 0:
        raise ValueError("a replay bank must contain at least one score")
    probabilities = softmax(scores)
    mass = [-1.0 / count for _ in scores]
    balance = (
        [probability - 1.0 / count for probability in probabilities]
        if count >= 2
        else [0.0]
    )
    weighted = [
        mass_weight * mass_value + balance_weight * balance_value
        for mass_value, balance_value in zip(mass, balance)
    ]
    return {
        "bank_probabilities": probabilities,
        "mass": mass,
        "balance": balance,
        "weighted": weighted,
    }


def retention_safe_score_gradients(
    scores: Sequence[float],
    *,
    mass_weight: float,
    balance_weight: float,
) -> dict[str, list[float]]:
    """Preserve rare-mode emphasis without a downward score-space derivative."""

    if not math.isfinite(mass_weight) or mass_weight <= 0.0:
        raise ValueError("retention-safe projection requires positive mass weight")
    raw = split_score_gradients(
        scores,
        mass_weight=mass_weight,
        balance_weight=balance_weight,
    )
    upward_pressure = [max(-value, 0.0) for value in raw["weighted"]]
    normalizer = sum(upward_pressure)
    if normalizer <= 0.0:
        raise RuntimeError("positive mass weight must leave upward score pressure")
    safe = [
        -mass_weight * pressure / normalizer
        for pressure in upward_pressure
    ]
    return {
        **raw,
        "retention_safe": safe,
    }


def retention_safe_balance_gradients(
    scores: Sequence[float],
    *,
    mass_weight: float,
    balance_weight: float,
) -> dict[str, list[float] | float]:
    """Use the largest balance weight that cannot push a bank score downward."""

    if not math.isfinite(mass_weight) or mass_weight <= 0.0:
        raise ValueError("retention-safe balance requires positive mass weight")
    if not math.isfinite(balance_weight) or balance_weight < 0.0:
        raise ValueError("retention-safe balance requires non-negative weight")
    raw = split_score_gradients(
        scores,
        mass_weight=mass_weight,
        balance_weight=balance_weight,
    )
    count = len(scores)
    dominance = count * max(raw["bank_probabilities"]) - 1.0
    safe_balance_weight = (
        balance_weight
        if dominance <= 0.0
        else min(balance_weight, mass_weight / dominance)
    )
    safe = split_score_gradients(
        scores,
        mass_weight=mass_weight,
        balance_weight=safe_balance_weight,
    )
    if any(value > 1e-12 for value in safe["weighted"]):
        raise RuntimeError("safe balance assigned downward score pressure")
    return {
        **raw,
        "requested_balance_weight": balance_weight,
        "safe_balance_weight": safe_balance_weight,
        "retention_safe": safe["weighted"],
    }


def descend_scores(
    scores: Sequence[float],
    *,
    steps: int,
    learning_rate: float,
    mass_weight: float,
    balance_weight: float,
) -> list[list[float]]:
    trajectory = [[float(value) for value in scores]]
    for _ in range(steps):
        gradients = split_score_gradients(
            trajectory[-1],
            mass_weight=mass_weight,
            balance_weight=balance_weight,
        )["weighted"]
        trajectory.append(
            [
                value - learning_rate * gradient
                for value, gradient in zip(trajectory[-1], gradients)
            ]
        )
    return trajectory


def build_report(*, steps: int = 12, learning_rate: float = 1.0) -> dict:
    initial_pair = [-0.2, -2.2]
    mass_only = descend_scores(
        initial_pair,
        steps=steps,
        learning_rate=learning_rate,
        mass_weight=0.1,
        balance_weight=0.0,
    )
    mass_plus_balance = descend_scores(
        initial_pair,
        steps=steps,
        learning_rate=learning_rate,
        mass_weight=0.1,
        balance_weight=0.1,
    )
    initial_gradients = split_score_gradients(
        initial_pair,
        mass_weight=0.1,
        balance_weight=0.1,
    )
    singleton_gradients = split_score_gradients(
        [-0.2],
        mass_weight=0.1,
        balance_weight=0.1,
    )
    four_mode_probabilities = [0.9, *([0.1 / 3.0] * 3)]
    four_mode_scores = [math.log(value) for value in four_mode_probabilities]
    four_mode_gradients = retention_safe_score_gradients(
        four_mode_scores,
        mass_weight=0.1,
        balance_weight=0.1,
    )
    four_mode_capped = retention_safe_balance_gradients(
        four_mode_scores,
        mass_weight=0.1,
        balance_weight=0.1,
    )

    def summary(trajectory: list[list[float]]) -> dict:
        start = trajectory[0]
        finish = trajectory[-1]
        return {
            "initial_scores": start,
            "final_scores": finish,
            "initial_bank_probabilities": softmax(start),
            "final_bank_probabilities": softmax(finish),
            "initial_score_gap": max(start) - min(start),
            "final_score_gap": max(finish) - min(finish),
        }

    return {
        "schema": "open_bank_maxent_score_toy_v2",
        "objective": {
            "mass": "-mean_i score_i",
            "balance": "KL(uniform_bank || softmax(bank_scores))",
            "combined": "0.1 * mass + 0.1 * balance",
            "safe_balance": "min(balance_weight, mass_weight / (n*q_max - 1))",
        },
        "interpretation": (
            "An unadmitted mode receives no derivative. On a singleton bank the "
            "balance derivative is exactly zero. Once a verified exemplar is "
            "admitted, reverse-KL balance gives the low-score mode the larger "
            "restorative derivative, while mass raises both modes."
        ),
        "settings": {"steps": steps, "score_learning_rate": learning_rate},
        "before_admission_singleton": {
            "scores": [-0.2],
            "gradients": singleton_gradients,
        },
        "first_step_after_admission": {
            "scores": initial_pair,
            "gradients": initial_gradients,
            "rare_to_common_update_ratio": (
                abs(initial_gradients["weighted"][1])
                / abs(initial_gradients["weighted"][0])
            ),
        },
        "four_mode_projection": {
            "probabilities": four_mode_probabilities,
            "scores": four_mode_scores,
            "gradients": four_mode_gradients,
            "raw_downward_modes": sum(value > 0.0 for value in four_mode_gradients["weighted"]),
            "safe_downward_modes": sum(value > 0.0 for value in four_mode_gradients["retention_safe"]),
        },
        "four_mode_balance_cap": {
            "probabilities": four_mode_probabilities,
            "scores": four_mode_scores,
            "gradients": four_mode_capped,
            "safe_downward_modes": sum(
                value > 1e-12 for value in four_mode_capped["retention_safe"]
            ),
        },
        "mass_only": summary(mass_only),
        "mass_plus_balance": summary(mass_plus_balance),
    }


def markdown(report: dict) -> str:
    pair = report["first_step_after_admission"]
    rows = []
    for name in ("mass_only", "mass_plus_balance"):
        result = report[name]
        rows.append(
            f"| {name} | {result['initial_score_gap']:.4f} | "
            f"{result['final_score_gap']:.4f} | "
            f"{result['final_bank_probabilities'][1]:.4f} |"
        )
    return "\n".join(
        [
            "| objective | initial gap | final gap | final rare-mode bank mass |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "First post-admission weighted score gradients: "
            f"{pair['gradients']['weighted']}; rare/common update ratio "
            f"{pair['rare_to_common_update_ratio']:.3f}.",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    if args.steps < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("steps must be non-negative and learning rate positive")
    report = build_report(steps=args.steps, learning_rate=args.learning_rate)
    if args.format == "markdown":
        print(markdown(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
