#!/usr/bin/env python3
"""Gate E10 prefix-ratio direct-MaxEnt comparative and stress smokes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGET_ENTROPY = 0.0519168465631
SEED = 9005
COMPARATIVE_STAMP = "gce10_prefix_maxent_smoke_v1"
STRESS_STAMP = "gce10_prefix_maxent_stress_a0p50_v1"


def _step(row: dict) -> int:
    return int(row.get("trainer/global_step", row.get("misc/global_step", -1)))


def discover_rows(root: Path, stamp: str, arm: str) -> list[dict]:
    suffix = f"{stamp}_{arm}_s{SEED}"
    attempts: list[tuple[int, Path]] = []
    if root.exists():
        for run_dir in root.iterdir():
            if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
                continue
            for path in run_dir.glob("debug_*/train_metrics.jsonl"):
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if rows:
                    attempts.append((max(_step(row) for row in rows), path))
    if not attempts:
        raise RuntimeError(f"no E10 metrics found for {suffix}")
    path = max(attempts, key=lambda item: (item[0], str(item[1])))[1]
    by_step = {
        _step(row): row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if _step(row) >= 0 and "train/pg_loss" in row
    }
    return [by_step[step] for step in sorted(by_step)]


def inspect_arm(
    rows: list[dict],
    *,
    arm: str,
    expected_alpha: float | None = None,
) -> dict[str, float | str]:
    if not rows or _step(rows[-1]) < 127:
        reached = _step(rows[-1]) if rows else -1
        raise RuntimeError(f"{arm} stopped at step {reached}; expected 127")
    last = rows[-1]
    finite_keys = (
        "train/pg_loss",
        "train/policy_grad_norm",
        "train/maxent_alpha_used",
        "train/maxent_sequence_entropy",
        "train/maxent_sampled_prefix_entropy",
        "train/maxent_entropy_surrogate",
        "train/maxent_entropy_loss",
        "train/maxent_prefix_ratio_mean",
        "train/maxent_prefix_ratio_max",
        "train/maxent_prefix_ratio_clipfrac",
        "actor/response_tok_len",
        "actor/no_eos_count",
        "actor/rewards",
        "eval/average/accuracy",
    )
    for key in finite_keys:
        if key not in last or not math.isfinite(float(last[key])):
            raise RuntimeError(f"{arm} missing finite terminal {key}")
    if expected_alpha is not None and not math.isclose(
        float(last["train/maxent_alpha_used"]),
        expected_alpha,
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError(f"{arm} did not hold alpha={expected_alpha}")
    ratio_mean = float(last["train/maxent_prefix_ratio_mean"])
    ratio_max = float(last["train/maxent_prefix_ratio_max"])
    clipfrac = float(last["train/maxent_prefix_ratio_clipfrac"])
    if ratio_mean <= 0 or ratio_max <= 0 or not 0 <= clipfrac <= 1:
        raise RuntimeError(f"{arm} has invalid prefix-ratio telemetry")

    length = float(last["actor/response_tok_len"])
    no_eos = float(last["actor/no_eos_count"])
    accuracy = float(last["eval/average/accuracy"])
    if not 2 < length <= 64:
        raise RuntimeError(f"{arm} terminal length {length:.3f} violates guardrail")
    if no_eos > 2:
        raise RuntimeError(f"{arm} has {no_eos:.0f}/16 terminal no-EOS rollouts")
    if accuracy <= 0.05:
        raise RuntimeError(f"{arm} terminal evaluation accuracy is {accuracy:.4f}")
    max_tail_reward = max(float(row.get("actor/rewards", 0.0)) for row in rows[-32:])
    if max_tail_reward <= 0:
        raise RuntimeError(f"{arm} has no positive reward in its trailing window")

    tail_entropy = sum(
        float(row["train/maxent_sequence_entropy"]) for row in rows[-16:]
    ) / min(16, len(rows))
    target_fraction = tail_entropy / TARGET_ENTROPY
    if arm in {"maxent_control", "maxent_dual"}:
        target_key = (
            "train/maxent_control_target_entropy"
            if arm == "maxent_control"
            else "train/maxent_dual_target_entropy"
        )
        observed_target = float(last.get(target_key, math.nan))
        if not math.isclose(
            observed_target, TARGET_ENTROPY, rel_tol=1e-5, abs_tol=1e-7
        ):
            raise RuntimeError(f"{arm} used the wrong frozen target")
        if target_fraction < 0.5:
            raise RuntimeError(
                f"{arm} retained {target_fraction:.3f} of target; require 0.500"
            )

    return {
        "arm": arm,
        "tail_entropy": tail_entropy,
        "target_fraction": target_fraction,
        "alpha": float(last["train/maxent_alpha_used"]),
        "length": length,
        "no_eos": no_eos,
        "accuracy": accuracy,
        "clipfrac": clipfrac,
        "prefix_ratio_max": ratio_max,
        "max_tail_reward": max_tail_reward,
    }


def inspect_all(root: Path) -> list[dict[str, float | str]]:
    summaries = [
        inspect_arm(
            discover_rows(root, COMPARATIVE_STAMP, arm),
            arm=arm,
            expected_alpha=0.05 if arm == "maxent" else None,
        )
        for arm in ("maxent", "maxent_control", "maxent_dual")
    ]
    summaries.append(
        inspect_arm(
            discover_rows(root, STRESS_STAMP, "maxent"),
            arm="maxent_stress_a0p50",
            expected_alpha=0.5,
        )
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    args = parser.parse_args()

    summaries = inspect_all(args.run_data_root)
    print("E10 prefix-ratio MaxEnt smoke passed.")
    for row in summaries:
        print(
            f"  {row['arm']}: tail_H={float(row['tail_entropy']):.6f} "
            f"target_fraction={float(row['target_fraction']):.3f} "
            f"alpha={float(row['alpha']):.4f} length={float(row['length']):.2f} "
            f"no_eos={float(row['no_eos']):.0f} accuracy={float(row['accuracy']):.3f} "
            f"prefix_max={float(row['prefix_ratio_max']):.3f} "
            f"clipfrac={float(row['clipfrac']):.3f}"
        )


if __name__ == "__main__":
    main()
