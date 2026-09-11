#!/usr/bin/env python3
"""Gate E9b fixed-dose and adaptive direct-MaxEnt actuator smokes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SEED = 9005
CALIBRATION_STAMP = "gce9b_maxent_calibration_v1"
DOSES = {
    "gce9b_maxent_dose_a0p05_v1": 0.05,
    "gce9b_maxent_dose_a0p10_v1": 0.10,
    "gce9b_maxent_dose_a0p20_v1": 0.20,
    "gce9b_maxent_dose_a0p50_v1": 0.50,
}
ADAPTIVE_STAMP = "gce9b_maxent_adaptive_v1"


def _step(row: dict) -> int:
    return int(row.get("trainer/global_step", row.get("misc/global_step", -1)))


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_rows(root: Path, stamp: str, arm: str) -> list[dict]:
    suffix = f"{stamp}_{arm}_s{SEED}"
    attempts: list[tuple[int, Path]] = []
    if root.exists():
        for run_dir in root.iterdir():
            if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
                continue
            for path in run_dir.glob("debug_*/train_metrics.jsonl"):
                rows = _read_rows(path)
                if rows:
                    attempts.append((max(_step(row) for row in rows), path))
    if not attempts:
        raise RuntimeError(f"no metrics found for {suffix}")
    path = max(attempts, key=lambda item: (item[0], str(item[1])))[1]
    by_step = {
        _step(row): row
        for row in _read_rows(path)
        if _step(row) >= 0 and "train/pg_loss" in row
    }
    return [by_step[step] for step in sorted(by_step)]


def calibrated_target(root: Path) -> float:
    rows = discover_rows(root, CALIBRATION_STAMP, "maxent")
    if len(rows) < 64 or _step(rows[-1]) < 63:
        raise RuntimeError("frozen-policy calibration has fewer than 64 updates")
    observations = rows[:64]
    for row in observations:
        lr = float(row.get("misc/lr", math.nan))
        entropy = float(row.get("train/maxent_sequence_entropy", math.nan))
        if lr != 0 or not math.isfinite(entropy) or entropy <= 0:
            raise RuntimeError("invalid frozen-policy calibration telemetry")
    return 0.8 * sum(
        float(row["train/maxent_sequence_entropy"]) for row in observations
    ) / len(observations)


def inspect_run(
    rows: list[dict],
    *,
    label: str,
    target: float,
    expected_alpha: float | None = None,
    controller: str | None = None,
) -> dict[str, float | str]:
    if not rows or _step(rows[-1]) < 127:
        reached = _step(rows[-1]) if rows else -1
        raise RuntimeError(f"{label} stopped at step {reached}; expected 127")
    last = rows[-1]
    required = (
        "train/pg_loss",
        "train/policy_grad_norm",
        "train/maxent_alpha_used",
        "train/maxent_sequence_entropy",
        "train/maxent_entropy_surrogate",
        "train/maxent_entropy_loss",
        "actor/response_tok_len",
        "actor/rewards",
    )
    for key in required:
        if key not in last or not math.isfinite(float(last[key])):
            raise RuntimeError(f"{label} missing finite terminal {key}")
    alphas = [float(row["train/maxent_alpha_used"]) for row in rows]
    if expected_alpha is not None and any(
        not math.isclose(alpha, expected_alpha, rel_tol=1e-5, abs_tol=1e-6)
        for alpha in alphas
    ):
        raise RuntimeError(f"{label} did not hold alpha={expected_alpha}")
    if float(last["actor/response_tok_len"]) <= 2:
        raise RuntimeError(f"{label} ended in EOS collapse")
    tail32 = rows[-32:]
    max_tail_reward = max(float(row.get("actor/rewards", 0.0)) for row in tail32)
    if max_tail_reward <= 0:
        raise RuntimeError(f"{label} has no positive trailing reward")
    tail_entropy = sum(
        float(row["train/maxent_sequence_entropy"]) for row in rows[-16:]
    ) / min(16, len(rows))
    target_fraction = tail_entropy / target

    if controller == "proportional":
        target_key = "train/maxent_control_target_entropy"
        if "train/maxent_control_relative_deficit" not in last:
            raise RuntimeError("proportional arm lacks relative-deficit telemetry")
    elif controller == "dual":
        target_key = "train/maxent_dual_target_entropy"
        if float(last.get("train/maxent_dual_optimizer_steps", 0)) < 127:
            raise RuntimeError("dual arm did not control from its first observation")
    else:
        target_key = None
    if target_key is not None:
        observed_target = float(last.get(target_key, math.nan))
        if not math.isclose(observed_target, target, rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError(f"{label} used target {observed_target}, expected {target}")

    return {
        "label": label,
        "terminal_alpha": alphas[-1],
        "max_alpha": max(alphas),
        "tail_entropy": tail_entropy,
        "target_fraction": target_fraction,
        "terminal_length": float(last["actor/response_tok_len"]),
        "max_tail_reward": max_tail_reward,
    }


def stage_a(root: Path) -> tuple[float, list[dict[str, float | str]]]:
    target = calibrated_target(root)
    summaries = [
        inspect_run(
            discover_rows(root, stamp, "maxent"),
            label=f"fixed alpha={alpha:.2f}",
            target=target,
            expected_alpha=alpha,
        )
        for stamp, alpha in DOSES.items()
    ]
    if not any(float(row["target_fraction"]) >= 0.5 for row in summaries):
        raise RuntimeError("no fixed-alpha dose retained 50% of calibrated target")
    return target, summaries


def stage_b(root: Path) -> tuple[float, list[dict[str, float | str]]]:
    target, _ = stage_a(root)
    summaries = [
        inspect_run(
            discover_rows(root, ADAPTIVE_STAMP, arm),
            label=arm,
            target=target,
            controller=controller,
        )
        for arm, controller in (
            ("maxent_control", "proportional"),
            ("maxent_dual", "dual"),
        )
    ]
    failed = [row for row in summaries if float(row["target_fraction"]) < 0.5]
    if failed:
        fractions = ", ".join(
            f"{row['label']}={float(row['target_fraction']):.3f}" for row in failed
        )
        raise RuntimeError(f"E9b adaptive retention gate failed: {fractions}")
    return target, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage-a", "stage-b"))
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()

    target, summaries = (
        stage_a(args.run_data_root)
        if args.stage == "stage-a"
        else stage_b(args.run_data_root)
    )
    if args.target_only:
        print(f"{target:.12g}")
        return
    print(f"E9b {args.stage} passed; calibrated target={target:.6f}")
    for row in summaries:
        print(
            f"  {row['label']}: tail_H={float(row['tail_entropy']):.6f} "
            f"target_fraction={float(row['target_fraction']):.3f} "
            f"alpha={float(row['terminal_alpha']):.4f} "
            f"max_alpha={float(row['max_alpha']):.4f} "
            f"length={float(row['terminal_length']):.2f} "
            f"tail_reward={float(row['max_tail_reward']):.3f}"
        )


if __name__ == "__main__":
    main()
