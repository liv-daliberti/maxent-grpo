#!/usr/bin/env python3
"""Fail-closed gate for E11's standard sequence-MaxEnt smokes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SEED = 9005
T_MAX = 192.0
RAW_TARGET = 9.9680345401152
LITERAL_STAMP = "gce11_standard_maxent_literal_a0p05_v1"
COMPARATIVE_STAMP = "gce11_standard_maxent_smoke_v1"


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
        raise RuntimeError(f"no E11 metrics found for {suffix}")
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
    expected_step: int,
    adaptive: bool = False,
    endpoint_step: int | None = None,
) -> dict[str, float | str]:
    reached = _step(rows[-1]) if rows else -1
    if reached < expected_step:
        raise RuntimeError(f"{arm} stopped at step {reached}; expected {expected_step}")
    if endpoint_step is not None:
        rows = [row for row in rows if _step(row) <= endpoint_step]
        if not rows or _step(rows[-1]) != endpoint_step:
            raise RuntimeError(f"{arm} lacks frozen endpoint step {endpoint_step}")
    finite_keys = (
        "train/pg_loss",
        "train/policy_grad_norm",
        "train/maxent_alpha_used",
        "train/maxent_sequence_entropy",
        "train/maxent_sequence_entropy_per_tmax",
        "train/maxent_entropy_surrogate",
        "train/maxent_entropy_loss",
        "train/maxent_prefix_ratio_mean",
        "train/maxent_prefix_ratio_max",
        "train/maxent_prefix_ratio_clipfrac",
        "actor/response_tok_len",
        "actor/no_eos_count",
        "actor/rewards",
    )
    for row in rows:
        for key in finite_keys:
            if key not in row or not math.isfinite(float(row[key])):
                raise RuntimeError(f"{arm} has missing or nonfinite {key}")
        raw = float(row["train/maxent_sequence_entropy"])
        normalized = float(row["train/maxent_sequence_entropy_per_tmax"])
        if not math.isclose(raw / T_MAX, normalized, rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError(f"{arm} raw/per-T_max entropy telemetry disagrees")

    last = rows[-1]
    if arm == "maxent" and not math.isclose(
        float(last["train/maxent_alpha_used"]),
        0.05,
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError(f"{arm} did not use literal standard alpha=0.05")
    accuracy = float(last.get("eval/average/accuracy", math.nan))
    if not math.isfinite(accuracy):
        raise RuntimeError(f"{arm} lacks terminal evaluation accuracy")
    length = float(last["actor/response_tok_len"])
    no_eos = float(last["actor/no_eos_count"])
    if not 2 < length <= 64:
        raise RuntimeError(f"{arm} terminal length {length:.3f} violates guardrail")
    if no_eos > 2:
        raise RuntimeError(f"{arm} has {no_eos:.0f}/16 terminal no-EOS rollouts")
    if accuracy <= 0.05:
        raise RuntimeError(f"{arm} terminal evaluation accuracy is {accuracy:.4f}")
    tail_window = rows[-16:]
    if max(float(row["actor/rewards"]) for row in rows[-16:]) <= 0:
        raise RuntimeError(f"{arm} has no positive trailing reward")
    tail_entropy = sum(
        float(row["train/maxent_sequence_entropy"]) for row in tail_window
    ) / len(tail_window)
    target_fraction = tail_entropy / RAW_TARGET
    if adaptive:
        target_key = (
            "train/maxent_control_target_entropy"
            if arm == "maxent_control"
            else "train/maxent_dual_target_entropy"
        )
        if not math.isclose(
            float(last.get(target_key, math.nan)),
            RAW_TARGET,
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise RuntimeError(f"{arm} used the wrong raw entropy target")
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
    }


def inspect_literal(root: Path) -> dict[str, float | str]:
    return inspect_arm(
        discover_rows(root, LITERAL_STAMP, "maxent"),
        arm="maxent",
        expected_step=32,
        endpoint_step=32,
    )


def inspect_all(root: Path) -> list[dict[str, float | str]]:
    summaries = [inspect_literal(root)]
    for arm in ("maxent", "maxent_control", "maxent_dual"):
        summaries.append(
            inspect_arm(
                discover_rows(root, COMPARATIVE_STAMP, arm),
                arm=arm,
                expected_step=127,
                adaptive=arm != "maxent",
            )
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--stage", choices=("literal", "all"), default="all")
    args = parser.parse_args()

    summaries = (
        [inspect_literal(args.run_data_root)]
        if args.stage == "literal"
        else inspect_all(args.run_data_root)
    )
    print(f"E11 standard-MaxEnt {args.stage} gate passed.")
    for row in summaries:
        print(
            f"  {row['arm']}: tail_H={float(row['tail_entropy']):.5f} "
            f"target_fraction={float(row['target_fraction']):.3f} "
            f"alpha={float(row['alpha']):.5f} "
            f"length={float(row['length']):.2f} no_eos={float(row['no_eos']):.0f} "
            f"accuracy={float(row['accuracy']):.3f}"
        )


if __name__ == "__main__":
    main()
