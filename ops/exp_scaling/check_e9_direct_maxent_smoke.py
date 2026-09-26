#!/usr/bin/env python3
"""Gate E9 direct-gradient MaxEnt on objective and collapse diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ARMS = ("maxent", "maxent_control", "maxent_dual")


def _step(row: dict) -> int:
    return int(row.get("trainer/global_step", row.get("misc/global_step", -1)))


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_metrics(root: Path, *, stamp: str, arm: str, seed: int) -> Path:
    suffix = f"{stamp}_{arm}_s{seed}"
    attempts: list[tuple[int, Path]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
            continue
        for path in run_dir.glob("debug_*/train_metrics.jsonl"):
            rows = _read_rows(path)
            if rows:
                attempts.append((max(_step(row) for row in rows), path))
    if not attempts:
        raise RuntimeError(f"no metrics found for {suffix}")
    return max(attempts, key=lambda item: (item[0], str(item[1])))[1]


def inspect_rows(
    rows: list[dict],
    *,
    arm: str,
    expected_step: int,
    num_samples: int = 16,
    trailing_steps: int = 32,
    entropy_guard_steps: int = 16,
    min_target_fraction: float = 0.5,
) -> dict[str, float | int | str]:
    training = [row for row in rows if _step(row) >= 0 and "train/pg_loss" in row]
    if not training:
        raise RuntimeError(f"{arm} has no training rows")
    last = max(training, key=_step)
    max_step = _step(last)
    if max_step < expected_step:
        raise RuntimeError(f"{arm} stopped at step {max_step}; expected {expected_step}")

    finite_keys = (
        "train/entropy",
        "train/pg_loss",
        "train/policy_grad_norm",
        "train/maxent_alpha_used",
        "train/maxent_sequence_entropy",
        "train/maxent_entropy_surrogate",
        "train/maxent_causal_score_term",
        "train/maxent_entropy_loss",
        "train/maxent_reward_estimator_scale",
        "actor/response_tok_len",
        "actor/rewards",
    )
    for key in finite_keys:
        if key not in last or not math.isfinite(float(last[key])):
            raise RuntimeError(f"{arm} missing finite terminal {key}")
    if float(last["train/maxent_alpha_used"]) <= 0:
        raise RuntimeError(f"{arm} did not apply a positive MaxEnt coefficient")
    if float(last["train/maxent_sequence_entropy"]) <= 0:
        raise RuntimeError(f"{arm} has non-positive sequence entropy")
    if float(last["train/maxent_entropy_loss"]) == 0:
        raise RuntimeError(f"{arm} has a zero direct entropy loss")
    expected_scale = float(num_samples - 1) / float(num_samples)
    if not math.isclose(
        float(last["train/maxent_reward_estimator_scale"]),
        expected_scale,
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise RuntimeError(f"{arm} has the wrong Dr.GRPO estimator scale")

    terminal_length = float(last["actor/response_tok_len"])
    if terminal_length <= 2.0:
        raise RuntimeError(
            f"{arm} terminal response length {terminal_length:.3g} is EOS collapse"
        )
    tail = [row for row in training if _step(row) > max_step - trailing_steps]
    tail_rewards = [float(row.get("actor/rewards", 0.0)) for row in tail]
    if not tail_rewards or max(tail_rewards) <= 0:
        raise RuntimeError(f"{arm} has no nonzero reward in its trailing window")
    tail_lengths = [float(row.get("actor/response_tok_len", 0.0)) for row in tail]
    if len(tail_lengths) >= 8 and all(value <= 2.0 for value in tail_lengths[-8:]):
        raise RuntimeError(f"{arm} converged to the two-token state")

    controller_key = None
    target_key = None
    if arm == "maxent_control":
        controller_key = "train/maxent_control_observations"
        target_key = "train/maxent_control_target_entropy"
    elif arm == "maxent_dual":
        controller_key = "train/maxent_dual_observations"
        target_key = "train/maxent_dual_target_entropy"
    if controller_key is not None:
        values = [
            float(row[controller_key]) for row in training if controller_key in row
        ]
        if not values or max(values) < 64:
            raise RuntimeError(f"{arm} did not cross controller warmup")
        target = float(last.get(target_key, math.nan))
        if not math.isfinite(target) or target <= 0:
            raise RuntimeError(f"{arm} has no finite positive entropy target")
        entropy_tail = training[-max(int(entropy_guard_steps), 1) :]
        tail_entropy = sum(
            float(row["train/maxent_sequence_entropy"]) for row in entropy_tail
        ) / float(len(entropy_tail))
        target_fraction = tail_entropy / target
        if target_fraction < float(min_target_fraction):
            raise RuntimeError(
                f"{arm} actuator guard failed: trailing entropy is "
                f"{target_fraction:.3f} of target; require "
                f"{float(min_target_fraction):.3f}"
            )

    return {
        "arm": arm,
        "max_step": max_step,
        "terminal_length": terminal_length,
        "max_tail_reward": max(tail_rewards),
        "alpha": float(last["train/maxent_alpha_used"]),
        "sequence_entropy": float(last["train/maxent_sequence_entropy"]),
        "entropy_loss": float(last["train/maxent_entropy_loss"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--stamp", default="gce9_direct_maxent_smoke_v1")
    parser.add_argument("--seed", type=int, default=9005)
    parser.add_argument("--expected-step", type=int, default=127)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--min-target-fraction", type=float, default=0.5)
    args = parser.parse_args()

    summaries = []
    for arm in ARMS:
        path = discover_metrics(
            args.run_data_root, stamp=args.stamp, arm=arm, seed=args.seed
        )
        summaries.append(
            inspect_rows(
                _read_rows(path),
                arm=arm,
                expected_step=args.expected_step,
                num_samples=args.num_samples,
                min_target_fraction=args.min_target_fraction,
            )
        )
    print("E9 direct-gradient MaxEnt smoke passed.")
    for row in summaries:
        print(
            f"  {row['arm']}: step={row['max_step']} "
            f"length={row['terminal_length']:.2f} "
            f"tail_reward={row['max_tail_reward']:.4f} "
            f"alpha={row['alpha']:.5f} Hseq={row['sequence_entropy']:.5f} "
            f"Lent={row['entropy_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
