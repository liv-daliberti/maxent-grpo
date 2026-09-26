#!/usr/bin/env python3
"""Fail-closed approval gate for E21's conditional-token MATH smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


DEFAULT_PREFIX = "mte21_math_conditional_token_smoke_v10"
SEED = 9008
END_STEP = 64
TAIL = 16
GROUP_SIZE = 16
ARMS = {
    "grpo": {"alpha": 0.0},
    "maxent": {"alpha": 0.00010},
    "maxent_control": {"alpha_min": 0.000075, "alpha_max": 0.00015},
    "maxent_dual": {"alpha_min": 0.00005, "alpha_max": 0.00015},
}


def _step(row: dict) -> int:
    return int(row.get("trainer/global_step", row.get("misc/global_step", -1)))


def _finite(row: dict, key: str, *, step: int) -> float:
    if key not in row:
        raise RuntimeError(f"step {step} is missing {key}")
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"step {step} has nonfinite {key}={value}")
    return value


def _read_attempt(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discover(
    root: Path, arm: str, prefix: str
) -> tuple[Path, list[dict], list[dict]]:
    suffix = f"{prefix}_{arm}_s{SEED}"
    attempts: list[tuple[int, Path, list[dict]]] = []
    for run_dir in root.glob(f"*{suffix}"):
        if not run_dir.is_dir():
            continue
        for path in run_dir.glob("debug_*/train_metrics.jsonl"):
            all_rows = _read_attempt(path)
            reached = max((_step(row) for row in all_rows), default=-1)
            attempts.append((reached, path, all_rows))
    if not attempts:
        raise RuntimeError(f"no E21 metrics found for {suffix}")
    _, path, all_rows = max(attempts, key=lambda item: (item[0], str(item[1])))
    by_step = {
        _step(row): row
        for row in all_rows
        if _step(row) >= 1 and "train/pg_loss" in row
    }
    return path, [by_step[step] for step in sorted(by_step)], all_rows


def inspect_arm(
    root: Path, arm: str, prefix: str = DEFAULT_PREFIX
) -> dict:
    path, rows, all_rows = discover(root, arm, prefix)
    by_step = {_step(row): row for row in rows}
    if END_STEP not in by_step:
        reached = max(by_step, default=-1)
        raise RuntimeError(f"{arm} reached step {reached}; expected {END_STEP}")
    missing = sorted(set(range(1, END_STEP + 1)) - set(by_step))
    if missing:
        raise RuntimeError(f"{arm} lacks contiguous telemetry: {missing}")
    rows = [by_step[step] for step in range(1, END_STEP + 1)]
    common = (
        "train/pg_loss",
        "train/policy_grad_norm",
        "actor/rewards",
        "actor/response_tok_len",
        "actor/no_eos_count",
    )
    for row in rows:
        step = _step(row)
        for key in common:
            _finite(row, key, step=step)

    eval_values = [
        float(row["eval/math/accuracy"])
        for row in all_rows
        if "eval/math/accuracy" in row
        and math.isfinite(float(row["eval/math/accuracy"]))
    ]
    if len(eval_values) < 2:
        raise RuntimeError(f"{arm} lacks initialization and terminal MATH pass@1")

    if arm == "grpo":
        forbidden = [
            key
            for row in rows
            for key in row
            if key.startswith("train/maxent_")
        ]
        if forbidden:
            raise RuntimeError(f"grpo unexpectedly logged MaxEnt metrics: {forbidden[0]}")
    else:
        finite_treatment = (
            "train/maxent_alpha_used",
            "train/maxent_conditional_token_entropy",
            "train/maxent_entropy_surrogate",
            "train/maxent_entropy_loss",
            "train/entropy",
            "train/maxent_eos_excluded",
            "train/maxent_state_distribution_detached",
            "train/maxent_response_equal_weight",
        )
        for row in rows:
            step = _step(row)
            for key in finite_treatment:
                _finite(row, key, step=step)
            for key in (
                "train/maxent_eos_excluded",
                "train/maxent_state_distribution_detached",
                "train/maxent_response_equal_weight",
            ):
                if not math.isclose(float(row[key]), 1.0, abs_tol=1e-8):
                    raise RuntimeError(f"{arm} step {step} violates {key}")
            forbidden_legacy = [
                key
                for key in row
                if key.startswith("train/maxent_sequence_entropy")
                or key.startswith("train/maxent_prefix_ratio")
                or key.startswith("train/maxent_sampled_prefix_entropy")
            ]
            if forbidden_legacy:
                raise RuntimeError(
                    f"{arm} step {step} logged legacy sequence/prefix telemetry: "
                    f"{forbidden_legacy[0]}"
                )

        if arm == "maxent":
            for row in rows:
                alpha = _finite(row, "train/maxent_alpha_used", step=_step(row))
                if not math.isclose(alpha, 0.00010, rel_tol=1e-5, abs_tol=1e-9):
                    raise RuntimeError(f"fixed arm used alpha={alpha}")
                if any(
                    key.startswith("train/maxent_control_")
                    or key.startswith("train/maxent_dual_")
                    for key in row
                ):
                    raise RuntimeError("fixed arm unexpectedly logged a controller")
        else:
            low = float(ARMS[arm]["alpha_min"])
            high = float(ARMS[arm]["alpha_max"])
            prefix = "maxent_control" if arm == "maxent_control" else "maxent_dual"
            target_key = f"train/{prefix}_target_entropy"
            targets = []
            for row in rows[16:]:
                step = _step(row)
                alpha = _finite(row, "train/maxent_alpha_used", step=step)
                if alpha < low - 1e-9 or alpha > high + 1e-9:
                    raise RuntimeError(
                        f"{arm} step {step} alpha={alpha} outside [{low}, {high}]"
                    )
                targets.append(_finite(row, target_key, step=step))
            if not targets or min(targets) <= 0:
                raise RuntimeError(f"{arm} did not land a positive warmup target")

    tail = rows[-TAIL:]
    return {
        "arm": arm,
        "metrics_path": str(path.resolve()),
        "end_step": END_STEP,
        "tail_mean_reward": sum(float(row["actor/rewards"]) for row in tail)
        / TAIL,
        "tail_max_reward": max(float(row["actor/rewards"]) for row in tail),
        "tail_mean_length": sum(
            float(row["actor/response_tok_len"]) for row in tail
        )
        / TAIL,
        "tail_no_eos_total": sum(
            float(row["actor/no_eos_count"]) for row in tail
        ),
        "tail_no_eos_rate": sum(
            float(row["actor/no_eos_count"]) for row in tail
        )
        / (TAIL * GROUP_SIZE),
        "tail_max_no_eos": max(float(row["actor/no_eos_count"]) for row in tail),
        "initial_pass1": eval_values[0],
        "terminal_pass1": eval_values[-1],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-surface-hash", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summaries = {
        arm: inspect_arm(args.run_data_root, arm, args.prefix) for arm in ARMS
    }
    control = summaries["grpo"]
    control_length = float(control["tail_mean_length"])
    control_no_eos_total = float(control["tail_no_eos_total"])
    length_limit = max(1.25 * control_length, control_length + 32.0)
    # A maximum-over-16-batches comparison is dominated by one noisy batch.
    # Aggregate the same one-extra-row-per-batch allowance across the frozen
    # final window, while retaining the absolute per-batch runaway guard.
    no_eos_total_limit = min(
        float(TAIL * GROUP_SIZE), control_no_eos_total + float(TAIL)
    )
    for arm in ("maxent", "maxent_control", "maxent_dual"):
        row = summaries[arm]
        if float(row["tail_max_reward"]) <= 0:
            raise RuntimeError(f"{arm} has no positive reward in the final 16 steps")
        if float(row["tail_mean_length"]) > length_limit:
            raise RuntimeError(
                f"{arm} tail length {row['tail_mean_length']:.3f} > {length_limit:.3f}"
            )
        if float(row["tail_mean_length"]) >= 768:
            raise RuntimeError(f"{arm} tail length reached the absolute guardrail")
        if float(row["tail_no_eos_total"]) > no_eos_total_limit:
            raise RuntimeError(
                f"{arm} aggregate no-EOS {row['tail_no_eos_total']:.0f} > "
                f"{no_eos_total_limit:.0f}"
            )
        if float(row["tail_max_no_eos"]) > 12:
            raise RuntimeError(f"{arm} exceeded the absolute no-EOS guardrail")

    manifest_rows = [
        line.split("\t")
        for line in args.manifest.read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    if len(manifest_rows) != 4 or {row[0] for row in manifest_rows} != set(ARMS):
        raise RuntimeError("E21 smoke manifest is not exactly the frozen four arms")
    if {row[1] for row in manifest_rows} != {str(SEED)}:
        raise RuntimeError(f"E21 smoke manifest is not exactly seed {SEED}")
    payload = {
        "schema": "e21_math_conditional_token_smoke_approval_v1",
        "approved": True,
        "prefix": args.prefix,
        "seed": SEED,
        "source_hash": args.source_hash,
        "execution_surface_hash": args.execution_surface_hash,
        "protocol_sha256": _sha256(args.protocol),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "length_limit": length_limit,
        "no_eos_total_limit": no_eos_total_limit,
        "arms": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + args.out.name + ".", dir=args.out.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, args.out)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
