#!/usr/bin/env python3
"""Fail-closed gate for E21 free-form conditional-token MATH MaxEnt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARMS = ("grpo", "maxent", "maxent_control", "maxent_dual")
SEED = 9007
EXPECTED_STEP = 64
WARMUP_STEPS = 16
DEFAULT_STAMP = "me21_math_token_policy_smoke_v1"


class GateError(RuntimeError):
    """A prospective E21 smoke condition was not met."""


def _step(row: dict[str, Any]) -> int:
    return int(row.get("trainer/global_step", row.get("misc/global_step", -1)))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GateError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise GateError(f"nondictionary metric at {path}:{line_number}")
            rows.append(value)
    return rows


def _finite(row: dict[str, Any], key: str, *, arm: str, step: int) -> float:
    if key not in row:
        raise GateError(f"{arm} step {step} is missing {key}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as error:
        raise GateError(f"{arm} step {step} has nonnumeric {key}") from error
    if not math.isfinite(value):
        raise GateError(f"{arm} step {step} has nonfinite {key}={value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_metrics(root: Path, *, stamp: str, arm: str) -> Path:
    suffix = f"{stamp}_{arm}_s{SEED}"
    attempts: list[tuple[int, Path]] = []
    if not root.is_dir():
        raise GateError(f"run-data root does not exist: {root}")
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.endswith(suffix):
            continue
        for path in run_dir.glob("debug_*/train_metrics.jsonl"):
            rows = _read_rows(path)
            if rows:
                attempts.append((max(_step(row) for row in rows), path))
    if not attempts:
        raise GateError(f"no E21 metrics found for {suffix}")
    return max(attempts, key=lambda item: (item[0], str(item[1])))[1]


def _training_rows(path: Path, *, arm: str) -> list[dict[str, Any]]:
    # Evaluation at an already-logged step can append a second row. Retain the
    # last row per step because it carries the same update metrics plus eval.
    by_step: dict[int, dict[str, Any]] = {}
    for row in _read_rows(path):
        step = _step(row)
        if step >= 1 and "train/pg_loss" in row:
            by_step[step] = row
    if not by_step:
        raise GateError(f"{arm} has no optimizer-update metrics in {path}")
    if max(by_step) < EXPECTED_STEP:
        raise GateError(
            f"{arm} stopped at step {max(by_step)}; expected {EXPECTED_STEP}"
        )
    rows = [by_step[step] for step in sorted(by_step) if step <= EXPECTED_STEP]
    if _step(rows[-1]) != EXPECTED_STEP:
        raise GateError(f"{arm} lacks frozen endpoint step {EXPECTED_STEP}")
    required_tail = set(range(EXPECTED_STEP - 15, EXPECTED_STEP + 1))
    missing = sorted(required_tail - {_step(row) for row in rows})
    if missing:
        raise GateError(f"{arm} lacks contiguous final-16 telemetry: {missing}")
    return rows


def _check_numerical_markers(row: dict[str, Any], *, arm: str, step: int) -> None:
    for key, raw in row.items():
        if not key.endswith(("_nan", "_inf")):
            continue
        value = _finite(row, key, arm=arm, step=step)
        if value != 0:
            raise GateError(f"{arm} step {step} reports numerical failure {key}={value}")


def inspect_arm(path: Path, *, arm: str) -> dict[str, Any]:
    rows = _training_rows(path, arm=arm)
    common = (
        "train/pg_loss",
        "train/policy_grad_norm",
        "actor/response_tok_len",
        "actor/no_eos_count",
        "actor/rewards",
        "misc/prompt_consumed",
    )
    maxent = (
        "train/maxent_alpha_used",
        "train/maxent_conditional_token_entropy",
        "train/maxent_entropy_surrogate",
        "train/maxent_entropy_loss",
        "train/maxent_state_distribution_detached",
        "train/maxent_eos_excluded",
        "train/maxent_response_equal_weight",
        "train/maxent_prefix_ratio_mean",
        "train/maxent_prefix_ratio_max",
        "train/maxent_prefix_ratio_clipfrac",
    )
    for row in rows:
        step = _step(row)
        _check_numerical_markers(row, arm=arm, step=step)
        for key in common:
            _finite(row, key, arm=arm, step=step)
        if arm == "grpo":
            forbidden = [key for key in row if key.startswith("train/maxent_")]
            if forbidden:
                raise GateError(
                    f"C0 step {step} contains MaxEnt treatment telemetry: {forbidden}"
                )
            continue
        for key in maxent:
            _finite(row, key, arm=arm, step=step)
        if any(
            key in row
            for key in (
                "train/maxent_sequence_entropy",
                "train/maxent_sequence_entropy_per_tmax",
            )
        ):
            raise GateError(f"{arm} step {step} used trajectory-entropy telemetry")
        for key in (
            "train/maxent_state_distribution_detached",
            "train/maxent_eos_excluded",
            "train/maxent_response_equal_weight",
        ):
            if float(row[key]) != 1.0:
                raise GateError(f"{arm} step {step} violated {key}")
        if not (
            math.isclose(
                float(row["train/maxent_prefix_ratio_mean"]),
                1.0,
                rel_tol=0,
                abs_tol=1e-7,
            )
            and math.isclose(
                float(row["train/maxent_prefix_ratio_max"]),
                1.0,
                rel_tol=0,
                abs_tol=1e-7,
            )
            and math.isclose(
                float(row["train/maxent_prefix_ratio_clipfrac"]),
                0.0,
                rel_tol=0,
                abs_tol=1e-7,
            )
        ):
            raise GateError(f"{arm} step {step} did not detach state visitation")
        alpha = float(row["train/maxent_alpha_used"])
        if not 0 < alpha <= 0.00015 + 1e-9:
            raise GateError(f"{arm} step {step} used out-of-contract alpha={alpha}")
        if arm == "maxent" and not math.isclose(
            alpha, 0.0001, rel_tol=1e-5, abs_tol=1e-9
        ):
            raise GateError(f"fixed arm step {step} used alpha={alpha}, not 0.0001")

    last = rows[-1]
    for key in (
        "eval/math/accuracy",
        "eval/math/sampled_any_correct_at_8",
        "eval/math/sampled_mean_at_8",
    ):
        _finite(last, key, arm=arm, step=EXPECTED_STEP)
    eval_count = _finite(last, "eval/math/eval_count", arm=arm, step=EXPECTED_STEP)
    if int(eval_count) != 500:
        raise GateError(f"{arm} terminal MATH eval_count={eval_count}; expected 500")

    if arm == "maxent_control":
        prefix = "train/maxent_control"
    elif arm == "maxent_dual":
        prefix = "train/maxent_dual"
    else:
        prefix = ""
    target = None
    observations = None
    if prefix:
        target = _finite(
            last, f"{prefix}_target_entropy", arm=arm, step=EXPECTED_STEP
        )
        observations = _finite(
            last, f"{prefix}_observations", arm=arm, step=EXPECTED_STEP
        )
        if target <= 0 or observations < EXPECTED_STEP:
            raise GateError(
                f"{arm} did not land and exercise its calibrated entropy target"
            )
        warmup_entropy = math.fsum(
            _finite(
                row,
                "train/maxent_conditional_token_entropy",
                arm=arm,
                step=_step(row),
            )
            for row in rows
            if 1 <= _step(row) <= WARMUP_STEPS
        ) / WARMUP_STEPS
        expected_target = 0.8 * warmup_entropy
        if not math.isclose(target, expected_target, rel_tol=1e-5, abs_tol=1e-6):
            raise GateError(
                f"{arm} target={target} does not equal 0.8 * warmup conditional entropy "
                f"({expected_target})"
            )
        next_alpha = _finite(
            last, f"{prefix}_next_alpha", arm=arm, step=EXPECTED_STEP
        )
        lower = 0.000075 if arm == "maxent_control" else 0.00005
        if not lower - 1e-9 <= next_alpha <= 0.00015 + 1e-9:
            raise GateError(f"{arm} proposed out-of-contract alpha={next_alpha}")
        if arm == "maxent_dual":
            optimizer_steps = _finite(
                last,
                "train/maxent_dual_optimizer_steps",
                arm=arm,
                step=EXPECTED_STEP,
            )
            if optimizer_steps < EXPECTED_STEP - WARMUP_STEPS:
                raise GateError(
                    f"dual optimizer ran {optimizer_steps} steps; expected at least "
                    f"{EXPECTED_STEP - WARMUP_STEPS}"
                )

    tail = rows[-16:]
    return {
        "arm": arm,
        "metrics_path": str(path.resolve()),
        "metrics_sha256": _sha256(path),
        "endpoint_step": EXPECTED_STEP,
        "tail_mean_reward": math.fsum(float(r["actor/rewards"]) for r in tail)
        / len(tail),
        "tail_max_reward": max(float(r["actor/rewards"]) for r in tail),
        "tail_mean_length": math.fsum(
            float(r["actor/response_tok_len"]) for r in tail
        )
        / len(tail),
        "tail_max_no_eos": max(float(r["actor/no_eos_count"]) for r in tail),
        "terminal_pass1": float(last["eval/math/accuracy"]),
        "terminal_pass8": float(last["eval/math/sampled_any_correct_at_8"]),
        "target_entropy": target,
        "controller_observations": observations,
    }


def apply_comparative_gate(summaries: list[dict[str, Any]]) -> None:
    by_arm = {row["arm"]: row for row in summaries}
    c0 = by_arm["grpo"]
    for arm in ARMS[1:]:
        row = by_arm[arm]
        if row["tail_max_reward"] <= 0:
            raise GateError(f"{arm} has no positive-reward batch in the final 16")
        relative_length_limit = max(
            1.25 * c0["tail_mean_length"], c0["tail_mean_length"] + 32.0
        )
        if row["tail_mean_length"] > relative_length_limit:
            raise GateError(
                f"{arm} tail mean length {row['tail_mean_length']:.2f} exceeds "
                f"matched C0 limit {relative_length_limit:.2f}"
            )
        no_eos_limit = min(16.0, c0["tail_max_no_eos"] + 1.0)
        if row["tail_max_no_eos"] > no_eos_limit:
            raise GateError(
                f"{arm} tail no-EOS {row['tail_max_no_eos']:.0f} exceeds "
                f"matched C0 limit {no_eos_limit:.0f}"
            )
        if row["tail_mean_length"] >= 768:
            raise GateError(f"{arm} tail mean length reached the 768-token guard")
        if row["tail_max_no_eos"] > 12:
            raise GateError(f"{arm} tail no-EOS exceeds 12/16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--stamp", default=DEFAULT_STAMP)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-surface-hash", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.unlink(missing_ok=True)
    if not args.protocol.is_file():
        raise GateError(f"protocol does not exist: {args.protocol}")
    summaries = [
        inspect_arm(
            discover_metrics(args.run_data_root, stamp=args.stamp, arm=arm), arm=arm
        )
        for arm in ARMS
    ]
    apply_comparative_gate(summaries)
    payload = {
        "schema": "e21_math_token_policy_smoke_approval_v1",
        "approved": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "E21",
        "label": "free-form conditional-token MaxEnt",
        "stamp": args.stamp,
        "seed": SEED,
        "source_hash": args.source_hash,
        "execution_surface_hash": args.execution_surface_hash,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "arms": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("E21 free-form conditional-token MATH smoke gate passed.")
    for row in summaries:
        print(
            f"  {row['arm']}: len={row['tail_mean_length']:.2f} "
            f"max_no_eos={row['tail_max_no_eos']:.0f} "
            f"reward={row['tail_mean_reward']:.4f} "
            f"pass1={row['terminal_pass1']:.4f} "
            f"pass8={row['terminal_pass8']:.4f}"
        )
    print(f"approval={args.out}")


if __name__ == "__main__":
    main()
