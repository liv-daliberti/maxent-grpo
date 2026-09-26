#!/usr/bin/env python3
"""Analyze E13's projected expected-length-dual engineering arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SEED = 9005
T_MAX = 192.0
ALPHA = 0.002
TARGET_LENGTH = 16.0
LAMBDA_INITIAL = 0.0
LAMBDA_MAX = 0.02
EMA_DECAY = 0.9
MIN_USEFUL_ENTROPY = 4.9840172700576
ARMS = {
    "eta5em5": 0.00005,
    "eta2em4": 0.00020,
}


def stamp(label: str) -> str:
    return f"gce13_length_constrained_maxent_{label}_v1"


def _step(row: dict) -> int:
    return int(row.get("trainer/global_step", row.get("misc/global_step", -1)))


def discover_rows(root: Path, label: str) -> list[dict]:
    suffix = f"{stamp(label)}_maxent_length_dual_s{SEED}"
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
        raise RuntimeError(f"no E13 metrics found for {suffix}")
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


def _require_close(
    observed: float,
    expected: float,
    *,
    label: str,
    step: int,
    quantity: str,
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-8,
) -> None:
    if not math.isclose(observed, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise RuntimeError(
            f"{label} has {quantity}={observed:.10g} at step {step}; "
            f"expected {expected:.10g}"
        )


def inspect_arm(rows: list[dict], *, label: str, dual_lr: float) -> dict:
    reached = _step(rows[-1]) if rows else -1
    if reached < 128:
        raise RuntimeError(f"{label} stopped at step {reached}; expected step 128")
    rows = [row for row in rows if _step(row) <= 128]
    if not rows or _step(rows[-1]) != 128:
        raise RuntimeError(f"{label} lacks the frozen step-128 endpoint")

    observed_steps = {_step(row) for row in rows}
    missing_tail_steps = sorted(set(range(97, 129)) - observed_steps)
    if missing_tail_steps:
        raise RuntimeError(
            f"{label} lacks contiguous final-32 telemetry; missing steps "
            + ", ".join(str(step) for step in missing_tail_steps)
        )

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
        "train/maxent_expected_length",
        "train/maxent_sampled_prefix_length",
        "train/maxent_length_surrogate",
        "train/maxent_length_loss",
        "train/maxent_length_target",
        "train/maxent_length_lambda_used",
        "train/maxent_length_lambda_next",
        "train/maxent_length_lambda_max",
        "train/maxent_length_ema",
        "train/maxent_length_ema_decay",
        "train/maxent_length_relative_violation",
        "train/maxent_length_dual_lr",
        "actor/response_tok_len",
        "actor/no_eos_count",
        "actor/rewards",
    )

    previous_ema = TARGET_LENGTH
    previous_next: float | None = None
    for row in rows:
        step = _step(row)
        controller_keys = [
            key
            for key in row
            if key.startswith("train/maxent_control_")
            or key.startswith("train/maxent_dual_")
        ]
        if controller_keys:
            raise RuntimeError(
                f"{label} unexpectedly logged entropy-alpha controller telemetry: "
                + ", ".join(sorted(controller_keys))
            )
        for key in finite_keys:
            if key not in row or not math.isfinite(float(row[key])):
                raise RuntimeError(f"{label} has missing or nonfinite {key}")

        _require_close(
            float(row["train/maxent_alpha_used"]),
            ALPHA,
            label=label,
            step=step,
            quantity="alpha",
        )
        _require_close(
            float(row["train/maxent_length_target"]),
            TARGET_LENGTH,
            label=label,
            step=step,
            quantity="length target",
        )
        _require_close(
            float(row["train/maxent_length_lambda_max"]),
            LAMBDA_MAX,
            label=label,
            step=step,
            quantity="lambda maximum",
        )
        _require_close(
            float(row["train/maxent_length_ema_decay"]),
            EMA_DECAY,
            label=label,
            step=step,
            quantity="EMA decay",
        )
        _require_close(
            float(row["train/maxent_length_dual_lr"]),
            dual_lr,
            label=label,
            step=step,
            quantity="dual step",
        )

        raw_entropy = float(row["train/maxent_sequence_entropy"])
        normalized_entropy = float(
            row["train/maxent_sequence_entropy_per_tmax"]
        )
        _require_close(
            normalized_entropy,
            raw_entropy / T_MAX,
            label=label,
            step=step,
            quantity="per-T_max entropy",
            abs_tol=1e-7,
        )
        _require_close(
            float(row["train/maxent_sampled_prefix_length"]),
            float(row["actor/response_tok_len"]),
            label=label,
            step=step,
            quantity="sampled-prefix length",
            abs_tol=1e-6,
        )

        lambda_used = float(row["train/maxent_length_lambda_used"])
        lambda_next = float(row["train/maxent_length_lambda_next"])
        if not 0.0 <= lambda_used <= LAMBDA_MAX:
            raise RuntimeError(f"{label} has out-of-range lambda_used at step {step}")
        if not 0.0 <= lambda_next <= LAMBDA_MAX:
            raise RuntimeError(f"{label} has out-of-range lambda_next at step {step}")
        if previous_next is None:
            _require_close(
                lambda_used,
                LAMBDA_INITIAL,
                label=label,
                step=step,
                quantity="initial lambda",
            )
        else:
            _require_close(
                lambda_used,
                previous_next,
                label=label,
                step=step,
                quantity="lambda continuity",
            )

        expected_length = float(row["train/maxent_expected_length"])
        expected_ema = EMA_DECAY * previous_ema + (1.0 - EMA_DECAY) * expected_length
        observed_ema = float(row["train/maxent_length_ema"])
        _require_close(
            observed_ema,
            expected_ema,
            label=label,
            step=step,
            quantity="length EMA",
        )
        expected_violation = (observed_ema - TARGET_LENGTH) / TARGET_LENGTH
        observed_violation = float(
            row["train/maxent_length_relative_violation"]
        )
        _require_close(
            observed_violation,
            expected_violation,
            label=label,
            step=step,
            quantity="relative length violation",
        )
        expected_next = min(
            LAMBDA_MAX,
            max(0.0, lambda_used + dual_lr * expected_violation),
        )
        _require_close(
            lambda_next,
            expected_next,
            label=label,
            step=step,
            quantity="projected lambda update",
        )
        previous_ema = observed_ema
        previous_next = lambda_next

    last = rows[-1]
    accuracy = float(last.get("eval/average/accuracy", math.nan))
    coverage = float(
        last.get("eval/average/sampled_mode_coverage_at_8", math.nan)
    )
    pass_at_8 = float(
        last.get("eval/multi_answer/sampled_any_correct_at_8", math.nan)
    )
    for key, value in (
        ("pass_at_1", accuracy),
        ("coverage_at_8", coverage),
        ("pass_at_8", pass_at_8),
    ):
        if not math.isfinite(value):
            raise RuntimeError(f"{label} lacks finite terminal {key}")

    tail16 = rows[-16:]
    tail32 = rows[-32:]
    tail_entropy = sum(
        float(row["train/maxent_sequence_entropy"]) for row in tail16
    ) / len(tail16)
    tail_actor_length = sum(
        float(row["actor/response_tok_len"]) for row in tail16
    ) / len(tail16)
    tail_expected_length = sum(
        float(row["train/maxent_expected_length"]) for row in tail16
    ) / len(tail16)
    max_tail_length = max(float(row["actor/response_tok_len"]) for row in tail16)
    max_tail_no_eos = max(float(row["actor/no_eos_count"]) for row in tail16)
    max_tail_reward = max(float(row["actor/rewards"]) for row in tail32)
    pinned_final_eight = all(
        math.isclose(
            float(row["train/maxent_length_lambda_next"]),
            LAMBDA_MAX,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        for row in rows[-8:]
    )

    failures: list[str] = []
    if max_tail_reward <= 0:
        failures.append("no positive trailing reward")
    if tail_actor_length > 18:
        failures.append(f"tail actor length {tail_actor_length:.2f} > 18")
    if tail_expected_length > 18:
        failures.append(f"tail expected length {tail_expected_length:.2f} > 18")
    if max_tail_length > 64:
        failures.append(f"tail max length {max_tail_length:.2f} > 64")
    if max_tail_no_eos > 2:
        failures.append(f"tail max no-EOS {max_tail_no_eos:.0f} > 2")
    if accuracy <= 0.05:
        failures.append(f"pass@1 {accuracy:.4f} <= 0.05")
    if pinned_final_eight:
        failures.append("lambda pinned at maximum for final eight updates")

    safe = not failures
    effective = tail_entropy >= MIN_USEFUL_ENTROPY
    return {
        "label": label,
        "dual_lr": dual_lr,
        "alpha": ALPHA,
        "target_length": TARGET_LENGTH,
        "tail_entropy": tail_entropy,
        "entropy_target_fraction": tail_entropy / (2 * MIN_USEFUL_ENTROPY),
        "tail_actor_length": tail_actor_length,
        "tail_expected_length": tail_expected_length,
        "tail_max_length": max_tail_length,
        "tail_max_no_eos": max_tail_no_eos,
        "tail_lambda": sum(
            float(row["train/maxent_length_lambda_next"]) for row in tail16
        )
        / len(tail16),
        "terminal_pass_at_1": accuracy,
        "terminal_pass_at_8": pass_at_8,
        "terminal_coverage_at_8": coverage,
        "peak_entropy": max(
            float(row["train/maxent_sequence_entropy"]) for row in rows
        ),
        "peak_length": max(float(row["actor/response_tok_len"]) for row in rows),
        "pinned_final_eight": pinned_final_eight,
        "constraint_safe": safe,
        "entropy_effective": effective,
        "viable": safe and effective,
        "failures": failures,
    }


def inspect_all(root: Path) -> tuple[list[dict], dict | None]:
    summaries = [
        inspect_arm(discover_rows(root, label), label=label, dual_lr=dual_lr)
        for label, dual_lr in ARMS.items()
    ]
    viable = [row for row in summaries if row["viable"]]
    if not viable:
        return summaries, None
    maximum_entropy = max(float(row["tail_entropy"]) for row in viable)
    tied = [
        row
        for row in viable
        if float(row["tail_entropy"]) >= 0.95 * maximum_entropy
    ]
    return summaries, min(tied, key=lambda row: float(row["dual_lr"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summaries, preferred = inspect_all(args.run_data_root)
    payload = {"arms": summaries, "preferred": preferred}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in summaries:
        status = "VIABLE" if row["viable"] else "REJECT"
        reasons = "; ".join(row["failures"]) or "none"
        print(
            f"{row['label']} eta={float(row['dual_lr']):.5g} {status}: "
            f"tail_H={float(row['tail_entropy']):.3f} "
            f"actor_len={float(row['tail_actor_length']):.2f} "
            f"expected_len={float(row['tail_expected_length']):.2f} "
            f"lambda={float(row['tail_lambda']):.5f} "
            f"pass@1={float(row['terminal_pass_at_1']):.3f} "
            f"pass@8={float(row['terminal_pass_at_8']):.3f} "
            f"coverage@8={float(row['terminal_coverage_at_8']):.3f}; {reasons}"
        )
    if preferred is None:
        raise SystemExit("E13 gate failed: no constraint-safe entropy-effective arm")
    print(
        "E13 preferred arm: "
        f"{preferred['label']} eta={float(preferred['dual_lr']):.6f}"
    )


if __name__ == "__main__":
    main()
