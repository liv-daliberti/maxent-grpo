#!/usr/bin/env python3
"""Analyze E12 fixed standard-MaxEnt doses and select a safe useful dose."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SEED = 9005
T_MAX = 192.0
MIN_USEFUL_ENTROPY = 4.9840172700576
DOSES = {
    "a0p0005": 0.0005,
    "a0p0010": 0.0010,
    "a0p0015": 0.0015,
    "a0p0020": 0.0020,
}


def stamp(label: str) -> str:
    return f"gce12_standard_maxent_dose_{label}_v2"


def _step(row: dict) -> int:
    return int(row.get("trainer/global_step", row.get("misc/global_step", -1)))


def discover_rows(root: Path, label: str) -> list[dict]:
    suffix = f"{stamp(label)}_maxent_s{SEED}"
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
        raise RuntimeError(f"no E12 metrics found for {suffix}")
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


def inspect_dose(rows: list[dict], *, label: str, alpha: float) -> dict:
    reached = _step(rows[-1]) if rows else -1
    if reached < 128:
        raise RuntimeError(f"{label} stopped at step {reached}; expected step 128")
    # Freeze the comparison at the preregistered endpoint even if a launcher
    # emits a later bookkeeping row.  A missing step 128 is a hard failure.
    rows = [row for row in rows if _step(row) <= 128]
    if not rows or _step(rows[-1]) != 128:
        raise RuntimeError(f"{label} lacks the frozen step-128 endpoint")
    observed_steps = {_step(row) for row in rows}
    required_tail_steps = set(range(97, 129))
    missing_tail_steps = sorted(required_tail_steps - observed_steps)
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
        "actor/response_tok_len",
        "actor/no_eos_count",
        "actor/rewards",
    )
    for row in rows:
        observed_alpha = float(row.get("train/maxent_alpha_used", math.nan))
        if not math.isclose(observed_alpha, alpha, rel_tol=1e-5, abs_tol=1e-8):
            raise RuntimeError(
                f"{label} used alpha={observed_alpha:.8f} at step {_step(row)}; "
                f"expected {alpha:.8f}"
            )
        controller_keys = [
            key
            for key in row
            if key.startswith("train/maxent_control_")
            or key.startswith("train/maxent_dual_")
        ]
        if controller_keys:
            raise RuntimeError(
                f"{label} unexpectedly logged adaptive-controller telemetry: "
                + ", ".join(sorted(controller_keys))
            )
        for key in finite_keys:
            if key not in row or not math.isfinite(float(row[key])):
                raise RuntimeError(f"{label} has missing or nonfinite {key}")
        raw = float(row["train/maxent_sequence_entropy"])
        normalized = float(row["train/maxent_sequence_entropy_per_tmax"])
        if not math.isclose(raw / T_MAX, normalized, rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError(f"{label} raw/per-T_max entropy telemetry disagrees")

    last = rows[-1]
    accuracy = float(last.get("eval/average/accuracy", math.nan))
    coverage = float(
        last.get("eval/average/sampled_mode_coverage_at_8", math.nan)
    )
    pass_at_8 = float(
        last.get("eval/multi_answer/sampled_any_correct_at_8", math.nan)
    )
    for key, value in (
        ("accuracy", accuracy),
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
    tail_length = sum(float(row["actor/response_tok_len"]) for row in tail16) / len(
        tail16
    )
    max_tail_length = max(float(row["actor/response_tok_len"]) for row in tail16)
    max_tail_no_eos = max(float(row["actor/no_eos_count"]) for row in tail16)
    max_tail_reward = max(float(row["actor/rewards"]) for row in tail32)

    failures: list[str] = []
    if max_tail_reward <= 0:
        failures.append("no positive trailing reward")
    if tail_length > 32:
        failures.append(f"tail mean length {tail_length:.2f} > 32")
    if max_tail_length > 64:
        failures.append(f"tail max length {max_tail_length:.2f} > 64")
    if max_tail_no_eos > 2:
        failures.append(f"tail max no-EOS {max_tail_no_eos:.0f} > 2")
    if accuracy <= 0.05:
        failures.append(f"accuracy {accuracy:.4f} <= 0.05")
    safe = not failures
    effective = tail_entropy >= MIN_USEFUL_ENTROPY
    return {
        "label": label,
        "alpha": alpha,
        "tail_entropy": tail_entropy,
        "entropy_target_fraction": tail_entropy / (2 * MIN_USEFUL_ENTROPY),
        "tail_mean_length": tail_length,
        "tail_max_length": max_tail_length,
        "tail_max_no_eos": max_tail_no_eos,
        "terminal_accuracy": accuracy,
        "terminal_pass_at_8": pass_at_8,
        "terminal_coverage_at_8": coverage,
        "peak_entropy": max(
            float(row["train/maxent_sequence_entropy"]) for row in rows
        ),
        "peak_length": max(float(row["actor/response_tok_len"]) for row in rows),
        "safe": safe,
        "entropy_effective": effective,
        "viable": safe and effective,
        "failures": failures,
    }


def inspect_all(root: Path) -> tuple[list[dict], dict | None]:
    summaries = [
        inspect_dose(discover_rows(root, label), label=label, alpha=alpha)
        for label, alpha in DOSES.items()
    ]
    viable = [row for row in summaries if row["viable"]]
    preferred = max(viable, key=lambda row: float(row["alpha"])) if viable else None
    return summaries, preferred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summaries, preferred = inspect_all(args.run_data_root)
    payload = {"doses": summaries, "preferred": preferred}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in summaries:
        status = "VIABLE" if row["viable"] else "REJECT"
        reasons = "; ".join(row["failures"]) or "none"
        print(
            f"{row['label']} alpha={float(row['alpha']):.4g} {status}: "
            f"tail_H={float(row['tail_entropy']):.3f} "
            f"tail_len={float(row['tail_mean_length']):.2f} "
            f"max_len={float(row['tail_max_length']):.2f} "
            f"max_no_eos={float(row['tail_max_no_eos']):.0f} "
            f"acc={float(row['terminal_accuracy']):.3f} "
            f"coverage={float(row['terminal_coverage_at_8']):.3f}; {reasons}"
        )
    if preferred is None:
        raise SystemExit("E12 calibration failed: no safe entropy-effective dose")
    print(
        "E12 preferred dose: "
        f"alpha={float(preferred['alpha']):.6f} ({preferred['label']})"
    )


if __name__ == "__main__":
    main()
