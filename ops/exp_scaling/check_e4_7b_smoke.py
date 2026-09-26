#!/usr/bin/env python3
"""Check only operational invariants for the staged E4 7B smoke.

This intentionally does not inspect or compare evaluation outcomes. It verifies
that all three arms ran far enough, logged finite training entropy, the
feedback arm crossed its warmup with controller diagnostics, and every run
wrote a non-empty checkpoint before a human approves the full 7B campaign.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


ARMS = ("grpo", "xdr_tau0p05", "xdr_tau_control")
STEP_RE = re.compile(r"step_(\d+)$")


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _step(row: dict) -> int:
    return int(row.get("trainer/global_step", row.get("misc/global_step", -1)))


def load_longest_attempt(run_dir: Path) -> tuple[list[dict], Path]:
    attempts = []
    for path in sorted(run_dir.glob("debug_*/train_metrics.jsonl")):
        rows = _read_rows(path)
        if rows:
            attempts.append((max(_step(row) for row in rows), path, rows))
    if not attempts:
        raise RuntimeError(f"no training metrics under {run_dir}")
    _max_step, path, rows = max(attempts, key=lambda item: (item[0], str(item[1])))
    return rows, path


def discover_run(run_data_root: Path, *, stamp: str, arm: str, seed: int) -> Path:
    suffix = f"{stamp}_{arm}_s{seed}"
    matches = sorted(
        path
        for path in run_data_root.iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one run ending in {suffix!r}, found {len(matches)}"
        )
    return matches[0]


def _finite_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        if key not in row:
            continue
        value = float(row[key])
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite {key} at step {_step(row)}")
        values.append(value)
    return values


def _latest_nonempty_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    checkpoints = []
    for path in run_dir.glob("debug_*/checkpoints/step_*"):
        match = STEP_RE.match(path.name)
        if match and path.is_dir() and any(item.is_file() for item in path.rglob("*")):
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=None, key=lambda item: (item[0], str(item[1])))


def inspect_run(
    run_dir: Path,
    *,
    arm: str,
    expected_step: int,
    warmup_steps: int,
) -> dict[str, object]:
    rows, metrics_path = load_longest_attempt(run_dir)
    max_step = max(_step(row) for row in rows)
    if max_step < expected_step:
        raise RuntimeError(
            f"{arm} stopped at step {max_step}; expected at least {expected_step}"
        )

    entropy = _finite_values(rows, "train/entropy")
    if not entropy:
        raise RuntimeError(f"{arm} has no finite train/entropy observations")

    checkpoint = _latest_nonempty_checkpoint(run_dir)
    if checkpoint is None or checkpoint[0] < expected_step:
        found = None if checkpoint is None else checkpoint[0]
        raise RuntimeError(
            f"{arm} latest non-empty checkpoint is {found}; expected {expected_step}"
        )

    result: dict[str, object] = {
        "arm": arm,
        "max_step": max_step,
        "entropy_observations": len(entropy),
        "checkpoint_step": checkpoint[0],
        "metrics_path": str(metrics_path),
        "checkpoint_path": str(checkpoint[1]),
    }

    if arm == "xdr_tau_control":
        required = (
            "train/xdr_tau_used",
            "train/xdr_tau_control_entropy_ema",
            "train/xdr_tau_control_next_tau",
            "train/xdr_tau_control_target_entropy",
            "train/xdr_tau_control_observations",
        )
        diagnostics = {key: _finite_values(rows, key) for key in required}
        missing = [key for key, values in diagnostics.items() if not values]
        if missing:
            raise RuntimeError(
                "feedback arm is missing controller diagnostics: " + ", ".join(missing)
            )
        observations = diagnostics["train/xdr_tau_control_observations"]
        if max(observations) < warmup_steps:
            raise RuntimeError(
                f"feedback controller reached {max(observations):.0f} observations; "
                f"expected at least {warmup_steps}"
            )
        result["controller_observations"] = int(max(observations))

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-data-root", type=Path, default=Path("var/data"))
    parser.add_argument("--stamp", default="gce4_taucontrol_7b_smoke")
    parser.add_argument("--seed", type=int, default=9001)
    # Five prompt-pool passes at the smoke's global batch geometry produce 160
    # optimizer steps; keep the operational check aligned with the campaign
    # ceiling rather than requiring the former eight-pass/256-step run.
    parser.add_argument("--expected-step", type=int, default=160)
    parser.add_argument("--warmup-steps", type=int, default=64)
    args = parser.parse_args()

    summaries = []
    for arm in ARMS:
        run_dir = discover_run(
            args.run_data_root,
            stamp=args.stamp,
            arm=arm,
            seed=args.seed,
        )
        summaries.append(
            inspect_run(
                run_dir,
                arm=arm,
                expected_step=args.expected_step,
                warmup_steps=args.warmup_steps,
            )
        )

    print("E4 7B smoke operational checks passed (evaluation outcomes uninspected).")
    for summary in summaries:
        controller = summary.get("controller_observations")
        controller_text = (
            "" if controller is None else f" controller_observations={controller}"
        )
        print(
            f"  {summary['arm']}: step={summary['max_step']} "
            f"checkpoint={summary['checkpoint_step']} "
            f"entropy_logs={summary['entropy_observations']}{controller_text}"
        )


if __name__ == "__main__":
    main()
