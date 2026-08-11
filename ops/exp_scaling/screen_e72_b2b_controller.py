#!/usr/bin/env python3
"""Calibrate B2b's entropy controller on a few short runs before spending a cohort.

Two full cohorts were discarded because a configuration that parsed, validated,
and looked reasonable could not hold its target. The offline validator cannot
catch that --- only telemetry can --- so the cheap version of that telemetry is
bought here: two domains, one seed, two passes, a handful of controller
settings.

What is being calibrated is the *instrument*, not the treatment. The target
stays the treatment's own measured entropy and no arm coefficient is tuned;
what varies is the coefficient floor and the speed at which the dual controller
moves, which decide whether the target is reachable at all.

The diagnosis these settings respond to: the dual coefficient's floor is a
strictly positive entropy bonus, and Dr.GRPO's reward on these domains is zero
for the first few hundred updates. With nothing opposing it, even a small
constant bonus drives the policy to near-uniform output, after which no verified
outcome is ever sampled, the advantage is identically zero, and the entropy term
is the only force left. The state absorbs. A floor near zero lets the controller
genuinely stop pushing while the policy is still above target, which is the
regime every one of these domains starts in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as launcher  # noqa: E402

# Each setting names the floor, the starting coefficient, and the adaptation
# rate. The floor is the variable under test; "published" reproduces what the
# discarded cohorts ran so the screen contains its own negative control.
SETTINGS: dict[str, dict[str, str]] = {
    "published": {
        "OAT_ZERO_MAXENT_ALPHA": "0.05",
        "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA": "0.005",
        "OAT_ZERO_MAXENT_DUAL_ALPHA_LR": "0.003",
    },
    "floor6_fast": {
        "OAT_ZERO_MAXENT_ALPHA": "0.000001",
        "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA": "0.000001",
        "OAT_ZERO_MAXENT_DUAL_ALPHA_LR": "0.01",
        "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA": "0.5",
    },
    "floor6_slow": {
        "OAT_ZERO_MAXENT_ALPHA": "0.000001",
        "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA": "0.000001",
        "OAT_ZERO_MAXENT_DUAL_ALPHA_LR": "0.003",
        "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA": "0.5",
    },
    "floor4_fast": {
        "OAT_ZERO_MAXENT_ALPHA": "0.0001",
        "OAT_ZERO_MAXENT_DUAL_MIN_ALPHA": "0.0001",
        "OAT_ZERO_MAXENT_DUAL_ALPHA_LR": "0.01",
        "OAT_ZERO_MAXENT_DUAL_MAX_ALPHA": "0.5",
    },
}

# Countdown fails fastest and Graph coloring carries the largest target, so
# between them they bracket the screen. One seed each: this is a go/no-go on the
# instrument, not an estimate of anything.
SCREEN_DOMAINS = ("countdown", "graph_coloring")
SCREEN_SEED = 43
SCREEN_PASSES = 2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "var" / "artifacts" / "e72_frontier_source_runs.json",
    )
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--partition", default="mltheory")
    parser.add_argument("--account", default="mltheory")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--time-limit", default="0-06:00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    chosen = [name for name in args.settings.split(",") if name]
    unknown = [name for name in chosen if name not in SETTINGS]
    if unknown:
        raise SystemExit(f"unknown settings: {unknown}")

    submitted: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        if (
            run["arm"] != launcher.REFERENCE_ARM
            or run["domain"] not in SCREEN_DOMAINS
            or int(run["seed"]) != SCREEN_SEED
        ):
            continue
        for name in chosen:
            stamp = f"{launcher.run_stamp(run, 'b2b')}_cal_{name}"
            target = (
                root
                / "var"
                / "data"
                / f"xdr_{launcher.MODEL_TAG}_"
                f"{launcher.ARM_SPECS['b2b']['variant']}_{stamp}"
            )
            env = launcher.build_export_vars(root, run, target, "b2b")
            env["RUN_STAMP"] = stamp
            env["SAVE_PATH"] = str(target)
            # Short horizon: the question is whether entropy tracks or runs
            # away, which the first two passes answer.
            env["OAT_ZERO_NUM_PROMPT_EPOCH"] = str(SCREEN_PASSES)
            env["OAT_ZERO_MAX_PROMPT_EPOCHS"] = str(SCREEN_PASSES)
            # No checkpointing: nothing here is resumed or reported.
            env["OAT_ZERO_SAVE_STEPS"] = "0"
            env["OAT_ZERO_WATCHDOG_REQUEUE"] = "0"
            env["OAT_ZERO_AUTO_RESUME"] = "0"
            env.update(SETTINGS[name])

            export_pairs = ",".join(f"{k}={v}" for k, v in env.items())
            sbatch = [
                "sbatch",
                "--parsable",
                f"--job-name=e72b2bcal-{name}-{run['domain'][:6]}",
                f"--export=ALL,{export_pairs}",
                f"--nodelist={run['source_node']}",
                f"--gres={args.gres}",
                f"--cpus-per-task={args.cpus}",
                f"--mem={args.memory}",
                f"--time={args.time_limit}",
                f"--partition={args.partition}",
                f"--account={args.account}",
                str(root / "ops" / "run_experiment.sh"),
            ]
            if args.dry_run:
                print(" ".join(sbatch[:6]) + " ...")
                submitted.append({"setting": name, "domain": run["domain"]})
                continue
            result = subprocess.run(sbatch, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  submit failed [{name}/{run['domain']}]: {result.stderr[:200]}")
                continue
            submitted.append(
                {
                    "setting": name,
                    "domain": run["domain"],
                    "job_id": result.stdout.strip(),
                    "run_dir": str(target),
                }
            )

    ledger = root / "var" / "artifacts" / "e72_b2b_controller_screen.json"
    ledger.write_text(
        json.dumps(
            {
                "schema": "e72_b2b_controller_screen_v1",
                "settings": SETTINGS,
                "passes": SCREEN_PASSES,
                "seed": SCREEN_SEED,
                "cells": submitted,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"[b2b-screen] submitted {len(submitted)} cells; ledger {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
