#!/usr/bin/env python3
"""Aggregate the E73 Falcon3-1B cross-family replication from run telemetry.

This exists because the first version of these numbers was extracted ad hoc and
reported the wrong quantity. `sampled_mode_coverage_at_8` is distinct verified
modes divided by the domain's total mode count; `sampled_distinct_correct_at_8`
is the count itself. The manuscript's `distinct@8` is the count, and the two
differ by more than a factor of four on these domains. Reading the coverage key
and labelling it `distinct@8` produced a table that violated the paper's own
Lemma 2.2, because coverage is a fraction of a large denominator and can sit
below `pass@8` while the count it derives from cannot.

So the key is named once, here, and the ordering the lemma guarantees is checked
against every emitted cell rather than trusted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

MODEL_TAG = "falcon3_1b_instruct"
TERMINAL_STEP = 4608

# The count, not the coverage fraction. This distinction is the whole reason
# this script exists; do not "simplify" it back to mode coverage.
METRIC_KEYS = {
    "pass_at_1": "eval/multi_answer/accuracy",
    "mean_at_8": "eval/multi_answer/sampled_mean_at_8",
    "pass_at_8": "eval/multi_answer/sampled_any_correct_at_8",
    "distinct_at_8": "eval/multi_answer/sampled_distinct_correct_at_8",
    "mode_coverage_at_8": "eval/multi_answer/sampled_mode_coverage_at_8",
}

ARMS = {
    "drgrpo": ("grpo_compute_matched", "grpo"),
    "xgrpo": (
        "verified_first_global_replay_canonical",
        "verified_first_global_replay_canonical",
    ),
}

# Cohorts relaunched after the originals OOM-looped supersede the originals
# wherever they exist --- the same supersede-don't-pool rule the manuscript
# applies elsewhere.
DOMAINS = (
    ("graph_coloring", "gce73_falcon3_1b_12pass", None),
    ("countdown", "cde73_falcon3_1b_12pass", None),
    ("python_factors", "pye73_falcon3_1b_12pass", "pye73_falcon3_1b_12pass_r1"),
    ("mathir", "mie73_falcon3_1b_12pass", "mie73_falcon3_1b_12pass_r1"),
    ("pantry_plan", "ppe73_falcon3_1b_12pass", None),
)
SEEDS = (43, 44, 45, 46, 47)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    """Every evaluated step of one run, keyed by optimizer step."""
    by_step: dict[int, dict[str, float]] = {}
    for path in glob.glob(str(run_dir / "debug_job*" / "train_metrics.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if METRIC_KEYS["distinct_at_8"] not in record:
                    continue
                step = int(record.get("misc/global_step", -1))
                by_step[step] = {
                    name: float(record[key])
                    for name, key in METRIC_KEYS.items()
                    if key in record
                }
    return by_step


def cell(root: Path, prefix: str, arm: str, seed: int) -> dict[int, dict[str, float]]:
    variant, stamp_arm = ARMS[arm]
    run_dir = (
        root
        / "var"
        / "data"
        / f"xdr_{MODEL_TAG}_{variant}_{prefix}_{stamp_arm}_s{seed}"
    )
    return run_metrics(run_dir) if run_dir.is_dir() else {}


def common_endpoint(cells: dict[tuple[str, int], dict[int, dict[str, float]]]) -> int:
    """Deepest step every arm and seed of a domain actually evaluated.

    The manuscript reports arms only where they can be compared at the same
    depth, so a domain's endpoint is the minimum over its cells, not the
    maximum over the ones that happen to be furthest along.
    """
    if not cells or any(not steps for steps in cells.values()):
        return 0
    return min(max(steps) for steps in cells.values())


def summarize(root: Path) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    violations: list[str] = []

    for domain, original, replacement in DOMAINS:
        prefix = original
        if replacement is not None:
            probe = {
                (arm, seed): cell(root, replacement, arm, seed)
                for arm in ARMS
                for seed in SEEDS
            }
            if any(probe.values()):
                prefix = replacement

        cells = {
            (arm, seed): cell(root, prefix, arm, seed)
            for arm in ARMS
            for seed in SEEDS
        }
        endpoint = common_endpoint(cells)
        rows = [
            {"arm": arm, "seed": seed, "step": endpoint, **cells[(arm, seed)][endpoint]}
            for (arm, seed) in sorted(cells)
            if endpoint and endpoint in cells[(arm, seed)]
        ]
        if len(rows) != len(ARMS) * len(SEEDS):
            domains[domain] = {
                "cohort": "r1" if prefix != original else "original",
                "complete": False,
                "terminal_cells": len(rows),
                "common_endpoint": endpoint,
            }
            continue

        # Lemma 2.2 orders the three quantities. A cell that breaks it means the
        # wrong key was read, so it is a hard error rather than a note.
        for row in rows:
            if not row["mean_at_8"] <= row["pass_at_8"] <= row["distinct_at_8"] + 1e-9:
                violations.append(
                    f"{domain} {row['arm']} s{row['seed']}: "
                    f"mean {row['mean_at_8']:.4f} pass {row['pass_at_8']:.4f} "
                    f"distinct {row['distinct_at_8']:.4f}"
                )

        means = {
            arm: {
                name: statistics.fmean(
                    row[name] for row in rows if row["arm"] == arm
                )
                for name in METRIC_KEYS
            }
            for arm in ARMS
        }
        paired = {
            name: sum(
                1
                for seed in SEEDS
                if cells[("xgrpo", seed)][endpoint][name]
                > cells[("drgrpo", seed)][endpoint][name]
            )
            for name in ("pass_at_8", "distinct_at_8")
        }
        domains[domain] = {
            "cohort": "r1" if prefix != original else "original",
            "complete": endpoint >= TERMINAL_STEP,
            "common_endpoint": endpoint,
            "means": means,
            "xgrpo_wins_of_5": paired,
            "rows": rows,
        }

    complete_entries = [entry for entry in domains.values() if entry.get("complete")]
    all_terminal = (
        len(complete_entries) == len(DOMAINS)
        and all(
            entry.get("common_endpoint") == TERMINAL_STEP
            and len(entry.get("rows", ())) == len(ARMS) * len(SEEDS)
            for entry in complete_entries
        )
        and not violations
    )
    directional = {
        "distinct_positive_domains": sum(
            entry["means"]["xgrpo"]["distinct_at_8"]
            > entry["means"]["drgrpo"]["distinct_at_8"]
            for entry in complete_entries
        ),
        "pass8_nonnegative_domains": sum(
            entry["means"]["xgrpo"]["pass_at_8"]
            >= entry["means"]["drgrpo"]["pass_at_8"]
            for entry in complete_entries
        ),
        "distinct_paired_wins": sum(
            entry["xgrpo_wins_of_5"]["distinct_at_8"]
            for entry in complete_entries
        ),
        "pass8_paired_wins": sum(
            entry["xgrpo_wins_of_5"]["pass_at_8"]
            for entry in complete_entries
        ),
    }

    return {
        "schema": "e73_cross_family_summary_v2",
        "model": "Falcon3-1B-Instruct",
        "prompt_epochs": 12,
        "terminal_step": TERMINAL_STEP,
        "all_50_cells_terminal_and_valid": all_terminal,
        "registered_directional_outcomes": directional,
        "reporting_rule": (
            "deepest checkpoint reached by all five seeds of both arms"
        ),
        "metric_keys": METRIC_KEYS,
        "lemma_violations": violations,
        "domains": domains,
    }


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "var" / "artifacts" / "e73_falcon_cross_family_summary_v2.json",
    )
    args = parser.parse_args()

    payload = summarize(root)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, args.output)

    for domain, entry in payload["domains"].items():
        if not entry.get("means"):
            print(
                f"  {domain:16} incomplete: {entry['terminal_cells']}/10 cells "
                f"at step {entry['common_endpoint']}"
            )
            continue
        drgrpo, xgrpo = entry["means"]["drgrpo"], entry["means"]["xgrpo"]
        print(
            f"  {domain:16} step {entry['common_endpoint']:5}  "
            f"p@1 {drgrpo['pass_at_1']:.3f}/{xgrpo['pass_at_1']:.3f}  "
            f"pass@8 {drgrpo['pass_at_8']:.3f}/{xgrpo['pass_at_8']:.3f}  "
            f"distinct@8 {drgrpo['distinct_at_8']:.3f}/{xgrpo['distinct_at_8']:.3f}  "
            f"(coverage {drgrpo['mode_coverage_at_8']:.3f}/"
            f"{xgrpo['mode_coverage_at_8']:.3f})"
        )
    if payload["lemma_violations"]:
        for problem in payload["lemma_violations"]:
            print(f"  LEMMA VIOLATION: {problem}")
        return 1
    if not payload["all_50_cells_terminal_and_valid"]:
        print("[e73] terminal audit failed: not all 50 cells are valid at step 4608")
        return 1
    print(f"[e73] wrote {args.output}; metric ordering holds in every cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
