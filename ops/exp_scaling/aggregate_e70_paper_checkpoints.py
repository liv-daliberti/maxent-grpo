#!/usr/bin/env python3
"""Aggregate the E70 five-domain surface onto the paper's fixed checkpoint grid.

The paper reports one common design: seeds 43--47, both arms, evaluated at
training passes 0, 3, 6, 9, and 12 on the ``multi_answer`` split. This script
reads the per-domain scaling curves, refuses any checkpoint that is not present
for every seed of both arms, and emits the long-form CSV plus the markdown
summary that the LaTeX tables are transcribed from.

A domain whose seeds have not all reached pass 12 yet reports its deepest
all-seed checkpoint and is marked ``provisional`` rather than averaged over the
seeds that happen to be available.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Display name -> (curve artifact, campaign stage). The first entry that exists
# wins, so a completed E71 384/128 cohort supersedes its E70 predecessor without
# an edit here; until then the E70 curve is used and the cohort is reported.
DOMAINS = {
    "Graph coloring": (
        ["gce71_scale384_05b_12pass_scaling_curve.json",
         "gce70_clean_stage_a_05b_12pass_scaling_curve.json"], "A"),
    "Countdown": (["cde70_clean_stage_a_05b_12pass_scaling_curve.json"], "A"),
    "Python factors": (["pye70_clean_stage_a_05b_12pass_scaling_curve.json"], "A"),
    "MathIR action menu": (["mie70_clean_stage_a_05b_12pass_scaling_curve.json"], "A"),
    "PantryPlan": (
        ["ppe71_scale384_05b_12pass_scaling_curve.json",
         "ppe70_clean_stage_b_05b_12pass_scaling_curve.json"], "B"),
}

ARMS = ("grpo", "verified_first_global_replay_canonical")
SEEDS = (43, 44, 45, 46, 47)
GRID = (0, 3, 6, 9, 12)
METRICS = ("greedy", "mean8", "pass8", "distinct8")
SPLIT = "multi_answer"


def load_curve(path: Path) -> tuple[dict, float]:
    """Index a curve by (arm, seed, training_pass) and return the pass unit.

    ``training_passes`` is null throughout these artifacts, so passes are
    derived from ``prompt_consumed``: evaluations land four times per pass, so
    one pass is four evaluation strides of prompt consumption.
    """
    rows = [r for r in json.loads(path.read_text()) if r["split"] == SPLIT]
    consumed = sorted({r["prompt_consumed"] for r in rows})
    prompts_per_pass = (consumed[1] - consumed[0]) * 4
    index = {}
    for row in rows:
        key = (row["arm"], row["seed"], row["prompt_consumed"] / prompts_per_pass)
        index[key] = row
    return index, prompts_per_pass


def deepest_common_pass(index: dict) -> int:
    """Largest grid checkpoint present for every seed of both arms."""
    best = 0
    for target in GRID:
        if all((arm, seed, float(target)) in index for arm in ARMS for seed in SEEDS):
            best = target
    return best


def resolve_curve(candidates: list[str], artifacts: Path):
    """Pick the curve that reports the deepest checkpoint, newest cohort first.

    A superseding cohort only takes over once it has actually caught up. A
    freshly launched replacement whose curve exists but has reached pass 0 must
    never silently pull a completed pass-12 row backwards, so selection is by
    depth first and cohort recency only as a tie-break.
    """
    resolved = []
    for order, name in enumerate(candidates):
        path = artifacts / name
        if not path.is_file():
            continue
        try:
            index, prompts_per_pass = load_curve(path)
        except (IndexError, ValueError, KeyError, json.JSONDecodeError):
            # A curve written mid-flight can be too sparse to infer a pass unit
            # from; skip it rather than crash the manuscript surface.
            continue
        resolved.append((deepest_common_pass(index), -order, name, index,
                         prompts_per_pass))
    if not resolved:
        raise SystemExit(f"no scaling curve found; tried {candidates}")
    depth, _, name, index, prompts_per_pass = max(resolved, key=lambda r: r[:2])
    return name, index, depth, prompts_per_pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ROOT / "var/artifacts")
    parser.add_argument("--out-csv", type=Path,
                        default=ROOT / "paper/results/e70_five_domain_fixed_checkpoints_live.csv")
    parser.add_argument("--out-md", type=Path,
                        default=ROOT / "paper/results/e70_five_domain_fixed_checkpoints_live.md")
    args = parser.parse_args()

    csv_rows = []
    endpoints = {}
    for domain, (candidates, stage) in DOMAINS.items():
        filename, index, terminal, prompts_per_pass = resolve_curve(
            candidates, args.artifacts
        )
        cohort = filename.split("_")[0]
        endpoints[domain] = {
            "stage": stage,
            "cohort": cohort,
            "curve": filename,
            "terminal_pass": terminal,
            "complete": terminal == 12,
            "prompts_per_pass": prompts_per_pass,
            "cells": {},
        }
        for arm in ARMS:
            for target in GRID:
                if target > terminal:
                    continue
                present = [index.get((arm, seed, float(target))) for seed in SEEDS]
                if any(row is None for row in present):
                    continue
                for metric in METRICS:
                    values = [row[metric] for row in present]
                    csv_rows.append({
                        "domain": domain,
                        "stage": stage,
                        "cohort": cohort,
                        "arm": arm,
                        "metric": metric,
                        "training_pass": target,
                        "optimizer_step": int(present[0]["step"]),
                        **{f"seed_{s}": v for s, v in zip(SEEDS, values)},
                        "mean": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "is_terminal": int(target == terminal),
                        "is_complete_12": int(terminal == 12),
                    })
                    if target == terminal:
                        endpoints[domain]["cells"].setdefault(arm, {})[metric] = statistics.mean(values)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = ["# E70 five-domain fixed-checkpoint surface (seeds 43-47)", ""]
    lines.append("Grid: passes 0, 3, 6, 9, 12 on the multi_answer split. A checkpoint is")
    lines.append("emitted only when all five seeds of both arms have reached it.")
    lines.append("")
    for domain, info in endpoints.items():
        flag = "complete" if info["complete"] else f"PROVISIONAL (deepest all-seed pass {info['terminal_pass']})"
        lines.append(f"## {domain} - Stage {info['stage']} - cohort {info['cohort']} - {flag}")
        lines.append("")
        lines.append("| arm | pass | pass@1 | mean@8 | pass@8 | distinct@8 | modes/success |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for arm in ARMS:
            for target in GRID:
                sel = [r for r in csv_rows
                       if r["domain"] == domain and r["arm"] == arm and r["training_pass"] == target]
                if not sel:
                    continue
                means = {r["metric"]: r["mean"] for r in sel}
                mps = means["distinct8"] / means["pass8"] if means["pass8"] else float("nan")
                label = "Dr.GRPO" if arm == "grpo" else "xGRPO"
                lines.append(
                    f"| {label} | {target} | {means['greedy']:.4f} | {means['mean8']:.4f} | "
                    f"{means['pass8']:.4f} | {means['distinct8']:.4f} | {mps:.2f} |"
                )
        lines.append("")
    args.out_md.write_text("\n".join(lines))

    for domain, info in endpoints.items():
        cells = info["cells"]
        row = " & ".join(
            f"{cells[arm][m]:.3f}".lstrip("0") for arm in ARMS for m in METRICS
        )
        print(f"{domain} [{info['cohort']}] (pass {info['terminal_pass']}"
              f"{'' if info['complete'] else ', PROVISIONAL'}): {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
