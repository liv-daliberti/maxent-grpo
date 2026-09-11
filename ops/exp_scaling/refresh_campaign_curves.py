#!/usr/bin/env python3
"""Refresh every curve consumed by the live MaxEnt compute-divergence figures.

Keeping this as one entry point prevents a figure refresh from updating only
the newest treatment extensions while leaving matched controls stale. Every
parse is constrained to the campaign's ten-pass horizon so an
obsolete over-budget attempt cannot outrank a later valid replacement.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
PARSER = ROOT / "ops/exp_scaling/parse_scaling_curve.py"
ARTIFACTS = ROOT / "var/artifacts"
RUN_DATA = ROOT / "var/data"

# (stamp prefix, prompt-pool rows, samples per prompt)
CELLS = (
    ("cde32_freeform_05b_ema_10ep_v4_preemptsafe", 384, 16),
    ("gce32_freeform_05b_ema_10ep_v5", 192, 16),
    ("cde33_freeform_3b_ema_10ep_v3_a100", 384, 16),
    ("gce33_freeform_3b_ema_10ep_v3_a100", 192, 16),
    ("cde16_canonical_maxent_05b_v2", 384, 16),
    ("gce16_canonical_maxent_05b_v2", 192, 16),
    ("cde17_canonical_maxent_3b_v5", 384, 16),
    ("gce17_canonical_maxent_3b_v5", 192, 16),
    ("cde18_canonical_drgrpo_3b_v1", 384, 16),
    ("gce18_canonical_drgrpo_3b_v1", 192, 16),
    ("cde19_canonical_drgrpo_05b_v1", 384, 16),
    ("gce19_canonical_drgrpo_05b_v1", 192, 16),
    ("cde23_canonical_maxent_7b_v6_4xa100_evalsync_fix", 384, 16),
    ("gce24_canonical_maxent_7b_v4_4xa100_evalsync_fix", 192, 16),
    ("cde22_freeform_conditional_dual_05b_v2", 384, 16),
    ("gce22_freeform_conditional_dual_05b_v2", 192, 16),
    ("cde27_freeform_conditional_dual_05b_v1", 384, 16),
    ("gce27_freeform_conditional_dual_05b_v1", 192, 16),
    ("cde25_freeform_conditional_dual_3b_v2", 384, 16),
    ("gce25_freeform_conditional_dual_3b_v2", 192, 16),
    ("cde28_freeform_drgrpo_3b_v1", 384, 16),
    ("gce28_freeform_drgrpo_3b_v1", 192, 16),
    ("cde29_freeform_7b_4gpu_v5_buffer_restore", 384, 16),
    ("gce29_freeform_7b_4gpu_v5_buffer_restore", 192, 16),
)

# Resume-ordering contamination discovered on 2026-07-22. Keep the valid
# predecessor boundary evaluation and reject every downstream point from the
# abandoned branch. Repair-v2 trajectories are published separately once
# clean replay data exist; they are never cosmetically spliced here.
MAX_CLEAN_STEP_BY_STAMP = {
    "cde25_freeform_conditional_dual_3b_v2": {43: 576, 44: 576, 45: 288},
    "gce25_freeform_conditional_dual_3b_v2": {43: 576, 44: 480, 45: 576},
    "cde28_freeform_drgrpo_3b_v1": {43: 96, 44: 96, 45: 864},
    "gce28_freeform_drgrpo_3b_v1": {43: 144, 44: 48, 45: 48},
}

# Repeated evaluations produced outside the historical training processes.
# These are merged only onto the matching terminal multi-answer row; every
# underlying trace path and evaluation seed remains attached as provenance.
REPEATED_EVAL_BACKFILLS = (
    (
        "cde22_freeform_conditional_dual_05b_v2",
        "cde30_freeform_05b_fixed_k8x4_v1",
        "grpo",
    ),
    (
        "cde27_freeform_conditional_dual_05b_v1",
        "cde30_freeform_05b_fixed_k8x4_v1",
        "maxent_dual",
    ),
    (
        "gce22_freeform_conditional_dual_05b_v2",
        "gce30_freeform_05b_fixed_k8x4_v1",
        "grpo",
    ),
    (
        "gce27_freeform_conditional_dual_05b_v1",
        "gce30_freeform_05b_fixed_k8x4_v1",
        "maxent_dual",
    ),
)

BACKFILL_METRICS = {
    "pass8": "any_correct_at_k",
    "mean8": "mean_at_k",
    "coverage8": "mode_coverage_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}


def write_evaluation_coverage_manifest() -> Path:
    """Record which plotted rows have full metrics and repeated uncertainty."""

    manifest = {
        "schema_version": 1,
        "evaluation_contract": {
            "pass_at_1": "deterministic greedy",
            "sampled_metrics": list(BACKFILL_METRICS),
            "sample_count": 8,
            "fixed_eval_seeds": [1001, 1002, 1003, 1004],
            "repeated_draw_statistics": ["mean", "sd", "se", "min", "max"],
            "smoothing": False,
        },
        "cells": {},
    }
    for stamp, _prompt_pool_size, _num_samples in CELLS:
        curve_path = ARTIFACTS / f"{stamp}_scaling_curve.json"
        rows = json.loads(curve_path.read_text(encoding="utf-8"))
        plotted = [row for row in rows if row.get("split") == "multi_answer"]
        repeated = [
            row
            for row in plotted
            if all(len(row.get(f"{metric}_draws") or []) == 4 for metric in BACKFILL_METRICS)
        ]
        complete = [
            row
            for row in plotted
            if row.get("greedy") is not None
            and all(row.get(metric) is not None for metric in BACKFILL_METRICS)
        ]
        manifest["cells"][stamp] = {
            "curve_path": str(curve_path.resolve()),
            "multi_answer_rows": len(plotted),
            "all_five_metrics_rows": len(complete),
            "fixed_k8x4_rows": len(repeated),
            "legacy_single_k8_rows": len(plotted) - len(repeated),
            "fixed_k8x4_points": [
                {
                    "arm": row["arm"],
                    "seed": row["seed"],
                    "step": row["step"],
                    "training_passes": row.get("training_passes"),
                    "trace_count": row.get("repeated_eval_trace_count"),
                    "source": row.get("repeated_eval_source", "inline"),
                }
                for row in repeated
            ],
        }
    path = ARTIFACTS / "compute_divergence_eval_coverage.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def merge_repeated_terminal_evaluation(
    curve_path: Path,
    summary_paths: list[Path],
    target_arm: str,
    greedy_summary_path: Path | None = None,
) -> int:
    """Replace matching terminal headlines with real repeated-draw summaries."""

    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    provenance: dict[int, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(summary_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed_match = re.search(r"_e(\d+)_coverage_summary\.json$", path.name)
        if seed_match is None:
            raise ValueError(f"cannot recover evaluation seed from {path}")
        eval_seed = int(seed_match.group(1))
        for checkpoint in payload["checkpoints"]:
            alias_match = re.fullmatch(r"(.+)_s(\d+)", checkpoint["alias"])
            if alias_match is None or alias_match.group(1) != target_arm:
                continue
            train_seed = int(alias_match.group(2))
            split = checkpoint["splits"]["multi_answer"]
            for short, source in BACKFILL_METRICS.items():
                grouped[train_seed][short].append(float(split["metrics"][source]))
            provenance[train_seed].append(
                {
                    "eval_seed": eval_seed,
                    "attempts_path": split["attempts_path"],
                    "prompt_metrics_path": split["prompt_metrics_path"],
                    "summary_path": str(path.resolve()),
                    "checkpoint_path": split["checkpoint_path"],
                }
            )

    rows = json.loads(curve_path.read_text(encoding="utf-8"))
    greedy_by_train_seed = {}
    greedy_provenance = {}
    if greedy_summary_path is not None:
        greedy_payload = json.loads(greedy_summary_path.read_text(encoding="utf-8"))
        for checkpoint in greedy_payload["checkpoints"]:
            alias_match = re.fullmatch(r"(.+)_s(\d+)", checkpoint["alias"])
            if alias_match is None or alias_match.group(1) != target_arm:
                continue
            train_seed = int(alias_match.group(2))
            split = checkpoint["splits"]["multi_answer"]
            greedy_by_train_seed[train_seed] = float(
                split["metrics"]["any_correct_at_k"]
            )
            greedy_provenance[train_seed] = {
                "attempts_path": split["attempts_path"],
                "prompt_metrics_path": split["prompt_metrics_path"],
                "summary_path": str(greedy_summary_path.resolve()),
                "checkpoint_path": split["checkpoint_path"],
            }
    merged = 0
    for train_seed, metrics in grouped.items():
        candidates = [
            row
            for row in rows
            if row["arm"] == target_arm
            and int(row["seed"]) == train_seed
            and row["split"] == "multi_answer"
        ]
        if not candidates:
            raise ValueError(
                f"no {target_arm} seed {train_seed} multi-answer row in {curve_path}"
            )
        terminal = max(
            candidates,
            key=lambda row: (
                -math.inf
                if row.get("training_passes") is None
                else float(row["training_passes"]),
                int(row["step"]),
            ),
        )
        eval_seeds = sorted(item["eval_seed"] for item in provenance[train_seed])
        if eval_seeds != [1001, 1002, 1003, 1004]:
            raise ValueError(
                f"expected fixed seeds 1001--1004 for {target_arm} seed "
                f"{train_seed}, found {eval_seeds}"
            )
        for short, draws in metrics.items():
            if len(draws) != 4:
                raise ValueError(
                    f"expected four {short} draws for {target_arm} seed "
                    f"{train_seed}, found {len(draws)}"
                )
            terminal[short] = statistics.fmean(draws)
            terminal[f"{short}_draws"] = draws
            terminal[f"{short}_draw_std"] = statistics.stdev(draws)
            terminal[f"{short}_draw_se"] = statistics.stdev(draws) / math.sqrt(4)
            terminal[f"{short}_draw_min"] = min(draws)
            terminal[f"{short}_draw_max"] = max(draws)
        terminal["repeated_eval_seeds"] = eval_seeds
        terminal["repeated_eval_trace_count"] = len(provenance[train_seed])
        terminal["repeated_eval_traces"] = sorted(
            provenance[train_seed], key=lambda item: item["eval_seed"]
        )
        terminal["repeated_eval_source"] = "external_fixed_k8x4_terminal_backfill"
        if train_seed in greedy_by_train_seed:
            terminal["greedy"] = greedy_by_train_seed[train_seed]
            terminal["greedy_eval_trace"] = greedy_provenance[train_seed]
            terminal["greedy_eval_source"] = "external_deterministic_terminal_backfill"
        merged += 1
    curve_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return merged


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for stamp, prompt_pool_size, num_samples in CELLS:
        output = ARTIFACTS / f"{stamp}_scaling_curve.json"
        subprocess.run(
            [
                sys.executable,
                str(PARSER),
                "--stamp-prefix",
                stamp,
                "--run-data-root",
                str(RUN_DATA),
                "--out",
                str(output),
                "--prompt-pool-size",
                str(prompt_pool_size),
                "--num-samples",
                str(num_samples),
                "--max-training-passes",
                "10",
            ],
            check=True,
            cwd=ROOT,
        )
        clean_limits = MAX_CLEAN_STEP_BY_STAMP.get(stamp)
        if clean_limits is not None:
            rows = json.loads(output.read_text(encoding="utf-8"))
            clean_rows = [
                row
                for row in rows
                if int(row["step"]) <= clean_limits[int(row["seed"])]
            ]
            output.write_text(
                json.dumps(clean_rows, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[refresh] clipped {stamp} to audited clean boundaries: "
                f"{clean_limits}; retained {len(clean_rows)}/{len(rows)} rows"
            )
    repeated_root = ARTIFACTS / "freeform_05b_repeated_eval_v1"
    for curve_stamp, eval_stamp, target_arm in REPEATED_EVAL_BACKFILLS:
        summary_paths = sorted(
            repeated_root.glob(f"{eval_stamp}_e*_coverage_summary.json")
        )
        if not summary_paths:
            print(
                f"[refresh] no repeated backfill found for {curve_stamp}; "
                "retaining legacy single-evaluation points"
            )
            continue
        merged = merge_repeated_terminal_evaluation(
            ARTIFACTS / f"{curve_stamp}_scaling_curve.json",
            summary_paths,
            target_arm,
            greedy_summary_path=(
                greedy_path
                if (
                    greedy_path := repeated_root
                    / f"{eval_stamp}_greedy_coverage_summary.json"
                ).is_file()
                else None
            ),
        )
        print(
            f"[refresh] merged four fixed draws into {merged} terminal "
            f"{target_arm} rows for {curve_stamp}"
        )
    manifest_path = write_evaluation_coverage_manifest()
    print(f"[refresh] wrote evaluation coverage manifest {manifest_path}")


if __name__ == "__main__":
    main()
