#!/usr/bin/env python3
"""Summarize the completed E72 decoding grid as the paper's decoding objection.

The objection this answers is that collapse is a decoding artifact: that a
narrow policy is really a broad one sampled too conservatively, so raising the
temperature, drawing more samples, or widening the nucleus would recover the
missing modes. Stage a varies temperature, stage b the sample budget, and
stage c the nucleus truncation, all on the same frozen terminal checkpoints
through the ordinary evaluation path. Each cell is therefore a pure function of
a checkpoint and its decoding settings.

Every number is recomputed from the retained per-draw records, whose canonical
keys were written by the same validator that graded them; nothing is resampled
or regraded here. \\pmd{} pools a prompt's draws before scoring it, matching the
aggregation used everywhere else in the paper, and a cell is reported only when
at least 30 prompts return two verified responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "var/data/e72_frontier"
MANIFEST = ROOT / "var/artifacts/e72_frontier_source_runs.json"
DEFAULT_OUTPUT = ROOT / "paper/results/decoding_objection_e72"

SEEDS = (43, 44, 45, 46, 47)
SUPPORT_BAR = 30
DOMAIN_ORDER = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
DOMAIN_LABEL = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
# The retained arm names predate the Re: naming; drgrpo is the verifier-only
# control and xgrpo the replay arm of that cohort.
ARMS = {"drgrpo": "control", "xgrpo": "replay"}
SETTINGS: dict[str, tuple[tuple[str, float, float, int], ...]] = {
    "a": (("T0p5_p1_K8_d4", 0.5, 1.0, 8), ("T0p7_p1_K8_d4", 0.7, 1.0, 8),
          ("T1_p1_K8_d4", 1.0, 1.0, 8), ("T1p3_p1_K8_d4", 1.3, 1.0, 8),
          ("T1p6_p1_K8_d4", 1.6, 1.0, 8), ("T2_p1_K8_d4", 2.0, 1.0, 8)),
    "b": (("T1_p1_K32_d2", 1.0, 1.0, 32), ("T1p3_p1_K32_d2", 1.3, 1.0, 32),
          ("T1p6_p1_K32_d2", 1.6, 1.0, 32)),
    "c": (("T1_p0p95_K8_d4", 1.0, 0.95, 8), ("T1p6_p0p95_K8_d4", 1.6, 0.95, 8)),
}
REFERENCE_SETTING = ("a", "T1_p1_K8_d4")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw_file(stage: str, domain: str, arm: str, seed: int, setting: str) -> Path | None:
    """The one admissible attempt for a cell, following the paper's source rule.

    Identical repeated attempts are deduplicated by content; attempts that
    disagree invalidate the cell rather than being chosen by metric value.
    """
    cell = SWEEP / stage / domain / f"{arm}_s{seed}" / setting
    if not cell.is_dir():
        return None
    hits = sorted(p for p in cell.glob("*/eval_mode_coverage_draws.jsonl") if p.is_file())
    hits = [p for p in hits if (p.parent / "EVAL_ONLY_COMPLETE.json").is_file()]
    if not hits:
        return None
    digests = {_digest(p) for p in hits}
    if len(digests) > 1:
        raise ValueError(
            f"{cell} retains {len(hits)} completed attempts that disagree; "
            "admit one before summarizing")
    return hits[0]


def read_cell(stage: str, domain: str, arm: str, seed: int, setting: str) -> dict[str, Any] | None:
    """pass@K and \\pmd{} for one checkpoint at one decoding setting."""
    path = _draw_file(stage, domain, arm, seed, setting)
    if path is None:
        return None
    keys_by_prompt: dict[int, list[str]] = defaultdict(list)
    draw_success: list[float] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # The greedy trace shares the file but is not one of the sampled draws.
        if record.get("evaluation_kind") != "fixed_seed_sampled_k_neutral":
            continue
        for prompt in record["prompts"]:
            verified = [key for key, reward in zip(prompt["answer_keys"], prompt["rewards"])
                        if key and reward and reward > 0]
            keys_by_prompt[prompt["prompt_index"]].extend(verified)
            draw_success.append(1.0 if verified else 0.0)
    if not draw_success:
        return None
    pmds = []
    for keys in keys_by_prompt.values():
        total = len(keys)
        if total < 2:
            continue
        counts: dict[str, int] = defaultdict(int)
        for key in keys:
            counts[key] += 1
        collision = sum(n * (n - 1) for n in counts.values()) / (total * (total - 1))
        pmds.append(1.0 - collision)
    return {
        "pass_at_k": sum(draw_success) / len(draw_success),
        "pmd": statistics.mean(pmds) if len(pmds) >= SUPPORT_BAR else None,
        "defined_prompts": len(pmds),
        "source": str(path.relative_to(ROOT)),
    }


def build_record() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    cells: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        for arm in ARMS:
            for stage, settings in SETTINGS.items():
                for setting, temperature, top_p, k in settings:
                    seed_values = {}
                    for seed in SEEDS:
                        measured = read_cell(stage, domain, arm, seed, setting)
                        if measured is not None:
                            seed_values[seed] = measured
                    if not seed_values:
                        continue
                    defined = [v["pmd"] for v in seed_values.values() if v["pmd"] is not None]
                    cells.append({
                        "domain": domain,
                        "arm": ARMS[arm],
                        "stage": stage,
                        "temperature": temperature,
                        "top_p": top_p,
                        "k": k,
                        "seeds": sorted(seed_values),
                        "pass_at_k": statistics.mean(v["pass_at_k"] for v in seed_values.values()),
                        "pmd": statistics.mean(defined) if defined else None,
                        "pmd_seed_n": len(defined),
                        "mean_defined_prompts": statistics.mean(
                            v["defined_prompts"] for v in seed_values.values()),
                    })
    return {
        "schema": "decoding-objection-e72-v1",
        "analysis_code_sha256": _digest(Path(__file__)),
        "source": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": _digest(MANIFEST)},
        "cohort": {
            "model_tag": manifest["model_tag"],
            "seeds": list(SEEDS),
            "arms": {"control": "drgrpo", "replay": "xgrpo"},
            "note": (
                "E72 terminal checkpoints: twelve training passes to step 4,609, and a "
                "replay arm that predates the Re:Dr objective contract. Same "
                "benchmark, splits, seeds and evaluation path as the main comparisons."
            ),
        },
        "support_bar": SUPPORT_BAR,
        "settings": {stage: [list(s) for s in settings] for stage, settings in SETTINGS.items()},
        "reference_setting": {"stage": REFERENCE_SETTING[0], "setting": REFERENCE_SETTING[1]},
        "cells": cells,
    }


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    """Per domain: the control's whole-grid range against the replay default."""
    out = {}
    for domain in DOMAIN_ORDER:
        control = [c for c in record["cells"]
                   if c["domain"] == domain and c["arm"] == "control" and c["pmd"] is not None]
        reference = [c for c in record["cells"]
                     if c["domain"] == domain and c["arm"] == "replay"
                     and c["stage"] == REFERENCE_SETTING[0]
                     and (c["temperature"], c["top_p"], c["k"]) == (1.0, 1.0, 8)]
        grid = [c for c in record["cells"] if c["domain"] == domain and c["arm"] == "control"]
        best = max(control, key=lambda c: c["pmd"]) if control else None
        out[domain] = {
            "settings_measured": len(grid),
            "settings_reportable": len(control),
            "control_pmd_min": min((c["pmd"] for c in control), default=None),
            "control_pmd_max": best["pmd"] if best else None,
            "control_best_setting": (
                {"temperature": best["temperature"], "top_p": best["top_p"], "k": best["k"]}
                if best else None),
            "replay_pmd_default": reference[0]["pmd"] if reference else None,
            "control_pass_default": next(
                (c["pass_at_k"] for c in grid
                 if (c["stage"], c["temperature"], c["top_p"], c["k"]) == ("a", 1.0, 1.0, 8)), None),
        }
    return out


def render_table(record: dict[str, Any]) -> str:
    """Table body: one row per domain, control grid range beside replay default."""
    summary = summarize(record)
    lines = []
    for domain in DOMAIN_ORDER:
        s = summary[domain]
        if s["control_pmd_max"] is None:
            span = r"\multicolumn{1}{c}{---}"
        elif f"{s['control_pmd_min']:.3f}" == f"{s['control_pmd_max']:.3f}":
            span = f"{s['control_pmd_max']:.3f}".lstrip("0")
        else:
            span = (f"{s['control_pmd_min']:.3f}".lstrip("0") + "--"
                    + f"{s['control_pmd_max']:.3f}".lstrip("0"))
        replay = (f"{s['replay_pmd_default']:.3f}".lstrip("0")
                  if s["replay_pmd_default"] is not None else r"\multicolumn{1}{c}{---}")
        lines.append(
            f"    {DOMAIN_LABEL[domain]} & {s['settings_reportable']}/{s['settings_measured']} "
            f"& {span} & {replay} \\\\"
        )
    # The generated body carries its own bottom rule: a bare \\ at end of file
    # would be scanned for an optional argument and expand \bottomrule too early.
    return "\n".join(lines) + "\n    \\bottomrule\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    record = build_record()
    record["summary"] = summarize(record)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    Path(str(args.output) + "_table_body.tex").write_text(render_table(record))
    if args.print_summary:
        for domain, s in record["summary"].items():
            print(f"{domain:16s} control {s['control_pmd_min']} - {s['control_pmd_max']} "
                  f"over {s['settings_reportable']}/{s['settings_measured']} settings; "
                  f"replay default {s['replay_pmd_default']}")
    print(f"Wrote {args.output.with_suffix('.json').name} and its table body "
          f"({len(record['cells'])} measured cells).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
