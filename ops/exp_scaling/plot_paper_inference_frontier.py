#!/usr/bin/env python3
"""Render cross-scale distinct@K for the full x-Mode treatment.

x-Mode Dr.GRPO is the registered combination of canonical verified replay and
adaptive semantic MaxEnt.  The figure uses model rows and static-domain
columns, plots distinct@K only, and compares against seed-matched Dr.GRPO at
the same training checkpoint.  Each cell uses every started seed pair with a
positive complete checkpoint and then freezes the deepest checkpoint shared by
that constant seed set.  Thus terminal and nonterminal cells can be disclosed
without changing n along a curve or pretending that progress is an endpoint.

Four independent registered K=8 draws are concatenated per prompt and split
into non-overlapping blocks for K in {1, 2, 4, 8, 16, 32}.  Consequently the
K=8 values exactly reproduce the registered pass@8 and distinct@8 estimators.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402


DEFAULT_OUTPUT = ROOT / "paper/figures/xmode_adaptive_cross_scale_distinct_at_k"
DEFAULT_TABLE_BODY = (
    ROOT / "paper/results/xmode_adaptive_cross_scale_endpoints_table_body.tex"
)
KS = (1, 2, 4, 8, 16, 32)
EXPECTED_DRAWS = 4
EXPECTED_SAMPLES_PER_DRAW = 8
EVALUATION_KIND = "fixed_seed_sampled_k_neutral"
DOMAIN_ORDER = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
DOMAIN_LABEL = {
    "graph_coloring": "Graph", "countdown": "Countdown",
    "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "Pantry",
}
DOMAIN_TABLE_LABEL = {
    "graph_coloring": "Graph coloring", "countdown": "Countdown",
    "python_factors": "Python factors", "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
SCALE_SPECS = (
    {
        "key": "qwen05b",
        "label": "Qwen2.5-0.5B",
        "control_ledger": ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
        "xmode_ledger": ROOT / "var/artifacts/e89_adaptive_semantic_maxent_reachable_05b_jobs.json",
    },
    {
        "key": "falcon1b",
        "label": "Falcon3-1B",
        "control_ledger": ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
        "xmode_ledger": ROOT / "var/artifacts/e91_falcon_adaptive_semantic_maxent_jobs.json",
    },
    {
        "key": "qwen3b",
        "label": "Qwen2.5-3B",
        "control_ledger": ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
        "xmode_ledger": ROOT / "var/artifacts/e92_qwen3b_adaptive_semantic_maxent_jobs.json",
    },
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_prefix(path: Path) -> tuple[bytes, dict[str, Any]]:
    byte_length = path.stat().st_size
    with path.open("rb") as handle:
        data = handle.read(byte_length)
    if len(data) != byte_length:
        raise RuntimeError(f"short read while freezing {path}")
    return data, {
        "path": str(path.relative_to(ROOT)),
        "byte_length": byte_length,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _complete_draws_by_step(run: dict[str, Any]) -> dict[str, Any] | None:
    """Return complete four-draw checkpoints and exact append-only prefixes."""

    records: dict[tuple[int, int], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    paths = sorted(
        Path(run["run_dir"]).glob("debug_job*/eval_mode_coverage_draws.jsonl")
    )
    if not paths:
        return None
    for path in paths:
        data, source = _frozen_prefix(path)
        source.update({"job_id": run.get("job_id")})
        sources.append(source)
        for raw_line in data.splitlines():
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if row.get("evaluation_kind") != EVALUATION_KIND:
                continue
            if row.get("sample_count") != EXPECTED_SAMPLES_PER_DRAW:
                continue
            step, draw = row.get("step"), row.get("draw_index")
            if not isinstance(step, int) or draw not in range(EXPECTED_DRAWS):
                continue
            records[(step, int(draw))] = row
    steps = sorted({step for step, _draw in records})
    complete = {
        step: {draw: records[(step, draw)] for draw in range(EXPECTED_DRAWS)}
        for step in steps
        if all((step, draw) in records for draw in range(EXPECTED_DRAWS))
    }
    if not complete:
        return None
    return {"draws": complete, "sources": sources}


def _frontier(draws: dict[int, dict[str, Any]]) -> dict[int, dict[str, float]]:
    prompt_count = len(draws[0]["prompts"])
    if any(len(draws[draw]["prompts"]) != prompt_count for draw in draws):
        raise RuntimeError("sampled draws disagree on prompt count")

    prompt_samples: list[tuple[list[Any], list[float]]] = []
    for prompt_index in range(prompt_count):
        keys: list[Any] = []
        rewards: list[float] = []
        for draw in range(EXPECTED_DRAWS):
            prompt = draws[draw]["prompts"][prompt_index]
            draw_keys = list(prompt["answer_keys"])
            draw_rewards = [float(value) for value in prompt["rewards"]]
            if len(draw_keys) != EXPECTED_SAMPLES_PER_DRAW:
                raise RuntimeError("unexpected answer-key count")
            if len(draw_rewards) != EXPECTED_SAMPLES_PER_DRAW:
                raise RuntimeError("unexpected sampled-reward count")
            keys.extend(draw_keys)
            rewards.extend(draw_rewards)
        prompt_samples.append((keys, rewards))

    result: dict[int, dict[str, float]] = {}
    total = EXPECTED_DRAWS * EXPECTED_SAMPLES_PER_DRAW
    for k in KS:
        values: dict[str, list[float]] = defaultdict(list)
        for keys, rewards in prompt_samples:
            for start in range(0, total, k):
                block_keys = keys[start : start + k]
                block_rewards = rewards[start : start + k]
                values["pass8"].append(float(any(value > 0 for value in block_rewards)))
                values["distinct8"].append(float(len({
                    str(key)
                    for key, reward in zip(block_keys, block_rewards)
                    if reward > 0 and key is not None
                })))
        result[k] = {
            metric: sum(samples) / len(samples)
            for metric, samples in values.items()
        }
    return result


def _run_map(ledger: dict[str, Any], *, arm: str | None) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for run in ledger["runs"]:
        if arm is not None and run.get("arm") != arm:
            continue
        key = (str(run["domain"]), int(run["seed"]))
        if key in result:
            raise RuntimeError(f"duplicate registered run {key}")
        result[key] = run
    return result


def _summarize_cell(
    *, scale: dict[str, Any], domain: str,
    controls: dict[tuple[str, int], dict[str, Any]],
    treatments: dict[tuple[str, int], dict[str, Any]],
    target_step: int, train_rows: int,
) -> dict[str, Any]:
    seed_records: dict[int, dict[str, Any]] = {}
    for seed in sorted({key[1] for key in treatments if key[0] == domain}):
        control_run = controls.get((domain, seed))
        treatment_run = treatments.get((domain, seed))
        if control_run is None or treatment_run is None:
            continue
        control = _complete_draws_by_step(control_run)
        treatment = _complete_draws_by_step(treatment_run)
        if control is None or treatment is None:
            continue
        common = sorted(set(control["draws"]) & set(treatment["draws"]))
        positive = [step for step in common if step > 0]
        if not positive:
            continue
        seed_records[seed] = {
            "control": control, "xmode": treatment,
            "common_steps": positive,
        }
    if not seed_records:
        raise RuntimeError(f"{scale['key']}/{domain}: no started matched pair")
    shared_steps = sorted(set.intersection(*(
        set(record["common_steps"]) for record in seed_records.values()
    )))
    if not shared_steps:
        raise RuntimeError(f"{scale['key']}/{domain}: no constant-n shared checkpoint")
    step = shared_steps[-1]
    per_seed: dict[str, dict[int, dict[str, float]]] = {
        "control": {}, "xmode": {},
    }
    sources: list[dict[str, Any]] = []
    for seed, record in seed_records.items():
        for method in ("control", "xmode"):
            per_seed[method][seed] = _frontier(record[method]["draws"][step])
            for source in record[method]["sources"]:
                sources.append({
                    **source, "scale": scale["key"], "domain": domain,
                    "seed": seed, "method": method,
                })

    summaries: dict[str, Any] = {}
    for method in ("control", "xmode"):
        summaries[method] = {}
        for k in KS:
            summaries[method][str(k)] = {}
            for metric in ("pass8", "distinct8"):
                values = {
                    str(seed): per_seed[method][seed][k][metric]
                    for seed in sorted(per_seed[method])
                }
                samples = list(values.values())
                summaries[method][str(k)][metric] = {
                    "mean": sum(samples) / len(samples),
                    "range": [min(samples), max(samples)],
                    "per_seed": values,
                }
    return {
        "scale": scale["key"], "scale_label": scale["label"],
        "domain": domain, "seeds": sorted(seed_records),
        "n": len(seed_records), "checkpoint_step": step,
        "checkpoint_training_pass": step / train_rows,
        "terminal_checkpoint": step == target_step,
        "summaries": summaries, "prefix_sources": sources,
    }


def build_snapshot() -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    ledger_sources: list[dict[str, Any]] = []
    for scale in SCALE_SPECS:
        if scale["control_ledger"] is None or scale["xmode_ledger"] is None:
            cells[scale["key"]] = {}
            continue
        control_ledger = _read(scale["control_ledger"])
        xmode_ledger = _read(scale["xmode_ledger"])
        if int(control_ledger["train_rows"]) != int(xmode_ledger["train_rows"]):
            raise RuntimeError(f"{scale['key']}: training-pass denominator mismatch")
        if int(control_ledger["target_steps"]) != int(xmode_ledger["target_steps"]):
            raise RuntimeError(f"{scale['key']}: target-step mismatch")
        controls = _run_map(control_ledger, arm="control")
        treatments = _run_map(xmode_ledger, arm=None)
        cells[scale["key"]] = {
            domain: _summarize_cell(
                scale=scale, domain=domain, controls=controls,
                treatments=treatments,
                target_step=int(xmode_ledger["target_steps"]),
                train_rows=int(xmode_ledger["train_rows"]),
            )
            for domain in DOMAIN_ORDER
        }
        for role in ("control_ledger", "xmode_ledger"):
            path = scale[role]
            ledger_sources.append({
                "scale": scale["key"], "role": role,
                "path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
            })
    return {
        "schema": "xmode-adaptive-cross-scale-frontier-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "mixed terminal and constant-n progress; checkpoint and n are "
            "reported per model-domain cell"
        ),
        "treatment": "Re:Dr.GRPO + Adaptive Semantic MaxEnt",
        "methods": ["drgrpo", "adaptive_semantic_replay"],
        "row_order": [scale["key"] for scale in SCALE_SPECS],
        "domain_order": list(DOMAIN_ORDER),
        "blank_rule": "model-domain cells without sufficient evidence remain blank",
        "ks": list(KS),
        "sample_reuse": (
            "four independent registered K=8 draws concatenated per prompt; "
            "non-overlapping K-block estimators"
        ),
        "selection_rule": (
            "all started seed pairs with a positive complete sampled checkpoint; "
            "deepest checkpoint shared by that constant seed set and both methods"
        ),
        "ledger_sources": ledger_sources,
        "cells": cells,
    }


def render(snapshot: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(SCALE_SPECS), len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 6.8), sharex=True, sharey=True, squeeze=False,
    )
    control_visual = method_visuals.method_style("drgrpo")
    xmode_visual = method_visuals.method_style("adaptive_semantic_replay")
    method_specs = {"control": control_visual, "xmode": xmode_visual}
    maximum = max(
        cell["summaries"][method][str(k)]["distinct8"]["range"][1]
        for row in snapshot["cells"].values()
        for cell in row.values()
        for method in method_specs for k in KS
    )
    high = max(2.0, math.ceil((maximum + 0.15) * 2) / 2)

    def cell_note(cell: dict[str, Any]) -> str:
        status = "terminal" if cell["terminal_checkpoint"] else "progress"
        return (
            f"n={cell['n']} \u00b7 {cell['checkpoint_training_pass']:g}p"
            f" \u00b7 {status}"
        )

    shared_note = style.dominant_note(
        cell_note(cell)
        for row in snapshot["cells"].values()
        for cell in row.values()
    )
    for row_index, scale in enumerate(SCALE_SPECS):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row_index][column]
            style.style_axis(
                axis, grid="both",
                title=DOMAIN_LABEL[domain] if row_index == 0 else None,
            )
            axis.set_xscale("log", base=2)
            axis.set_xlim(1, 32)
            axis.set_xticks(KS, [str(k) for k in KS])
            axis.set_ylim(-0.05, high)
            cell = snapshot["cells"][scale["key"]].get(domain)
            if cell is None:
                if column == 0:
                    axis.set_ylabel(
                        f"{scale['label']}\nmean distinct@K",
                        fontsize=style.LABEL_FONT,
                    )
                else:
                    axis.tick_params(labelleft=False)
                if row_index == len(SCALE_SPECS) - 1:
                    axis.set_xlabel("samples K", fontsize=style.LABEL_FONT)
                continue
            for method, visual in method_specs.items():
                means = [
                    cell["summaries"][method][str(k)]["distinct8"]["mean"]
                    for k in KS
                ]
                lows = [
                    cell["summaries"][method][str(k)]["distinct8"]["range"][0]
                    for k in KS
                ]
                highs = [
                    cell["summaries"][method][str(k)]["distinct8"]["range"][1]
                    for k in KS
                ]
                axis.fill_between(
                    KS, lows, highs, color=visual["color"],
                    alpha=style.BAND_ALPHA, linewidth=0, zorder=1,
                )
                axis.plot(
                    KS, means, color=visual["color"],
                    linestyle=visual["linestyle"], linewidth=style.MEAN_LW,
                    marker=visual["marker"], markersize=2.7,
                    markeredgewidth=0.45, zorder=3,
                )
            note = cell_note(cell)
            if note != shared_note:
                axis.text(
                    0.03, 0.96, note,
                    transform=axis.transAxes, ha="left", va="top",
                    fontsize=style.SMALL_FONT, color=style.MUTED,
                )
            if column == 0:
                axis.set_ylabel(
                    f"{scale['label']}\nmean distinct@K",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)
            if row_index == len(SCALE_SPECS) - 1:
                axis.set_xlabel("samples K", fontsize=style.LABEL_FONT)

    handles = [
        Line2D(
            [0], [0], color=control_visual["color"],
            linestyle=control_visual["linestyle"], linewidth=style.MEAN_LW,
            marker=control_visual["marker"], markersize=3,
        ),
        Line2D(
            [0], [0], color=xmode_visual["color"],
            linestyle=xmode_visual["linestyle"], linewidth=style.MEAN_LW,
            marker=xmode_visual["marker"], markersize=3,
        ),
    ]
    style.bottom_legend(
        figure, handles,
        ["matched Dr.GRPO", "x-Mode Dr.GRPO (replay + adaptive semantic MaxEnt)"],
        y=0.003, ncol=2,
    )
    figure.suptitle(
        "Verified-mode breadth for full x-Mode Dr.GRPO",
        fontsize=style.TITLE_FONT, color=style.INK, y=0.995,
    )
    figure.text(
        0.5, 0.96,
        "Rows: Qwen2.5-0.5B · Falcon3-1B · Qwen2.5-3B; unavailable cells are blank."
        + (f"  All panels {shared_note} unless marked." if shared_note else ""),
        ha="center", va="top", fontsize=style.SMALL_FONT, color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.105, right=0.995, top=0.90, bottom=0.105,
        hspace=0.30, wspace=0.24,
    )
    style.save(figure, output, png=True, dpi=260)
    plt.close(figure)


def _decimal(value: float) -> str:
    rendered = f"{value:.3f}"
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-" + rendered[2:]
    return rendered


def write_table(snapshot: dict[str, Any], path: Path) -> None:
    lines = []
    for scale_index, scale in enumerate(SCALE_SPECS):
        if scale_index:
            lines.append(r"\midrule")
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            cell = snapshot["cells"][scale["key"]].get(domain)
            scale_label = scale["label"] if domain_index == 0 else ""
            if cell is None:
                lines.append(
                    f"{scale_label} & {DOMAIN_TABLE_LABEL[domain]} & "
                    " &  &  &  &  &  &  \\\\"
                )
                continue
            control = cell["summaries"]["control"]["8"]
            xmode = cell["summaries"]["xmode"]["8"]
            delta = (
                xmode["distinct8"]["mean"] - control["distinct8"]["mean"]
            )
            lines.append(
                f"{scale_label} & {DOMAIN_TABLE_LABEL[domain]} & {cell['n']} & "
                f"{cell['checkpoint_training_pass']:g} & "
                f"{_decimal(control['pass8']['mean'])} & "
                f"{_decimal(control['distinct8']['mean'])} & "
                f"{_decimal(xmode['pass8']['mean'])} & "
                f"{_decimal(xmode['distinct8']['mean'])} & "
                f"{_decimal(delta)} \\\\"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-body", type=Path, default=DEFAULT_TABLE_BODY)
    args = parser.parse_args()
    snapshot = build_snapshot()
    output = args.output.resolve()
    render(snapshot, output)
    output.with_suffix(".json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    write_table(snapshot, args.table_body.resolve())
    print(
        f"wrote {output.with_suffix('.pdf')}, {output.with_suffix('.png')}, "
        f"{output.with_suffix('.json')}, and {args.table_body.resolve()}"
    )


if __name__ == "__main__":
    main()
