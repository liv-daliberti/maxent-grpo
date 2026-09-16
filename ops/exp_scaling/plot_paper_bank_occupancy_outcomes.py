#!/usr/bin/env python3
"""Align replay-bank occupancy with correctness-adjusted breadth outcomes.

The join is observational.  It uses every terminal Qwen2.5-0.5B replay run,
never fits or pools a cross-domain association, and does not pretend that the
aggregate logs identify which bank exemplar later reappeared in evaluation.
The first two panels match time windows: whole-training occupancy with
normalized trajectory AUC, and passes 6--8 occupancy with the terminal
endpoint. The third exposes capacity-hit frequency against the endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402


DEFAULT_TELEMETRY = (
    ROOT / "paper" / "figures" / "replay_mechanism_telemetry_qwen05b.json"
)
DEFAULT_RESULTS = ROOT / "paper" / "results" / "e78_terminal_05b.json"
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "bank_occupancy_retained_breadth"
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
DOMAIN_LABELS = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "Pantry",
}
DOMAIN_COLORS = {
    "graph_coloring": style.METHOD,
    "countdown": style.CONTROL,
    "python_factors": style.ABLATION,
    "mathir": style.ADD_ON,
    "pantry_plan": style.INK,
}
SEEDS = (43, 44, 45, 46, 47)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "byte_length": path.stat().st_size,
        "sha256": sha256(path),
    }


def finite(value: Any, *, where: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{where}: expected finite numeric value, got {value!r}")
    return float(value)


def mean_record(per_seed: dict[str, dict[str, float]]) -> dict[str, float]:
    fields = tuple(next(iter(per_seed.values())))
    return {
        field: statistics.fmean(record[field] for record in per_seed.values())
        for field in fields
    }


def build_payload(
    telemetry_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if (
        telemetry.get("schema") != "paper-replay-mechanism-telemetry-v1"
        or telemetry.get("status") != "terminal observational mechanism telemetry"
        or telemetry.get("capacity") != 16
        or telemetry.get("replay_runs") != 25
        or telemetry.get("replay_updates") != 76800
    ):
        raise RuntimeError("replay telemetry is not the complete terminal cohort")
    if (
        results.get("schema") != "e78_terminal_paper_results_v1"
        or results.get("design", {}).get("model") != "Qwen2.5-0.5B-Instruct"
        or results.get("design", {}).get("domains") != list(DOMAINS)
        or results.get("design", {}).get("paired_seeds") != list(SEEDS)
        or results.get("design", {}).get("target_steps") != 3072
    ):
        raise RuntimeError("terminal outcome artifact is not the registered cohort")

    cells: list[dict[str, Any]] = []
    for domain in DOMAINS:
        mechanism = telemetry["domains"].get(domain)
        outcome = results["domains"].get(domain)
        if (
            mechanism is None
            or outcome is None
            or mechanism.get("seeds") != list(SEEDS)
        ):
            raise RuntimeError(f"{domain}: missing complete mechanism/outcome cell")
        effects = outcome["paired_effects"]
        required_effects = {
            "normalized_auc_pass8",
            "normalized_auc_distinct8",
            "terminal_pass8",
            "terminal_distinct8",
            "terminal_excess_modes8",
        }
        if not required_effects <= set(effects):
            raise RuntimeError(f"{domain}: terminal effect fields are incomplete")

        per_seed: dict[str, dict[str, float]] = {}
        for seed in SEEDS:
            key = str(seed)
            replay_summary = mechanism["replay_seed_summaries"].get(key)
            if replay_summary is None or replay_summary.get("updates") != 3072:
                raise RuntimeError(f"{domain}/seed{seed}: telemetry grid incomplete")
            mean_occupancy = finite(
                replay_summary.get("mean_available_modes"),
                where=f"{domain}/seed{seed}/mean occupancy",
            )
            late_occupancy = finite(
                mechanism["late_occupancy"]["seed_values"].get(key),
                where=f"{domain}/seed{seed}/late occupancy",
            )
            auc_pass = finite(
                effects["normalized_auc_pass8"]["per_seed"].get(key),
                where=f"{domain}/seed{seed}/AUC pass effect",
            )
            auc_distinct = finite(
                effects["normalized_auc_distinct8"]["per_seed"].get(key),
                where=f"{domain}/seed{seed}/AUC distinct effect",
            )
            terminal_pass = finite(
                effects["terminal_pass8"]["per_seed"].get(key),
                where=f"{domain}/seed{seed}/terminal pass effect",
            )
            terminal_distinct = finite(
                effects["terminal_distinct8"]["per_seed"].get(key),
                where=f"{domain}/seed{seed}/terminal distinct effect",
            )
            terminal_adjusted = finite(
                effects["terminal_excess_modes8"]["per_seed"].get(key),
                where=f"{domain}/seed{seed}/terminal adjusted effect",
            )
            if not math.isclose(
                terminal_adjusted,
                terminal_distinct - terminal_pass,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{domain}/seed{seed}: adjusted endpoint arithmetic drifted"
                )
            per_seed[key] = {
                "mean_bank_occupancy": mean_occupancy,
                "capacity_hit_updates": int(
                    replay_summary.get("capacity_hit_updates")
                ),
                "capacity_hit_fraction": finite(
                    replay_summary.get("capacity_hit_fraction"),
                    where=f"{domain}/seed{seed}/capacity-hit fraction",
                ),
                "normalized_auc_pass8_effect": auc_pass,
                "normalized_auc_distinct8_effect": auc_distinct,
                "normalized_auc_adjusted_breadth_effect": auc_distinct - auc_pass,
                "late_bank_occupancy": late_occupancy,
                "terminal_pass8_effect": terminal_pass,
                "terminal_distinct8_effect": terminal_distinct,
                "terminal_adjusted_breadth_effect": terminal_adjusted,
            }
        means = mean_record(per_seed)
        if not math.isclose(
            means["terminal_adjusted_breadth_effect"],
            finite(
                effects["terminal_excess_modes8"].get("mean"),
                where=f"{domain}/terminal adjusted mean",
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"{domain}: terminal adjusted mean drifted")
        cells.append(
            {
                "domain": domain,
                "label": DOMAIN_LABELS[domain],
                "n": len(SEEDS),
                "seeds": list(SEEDS),
                "per_seed": per_seed,
                "means": means,
            }
        )

    return {
        "schema": "paper-bank-occupancy-retained-breadth-v2",
        "status": (
            "terminal observational cross-measure alignment; "
            "no pooled or causal estimate"
        ),
        "model": "Qwen2.5-0.5B-Instruct",
        "comparison": "Re:Dr minus matched Dr.GRPO",
        "domain_order": list(DOMAINS),
        "registered_seeds": list(SEEDS),
        "capacity": 16,
        "selection_rule": (
            "all 25 terminal replay runs; whole-training occupancy is paired "
            "with normalized trajectory AUC and passes 6--8 occupancy with "
            "the pass-8 endpoint; capacity-hit frequency is paired with the "
            "same endpoint; no run or domain is selected by outcome"
        ),
        "panels": {
            "trajectory": {
                "x": "mean available modes in scheduled bank, passes 0--8",
                "y": "normalized AUC effect in distinct@8 minus pass@8",
            },
            "terminal": {
                "x": "mean available modes in scheduled bank, passes 6--8",
                "y": "terminal effect in distinct@8 minus pass@8",
            },
            "capacity": {
                "x": "fraction of optimizer updates at the 16-mode cap",
                "y": "terminal effect in distinct@8 minus pass@8",
            },
        },
        "aggregation": (
            "five seed points and one descriptive within-domain mean; "
            "no regression, correlation, interval, or cross-domain pooling"
        ),
        "input_sha256": {
            str(telemetry_path.relative_to(ROOT)): file_record(telemetry_path),
            str(results_path.relative_to(ROOT)): file_record(results_path),
        },
        "cells": cells,
        "limitations": [
            "Occupancy and evaluated breadth are run-level aggregates, not identities of the same modes.",
            "Bank-exemplar survival and later evaluation re-observation cannot be reconstructed from the retained logs.",
            "Domain difficulty and support size affect both axes, so cross-domain association is not a causal replay-dose estimate.",
        ],
    }


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(1, 3, figsize=(style.WIDTH, 2.82))
    panels = (
        (
            axes[0],
            "A  Whole trajectory",
            "mean_bank_occupancy",
            "normalized_auc_adjusted_breadth_effect",
            "mean bank occupancy, passes 0--8",
            r"normalized AUC  $\Delta(D-P)$",
            (0.65, 7.35),
            (-0.09, 1.34),
            1.0,
        ),
        (
            axes[1],
            "B  Late training and endpoint",
            "late_bank_occupancy",
            "terminal_adjusted_breadth_effect",
            "mean bank occupancy, passes 6--8",
            r"terminal  $\Delta(D-P)$",
            (0.65, 9.15),
            (-0.09, 1.68),
            1.0,
        ),
        (
            axes[2],
            "C  Capacity contact",
            "capacity_hit_fraction",
            "terminal_adjusted_breadth_effect",
            "updates at 16-mode cap (%)",
            r"terminal  $\Delta(D-P)$",
            (-0.25, 5.2),
            (-0.09, 1.68),
            100.0,
        ),
    )
    label_offsets = {
        "graph_coloring": (4.0, 3.0),
        "countdown": (4.0, 3.0),
        "python_factors": (4.0, 5.0),
        "mathir": (4.0, -8.0),
        "pantry_plan": (4.0, 3.0),
    }
    for axis, title, x_field, y_field, xlabel, ylabel, xlim, ylim, x_scale in panels:
        style.style_axis(axis, title=title)
        axis.axhline(
            0.0,
            color=style.MUTED,
            linewidth=0.8,
            linestyle=(0, (4, 1.5)),
            zorder=1,
        )
        for cell in payload["cells"]:
            domain = cell["domain"]
            values = list(cell["per_seed"].values())
            xs = [record[x_field] * x_scale for record in values]
            ys = [record[y_field] for record in values]
            color = DOMAIN_COLORS[domain]
            axis.scatter(
                xs,
                ys,
                s=18,
                marker="o",
                facecolor=style.WHITE,
                edgecolor=color,
                linewidth=0.75,
                zorder=3,
            )
            mean_x = cell["means"][x_field] * x_scale
            mean_y = cell["means"][y_field]
            axis.scatter(
                [mean_x],
                [mean_y],
                s=38,
                marker="D",
                color=color,
                edgecolor=style.WHITE,
                linewidth=0.55,
                zorder=4,
            )
            axis.annotate(
                cell["label"],
                (mean_x, mean_y),
                xytext=label_offsets[domain],
                textcoords="offset points",
                fontsize=style.SMALL_FONT,
                color=color,
                zorder=5,
            )
        axis.set_xlabel(xlabel, fontsize=style.LABEL_FONT)
        axis.set_ylabel(ylabel, fontsize=style.LABEL_FONT)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)

    figure.suptitle(
        "Bank occupancy versus retained breadth",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        (
            "Qwen2.5-0.5B · open circles are paired seeds · diamonds are "
            "domain means · n=5/domain · no pooled fit"
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.84,
        bottom=0.19,
        wspace=0.37,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, default=DEFAULT_TELEMETRY)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    telemetry = args.telemetry.resolve()
    results = args.results.resolve()
    output = args.output.resolve()
    for path in (telemetry, results):
        if not path.is_file():
            raise SystemExit(f"missing source artifact: {path}")
    payload = build_payload(telemetry, results)
    render(payload, output)
    write_json(output.with_suffix(".json"), payload)
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
