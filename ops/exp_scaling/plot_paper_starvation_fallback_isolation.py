#!/usr/bin/env python3
"""Plot terminal E103 starvation-fallback isolation outcomes and actuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as style  # noqa: E402
from plot_paper_direct_comparator_endpoint_effects import (  # noqa: E402
    _completion_marker,
    _interval,
    _relative,
    _runs,
    _sha256,
    _terminal_metrics,
)
from plot_paper_open_bank_bundle_effects import (  # noqa: E402
    DOMAIN_LABEL,
    DOMAIN_ORDER,
    SEEDS,
    TARGET,
    _effect,
    _read,
)

E102_LEDGER = ROOT / "var/artifacts/e102_full_open_bank_maxent_replay_05b_jobs.json"
E103_LEDGER = ROOT / "var/artifacts/e103_starvation_fallback_maxent_replay_05b_jobs.json"
E103_AUDIT = ROOT / "var/artifacts/e103_starvation_fallback_audit_latest.json"
PROTOCOL = ROOT / "paper/preregistration/e103_starvation_fallback_maxent_replay_05b_20260817.md"
DEFAULT_OUTPUT = ROOT / "paper/figures/starvation_fallback_isolation"
EFFECT_FIELDS = ("pass8", "distinct8")
EFFECT_LABEL = {"pass8": "pass@8", "distinct8": "distinct@8"}
MECHANISM_FIELDS = (
    "fallback_activations",
    "fallback_admissions",
    "fallback_extra_groups",
)
MECHANISM_LABEL = {
    "fallback_activations": "activations",
    "fallback_admissions": "admissions",
    "fallback_extra_groups": "extra groups",
}


def build() -> dict[str, Any]:
    parent, treatment, audit = map(_read, (E102_LEDGER, E103_LEDGER, E103_AUDIT))
    if (
        audit.get("schema") != "e103-starvation-fallback-audit-v1"
        or audit.get("terminal") is not True
        or audit.get("passed") is not True
        or audit.get("violations") != []
    ):
        raise RuntimeError("E103 terminal mechanism audit has not passed")
    if (
        parent.get("target_steps") != TARGET
        or treatment.get("target_steps") != TARGET
        or tuple(treatment.get("domains", ())) != DOMAIN_ORDER
        or tuple(treatment.get("seeds", ())) != SEEDS
        or len(treatment.get("runs", ())) != len(DOMAIN_ORDER) * len(SEEDS)
    ):
        raise RuntimeError("E102/E103 design drifted")

    parent_runs = _runs(parent)
    treatment_runs = _runs(treatment)
    audit_runs = {
        (str(run["domain"]), int(run["seed"])): run for run in audit["runs"]
    }
    source_paths: set[Path] = {
        E102_LEDGER, E103_LEDGER, E103_AUDIT, PROTOCOL, Path(__file__)
    }
    cells: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            parent_dir = Path(parent_runs[(domain, seed)]["run_dir"])
            treatment_run = treatment_runs[(domain, seed)]
            treatment_dir = Path(treatment_run["run_dir"])
            parent_marker = _completion_marker(parent_dir, target=TARGET)
            treatment_marker = _completion_marker(treatment_dir, target=TARGET)
            if parent_marker is None or treatment_marker is None:
                raise RuntimeError(f"{domain}/seed {seed}: nonterminal E102/E103 pair")
            parent_endpoint, parent_sources = _terminal_metrics(
                parent_dir, target=TARGET
            )
            treatment_endpoint, treatment_sources = _terminal_metrics(
                treatment_dir, target=TARGET
            )
            source_paths.update(
                parent_sources | treatment_sources | {parent_marker, treatment_marker}
            )
            report = audit_runs[(domain, seed)]["report"]
            per_seed[str(seed)] = {
                "job_id": int(treatment_run["job_id"]),
                "parent": parent_endpoint,
                "treatment": treatment_endpoint,
                "effect": _effect(treatment_endpoint, parent_endpoint),
                "mechanism": {
                    field: float(report[field]) for field in MECHANISM_FIELDS
                },
            }
        cells.append(
            {
                "domain": domain,
                "n": len(SEEDS),
                "seeds": list(SEEDS),
                "per_seed": per_seed,
                "effect_summaries": {
                    field: _interval(
                        [per_seed[str(seed)]["effect"][field] for seed in SEEDS]
                    )
                    for field in EFFECT_FIELDS
                },
                "mechanism_summaries": {
                    field: {
                        "mean": statistics.fmean(
                            per_seed[str(seed)]["mechanism"][field] for seed in SEEDS
                        ),
                        "total": sum(
                            per_seed[str(seed)]["mechanism"][field] for seed in SEEDS
                        ),
                    }
                    for field in MECHANISM_FIELDS
                },
            }
        )
    return {
        "schema": "paper-starvation-fallback-isolation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "terminal; 25/25 E103 cells passed the outcome-blind mechanism audit",
        "model": "Qwen2.5-0.5B",
        "comparison": "E103 starvation fallback minus matched E102 open-bank bundle",
        "domain_order": list(DOMAIN_ORDER),
        "seeds": list(SEEDS),
        "target_steps": TARGET,
        "cells": cells,
        "outcome_metrics_used_for_mechanism_gate": False,
        "audit_violations": audit["violations"],
        "input_sha256": {
            _relative(path): {"byte_length": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(source_paths)
        },
    }


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        2, len(DOMAIN_ORDER), figsize=(style.WIDTH, 4.75), squeeze=False
    )
    effect_values = [
        cell["per_seed"][str(seed)]["effect"][field]
        for cell in payload["cells"] for seed in SEEDS for field in EFFECT_FIELDS
    ]
    limit = max(abs(min(effect_values)), abs(max(effect_values))) + 0.08
    jitter = (-0.036, -0.018, 0.0, 0.018, 0.036)
    for column, cell in enumerate(payload["cells"]):
        effect_axis = axes[0][column]
        style.style_axis(effect_axis, grid="both")
        effect_axis.axhline(0.0, color=style.MUTED, lw=0.8, linestyle=(0, (2, 2)))
        for x, field in enumerate(EFFECT_FIELDS):
            values = [
                cell["per_seed"][str(seed)]["effect"][field] for seed in SEEDS
            ]
            effect_axis.scatter(
                [x + value for value in jitter], values, s=9,
                facecolor=style.WHITE, edgecolor=style.ADD_ON, linewidth=0.7,
                zorder=3,
            )
            summary = cell["effect_summaries"][field]
            low, high = summary["student_t_95"]
            effect_axis.vlines(x, low, high, color=style.ADD_ON, linewidth=1.35)
            effect_axis.scatter(
                [x], [summary["mean"]], marker="D", s=25, color=style.ADD_ON,
                edgecolor=style.WHITE, linewidth=0.5, zorder=4,
            )
        effect_axis.set_xlim(-0.42, 1.42)
        effect_axis.set_ylim(-limit, limit)
        effect_axis.set_xticks(
            range(len(EFFECT_FIELDS)),
            [EFFECT_LABEL[field] for field in EFFECT_FIELDS],
        )
        effect_axis.set_title(DOMAIN_LABEL[cell["domain"]], fontsize=style.LABEL_FONT)
        effect_axis.text(
            0.03, 0.96, "paired n=5", transform=effect_axis.transAxes,
            ha="left", va="top", fontsize=style.SMALL_FONT, color=style.MUTED,
        )
        if column == 0:
            effect_axis.set_ylabel("fallback - base-bundle effect", fontsize=style.LABEL_FONT)
        else:
            effect_axis.tick_params(labelleft=False)

        mechanism_axis = axes[1][column]
        style.style_axis(mechanism_axis, grid="both")
        mechanism_axis.set_yscale("symlog", linthresh=1.0)
        for x, field in enumerate(MECHANISM_FIELDS):
            values = [
                cell["per_seed"][str(seed)]["mechanism"][field] for seed in SEEDS
            ]
            mechanism_axis.scatter(
                [x + value for value in jitter], values, s=9,
                facecolor=style.WHITE, edgecolor=style.METHOD, linewidth=0.7,
                zorder=3,
            )
            mechanism_axis.scatter(
                [x], [statistics.fmean(values)], marker="D", s=25,
                color=style.METHOD, edgecolor=style.WHITE, linewidth=0.5, zorder=4,
            )
        mechanism_axis.set_xlim(-0.42, 2.42)
        mechanism_axis.set_xticks(
            range(len(MECHANISM_FIELDS)),
            [MECHANISM_LABEL[field] for field in MECHANISM_FIELDS],
            rotation=24,
            ha="right",
        )
        if column == 0:
            mechanism_axis.set_ylabel(
                "fallback count per run\n(symlog)", fontsize=style.LABEL_FONT
            )
        else:
            mechanism_axis.tick_params(labelleft=False)

    style.bottom_legend(
        figure,
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=style.WHITE,
                   markeredgecolor=style.ADD_ON, markersize=4),
            Line2D([0], [0], marker="D", color=style.ADD_ON,
                   markeredgecolor=style.WHITE, markersize=4),
        ],
        ["paired seed", "mean + 95% Student-t interval (outcomes); mean (mechanism)"],
        y=0.004,
        ncol=2,
    )
    figure.suptitle(
        "Starvation fallback: actuation without terminal breadth gain",
        fontsize=style.TITLE_FONT, color=style.INK, y=0.995,
    )
    figure.text(
        0.5, 0.962,
        "Qwen2.5-0.5B · pass 8 · only bounded explorer fallback differs from the base bundle.",
        ha="center", va="top", fontsize=style.SMALL_FONT, color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.095, right=0.995, top=0.88, bottom=0.19, hspace=0.34, wspace=0.25
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    render(payload, output)
    for suffix in (".json", ".pdf", ".png"):
        print(f"wrote {output.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
