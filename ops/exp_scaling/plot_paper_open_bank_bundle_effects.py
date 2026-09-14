#!/usr/bin/env python3
"""Plot the preregistered E102 open-bank bundle against both E78 baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
import paper_style as style  # noqa: E402
from plot_paper_direct_comparator_endpoint_effects import (  # noqa: E402
    _completion_marker,
    _interval,
    _relative,
    _runs,
    _sha256,
    _terminal_metrics,
)


CORE_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
E102_LEDGER = ROOT / "var/artifacts/e102_full_open_bank_maxent_replay_05b_jobs.json"
E102_AUDIT = ROOT / "var/artifacts/e102_full_open_bank_maxent_replay_05b_audit_latest.json"
PROTOCOL = ROOT / "paper/preregistration/e102_full_open_bank_maxent_replay_05b_20260814.md"
DEFAULT_OUTPUT = ROOT / "paper/figures/open_bank_bundle_endpoint_effects"
DOMAIN_ORDER = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
DOMAIN_LABEL = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
SEEDS = (43, 44, 45, 46, 47)
TARGET = 3072
BASELINES = ("replay", "control")
BASELINE_LABEL = {"replay": "Re:Dr.GRPO", "control": "Dr.GRPO"}
EFFECT_FIELDS = ("pass8", "adjusted_breadth8")
EFFECT_LABEL = {"pass8": "pass@8", "adjusted_breadth8": "D-P"}
MECHANISM_FIELDS = ("admissions", "priority_replay_groups")
MECHANISM_LABEL = {"admissions": "admissions", "priority_replay_groups": "priority groups"}
OPEN_COLOR = style.ADAPTIVE


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _effect(treatment: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    delta_pass = treatment["pass8"] - baseline["pass8"]
    delta_distinct = treatment["distinct8"] - baseline["distinct8"]
    return {
        "pass8": delta_pass,
        "distinct8": delta_distinct,
        "adjusted_breadth8": delta_distinct - delta_pass,
    }


def build() -> dict[str, Any]:
    core, treatment, audit = map(_read, (CORE_LEDGER, E102_LEDGER, E102_AUDIT))
    if (
        audit.get("schema") != "e102-full-open-bank-campaign-audit-v1"
        or audit.get("terminal") is not True
        or audit.get("passed_so_far") is not True
        or audit.get("violations") != []
        or set(audit.get("domain_gate", {})) != set(DOMAIN_ORDER)
        or not all(item.get("passed") for item in audit["domain_gate"].values())
    ):
        raise RuntimeError("E102 terminal mechanism audit has not passed")
    if "compute_accounting" not in audit:
        raise RuntimeError("E102 audit lacks preregistered compute accounting")
    if (
        core.get("target_steps") != TARGET
        or treatment.get("target_steps") != TARGET
        or tuple(treatment.get("domains", ())) != DOMAIN_ORDER
        or tuple(treatment.get("seeds", ())) != SEEDS
        or len(treatment.get("runs", ())) != len(DOMAIN_ORDER) * len(SEEDS)
    ):
        raise RuntimeError("E102 or matched E78 design drifted")

    treatment_runs = _runs(treatment)
    baseline_runs = {arm: _runs(core, arm=arm) for arm in BASELINES}
    audit_runs = {
        (str(run["domain"]), int(run["seed"])): run
        for run in audit["runs"]
    }
    source_paths: set[Path] = {CORE_LEDGER, E102_LEDGER, E102_AUDIT, PROTOCOL}
    cells: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            run = treatment_runs[(domain, seed)]
            run_dir = Path(run["run_dir"])
            marker = _completion_marker(run_dir, target=TARGET)
            if marker is None:
                raise RuntimeError(f"E102 {domain}/seed {seed} is not terminal")
            endpoint, endpoint_sources = _terminal_metrics(run_dir, target=TARGET)
            source_paths.update(endpoint_sources | {marker})
            baselines: dict[str, Any] = {}
            effects: dict[str, Any] = {}
            for arm in BASELINES:
                baseline_dir = Path(baseline_runs[arm][(domain, seed)]["run_dir"])
                baseline_marker = _completion_marker(baseline_dir, target=TARGET)
                if baseline_marker is None:
                    raise RuntimeError(f"E78 {arm} {domain}/seed {seed} is not terminal")
                baseline, baseline_sources = _terminal_metrics(
                    baseline_dir, target=TARGET
                )
                source_paths.update(baseline_sources | {baseline_marker})
                baselines[arm] = baseline
                effects[arm] = _effect(endpoint, baseline)
            audit_run = audit_runs[(domain, seed)]
            report = audit_run["report"]
            per_seed[str(seed)] = {
                "job_id": int(run["job_id"]),
                "endpoint": endpoint,
                "baselines": baselines,
                "effects": effects,
                "mechanism": {
                    "admissions": float(report["proposal_cumulative_admissions"]),
                    "priority_replay_groups": float(
                        report["priority_replay_groups_cumulative"]
                    ),
                    "proposal_groups": float(report["proposal_groups_generated"]),
                    "proposal_rows": float(report["proposal_rows_generated"]),
                    "replay_actuation_updates": int(report["replay_actuation_updates"]),
                    "balance_eligible_updates": int(report["balance_eligible_updates"]),
                },
            }
        summaries = {
            arm: {
                field: _interval([
                    per_seed[str(seed)]["effects"][arm][field] for seed in SEEDS
                ])
                for field in EFFECT_FIELDS
            }
            for arm in BASELINES
        }
        mechanism_summaries = {
            field: {
                "mean": statistics.fmean(
                    per_seed[str(seed)]["mechanism"][field] for seed in SEEDS
                ),
                "range": [
                    min(per_seed[str(seed)]["mechanism"][field] for seed in SEEDS),
                    max(per_seed[str(seed)]["mechanism"][field] for seed in SEEDS),
                ],
            }
            for field in MECHANISM_FIELDS
        }
        cells.append({
            "domain": domain,
            "n": len(SEEDS),
            "seeds": list(SEEDS),
            "per_seed": per_seed,
            "effect_summaries": summaries,
            "mechanism_summaries": mechanism_summaries,
            "domain_gate": audit["domain_gate"][domain],
            "compute_accounting": audit["compute_accounting"]["by_domain"][domain],
        })
    return {
        "schema": "paper-open-bank-bundle-endpoint-effects-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "terminal preregistered follow-up; 25/25 cells passed mechanism audit",
        "matrix_scope": (
            "E102 is a separately preregistered bundled follow-up outside the "
            "canonical ten-method 750-cell organizing matrix"
        ),
        "model": "Qwen2.5-0.5B",
        "treatment": (
            "Re:Dr.GRPO plus retention-safe whole-bank balance, one fresh "
            "original-prompt proposal group per update, and four-visit 4x mass "
            "priority for newly admitted verified modes"
        ),
        "comparators": ["Re:Dr.GRPO", "Dr.GRPO"],
        "endpoint_metrics": ["pass@8", "distinct@8", "distinct@8-pass@8"],
        "domain_order": list(DOMAIN_ORDER),
        "seeds": list(SEEDS),
        "target_steps": TARGET,
        "cells": cells,
        "mechanism_audit": {
            "domain_gate": audit["domain_gate"],
            "compute_accounting": audit["compute_accounting"],
            "violations": audit["violations"],
        },
        "reporting_gap": audit["compute_accounting"]["proposal_realized_tokens"],
        "input_sha256": {
            _relative(path): {
                "byte_length": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(source_paths)
        },
    }


def _effect_panel(axis: Any, cell: dict[str, Any], arm: str) -> None:
    style.style_axis(axis, grid="both")
    axis.axhline(0.0, color=style.MUTED, lw=0.8, linestyle=(0, (2, 2)))
    jitter = [-0.036, -0.018, 0.0, 0.018, 0.036]
    for x, field in enumerate(EFFECT_FIELDS):
        values = [
            cell["per_seed"][str(seed)]["effects"][arm][field] for seed in SEEDS
        ]
        axis.scatter(
            [x + value for value in jitter], values, s=9,
            facecolor=style.WHITE, edgecolor=OPEN_COLOR, linewidth=0.7, zorder=3,
        )
        summary = cell["effect_summaries"][arm][field]
        low, high = summary["student_t_95"]
        axis.vlines(x, low, high, color=OPEN_COLOR, linewidth=1.35, zorder=4)
        axis.scatter(
            [x], [summary["mean"]], s=25, marker="D", color=OPEN_COLOR,
            edgecolor=style.WHITE, linewidth=0.5, zorder=5,
        )
    axis.set_xlim(-0.42, 1.42)
    axis.set_xticks((0, 1), [EFFECT_LABEL[field] for field in EFFECT_FIELDS])
    # No per-panel "paired n=5": the subtitle already states five paired seeds
    # per domain, and ten copies of a constant only crowd the panels.


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        3, len(DOMAIN_ORDER), figsize=(style.WIDTH, 6.6), squeeze=False,
    )
    effects = [
        cell["per_seed"][str(seed)]["effects"][arm][field]
        for cell in payload["cells"] for seed in SEEDS
        for arm in BASELINES for field in EFFECT_FIELDS
    ]
    limit = max(abs(min(effects)), abs(max(effects))) + 0.15
    # The mechanism row is log-scaled. Each panel used to carry its own scale
    # with only its major labels suppressed, so matplotlib's sub-decade minor
    # labels ("2 x 10^3", "6 x 10^2", ...) still printed at every panel's left
    # edge --- landing on top of the neighbouring panel's data. One shared
    # scale for the row fixes the overprinting and is what makes the counts
    # comparable across domains, which is the row's whole purpose.
    mechanism_values = [
        cell["per_seed"][str(seed)]["mechanism"][field]
        for cell in payload["cells"] for seed in SEEDS
        for field in MECHANISM_FIELDS
    ]
    mechanism_low = min(value for value in mechanism_values if value > 0) / 1.8
    mechanism_high = max(mechanism_values) * 1.8
    for column, cell in enumerate(payload["cells"]):
        for row, arm in enumerate(BASELINES):
            axis = axes[row][column]
            _effect_panel(axis, cell, arm)
            axis.set_ylim(-limit, limit)
            if column == 0:
                axis.set_ylabel(
                    f"open-bank bundle - {BASELINE_LABEL[arm]}\neffect",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)
        axis = axes[2][column]
        style.style_axis(axis, grid="both")
        axis.set_yscale("log")
        for x, field in enumerate(MECHANISM_FIELDS):
            values = [
                cell["per_seed"][str(seed)]["mechanism"][field] for seed in SEEDS
            ]
            axis.scatter(
                [x + value for value in (-0.036, -0.018, 0.0, 0.018, 0.036)],
                values, s=9, facecolor=style.WHITE, edgecolor=style.ADD_ON,
                linewidth=0.7, zorder=3,
            )
            axis.scatter(
                [x], [statistics.fmean(values)], s=25, marker="D",
                color=style.ADD_ON, edgecolor=style.WHITE, linewidth=0.5, zorder=5,
            )
        axis.set_xlim(-0.42, 1.42)
        axis.set_ylim(mechanism_low, mechanism_high)
        axis.set_xticks(
            (0, 1),
            [MECHANISM_LABEL[field] for field in MECHANISM_FIELDS],
            rotation=18,
            ha="right",
        )
        if column == 0:
            axis.set_ylabel("mechanism count per run\n(log scale)", fontsize=style.LABEL_FONT)
        else:
            # `which="both"`: the default only silences the major labels, and
            # on a sub-decade log axis the minor ones are the noisy ones.
            axis.tick_params(labelleft=False, which="both")
        axes[0][column].set_title(DOMAIN_LABEL[cell["domain"]], fontsize=style.LABEL_FONT)
    style.bottom_legend(
        figure,
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=style.WHITE,
                   markeredgecolor=OPEN_COLOR, markersize=4),
            Line2D([0], [0], marker="D", color=OPEN_COLOR,
                   markeredgecolor=style.WHITE, markersize=4),
        ],
        ["paired seed", "mean + 95% Student-t interval (outcomes); mean (mechanism)"],
        y=0.003, ncol=2,
    )
    figure.suptitle(
        "Open-bank discovery bundle: terminal outcomes and mechanism",
        fontsize=style.TITLE_FONT, color=style.INK, y=0.995,
    )
    figure.text(
        0.5, 0.962,
        "Qwen2.5-0.5B · pass 8 · five paired seeds per domain; domains are not pooled.",
        ha="center", va="top", fontsize=style.SMALL_FONT, color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.095, right=0.995, top=0.89, bottom=0.15, hspace=0.32, wspace=0.23,
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
