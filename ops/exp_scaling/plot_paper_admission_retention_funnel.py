#!/usr/bin/env python3
"""Plot the terminal E108 admission-to-retention mechanism funnel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

AUDIT = ROOT / "var/artifacts/e108_admission_retention_mechanism_gate_audit_latest.json"
LEDGER = ROOT / "var/artifacts/e108_admission_retention_mechanism_gate_jobs.json"
PROTOCOL = ROOT / "paper/preregistration/e108_admission_retention_mechanism_gate_20260817.md"
DEFAULT_OUTPUT = ROOT / "paper/figures/admission_retention_funnel"
DOMAIN_ORDER = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
DOMAIN_LABEL = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "Pantry",
}
ARM_ORDER = ("retention_tracking", "adaptive_retention")
ARM_LABEL = {
    "retention_tracking": "Passive tracking",
    "adaptive_retention": "Adaptive priority",
}
STAGES = (
    "tracked_admissions",
    "score_observed_admissions",
    "score_followup_admissions",
    "score_retained_admissions",
    "rollout_eligible_admissions",
    "rollout_converted_admissions",
)
STAGE_LABEL = ("tracked", "observed", "followed up", "retained", "eligible", "converted")
ACTION_FIELDS = ("refresh_requests_cumulative", "priority_visits_added_cumulative")
ACTION_LABEL = ("refresh\nrequests", "priority\nvisits")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    audit = _read(AUDIT)
    ledger = _read(LEDGER)
    if (
        audit.get("schema") != "e108_admission_retention_mechanism_gate_audit_v1"
        or audit.get("terminal") is not True
        or audit.get("passed") is not True
        or audit.get("outcomes_used_for_gate") is not False
        or audit.get("violations") != []
    ):
        raise RuntimeError("E108 terminal outcome-blind mechanism gate has not passed")
    if len(audit.get("runs", ())) != 10 or len(ledger.get("runs", ())) != 10:
        raise RuntimeError("E108 must contain the frozen two-arm five-domain grid")

    rows: dict[str, dict[str, Any]] = {arm: {} for arm in ARM_ORDER}
    for run in audit["runs"]:
        arm = str(run["arm"])
        domain = str(run["domain"])
        if arm not in rows or domain not in DOMAIN_ORDER:
            raise RuntimeError(f"unexpected E108 cell {arm}/{domain}")
        maxima = run["report"]["retention_maxima"]
        rows[arm][domain] = {
            "seed": int(run["seed"]),
            "job_id": int(run["job_id"]),
            "stages": {field: int(maxima[field]) for field in STAGES},
            "actions": {field: int(maxima[field]) for field in ACTION_FIELDS},
            "largest_mean_token_logprob_drop": float(
                run["report"]["retention_latest"]["mean_logprob_drop_max"]
            ),
        }
    if any(set(rows[arm]) != set(DOMAIN_ORDER) for arm in ARM_ORDER):
        raise RuntimeError("E108 grid is incomplete")

    return {
        "schema": "paper-admission-retention-funnel-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "terminal; all 10 mechanism cells passed the outcome-blind gate",
        "model": "Qwen2.5-0.5B",
        "seed": 43,
        "target_steps": 64,
        "domain_order": list(DOMAIN_ORDER),
        "arm_order": list(ARM_ORDER),
        "stage_order": list(STAGES),
        "action_order": list(ACTION_FIELDS),
        "cells": rows,
        "arm_totals": audit["arm_totals"],
        "outcomes_used_for_gate": False,
        "audit_violations": audit["violations"],
        "input_sha256": {
            str(path.relative_to(ROOT)): {
                "byte_length": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (AUDIT, LEDGER, PROTOCOL, Path(__file__))
        },
    }


def _heatmap(axis: Any, payload: dict[str, Any], arm: str) -> None:
    values = np.array(
        [
            [
                payload["cells"][arm][domain]["stages"][stage]
                for stage in STAGES
            ]
            for domain in DOMAIN_ORDER
        ],
        dtype=float,
    )
    # The manuscript's own magnitude ramp rather than matplotlib's "Blues", so
    # all three heatmaps in the paper share one language.
    cmap = style.sequential_cmap()
    high = max(1.0, float(values.max()))
    image = axis.imshow(
        values,
        cmap=cmap,
        norm=colors.Normalize(vmin=0.0, vmax=high),
        aspect="auto",
    )
    del image
    axis.set_xticks(
        range(len(STAGES)),
        STAGE_LABEL,
        rotation=32,
        ha="right",
    )
    axis.set_yticks(
        range(len(DOMAIN_ORDER)),
        [DOMAIN_LABEL[domain] for domain in DOMAIN_ORDER],
    )
    axis.set_title(ARM_LABEL[arm], fontsize=style.TITLE_FONT, color=style.INK)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = int(values[row, column])
            axis.text(
                column, row, str(value), ha="center", va="center",
                color=style.cell_ink(cmap(value / high)),
                fontsize=style.SMALL_FONT,
            )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_color(style.MUTED)
        spine.set_linewidth(style.SPINE_LW)


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure = plt.figure(figsize=(style.WIDTH, 3.35))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.72))
    passive = figure.add_subplot(grid[0, 0])
    adaptive = figure.add_subplot(grid[0, 1])
    actions = figure.add_subplot(grid[0, 2])
    _heatmap(passive, payload, "retention_tracking")
    _heatmap(adaptive, payload, "adaptive_retention")

    style.style_axis(actions, grid="both")
    x = np.arange(len(DOMAIN_ORDER))
    width = 0.34
    for offset, field, color, hatch in (
        (-width / 2, ACTION_FIELDS[0], style.ADD_ON, "///"),
        (width / 2, ACTION_FIELDS[1], style.METHOD, ""),
    ):
        values = [
            payload["cells"]["adaptive_retention"][domain]["actions"][field]
            for domain in DOMAIN_ORDER
        ]
        actions.bar(
            x + offset, values, width=width, color=color, alpha=0.82,
            hatch=hatch, edgecolor=style.INK, linewidth=0.35,
            label=ACTION_LABEL[ACTION_FIELDS.index(field)].replace("\n", " "),
        )
    actions.set_xticks(
        x, [DOMAIN_LABEL[domain] for domain in DOMAIN_ORDER],
        rotation=28, ha="right",
    )
    actions.set_ylabel("adaptive action count", fontsize=style.LABEL_FONT)
    actions.set_title("Bounded actuation", fontsize=style.TITLE_FONT, color=style.INK)
    actions.legend(
        frameon=False, fontsize=style.SMALL_FONT, loc="upper left",
        handlelength=1.2, borderaxespad=0.2,
    )

    figure.suptitle(
        "Admission-to-retention gate: measured survival and bounded priority",
        fontsize=style.TITLE_FONT, color=style.INK, y=0.995,
    )
    figure.text(
        0.5, 0.952,
        "Qwen2.5-0.5B · seed 43 · 64 updates · counts are post-treatment diagnostics, not efficacy estimates.",
        ha="center", va="top", fontsize=style.SMALL_FONT, color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.085, right=0.995, top=0.82, bottom=0.24, wspace=0.34
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
