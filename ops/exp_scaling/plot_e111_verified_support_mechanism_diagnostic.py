#!/usr/bin/env python3
"""Render the completed verified-support mechanism validation."""

from __future__ import annotations

import argparse
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
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402


DEFAULT_INPUT = ROOT / "var/artifacts/e111_verified_support_discovery_mechanism_gate_audit.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/e111_verified_support_mechanism_diagnostic"
SCALES = ("qwen05b", "falcon1b", "qwen3b")
SCALE_LABELS = {"qwen05b": "Qwen 0.5B", "falcon1b": "Falcon 1B", "qwen3b": "Qwen 3B"}
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
DOMAIN_LABELS = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "Pantry",
}
METRICS = (
    ("semantic_eligible_fraction_max", "Semantic-eligible fraction (max)", ".2f"),
    ("support_at_least_two_eligible_fraction_max", "Eligible support with >=2 modes (max)", ".2f"),
    ("semantic_rms_max", "Verified-support semantic RMS (max)", ".3f"),
    ("proposal_cumulative_admissions", "Proposal admissions", ".0f"),
    ("replay_gradient_l2_max", "Applied replay gradient L2 (max)", ".3f"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{where}: expected finite numeric value, got {value!r}")
    return float(value)


def build(audit: dict[str, Any], *, input_path: Path) -> dict[str, Any]:
    if audit.get("schema") != "e111_verified_support_discovery_mechanism_gate_audit_v1":
        raise RuntimeError("unexpected verified-support validation schema")
    if audit.get("terminal") is not True or audit.get("passed") is not True or audit.get("violations") != []:
        raise RuntimeError("diagnostic requires a complete, passing, violation-free validation")
    if audit.get("pointmaze") != "excluded" or audit.get("outcomes_used_for_gate") is not False:
        raise RuntimeError("diagnostic requires PointMaze exclusion and no task-endpoint input")

    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for run in audit.get("runs", []):
        scale, domain = str(run.get("scale")), str(run.get("domain"))
        key = (scale, domain)
        if scale not in SCALES or domain not in DOMAINS or key in seen:
            raise RuntimeError(f"unexpected or duplicate E111 cell {key}")
        if run.get("scheduler_state") != "COMPLETED" or run.get("violations") != []:
            raise RuntimeError(f"E111 cell is not terminal and clean: {key}")
        report = run.get("report")
        if not isinstance(report, dict):
            raise RuntimeError(f"E111 cell lacks mechanism report: {key}")
        values = {
            metric: _finite(report.get(metric), where=f"{scale}/{domain}/{metric}")
            for metric, _title, _format in METRICS
        }
        invariants = {
            "v7_active": _finite(report.get("v7_active_min"), where=f"{key}/v7") >= 1.0,
            "v5_inactive": _finite(report.get("v5_active_max"), where=f"{key}/v5") == 0.0,
            "v6_inactive": _finite(report.get("v6_active_max"), where=f"{key}/v6") == 0.0,
            "controller_inactive": _finite(report.get("controller_active_max"), where=f"{key}/controller") == 0.0,
            "proposal_not_in_ppo": _finite(report.get("proposal_rows_to_ppo_max_abs"), where=f"{key}/proposal_rows_to_ppo") == 0.0,
            "proposal_outcome_neutral": _finite(report.get("proposal_objective_outcome_delta_max_abs"), where=f"{key}/proposal_outcome") == 0.0,
            "replay_actuated": values["replay_gradient_l2_max"] > 0.0,
        }
        if not all(invariants.values()):
            failed = [name for name, passed in invariants.items() if not passed]
            raise RuntimeError(f"E111 cell violates invariants {key}: {failed}")
        cells.append(
            {
                "scale": scale,
                "model": SCALE_LABELS[scale],
                "domain": domain,
                "seed": int(run["seed"]),
                "job_id": int(run["job_id"]),
                "effective_job_id": int(run["effective_job_id"]),
                "metrics": values,
                "completed_causal_chain": bool(run["completed_causal_chain"]),
                "invariants": invariants,
            }
        )
        seen.add(key)
    expected = {(scale, domain) for scale in SCALES for domain in DOMAINS}
    if seen != expected or len(cells) != 15:
        raise RuntimeError("E111 diagnostic requires the exact 3-by-5 grid")
    cells.sort(key=lambda cell: (SCALES.index(cell["scale"]), DOMAINS.index(cell["domain"])))
    return {
        "schema": "paper-e111-verified-support-mechanism-diagnostic-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pointmaze": "excluded",
        "outcome_metrics_used_for_gate": False,
        "status": "complete 15-cell mechanism validation; no efficacy result",
        "input": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "plotter": str(Path(__file__).resolve()),
        "plotter_sha256": _sha256(Path(__file__)),
        "scale_order": list(SCALES),
        "domain_order": list(DOMAINS),
        "metric_order": [metric for metric, _title, _format in METRICS],
        "cells": cells,
        "completed_causal_chain_cells": sum(bool(cell["completed_causal_chain"]) for cell in cells),
        "invariant_counts": {
            name: sum(bool(cell["invariants"][name]) for cell in cells)
            for name in cells[0]["invariants"]
        },
    }


def _matrix(payload: dict[str, Any], metric: str) -> np.ndarray:
    indexed = {(cell["scale"], cell["domain"]): cell for cell in payload["cells"]}
    return np.asarray(
        [[indexed[(scale, domain)]["metrics"][metric] for domain in DOMAINS] for scale in SCALES],
        dtype=float,
    )


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(2, 3, figsize=(style.WIDTH, 5.0), squeeze=False)
    flat = list(axes.flat)
    chain = {
        (cell["scale"], cell["domain"])
        for cell in payload["cells"]
        if cell["completed_causal_chain"]
    }
    for axis, (metric, title, number_format) in zip(flat, METRICS):
        values = _matrix(payload, metric)
        vmax = max(float(values.max()), 1e-12)
        cmap = style.sequential_cmap()
        image = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        axis.set_title(title, fontsize=7.2)
        axis.set_xticks(range(len(DOMAINS)), [DOMAIN_LABELS[d] for d in DOMAINS], rotation=30, ha="right")
        axis.set_yticks(range(len(SCALES)), [SCALE_LABELS[s] for s in SCALES])
        for row, scale in enumerate(SCALES):
            for column, domain in enumerate(DOMAINS):
                value = values[row, column]
                axis.text(
                    column,
                    row,
                    format(value, number_format),
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    color=style.cell_ink(cmap(value / vmax)),
                )
                if metric == "semantic_rms_max" and (scale, domain) in chain:
                    axis.add_patch(
                        Rectangle(
                            (column - 0.48, row - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor=style.ADD_ON,
                            linewidth=1.2,
                        )
                    )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)

    summary = flat[-1]
    summary.axis("off")
    counts = payload["invariant_counts"]
    summary.text(
        0.02,
        0.98,
        "Validation invariants\n\n"
        f"verified-support estimator active: {counts['v7_active']}/15\n"
        f"prior estimators inactive: {counts['v5_inactive']}/15, {counts['v6_inactive']}/15\n"
        f"controller inactive: {counts['controller_inactive']}/15\n"
        f"proposal rows excluded from PPO: {counts['proposal_not_in_ppo']}/15\n"
        f"proposal objective outcome-neutral: {counts['proposal_outcome_neutral']}/15\n"
        f"verified replay actuated: {counts['replay_actuated']}/15\n\n"
        f"complete discovery-to-pressure chain: {payload['completed_causal_chain_cells']}/15\n"
        "(outlined in the RMS panel)\n\n"
        "Mechanism validation uses no task endpoints.",
        transform=summary.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        linespacing=1.23,
    )
    figure.suptitle("Verified-support Semantic-MaxEnt + Re:Dr.GRPO: mechanism validation", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(json.loads(args.input.read_text(encoding="utf-8")), input_path=args.input)
    render(payload, args.output)
    print(f"wrote {args.output.with_suffix('.pdf')}, {args.output.with_suffix('.png')}, and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
