#!/usr/bin/env python3
"""Render the complete outcome-blind superseded group-centered mechanism grid."""

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

DEFAULT_INPUT = ROOT / "var/artifacts/e106_python_lambda_normalization_combined_gate.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/e106_group_centered_mechanism_diagnostic"
SCALES = ("qwen05b", "falcon1b", "qwen3b")
SCALE_LABELS = {"qwen05b": "Qwen 0.5B", "falcon1b": "Falcon 1B", "qwen3b": "Qwen 3B"}
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
DOMAIN_LABELS = {
    "graph_coloring": "Graph", "countdown": "Countdown", "python_factors": "Python",
    "mathir": "MathIR", "pantry_plan": "Pantry",
}
MEAN_TOLERANCE = 1e-8
METRICS = (
    ("semantic_eligible_fraction_max", "Semantic-eligible fraction (max)", 1.0, ".2f"),
    ("support_at_least_two_prompt_fraction_max", "Prompts with at least two verified modes (max)", 1.0, ".2f"),
    ("semantic_rms_max", "Centered semantic advantage RMS (max)", 0.1, ".3f"),
    ("semantic_effective_mean_abs_max", "Absolute centered semantic mean (max)", MEAN_TOLERANCE, ".1e"),
    ("replay_gradient_l2_max", "Applied replay score-gradient L2 (max)", 0.1, ".3f"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def plot_payload(audit: dict[str, Any], *, input_path: Path) -> dict[str, Any]:
    if audit.get("schema") != "e106_python_lambda_normalization_combined_gate_v1":
        raise RuntimeError("unexpected E106 combined-gate schema")
    if audit.get("complete") is not True or audit.get("passed") is not True:
        raise RuntimeError("E106 mechanism diagnostic requires a passed 15-cell gate")
    if audit.get("pointmaze") != "excluded":
        raise RuntimeError("E106 mechanism diagnostic requires PointMaze exclusion")
    if audit.get("mechanism_gate_used_outcome_metrics") is not False or audit.get("post_update_outcome_metrics_inspected") is not False:
        raise RuntimeError("E106 mechanism diagnostic requires outcome blinding")
    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in audit.get("runs", []):
        scale, domain = str(item.get("scale")), str(item.get("domain"))
        key = (scale, domain)
        if scale not in SCALES or domain not in DOMAINS or key in seen:
            raise RuntimeError(f"unexpected or duplicate E106 cell: {key}")
        if item.get("runtime_complete") is not True:
            raise RuntimeError(f"E106 cell lacks receipt-backed completion: {key}")
        report = item.get("report")
        if not isinstance(report, dict):
            raise RuntimeError(f"E106 cell lacks mechanism report: {key}")
        values = {metric: finite(report.get(metric), where=f"{scale}/{domain}/{metric}") for metric, _title, _vmax, _fmt in METRICS}
        invariants = {
            "v6_group_centered_active": finite(report.get("group_centered_active_min"), where=f"{key}/group_centered_active_min") >= 1.0,
            "legacy_estimator_inactive": finite(report.get("legacy_active_max"), where=f"{key}/legacy_active_max") == 0.0,
            "rms_controller_inactive": finite(report.get("controller_active_max"), where=f"{key}/controller_active_max") == 0.0,
            "centered_mean_within_tolerance": values["semantic_effective_mean_abs_max"] <= MEAN_TOLERANCE,
            "replay_actuated": values["replay_gradient_l2_max"] > 0.0,
        }
        if not all(invariants.values()):
            failed = [name for name, passed in invariants.items() if not passed]
            raise RuntimeError(f"E106 cell violates mechanism invariants {key}: {failed}")
        cells.append({
            "scale": scale, "model": SCALE_LABELS[scale], "domain": domain,
            "source": item.get("source"), "job_id": int(item["job_id"]),
            "target_steps": int(item["target_steps"]),
            "completion_receipt_step": int(item["completion_receipt_step"]),
            "scheduler_state": str(item["scheduler_state"]), "metrics": values,
            "invariants": invariants, "zero_semantic_pressure": values["semantic_rms_max"] == 0.0,
        })
        seen.add(key)
    expected = {(scale, domain) for scale in SCALES for domain in DOMAINS}
    if seen != expected or len(cells) != 15:
        raise RuntimeError("E106 mechanism diagnostic requires the exact 3-by-5 grid")
    cells.sort(key=lambda cell: (SCALES.index(cell["scale"]), DOMAINS.index(cell["domain"])))
    zero = [{"scale": cell["scale"], "domain": cell["domain"], "job_id": cell["job_id"]} for cell in cells if cell["zero_semantic_pressure"]]
    return {
        "schema": "paper-e106-group-centered-mechanism-diagnostic-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(), "pointmaze": "excluded",
        "outcome_metrics_used": False, "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path), "plotter": str(Path(__file__).resolve()),
        "plotter_sha256": sha256(Path(__file__)), "scale_order": list(SCALES),
        "domain_order": list(DOMAINS), "metric_order": [metric for metric, *_ in METRICS],
        "mean_tolerance": MEAN_TOLERANCE, "cells": cells,
        "zero_semantic_pressure_cells": zero,
        "invariant_counts": {name: sum(bool(cell["invariants"][name]) for cell in cells) for name in cells[0]["invariants"]},
    }


def _matrix(payload: dict[str, Any], metric: str) -> np.ndarray:
    indexed = {(cell["scale"], cell["domain"]): cell for cell in payload["cells"]}
    return np.asarray([[indexed[(scale, domain)]["metrics"][metric] for domain in DOMAINS] for scale in SCALES], dtype=float)


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(2, 3, figsize=(style.WIDTH, 5.0), squeeze=False)
    flat = list(axes.flat)
    zero = {(cell["scale"], cell["domain"]) for cell in payload["cells"] if cell["zero_semantic_pressure"]}
    for axis, (metric, title, vmax, number_format) in zip(flat, METRICS):
        values = _matrix(payload, metric)
        cmap = style.sequential_cmap()
        image = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        axis.set_title(title, fontsize=7.2)
        axis.set_xticks(range(len(DOMAINS)), [DOMAIN_LABELS[d] for d in DOMAINS], rotation=30, ha="right")
        axis.set_yticks(range(len(SCALES)), [SCALE_LABELS[s] for s in SCALES])
        for row, scale in enumerate(SCALES):
            for column, domain in enumerate(DOMAINS):
                value = values[row, column]
                if metric != "semantic_effective_mean_abs_max":
                    axis.text(column, row, format(value, number_format), ha="center", va="center", fontsize=5.6, color=style.cell_ink(cmap(value / vmax)))
                if metric == "semantic_rms_max" and (scale, domain) in zero:
                    axis.add_patch(Rectangle((column - 0.48, row - 0.48), 0.96, 0.96, fill=False, edgecolor=style.ADD_ON, linewidth=1.2))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
    summary = flat[-1]
    summary.axis("off")
    counts, zero_cells = payload["invariant_counts"], payload["zero_semantic_pressure_cells"]
    summary.text(0.02, 0.98, "Diagnostic invariants\n\n" f"group-centered estimator active: {counts['v6_group_centered_active']}/15\n" f"legacy estimator inactive: {counts['legacy_estimator_inactive']}/15\n" f"RMS controller inactive: {counts['rms_controller_inactive']}/15\n" f"centered mean <= 1e-8: {counts['centered_mean_within_tolerance']}/15\n" f"verified replay actuated: {counts['replay_actuated']}/15\n\n" f"zero semantic-RMS cells: {len(zero_cells)}\n" "(outlined; retained, not filtered)\n\n" "Mechanism-only diagnostic;\nno post-update outcomes used.", transform=summary.transAxes, ha="left", va="top", fontsize=7.0, linespacing=1.25)
    figure.suptitle("Group-centered Semantic-MaxEnt + Re:Dr: superseded diagnostic", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = plot_payload(json.loads(args.input.read_text(encoding="utf-8")), input_path=args.input)
    render(payload, args.output)
    print(f"wrote {args.output.with_suffix('.pdf')}, {args.output.with_suffix('.png')}, and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
