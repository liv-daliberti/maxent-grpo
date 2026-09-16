#!/usr/bin/env python3
"""Restore the historical frozen-model admission evidence in Methods.

This plot contains only the measured Level-1/Level-2 development check. The
all-model, all-level held-out grid has a separate renderer and completeness gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paper_style as style

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "paper/results/modebench_level_comparison_snapshot.json"
OUT = ROOT / "paper/figures/modebench_level_construction"
SCRIPT = Path(__file__).resolve()
DOMAINS = (
    ("graph_coloring", "Graph coloring"), ("countdown", "Countdown"),
    ("python_factors", "Python factors"), ("mathir", "MathIR"),
    ("pantry", "PantryPlan"),
)
LEVEL1_COLOR = "#475569"
LEVEL2_COLOR = "#0F766E"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(snapshot: dict, snapshot_path: Path) -> dict:
    if snapshot.get("schema") != "modebench-level-comparison-frozen-snapshot-v1":
        raise ValueError("wrong frozen Level-1/Level-2 comparison snapshot")
    reference = snapshot["reference_figure"]
    candidates = [(Path(path), sha) for path, sha in reference["sources"].items()
                  if Path(path).name == "admission_fairness_report.json"]
    if len(candidates) != 1:
        raise ValueError("construction requires one frozen admission report")
    report_path, expected = candidates[0]
    if digest(report_path) != expected:
        raise ValueError("historical admission report no longer matches its frozen hash")
    report = json.loads(report_path.read_text())
    if report["status"] != "pass" or report["decision"] != "admit_all_domains_for_treatment_training":
        raise ValueError("historical Level-2 admission did not pass")
    rows = []
    for domain, label in DOMAINS:
        cell = report["frozen_base_model_viability"][domain]["qwen-0.5b"]
        if cell["status"] != "pass" or not all(cell["checks"].values()):
            raise ValueError(f"historical admission failed for {domain}")
        rows.append({"domain": label,
                     "level1_pass8": cell["level1_pass_at_8"],
                     "level2_pass8": cell["level2_pass_at_8"]})
    if rows != reference["admission_rows"]:
        raise ValueError("development points differ from the frozen reference figure")
    return {
        "schema": "modebench-level-construction-v1",
        "model": "Qwen2.5-0.5B-Instruct",
        "population": "historical development admission; not the held-out all-level grid",
        "levels": [1, 2],
        "metric": "empirical pass@8",
        "admission_rows": rows,
        "viability_gate": report["viability_gate"],
        "protocol": report["comparable_prompt_and_response_budget"],
        "snapshot_path": str(snapshot_path.resolve()),
        "sources": {str(snapshot_path.resolve()): digest(snapshot_path),
                    str(report_path): expected},
        "renderer": {"path": str(SCRIPT), "sha256": digest(SCRIPT),
                     "style_path": str(Path(style.__file__).resolve()),
                     "style_sha256": digest(Path(style.__file__))},
    }


def render(record: dict, output: Path) -> None:
    style.apply_rcparams()
    figure = plt.figure(figsize=(style.WIDTH, 1.30))
    axis = figure.add_axes([.18, .26, .64, .66])
    axis.set_facecolor(style.PANEL)
    axis.axvspan(.10, .90, color=LEVEL2_COLOR, alpha=.06, lw=0)
    axis.grid(axis="x", color=style.GRID, lw=.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    rows = record["admission_rows"]
    for y, row in enumerate(rows):
        first, second = row["level1_pass8"], row["level2_pass8"]
        axis.plot([first, second], [y, y], color=style.GRID, lw=2, zorder=1)
        axis.scatter(first, y, s=25, facecolors="none", edgecolors=LEVEL1_COLOR, lw=1.2, zorder=3)
        axis.scatter(second, y, s=25, color=LEVEL2_COLOR, zorder=4)
    axis.set_yticks(range(5), [row["domain"] for row in rows], fontsize=7)
    axis.tick_params(axis="y", length=0)
    axis.set_ylim(4.5, -.5)
    axis.set_xlim(0, 1)
    axis.set_xticks((0, .25, .5, .75, 1), ("0", ".25", ".5", ".75", "1"), fontsize=7)
    axis.set_xlabel("Frozen base-model development pass@8", fontsize=7.4, labelpad=3)
    figure.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=LEVEL1_COLOR, label="Level 1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL2_COLOR,
               markeredgecolor=LEVEL2_COLOR, label="Level 2"),
    ], loc="center right", bbox_to_anchor=(1, .60), frameon=False, fontsize=7.2)
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def validate_record(record: dict, *, output: Path = OUT, snapshot_path: Path = SNAPSHOT) -> None:
    if Path(record["snapshot_path"]).resolve() != snapshot_path.resolve():
        raise ValueError("construction figure references an unexpected frozen snapshot")
    expected = build_record(json.loads(snapshot_path.read_text()), snapshot_path)
    if {key: value for key, value in record.items() if key != "outputs"} != expected:
        raise ValueError("construction figure record differs from frozen admission evidence")
    expected_outputs = {str(output.with_suffix(suffix).resolve()) for suffix in (".pdf", ".png")}
    if set(record.get("outputs", {})) != expected_outputs:
        raise ValueError("construction figure requires its published PDF and PNG bindings")
    for path, sha in record["outputs"].items():
        if digest(Path(path)) != sha:
            raise ValueError(f"construction figure output changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    record = build_record(json.loads(args.snapshot.read_text()), args.snapshot)
    render(record, args.output)
    record["outputs"] = {str(args.output.with_suffix(suffix).resolve()):
                         digest(args.output.with_suffix(suffix)) for suffix in (".pdf", ".png")}
    validate_record(record, output=args.output, snapshot_path=args.snapshot)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
