#!/usr/bin/env python3
"""Render the B3a training curves: three arms, five domains, two metrics.

The endpoint table says where the arms land; it cannot say when they part. That
matters here, because the claim is that removing the replay gradient removes the
mechanism rather than merely slowing it: if B3a tracked xGRPO for several passes
before falling away, "slower" would be the better description. The trajectory is
what distinguishes the two readings, so it is plotted.

Form. Breadth against training pass, one panel per domain, three arms per panel,
means over five seeds with the seed range as a band. Arms are small multiples of
one comparison, so panels share a metric row and each row keeps its own y scale
across domains only where the scales are commensurate --- they are not, so each
panel is scaled to its own domain and the shared quantity is the shape.

Color. The manuscript's established pair carries Dr.GRPO and xGRPO; B3a takes a
third hue. The method teal sits under the chroma floor and its tritan separation
from the control orange is weak (validator: chroma FAIL, CVD PASS on protan and
deutan), so identity is never colour alone: each arm also has its own dash
pattern and a direct label on the panel where the curves are furthest apart.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#19324A"
MUTED = "#607487"
GRID = "#D8E2EA"
WHITE = "#FFFFFF"
CONTROL = "#C76A3A"
METHOD = "#087F8C"
ABLATION = "#6C5CE7"

STEPS_PER_PASS = 384
TERMINAL_STEP = 4608

DOMAINS = (
    ("graph_coloring", "Graph coloring", "gce71_scale384_05b_12pass", "gc"),
    ("countdown", "Countdown", "cde70_clean_stage_a_05b_12pass", "cd"),
    ("python_factors", "Python factors", "pye70_clean_stage_a_05b_12pass", "py"),
    ("mathir", "MathIR action menu", "mie70_clean_stage_a_05b_12pass", "mi"),
    ("pantry_plan", "PantryPlan", "ppe71_scale384_05b_12pass", "pp"),
)

ARMS = (
    ("drgrpo", "matched Dr.GRPO", CONTROL, (0, (5, 1.6))),
    ("b3a", "B3a (replay gradient removed)", ABLATION, (0, (1.6, 1.4))),
    ("xgrpo", "xGRPO", METHOD, "solid"),
)

METRICS = (
    ("distinct8", "eval/multi_answer/sampled_distinct_correct_at_8", "breadth  (# distinct@8)"),
    ("pass8", "eval/multi_answer/sampled_any_correct_at_8", "pass@8"),
)

B3A_GLOB = "var/data/xdr_qwen25_0p5b_instruct_verified_first_replay_gradient_ablation_*"
B3A_NAME = re.compile(r"_(gc|cd|py|mi|pp)e7[01]_.*_b3a_s(?P<seed>\d+)$")

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
    }
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def reference_series(root: Path, curve: str) -> dict[tuple[str, str], dict[int, list[float]]]:
    """Per-arm, per-metric trajectories from a frozen scaling curve."""
    rows = json.loads((root / "var" / "artifacts" / curve).read_text())
    label_to_arm = {"grpo": "drgrpo", "verified_first_global_replay_canonical": "xgrpo"}
    out: dict[tuple[str, str], dict[int, list[float]]] = {}
    for row in rows:
        if row.get("split") != "multi_answer":
            continue
        arm = label_to_arm.get(str(row.get("arm")))
        if arm is None:
            continue
        for metric, _, _ in METRICS:
            value = row.get(metric)
            if value is None:
                continue
            out.setdefault((arm, metric), {}).setdefault(int(row["step"]), []).append(
                float(value)
            )
    return out


def b3a_series(root: Path, prefix: str) -> dict[tuple[str, str], dict[int, list[float]]]:
    out: dict[tuple[str, str], dict[int, list[float]]] = {}
    for run_dir in sorted(root.glob(B3A_GLOB.replace("var/data/", "var/data/"))):
        match = B3A_NAME.search(run_dir.name)
        if match is None or match.group(1) != prefix:
            continue
        for path in glob.glob(str(run_dir / "debug_job*" / "train_metrics.jsonl")):
            for line in open(path, encoding="utf-8"):
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                step = int(record.get("misc/global_step", -1))
                for metric, key, _ in METRICS:
                    if key in record:
                        out.setdefault(("b3a", metric), {}).setdefault(step, []).append(
                            float(record[key])
                        )
    return out


def render(root: Path, output: Path) -> dict[str, Any]:
    figure, axes = plt.subplots(
        len(METRICS), len(DOMAINS), figsize=(7.35, 3.5), constrained_layout=True
    )
    drawn: list[dict[str, Any]] = []

    for column, (domain, title, curve, prefix) in enumerate(DOMAINS):
        series = reference_series(root, f"{curve}_scaling_curve.json")
        series.update(b3a_series(root, prefix))

        for row, (metric, _, ylabel) in enumerate(METRICS):
            axis = axes[row][column]
            axis.grid(True, color=GRID, linewidth=0.55, zorder=0)
            axis.set_axisbelow(True)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            if row == 0:
                axis.set_title(title, fontsize=7.6, color=INK, pad=3)

            for arm, label, color, dash in ARMS:
                points = series.get((arm, metric))
                if not points:
                    continue
                steps = sorted(step for step in points if 0 <= step <= TERMINAL_STEP)
                passes = [step / STEPS_PER_PASS for step in steps]
                means = [statistics.fmean(points[step]) for step in steps]
                lows = [min(points[step]) for step in steps]
                highs = [max(points[step]) for step in steps]
                axis.fill_between(
                    passes, lows, highs, color=color, alpha=0.13, linewidth=0, zorder=1
                )
                axis.plot(
                    passes,
                    means,
                    color=color,
                    linewidth=1.35,
                    linestyle=dash,
                    zorder=3,
                    label=label,
                )
                drawn.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "arm": arm,
                        "n_seeds_terminal": len(points.get(TERMINAL_STEP, [])),
                        "terminal_mean": means[-1] if means else None,
                    }
                )

            axis.set_xlim(0, 12)
            axis.set_xticks([0, 3, 6, 9, 12])
            axis.tick_params(length=2.0, width=0.55, pad=1.5)
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=7.0, labelpad=2)

    figure.supxlabel("training pass", fontsize=7.2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=7.2,
        bbox_to_anchor=(0.5, -0.055),
        handlelength=2.6,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return {"series": drawn}


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper" / "figures" / "e72_b3a_curves.pdf",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "var" / "artifacts" / "e72_b3a_summary.json",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    reportable = summary.get("reportable_domains", [])
    if len(reportable) != len(DOMAINS) and not args.allow_incomplete:
        raise SystemExit(
            f"refusing to render: {len(reportable)}/{len(DOMAINS)} domains reportable"
        )

    drawn = render(root, args.output)
    provenance = {
        "schema": "e72_b3a_curves_figure_v1",
        "figure": str(args.output),
        "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "reportable_domains": reportable,
        **drawn,
    }
    path = root / "var" / "artifacts" / "e72_b3a_curves_figure_provenance.json"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(provenance, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, path)
    print(f"[e72-b3a-curves] wrote {args.output} and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
