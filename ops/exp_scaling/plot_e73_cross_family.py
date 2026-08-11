#!/usr/bin/env python3
"""Render the Falcon3-1B cross-family replication as trajectories, not endpoints.

All five domains are terminal. The renderer still derives each domain's common
endpoint from all ten cells and refuses to draw past it, so a missing or
regressed cell cannot be hidden by carrying another seed forward. This makes
the trajectory figures a terminal counterpart to the endpoint table rather
than an interim progress view.

Form. Two rows --- breadth and accuracy --- against training pass, one column
per domain, five-seed means with the seed range as a band. Breadth is the row
that carries the claim; accuracy is directly under it so a reader can check the
band collapse and the accuracy story in one vertical glance.

Color follows the manuscript's entities: control orange for matched Dr.GRPO,
historical-treatment teal, each with its own dash pattern so identity never
rests on hue alone.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_OPS = Path(__file__).resolve().parents[1]
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))
import paper_style as style  # noqa: E402

STEPS_PER_PASS = 384
TERMINAL_STEP = 4608
MODEL_TAG = "falcon3_1b_instruct"

DOMAINS = (
    ("graph_coloring", "Graph coloring"),
    ("countdown", "Countdown"),
    ("python_factors", "Python factors"),
    ("mathir", "MathIR action menu"),
    ("pantry_plan", "PantryPlan"),
)

ARMS = (
    ("drgrpo", "matched Dr.GRPO", style.CONTROL),
    ("xgrpo", "historical treatment", style.METHOD),
)
ARM_VARIANT = {
    "drgrpo": ("grpo_compute_matched", "grpo"),
    "xgrpo": (
        "verified_first_global_replay_canonical",
        "verified_first_global_replay_canonical",
    ),
}

METRICS = (
    (
        "distinct_at_8",
        "eval/multi_answer/sampled_distinct_correct_at_8",
        "breadth  (# distinct@8)",
    ),
    ("pass_at_8", "eval/multi_answer/sampled_any_correct_at_8", "accuracy  (pass@8)"),
)

# The band the manuscript's claim is about: four metrics that coincide when a
# policy has collapsed to one execution and separate when it has not. Two of
# them cannot show that, so the full grid carries all four in reporting order.
BAND_METRICS = (
    ("greedy", "eval/multi_answer/accuracy", "greedy  pass@1"),
    ("mean_at_8", "eval/multi_answer/sampled_mean_at_8", "mean@8"),
    ("pass_at_8", "eval/multi_answer/sampled_any_correct_at_8", "pass@8"),
    (
        "distinct_at_8",
        "eval/multi_answer/sampled_distinct_correct_at_8",
        "# distinct@8",
    ),
    # Derived, and the row that actually shows the band closing: four metrics
    # on four independent scales cannot be seen to coincide, but their ratio
    # can. One means every successful draw returned the same execution.
    ("modes_per_success", None, "modes-per-success"),
)

PREFIXES = {
    "graph_coloring": ("gce73_falcon3_1b_12pass",),
    "countdown": ("cde73_falcon3_1b_12pass",),
    "python_factors": ("pye73_falcon3_1b_12pass_r1", "pye73_falcon3_1b_12pass"),
    "mathir": ("mie73_falcon3_1b_12pass_r1", "mie73_falcon3_1b_12pass"),
    "pantry_plan": ("ppe73_falcon3_1b_12pass",),
}

style.apply_rcparams()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def series(
    root: Path, domain: str, arm: str, metrics=METRICS
) -> dict[str, dict[int, list[float]]]:
    """Per-step values for one domain and arm, pooled across seeds."""
    variant, stamp_arm = ARM_VARIANT[arm]
    out: dict[str, dict[int, list[float]]] = {name: {} for name, _, _ in metrics}
    for prefix in PREFIXES[domain]:
        pattern = (
            f"var/data/xdr_{MODEL_TAG}_{variant}_{prefix}_{stamp_arm}_s4[3-7]"
        )
        directories = sorted(root.glob(pattern))
        if not directories:
            continue
        for run_dir in directories:
            for path in glob.glob(str(run_dir / "debug_job*" / "train_metrics.jsonl")):
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        step = int(record.get("misc/global_step", -1))
                        for name, key, _ in metrics:
                            if key is not None and key in record:
                                out[name].setdefault(step, []).append(
                                    float(record[key])
                                )
        # The relaunched cohort supersedes the original wherever it exists;
        # never pool the two.
        break
    return out


def render(
    root: Path, summary: dict[str, Any], output: Path, metrics=METRICS
) -> dict[str, Any]:
    # A derived row is computed from logged ones, so those must be gathered
    # whether or not they are themselves drawn.
    collect_metrics = tuple(m for m in metrics if m[1] is not None)
    figure, axes = plt.subplots(
        len(metrics),
        len(DOMAINS),
        figsize=(style.WIDTH, style.panel_height(len(metrics))),
        constrained_layout=True,
    )
    drawn: list[dict[str, Any]] = []

    for column, (domain, title) in enumerate(DOMAINS):
        entry = summary["domains"].get(domain, {})
        endpoint = int(entry.get("common_endpoint") or 0)
        complete = bool(entry.get("complete"))
        collected = {
            arm: series(root, domain, arm, collect_metrics) for arm, _, _ in ARMS
        }

        for row, (metric, _, ylabel) in enumerate(metrics):
            axis = axes[row][column]
            style.style_axis(axis, title=title if row == 0 else None)

            for arm, label, color in ARMS:
                if metric == "modes_per_success":
                    distinct = collected[arm]["distinct_at_8"]
                    passed = collected[arm]["pass_at_8"]
                    points = {
                        step: [
                            statistics.fmean(distinct[step])
                            / statistics.fmean(passed[step])
                        ]
                        for step in distinct
                        if step in passed and statistics.fmean(passed[step]) > 0
                    }
                else:
                    points = collected[arm][metric]
                # Never draw past the depth every cell of this domain reached.
                steps = sorted(s for s in points if 0 <= s <= endpoint)
                if not steps:
                    continue
                passes = [s / STEPS_PER_PASS for s in steps]
                means = [statistics.fmean(points[s]) for s in steps]
                axis.fill_between(
                    passes,
                    [min(points[s]) for s in steps],
                    [max(points[s]) for s in steps],
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                    zorder=1,
                )
                axis.plot(
                    passes,
                    means,
                    color=color,
                    linewidth=style.MEAN_LW,
                    linestyle=style.ARM_DASH[color],
                    zorder=3,
                    label=label,
                )
                drawn.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "arm": arm,
                        "common_endpoint": endpoint,
                        "terminal_mean": means[-1],
                    }
                )

            axis.set_xlim(0, 12)
            axis.set_xticks([0, 3, 6, 9, 12])
            if row == 0 and not complete:
                # State the depth on the panel. A caption-level qualifier is too
                # far from the axis it qualifies.
                axis.annotate(
                    f"to pass {endpoint / STEPS_PER_PASS:.3g}",
                    xy=(0.97, 0.06),
                    xycoords="axes fraction",
                    ha="right",
                    fontsize=style.SMALL_FONT,
                    color=style.MUTED,
                )
            if endpoint < TERMINAL_STEP:
                axis.axvspan(
                    endpoint / STEPS_PER_PASS,
                    12,
                    color=style.GRID,
                    alpha=0.45,
                    linewidth=0,
                    zorder=0,
                )
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=style.LABEL_FONT, labelpad=2)

    figure.supxlabel("training pass", fontsize=style.FONT)
    handles, labels = axes[0][0].get_legend_handles_labels()
    style.bottom_legend(figure, handles, labels, y=-0.055)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return {"series": drawn}


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "var" / "artifacts" / "e73_falcon_cross_family_summary_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "paper" / "figures" / "e73_cross_family_curves.pdf",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    if summary.get("lemma_violations"):
        raise SystemExit(
            "refusing to render: the summary reports metric-ordering violations"
        )

    drawn = render(root, summary, args.output)
    # The full band as well: four metrics coinciding is the claim, and the
    # two-row figure cannot show a band closing.
    grid_output = args.output.with_name(
        args.output.name.replace("_curves", "_band")
    )
    render(root, summary, grid_output, BAND_METRICS)
    print(f"[e73-curves] wrote {grid_output}")
    provenance = {
        "schema": "e73_cross_family_curves_figure_v1",
        "figure": str(args.output),
        "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "common_endpoints": {
            domain: entry.get("common_endpoint")
            for domain, entry in summary["domains"].items()
        },
        **drawn,
    }
    path = root / "var" / "artifacts" / "e73_cross_family_curves_provenance.json"
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(provenance, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, path)
    print(f"[e73-curves] wrote {args.output} and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
