#!/usr/bin/env python3
"""Render the primary four-method training histories from one frozen snapshot.

The Level-1 plots have one physical row per model scale. Accuracy and mode
count are separate figures so that every domain retains a readable axis.
Level 2 uses the same methods and identities, with one row per metric.
Incomplete registered checkpoints break both the line and its seed-range
band; they never shrink a fixed paired cohort or create an interpolated point.

    python ops/exp_scaling/plot_paper_training_curves.py \
        --snapshot paper/results/training_curve_snapshot_20260911.json \
        --output-dir paper/figures
"""

from __future__ import annotations

import argparse
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
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

DEFAULT_SNAPSHOT = ROOT / "paper/results/training_curve_snapshot_20260911.json"
SNAPSHOT_SCHEMA = "training-curve-frozen-snapshot-v1"
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
DOMAIN_LABELS = ("Graph", "Countdown", "Python", "MathIR", "Pantry")
SCALES = ("qwen05b", "falcon1b", "qwen3b")
SCALE_LABELS = {
    "qwen05b": "Qwen2.5-0.5B", "falcon1b": "Falcon3-1B", "qwen3b": "Qwen2.5-3B",
}
METRICS = {"pass8": "Accuracy (pass@8)", "distinct8": "Correct modes (distinct@8)"}
# Match the main factorial figure, including its objective-specific markers.
METHODS = {
    "drgrpo": {"label": "Dr.GRPO", "color": style.CONTROL,
               "marker": "o", "fill": "none", "dash": (0, (4, 1.7))},
    "replay_drgrpo": {"label": "Re:Dr.GRPO", "color": style.ADAPTIVE,
                      "marker": "o", "fill": style.ADAPTIVE, "dash": "-"},
    "maxrl": {"label": "MaxRL", "color": style.COMPARATOR,
              "marker": "s", "fill": "none", "dash": (0, (4, 1.7))},
    "replay_maxrl": {"label": "Re:MaxRL", "color": style.ABLATION,
                     "marker": "s", "fill": style.ABLATION, "dash": "-"},
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def contiguous_segments(
    points: list[dict[str, Any]], registered_steps: list[int],
) -> list[list[dict[str, Any]]]:
    """Split at missing registered checkpoints, including explicitly null ones.

    Callers select usable points first. Adjacency is defined by the registered
    evaluation grid rather than by the distance between observed points.
    """
    positions = {step: index for index, step in enumerate(registered_steps)}
    if len(positions) != len(registered_steps) or registered_steps != sorted(registered_steps):
        raise ValueError("registered checkpoints must be unique and increasing")
    by_step: dict[int, dict[str, Any]] = {}
    for point in points:
        step = point["step"]
        if step not in positions or step in by_step:
            raise ValueError(f"unexpected or duplicate checkpoint: {step}")
        by_step[step] = point
    segments: list[list[dict[str, Any]]] = []
    previous = -2
    for step in sorted(by_step):
        position = positions[step]
        if position != previous + 1:
            segments.append([])
        segments[-1].append(by_step[step])
        previous = position
    return segments


def selected_methods(panel: dict[str, Any]) -> dict[str, tuple[dict[str, Any], bool]]:
    """Keep main-paper paired cohorts; expose an unpaired arm only if absent."""
    selected = {}
    for method in METHODS:
        primary = panel.get("methods", {}).get(method, {})
        if primary.get("cohort_n", 0):
            selected[method] = (primary, False)
            continue
        independent = panel.get("supplementary_methods", {}).get(method, {})
        if independent.get("cohort_n", 0):
            selected[method] = (independent, True)
    # Paired comparisons use the same registered checkpoints as well as the
    # same seeds. Preserve raw per-method observations in the input snapshot.
    for left, right in (("drgrpo", "replay_drgrpo"), ("maxrl", "replay_maxrl")):
        if left not in selected or right not in selected:
            continue
        left_record, left_partial = selected[left]
        right_record, right_partial = selected[right]
        if left_partial or right_partial:
            continue
        if set(left_record["cohort_seeds"]) != set(right_record["cohort_seeds"]):
            raise ValueError("primary objective pair must share its fixed seed cohort")
        shared_steps = sorted({point["step"] for point in left_record["points"] if point["complete"]}
                              & {point["step"] for point in right_record["points"] if point["complete"]})
        for method, record in ((left, left_record), (right, right_record)):
            selected[method] = ({**record, "paired_checkpoint_steps": shared_steps,
                                 "points": [point if point["step"] in shared_steps else
                                            {**point, "complete": False, "mean": {metric: None for metric in METRICS}}
                                            for point in record["points"]]}, False)
    return selected


def cohort_note(panel: dict[str, Any]) -> str:
    """Every panel states the sample sizes without implying unpaired pairing."""
    selected = selected_methods(panel)
    notes = []
    for label, methods in (("Dr", ("drgrpo", "replay_drgrpo")),
                           ("Max", ("maxrl", "replay_maxrl"))):
        counts = [selected.get(method, ({"cohort_n": 0}, False))[0]["cohort_n"]
                  for method in methods]
        partial = any(selected.get(method, ({}, False))[1] for method in methods)
        paired_seeds = [selected.get(method, ({"cohort_seeds": []}, False))[0]["cohort_seeds"]
                        for method in methods]
        if not partial and counts[0] and paired_seeds[0] == paired_seeds[1]:
            notes.append(f"{label} pair n={counts[0]}")
        else:
            notes.append(f"{label} histories\nn={counts[0]}/{counts[1]}†" if partial else
                         f"{label} n={counts[0]}/{counts[1]}")
    return "\n".join(notes)


def primary_points(record: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    """Accept complete, fixed-cohort means and compute descriptive seed ranges."""
    seeds = list(map(str, record["cohort_seeds"]))
    if len(set(seeds)) != record["cohort_n"] or not seeds:
        raise ValueError("primary cohort size must match its nonempty seed list")
    result = []
    for point in record["points"]:
        mean = point.get("mean", {}).get(metric)
        if not point.get("complete") or mean is None:
            continue
        values = [point.get("per_seed", {}).get(seed, {}).get(metric) for seed in seeds]
        if not all(_finite(value) for value in [mean, *values]):
            raise ValueError("a complete checkpoint contains missing or invalid cohort metrics")
        if not math.isclose(sum(values) / len(values), mean, abs_tol=1e-12):
            raise ValueError("checkpoint mean does not describe its fixed cohort")
        result.append({"step": point["step"], "training_pass": point["training_pass"],
                       "mean": mean, "range": [min(values), max(values)],
                       "per_seed": {seed: value for seed, value in zip(seeds, values)}})
    return result


def individual_points(record: dict[str, Any], seed: str, metric: str) -> list[dict[str, Any]]:
    return [{"step": point["step"], "training_pass": point["training_pass"],
             "value": point["per_seed"][seed][metric]}
            for point in record["points"]
            if _finite(point.get("per_seed", {}).get(seed, {}).get(metric))]


def draw_method(
    axis: plt.Axes, method: str, record: dict[str, Any], partial: bool,
    metric: str, registered_steps: list[int],
) -> dict[str, Any]:
    spec = METHODS[method]
    common = dict(color=spec["color"], linestyle=spec["dash"], marker=spec["marker"],
                  markerfacecolor=spec["fill"], markeredgewidth=.7, markersize=2.7)
    audit: dict[str, Any] = {"cohort_n": record["cohort_n"],
                            "cohort_seeds": record["cohort_seeds"],
                            "cohort_policy": record["cohort_policy"], "partial_unpaired": partial}
    if partial:
        per_seed = {}
        for seed in map(str, record["cohort_seeds"]):
            points = individual_points(record, seed, metric)
            segments = contiguous_segments(points, registered_steps)
            per_seed[seed] = {"points": points,
                              "segments": [[point["step"] for point in segment] for segment in segments]}
            for segment in segments:
                line, = axis.plot([point["training_pass"] for point in segment],
                                  [point["value"] for point in segment],
                                  lw=.95, alpha=.8 if record["cohort_n"] == 1 else .5,
                                  zorder=3, **common)
                line.set_gid(f"{method}:partial:{seed}")
        audit.update(per_seed=per_seed, band="none: independent partial histories")
        return audit

    points = primary_points(record, metric)
    audit["paired_checkpoint_steps"] = record.get("paired_checkpoint_steps", [point["step"] for point in points])
    segments = contiguous_segments(points, registered_steps)
    for segment in segments:
        xs = [point["training_pass"] for point in segment]
        if record["cohort_n"] > 1 and len(segment) > 1:
            band = axis.fill_between(xs, [point["range"][0] for point in segment],
                                     [point["range"][1] for point in segment],
                                     color=spec["color"], alpha=.09, linewidth=0, zorder=1)
            band.set_gid(f"{method}:seed-range")
        line, = axis.plot(xs, [point["mean"] for point in segment], lw=1.1,
                          markevery=2 if len(segment) > 3 else 1, zorder=4, **common)
        line.set_gid(f"{method}:paired-mean")
    audit.update(points=points, segments=[[point["step"] for point in segment] for segment in segments],
                 band="min–max across the fixed paired seeds" if record["cohort_n"] > 1 else "none: n=1")
    return audit


def _domain_mode_limits(panels: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    limits = {}
    for domain in DOMAINS:
        values = []
        for panel in panels:
            if panel["domain"] != domain:
                continue
            for record, partial in selected_methods(panel).values():
                if partial:
                    values.extend(point["value"] for seed in map(str, record["cohort_seeds"])
                                  for point in individual_points(record, seed, "distinct8"))
                else:
                    values.extend(point["range"][1] for point in primary_points(record, "distinct8"))
        top = max(values, default=1) * 1.04
        ticks = MaxNLocator(nbins=3, min_n_ticks=3).tick_values(0, max(.1, top))
        limits[domain] = (0., min(8., float(ticks[-1])))
    return limits


def build_figure(
    payload: dict[str, Any], *, level: str, metric: str | None = None,
) -> tuple[plt.Figure, dict[str, Any]]:
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("unexpected training-curve snapshot schema")
    if level not in ("level1", "level2") or (level == "level1" and metric not in METRICS):
        raise ValueError("select a Level-1 metric or the two-metric Level-2 figure")
    panels = [panel for panel in payload["panels"] if panel["level"] == level]
    by_key = {(panel["scale"], panel["domain"]): panel for panel in panels}
    if len(by_key) != len(panels):
        raise ValueError("snapshot repeats a model/domain panel")
    rows = [(scale, metric) for scale in SCALES] if level == "level1" else [
        ("qwen05b", "pass8"), ("qwen05b", "distinct8")]
    mode_limits = _domain_mode_limits(panels)
    rc = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold",
          "text.color": style.INK, "axes.labelcolor": style.INK,
          "xtick.color": style.INK, "ytick.color": style.INK,
          "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(rc):
        figure, axes = plt.subplots(len(rows), 5, figsize=(style.WIDTH, 5.75 if len(rows) == 3 else 4.25),
                                    squeeze=False)
        figure.subplots_adjust(left=.105, right=.991, bottom=.105 if len(rows) == 3 else .14,
                               top=.80 if len(rows) == 3 else .73, wspace=.33, hspace=.58)
        audit: dict[str, Any] = {"schema": "primary-training-curves-figure-v1", "level": level,
                                "metrics": [metric] if metric else list(METRICS),
                                "methods": list(METHODS), "models": list(dict.fromkeys(scale for scale, _ in rows)),
                                "x_limits": [0, 8], "panels": []}
        for row, (scale, row_metric) in enumerate(rows):
            for col, (domain, domain_label) in enumerate(zip(DOMAINS, DOMAIN_LABELS)):
                panel = by_key[(scale, domain)]
                axis = axes[row, col]
                axis.set_facecolor(style.PANEL)
                axis.set_xlim(0, 8)
                axis.set_xticks([0, 4, 8])
                limits = (0, 1) if row_metric == "pass8" else mode_limits[domain]
                axis.set_ylim(*limits)
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
                if row_metric == "pass8":
                    axis.set_yticks([0, .5, 1])
                axis.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
                axis.tick_params(labelsize=8.3, width=.6, length=2.3, pad=2)
                axis.grid(axis="both", color=style.GRID, lw=.6, zorder=0)
                axis.spines[["top", "right"]].set_visible(False)
                for spine in ("left", "bottom"):
                    axis.spines[spine].set_color(style.MUTED)
                    axis.spines[spine].set_linewidth(.6)
                if row == 0:
                    axis.set_title(domain_label, fontsize=9.5, pad=29)
                if col == 0:
                    axis.set_ylabel(SCALE_LABELS[scale] if level == "level1" else
                                    ("Accuracy\n(pass@8)" if row_metric == "pass8" else
                                     "Correct modes\n(distinct@8)"), fontsize=9.4, labelpad=7)
                note = axis.text(.5, 1.035, cohort_note(panel), ha="center", va="bottom",
                                 transform=axis.transAxes, fontsize=8.0, linespacing=1.08,
                                 color=style.MUTED)
                note.set_gid("cohort-note")
                panel_audit = {"scale": scale, "domain": domain, "metric": row_metric,
                               "y_limits": list(limits), "cohort_note": cohort_note(panel), "methods": {}}
                for method, (record, partial) in selected_methods(panel).items():
                    panel_audit["methods"][method] = draw_method(
                        axis, method, record, partial, row_metric, payload["registered_steps"])
                audit["panels"].append(panel_audit)
        handles = [Line2D([0], [0], color=spec["color"], linestyle=spec["dash"],
                          marker=spec["marker"], markerfacecolor=spec["fill"],
                          markeredgewidth=.8, markersize=4, lw=1.1, label=spec["label"])
                   for spec in METHODS.values()]
        figure.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.55, .985),
                      fontsize=9, frameon=False, handlelength=2, handletextpad=.45, columnspacing=1.1)
        figure.text(.55, .918 if len(rows) == 3 else .89,
                    f"Level 1 · {METRICS[metric]}" if level == "level1" else "Level 2 · Qwen2.5-0.5B",
                    ha="center", va="center", fontsize=10, weight="bold")
        figure.text(.55, .041 if len(rows) == 3 else .055, "Training passes", ha="center", va="center", fontsize=9.4)
        footer = ("Paired means + seed ranges · † Independent partial histories · Missing checkpoints leave gaps"
                  if level == "level2" else
                  "Paired lines: fixed-cohort means · Shading: seed range · Missing checkpoints leave gaps")
        figure.text(.55, .008, footer, ha="center", va="bottom", fontsize=8.0, color=style.MUTED)
    return figure, audit


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def render(snapshot: Path, output_dir: Path) -> list[Path]:
    raw = snapshot.read_bytes()
    payload = json.loads(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    specifications = [("level1", metric, f"factorial_training_curves_{metric}") for metric in METRICS]
    specifications.append(("level2", None, "level2_factorial_training_curves"))
    for level, metric, name in specifications:
        figure, audit = build_figure(payload, level=level, metric=metric)
        stem = output_dir / name
        audit.update(source_snapshot=_relative(snapshot), source_sha256=hashlib.sha256(raw).hexdigest(),
                     builder={"path": _relative(Path(__file__)),
                              "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        figure.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
        figure.savefig(stem.with_suffix(".png"), dpi=220)
        plt.close(figure)
        stem.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(stem.with_suffix(".pdf"))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    for output in render(args.snapshot, args.output_dir):
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
