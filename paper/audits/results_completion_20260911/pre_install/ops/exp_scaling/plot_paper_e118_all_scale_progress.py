#!/usr/bin/env python3
"""Render three-model replay evidence with explicit incomplete-domain coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402
from exp_scaling.build_paper_core_terminal_endpoints import sampled_endpoint

ENDPOINT_AUDIT: list[dict] = []

LEDGER = ROOT / "var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json"
BASE = ROOT / "paper/results/core_terminal_endpoints.json"
PRECHECK = ROOT / "paper/results/baseline_collapse_precheck.json"
TRAJECTORY = ROOT / "paper/figures/direct_baseline_learning_curves_static_strip.json"
OUT = ROOT / "paper/figures/e118_all_scale_factorial_progress"
APPENDIX_OUT = ROOT / "paper/figures/e118_scale_extensions_appendix"
MODELS = (
    ("qwen05b", "Qwen2.5-0.5B", (43, 44, 45, 46, 47)),
    ("falcon1b", "Falcon3-1B", (55, 56, 57, 58, 59)),
    ("qwen3b", "Qwen2.5-3B", (70, 71, 72, 73, 74)),
)
DOMAINS = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
LABELS = ("Graph", "Countdown", "Python", "MathIR", "Pantry")
DOMAIN_WASH = {
    "graph_coloring": "#E8F1FA", "countdown": "#E7FBF6",
    "python_factors": "#EFFBE7", "mathir": "#E7FBEE",
    "pantry_plan": "#E7ECFB",
}
METHODS = {
    "before_training": ("Untrained", "#6B7280", "D", "#6B7280"),
    "drgrpo": ("Dr.GRPO", style.CONTROL, "o", "none"),
    "replay_drgrpo": ("ReplayDr.GRPO (ours)", style.ADAPTIVE, "o", style.ADAPTIVE),
    "maxrl": ("MaxRL", style.COMPARATOR, "s", "none"),
    "replay_maxrl": ("ReplayMaxRL (ours)", style.ABLATION, "s", style.ABLATION),
}
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_lines(path: Path, max_lines: int = 96, chunk_bytes: int = 1_048_576):
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks = []
        newlines = 0
        while position > 0 and newlines <= max_lines:
            take = min(chunk_bytes, position)
            position -= take
            handle.seek(position)
            chunk = handle.read(take)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    return data.decode("utf-8", errors="replace").splitlines()[-max_lines:]


def endpoint(run: dict) -> dict[str, float] | None:
    values = sampled_endpoint(Path(run["run_dir"]), step=3072, audit=ENDPOINT_AUDIT)
    return {metric: values[metric] for metric in ("pass8", "distinct8")} if values else None


def before_values(
    trajectory: dict, scale: str, domain: str, metric: str, seeds: list[int],
) -> list[float]:
    source = trajectory["cells"][f"{scale}/{domain}"]["methods"]["drgrpo"]
    per_seed = source["summaries"]["0"][metric]["per_seed"]
    return [float(per_seed[str(seed)]) for seed in seeds]


def initial_method(start: str) -> str:
    """Give each track an initial reference with its own paired seeds."""
    return "before_training_drgrpo" if start == "drgrpo" else "before_training"


def attach_reference_methods(
    cell: dict, *, base: dict, trajectory: dict, scale: str,
    model: str, domain: str,
) -> None:
    """Keep the MaxRL intersection and separately match the Dr.GRPO track."""
    common = cell["matched_seeds"]
    sources = base["models"][model]["domains"][domain]["methods"]
    dr_seeds = [
        seed for seed in common
        if all(str(seed) in sources[arm]["per_seed"] for arm in ("control", "replay"))
    ]
    cell["method_seeds"] = {
        "maxrl": list(common), "replay_maxrl": list(common),
        "before_training": list(common),
        "drgrpo": dr_seeds, "replay_drgrpo": dr_seeds,
        "before_training_drgrpo": dr_seeds,
    }
    for method, arm in (("drgrpo", "control"), ("replay_drgrpo", "replay")):
        cell["methods"][method] = {
            metric: [
                float(sources[arm]["per_seed"][str(seed)][metric])
                for seed in dr_seeds
            ]
            for metric in ("pass8", "distinct8")
        }
    if scale != "qwen3b":
        for method in ("before_training", "before_training_drgrpo"):
            cell["methods"][method] = {
                metric: before_values(
                    trajectory, scale, domain, metric, cell["method_seeds"][method],
                )
                for metric in ("pass8", "distinct8")
            }


MAIN_FIGURE_DESCRIPTION = (
    "three-model cross-domain pass@8 and distinct@8 with MaxRL/replay and "
    "Dr.GRPO/replay tracks; Qwen2.5-3B MaxRL uses descriptive equal-domain means "
    "of available within-domain paired seeds"
)
QWEN3B_DESCRIPTION = (
    "complete five-seed E80-R1 Dr.GRPO/replay cross-domain track and descriptive "
    "MaxRL/replay equal-domain means in the main model row; MaxRL pairs differ "
    "by domain, with exact domain counts and no pooled seed paths or intervals; "
    "all domains and both metrics in appendix panels"
)
AVAILABLE_DOMAIN_DEFINITION = "equal domain average of domain-specific paired-seed means"


def attach_complete_qwen3b_reference_track(
    record: dict, *, base: dict, precheck: dict,
) -> None:
    """Attach the completed E80-R1 track independently of E118 MaxRL progress."""
    seeds = [70, 71, 72, 73, 74]
    expected = {str(seed) for seed in seeds}
    domains = base["models"]["Qwen2.5-3B"]["domains"]
    initial = precheck["arms"]["drgrpo"]["scales"]["qwen3b"]
    if initial.get("target_step") != 3072:
        raise RuntimeError("Qwen3B initial-reference horizon drifted")
    for domain in DOMAINS:
        sources = domains[domain]["methods"]
        origin = initial["domains"][domain]["per_seed"]
        if set(origin) != expected or any(
            set(sources[arm]["per_seed"]) != expected for arm in ("control", "replay")
        ):
            raise RuntimeError(f"Qwen3B reference track requires five paired seeds: {domain}")
        cell = record["cells"]["qwen3b"][domain]
        for method, arm in (("drgrpo", "control"), ("replay_drgrpo", "replay")):
            cell["method_seeds"][method] = list(seeds)
            cell["methods"][method] = {
                metric: [float(sources[arm]["per_seed"][str(seed)][metric]) for seed in seeds]
                for metric in ("pass8", "distinct8")
            }
        cell["method_seeds"]["before_training_drgrpo"] = list(seeds)
        cell["methods"]["before_training_drgrpo"] = {
            metric: [float(origin[str(seed)]["pass0"][metric]) for seed in seeds]
            for metric in ("pass8", "distinct8")
        }
        maxrl_seeds = cell["method_seeds"]["maxrl"]
        cell["method_seeds"]["before_training"] = list(maxrl_seeds)
        cell["methods"]["before_training"] = {
            metric: [float(origin[str(seed)]["pass0"][metric]) for seed in maxrl_seeds]
            for metric in ("pass8", "distinct8")
        }
        for seed in seeds:
            for metric in ("pass8", "distinct8"):
                if origin[str(seed)]["pass8"][metric] != sources["control"]["per_seed"][str(seed)][metric]:
                    raise RuntimeError(f"Qwen3B initial/control reference mismatch: {domain}/{seed}")
        if any(
            not math.isfinite(value)
            for method in ("before_training_drgrpo", "drgrpo", "replay_drgrpo")
            for values in cell["methods"][method].values() for value in values
        ):
            raise RuntimeError(f"non-finite Qwen3B reference value: {domain}")


def add_qwen3b_main_figure_track(record: dict) -> None:
    """Extend a frozen figure record without refreshing ongoing E118 endpoints."""
    base = json.loads(BASE.read_text())
    precheck = json.loads(PRECHECK.read_text())
    source = base["sources"]["Qwen2.5-3B"]
    ledger = ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"
    registered = json.loads(ledger.read_text())
    if (
        Path(source["path"]).resolve() != ledger.resolve()
        or sha(ledger) != source["sha256"]
        or precheck["ledgers"][ledger.name]["sha256"] != source["sha256"]
        or registered["seeds"] != [70, 71, 72, 73, 74]
        or len(registered["runs"]) != 50
    ):
        raise RuntimeError("Qwen3B reference provenance does not match E80-R1")
    audits = [row for row in base["endpoint_audit"] if row.get("model") == "Qwen2.5-3B"]
    if len(audits) != 50 or any(
        row.get("status") != "admitted" or row.get("step") != 3072
        or row.get("observed_draws") != [0, 1, 2, 3]
        or row.get("conflicting_retry_selected")
        for row in audits
    ):
        raise RuntimeError("Qwen3B reference track lacks 50 admitted four-draw endpoints")
    attach_complete_qwen3b_reference_track(record, base=base, precheck=precheck)
    record["main_figure_scales"] = [scale for scale, _label, _seeds in MODELS]
    record["main_figure_tracks"] = {
        scale: ["maxrl", "drgrpo"] for scale, _label, _seeds in MODELS
    }
    record["main_cross_domain_tracks"] = {
        scale: ["maxrl", "drgrpo"] for scale, _label, _seeds in MODELS
    }
    record["main_domain_tracks"] = {}
    record["source_sha256"][str(PRECHECK.relative_to(ROOT))] = sha(PRECHECK)
    record["qwen3b_reference_source"] = {
        "ledger": str(ledger.relative_to(ROOT)), "ledger_sha256": sha(ledger),
        "terminal_endpoints": str(BASE.relative_to(ROOT)),
        "initial_reference": str(PRECHECK.relative_to(ROOT)),
        "admitted_terminal_cells": 50, "seeds": [70, 71, 72, 73, 74],
        "aggregate_status": "post-hoc descriptive; no pooled inference",
    }
    if "display_contract" in record:
        record["display_contract"].update(
            main_figure=MAIN_FIGURE_DESCRIPTION, qwen3b_display=QWEN3B_DESCRIPTION,
        )


def absolute_cross_domain_averages(record: dict) -> dict:
    """Average domains within the common seed set of each matched track."""
    averages = {}
    methods = (
        "before_training", "before_training_drgrpo", "drgrpo",
        "replay_drgrpo", "maxrl", "replay_maxrl",
    )
    for scale, _model, _seeds in MODELS:
        if scale not in record["cells"]:
            continue
        cells = record["cells"][scale]
        averages[scale] = {metric: {} for metric in ("pass8", "distinct8")}
        scale_methods = ("before_training_drgrpo", "drgrpo", "replay_drgrpo") if scale == "qwen3b" else methods
        for method in scale_methods:
            seeds = sorted(set.intersection(*(
                set(cells[domain]["method_seeds"][method]) for domain in DOMAINS
            )))
            if not seeds:
                raise RuntimeError(f"no common cross-domain seeds for {scale}/{method}")
            for metric in ("pass8", "distinct8"):
                domain_values = {
                    domain: dict(zip(
                        cells[domain]["method_seeds"][method],
                        cells[domain]["methods"][method][metric], strict=True,
                    ))
                    for domain in DOMAINS
                }
                per_seed = {
                    str(seed): statistics.fmean(
                        domain_values[domain][seed] for domain in DOMAINS
                    )
                    for seed in seeds
                }
                averages[scale][metric][method] = {
                    "mean": statistics.fmean(per_seed.values()),
                    "per_seed": per_seed, "seeds": seeds, "n": len(seeds),
                    "definition": "equal domain average within paired seed",
                    "status": (
                        "post-hoc descriptive cross-domain average" if len(seeds) == 5
                        else "descriptive paired prefix; no uncertainty interval"
                    ),
                }
    return averages


def descriptive_available_domain_averages(record: dict) -> dict:
    """Balance domains while retaining each domain's own valid MaxRL pairs.

    This descriptive estimand has no shared cross-domain seed observations.
    Keep it separate from the common-seed macro averages and never attach a
    seed-level standard error, confidence interval, or inferred missing value.
    """
    scale = "qwen3b"
    cells = record["cells"].get(scale, {})
    if any(not cells.get(domain, {}).get("matched_seeds") for domain in DOMAINS):
        return {}
    methods = ("before_training", "maxrl", "replay_maxrl")
    output = {scale: {metric: {} for metric in ("pass8", "distinct8")}}
    for method in methods:
        for metric in ("pass8", "distinct8"):
            per_domain = {}
            for domain in DOMAINS:
                cell = cells[domain]
                seeds = cell["method_seeds"][method]
                if seeds != cell["matched_seeds"]:
                    raise RuntimeError(f"unmatched available-domain reference: {domain}/{method}")
                values = cell["methods"][method][metric]
                if any(not math.isfinite(value) for value in values):
                    raise RuntimeError(f"non-finite available-domain value: {domain}/{method}")
                per_domain[domain] = dict(zip(map(str, seeds), values, strict=True))
            domain_means = {
                domain: statistics.fmean(values.values())
                for domain, values in per_domain.items()
            }
            output[scale][metric][method] = {
                "mean": statistics.fmean(domain_means.values()),
                "n_domains": len(DOMAINS),
                "domain_means": domain_means,
                "per_domain_per_seed": per_domain,
                "domain_seed_counts": {
                    domain: len(values) for domain, values in per_domain.items()
                },
                "domain_weights": {domain: 1 / len(DOMAINS) for domain in DOMAINS},
                "definition": AVAILABLE_DOMAIN_DEFINITION,
                "status": "descriptive available-pair domain mean; no cross-domain seed inference",
                "uncertainty": "none; domain-specific seed sets are not pooled",
            }
    return output


PAIR_TRACKS = (
    (
        "maxrl", "replay_maxrl", 0.18, "s",
        style.COMPARATOR, style.ABLATION,
    ),
    (
        "drgrpo", "replay_drgrpo", -0.18, "o",
        style.CONTROL, style.ADAPTIVE,
    ),
)


def pair_legend_handles(*, include_untrained: bool = False) -> list[Line2D]:
    handles = []
    if include_untrained:
        handles.append(
            Line2D(
                [0], [0], marker="D", linestyle="none", markersize=4.8,
                markerfacecolor="#6B7280", markeredgecolor="#6B7280",
                label="Untrained",
            )
        )
    handles.extend([
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=5.2,
            markerfacecolor=style.WHITE, markeredgecolor=style.COMPARATOR,
            markeredgewidth=1.2, label="MaxRL",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=5.4,
            markerfacecolor=style.ABLATION, markeredgecolor=style.ABLATION,
            label="ReplayMaxRL (ours)",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.4,
            markerfacecolor=style.WHITE, markeredgecolor=style.CONTROL,
            markeredgewidth=1.2, label="Dr.GRPO",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.6,
            markerfacecolor=style.ADAPTIVE, markeredgecolor=style.ADAPTIVE,
            label="ReplayDr.GRPO (ours)",
        ),
    ])
    return handles


def draw_pair_panel(
    axis,
    *,
    record: dict,
    absolute_averages: dict,
    scale: str,
    metric: str,
    title: str,
    rows: tuple[tuple[str, str], ...],
    xlim: tuple[float, float],
    show_untrained: bool = False,
    tracks: tuple = PAIR_TRACKS,
    counts_in_labels: bool = False,
) -> None:
    positions = {
        key: len(rows) - 1 - index
        for index, (key, _label) in enumerate(rows)
    }
    axis.set_facecolor(style.WHITE)
    axis.set_title(title, loc="left", fontsize=9.2, fontweight="bold", pad=5)
    axis.set_xlim(*xlim)
    axis.set_ylim(-0.48, len(rows) - 0.52)
    axis.grid(axis="x", color=style.GRID, linewidth=0.65, zorder=1)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=7.7)
    axis.tick_params(axis="y", length=0, pad=6)

    for key, _label in rows:
        middle = positions[key]
        wash = "#F3F5F7" if key == "average" else DOMAIN_WASH[key]
        axis.axhspan(
            middle - 0.43, middle + 0.43,
            facecolor=wash, edgecolor="none", zorder=0,
        )
        if key == "average":
            values = {
                method: absolute_averages[scale][metric][method]["mean"]
                for method in (
                    (("before_training", "before_training_drgrpo") if show_untrained else ())
                    + ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
                )
                if method in absolute_averages[scale][metric]
            }
        else:
            cell = record["cells"][scale][key]
            values = {}
            for method in (
                (("before_training", "before_training_drgrpo") if show_untrained else ())
                + ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
            ):
                samples = cell["methods"].get(method, {}).get(metric, [])
                if samples:
                    values[method] = statistics.fmean(samples)

        for start, finish, offset, marker, start_color, finish_color in tracks:
            if start not in values or finish not in values:
                continue
            if key == "average":
                seeds = absolute_averages[scale][metric][start]["seeds"]
            else:
                seeds = cell["method_seeds"][start]
            complete = len(seeds) == 5
            initial = initial_method(start)
            y = middle + (offset if len(tracks) > 1 else 0.0)
            line_style = "-" if complete else (0, (2.5, 1.8))
            if not complete and not counts_in_labels:
                axis.text(
                    xlim[1] - 0.015 * (xlim[1] - xlim[0]), y + 0.065,
                    f"n={len(seeds)}", fontsize=6.3, color=start_color,
                    ha="right", va="bottom", zorder=6,
                )
            if show_untrained and initial in values:
                axis.plot(
                    [values[initial], values[start]], [y, y],
                    color="#6B7280", linewidth=1.0, linestyle=line_style,
                    solid_capstyle="round", zorder=2,
                )
                axis.plot(
                    values[initial], y, marker="D", linestyle="none",
                    markersize=4.5, markerfacecolor="#6B7280",
                    markeredgecolor="#6B7280", markeredgewidth=0.9, zorder=4,
                )
            axis.annotate(
                "",
                xy=(values[finish], y),
                xytext=(values[start], y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": start_color,
                    "linewidth": 1.25,
                    "linestyle": line_style,
                    "mutation_scale": 7.5,
                    "shrinkA": 3.5,
                    "shrinkB": 4.5,
                },
                zorder=2,
            )
            axis.plot(
                values[start], y, marker=marker, linestyle="none",
                markersize=5.0, markerfacecolor=style.WHITE,
                markeredgecolor=start_color, markeredgewidth=1.2, zorder=3,
            )
            axis.plot(
                values[finish], y, marker=marker, linestyle="none",
                markersize=5.3, markerfacecolor=finish_color,
                markeredgecolor=finish_color, markeredgewidth=1.0, zorder=4,
            )

    axis.set_yticks(
        [positions[key] for key, _label in rows],
        [
            f"{label} (n={len(record['cells'][scale][key]['matched_seeds'])})"
            if counts_in_labels else label
            for key, label in rows
        ],
        fontsize=7.9,
    )
    if rows and rows[0][0] == "average":
        tick_labels = axis.get_yticklabels()
        if tick_labels:
            tick_labels[0].set_fontweight("bold")
    axis.set_xlabel(
        "pass@8" if metric == "pass8" else "distinct@8 (verified modes)",
        fontsize=8.2, labelpad=3,
    )


def cross_domain_seed_values(
    absolute_averages: dict,
    scale: str,
    metric: str,
    method: str,
    seeds: tuple[int, ...],
) -> list[float]:
    source = absolute_averages[scale][metric][method]
    if source.get("definition") != "equal domain average within paired seed":
        raise RuntimeError(f"non-comparable average for {scale}/{metric}/{method}")
    per_seed = source.get("per_seed", {})
    if set(per_seed) != {str(seed) for seed in seeds}:
        raise RuntimeError(f"incomplete seed set for {scale}/{metric}/{method}")
    return [float(per_seed[str(seed)]) for seed in seeds]


def draw_cross_domain_panel(
    axis,
    *,
    absolute_averages: dict,
    metric: str,
    title: str,
    xlim: tuple[float, float],
    descriptive_averages: dict | None = None,
) -> None:
    axis.set_facecolor(style.WHITE)
    axis.set_title(title, loc="left", fontsize=9.4, fontweight="bold", pad=6)
    axis.set_xlim(*xlim)
    completed = tuple(model for model in MODELS if model[0] in absolute_averages)
    positions = tuple(float(index) for index in reversed(range(len(completed))))
    axis.set_ylim(-0.48, len(completed) - 0.52)
    axis.grid(axis="x", color=style.GRID, linewidth=0.65, zorder=1)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=7.7)
    axis.tick_params(axis="y", length=0, pad=7)
    for index, row in enumerate(positions):
        axis.axhspan(row - 0.45, row + 0.45, color=("#F3F6F8", "#FAFBFC")[index % 2], zorder=0)

    for row, (scale, _label, _registered_seeds) in zip(positions, completed):
        descriptive = (descriptive_averages or {}).get(scale, {}).get(metric, {})
        regular = absolute_averages[scale][metric]
        tracks = tuple(track for track in PAIR_TRACKS
                       if track[0] in regular or track[0] in descriptive)
        for start, finish, offset, marker, start_color, finish_color in tracks:
            y = row + (offset if len(tracks) > 1 else 0.0)
            if start in descriptive:
                source = descriptive[start]
                if source["definition"] != AVAILABLE_DOMAIN_DEFINITION:
                    raise RuntimeError("unsupported descriptive domain aggregate")
                counts = list(source["domain_seed_counts"].values())
                count_label = f"n={min(counts)}–{max(counts)}/domain"
                complete = False
                initial_mean, start_mean, finish_mean = (
                    descriptive[method]["mean"]
                    for method in (initial_method(start), start, finish)
                )
                # There is no common seed cohort here: draw only the domain-
                # balanced descriptive means, with no seed paths or interval.
            else:
                seeds = tuple(regular[start]["seeds"])
                complete = len(seeds) == 5
                count_label = f"n={len(seeds)}"
                seed_jitter = [
                    -0.042 + 0.084 * index / (len(seeds) - 1)
                    if len(seeds) > 1 else 0.0
                    for index in range(len(seeds))
                ]
                before, starts, finishes = (
                    cross_domain_seed_values(absolute_averages, scale, metric, method, seeds)
                    for method in (initial_method(start), start, finish)
                )
                for initial, control, replay, jitter in zip(before, starts, finishes, seed_jitter):
                    seed_y = y + jitter
                    axis.plot(
                        [initial, control], [seed_y, seed_y], color="#9AA6B2",
                        linewidth=0.55, alpha=0.34, zorder=2,
                    )
                    axis.plot(
                        [control, replay], [seed_y, seed_y], color=start_color,
                        linewidth=0.65, alpha=0.25, zorder=2,
                    )
                    axis.plot(
                        [control, replay], [seed_y, seed_y], linestyle="none",
                        marker=marker, markersize=2.2, markerfacecolor=style.WHITE,
                        markeredgecolor=start_color, markeredgewidth=0.45,
                        alpha=0.38, zorder=3,
                    )
                initial_mean, start_mean, finish_mean = (
                    statistics.fmean(values) for values in (before, starts, finishes)
                )
            line_style = "-" if complete else (0, (2.5, 1.8))
            if not complete:
                axis.text(
                    xlim[1] - 0.015 * (xlim[1] - xlim[0]), y + 0.065,
                    count_label, fontsize=6.5, color=start_color,
                    ha="right", va="bottom", zorder=7,
                )
            axis.plot(
                [initial_mean, start_mean], [y, y], color="#6B7280",
                linewidth=1.15, linestyle=line_style,
                solid_capstyle="round", zorder=4,
            )
            axis.annotate(
                "", xy=(finish_mean, y), xytext=(start_mean, y),
                arrowprops={
                    "arrowstyle": "-|>", "color": start_color,
                    "linewidth": 1.55, "mutation_scale": 8.5,
                    "linestyle": line_style,
                    "shrinkA": 4.0, "shrinkB": 5.0,
                },
                zorder=4,
            )
            axis.plot(
                initial_mean, y, marker="D", linestyle="none", markersize=5.0,
                markerfacecolor="#6B7280", markeredgecolor="#6B7280", zorder=5,
            )
            axis.plot(
                start_mean, y, marker=marker, linestyle="none", markersize=5.7,
                markerfacecolor=style.WHITE, markeredgecolor=start_color,
                markeredgewidth=1.35, zorder=5,
            )
            axis.plot(
                finish_mean, y, marker=marker, linestyle="none", markersize=6.0,
                markerfacecolor=finish_color, markeredgecolor=finish_color,
                markeredgewidth=1.0, zorder=6,
            )

    axis.set_yticks(
        positions, [model for _scale, model, _seeds in completed],
    )
    axis.set_xlabel(
        "probability" if metric == "pass8" else "verified modes (raw count)",
        fontsize=8.1, labelpad=3,
    )


def distinct_axis_upper(record: dict, scales: tuple[str, ...]) -> float:
    """Keep every displayed absolute domain mean inside the shared metric axis."""
    maximum = max(
        statistics.fmean(values["distinct8"])
        for scale in scales for cell in record["cells"][scale].values()
        for values in cell["methods"].values() if values.get("distinct8")
    )
    return max(2.5, math.ceil(maximum * 1.04 * 10) / 10)


def render_main_figure(record: dict, absolute_averages: dict) -> None:
    style.apply_rcparams(font_size=8.8)
    figure, axes = plt.subplots(
        1, 2, figsize=(style.WIDTH, 2.70), gridspec_kw={"wspace": 0.20},
    )
    descriptive = record["descriptive_available_domain_average"]
    draw_cross_domain_panel(
        axes[0], absolute_averages=absolute_averages, descriptive_averages=descriptive,
        metric="pass8", title="A  Cross-domain pass@8", xlim=(0.0, 1.0),
    )
    maximum_distinct = max(
        [value for scale in absolute_averages.values()
         for method in scale["distinct8"].values()
         for value in method["per_seed"].values()]
        + [method["mean"] for scale in descriptive.values()
           for method in scale["distinct8"].values()]
    )
    distinct_upper = max(1.62, math.ceil(maximum_distinct * 1.04 * 10) / 10)
    record["display_contract"]["main_axis_limits"] = {
        "pass8": [0.0, 1.0], "distinct8": [0.0, distinct_upper],
    }
    draw_cross_domain_panel(
        axes[1], absolute_averages=absolute_averages, descriptive_averages=descriptive,
        metric="distinct8", title="B  Cross-domain distinct@8", xlim=(0.0, distinct_upper),
    )
    axes[1].tick_params(labelleft=False)
    figure.legend(
        handles=pair_legend_handles(include_untrained=True), ncol=5,
        loc="upper center", bbox_to_anchor=(0.57, 0.995), frameon=False,
        fontsize=7.0, columnspacing=0.70, handletextpad=0.30,
    )
    figure.text(
        0.99, 0.025, "3B MaxRL: equal domain weights; domain-specific paired seeds; descriptive only.",
        ha="right", fontsize=6.8, color=style.MUTED,
    )
    figure.subplots_adjust(top=0.80, bottom=0.22, left=0.19, right=0.99)
    for extension in ("pdf", "png"):
        figure.savefig(
            OUT.with_suffix("." + extension), dpi=220,
            bbox_inches="tight", pad_inches=0.02,
        )
    plt.close(figure)


def render_appendix_figure(record: dict, absolute_averages: dict) -> None:
    style.apply_rcparams(font_size=8.8)
    figure, axes = plt.subplots(
        3, 2, figsize=(style.WIDTH, 7.25),
        gridspec_kw={"hspace": 0.43, "wspace": 0.17},
    )
    domain_rows = tuple(zip(DOMAINS, LABELS))
    upper = distinct_axis_upper(record, tuple(scale for scale, _, _ in MODELS))
    record["display_contract"]["appendix_axis_limits"] = {
        "pass8": [-0.04, 1.04], "distinct8": [-0.08, upper],
    }
    for row, (scale, label, _seeds) in enumerate(MODELS):
        for col, metric in enumerate(("pass8", "distinct8")):
            letter = chr(ord("A") + 2 * row + col)
            metric_label = "pass@8" if metric == "pass8" else "distinct@8"
            draw_pair_panel(
                axes[row, col], record=record, absolute_averages=absolute_averages,
                scale=scale, metric=metric, title=f"{letter}  {label} · {metric_label}",
                rows=domain_rows,
                xlim=(-0.04, 1.04) if metric == "pass8" else (-0.08, upper),
                show_untrained=True,
            )
        axes[row, 1].tick_params(labelleft=False)
    figure.legend(
        handles=pair_legend_handles(include_untrained=True), ncol=5,
        loc="upper center", bbox_to_anchor=(0.57, 0.995), frameon=False,
        fontsize=7.0, columnspacing=0.70, handletextpad=0.30,
    )
    figure.text(
        0.99, 0.013, "Solid: n=5; dashed: available paired seeds (descriptive).",
        ha="right", fontsize=7.0, color=style.MUTED,
    )
    figure.subplots_adjust(top=0.94, bottom=0.07, left=0.17, right=0.99)
    for extension in ("pdf", "png"):
        figure.savefig(
            APPENDIX_OUT.with_suffix("." + extension), dpi=220,
            bbox_inches="tight", pad_inches=0.02,
        )
    plt.close(figure)


def write_qwen3b_python_table(record: dict) -> None:
    """Emit the completed domain block without aggregating incomplete scales."""
    cell = record["cells"]["qwen3b"]["python_factors"]
    expected = [70, 71, 72, 73, 74]
    methods = (("drgrpo", "Dr.GRPO"), ("replay_drgrpo", "ReplayDr.GRPO"),
               ("maxrl", "MaxRL"), ("replay_maxrl", "ReplayMaxRL"))
    if any(cell["method_seeds"][method] != expected for method, _ in methods):
        raise RuntimeError("Qwen3B Python table requires five valid four-arm seeds")
    lines = []
    for method, label in methods:
        values = cell["methods"][method]
        lines.append(f"{label} & 5 & {statistics.fmean(values['pass8']):.3f} & "
                     f"{statistics.fmean(values['distinct8']):.3f} " + r"\\")
    path = ROOT / "paper/results/e118_qwen3b_python_terminal_table_body.tex"
    path.write_text("\n".join(lines) + "\n    \\bottomrule\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-audit", type=Path,
                        help="render a retained dated terminal census without rescanning live runs")
    args = parser.parse_args()
    ENDPOINT_AUDIT.clear()
    read_endpoint = endpoint
    snapshot_source = None
    if args.endpoint_audit:
        frozen = json.loads(args.endpoint_audit.read_text())
        frozen_rows = frozen["campaigns"]["e118"]["rows"]
        by_run = {row["run_dir"]: row for row in frozen_rows}
        if len(by_run) != 150:
            raise RuntimeError("frozen E118 census must contain all 150 registered runs")
        def read_endpoint(run):
            row = by_run[run["run_dir"]]
            if any(row[key] != run[key] for key in ("scale", "domain", "arm", "seed")):
                raise RuntimeError("frozen E118 registered science identity changed")
            ENDPOINT_AUDIT.append(row["integrity_audit"])
            value = row["endpoint"]
            return {metric: value[metric] for metric in ("pass8", "distinct8")} if value else None
        snapshot_source = {"path": str(args.endpoint_audit.resolve().relative_to(ROOT)),
                           "sha256": sha(args.endpoint_audit),
                           "collection_started_at_utc": frozen["collected_at_utc"],
                           "collection_finished_at_utc": frozen["finished_at_utc"]}
    base = json.loads(BASE.read_text())
    ledger = json.loads(LEDGER.read_text())
    trajectory = json.loads(TRAJECTORY.read_text())
    if trajectory.get("schema") != "paper-aligned-domain-strip-v2":
        raise RuntimeError("before-training trajectory schema drifted")
    runs = {
        (run["scale"], run["domain"], run["arm"], int(run["seed"])): run
        for run in ledger["runs"]
    }
    record = {
        "schema": "e118-all-scale-terminal-progress-v6",
        "target_step": 3072,
        "before_training_step": 0,
        "before_training_definition": (
            "shared initial checkpoint evaluated over each track's own "
            "admissible terminal seed intersection"
        ),
        "main_figure_scales": ["qwen05b", "falcon1b"],
        "appendix_figure_scales": [scale for scale, _label, _seeds in MODELS],
        "incomplete_scales": ["qwen3b"],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (BASE, LEDGER, TRAJECTORY)
        },
        "cells": {},
    }
    # Keep the absolute arm values in the machine-readable record because
    # Figure 4 consumes them, but make the visual answer Figure 5's actual
    # question directly: what does replay add to MaxRL on matched seeds?
    for scale, model, seeds in MODELS:
        for domain in DOMAINS:
            fresh = {
                arm: {
                    seed: read_endpoint(runs[(scale, domain, arm, seed)])
                    for seed in seeds
                }
                for arm in ("maxrl", "replay_maxrl")
            }
            common = [
                seed for seed in seeds
                if fresh["maxrl"][seed] and fresh["replay_maxrl"][seed]
            ]
            cell = (
                record["cells"].setdefault(scale, {}).setdefault(
                    domain, {"matched_seeds": common, "methods": {}}
                )
            )
            for method in ("maxrl", "replay_maxrl"):
                cell["methods"][method] = {
                    metric: [fresh[method][seed][metric] for seed in common]
                    for metric in ("pass8", "distinct8")
                }
            attach_reference_methods(
                cell, base=base, trajectory=trajectory, scale=scale,
                model=model, domain=domain,
            )

            effect_by_seed = {
                str(seed): {
                    metric: (
                        float(fresh["replay_maxrl"][seed][metric])
                        - float(fresh["maxrl"][seed][metric])
                    )
                    for metric in ("pass8", "distinct8")
                }
                for seed in common
            }
            cell["replay_maxrl_minus_maxrl"] = {
                "per_seed": effect_by_seed,
                "summaries": {},
                "status": (
                    "complete paired block" if len(common) == 5
                    else "terminal paired prefix; descriptive only"
                ),
            }
            for metric in ("pass8", "distinct8"):
                values = [effect_by_seed[str(seed)][metric] for seed in common]
                summary = {"n": len(values)}
                if values:
                    summary["mean"] = statistics.fmean(values)
                if len(values) == 5:
                    half = 2.776445105 * statistics.stdev(values) / math.sqrt(5)
                    summary["student_t_95"] = [
                        summary["mean"] - half,
                        summary["mean"] + half,
                    ]
                cell["replay_maxrl_minus_maxrl"]["summaries"][metric] = summary

    record["endpoint_integrity_audit"] = ENDPOINT_AUDIT
    if snapshot_source is not None:
        record["endpoint_snapshot"] = snapshot_source
    qwen_cells = record["cells"]["qwen05b"]
    expected_qwen_seeds = {"43", "44", "45", "46", "47"}
    if any(
        set(qwen_cells[domain]["replay_maxrl_minus_maxrl"]["per_seed"])
        != expected_qwen_seeds
        for domain in DOMAINS
    ):
        raise RuntimeError("Qwen cross-domain average requires five common seeds")
    qwen_average_per_seed = {
        seed: {
            metric: statistics.fmean(
                qwen_cells[domain]["replay_maxrl_minus_maxrl"]["per_seed"][seed][metric]
                for domain in DOMAINS
            )
            for metric in ("pass8", "distinct8")
        }
        for seed in sorted(expected_qwen_seeds, key=int)
    }
    falcon_cells = record["cells"]["falcon1b"]
    falcon_expected_seeds = {55, 56, 57, 58, 59}
    falcon_complete = all(
        set(falcon_cells[domain]["matched_seeds"]) == falcon_expected_seeds
        for domain in DOMAINS
    )
    falcon_average_per_seed = {
        str(seed): {
            metric: statistics.fmean(
                falcon_cells[domain]["replay_maxrl_minus_maxrl"]
                ["per_seed"][str(seed)][metric]
                for domain in DOMAINS
            )
            for metric in ("pass8", "distinct8")
        }
        for seed in sorted(falcon_expected_seeds)
    } if falcon_complete else {}
    averages = {
        "qwen05b": {
            "definition": "equal domain average within paired seed",
            "per_seed": qwen_average_per_seed,
            "summaries": {},
            "status": "post-hoc descriptive cross-domain average",
        },
        "falcon1b": {
            "definition": (
                "equal domain average within paired seed" if falcon_complete
                else "equal average of the five displayed domain-prefix means"
            ),
            "per_seed": falcon_average_per_seed,
            "summaries": {},
            "status": (
                "post-hoc descriptive cross-domain average" if falcon_complete
                else "descriptive partial-prefix average; no uncertainty interval"
            ),
        },
    }
    for metric in ("pass8", "distinct8"):
        qwen_values = [row[metric] for row in qwen_average_per_seed.values()]
        qwen_mean = statistics.fmean(qwen_values)
        qwen_half = (
            2.776445105 * statistics.stdev(qwen_values) / math.sqrt(5)
        )
        averages["qwen05b"]["summaries"][metric] = {
            "n": 5,
            "mean": qwen_mean,
            "student_t_95": [qwen_mean - qwen_half, qwen_mean + qwen_half],
        }
        if falcon_complete:
            falcon_values = [
                row[metric] for row in falcon_average_per_seed.values()
            ]
            falcon_mean = statistics.fmean(falcon_values)
            falcon_half = (
                2.776445105 * statistics.stdev(falcon_values) / math.sqrt(5)
            )
            averages["falcon1b"]["summaries"][metric] = {
                "n": 5,
                "mean": falcon_mean,
                "student_t_95": [
                    falcon_mean - falcon_half,
                    falcon_mean + falcon_half,
                ],
            }
        else:
            averages["falcon1b"]["summaries"][metric] = {
                "n_domains": 5,
                "domain_seed_counts": {
                    domain: len(falcon_cells[domain]["matched_seeds"])
                    for domain in DOMAINS
                },
                "mean": statistics.fmean(
                    falcon_cells[domain]["replay_maxrl_minus_maxrl"]
                    ["summaries"][metric]["mean"]
                    for domain in DOMAINS
                ),
            }
    record["cross_domain_average"] = averages
    record["display_contract"] = {
        "estimand": "ReplayMaxRL minus MaxRL on matched terminal seeds",
        "filled_marker": "complete five-seed block",
        "open_marker": "terminal paired prefix; no interval",
        "intervals": "unadjusted descriptive 95% Student-t; n=5 only",
        "qwen_average": averages["qwen05b"]["definition"],
        "falcon_average": averages["falcon1b"]["definition"],
    }

    # The completed E80-R1 track is independent of unfinished E118 MaxRL cells.
    add_qwen3b_main_figure_track(record)
    # Each track averages the same admissible seeds across every domain.
    absolute_averages = absolute_cross_domain_averages(record)
    record["absolute_cross_domain_average"] = absolute_averages
    record["descriptive_available_domain_average"] = descriptive_available_domain_averages(record)
    record["display_contract"] = {
        "main_figure": MAIN_FIGURE_DESCRIPTION,
        "appendix_figure": (
            "all three models across five domains and both terminal metrics"
        ),
        "qwen3b_display": QWEN3B_DESCRIPTION,
        "estimands": [
            "ReplayMaxRL minus MaxRL on matched terminal seeds",
            "ReplayDr.GRPO minus Dr.GRPO on its admissible paired seed intersection",
        ],
        "tracks": {
            "upper": "Untrained to MaxRL to ReplayMaxRL",
            "lower": "Untrained to Dr.GRPO to ReplayDr.GRPO",
        },
        "untrained_reference": (
            "shared frozen step-0 checkpoint restricted to each track's paired seeds; "
            "arms share initialization but are trained separately"
        ),
        "domain_backgrounds": (
            "exact Figure 2 domain-card tints; color is contextual, not data"
        ),
        "solid_connector": "complete five-seed block",
        "dashed_connector": "partial or domain-specific paired seeds with exact counts and no interval",
        "numeric_annotations": (
            "Qwen3B MaxRL track shows n=1–5/domain; other partial tracks show exact n; "
            "all domain-specific counts and endpoints remain in JSON and appendix"
        ),
        "main_panels": {
            "A": "cross-domain pass@8", "B": "cross-domain distinct@8",
        },
        "qwen_average": "equal domain average within paired seed",
        "falcon_average": averages["falcon1b"]["definition"],
        "qwen3b_maxrl_average": AVAILABLE_DOMAIN_DEFINITION,
        "qwen3b_maxrl_counts": {
            domain: len(record["cells"]["qwen3b"][domain]["matched_seeds"])
            for domain in DOMAINS
        },
        "qwen3b_maxrl_uncertainty": "none; no pooled seed paths or cross-domain interval",
    }
    write_qwen3b_python_table(record)
    render_main_figure(record, absolute_averages)
    render_appendix_figure(record, absolute_averages)
    for path in (
        OUT.with_suffix(".json"), APPENDIX_OUT.with_suffix(".json"),
    ):
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(OUT.with_suffix(".pdf"))
    print(APPENDIX_OUT.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
