#!/usr/bin/env python3
"""Build Figure 4: cross-scale retention plus fully matched direct alternatives.

Panel A summarizes admissible Re:Dr.GRPO-minus-Dr.GRPO endpoints over
three scales and five domains. Panel B holds scale fixed at Qwen2.5-0.5B so
Re:Dr.GRPO can be compared fairly with ordinary GRPO, UCPO, sparse RLEP-Dr,
binary MaxRL without replay, and fixed Semantic-MaxEnt without replay.
"""

from __future__ import annotations

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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402


PRIMARY = ROOT / "paper/figures/cross_scale_terminal_endpoint_effects.json"
DIRECT = ROOT / "paper/figures/direct_comparator_endpoint_effects.json"
MAXRL = ROOT / "paper/figures/e118_all_scale_factorial_progress.json"
SEMANTIC = ROOT / "paper/figures/fixed_semantic_factorial_effects.json"
TRAJECTORY = ROOT / "paper/figures/direct_baseline_learning_curves_static_strip.json"
OUTPUT = ROOT / "paper/figures/experiment1_retention_comparator_matrix"

MODELS = ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")
MODEL_LABELS = ("Qwen 0.5B", "Falcon 1B", "Qwen 3B")
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
DOMAIN_LABELS = ("Graph", "Countdown", "Python", "MathIR", "Pantry")
AVERAGE_KEY = "average"
DISPLAY_COLUMNS = DOMAINS + (AVERAGE_KEY,)
DISPLAY_LABELS = DOMAIN_LABELS + ("Average",)
METHODS = (
    "before_training",
    "replay_drgrpo",
    "maxrl",
    "ucpo",
    "rlep_dr",
    "semantic_maxent",
    "grpo",
)
METHOD_LABELS = (
    "Before training",
    "Re:Dr.GRPO (ours)",
    "MaxRL (no replay)",
    "UCPO",
    "RLEP-Dr",
    "Semantic-MaxEnt",
    "GRPO",
)
METRICS = ("pass8", "distinct8")
METRIC_TITLES = (
    r"$\Delta$ pass@8",
    r"$\Delta$ distinct@8",
)
T_CRIT_DF4 = 2.7764451051977987
AVERAGE_GREEN = "#008A5A"
REPLAY_PURPLE = style.ADAPTIVE


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(per_seed: dict[str, dict[str, float]]) -> dict[str, Any]:
    seeds = sorted(per_seed, key=int)
    if not 1 <= len(seeds) <= 5:
        raise RuntimeError(f"invalid paired seed count: {seeds}")
    output: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(per_seed[seed][metric]) for seed in seeds]
        mean = statistics.fmean(values)
        summary = {
            "mean": mean,
            "n": len(seeds),
            "per_seed": {seed: value for seed, value in zip(seeds, values)},
        }
        if len(seeds) == 5:
            half = T_CRIT_DF4 * statistics.stdev(values) / math.sqrt(5)
            summary["student_t_95"] = [mean - half, mean + half]
        output[metric] = summary
    return output


def _macro_average(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Average all five domains only on their admissible common seed set."""
    seed_sets = [set(cells[domain]["per_seed"]) for domain in DOMAINS]
    seeds = sorted(set.intersection(*seed_sets), key=int)
    if not seeds:
        raise RuntimeError("cross-domain average has no common paired seeds")
    per_seed = {
        seed: {
            metric: statistics.fmean(
                float(cells[domain]["per_seed"][seed][metric])
                for domain in DOMAINS
            )
            for metric in METRICS
        }
        for seed in seeds
    }
    return {
        "n": len(seeds),
        "seeds": list(map(int, seeds)),
        "per_seed": per_seed,
        "summaries": _summary(per_seed),
        "status": "post-hoc descriptive; excluded from inference",
    }


def _two_sided_sign_p(positive: int, negative: int) -> float:
    """Exact two-sided binomial sign test after omitting zero ties."""
    n = positive + negative
    if n == 0:
        return 1.0
    tail = min(positive, negative)
    probability = 2.0 * sum(math.comb(n, i) for i in range(tail + 1)) / 2**n
    return min(1.0, probability)


def _holm_adjust(raw_p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment, returned in the original key order."""
    ordered = sorted(raw_p, key=lambda key: (raw_p[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * raw_p[key])
        running = max(running, candidate)
        adjusted[key] = running
    return {key: adjusted[key] for key in raw_p}


def _omnibus_sign_tests(
    cells: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Exploratory signs of complete five-seed blocks only."""
    tests: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for metric in METRICS:
        values = [
            float(cells[model][domain]["summaries"][metric]["mean"])
            for model in MODELS
            for domain in DOMAINS
            if cells[model][domain]["n"] == 5
        ]
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        ties = len(values) - positive - negative
        raw_p[metric] = _two_sided_sign_p(positive, negative)
        tests[metric] = {
            "unit": "model-domain block mean",
            "positive": positive,
            "negative": negative,
            "ties": ties,
            "n_nonzero": positive + negative,
            "two_sided_exact_sign_p": raw_p[metric],
        }
    adjusted = _holm_adjust(raw_p)
    for metric in METRICS:
        tests[metric]["holm_adjusted_p_across_two_coprimary_endpoints"] = (
            adjusted[metric]
        )
    return {
        "status": "post-hoc omnibus consistency check",
        "assumption": "independent equiprobable block signs under the null; cross-block dependence is not modeled",
        "incomplete_blocks": [
            {"model": model, "domain": domain, "n": cells[model][domain]["n"]}
            for model in MODELS for domain in DOMAINS
            if cells[model][domain]["n"] != 5
        ],
        "magnitude_pooling": False,
        "family": list(METRICS),
        "tests": tests,
    }


def _build_record() -> dict[str, Any]:
    primary = _read(PRIMARY)
    direct = _read(DIRECT)
    maxrl = _read(MAXRL)
    semantic = _read(SEMANTIC)
    trajectory = _read(TRAJECTORY)
    if primary.get("schema") != "paper-cross-scale-endpoint-effects-v2":
        raise RuntimeError("cross-scale retention source schema drifted")
    if direct.get("schema") != "paper-direct-comparator-endpoint-effects-v2":
        raise RuntimeError("direct-comparator source schema drifted")
    if maxrl.get("schema") != "e118-all-scale-terminal-progress-v6":
        raise RuntimeError("MaxRL source schema drifted")
    if semantic.get("schema") != "paper-fixed-semantic-factorial-effects-v1":
        raise RuntimeError("Semantic-MaxEnt source schema drifted")
    if trajectory.get("schema") != "paper-aligned-domain-strip-v2":
        raise RuntimeError("before-training trajectory schema drifted")

    primary_rows: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        primary_rows[model] = {}
        for domain in DOMAINS:
            source = next(
                row for row in primary["rows"]
                if row["model"] == model and row["domain"] == domain
            )
            per_seed = {
                seed: {
                    "pass8": float(values["pass8"]),
                    "distinct8": (
                        float(values["pass8"])
                        + float(values["adjusted_breadth8"])
                    ),
                }
                for seed, values in source["per_seed_effects"].items()
            }
            primary_rows[model][domain] = {
                "n": len(per_seed),
                "seeds": sorted(map(int, per_seed)),
                "per_seed": per_seed,
                "summaries": _summary(per_seed),
            }
        primary_rows[model][AVERAGE_KEY] = _macro_average(primary_rows[model])

    qwen: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    for domain in DOMAINS:
        primary_source = primary_rows["Qwen2.5-0.5B"][domain]
        qwen["replay_drgrpo"][domain] = primary_source

        trajectory_source = trajectory["cells"][
            f"qwen05b/{domain}"
        ]["methods"]["drgrpo"]["summaries"]
        initial_source = trajectory_source["0"]
        terminal_source = trajectory_source["3072"]
        initial_seeds = set(initial_source["pass8"]["per_seed"])
        if initial_seeds != {"43", "44", "45", "46", "47"}:
            raise RuntimeError(f"{domain}: incomplete before-training seed set")
        per_seed = {}
        for seed in sorted(initial_seeds, key=int):
            initial_pass = float(initial_source["pass8"]["per_seed"][seed])
            initial_distinct = float(
                initial_source["distinct8"]["per_seed"][seed]
            )
            terminal_pass = float(terminal_source["pass8"]["per_seed"][seed])
            terminal_distinct = float(
                terminal_source["distinct8"]["per_seed"][seed]
            )
            per_seed[seed] = {
                "pass8": initial_pass - terminal_pass,
                "distinct8": initial_distinct - terminal_distinct,
            }
        qwen["before_training"][domain] = {
            "n": 5,
            "seeds": sorted(map(int, per_seed)),
            "per_seed": per_seed,
            "summaries": _summary(per_seed),
        }

        direct_source = next(
            cell for cell in direct["cells"]
            if cell["scale"] == "qwen05b" and cell["domain"] == domain
        )
        for method in ("grpo", "ucpo", "rlep_dr"):
            source_method = direct_source["methods"][method]
            per_seed = {
                seed: {
                    "pass8": float(seed_record["effect"]["pass8"]),
                    "distinct8": (
                        float(seed_record["effect"]["pass8"])
                        + float(seed_record["effect"]["adjusted_breadth8"])
                    ),
                }
                for seed, seed_record in source_method["per_seed"].items()
            }
            qwen[method][domain] = {
                "n": len(per_seed),
                "seeds": sorted(map(int, per_seed)),
                "per_seed": per_seed,
                "summaries": _summary(per_seed),
            }

        maxrl_source = maxrl["cells"]["qwen05b"][domain]
        seeds = list(maxrl_source["matched_seeds"])
        if seeds != [43, 44, 45, 46, 47]:
            raise RuntimeError(f"{domain}: MaxRL block is not the complete seed set")
        maxrl_methods = maxrl_source["methods"]
        per_seed: dict[str, dict[str, float]] = {}
        for index, seed in enumerate(seeds):
            control_pass = float(maxrl_methods["drgrpo"]["pass8"][index])
            control_distinct = float(maxrl_methods["drgrpo"]["distinct8"][index])
            method_pass = float(maxrl_methods["maxrl"]["pass8"][index])
            method_distinct = float(maxrl_methods["maxrl"]["distinct8"][index])
            per_seed[str(seed)] = {
                "pass8": method_pass - control_pass,
                "distinct8": method_distinct - control_distinct,
            }
        qwen["maxrl"][domain] = {
            "n": 5,
            "seeds": seeds,
            "per_seed": per_seed,
            "summaries": _summary(per_seed),
        }

        semantic_source = next(
            cell for cell in semantic["cells"]
            if cell["scale"] == "qwen05b" and cell["domain"] == domain
        )["contrasts"]["maxent_without_replay"]
        per_seed = {
            seed: {
                "pass8": float(values["pass8"]),
                "distinct8": (
                    float(values["pass8"])
                    + float(values["adjusted_breadth"])
                ),
            }
            for seed, values in semantic_source["per_seed"].items()
        }
        qwen["semantic_maxent"][domain] = {
            "n": len(per_seed),
            "seeds": sorted(map(int, per_seed)),
            "per_seed": per_seed,
            "summaries": _summary(per_seed),
        }

    for method in METHODS:
        qwen[method][AVERAGE_KEY] = _macro_average(qwen[method])

    return {
        "schema": "paper-experiment1-retention-comparator-matrix-v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "estimands": {
            "pass8": "method minus matched Dr.GRPO terminal pass@8",
            "distinct8": "method minus matched Dr.GRPO terminal distinct@8",
        },
        "uncertainty": (
            "paired unadjusted two-sided 95% Student-t estimation intervals over "
            "five terminal seeds only; smaller blocks are descriptive, marked with a dagger, "
            "and have no interval; retained in this artifact but not encoded as "
            "cell-level significance decisions in the figure"
        ),
        "omnibus_consistency_checks": _omnibus_sign_tests(primary_rows),
        "panel_a": {
            "description": "Re:Dr.GRPO minus Dr.GRPO across scale",
            "models": list(MODELS),
            "domains": list(DOMAINS),
            "cells": primary_rows,
            "display_columns": list(DISPLAY_COLUMNS),
            "average_definition": (
                "post-hoc descriptive unweighted macro-average across five "
                "domains within paired seed; excluded from inference"
            ),
        },
        "panel_b": {
            "description": (
                "Qwen2.5-0.5B before-training reference and direct alternatives minus terminal Dr.GRPO"
            ),
            "methods": list(METHODS),
            "method_labels": list(METHOD_LABELS),
            "domains": list(DOMAINS),
            "cells": qwen,
            "display_columns": list(DISPLAY_COLUMNS),
            "average_definition": (
                "post-hoc descriptive unweighted macro-average across five "
                "domains within paired seed; excluded from inference"
            ),
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (PRIMARY, DIRECT, MAXRL, SEMANTIC, TRAJECTORY)
        },
    }


def _draw_matrix(
    axis,
    *,
    rows: tuple[str, ...],
    row_labels: tuple[str, ...],
    cells: dict[str, dict[str, Any]],
    metric: str,
    bound: float,
    emphasized_rows: tuple[str, ...],
    best_rows: tuple[str, ...],
) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "signed_effect", ("#A23B53", "#F7FAFC", style.METHOD)
    )
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    matrix = [
        [cells[row][domain]["summaries"][metric]["mean"] for domain in DISPLAY_COLUMNS]
        for row in rows
    ]
    column_best = {
        domain: max(
            float(cells[row][domain]["summaries"][metric]["mean"])
            for row in best_rows
        )
        for domain in DISPLAY_COLUMNS
    } if best_rows else {}
    axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    axis.set_xticks(range(len(DISPLAY_COLUMNS)), DISPLAY_LABELS)
    average_label = axis.get_xticklabels()[-1]
    average_label.set_fontweight("bold")
    average_label.set_color(AVERAGE_GREEN)
    axis.set_yticks(range(len(rows)), row_labels)
    for label in axis.get_yticklabels():
        if label.get_text() == "Before training":
            label.set_color("#6B7280")
            label.set_fontweight("bold")
    axis.tick_params(axis="both", length=0, labelsize=6.6)
    # Keep adjacent Countdown/Python labels distinct at manuscript print size.
    axis.tick_params(axis="x", labelsize=6.0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks(
        [index - 0.5 for index in range(1, len(DISPLAY_COLUMNS))], minor=True
    )
    axis.set_yticks(
        [index - 0.5 for index in range(1, len(rows))], minor=True
    )
    axis.grid(which="minor", color=style.WHITE, linewidth=1.4)
    axis.tick_params(which="minor", bottom=False, left=False)
    for row_index, row in enumerate(rows):
        for domain_index, domain in enumerate(DISPLAY_COLUMNS):
            summary = cells[row][domain]["summaries"][metric]
            value = float(summary["mean"])
            face = cmap(norm(value))
            axis.text(
                domain_index,
                row_index,
                f"{value:+.2f}" + (r"$^{\dagger}$" if cells[row][domain]["n"] < 5 else ""),
                ha="center",
                va="center",
                fontsize=6.3,
                fontweight=(
                    "bold" if row in best_rows and math.isclose(
                        value, column_best[domain], abs_tol=1e-12
                    ) else "normal"
                ),
                color=style.cell_ink(face),
            )
    average_index = len(DOMAINS)
    axis.add_patch(
        Rectangle(
            (average_index - 0.49, -0.49),
            0.98,
            len(rows) - 0.02,
            fill=False,
            edgecolor=AVERAGE_GREEN,
            linewidth=1.5,
            zorder=5,
            clip_on=False,
        )
    )

    for emphasized_row in emphasized_rows:
        row_index = rows.index(emphasized_row)
        axis.add_patch(
            Rectangle(
                (-0.49, row_index - 0.49),
                len(DISPLAY_COLUMNS) - 0.02,
                0.98,
                fill=False,
                edgecolor=REPLAY_PURPLE,
                linewidth=1.9,
                zorder=6,
                clip_on=False,
            )
        )


def render(record: dict[str, Any], output: Path = OUTPUT) -> None:
    style.apply_rcparams(font_size=6.8)
    # Give the panel headings and metric titles enough vertical breathing room
    # at the manuscript single-column print size. The matrices deliberately
    # keep their existing proportions; the added height separates the panels.
    figure = plt.figure(figsize=(style.WIDTH, 3.82))
    grid = figure.add_gridspec(
        2, 3, height_ratios=(1.0, 2.0), width_ratios=(1.0, 0.45, 1.0),
        hspace=0.42, wspace=0.0,
    )
    plot_columns = (0, 2)
    axes = [
        [figure.add_subplot(grid[row, column]) for column in plot_columns]
        for row in range(2)
    ]
    bounds = {"pass8": 1.0, "distinct8": 2.4}
    for column, (metric, title) in enumerate(zip(METRICS, METRIC_TITLES)):
        _draw_matrix(
            axes[0][column],
            rows=MODELS,
            row_labels=MODEL_LABELS,
            cells=record["panel_a"]["cells"],
            metric=metric,
            bound=bounds[metric],
            emphasized_rows=(),
            best_rows=(),
        )
        axes[0][column].set_title(title, fontsize=7.5, pad=6, color=style.INK)
        _draw_matrix(
            axes[1][column],
            rows=METHODS,
            row_labels=METHOD_LABELS,
            cells=record["panel_b"]["cells"],
            metric=metric,
            bound=bounds[metric],
            emphasized_rows=("replay_drgrpo",),
            best_rows=tuple(row for row in METHODS if row != "before_training"),
        )
        axes[1][column].set_title(title, fontsize=7.5, pad=6, color=style.INK)

    figure.text(
        0.015, 0.975,
        "A  Re:Dr.GRPO (ours) − Dr.GRPO across three model scales",
        ha="left", va="top", fontsize=7.5, fontweight="bold", color=style.INK,
    )
    figure.text(
        0.015, 0.590,
        "B  Initial reference + alternatives − terminal Dr.GRPO at Qwen2.5-0.5B",
        ha="left", va="top", fontsize=7.5, fontweight="bold", color=style.INK,
    )
    figure.subplots_adjust(left=0.15, right=0.998, top=0.895, bottom=0.065)
    # Add 5pt below the panel B heading without changing matrix dimensions.
    panel_b_gap = 5.0 / 72.0 / figure.get_figheight()
    for axis in axes[1]:
        axis.set_position(axis.get_position().translated(0, -panel_b_gap))
    style.save(figure, output, png=True, dpi=260)
    plt.close(figure)


def main() -> int:
    record = _build_record()
    render(record)
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
