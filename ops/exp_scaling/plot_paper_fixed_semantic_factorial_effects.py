#!/usr/bin/env python3
"""Render paired fixed-semantic effects and their factorial interaction.

The forest shows the semantic-MaxEnt effect without replay, its effect on top
of Re:Dr.GRPO, and their difference-in-differences interaction.  Qwen2.5-
0.5B is read from the completed paper factorial record.  Falcon3-1B is rebuilt
from exact terminal draws for all five balanced four-arm domains.  Raw paired
seeds, means, and two-sided Student-t intervals are
shown without pooling models or domains.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402
from plot_paper_direct_comparator_endpoint_effects import (  # noqa: E402
    _completion_marker,
    _interval,
    _relative,
    _runs,
    _sha256,
    _terminal_metrics,
)


DEFAULT_OUTPUT = ROOT / "paper/figures/fixed_semantic_factorial_effects"
QWEN_RESULT = ROOT / "paper/results/maxent_factorial_05b.json"
FALCON_LEDGERS = {
    "control_replay": (
        ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"
    ),
    "semantic": (
        ROOT / "var/artifacts/e82_falcon_semantic_maxent_verified_replay_jobs.json"
    ),
    "semantic_only": (
        ROOT / "var/artifacts/e86_falcon_semantic_maxent_without_replay_jobs.json"
    ),
}
MODELS = ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")
MODEL_SHORT = {
    "Qwen2.5-0.5B": "Qwen 0.5B",
    "Falcon3-1B": "Falcon 1B",
    "Qwen2.5-3B": "Qwen 3B",
}
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
FALCON_DOMAINS = DOMAIN_ORDER
FALCON_SEEDS = (55, 56, 57, 58, 59)
CONTRASTS = (
    "maxent_without_replay",
    "maxent_with_replay",
    "factorial_interaction",
)
CONTRAST_LABEL = {
    "maxent_without_replay": "MaxEnt − Dr.GRPO",
    "maxent_with_replay": "MaxEnt+Replay − Replay",
    "factorial_interaction": "interaction",
}
CONTRAST_TICK = {
    "maxent_without_replay": "MaxEnt\nno replay",
    "maxent_with_replay": "MaxEnt\non replay",
    "factorial_interaction": "interaction",
}
METRICS = ("pass8", "distinct8", "adjusted_breadth")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _effect(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    delta_pass = left["pass8"] - right["pass8"]
    delta_distinct = left["distinct8"] - right["distinct8"]
    return {
        "pass8": delta_pass,
        "distinct8": delta_distinct,
        "adjusted_breadth": delta_distinct - delta_pass,
    }


def _summaries(per_seed: dict[str, dict[str, float]]) -> dict[str, Any]:
    if set(per_seed) not in (
        {"43", "44", "45", "46", "47"},
        {"55", "56", "57", "58", "59"},
    ):
        raise RuntimeError(f"unexpected factorial seed set {sorted(per_seed)}")
    return {
        metric: _interval([values[metric] for values in per_seed.values()])
        for metric in METRICS
    }


def _qwen_cells(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    design = result.get("design", {})
    if (
        result.get("schema") != "maxent_factorial_05b_paper_results_v1"
        or design.get("domains") != list(DOMAIN_ORDER)
        or design.get("arms")
        != ["control", "replay", "semantic_only", "semantic"]
        or design.get("paired_seeds") != [43, 44, 45, 46, 47]
        or design.get("evaluation_draws") != 4
        or design.get("target_steps") != 3072
    ):
        raise RuntimeError("completed Qwen factorial record drifted")
    cells: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        source_contrasts = result["domains"][domain].get("contrasts", {})
        if set(source_contrasts) != set(CONTRASTS):
            raise RuntimeError(f"{domain}: incomplete Qwen contrast set")
        contrasts: dict[str, Any] = {}
        for contrast in CONTRASTS:
            source = source_contrasts[contrast]
            per_seed = {
                str(seed): {
                    metric: float(source[metric]["per_seed"][str(seed)])
                    for metric in METRICS
                }
                for seed in (43, 44, 45, 46, 47)
            }
            contrasts[contrast] = {
                "label": CONTRAST_LABEL[contrast],
                "per_seed": per_seed,
                "summaries": {
                    metric: source[metric]
                    for metric in METRICS
                },
            }
        cells[domain] = {
            "scale": "qwen05b",
            "model": "Qwen2.5-0.5B",
            "domain": domain,
            "n": 5,
            "seeds": [43, 44, 45, 46, 47],
            "evidence": "balanced_five_seed_terminal_factorial",
            "contrasts": contrasts,
        }
    return cells


def _falcon_cells(source_paths: set[Path]) -> dict[str, dict[str, Any]]:
    ledgers = {name: _read(path) for name, path in FALCON_LEDGERS.items()}
    source_paths.update(FALCON_LEDGERS.values())
    targets = {int(ledger["target_steps"]) for ledger in ledgers.values()}
    if targets != {3072}:
        raise RuntimeError(f"Falcon factorial target-step mismatch: {targets}")
    arm_runs = {
        "control": _runs(ledgers["control_replay"], arm="control"),
        "replay": _runs(ledgers["control_replay"], arm="replay"),
        "semantic": _runs(ledgers["semantic"], arm="semantic"),
        "semantic_only": _runs(
            ledgers["semantic_only"], arm="semantic_only"
        ),
    }
    endpoint_cache: dict[Path, dict[str, float]] = {}

    def endpoint(run: dict[str, Any], *, domain: str, seed: int) -> dict[str, float]:
        run_dir = Path(str(run["run_dir"]))
        marker = _completion_marker(run_dir, target=3072)
        if marker is None:
            raise RuntimeError(
                f"Falcon {domain}/seed {seed}/{run['arm']} is not terminal"
            )
        source_paths.add(marker)
        if run_dir not in endpoint_cache:
            values, paths = _terminal_metrics(run_dir, target=3072)
            endpoint_cache[run_dir] = values
            source_paths.update(paths)
        return endpoint_cache[run_dir]

    cells: dict[str, dict[str, Any]] = {}
    for domain in FALCON_DOMAINS:
        arms_by_seed: dict[int, dict[str, dict[str, float]]] = {}
        for seed in FALCON_SEEDS:
            arms_by_seed[seed] = {}
            for arm, runs in arm_runs.items():
                run = runs.get((domain, seed))
                if run is None:
                    raise RuntimeError(f"Falcon {domain}/seed {seed}/{arm} missing")
                arms_by_seed[seed][arm] = endpoint(
                    run, domain=domain, seed=seed
                )
        simple: dict[str, dict[str, dict[str, float]]] = {
            "maxent_without_replay": {},
            "maxent_with_replay": {},
        }
        for seed in FALCON_SEEDS:
            arms = arms_by_seed[seed]
            simple["maxent_without_replay"][str(seed)] = _effect(
                arms["semantic_only"], arms["control"]
            )
            simple["maxent_with_replay"][str(seed)] = _effect(
                arms["semantic"], arms["replay"]
            )
        interaction = {
            str(seed): {
                metric: (
                    simple["maxent_with_replay"][str(seed)][metric]
                    - simple["maxent_without_replay"][str(seed)][metric]
                )
                for metric in METRICS
            }
            for seed in FALCON_SEEDS
        }
        per_contrast = {**simple, "factorial_interaction": interaction}
        cells[domain] = {
            "scale": "falcon1b",
            "model": "Falcon3-1B",
            "domain": domain,
            "n": 5,
            "seeds": list(FALCON_SEEDS),
            "evidence": "balanced_five_seed_terminal_factorial",
            "contrasts": {
                contrast: {
                    "label": CONTRAST_LABEL[contrast],
                    "per_seed": per_contrast[contrast],
                    "summaries": _summaries(per_contrast[contrast]),
                }
                for contrast in CONTRASTS
            },
        }
    return cells


def build() -> dict[str, Any]:
    source_paths: set[Path] = {QWEN_RESULT}
    qwen = _qwen_cells(_read(QWEN_RESULT))
    falcon = _falcon_cells(source_paths)
    cells: list[dict[str, Any]] = []
    for model in MODELS:
        for domain in DOMAIN_ORDER:
            if model == "Qwen2.5-0.5B":
                cells.append(qwen[domain])
            elif model == "Falcon3-1B" and domain in falcon:
                cells.append(falcon[domain])
            else:
                cells.append(
                    {
                        "scale": (
                            "falcon1b" if model == "Falcon3-1B" else "qwen3b"
                        ),
                        "model": model,
                        "domain": domain,
                        "n": 0,
                        "seeds": [],
                        "evidence": "blank",
                        "contrasts": {},
                        "blank_reason": (
                            "no balanced terminal four-arm factorial at freeze"
                        ),
                    }
                )
    return {
        "schema": "paper-fixed-semantic-factorial-effects-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "balanced five-seed terminal four-arm factorials: all five "
            "domains for Qwen2.5-0.5B and Falcon3-1B"
        ),
        "estimand": "fixed Semantic MaxEnt effects on terminal distinct@8",
        "model_rows": list(MODELS),
        "domain_order": list(DOMAIN_ORDER),
        "contrasts": list(CONTRASTS),
        "contrast_definitions": {
            "maxent_without_replay": "semantic-only minus matched Dr.GRPO",
            "maxent_with_replay": "semantic-plus-replay minus Re:Dr.GRPO",
            "factorial_interaction": (
                "(semantic-plus-replay minus replay) minus "
                "(semantic-only minus control)"
            ),
        },
        "metrics": {
            "pass8": "terminal pass@8 effect",
            "distinct8": "terminal distinct@8 effect plotted in the forest",
            "adjusted_breadth": "distinct@8 effect minus pass@8 effect",
        },
        "uncertainty": (
            "two-sided paired 95% Student-t intervals over five seeds (df=4); "
            "no pooling across domains or models"
        ),
        "cells": cells,
        "input_sha256": {
            _relative(path): {
                "byte_length": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(source_paths)
        },
    }


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(MODELS),
        len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 5.35),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    contrast_visual = {
        "maxent_without_replay": {
            **method_visuals.method_style("semantic_maxent"),
            "marker": "^",
        },
        "maxent_with_replay": {
            **method_visuals.method_style("replay_semantic_maxent"),
            "marker": "P",
        },
        "factorial_interaction": {
            "color": style.INK,
            "marker": "D",
            "linestyle": (0, (2, 1.4)),
        },
    }
    cells = {
        (cell["model"], cell["domain"]): cell for cell in payload["cells"]
    }
    plotted = [0.0]
    for cell in payload["cells"]:
        for contrast in cell["contrasts"].values():
            summary = contrast["summaries"]["distinct8"]
            plotted.extend(summary["student_t_95"])
            plotted.extend(
                values["distinct8"]
                for values in contrast["per_seed"].values()
            )
    bound = max(0.5, math.ceil((max(abs(value) for value in plotted) + 0.08) * 2) / 2)
    y_positions = {
        "maxent_without_replay": 2.0,
        "maxent_with_replay": 1.0,
        "factorial_interaction": 0.0,
    }
    jitter = (-0.08, -0.04, 0.0, 0.04, 0.08)
    for row, model in enumerate(MODELS):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row][column]
            style.style_axis(
                axis,
                grid="x",
                title=DOMAIN_LABEL[domain] if row == 0 else None,
            )
            axis.axvline(0.0, color=style.MUTED, linewidth=0.75, linestyle=(0, (2, 2)))
            axis.set_xlim(-bound, bound)
            axis.set_ylim(-0.55, 2.55)
            axis.set_yticks(
                [y_positions[key] for key in CONTRASTS],
                [CONTRAST_TICK[key] for key in CONTRASTS],
            )
            cell = cells[(model, domain)]
            if cell["contrasts"]:
                for contrast_name in CONTRASTS:
                    contrast = cell["contrasts"][contrast_name]
                    visual = contrast_visual[contrast_name]
                    seeds = cell["seeds"]
                    values = [
                        contrast["per_seed"][str(seed)]["distinct8"]
                        for seed in seeds
                    ]
                    y = y_positions[contrast_name]
                    axis.scatter(
                        values,
                        [y + offset for offset in jitter],
                        s=11,
                        facecolors="none",
                        edgecolors=visual["color"],
                        linewidths=0.65,
                        zorder=3,
                    )
                    summary = contrast["summaries"]["distinct8"]
                    low, high = summary["student_t_95"]
                    axis.plot(
                        [low, high],
                        [y, y],
                        color=visual["color"],
                        linewidth=1.15,
                        zorder=4,
                    )
                    axis.scatter(
                        [summary["mean"]],
                        [y],
                        s=24,
                        marker="D",
                        facecolors=visual["color"],
                        edgecolors="white",
                        linewidths=0.4,
                        zorder=5,
                    )
            if column != 0:
                axis.tick_params(labelleft=False)
            if row == len(MODELS) - 1:
                axis.set_xlabel(r"$\Delta$ distinct@8", fontsize=style.LABEL_FONT)

    handles = [
        Line2D(
            [0],
            [0],
            color=contrast_visual[key]["color"],
            marker=contrast_visual[key]["marker"],
            linestyle="none",
            markersize=4,
            label=CONTRAST_LABEL[key],
        )
        for key in CONTRASTS
    ]
    handles.extend(
        [
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=3.5,
                markerfacecolor="none", markeredgecolor=style.MUTED,
                label="paired seed",
            ),
            Line2D(
                [0, 1], [0, 0], color=style.MUTED, marker="D",
                linewidth=1.0, markersize=3.5, label="mean + 95% interval",
            ),
        ]
    )
    style.bottom_legend(
        figure,
        handles,
        [handle.get_label() for handle in handles],
        y=0.003,
        ncol=5,
    )
    figure.suptitle(
        "Fixed semantic-MaxEnt factorial effects",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.96,
        (
            "Terminal distinct@8 · raw paired seeds, diamonds, and paired 95% "
            "Student-t intervals at n=5; blank cells lack a complete four-arm "
            "block."
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.13,
        right=0.995,
        top=0.87,
        bottom=0.19,
        hspace=0.30,
        wspace=0.24,
    )
    # Placed from each row's own box rather than the hand-tuned
    # `0.775 - row * 0.265`, so the labels stay centred if the geometry is ever
    # retuned. The three physical model rows themselves are invariant across
    # every cross-model figure in the paper and are never dropped, even when a
    # row is entirely blank --- that is what lets a reader carry one set of
    # axes from figure to figure.
    for row, model in enumerate(MODELS):
        box = axes[row][0].get_position()
        figure.text(
            0.018,
            (box.y0 + box.y1) / 2,
            MODEL_SHORT[model],
            rotation=90,
            ha="center",
            va="center",
            fontsize=style.LABEL_FONT,
            color=style.INK,
        )
    style.save(figure, output, png=True, dpi=260)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    output = args.output.resolve()
    render(payload, output)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output.with_suffix('.pdf')}, {output.with_suffix('.png')}, "
        f"and {output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
