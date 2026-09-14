#!/usr/bin/env python3
"""Preview Experiment 2 using only Qwen/Falcon cross-domain averages."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402


SOURCE = ROOT / "paper/figures/e118_all_scale_factorial_progress.json"
OUT = ROOT / "paper/figures/e118_cross_domain_only_preview"
SCALES = (
    ("qwen05b", "Qwen2.5-0.5B", (43, 44, 45, 46, 47)),
    ("falcon1b", "Falcon3-1B", (55, 56, 57, 58, 59)),
)
TRACKS = (
    (
        "maxrl", "replay_maxrl", +0.15, "s",
        style.COMPARATOR, style.ABLATION,
    ),
    (
        "drgrpo", "replay_drgrpo", -0.15, "o",
        style.CONTROL, style.ADAPTIVE,
    ),
)
METRICS = (
    ("pass8", "A  Cross-domain pass@8", (0.0, 1.0)),
    ("distinct8", "B  Cross-domain distinct@8", (0.0, 1.62)),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=4.8,
            markerfacecolor="#6B7280", markeredgecolor="#6B7280",
            label="Untrained",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=5.2,
            markerfacecolor=style.WHITE, markeredgecolor=style.COMPARATOR,
            markeredgewidth=1.2, label="MaxRL",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=5.4,
            markerfacecolor=style.ABLATION, markeredgecolor=style.ABLATION,
            label="Re:MaxRL (ours)",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.2,
            markerfacecolor=style.WHITE, markeredgecolor=style.CONTROL,
            markeredgewidth=1.2, label="Dr.GRPO",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.4,
            markerfacecolor=style.ADAPTIVE, markeredgecolor=style.ADAPTIVE,
            label="Re:Dr.GRPO (ours)",
        ),
    ]


def values(
    absolute: dict, scale: str, metric: str, method: str, seeds: tuple[int, ...],
) -> list[float]:
    record = absolute[scale][metric][method]
    if record.get("definition") != "equal domain average within paired seed":
        raise RuntimeError(f"non-comparable average for {scale}/{metric}/{method}")
    per_seed = record.get("per_seed", {})
    if set(per_seed) != {str(seed) for seed in seeds}:
        raise RuntimeError(f"incomplete seed set for {scale}/{metric}/{method}")
    return [float(per_seed[str(seed)]) for seed in seeds]


def draw_panel(axis, absolute: dict, metric: str, title: str, xlim: tuple[float, float]) -> None:
    axis.set_facecolor(style.WHITE)
    axis.set_title(title, loc="left", fontsize=9.4, fontweight="bold", pad=6)
    axis.set_xlim(*xlim)
    axis.set_ylim(-0.48, 1.48)
    axis.grid(axis="x", color=style.GRID, linewidth=0.65, zorder=1)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=7.7)
    axis.tick_params(axis="y", length=0, pad=7)
    axis.axhspan(0.52, 1.45, color="#F3F6F8", zorder=0)
    axis.axhspan(-0.45, 0.48, color="#FAFBFC", zorder=0)

    seed_jitter = (-0.042, -0.021, 0.0, 0.021, 0.042)
    for row, (scale, _label, seeds) in zip((1.0, 0.0), SCALES):
        for start, finish, offset, marker, start_color, finish_color in TRACKS:
            before = values(absolute, scale, metric, "before_training", seeds)
            starts = values(absolute, scale, metric, start, seeds)
            finishes = values(absolute, scale, metric, finish, seeds)
            y = row + offset

            for initial, control, replay, jitter in zip(
                before, starts, finishes, seed_jitter
            ):
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

            means = tuple(
                float(absolute[scale][metric][method]["mean"])
                for method in ("before_training", start, finish)
            )
            initial_mean, start_mean, finish_mean = means
            axis.plot(
                [initial_mean, start_mean], [y, y], color="#6B7280",
                linewidth=1.15, solid_capstyle="round", zorder=4,
            )
            axis.annotate(
                "", xy=(finish_mean, y), xytext=(start_mean, y),
                arrowprops={
                    "arrowstyle": "-|>", "color": start_color,
                    "linewidth": 1.55, "mutation_scale": 8.5,
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

    axis.set_yticks((1.0, 0.0), [label for _scale, label, _seeds in SCALES])
    axis.set_xlabel(
        "probability" if metric == "pass8" else "verified modes (raw count)",
        fontsize=8.1, labelpad=3,
    )


def main() -> None:
    record = json.loads(SOURCE.read_text(encoding="utf-8"))
    if record.get("schema") != "e118-all-scale-terminal-progress-v4":
        raise RuntimeError("Figure 5 source schema drifted")
    absolute = record["absolute_cross_domain_average"]

    style.apply_rcparams(font_size=8.8)
    figure, axes = plt.subplots(
        1, 2, figsize=(style.WIDTH, 2.35), gridspec_kw={"wspace": 0.20},
    )
    for axis, (metric, title, xlim) in zip(axes, METRICS):
        draw_panel(axis, absolute, metric, title, xlim)
    axes[1].tick_params(labelleft=False)
    figure.legend(
        handles=legend_handles(), ncol=5, loc="upper center",
        bbox_to_anchor=(0.57, 0.995), frameon=False, fontsize=7.0,
        columnspacing=0.70, handletextpad=0.30,
    )
    figure.subplots_adjust(top=0.80, bottom=0.19, left=0.19, right=0.99)
    for extension in ("pdf", "png"):
        figure.savefig(
            OUT.with_suffix("." + extension), dpi=240,
            bbox_inches="tight", pad_inches=0.02,
        )
    plt.close(figure)

    payload = {
        "schema": "e118-cross-domain-only-preview-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "scales": [scale for scale, _label, _seeds in SCALES],
        "metrics": [metric for metric, _title, _xlim in METRICS],
        "estimand": "equal-domain average computed within seed",
        "seed_paths": "five paired cross-domain averages per scale",
        "numeric_annotations": "none",
        "manuscript_replacement": False,
        "absolute_cross_domain_average": {
            scale: absolute[scale] for scale, _label, _seeds in SCALES
        },
    }
    OUT.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
