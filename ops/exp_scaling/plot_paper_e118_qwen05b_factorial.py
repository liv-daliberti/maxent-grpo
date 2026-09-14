#!/usr/bin/env python3
"""Render the page-1 Qwen-0.5B replay x MaxRL terminal factorial."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

SOURCE = ROOT / "paper/results/e118_qwen05b_factorial.json"
OUT = ROOT / "paper/figures/replay_maxrl_qwen05b"
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
LABELS = ("Graph", "Countdown", "Python", "MathIR", "Pantry")
METHODS = {
    "drgrpo": {"label": "Dr.GRPO", "color": style.CONTROL, "marker": "o", "fill": "none"},
    "replay_drgrpo": {"label": "Re:Dr.GRPO", "color": style.METHOD, "marker": "o", "fill": style.METHOD},
    "maxrl": {"label": "MaxRL", "color": style.COMPARATOR, "marker": "s", "fill": "none"},
    "replay_maxrl": {"label": "Re:MaxRL", "color": style.ABLATION, "marker": "s", "fill": style.ABLATION},
}


def main() -> int:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if data.get("schema") != "e118-qwen05b-terminal-factorial-v1" or data.get("seeds") != [43, 44, 45, 46, 47]:
        raise RuntimeError("E118 page-1 result contract drifted")
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 2.42), gridspec_kw={"wspace": .25})
    metrics = (("pass8", "pass@8", (0, 1.04)), ("distinct8", "distinct correct modes @ 8", (0, 2.72)))
    offsets = {"drgrpo": -.18, "replay_drgrpo": -.06, "maxrl": .06, "replay_maxrl": .18}
    for ax, (metric, title, ylim) in zip(axes, metrics):
        ax.set_facecolor(style.PANEL)
        for x, domain in enumerate(DOMAINS):
            d = data["domains"][domain]
            for left, right, color in (("drgrpo", "replay_drgrpo", style.METHOD), ("maxrl", "replay_maxrl", style.ABLATION)):
                y0 = d["summaries"][left][metric]["mean"]
                y1 = d["summaries"][right][metric]["mean"]
                ax.annotate("", xy=(x + offsets[right], y1), xytext=(x + offsets[left], y0),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.15, "alpha": .70, "shrinkA": 3, "shrinkB": 3})
            for method, spec in METHODS.items():
                s = d["summaries"][method][metric]
                mean, ci = s["mean"], s["student_t_95"]
                ax.errorbar(x + offsets[method], mean, yerr=[[mean-ci[0]], [ci[1]-mean]], fmt=spec["marker"],
                    ms=5.2, mfc=spec["fill"], mec=spec["color"], mew=1.15, color=spec["color"],
                    ecolor=spec["color"], elinewidth=.9, capsize=1.8, zorder=4)
        ax.set_title(title, color=style.INK, pad=5)
        ax.set_xlim(-.48, 4.48); ax.set_ylim(*ylim)
        ax.set_xticks(range(5), LABELS)
        ax.grid(color=style.GRID, lw=.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(style.MUTED)
        ax.tick_params(colors=style.INK, labelsize=7.5)
    axes[0].set_yticks([0, .25, .5, .75, 1.0])
    axes[1].set_yticks([0, .5, 1, 1.5, 2, 2.5])
    handles = [Line2D([0], [0], marker=s["marker"], color="none", markerfacecolor=s["fill"],
        markeredgecolor=s["color"], markeredgewidth=1.15, markersize=5.4, label=s["label"]) for s in METHODS.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=4,
        frameon=False, handletextpad=.35, columnspacing=1.3, fontsize=7.8)
    fig.text(.5, .005, "Arrows add verified replay within the same fresh-rollout objective; mean ± 95% t interval across seeds (n=5).",
        ha="center", va="bottom", color=style.MUTED, fontsize=7.2)
    fig.subplots_adjust(top=.78, bottom=.19, left=.07, right=.985)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=.02)
    audit = {"schema": "page1-e118-qwen05b-factorial-figure-v1", "source": str(SOURCE),
        "output": str(OUT.with_suffix(".pdf")), "metrics": [m[0] for m in metrics], "methods": list(METHODS)}
    OUT.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
