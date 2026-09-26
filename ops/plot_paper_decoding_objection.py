#!/usr/bin/env python3
"""Render both decoding sweeps as one plate.

The two sweeps used to be two tables of identical shape -- domain, settings
reportable, the control's PCMD range over the sweep, and the replay arm at the
default setting. Printed as numbers the shared result is easy to miss: the
control's whole range sits against zero at every setting either sweep visits,
while replay at the default setting sits far to the right of all of it. Drawn,
that is the only thing a reader has to see, and it takes a third of the space.

Nothing is recomputed. Both panels read the summaries the two decoding builders
already wrote, so the drawn spans are the same numbers the tables printed.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402

E125 = ROOT / "paper/results/decoding_objection_e125.json"
E72 = ROOT / "paper/results/decoding_objection_e72.json"
OUTPUT = ROOT / "paper/figures/decoding_objection"

#: Rows top to bottom, and the tints of each domain's card in Fig. 2 so a
#: domain keeps one colour across the paper.
DOMAINS = (
    ("graph_coloring", "Graph", "#E8F1FA"),
    ("countdown", "Countdown", "#E7FBF6"),
    ("python_factors", "Python", "#EFFBE7"),
    ("mathir", "MathIR", "#E7FBEE"),
    ("pantry_plan", "PantryPlan", "#E7ECFB"),
)

#: The setting count rides in the title rather than a second line: a subtitle
#: under a two-panel strip this short collides with the row above it.
PANELS = (
    ("A", E125, "Eight passes: 6 temperature settings"),
    ("B", E72, "Twelve passes: 11 decoding settings"),
)

#: A span narrower than this would draw as a bare hairline and read as a
#: missing value rather than a range that is genuinely tight. Graph's E72
#: control is exactly zero at all eleven settings, which is the strongest
#: version of the result and the one most worth not losing.
MIN_SPAN = 0.004


def draw_panel(axis, record: dict, letter: str, title: str,
               *, show_ylabels: bool) -> None:
    summary = record["summary"]
    for row, (key, label, tint) in enumerate(DOMAINS):
        cell = summary.get(key, {})
        axis.axhspan(row - 0.5, row + 0.5, color=tint, zorder=0, linewidth=0)
        low, high = cell.get("control_pmd_min"), cell.get("control_pmd_max")
        if low is None or high is None:
            # No setting in the sweep clears the support bar, so the control
            # has no range to draw. Saying so beats an invented zero.
            # Above the row's centre line, so it cannot collide with the
            # replay marker, which in one panel sits at zero and in the other
            # near the middle of the axis.
            axis.text(
                0.0, row - 0.28, "fewer than 30 eligible prompts",
                fontsize=6.0, style="italic", color=style.MUTED,
                ha="left", va="center", zorder=4,
            )
        else:
            if high - low < MIN_SPAN:
                middle = (low + high) / 2
                low, high = middle - MIN_SPAN / 2, middle + MIN_SPAN / 2
            axis.plot(
                [low, high], [row, row], color=style.CONTROL, linewidth=4.2,
                solid_capstyle="butt", zorder=3, alpha=0.95,
            )
        replay = cell.get("replay_pmd_default")
        if replay is not None:
            axis.plot(
                replay, row, marker="o", linestyle="none", markersize=5.6,
                markerfacecolor=style.ADAPTIVE, markeredgecolor=style.WHITE,
                markeredgewidth=0.9, zorder=5,
            )
        reportable = cell.get("settings_reportable")
        measured = cell.get("settings_measured")
        if reportable is not None and measured:
            axis.text(
                0.675, row, f"{reportable}/{measured}", fontsize=5.9,
                color=style.MUTED, ha="right", va="center", zorder=4,
            )

    axis.set_xlim(-0.025, 0.685)
    axis.set_ylim(len(DOMAINS) - 0.5, -0.5)
    axis.set_yticks(range(len(DOMAINS)))
    # The panels share the y-axis, so clearing the right panel's labels would
    # clear the shared ticks and blank both. Hide them on the axis instead.
    axis.set_yticklabels([label for _k, label, _t in DOMAINS], fontsize=7.4)
    if not show_ylabels:
        axis.tick_params(axis="y", labelleft=False)
    axis.set_xticks([0.0, 0.2, 0.4, 0.6])
    axis.tick_params(axis="x", labelsize=7.0)
    axis.set_xlabel("PCMD", fontsize=7.8, labelpad=2)
    axis.set_title(f"{letter}  {title}", fontsize=7.8, loc="left", pad=5)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(style.MUTED)
    axis.grid(axis="x", color=style.MUTED, alpha=0.18, linewidth=0.5)
    axis.set_axisbelow(True)


def main() -> None:
    style.apply_rcparams()
    records = {}
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.05), sharey=True)
    for axis, (letter, path, title) in zip(axes, PANELS, strict=True):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[path.name] = record
        draw_panel(axis, record, letter, title, show_ylabels=axis is axes[0])

    handles = [
        Line2D([0], [0], color=style.CONTROL, linewidth=4.2,
               label="Dr.GRPO control, range across settings"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=5.6,
               markerfacecolor=style.ADAPTIVE, markeredgecolor=style.WHITE,
               label="Replay variant at $T=1$"),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False,
        fontsize=6.9, bbox_to_anchor=(0.5, -0.20), handlelength=1.8,
    )
    figure.text(
        0.5, -0.34,
        "Grey counts: settings with at least one seed reaching 30 eligible prompts.",
        fontsize=6.2, color=style.MUTED, ha="center", va="bottom",
    )
    figure.subplots_adjust(wspace=0.08)
    style.save(figure, OUTPUT)
    plt.close(figure)

    OUTPUT.with_suffix(".json").write_text(json.dumps({
        "schema": "paper-decoding-objection-plate-v1",
        "figure_key": "decoding_objection",
        "estimand": "terminal PCMD of the RLVR-only control across every "
                    "decoding setting, against the replay arm at the default",
        "encoding": "per domain, a span over the control's PCMD range across "
                    "the sweep and a marker at the replay arm's default "
                    "setting; grey counts are settings with at least one seed reaching 30 eligible prompts",
        "panels": {letter: path.name for letter, path, _t in PANELS},
        "support_bar": records[E125.name]["support_bar"],
        "min_drawn_span": MIN_SPAN,
        "source": {path.name: record.get("analysis_code_sha256")
                   for path, record in
                   ((E125, records[E125.name]), (E72, records[E72.name]))},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
