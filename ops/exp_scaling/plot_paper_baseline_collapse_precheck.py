#!/usr/bin/env python3
"""Draw the problem precheck: correctness up, verified breadth gone, everywhere.

The main text used to state this as three Qwen2.5-0.5B rows of numbers. Read as
a figure it has to answer the question a reader actually asks --- *is this one
small model, or is it the objective?* --- so the grid is the manuscript's
invariant three-model rows by five domain columns, and both verifier-only
objectives appear in every panel.

Two decisions about the encoding:

- Each bar is *stacked into the two halves of ``distinct@8``*: the first correct
  mode (``pass@8``) below, and everything beyond it (``distinct@8 - pass@8``)
  above. That is the whole claim in one mark --- the lower segment grows while
  the upper one goes to the floor --- and it works because both halves are the
  same unit, verified modes per prompt, and sum to the metric the paper is
  about. Drawing correctness and breadth as two separate panels would make the
  reader do that addition in their head.
- Colour names the *metric*, never an arm: ``paper_style.METRIC`` for the first
  correct mode, ``ADD_ON`` for the extra modes. Both are metric slots under the
  rule recorded in ``paper_style``, and it holds here because no arm is drawn in
  its own colour anywhere in this figure --- Dr.GRPO and GRPO are separated
  positionally and labelled under the axis. The pair meets as touching segments
  of a stacked bar, which is what the validated ``{METRIC, METHOD, ADD_ON}``
  trio was checked for under ``--pairs all``.

The retained-breadth badge over each arm is printed only where pass-0 breadth
clears the builder's floor. Three of the five domains start with essentially no
extra modes, and a percentage against .003 modes per prompt would invent a
result out of measurement noise.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))

import paper_style as style  # noqa: E402

DEFAULT_INPUT = ROOT / "paper/results/baseline_collapse_precheck.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/baseline_collapse_precheck"
SCALES = ("qwen05b", "falcon1b", "qwen3b")
SCALE_LABEL = {
    "qwen05b": "Qwen2.5-0.5B",
    "falcon1b": "Falcon3-1B",
    "qwen3b": "Qwen2.5-3B",
}
DOMAIN_ORDER = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
DOMAIN_LABEL = {
    "graph_coloring": "Graph", "countdown": "Countdown",
    "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "PantryPlan",
}
ARMS = ("drgrpo", "grpo")
ARM_LABEL = {"drgrpo": "Dr.GRPO", "grpo": "GRPO"}
# Two bars per arm (pass 0, pass 8), the arms held apart by a wider gap than
# separates the endpoints inside one arm, so the eye groups by arm first.
BAR_WIDTH = 0.78
BAR_X = {("drgrpo", "pass0"): 0.0, ("drgrpo", "pass8"): 0.86,
         ("grpo", "pass0"): 2.14, ("grpo", "pass8"): 3.0}
FIRST_MODE_COLOR = style.METRIC
EXTRA_MODE_COLOR = style.ADD_ON


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_centre(arm: str) -> float:
    """x midpoint of one arm's pass-0/pass-8 pair."""

    return (BAR_X[(arm, "pass0")] + BAR_X[(arm, "pass8")]) / 2


def cell(payload: dict[str, Any], arm: str, scale: str, domain: str):
    return payload["arms"][arm]["scales"][scale]["domains"].get(domain)


def retained(payload: dict[str, Any], record: dict[str, Any]) -> float | None:
    """Terminal extra modes over initial extra modes, or None under the floor."""

    origin = record["endpoints"]["pass0"]["extra_modes"]["mean"]
    if origin <= payload["breadth_floor"]:
        return None
    return record["endpoints"]["pass8"]["extra_modes"]["mean"] / origin


def column_limit(payload: dict[str, Any], domain: str) -> float:
    """One y limit per domain column, so scale rows are directly comparable."""

    tops = [0.0]
    for arm in ARMS:
        for scale in SCALES:
            record = cell(payload, arm, scale, domain)
            if record is None:
                continue
            for endpoint in ("pass0", "pass8"):
                block = record["endpoints"][endpoint]
                tops.append(block["distinct8"]["mean"])
                tops.append(block["distinct8"]["range"][1])
    return max(tops) * 1.30 or 1.0


def draw_panel(axis, payload: dict[str, Any], scale: str, domain: str) -> str | None:
    """Stack both halves of distinct@8 for both arms at both endpoints."""

    missing = []
    for arm in ARMS:
        record = cell(payload, arm, scale, domain)
        if record is None:
            missing.append(ARM_LABEL[arm])
            continue
        for endpoint in ("pass0", "pass8"):
            block = record["endpoints"][endpoint]
            first = block["pass8"]["mean"]
            extra = block["extra_modes"]["mean"]
            x = BAR_X[(arm, endpoint)]
            axis.bar(
                x, first, width=BAR_WIDTH, color=FIRST_MODE_COLOR,
                edgecolor=style.WHITE, linewidth=0.35, zorder=3,
            )
            axis.bar(
                x, extra, width=BAR_WIDTH, bottom=first, color=EXTRA_MODE_COLOR,
                edgecolor=style.WHITE, linewidth=0.35, zorder=3,
            )
            # Seed spread on the total, drawn as a thin whisker rather than a
            # cap-and-bar: at n=5 this is a range, not an interval, and it must
            # not read like one.
            low, high = block["distinct8"]["range"]
            if high > low:
                axis.plot(
                    (x, x), (low, high), color=style.INK, linewidth=0.6,
                    solid_capstyle="butt", zorder=4,
                )
        # Only the reportable cells get a badge. The floor cells used to print
        # "breadth at floor" here, which put eighteen two-line notes on the
        # grid to say that nothing happened, and they crowded out the six
        # numbers that carry the result. The caption states the floor instead.
        share = retained(payload, record)
        if share is not None:
            axis.annotate(
                f"{-100 * (1 - share):+.0f}%", (arm_centre(arm), 1.0),
                xycoords=("data", "axes fraction"), ha="center", va="top",
                fontsize=style.SMALL_FONT, color=style.INK,
            )
    axis.set_xlim(-0.75, 3.75)
    axis.set_xticks([BAR_X[(arm, endpoint)] for arm in ARMS
                     for endpoint in ("pass0", "pass8")])
    axis.set_xticklabels(["0", "8", "0", "8"], fontsize=style.SMALL_FONT)
    # The arm names sit on a second tick level rather than being spaced into
    # one xlabel string: a padded label only lines up at one figure width, and
    # this grid gets re-rendered at whatever width the template asks for.
    axis.set_xticks([arm_centre(arm) for arm in ARMS], minor=True)
    axis.set_xticklabels(
        [ARM_LABEL[arm] for arm in ARMS], minor=True,
        fontsize=style.SMALL_FONT, color=style.INK,
    )
    axis.tick_params(axis="x", which="minor", length=0, pad=13)
    return ", ".join(missing) or None


def render(payload: dict[str, Any], output: Path) -> dict[str, Any]:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(SCALES), len(DOMAIN_ORDER),
        figsize=(style.WIDTH, style.panel_height(len(SCALES), per_row=1.42)),
        squeeze=False,
    )
    # One short title only. This carried a two-line subtitle restating the
    # encoding, which at 6.4pt still overran the panel titles beneath it --- and
    # every word of it belonged in the LaTeX caption, where it is not competing
    # with the data for canvas.
    figure.suptitle(
        "RLVR-only training reduces average extra verified modes at every tested scale",
        fontsize=style.TITLE_FONT, color=style.INK, y=0.995,
    )
    # The metric name is written once, down the left edge. Repeating it inside
    # each row's ylabel put three two-line rotated labels in a column narrower
    # than the text, and they overprinted each other.
    figure.text(
        0.008, 0.53, "verified modes per prompt", rotation=90,
        ha="left", va="center", fontsize=style.LABEL_FONT, color=style.INK,
    )

    limits = {domain: column_limit(payload, domain) for domain in DOMAIN_ORDER}
    blanks: dict[str, str] = {}
    for row, scale in enumerate(SCALES):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row][column]
            style.style_axis(
                axis, grid="y",
                title=DOMAIN_LABEL[domain] if row == 0 else None,
            )
            missing = draw_panel(axis, payload, scale, domain)
            if missing:
                blanks[f"{scale}/{domain}"] = f"no registered cells for {missing}"
            axis.set_ylim(0, limits[domain])
            if column == 0:
                axis.set_ylabel(SCALE_LABEL[scale], fontsize=style.LABEL_FONT)
            # No x label: the two tick levels already read "pass index, then
            # arm", and an xlabel is placed *below* the padded minor labels, so
            # it would print "training pass" between the pass indices and the
            # arm names they do not belong to. The caption carries the axis.
            if row != len(SCALES) - 1:
                axis.tick_params(axis="x", which="minor", labelbottom=False)
    handles = [
        Patch(facecolor=FIRST_MODE_COLOR, edgecolor=style.WHITE,
              label="first correct mode (pass@8)"),
        Patch(facecolor=EXTRA_MODE_COLOR, edgecolor=style.WHITE,
              label="extra verified modes (distinct@8 \u2212 pass@8)"),
    ]
    style.bottom_legend(
        figure, handles, [handle.get_label() for handle in handles],
        y=0.0, ncol=2,
    )
    figure.subplots_adjust(
        left=0.105, right=0.995, top=0.900, bottom=0.145,
        hspace=0.42, wspace=0.30,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)
    return blanks


def write_provenance(
    payload: dict[str, Any], output: Path, input_path: Path,
    blanks: dict[str, str],
) -> None:
    panels: dict[str, Any] = {}
    for scale in SCALES:
        for domain in DOMAIN_ORDER:
            entry: dict[str, Any] = {}
            for arm in ARMS:
                record = cell(payload, arm, scale, domain)
                if record is None:
                    entry[arm] = {"status": "blank", "reason": blanks.get(
                        f"{scale}/{domain}", "no registered cells")}
                    continue
                share = retained(payload, record)
                entry[arm] = {
                    "status": "available",
                    "seeds": sorted(int(seed) for seed in record["per_seed"]),
                    "pass0": {
                        metric: record["endpoints"]["pass0"][metric]["mean"]
                        for metric in ("pass8", "distinct8", "extra_modes")
                    },
                    "pass8": {
                        metric: record["endpoints"]["pass8"][metric]["mean"]
                        for metric in ("pass8", "distinct8", "extra_modes")
                    },
                    "paired_change": {
                        metric: record["paired_change"][metric]
                        for metric in ("pass8", "distinct8", "extra_modes")
                    },
                    "extra_modes_retained_fraction": share,
                    "retained_fraction_reportable": share is not None,
                }
            panels[f"{scale}/{domain}"] = entry
    provenance = {
        "schema": "paper-baseline-collapse-precheck-figure-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figure_key": "baseline-collapse-precheck",
        "comparison": (
            "unmodified Dr.GRPO and unmodified GRPO, pass 0 against pass 8, on "
            "five ModeBench domains at three model scales"
        ),
        "layout": "three physical model rows by five static-domain columns",
        "encoding": (
            "each bar stacks pass@8 under (distinct@8 - pass@8); colour names "
            "the metric, arms are positional; whisker is the seed range of "
            "distinct@8"
        ),
        "metric": "verified modes per prompt at K=8",
        "arms": [ARM_LABEL[arm] for arm in ARMS],
        "breadth_floor": payload["breadth_floor"],
        "breadth_floor_note": payload["breadth_floor_note"],
        "selection_rule": payload["selection_rule"],
        "source": {
            "path": str(input_path.relative_to(ROOT)),
            "sha256": sha256(input_path),
            "schema": payload["schema"],
        },
        "ledgers": payload["ledgers"],
        "pass0_arm_agreement": payload["pass0_arm_agreement"],
        "macro_by_scale": {
            arm: {
                scale: payload["arms"][arm]["scales"][scale]["macro"]
                for scale in SCALES
            }
            for arm in ARMS
        },
        "panels": panels,
    }
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(
            f"{args.input} is missing; run "
            "ops/exp_scaling/build_paper_baseline_collapse_precheck.py first"
        )
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("schema") != "paper-baseline-collapse-precheck-v1":
        raise SystemExit(f"unexpected precheck schema: {payload.get('schema')!r}")
    blanks = render(payload, args.output)
    write_provenance(payload, args.output, args.input, blanks)
    print(f"wrote {args.output.with_suffix('.pdf').relative_to(ROOT)}")
    print(f"wrote {args.output.with_suffix('.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
