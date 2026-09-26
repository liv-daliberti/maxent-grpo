#!/usr/bin/env python3
"""Render an explicitly interim Figure 4 preview from live E78 evaluations.

This is an internal monitoring view, not a reportable result. It reads only
registered half-pass evaluations and never substitutes training progress for
evaluation outcomes. Thin lines are every currently available seed; thick
lines and ranges use only seeds for which both arms are present at that
checkpoint. If its separate prospective ledger exists, PointMaze is appended
as the sixth panel rather than retroactively inserted into E78's estimator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import re
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
import paper_style as style  # noqa: E402


DEFAULT_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
POINT_LEDGER = (
    ROOT / "var/artifacts/e78pm_point_maze_verified_replay_only_05b_jobs.json"
)
DEFAULT_OUTPUT = ROOT / "var/artifacts/e78_figure4_interim"
from status_e78 import DOMAIN_TITLES  # noqa: E402,F401
ARM_STYLE = {
    "control": (style.CONTROL, style.ARM_DASH[style.CONTROL], "matched Dr.GRPO"),
    "replay": (style.METHOD, style.ARM_DASH[style.METHOD], "Re:Dr"),
    # E81/E82 add one arm on top of `replay`; E83 adds the same term on top of
    # `control` instead. Both are add-one arms, so each carries its own colour
    # and dash rather than a shade of the arm it extends.
    "semantic": (
        style.ABLATION,
        style.ARM_DASH[style.ABLATION],
        "Re:Dr + Semantic MaxEnt",
    ),
    # E90 is verified replay with the dose set by bank occupancy rather than by
    # a fixed coefficient. It is the same mechanism at a different dose, so it
    # keeps the method hue and separates on dash; a sixth hue would read as an
    # unrelated treatment.
    "bank_normalized_replay": (
        style.METHOD,
        style.METHOD_DOSE_DASH,
        "Adaptive Re:Dr",
    ),
    "semantic_only": (
        style.ADD_ON,
        style.ARM_DASH[style.ADD_ON],
        "Semantic MaxEnt (no replay)",
    ),
    # The adapted-coefficient variant of `semantic`, read against the same
    # comparator, so it shares that arm's hue family and differs by dash.
    "adaptive_semantic": (
        style.ADAPTIVE,
        style.ARM_DASH[style.ADAPTIVE],
        "Adaptive Semantic MaxEnt + Re:Dr",
    ),
}
EXPECTED_DRAWS = 4


def _run_curve(run_dir: Path, *, interval: int, target: int) -> dict[int, float]:
    """Return complete four-draw distinct@8 means at registered checkpoints."""

    records: dict[tuple[int, int], float] = {}
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except (json.JSONDecodeError, MemoryError):
                    continue
                if row.get("evaluation_kind") != "fixed_seed_sampled_k_neutral":
                    continue
                step = row.get("step")
                draw = row.get("draw_index")
                metrics = row.get("metrics")
                if not isinstance(step, int) or not isinstance(draw, int):
                    continue
                if step < 0 or step > target or step % interval:
                    continue
                if not isinstance(metrics, dict):
                    continue
                value = metrics.get("distinct_correct_modes_at_k")
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                records[(step, draw)] = float(value)

    by_step: dict[int, list[float]] = defaultdict(list)
    for (step, _draw), value in records.items():
        by_step[step].append(value)
    return {
        step: sum(values) / len(values)
        for step, values in by_step.items()
        if len(values) == EXPECTED_DRAWS
    }


from status_e78 import POINT_EVAL_SCHEMA  # noqa: E402


def _point_curve(
    metrics_path: Path,
    *,
    interval: int,
    target: int,
    point_steps_per_pass: int,
    common_steps_per_pass: int,
) -> dict[int, float]:
    """Map registered PointMaze updates onto E78's common pass coordinate."""

    result = {}
    try:
        handle = metrics_path.open(encoding="utf-8", errors="replace")
    except OSError:
        return result
    with handle:
        for raw in handle:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if POINT_EVAL_SCHEMA.match(str(row.get("schema", ""))) is None:
                continue
            update = row.get("learning_round")
            value = row.get("distinct8")
            if (
                not isinstance(update, int)
                or update < 0
                or update > target
                or update % interval
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                continue
            numerator = update * common_steps_per_pass
            if numerator % point_steps_per_pass:
                continue
            result[numerator // point_steps_per_pass] = float(value)
    return result


def _snapshot(ledger_path: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    interval = int(ledger["checkpoint_interval_steps"])
    target = int(ledger["target_steps"])
    curves: dict[str, dict[str, dict[int, dict[int, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for run in ledger["runs"]:
        curve = _run_curve(
            Path(run["run_dir"]), interval=interval, target=target
        )
        if curve:
            curves[str(run["domain"])][str(run["arm"])][int(run["seed"])] = curve

    domains = [str(domain) for domain in ledger["domains"]]
    total_runs = len(ledger["runs"])
    point_included = False
    point_eval_rows = None
    if POINT_LEDGER.is_file():
        point = json.loads(POINT_LEDGER.read_text(encoding="utf-8"))
        for run in point["runs"]:
            curve = _point_curve(
                Path(run["metrics_path"]),
                interval=int(point["checkpoint_interval_steps"]),
                target=int(point["target_steps"]),
                point_steps_per_pass=int(point["train_rows"]),
                common_steps_per_pass=int(ledger["train_rows"]),
            )
            if curve:
                curves["point_maze"][str(run["arm"])][int(run["seed"])] = curve
        domains.append("point_maze")
        total_runs += len(point["runs"])
        point_included = True
        point_eval_rows = int(point["eval_rows"])

    return {
        "curves": curves,
        "domains": domains,
        "interval": interval,
        "passes": int(ledger["passes"]),
        "steps_per_pass": int(ledger["train_rows"]),
        "target": target,
        "total_runs": total_runs,
        "point_extension_included": point_included,
        "point_eval_rows": point_eval_rows,
    }


def _paired_values(
    domain_curves: dict[str, dict[int, dict[int, float]]],
    step: int,
) -> tuple[list[float], list[float], list[int]]:
    control = domain_curves.get("control", {})
    replay = domain_curves.get("replay", {})
    paired = sorted(
        seed
        for seed in set(control) & set(replay)
        if step in control[seed] and step in replay[seed]
    )
    return (
        [control[seed][step] for seed in paired],
        [replay[seed][step] for seed in paired],
        paired,
    )


def render(snapshot: dict[str, Any], output: Path) -> dict[str, Any]:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        1,
        len(snapshot["domains"]),
        figsize=(
            style.WIDTH * (1.18 if len(snapshot["domains"]) == 6 else 1.0),
            2.35,
        ),
        sharex=True,
        sharey=True,
    )
    curves = snapshot["curves"]
    steps_per_pass = snapshot["steps_per_pass"]
    interval = snapshot["interval"]
    registered_steps = list(range(0, snapshot["target"] + 1, interval))
    observed_runs = sum(
        len(arm_curves)
        for domain_curves in curves.values()
        for arm_curves in domain_curves.values()
    )
    provenance: dict[str, Any] = {
        "schema": "e78_figure4_interim_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ledger": str(DEFAULT_LEDGER.resolve()),
        "metric": (
            "static domains: mean across four distinct_correct_modes_at_k draws; "
            "PointMaze: one common-random-number eight-trajectory draw over "
            f"{snapshot['point_eval_rows']} maps"
        ),
        "registered_checkpoint_interval": interval,
        "observed_runs": observed_runs,
        "total_runs": snapshot["total_runs"],
        "domains": {},
        "point_extension_included": snapshot["point_extension_included"],
    }

    maximum = 1.0
    for domain_curves in curves.values():
        for arm_curves in domain_curves.values():
            for curve in arm_curves.values():
                maximum = max(maximum, *curve.values())
    y_high = max(2.0, math.ceil(maximum * 2) / 2 + 0.25)

    for index, (axis, domain) in enumerate(zip(axes, snapshot["domains"])):
        title = DOMAIN_TITLES.get(domain, domain)
        style.style_axis(axis, title=title)
        axis.set_xlim(0, snapshot["passes"])
        axis.set_ylim(0, y_high)
        axis.set_xticks([0, 2, 4, 6, 8])
        axis.axhline(1.0, color=style.MUTED, linewidth=0.65, linestyle=(0, (2, 2)))
        domain_curves = curves.get(domain, {})
        available_steps: set[int] = set()
        domain_record: dict[str, Any] = {"arms": {}, "paired_seeds_by_pass": {}}

        for arm in ("control", "replay"):
            color, dash, _label = ARM_STYLE[arm]
            arm_curves = domain_curves.get(arm, {})
            domain_record["arms"][arm] = sorted(arm_curves)
            for seed, curve in sorted(arm_curves.items()):
                points = sorted(curve.items())
                if not points:
                    continue
                available_steps.update(step for step, _value in points)
                axis.plot(
                    [step / steps_per_pass for step, _value in points],
                    [value for _step, value in points],
                    color=color,
                    linestyle=dash,
                    linewidth=style.SEED_LW,
                    alpha=0.38,
                    zorder=2,
                )

        for arm in ("control", "replay"):
            color, dash, _label = ARM_STYLE[arm]
            xs: list[float] = []
            means: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for step in registered_steps:
                control_values, replay_values, paired = _paired_values(
                    domain_curves, step
                )
                values = control_values if arm == "control" else replay_values
                if not paired:
                    continue
                pass_value = step / steps_per_pass
                domain_record["paired_seeds_by_pass"][str(pass_value)] = paired
                xs.append(pass_value)
                means.append(sum(values) / len(values))
                lows.append(min(values))
                highs.append(max(values))
            if xs:
                axis.fill_between(
                    xs,
                    lows,
                    highs,
                    color=color,
                    alpha=style.BAND_ALPHA,
                    linewidth=0,
                    zorder=1,
                )
                axis.plot(
                    xs,
                    means,
                    color=color,
                    linestyle=dash,
                    linewidth=style.MEAN_LW,
                    zorder=3,
                )

        if available_steps:
            deepest = max(available_steps) / steps_per_pass
            if deepest < snapshot["passes"]:
                axis.axvspan(
                    deepest,
                    snapshot["passes"],
                    color="#F2F4F6",
                    alpha=0.82,
                    linewidth=0,
                    zorder=-2,
                )
            paired_any = sorted(
                set(domain_curves.get("control", {}))
                & set(domain_curves.get("replay", {}))
            )
            if paired_any:
                note = "paired live seeds " + ", ".join(map(str, paired_any))
            else:
                note = "live seeds are not paired yet"
            axis.text(
                0.04,
                0.96,
                note,
                transform=axis.transAxes,
                va="top",
                fontsize=style.SMALL_FONT,
                color=style.MUTED,
            )
        else:
            axis.axvspan(
                0,
                snapshot["passes"],
                color="#F2F4F6",
                alpha=0.82,
                linewidth=0,
                zorder=-2,
            )
            axis.text(
                0.5,
                0.52,
                "awaiting first\nregistered evaluation",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=style.SMALL_FONT,
                color=style.MUTED,
            )
        if index == 0:
            axis.set_ylabel("mean distinct correct@8", fontsize=style.LABEL_FONT)
        axis.set_xlabel("training pass", fontsize=style.LABEL_FONT)
        provenance["domains"][domain] = domain_record

    control_color, control_dash, control_label = ARM_STYLE["control"]
    replay_color, replay_dash, replay_label = ARM_STYLE["replay"]
    handles = [
        Line2D([0], [0], color=control_color, linestyle=control_dash, lw=1.4),
        Line2D([0], [0], color=replay_color, linestyle=replay_dash, lw=1.4),
        Line2D([0], [0], color=style.MUTED, lw=style.SEED_LW, alpha=0.55),
    ]
    labels = [control_label, replay_label, "available seed (thin)"]
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=style.FONT,
        bbox_to_anchor=(0.5, -0.015),
        handlelength=2.7,
    )
    figure.suptitle(
        (
            "INTERIM E78 + PROSPECTIVE POINTMAZE PREVIEW — not a reportable result"
            if snapshot["point_extension_included"]
            else "INTERIM E78 PREVIEW — asynchronous progress, not a reportable result"
        ),
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=1.015,
    )
    figure.text(
        0.5,
        0.955,
        f"{observed_runs}/{snapshot['total_runs']} cells have registered evaluations; "
        "thick curves use paired seeds only",
        ha="center",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(left=0.075, right=0.995, top=0.79, bottom=0.23, wspace=0.20)
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    provenance = render(_snapshot(args.ledger.resolve()), args.output.resolve())
    print(
        f"wrote {args.output.with_suffix('.png')} and "
        f"{args.output.with_suffix('.pdf')} from "
        f"{provenance['observed_runs']}/{provenance['total_runs']} live cells"
    )


if __name__ == "__main__":
    main()
