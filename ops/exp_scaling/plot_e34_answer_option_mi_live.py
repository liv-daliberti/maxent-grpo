#!/usr/bin/env python3
"""Render the live E34 Dr.GRPO-versus-answer-option-MI pilot with gnuplot.

This deliberately uses the system gnuplot rather than matplotlib so the live
monitor can always publish the pilot plot from the lightweight login-node
environment.  E34 has one training seed; error bars therefore describe only
the four fixed K=8 evaluation draws, never training-seed uncertainty.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = ROOT / "var/artifacts"
OUT = ARTIFACTS / "e34_answer_option_mi_05b_live.png"
T_975_DF3 = 3.182446305284263
TASKS = (
    (
        "Countdown",
        ARTIFACTS / "cde34_answer_option_mi_05b_pilot_v2_scaling_curve.json",
    ),
    (
        "Graph coloring",
        ARTIFACTS / "gce34_answer_option_mi_05b_pilot_v2_scaling_curve.json",
    ),
)
METRICS = (
    ("greedy", "pass@1", 1.0),
    ("mean8", "mean@8", 1.0),
    ("pass8", "pass@8", 1.0),
    ("coverage8", "coverage@8", 1.0),
    ("distinct8", "distinct@8", 8.0),
)
ARMS = (
    ("grpo", "Dr.GRPO", "#3f3f3f", 2),
    ("diayn", "Answer-option MI", "#D55E00", 1),
)


def _load(path: Path) -> dict[str, list[dict[str, object]]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        rows = []
    by_arm = {arm: [] for arm, *_ in ARMS}
    for row in rows:
        arm = row.get("arm")
        if row.get("split") != "multi_answer" or arm not in by_arm:
            continue
        passes = row.get("training_passes")
        if passes is None or float(passes) > 5.0 + 1e-9:
            continue
        by_arm[arm].append(row)
    for arm_rows in by_arm.values():
        arm_rows.sort(key=lambda row: float(row["training_passes"]))
    return by_arm


def _interval(row: dict[str, object], metric: str, upper: float) -> tuple[float, float]:
    value = row.get(metric)
    draws = row.get(f"{metric}_draws") or []
    if value is None or len(draws) != 4:
        return math.nan, math.nan
    draw_values = [float(item) for item in draws]
    center = statistics.fmean(draw_values)
    margin = T_975_DF3 * statistics.stdev(draw_values) / math.sqrt(4)
    return max(0.0, center - margin), min(upper, center + margin)


def _block(
    name: str,
    rows: list[dict[str, object]],
    metric: str,
    upper: float,
) -> str:
    lines = [f"${name} << EOD"]
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        low, high = _interval(row, metric, upper)
        lines.append(
            f"{float(row['training_passes']):.9g} {float(value):.9g} "
            f"{low:.9g} {high:.9g}"
        )
    lines.append("EOD")
    return "\n".join(lines)


def _quoted(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data = [(task, _load(path)) for task, path in TASKS]
    temporary = OUT.with_name(f".{OUT.stem}.{os.getpid()}.tmp{OUT.suffix}")
    commands = [
        "set terminal pngcairo size 2400,1000 enhanced font 'DejaVu Sans,13'",
        f"set output '{_quoted(temporary)}'",
        "set multiplot layout 2,5 rowsfirst title 'E34 0.5B free-form pilot: Dr.GRPO vs answer-option MI (seed 3401; error bars = four fixed K=8 draws)' font ',18'",
        "set xrange [0:5]",
        "set xtics 1",
        "set grid ytics lc rgb '#dddddd' lw 1",
        "set border 3",
        "set tics nomirror",
        "set key top left reverse Left samplen 2",
    ]
    block_index = 0
    for task_index, (task, by_arm) in enumerate(data):
        for metric_index, (metric, title, upper) in enumerate(METRICS):
            blocks = []
            plots = []
            for arm, label, color, dash_type in ARMS:
                block_name = f"D{block_index}"
                block_index += 1
                blocks.append(_block(block_name, by_arm[arm], metric, upper))
                plots.extend(
                    [
                        f"${block_name} using 1:2:3:4 with yerrorbars lc rgb '{color}' pt 7 ps 0.35 lw 1 notitle",
                        f"${block_name} using 1:2 with linespoints lc rgb '{color}' dt {dash_type} pt 7 ps 0.55 lw 2 title '{label}'",
                    ]
                )
            commands.extend(blocks)
            commands.append(f"set title '{title}'")
            commands.append(f"set yrange [0:{upper}]")
            commands.append(f"set ylabel '{task}'" if metric_index == 0 else "unset ylabel")
            commands.append("set xlabel 'training passes'" if task_index == 1 else "unset xlabel")
            commands.append("set key top left" if task_index == 0 and metric_index == 0 else "unset key")
            commands.append("plot " + ", ".join(plots))
    commands.extend(
        [
            "unset multiplot",
            "unset output",
        ]
    )
    try:
        subprocess.run(
            ["gnuplot"],
            input="\n".join(commands) + "\n",
            text=True,
            cwd=ROOT,
            check=True,
        )
        temporary.replace(OUT)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
