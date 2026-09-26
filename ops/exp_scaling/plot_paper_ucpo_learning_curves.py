#!/usr/bin/env python3
"""Plot the balanced Dr.GRPO--UCPO--x-Mode Dr.GRPO training trajectories.

The seed selection is read from the frozen paper-facing UCPO result artifact,
so this figure and the adjacent endpoint table always describe the same slice.
Every plotted checkpoint is the mean of the four registered temperature-one,
K=8 evaluation draws; bands span the selected paired seeds.
"""

from __future__ import annotations

from collections import defaultdict
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
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402


UCPO_LEDGER = ROOT / "var/artifacts/e97_ucpo_05b_jobs.json"
CORE_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
RESULT_ARTIFACT = ROOT / "paper/results/ucpo_interim_05b.json"
OUTPUT = ROOT / "paper/figures/ucpo_interim_learning_curves"
EXPECTED_DRAWS = 4
METRICS = {
    "pass8": "any_correct_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}
ARM_SPECS = {
    "control": (style.CONTROL, style.ARM_DASH[style.CONTROL], "matched Dr.GRPO"),
    "ucpo": (style.ABLATION, style.ARM_DASH[style.ABLATION], "UCPO"),
    "xmode": (style.METHOD, style.ARM_DASH[style.METHOD], "x-Mode Dr.GRPO"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_curve(
    run_dir: Path,
    *,
    interval: int,
    target: int,
) -> tuple[dict[int, dict[str, float]], list[Path]]:
    """Return complete four-draw metrics at every registered checkpoint."""

    records: dict[tuple[int, int], dict[str, Any]] = {}
    paths = sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl"))
    if not paths:
        raise RuntimeError(f"{run_dir}: no sampled-evaluation log")
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
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
                if (
                    not isinstance(step, int)
                    or step < 0
                    or step > target
                    or step % interval
                    or not isinstance(draw, int)
                    or draw not in range(EXPECTED_DRAWS)
                    or not isinstance(metrics, dict)
                ):
                    continue
                records[(step, draw)] = metrics

    expected_steps = tuple(range(0, target + 1, interval))
    curve: dict[int, dict[str, float]] = {}
    for step in expected_steps:
        if any((step, draw) not in records for draw in range(EXPECTED_DRAWS)):
            raise RuntimeError(f"{run_dir}: incomplete registered checkpoint {step}")
        curve[step] = {}
        for output_name, source_name in METRICS.items():
            values = [records[(step, draw)].get(source_name) for draw in range(EXPECTED_DRAWS)]
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in values
            ):
                raise RuntimeError(f"{run_dir}:{step}: invalid {source_name}")
            curve[step][output_name] = statistics.fmean(float(value) for value in values)
    return curve, paths


def main() -> None:
    result = json.loads(RESULT_ARTIFACT.read_text(encoding="utf-8"))
    if result.get("schema") != "ucpo_interim_paper_results_v1":
        raise RuntimeError("unexpected UCPO paper-result schema")
    design = result["design"]
    domains = [str(domain) for domain in design["domains"]]
    seeds = [int(seed) for seed in design["balanced_terminal_seeds"]]
    target = int(design["terminal_step"])

    ucpo_ledger = json.loads(UCPO_LEDGER.read_text(encoding="utf-8"))
    core_ledger = json.loads(CORE_LEDGER.read_text(encoding="utf-8"))
    interval = int(ucpo_ledger["checkpoint_interval_steps"])
    steps_per_pass = int(ucpo_ledger["train_rows"])
    if target % steps_per_pass:
        raise RuntimeError("terminal step is not an integer number of training passes")

    ucpo_runs = {
        (str(run["domain"]), int(run["seed"])): run
        for run in ucpo_ledger["runs"]
    }
    core_runs = {
        (str(run["domain"]), str(run["arm"]), int(run["seed"])): run
        for run in core_ledger["runs"]
    }
    curves: dict[str, dict[str, dict[int, dict[int, dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    input_paths: set[Path] = {UCPO_LEDGER, CORE_LEDGER, RESULT_ARTIFACT}
    sources: list[dict[str, Any]] = []
    for domain in domains:
        for seed in seeds:
            arm_runs = {
                "control": core_runs[(domain, "control", seed)],
                "ucpo": ucpo_runs[(domain, seed)],
                "xmode": core_runs[(domain, "replay", seed)],
            }
            for arm, run in arm_runs.items():
                run_dir = Path(run["run_dir"])
                curve, paths = _run_curve(run_dir, interval=interval, target=target)
                curves[domain][arm][seed] = curve
                input_paths.update(paths)
                sources.append(
                    {
                        "domain": domain,
                        "arm": arm,
                        "seed": seed,
                        "run_dir": str(run_dir),
                    }
                )

    summaries: dict[str, Any] = {}
    for domain in domains:
        summaries[domain] = {}
        for arm in ARM_SPECS:
            summaries[domain][arm] = {}
            for step in range(0, target + 1, interval):
                summary: dict[str, Any] = {"training_pass": step / steps_per_pass}
                for metric in METRICS:
                    values = [curves[domain][arm][seed][step][metric] for seed in seeds]
                    summary[metric] = {
                        "mean": statistics.fmean(values),
                        "range": [min(values), max(values)],
                        "per_seed": dict(zip(map(str, seeds), values)),
                    }
                summaries[domain][arm][str(step)] = summary

            terminal = summaries[domain][arm][str(target)]
            expected = result["domains"][domain]["arms"][arm]
            for metric in METRICS:
                if not math.isclose(
                    terminal[metric]["mean"], float(expected[metric]), abs_tol=1e-12
                ):
                    raise RuntimeError(
                        f"{domain}/{arm}/{metric}: plot endpoint disagrees with table"
                    )

    provenance = {
        "schema": "ucpo_interim_learning_curves_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result_artifact": str(RESULT_ARTIFACT),
        "selection_rule": result["selection_rule"],
        "model": design["model"],
        "domains": domains,
        "balanced_terminal_seeds": seeds,
        "checkpoint_interval_steps": interval,
        "steps_per_pass": steps_per_pass,
        "terminal_step": target,
        "evaluation_draws": EXPECTED_DRAWS,
        "summaries": summaries,
        "sources": sources,
        "input_sha256": {
            str(path.resolve()): _sha256(path) for path in sorted(input_paths)
        },
    }

    style.apply_rcparams()
    figure, axes = plt.subplots(
        1,
        len(domains),
        figsize=(style.WIDTH, 2.05),
        sharex=True,
        sharey=True,
    )
    maximum = max(
        summaries[domain][arm][step]["distinct8"]["range"][1]
        for domain in domains
        for arm in ARM_SPECS
        for step in summaries[domain][arm]
    )
    y_top = max(1.0, math.ceil((maximum + 0.15) * 2) / 2)
    pass_ticks = [0, 2, 4, 6, 8]
    for index, domain in enumerate(domains):
        axis = axes[index]
        style.style_axis(axis, title=result["domains"][domain]["title"])
        axis.set_xlim(0, target / steps_per_pass)
        axis.set_xticks(pass_ticks)
        axis.set_ylim(-0.05, y_top)
        axis.set_xlabel("training pass", fontsize=style.LABEL_FONT)
        if index == 0:
            axis.set_ylabel("distinct@8", fontsize=style.LABEL_FONT)
        for arm, (color, dash, _label) in ARM_SPECS.items():
            arm_summary = summaries[domain][arm]
            steps = sorted(map(int, arm_summary))
            xs = [arm_summary[str(step)]["training_pass"] for step in steps]
            means = [arm_summary[str(step)]["distinct8"]["mean"] for step in steps]
            lows = [arm_summary[str(step)]["distinct8"]["range"][0] for step in steps]
            highs = [arm_summary[str(step)]["distinct8"]["range"][1] for step in steps]
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
            axis.scatter(
                [xs[-1]],
                [means[-1]],
                color=color,
                s=8,
                zorder=4,
            )
        axis.text(
            0.04,
            0.95,
            "seeds 43--44",
            transform=axis.transAxes,
            va="top",
            fontsize=style.SMALL_FONT,
            color=style.MUTED,
        )

    handles = [
        Line2D([0], [0], color=color, linestyle=dash, lw=style.MEAN_LW)
        for color, dash, _label in ARM_SPECS.values()
    ]
    labels = [label for _color, _dash, label in ARM_SPECS.values()]
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=style.FONT,
        bbox_to_anchor=(0.5, -0.005),
        handlelength=2.7,
    )
    figure.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.28, wspace=0.16)
    style.save(figure, OUTPUT, png=True, dpi=240)
    plt.close(figure)
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {OUTPUT.with_suffix('.pdf')}, {OUTPUT.with_suffix('.png')}, and "
        f"{OUTPUT.with_suffix('.json')} for balanced seeds {seeds}"
    )


if __name__ == "__main__":
    main()
