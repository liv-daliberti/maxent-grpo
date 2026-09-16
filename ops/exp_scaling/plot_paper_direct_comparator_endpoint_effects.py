#!/usr/bin/env python3
"""Plot exact terminal direct-comparator effects versus matched Dr.GRPO.

The figure deliberately mirrors ``cross_scale_terminal_endpoint_effects``:
raw paired seeds are circles, a diamond and paired Student-t interval appear
only for a complete five-seed block, and the two reported effects are pass@8
and correctness-adjusted breadth.  Unlike the terminal Re:Dr forest,
this comparator view also admits exact terminal prefixes.  Their sample size is
printed in-panel and they never receive a mean marker or interval.
"""

from __future__ import annotations

import argparse
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
import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402


DEFAULT_OUTPUT = ROOT / "paper/figures/direct_comparator_endpoint_effects"
MODELS = ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")
MODEL_SHORT = {
    "Qwen2.5-0.5B": "Qwen 0.5B",
    "Falcon3-1B": "Falcon 1B",
    "Qwen2.5-3B": "Qwen 3B",
}
DOMAIN_ORDER = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
DOMAIN_LABELS = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
COMPARATORS = ("grpo", "ucpo", "rlep_dr")
COMPARATOR_SHORT = {"grpo": "G", "ucpo": "U", "rlep_dr": "R"}
COMPARATOR_LABELS = {
    key: method_visuals.method_style(key)["label"] for key in COMPARATORS
}
METRIC_FIELDS = {
    "pass8": "any_correct_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}
EXPECTED_DRAWS = 4
T_CRIT_DF4 = 2.7764451051977987

SCALE_SPECS: dict[str, dict[str, Any]] = {
    "qwen05b": {
        "model": "Qwen2.5-0.5B",
        "seeds": (43, 44, 45, 46, 47),
        "baseline": ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
        "comparators": {
            "grpo": (ROOT / "var/artifacts/e95_plain_grpo_Qwen25-05B_jobs.json",),
            "ucpo": (
                ROOT / "var/artifacts/e97_ucpo_05b_jobs.json",
                ROOT / "var/artifacts/e115_ucpo_qwen05b_domain_extension_jobs.json",
            ),
            "rlep_dr": (
                ROOT / "var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json",
                ROOT / "var/artifacts/e116_sparse_rlep_qwen05b_domain_extension_jobs.json",
            ),
        },
    },
    "falcon1b": {
        "model": "Falcon3-1B",
        "seeds": (55, 56, 57, 58, 59),
        "baseline": ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
        "comparators": {
            "grpo": (ROOT / "var/artifacts/e95_plain_grpo_Falcon3-1B_jobs.json",),
            "ucpo": (ROOT / "var/artifacts/e99_ucpo_falcon1b_jobs.json",),
            "rlep_dr": (ROOT / "var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json",),
        },
    },
    "qwen3b": {
        "model": "Qwen2.5-3B",
        "seeds": (70, 71, 72, 73, 74),
        "baseline": ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
        "comparators": {
            "grpo": (
                ROOT / "var/artifacts/e95_plain_grpo_Qwen25-3B_jobs.json",
                ROOT / "var/artifacts/e114_plain_grpo_qwen3b_extension_jobs.json",
            ),
            "ucpo": (ROOT / "var/artifacts/e115_ucpo_qwen3b_jobs.json",),
            "rlep_dr": (ROOT / "var/artifacts/e116_sparse_rlep_qwen3b_jobs.json",),
        },
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _completion_marker(run_dir: Path, *, target: int) -> Path | None:
    marker = run_dir / "TRAINING_COMPLETE.json"
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    terminal_step = payload.get("terminal_step")
    if not isinstance(terminal_step, int) or terminal_step < target:
        raise RuntimeError(
            f"{marker}: terminal step {terminal_step!r} is below {target}"
        )
    return marker


def _finite(value: Any, *, where: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{where}: expected finite numeric value, got {value!r}")
    return float(value)


def _reverse_lines(path: Path):
    """Yield complete binary lines from the end without decoding the full log."""

    block_size = 1 << 20
    with path.open("rb") as handle:
        position = handle.seek(0, 2)
        remainder = b""
        while position:
            size = min(block_size, position)
            position -= size
            handle.seek(position)
            data = handle.read(size) + remainder
            lines = data.split(b"\n")
            if position:
                remainder = lines[0]
                complete = lines[1:]
            else:
                remainder = b""
                complete = lines
            for raw in reversed(complete):
                if raw:
                    yield raw


def _terminal_metrics(
    run_dir: Path,
    *,
    target: int,
) -> tuple[dict[str, float], set[Path]]:
    """Read the exact four registered terminal draws and their source files."""

    records: dict[int, tuple[dict[str, Any], Path]] = {}
    paths = sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl"))
    if not paths:
        raise RuntimeError(f"{run_dir}: no sampled-evaluation log")
    for path in paths:
        path_records: dict[int, tuple[dict[str, Any], Path]] = {}
        for raw in _reverse_lines(path):
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if (
                row.get("evaluation_kind") == "fixed_seed_sampled_k_neutral"
                and row.get("step") == target
                and isinstance(row.get("draw_index"), int)
                and row["draw_index"] in range(EXPECTED_DRAWS)
                and isinstance(row.get("metrics"), dict)
            ):
                draw = int(row["draw_index"])
                path_records.setdefault(draw, (row["metrics"], path))
                if len(path_records) == EXPECTED_DRAWS:
                    break
        records.update(path_records)
    if tuple(sorted(records)) != tuple(range(EXPECTED_DRAWS)):
        raise RuntimeError(
            f"{run_dir}: terminal draw indexes {sorted(records)}, expected 0--3"
        )
    metrics = {
        output_name: statistics.fmean(
            _finite(
                records[draw][0].get(source_name),
                where=f"{run_dir}:{target}:draw{draw}:{source_name}",
            )
            for draw in range(EXPECTED_DRAWS)
        )
        for output_name, source_name in METRIC_FIELDS.items()
    }
    return metrics, {records[draw][1] for draw in range(EXPECTED_DRAWS)}


def _runs(
    ledger: dict[str, Any],
    *,
    arm: str | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for run in ledger.get("runs", []):
        if arm is not None and str(run.get("arm")) != arm:
            continue
        key = (str(run["domain"]), int(run["seed"]))
        if key in output:
            raise RuntimeError(f"duplicate ledger cell {key}")
        output[key] = run
    return output


def _interval(values: list[float]) -> dict[str, Any]:
    if len(values) != 5:
        raise RuntimeError(f"interval requires five paired seeds, got {len(values)}")
    mean = statistics.fmean(values)
    half = T_CRIT_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "student_t_95": [mean - half, mean + half],
        "range": [min(values), max(values)],
    }


def _effect(method: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "pass8": method["pass8"] - baseline["pass8"],
        "adjusted_breadth8": (
            method["distinct8"]
            - method["pass8"]
            - baseline["distinct8"]
            + baseline["pass8"]
        ),
    }


def build() -> dict[str, Any]:
    """Assemble every exact terminal comparator/baseline seed pair."""

    source_paths: set[Path] = set()
    cells: list[dict[str, Any]] = []
    for scale_key, spec in SCALE_SPECS.items():
        baseline_path = Path(spec["baseline"])
        baseline_ledger = json.loads(baseline_path.read_text(encoding="utf-8"))
        source_paths.add(baseline_path)
        baseline_target = int(baseline_ledger["target_steps"])
        baseline_runs = _runs(baseline_ledger, arm="control")

        comparator_runs_by_method: dict[
            str, dict[tuple[str, int], dict[str, Any]]
        ] = {}
        for method, ledger_paths_raw in spec["comparators"].items():
            merged_runs: dict[tuple[str, int], dict[str, Any]] = {}
            for ledger_path_raw in ledger_paths_raw:
                ledger_path = Path(ledger_path_raw)
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                if int(ledger["target_steps"]) != baseline_target:
                    raise RuntimeError(
                        f"{ledger_path}: target differs from baseline"
                    )
                run_map = _runs(ledger)
                overlap = merged_runs.keys() & run_map.keys()
                if overlap:
                    raise RuntimeError(
                        f"{method}: duplicate registered cells {sorted(overlap)}"
                    )
                merged_runs.update(run_map)
                source_paths.add(ledger_path)
            comparator_runs_by_method[method] = merged_runs

        for domain in DOMAIN_ORDER:
            method_records: dict[str, Any] = {}
            blank_methods: dict[str, str] = {}
            for method in COMPARATORS:
                if method not in comparator_runs_by_method:
                    blank_methods[method] = "comparator not registered at this scale"
                    continue
                comparator_runs = comparator_runs_by_method[method]
                per_seed: dict[str, Any] = {}
                for seed in spec["seeds"]:
                    run = comparator_runs.get((domain, seed))
                    if run is None:
                        continue
                    run_dir = Path(str(run["run_dir"]))
                    marker = _completion_marker(run_dir, target=baseline_target)
                    if marker is None:
                        continue
                    baseline_run = baseline_runs.get((domain, seed))
                    if baseline_run is None:
                        raise RuntimeError(
                            f"{scale_key}/{domain}/seed {seed}: no matched baseline"
                        )
                    baseline_dir = Path(str(baseline_run["run_dir"]))
                    baseline_marker = _completion_marker(
                        baseline_dir, target=baseline_target
                    )
                    if baseline_marker is None:
                        raise RuntimeError(
                            f"{scale_key}/{domain}/seed {seed}: comparator terminal "
                            "but matched baseline is not terminal"
                        )
                    method_endpoint, method_sources = _terminal_metrics(
                        run_dir, target=baseline_target
                    )
                    baseline_endpoint, baseline_sources = _terminal_metrics(
                        baseline_dir, target=baseline_target
                    )
                    source_paths.update(
                        method_sources
                        | baseline_sources
                        | {marker, baseline_marker}
                    )
                    per_seed[str(seed)] = {
                        "baseline": baseline_endpoint,
                        "comparator": method_endpoint,
                        "effect": _effect(method_endpoint, baseline_endpoint),
                        "comparator_job_id": int(run["job_id"]),
                        "baseline_job_id": int(baseline_run["job_id"]),
                    }
                if not per_seed:
                    blank_methods[method] = (
                        "no terminal comparator seed paired to a terminal baseline"
                    )
                    continue
                seeds = sorted(int(seed) for seed in per_seed)
                record: dict[str, Any] = {
                    "label": COMPARATOR_LABELS[method],
                    "n": len(seeds),
                    "seeds": seeds,
                    "evidence": (
                        "balanced_five_seed_terminal"
                        if len(seeds) == 5
                        else "exact_terminal_prefix"
                    ),
                    "per_seed": per_seed,
                }
                if len(seeds) == 5:
                    record["summaries"] = {
                        metric: _interval(
                            [per_seed[str(seed)]["effect"][metric] for seed in seeds]
                        )
                        for metric in ("pass8", "adjusted_breadth8")
                    }
                method_records[method] = record
            cells.append(
                {
                    "scale": scale_key,
                    "model": spec["model"],
                    "domain": domain,
                    "methods": method_records,
                    "blank_methods": blank_methods,
                }
            )

    expected_cells = len(MODELS) * len(DOMAIN_ORDER)
    if len(cells) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} cells, got {len(cells)}")
    if [cell["model"] for cell in cells[:: len(DOMAIN_ORDER)]] != list(MODELS):
        raise RuntimeError("model row order drifted")

    return {
        "schema": "paper-direct-comparator-endpoint-effects-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "mixed terminal balanced blocks and exact terminal prefixes; "
            "only n=5 receives a mean and paired 95% Student-t interval"
        ),
        "selection_rule": (
            "all registered comparator seeds with a terminal completion marker "
            "and four complete pass-8 sampled-evaluation draws, paired to the "
            "same seed's terminal matched Dr.GRPO control"
        ),
        "model_rows": list(MODELS),
        "domain_order": list(DOMAIN_ORDER),
        "comparators": list(COMPARATORS),
        "baseline": "matched Dr.GRPO",
        "metrics": {
            "pass8": "comparator pass@8 minus matched Dr.GRPO pass@8",
            "adjusted_breadth8": (
                "comparator (distinct@8-pass@8) minus matched Dr.GRPO "
                "(distinct@8-pass@8)"
            ),
        },
        "evidence_encoding": {
            "paired_seed": "open circle",
            "balanced_five_seed_mean": "filled diamond",
            "balanced_five_seed_interval": "paired two-sided 95% Student-t, df=4",
            "exact_terminal_prefix": "open circles only; no mean or interval",
        },
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
    """Render in the same grid and effect language as the core endpoint forest."""

    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(MODELS),
        len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 5.35),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    metrics = (
        ("pass8", r"$\Delta P$"),
        ("adjusted_breadth8", r"$\Delta(D-P)$"),
    )
    x_positions = (0.0, 1.0)
    method_offsets = {"grpo": -0.14, "ucpo": 0.0, "rlep_dr": 0.14}
    cells = {
        (cell["model"], cell["domain"]): cell for cell in payload["cells"]
    }
    plotted_values = [0.0]
    for cell in payload["cells"]:
        for method in cell["methods"].values():
            for seed in method["seeds"]:
                plotted_values.extend(
                    method["per_seed"][str(seed)]["effect"].values()
                )
            for summary in method.get("summaries", {}).values():
                plotted_values.extend(summary["student_t_95"])
    y_low = math.floor((min(plotted_values) - 0.12) * 10) / 10
    y_high = math.ceil((max(plotted_values) + 0.12) * 10) / 10

    def cell_counts(cell: dict[str, Any]) -> str:
        return " · ".join(
            f"{COMPARATOR_SHORT[key]} {cell['methods'][key]['n']}"
            for key in COMPARATORS
            if key in cell["methods"]
        )

    # Twelve of the fifteen panels carry the same seed counts. Say that once in
    # the subtitle and badge only the panels that differ, so the badge marks a
    # departure instead of blending into fifteen identical stamps.
    shared_counts = style.dominant_note(
        cell_counts(cell) for cell in payload["cells"] if cell_counts(cell)
    )

    for model_index, model in enumerate(MODELS):
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            axis = axes[model_index][domain_index]
            style.style_axis(
                axis,
                grid="both",
                title=DOMAIN_LABELS[domain] if model_index == 0 else None,
            )
            axis.axhline(0.0, color=style.MUTED, lw=0.8, linestyle=(0, (2, 2)))
            cell = cells[(model, domain)]
            for method_key in COMPARATORS:
                method = cell["methods"].get(method_key)
                if method is None:
                    continue
                visual = method_visuals.method_style(method_key)
                seed_count = len(method["seeds"])
                seed_jitter = [
                    (index - (seed_count - 1) / 2) * 0.018
                    for index in range(seed_count)
                ]
                for x, (metric, _label) in zip(x_positions, metrics):
                    center = x + method_offsets[method_key]
                    values = [
                        method["per_seed"][str(seed)]["effect"][metric]
                        for seed in method["seeds"]
                    ]
                    axis.scatter(
                        [center + offset for offset in seed_jitter],
                        values,
                        s=9,
                        marker=visual["marker"],
                        facecolor=style.WHITE,
                        edgecolor=visual["color"],
                        linewidth=0.65,
                        zorder=3,
                    )
                    summary = method.get("summaries", {}).get(metric)
                    if summary is None:
                        continue
                    low, high = summary["student_t_95"]
                    axis.vlines(
                        center,
                        low,
                        high,
                        color=visual["color"],
                        linewidth=1.35,
                        zorder=4,
                    )
                    axis.scatter(
                        [center],
                        [summary["mean"]],
                        s=25,
                        marker=visual["marker"],
                        color=visual["color"],
                        edgecolor=style.WHITE,
                        linewidth=0.5,
                        zorder=5,
                    )
            counts = cell_counts(cell)
            if counts and counts != shared_counts:
                axis.text(
                    0.03,
                    0.96,
                    f"n: {counts}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=5.2,
                    color=style.MUTED,
                    zorder=6,
                )
            axis.set_xlim(-0.42, 1.42)
            axis.set_ylim(y_low, y_high)
            axis.set_xticks(x_positions, [item[1] for item in metrics])
            if domain_index == 0:
                axis.set_ylabel(
                    f"{MODEL_SHORT[model]}\neffect vs Dr.GRPO",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=method_visuals.method_style(key)["marker"],
            color="none",
            markerfacecolor=method_visuals.method_style(key)["color"],
            markeredgecolor=style.WHITE,
            markeredgewidth=0.5,
            markersize=4.6,
        )
        for key in COMPARATORS
    ]
    evidence_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=style.WHITE,
            markeredgecolor=style.MUTED,
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=style.MUTED,
            markeredgecolor=style.WHITE,
            markersize=4,
        ),
    ]
    style.bottom_legend(
        figure,
        method_handles + evidence_handles,
        [COMPARATOR_LABELS[key] for key in COMPARATORS]
        + ["paired seed", "n=5 mean + 95% interval"],
        y=0.003,
        ncol=5,
    )
    figure.suptitle(
        "Direct comparator endpoint effects vs matched Dr.GRPO",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.96,
        (
            "Pass 8 only · exact terminal paired seeds; diamonds and intervals "
            "require all five registered seeds."
            + (f"  All panels n: {shared_counts} unless marked." if shared_counts else "")
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.85,
        bottom=0.16,
        hspace=0.30,
        wspace=0.20,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    payload = build()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    render(payload, output)
    print(f"wrote {output.with_suffix('.json')}")
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
