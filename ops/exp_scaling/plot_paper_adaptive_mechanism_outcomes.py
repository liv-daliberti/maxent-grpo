#!/usr/bin/env python3
"""Connect adaptive-semantic controller telemetry to outcomes.

The left column retains the failed rho=.05 cohort strictly as mechanism-only
context. For every terminal Qwen2.5-0.5B E89 seed, the other panels pair the
reachable controller's final realized semantic/task RMS ratio and coefficient
with the same seed's terminal effect relative to fixed Semantic MaxEnt +
Re:Dr. Complete five-seed domains receive outcome Student-t intervals;
incomplete domains retain only their exact terminal seeds. Domains are never
pooled.
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
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402
from plot_paper_direct_comparator_endpoint_effects import (  # noqa: E402
    _completion_marker,
    _interval,
    _relative,
    _runs,
    _sha256,
    _terminal_metrics,
)


E89_LEDGER = (
    ROOT / "var/artifacts/e89_adaptive_semantic_maxent_reachable_05b_jobs.json"
)
E89_PROTOCOL = (
    ROOT
    / "paper/preregistration/e89_adaptive_semantic_maxent_reachable_05b_20260810.md"
)
FAILED_GATE = ROOT / "paper/figures/adaptive_semantic_gate_e88.json"
FIXED_RESULT = ROOT / "paper/results/maxent_factorial_05b.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/adaptive_mechanism_outcomes_qwen05b"
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
DOMAIN_COLOR = {
    "graph_coloring": style.METHOD,
    "countdown": style.CONTROL,
    "python_factors": style.ABLATION,
    "mathir": style.ADD_ON,
    "pantry_plan": style.INK,
}
SEEDS = (43, 44, 45, 46, 47)
TARGET = 3072
TARGET_RATIO = 0.015
MAX_BOUND_HIT_FRACTION = 0.20
NOT_ADAPTED_FROZEN_FRACTION = 0.80
FIELDS = (
    "final_realized_ratio",
    "final_coefficient",
    "delta_pass8",
    "delta_adjusted_breadth8",
)
FIELD_LABEL = {
    "final_realized_ratio": "final realized RMS ratio",
    "final_coefficient": r"final coefficient $\eta$",
    "delta_pass8": r"$\Delta$ pass@8",
    "delta_adjusted_breadth8": r"$\Delta(D-P)$ at pass 8",
}
CONTROLLER_KEYS = {
    "final_realized_ratio": "train/semantic_rms_controller_realized_ratio",
    "final_coefficient": "train/semantic_rms_controller_coefficient",
    "bound_hit_fraction": "train/semantic_rms_controller_bound_hit_fraction",
    "frozen_fraction": "train/semantic_rms_controller_frozen_fraction",
    "observations": "train/semantic_rms_controller_observations",
    "updates_applied": "train/semantic_rms_controller_updates_applied",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any, *, where: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"{where}: expected finite value, got {value!r}")
    return float(value)


def _controller_terminal(run_dir: Path) -> tuple[dict[str, Any], Path]:
    paths = sorted(run_dir.glob("debug_job*/train_metrics.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"{run_dir}: expected one E89 training log, got {paths}")
    path = paths[0]
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if all(key in row for key in CONTROLLER_KEYS.values()):
                step = int(_finite(row.get("trainer/global_step"), where=str(path)))
                candidates.append((step, line_number, row))
    if not candidates:
        raise RuntimeError(f"{path}: no complete controller telemetry row")
    step, line_number, row = max(candidates, key=lambda item: (item[0], item[1]))
    if step < TARGET:
        raise RuntimeError(f"{path}: final controller step {step} is below {TARGET}")
    values = {
        output: _finite(row[source], where=f"{path}:{line_number}:{source}")
        for output, source in CONTROLLER_KEYS.items()
    }
    return {
        **values,
        "global_step": step,
        "source_line": line_number,
        # E89's registered gate is a factor-of-two band by pass 2 in at
        # least four of five domains.  It does not authorize dropping an
        # otherwise terminal seed because its final observation falls outside
        # a tighter post-hoc absolute tolerance.  Retain every terminal seed
        # and expose the final-band result for the paper audit instead.
        "final_within_factor_two": (
            TARGET_RATIO / 2
            <= values["final_realized_ratio"]
            <= TARGET_RATIO * 2
        ),
    }, path


def _fixed_endpoints(result: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    design = result.get("design", {})
    if (
        result.get("schema") != "maxent_factorial_05b_paper_results_v1"
        or design.get("domains") != list(DOMAIN_ORDER)
        or design.get("paired_seeds") != list(SEEDS)
        or design.get("evaluation_draws") != 4
        or design.get("target_steps") != TARGET
    ):
        raise RuntimeError("fixed Semantic-MaxEnt paper result drifted")
    output: dict[str, dict[int, dict[str, float]]] = {}
    for domain in DOMAIN_ORDER:
        source = result["domains"][domain]["arms"]["semantic"]["terminal_per_seed"]
        if set(source) != {str(seed) for seed in SEEDS}:
            raise RuntimeError(f"{domain}: incomplete fixed-semantic endpoint block")
        output[domain] = {
            seed: {
                "pass8": float(source[str(seed)]["pass8"]),
                "distinct8": float(source[str(seed)]["distinct8"]),
            }
            for seed in SEEDS
        }
    return output


def build() -> dict[str, Any]:
    ledger = _read(E89_LEDGER)
    failed_gate = _read(FAILED_GATE)
    if (
        failed_gate.get("schema") != "paper-adaptive-dose-gate-figure-v1"
        or failed_gate.get("status") != "closed registered mechanism-gate evidence"
        or failed_gate.get("target_ratio") != 0.05
        or failed_gate.get("maximum_bound_hit_fraction") != 0.20
        or len(failed_gate.get("records", [])) != 9
    ):
        raise RuntimeError("E88 failed-controller mechanism gate drifted")
    controller = ledger.get("controller", {})
    if (
        ledger.get("schema") != "e89_adaptive_semantic_maxent_reachable_05b_jobs_v1"
        or ledger.get("target_steps") != TARGET
        or ledger.get("domains") != list(DOMAIN_ORDER)
        or controller.get("kind") != "semantic_task_advantage_rms_ratio_v1"
        or controller.get("target_ratio") != TARGET_RATIO
        or controller.get("max_coefficient") != 0.4
    ):
        raise RuntimeError("E89 reachable-controller ledger drifted")
    fixed_result = _read(FIXED_RESULT)
    fixed = _fixed_endpoints(fixed_result)
    runs = _runs(ledger, arm="adaptive_semantic_reachable")
    source_paths: set[Path] = {
        E89_LEDGER, E89_PROTOCOL, FAILED_GATE, FIXED_RESULT,
    }
    cells: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            run = runs.get((domain, seed))
            if run is None:
                raise RuntimeError(f"E89 ledger lacks {domain}/seed {seed}")
            run_dir = Path(str(run["run_dir"]))
            marker = _completion_marker(run_dir, target=TARGET)
            if marker is None:
                continue
            adaptive, endpoint_paths = _terminal_metrics(run_dir, target=TARGET)
            mechanism, training_path = _controller_terminal(run_dir)
            source_paths.update(endpoint_paths | {marker, training_path})
            fixed_endpoint = fixed[domain][seed]
            delta_pass = adaptive["pass8"] - fixed_endpoint["pass8"]
            delta_distinct = adaptive["distinct8"] - fixed_endpoint["distinct8"]
            per_seed[str(seed)] = {
                "job_id": int(run["job_id"]),
                "mechanism": mechanism,
                "fixed_endpoint": fixed_endpoint,
                "adaptive_endpoint": adaptive,
                "effects": {
                    "delta_pass8": delta_pass,
                    "delta_distinct8": delta_distinct,
                    "delta_adjusted_breadth8": delta_distinct - delta_pass,
                },
            }
        seeds = sorted(int(seed) for seed in per_seed)
        if not seeds:
            cells.append(
                {
                    "domain": domain,
                    "n": 0,
                    "seeds": [],
                    "evidence": "no_terminal_adaptive_seed",
                    "per_seed": {},
                }
            )
            continue
        cell: dict[str, Any] = {
            "domain": domain,
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
            cell["summaries"] = {
                field: _interval(
                    [
                        (
                            per_seed[str(seed)]["mechanism"][field]
                            if field.startswith("final_")
                            else per_seed[str(seed)]["effects"][field]
                        )
                        for seed in seeds
                    ]
                )
                for field in FIELDS
            }
        cells.append(cell)
    bound_violations = [
        {
            "domain": cell["domain"],
            "seed": int(seed),
            "bound_hit_fraction": record["mechanism"]["bound_hit_fraction"],
        }
        for cell in cells
        for seed, record in cell["per_seed"].items()
        if record["mechanism"]["bound_hit_fraction"]
        > MAX_BOUND_HIT_FRACTION
    ]
    not_adapted = [
        {
            "domain": cell["domain"],
            "seed": int(seed),
            "frozen_fraction": record["mechanism"]["frozen_fraction"],
        }
        for cell in cells
        for seed, record in cell["per_seed"].items()
        if record["mechanism"]["frozen_fraction"]
        > NOT_ADAPTED_FROZEN_FRACTION
    ]
    final_band_misses = [
        {
            "domain": cell["domain"],
            "seed": int(seed),
            "final_realized_ratio": record["mechanism"][
                "final_realized_ratio"
            ],
        }
        for cell in cells
        for seed, record in cell["per_seed"].items()
        if not record["mechanism"]["final_within_factor_two"]
    ]
    return {
        "schema": "paper-adaptive-mechanism-outcomes-qwen05b-v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "all terminal E89 reachable-controller seeds; balanced summaries "
            "only for complete five-seed domains"
        ),
        "model": "Qwen2.5-0.5B-Instruct",
        "treatment": "Adaptive Semantic MaxEnt + Re:Dr, rho=.015",
        "outcome_baseline": "Fixed Semantic MaxEnt + Re:Dr, eta=.10",
        "domain_order": list(DOMAIN_ORDER),
        "registered_seeds": list(SEEDS),
        "target_ratio": TARGET_RATIO,
        "final_target_band": [TARGET_RATIO / 2, TARGET_RATIO * 2],
        "target_steps": TARGET,
        "fields": list(FIELDS),
        "e89_mechanism_gate": {
            "status": "failed",
            "decisive_registered_criterion": 2,
            "maximum_bound_hit_fraction": MAX_BOUND_HIT_FRACTION,
            "bound_hit_violations": bound_violations,
            "not_adapted_frozen_fraction": NOT_ADAPTED_FROZEN_FRACTION,
            "not_adapted_cells": not_adapted,
            "final_factor_two_band": [TARGET_RATIO / 2, TARGET_RATIO * 2],
            "final_band_misses": final_band_misses,
            "interpretation": (
                "criterion 2 fails because at least one terminal run spends "
                "more than 20% of applied controller updates at a bound; "
                "cells frozen above 80% are reported as not adapted. The "
                "final-ratio diagnostic does not replace the separately "
                "registered by-pass-2 criterion."
            ),
            "protocol": _relative(E89_PROTOCOL),
        },
        "failed_rho05_context": {
            "evidence": "mechanism_only; no outcome comparison",
            "status": failed_gate["status"],
            "outcome": failed_gate["outcome"],
            "target_ratio": failed_gate["target_ratio"],
            "maximum_bound_hit_fraction": failed_gate[
                "maximum_bound_hit_fraction"
            ],
            "records": failed_gate["records"],
            "source_json": _relative(FAILED_GATE),
        },
        "selection_rule": (
            "every registered E89 run with a step-3072 completion marker, a "
            "terminal controller-telemetry row, and four terminal sampled draws"
        ),
        "evidence_encoding": {
            "terminal_seed": "open marker",
            "balanced_five_seed_mean": "filled diamond in reachable-target panels",
            "balanced_five_seed_outcome_interval": (
                "vertical two-sided paired Student-t interval, df=4"
            ),
            "exact_terminal_prefix": "open markers only; no mean or interval",
            "failed_rho05_record": "mechanism-only point; no outcome attached",
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


def _value(seed_record: dict[str, Any], field: str) -> float:
    if field.startswith("final_"):
        return float(seed_record["mechanism"][field])
    return float(seed_record["effects"][field])


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        2, 3, figsize=(style.WIDTH, 5.05),
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.0]},
    )
    cells = {cell["domain"]: cell for cell in payload["cells"]}
    y_positions = {
        domain: float(len(DOMAIN_ORDER) - index - 1)
        for index, domain in enumerate(DOMAIN_ORDER)
    }
    current_panels = (
        (
            axes[0, 0], "A  Reachable-target final ratio",
            "final_realized_ratio", TARGET_RATIO,
            r"final realized RMS ratio ($\rho=.015$)", (0.0, 0.032),
        ),
        (
            axes[1, 0], "B  Reachable-target bound occupancy",
            "bound_hit_fraction", MAX_BOUND_HIT_FRACTION,
            "controller updates at a bound", (0.0, 0.31),
        ),
    )
    for axis, title, field, reference, xlabel, xlim in current_panels:
        style.style_axis(axis, grid="both", title=title)
        axis.axvline(
            reference, color=style.CONTROL, linewidth=0.9,
            linestyle=style.ARM_DASH[style.CONTROL], zorder=2,
        )
        if field == "final_realized_ratio":
            axis.axvspan(
                TARGET_RATIO / 2, TARGET_RATIO * 2,
                color=style.CONTROL, alpha=0.06, zorder=0,
            )
        for domain in DOMAIN_ORDER:
            records = [
                cells[domain]["per_seed"][str(seed)]
                for seed in cells[domain]["seeds"]
            ]
            offsets = [
                (index - (len(records) - 1) / 2) * 0.06
                for index in range(len(records))
            ]
            axis.scatter(
                [
                    float(
                        record["mechanism"][
                            (
                                "final_realized_ratio"
                                if field == "final_realized_ratio"
                                else "bound_hit_fraction"
                            )
                        ]
                    )
                    for record in records
                ],
                [y_positions[domain] + offset for offset in offsets],
                s=22,
                marker="D" if field == "bound_hit_fraction" else "o",
                facecolors=[
                    (
                        DOMAIN_COLOR[domain]
                        if record["mechanism"]["frozen_fraction"]
                        > NOT_ADAPTED_FROZEN_FRACTION
                        else style.WHITE
                    )
                    for record in records
                ],
                edgecolors=DOMAIN_COLOR[domain], linewidths=0.75, zorder=3,
            )
        axis.set_xlim(*xlim)
        axis.set_ylim(-0.55, len(DOMAIN_ORDER) - 0.45)
        axis.set_yticks(
            [y_positions[domain] for domain in DOMAIN_ORDER],
            [
                f"{DOMAIN_LABEL[domain]}  n={cells[domain]['n']}"
                for domain in DOMAIN_ORDER
            ],
        )
        axis.set_xlabel(xlabel, fontsize=style.LABEL_FONT)

    scatter_specs = (
        (axes[0, 1], "C  Realized ratio vs correctness", "final_realized_ratio", "delta_pass8"),
        (axes[0, 2], r"D  Final $\eta$ vs correctness", "final_coefficient", "delta_pass8"),
        (axes[1, 1], "E  Realized ratio vs adjusted breadth", "final_realized_ratio", "delta_adjusted_breadth8"),
        (axes[1, 2], r"F  Final $\eta$ vs adjusted breadth", "final_coefficient", "delta_adjusted_breadth8"),
    )
    for axis, title, x_field, y_field in scatter_specs:
        style.style_axis(axis, grid="both", title=title)
        axis.axhline(0.0, color=style.MUTED, linewidth=0.75, linestyle=(0, (2, 2)))
        if x_field == "final_realized_ratio":
            axis.axvline(
                TARGET_RATIO, color=style.CONTROL, linewidth=0.8,
                linestyle=style.ARM_DASH[style.CONTROL], zorder=1,
            )
        for domain in DOMAIN_ORDER:
            cell = cells[domain]
            seeds = cell["seeds"]
            xs = [_value(cell["per_seed"][str(seed)], x_field) for seed in seeds]
            ys = [_value(cell["per_seed"][str(seed)], y_field) for seed in seeds]
            color = DOMAIN_COLOR[domain]
            axis.scatter(
                xs, ys, s=18, marker="o", facecolors="none",
                edgecolors=color, linewidths=0.75, zorder=3,
            )
            if len(seeds) == 5:
                x_mean = cell["summaries"][x_field]["mean"]
                y_summary = cell["summaries"][y_field]
                axis.plot(
                    [x_mean, x_mean], y_summary["student_t_95"],
                    color=color, linewidth=1.0, zorder=4,
                )
                axis.scatter(
                    [x_mean], [y_summary["mean"]], s=30, marker="D",
                    facecolors=color, edgecolors=style.WHITE,
                    linewidths=0.45, zorder=5,
                )
        # These panels span about 0.007--0.016, where the default locator asks
        # for 0.0075/0.0100/0.0125/0.0150 and prints them as one unbroken run
        # of digits in a panel this narrow. Three ticks fit and stay readable.
        axis.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        axis.set_xlabel(FIELD_LABEL[x_field], fontsize=style.LABEL_FONT)
        axis.set_ylabel(FIELD_LABEL[y_field], fontsize=style.LABEL_FONT)

    handles = [
        *[
            Line2D(
                [0], [0], marker="o", linestyle="none", markersize=4,
                markerfacecolor="none", markeredgecolor=DOMAIN_COLOR[domain],
                label=f"{DOMAIN_LABEL[domain]}  n={cells[domain]['n']}",
            )
            for domain in DOMAIN_ORDER
        ],
        Line2D(
            [0, 1], [0, 0], color=style.MUTED, marker="D",
            linewidth=1.0, markersize=4, label="n=5 mean + outcome 95% interval",
        ),
        Line2D(
            [0, 1], [0, 0], color=style.CONTROL,
            linestyle=style.ARM_DASH[style.CONTROL], linewidth=1.0,
            label="registered target / gate threshold",
        ),
        Line2D(
            [0], [0], marker="D", linestyle="none", markersize=4,
            markerfacecolor=style.INK, markeredgecolor=style.INK,
            label="filled gate marker: frozen >80% (not adapted)",
        ),
    ]
    style.bottom_legend(
        figure,
        handles,
        [handle.get_label() for handle in handles],
        y=0.002,
        ncol=4,
    )
    figure.suptitle(
        "Adaptive semantic dose: completed outcomes, failed mechanism gate",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.945,
        (
            "A--B: terminal rho=.015 controller diagnostics · C--F: "
            "adaptive minus fixed treatment at the same seed"
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.16,
        right=0.995,
        top=0.88,
        bottom=0.20,
        hspace=0.42,
        wspace=0.34,
    )
    style.save(figure, output, png=True, dpi=260)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    payload = build()
    render(payload, output)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output.with_suffix('.pdf')}, {output.with_suffix('.png')}, "
        f"and {output.with_suffix('.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
