#!/usr/bin/env python3
"""Reflow frozen paper summaries into one invariant three-model domain grid.

Every static cross-model trajectory figure uses the same physical 3x5 layout.
Rows are always Qwen2.5-0.5B | Falcon3-1B | Qwen2.5-3B and columns are always
Graph | Countdown | Python | MathIR | Pantry. A cell without enough evidence
for the figure's stated gate is left blank; rows are never dropped or reordered.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))

import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402
from exp_scaling.build_paper_core_terminal_endpoints import approved_run_exclusion  # noqa: E402

DOMAIN_ORDER = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
DOMAIN_LABEL = {
    "graph_coloring": "Graph", "countdown": "Countdown",
    "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "Pantry",
}
SCALE_LABEL = {
    "qwen05b": "Qwen2.5-0.5B", "falcon1b": "Falcon3-1B",
    "qwen3b": "Qwen2.5-3B",
}
MODEL_ORDER = ("qwen05b", "falcon1b", "qwen3b")
COMPARISON_DIR = ROOT / "paper/figures/comparisons"
FIGURE_DIR = ROOT / "paper/figures"
INPUTS = {
    "fixed": (COMPARISON_DIR / "fixed_semantic_factorial_cross_scale.json",),
    "core_falcon": (
        COMPARISON_DIR / "core_retention_qwen05b_part1.json",
        COMPARISON_DIR / "core_retention_qwen05b_part2.json",
        COMPARISON_DIR / "core_retention_falcon1b_part1.json",
        COMPARISON_DIR / "core_retention_falcon1b_part2.json",
        COMPARISON_DIR / "core_retention_qwen3b_part1.json",
        COMPARISON_DIR / "core_retention_qwen3b_part2.json",
    ),
    "adaptive": (
        COMPARISON_DIR / "adaptive_semantic_replay_qwen05b_part1.json",
        COMPARISON_DIR / "adaptive_semantic_replay_qwen05b_part2.json",
        COMPARISON_DIR / "adaptive_semantic_replay_falcon1b_part1.json",
        COMPARISON_DIR / "adaptive_semantic_replay_falcon1b_part2.json",
        COMPARISON_DIR / "adaptive_semantic_replay_qwen3b_part1.json",
        COMPARISON_DIR / "adaptive_semantic_replay_qwen3b_part2.json",
    ),
    "replay_dose": (
        COMPARISON_DIR / "replay_dose_qwen05b_progress_part1.json",
        COMPARISON_DIR / "replay_dose_qwen05b_progress_part2.json",
    ),
}
CORE_LEDGER_BY_SCALE = {
    "qwen05b": ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
    "falcon1b": ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
    "qwen3b": ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
}
PLAIN_GRPO_LEDGER_BY_SCALE = {
    "qwen05b": (ROOT / "var/artifacts/e95_plain_grpo_Qwen25-05B_jobs.json",),
    "falcon1b": (ROOT / "var/artifacts/e95_plain_grpo_Falcon3-1B_jobs.json",),
    "qwen3b": (
        ROOT / "var/artifacts/e95_plain_grpo_Qwen25-3B_jobs.json",
        ROOT / "var/artifacts/e114_plain_grpo_qwen3b_extension_jobs.json",
    ),
}
CORE_METRICS = {
    "distinct8": ("distinct_correct_modes_at_k", "distinct@8"),
    "pass8": ("any_correct_at_k", "pass@8"),
    "mean8": ("mean_at_k", "mean@8"),
}
OUTPUTS = {
    "fixed": COMPARISON_DIR / "fixed_semantic_factorial_cross_scale_strip",
    "core_falcon": COMPARISON_DIR / "core_retention_falcon1b_static_strip",
    "core_pass8": COMPARISON_DIR / "core_retention_falcon1b_pass8_static_strip",
    "core_mean8": COMPARISON_DIR / "core_retention_falcon1b_mean8_static_strip",
    "adaptive": COMPARISON_DIR / "adaptive_semantic_replay_cross_scale_strip",
    "replay_dose": COMPARISON_DIR / "replay_dose_qwen05b_progress_static_strip",
    "ucpo": FIGURE_DIR / "direct_baseline_learning_curves_static_strip",
}
RLEP_LEDGER = ROOT / "var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json"
UCPO_LEDGER_BY_SCALE = {
    "qwen05b": (
        ROOT / "var/artifacts/e97_ucpo_05b_jobs.json",
        ROOT / "var/artifacts/e115_ucpo_qwen05b_domain_extension_jobs.json",
    ),
    "falcon1b": (ROOT / "var/artifacts/e99_ucpo_falcon1b_jobs.json",),
    "qwen3b": (ROOT / "var/artifacts/e115_ucpo_qwen3b_jobs.json",),
}
RLEP_LEDGER_BY_SCALE = {
    "qwen05b": (
        RLEP_LEDGER,
        ROOT / "var/artifacts/e116_sparse_rlep_qwen05b_domain_extension_jobs.json",
    ),
    "falcon1b": (ROOT / "var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json",),
    "qwen3b": (ROOT / "var/artifacts/e116_sparse_rlep_qwen3b_jobs.json",),
}
RLEP_EXPECTED_DRAWS = 4
RLEP_EVALUATION_KIND = "fixed_seed_sampled_k_neutral"
RLEP_METRICS = {
    "pass8": "any_correct_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}
_CORE_CURVE_CACHE: dict[
    tuple[Path, int, int], tuple[dict[str, dict[int, float]], list[dict[str, Any]]]
] = {}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def registered_paths(mapping: dict[str, tuple[Path, ...]]) -> tuple[Path, ...]:
    return tuple(path for paths in mapping.values() for path in paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for path in paths
    ]


def frozen_prefix(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read and identify exactly the append-only prefix used by this render."""

    byte_length = path.stat().st_size
    with path.open("rb") as handle:
        data = handle.read(byte_length)
    if len(data) != byte_length:
        raise RuntimeError(f"short read while freezing live prefix: {path}")
    return data, {
        "path": str(path.relative_to(ROOT)),
        "byte_length": byte_length,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def registered_evaluation_sources(
    run: dict[str, Any],
) -> tuple[list[Path], dict[str, Any]]:
    """Bind outcomes to the ledger's current job, preserving recovery provenance.

    Recovery ledgers retain the run directory while replacing a failed job.
    Its old debug files are historical evidence, not samples from the replacement.
    Undocumented job directories fail closed instead of being silently pooled.
    """

    run_dir = Path(run["run_dir"])
    job_id = int(run["job_id"])
    replaced = {int(value) for value in run.get("replaced_job_ids", [])}
    if run.get("replaces_job_id") is not None:
        replaced.add(int(run["replaces_job_id"]))
    if job_id in replaced:
        raise RuntimeError(f"current job is also marked superseded: {run_dir}")

    recovery_history = []
    for item in run.get("repair_history", []):
        protocol = Path(item["protocol"])
        if not protocol.is_absolute():
            protocol = ROOT / protocol
        if not protocol.is_file():
            raise FileNotFoundError(f"registered recovery amendment missing: {protocol}")
        recovery_history.append({
            "old_job_id": int(item["old_job_id"]),
            "new_job_id": int(item["new_job_id"]),
            "protocol": str(protocol.relative_to(ROOT)),
            "protocol_sha256": sha256(protocol),
        })

    selected = []
    excluded = []
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        source_id = path.parent.name.removeprefix("debug_job")
        if not source_id.isdigit():
            raise RuntimeError(f"unregistered evaluation source: {path}")
        source_job = int(source_id)
        if source_job == job_id:
            selected.append(path)
        elif source_job in replaced:
            # Hash the discarded source, but never inspect its outcomes to decide
            # whether it should replace or supplement the registered job.
            excluded.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "job_id": source_job,
                "status": "superseded_registered_job",
                "outcome_value_selected": False,
            })
        else:
            raise RuntimeError(f"unregistered evaluation source: {path}")
    return selected, {
        "policy": "current_registered_job_only; never pool superseded attempts",
        "registered_job_id": job_id,
        "superseded_job_ids": sorted(replaced),
        "recovery_history": recovery_history,
        "excluded_sources": excluded,
    }


def rlep_run_curve(run: dict[str, Any]) -> dict[str, Any] | None:
    """Extract complete registered four-draw checkpoints from one live run."""

    run_dir = Path(run["run_dir"])
    if approved_run_exclusion(run_dir) is not None:
        return None
    log_paths, source_binding = registered_evaluation_sources(run)
    if not log_paths:
        return None
    records: dict[tuple[int, int], dict[str, float]] = {}
    conflicting_keys: set[tuple[int, int]] = set()
    prefix_sources: list[dict[str, Any]] = []
    for log_path in log_paths:
        data, source = frozen_prefix(log_path)
        source.update({
            "domain": run["domain"], "seed": int(run["seed"]),
            "job_id": int(run["job_id"]),
            "source_binding": source_binding,
        })
        prefix_sources.append(source)
        for raw_line in data.splitlines():
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                # A live writer may leave the final prefix line incomplete.
                continue
            if row.get("evaluation_kind") != RLEP_EVALUATION_KIND:
                continue
            step, draw = row.get("step"), row.get("draw_index")
            metrics = row.get("metrics")
            if not isinstance(step, int) or not isinstance(draw, int):
                continue
            if draw not in range(RLEP_EXPECTED_DRAWS) or not isinstance(metrics, dict):
                continue
            values = {
                metric: float(metrics[source_name])
                for metric, source_name in RLEP_METRICS.items()
                if source_name in metrics
            }
            if len(values) != len(RLEP_METRICS) or not all(
                math.isfinite(value) for value in values.values()
            ):
                continue
            identity = (step, draw)
            if identity in records and records[identity] != values:
                conflicting_keys.add(identity)
            else:
                records[identity] = values

    # A nonterminal descriptive point can be omitted without selecting a retry.
    # A conflicted final endpoint remains a hard failure unless explicitly excluded.
    target = int(run.get("target_steps", 3072))
    if any(step == target for step, _draw in conflicting_keys):
        raise RuntimeError(f"conflicting terminal sampled evaluation: {run_dir}")
    excluded_steps = sorted({step for step, _draw in conflicting_keys})
    if excluded_steps:
        for source in prefix_sources:
            source["excluded_conflicting_checkpoints"] = excluded_steps
            source["outcome_value_selected_for_conflicts"] = False
    steps = sorted({step for step, _draw in records})
    complete_steps = [
        step for step in steps
        if step not in excluded_steps
        and all((step, draw) in records for draw in range(RLEP_EXPECTED_DRAWS))
    ]
    if not complete_steps:
        return None
    curve = {
        step: {
            metric: sum(records[(step, draw)][metric]
                        for draw in range(RLEP_EXPECTED_DRAWS))
            / RLEP_EXPECTED_DRAWS
            for metric in RLEP_METRICS
        }
        for step in complete_steps
    }
    return {"curve": curve, "prefix_sources": prefix_sources}


def rlep_progress_snapshot() -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]
]:
    """Freeze the available E98-R1 progress with constant n within each panel."""

    ledger = read(RLEP_LEDGER)
    if ledger.get("schema") != "e98r1_sparse_rlep_dr_05b_jobs_v1":
        raise RuntimeError("unexpected sparse RLEP-Dr ledger schema")
    train_rows = int(ledger["train_rows"])
    target_steps = int(ledger["target_steps"])
    by_domain: dict[str, list[tuple[int, dict[int, dict[str, float]]]]] = {}
    prefix_sources: list[dict[str, Any]] = []
    for run in ledger["runs"]:
        if run.get("domain") not in DOMAIN_ORDER:
            continue
        result = rlep_run_curve(run)
        if result is None:
            continue
        seed = int(run["seed"])
        by_domain.setdefault(run["domain"], []).append((seed, result["curve"]))
        prefix_sources.extend(result["prefix_sources"])

    summaries: dict[str, dict[str, Any]] = {}
    details: dict[str, dict[str, Any]] = {}
    for domain, seed_curves in by_domain.items():
        seed_curves.sort(key=lambda item: item[0])
        shared_steps = sorted(set.intersection(*(
            set(curve) for _seed, curve in seed_curves
        )))
        shared_steps = [step for step in shared_steps if 0 <= step <= target_steps]
        if not shared_steps:
            continue
        domain_summary: dict[str, Any] = {}
        for step in shared_steps:
            point: dict[str, Any] = {"training_pass": step / train_rows}
            for metric in RLEP_METRICS:
                per_seed = {
                    str(seed): float(curve[step][metric])
                    for seed, curve in seed_curves
                }
                values = list(per_seed.values())
                point[metric] = {
                    "mean": sum(values) / len(values),
                    "range": [min(values), max(values)],
                    "per_seed": per_seed,
                }
            domain_summary[str(step)] = point
        summaries[domain] = domain_summary
        details[domain] = {
            "seeds": [seed for seed, _curve in seed_curves],
            "available_seed_count": len(seed_curves),
            "complete_shared_checkpoint_count": len(shared_steps),
            "deepest_step": shared_steps[-1],
            "deepest_training_pass": shared_steps[-1] / train_rows,
        }
    return summaries, details, prefix_sources


def ordered(summary: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
    return sorted((float(step), value) for step, value in summary.items())


METHOD_SERIES = {
    "drgrpo": ("paired", "control"),
    "grpo": ("paired", "grpo"),
    "replay_grpo": ("paired", "replay"),
    "semantic_maxent": ("semantic", "semantic_only"),
    "replay_semantic_maxent": ("semantic", "semantic"),
    "adaptive_semantic_replay": ("semantic", "adaptive_semantic"),
    "adaptive_replay_grpo": ("semantic", "bank_normalized_replay"),
}


def series(record: dict[str, Any], method: str):
    kind, arm = METHOD_SERIES[method]
    if kind == "paired":
        points = ordered(record["paired_summary_by_pass"])
        prefix = arm
        points = [
            item for item in points if f"{prefix}_mean" in item[1]
        ]
        return (
            [step for step, _ in points],
            [float(value[f"{prefix}_mean"]) for _, value in points],
            [float(value[f"{prefix}_range"][0]) for _, value in points],
            [float(value[f"{prefix}_range"][1]) for _, value in points],
        )
    points = ordered(record["semantic_summary_by_arm"][arm])
    return (
        [step for step, _ in points],
        [float(value["semantic_mean"]) for _, value in points],
        [float(value["semantic_range"][0]) for _, value in points],
        [float(value["semantic_range"][1]) for _, value in points],
    )


def available(record: dict[str, Any], method: str) -> bool:
    kind, arm = METHOD_SERIES[method]
    if kind == "paired":
        points = record.get("paired_summary_by_pass", {})
        return any(f"{arm}_mean" in value for value in points.values())
    return bool(record.get("semantic_summary_by_arm", {}).get(arm))


def seed_count(record: dict[str, Any], method: str) -> int:
    kind, arm = METHOD_SERIES[method]
    if kind == "paired":
        points = ordered(record["paired_summary_by_pass"])
        points = [
            item for item in points if f"{arm}_mean" in item[1]
        ]
    else:
        points = ordered(record["semantic_summary_by_arm"][arm])
    return min(
        len(value.get(f"{arm}_per_seed", value["paired_seeds"]))
        for _, value in points
    )


def plot_record(axis, record, methods, *, marker=None, band_alpha=style.BAND_ALPHA):
    for method in methods:
        if not available(record, method):
            continue
        visual = method_visuals.method_style(method)
        x, mean, low, high = series(record, method)
        axis.fill_between(
            x, low, high, color=visual["color"], alpha=band_alpha,
            linewidth=0, zorder=1,
        )
        axis.plot(
            x, mean, color=visual["color"], linestyle=visual["linestyle"],
            linewidth=style.MEAN_LW, marker=marker,
            markevery=2 if marker else None,
            markersize=2.5 if marker else None,
            markeredgewidth=0.45 if marker else None, zorder=3,
        )


def y_max(panels, methods) -> float:
    maximum = 1.0
    for entries in panels.values():
        for _scale, record in entries:
            for method in methods:
                if available(record, method):
                    maximum = max(maximum, *series(record, method)[3])
    return max(2.0, math.ceil(maximum * 2) / 2 + 0.25)


def method_handles(methods):
    handles, labels = [], []
    for method in methods:
        visual = method_visuals.method_style(method)
        handles.append(Line2D(
            [0], [0], color=visual["color"],
            linestyle=visual["linestyle"], linewidth=style.MEAN_LW,
        ))
        labels.append(str(visual["label"]))
    return handles, labels


def model_row_axes(
    title: str,
    subtitle: str,
    *,
    ylabel: str = "mean distinct correct@8",
    height: float = 7.0,
):
    """Use the invariant three-model row order and fixed domain columns."""

    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(MODEL_ORDER), len(DOMAIN_ORDER), figsize=(style.WIDTH, height),
        sharex=True, sharey=True, squeeze=False,
    )
    figure.suptitle(title, fontsize=style.TITLE_FONT, color=style.INK, y=0.995)
    figure.text(
        0.5, 0.96, subtitle, ha="center", va="top",
        fontsize=style.SMALL_FONT, color=style.MUTED,
    )
    for row, scale in enumerate(MODEL_ORDER):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row][column]
            style.style_axis(
                axis, grid="both",
                title=DOMAIN_LABEL[domain] if row == 0 else None,
            )
            axis.set_xlim(0, 8)
            axis.set_xticks((0, 2, 4, 6, 8))
            if column == 0:
                axis.set_ylabel(
                    f"{SCALE_LABEL[scale]}\n{ylabel}",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)
            if row == len(MODEL_ORDER) - 1:
                axis.set_xlabel("training pass", fontsize=style.LABEL_FONT)
    return figure, axes


def save_model_rows(figure, output, handles, labels, *, ncol):
    style.bottom_legend(figure, handles, labels, y=0.003, ncol=ncol)
    figure.subplots_adjust(
        left=0.105, right=0.995, top=0.89, bottom=0.13,
        hspace=0.30, wspace=0.24,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def write_provenance(
    output, *, figure_key, comparison, evidence, methods, panels,
    source_paths, missing_reasons=None,
    metric="mean distinct correct@8 with paired-seed range",
    telemetry_sources=None,
):
    panel_payload = {}
    for domain in DOMAIN_ORDER:
        available_scales = {
            scale for scale, _record in panels.get(domain, [])
        }
        panel_payload[domain] = {
            "available": [
                {
                    "scale": scale,
                    "scale_label": SCALE_LABEL[scale],
                    "minimum_paired_seed_count_by_method": {
                        method: seed_count(record, method)
                        for method in methods if available(record, method)
                    },
                    "available_methods": [
                        method for method in methods if available(record, method)
                    ],
                }
                for scale, record in panels.get(domain, [])
            ],
            "cells": {
                scale: (
                    {"status": "available"}
                    if scale in available_scales
                    else {
                        "status": "blank",
                        "reason": (missing_reasons or {}).get(
                            (scale, domain), "insufficient evidence"
                        ),
                    }
                )
                for scale in MODEL_ORDER
            },
        }
    payload = {
        "schema": "paper-aligned-domain-strip-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figure_key": figure_key,
        "comparison": comparison,
        "layout": "three physical model rows by five static-domain columns",
        "environment_columns": list(DOMAIN_ORDER),
        "model_rows": list(MODEL_ORDER),
        "model_encoding": (
            "row 1 Qwen2.5-0.5B; row 2 Falcon3-1B; row 3 Qwen2.5-3B; "
            "insufficient cells are blank"
        ),
        "evidence": evidence,
        "metric": metric,
        "methods": list(methods),
        "sources": sources(source_paths),
        "panels": panel_payload,
    }
    if telemetry_sources is not None:
        payload["telemetry_sources"] = telemetry_sources
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def render_fixed() -> Path:
    source_paths = INPUTS["fixed"]
    payload = read(source_paths[0])
    rows = {row["scale"]: row["records"] for row in payload["rows"]}
    methods = (
        "drgrpo", "semantic_maxent", "replay_grpo", "replay_semantic_maxent",
    )
    panels = {
        domain: [
            (scale, rows[scale][domain])
            for scale in MODEL_ORDER
            if domain in rows.get(scale, {})
        ] for domain in DOMAIN_ORDER
    }
    figure, axes = model_row_axes(
        "Fixed semantic-MaxEnt factorial across model scales",
        "Five-seed factorial cells; Qwen2.5-3B shows matched seed-70 parent/treatment curves (n=1).",
    )
    high, notes = y_max(panels, methods), {}
    by_cell = {
        (scale, domain): record
        for domain, entries in panels.items() for scale, record in entries
    }
    for row, scale in enumerate(MODEL_ORDER):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row][column]
            axis.set_ylim(0, high)
            record = by_cell.get((scale, domain))
            if record is None:
                notes[(scale, domain)] = (
                    "no admitted terminal fixed-semantic comparison"
                )
            else:
                plot_record(axis, record, methods)
    handles, labels = method_handles(methods)
    output = OUTPUTS["fixed"]
    save_model_rows(figure, output, handles, labels, ncol=4)
    write_provenance(
        output, figure_key=output.name, comparison="fixed_semantic_factorial",
        evidence="terminal five-seed factorials plus Qwen2.5-3B n=1",
        methods=methods, panels=panels,
        source_paths=source_paths, missing_reasons=notes,
    )
    return output


def merge_family_sources(paths):
    scale, records = "", {}
    for path in paths:
        payload = read(path)
        current = str(payload["scale"])
        if scale and current != scale:
            raise ValueError(f"mixed scales: {scale} and {current}")
        scale = current
        overlap = records.keys() & payload["records"].keys()
        if overlap:
            raise ValueError(f"duplicate domains: {sorted(overlap)}")
        records.update(payload["records"])
    return scale, records


def merge_family_sources_by_scale(paths):
    """Merge chunked terminal summaries without mixing model identities."""

    by_scale: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = read(path)
        scale = str(payload["scale"])
        records = by_scale.setdefault(scale, {})
        overlap = records.keys() & payload["records"].keys()
        if overlap:
            raise ValueError(f"duplicate {scale} domains: {sorted(overlap)}")
        records.update(payload["records"])
    return by_scale


def attach_plain_grpo(panels, metric_key: str) -> list[dict[str, Any]]:
    """Compatibility wrapper for the all-available GRPO reader."""

    return attach_available_plain_grpo(panels, metric_key)


def render_core_falcon() -> Path:
    source_paths = (
        *INPUTS["core_falcon"],
        *registered_paths(PLAIN_GRPO_LEDGER_BY_SCALE),
    )
    rows = merge_family_sources_by_scale(INPUTS["core_falcon"])
    panels = {
        domain: [
            (scale, rows[scale][domain])
            for scale in MODEL_ORDER if domain in rows.get(scale, {})
        ]
        for domain in DOMAIN_ORDER
    }
    telemetry_sources = attach_available_plain_grpo(panels, "distinct8")
    methods = ("grpo", "drgrpo", "replay_grpo")
    figure, axes = model_row_axes(
        "GRPO controls and canonical verified replay — distinct@8",
        "Every available seed and checkpoint is shown; exact n is retained per method.",
    )
    high, missing_reasons = y_max(panels, methods), {}
    by_cell = {
        (scale, domain): record
        for domain, entries in panels.items() for scale, record in entries
    }
    for row, scale in enumerate(MODEL_ORDER):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row][column]
            axis.set_ylim(0, high)
            record = by_cell.get((scale, domain))
            if record is None:
                missing_reasons[(scale, domain)] = (
                    "no sampled checkpoint available"
                )
            else:
                plot_record(axis, record, methods)
    handles, labels = method_handles(methods)
    output = OUTPUTS["core_falcon"]
    save_model_rows(figure, output, handles, labels, ncol=3)
    write_provenance(
        output, figure_key=output.name, comparison="core_retention",
        evidence="all available trajectories; no minimum seed count", methods=methods,
        panels=panels, source_paths=source_paths,
        missing_reasons=missing_reasons,
        telemetry_sources=telemetry_sources,
    )
    return output


def core_metric_run_curve(
    run_dir: Path,
    *,
    interval: int,
    target: int,
    source_field: str,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    """Read every complete registered four-draw checkpoint for one metric."""

    exclusion = approved_run_exclusion(run_dir)
    if exclusion is not None:
        return {}, [exclusion]
    cache_key = (run_dir.resolve(), interval, target)
    cached = _CORE_CURVE_CACHE.get(cache_key)
    if cached is not None:
        curves, prefix_sources = cached
        return curves.get(source_field, {}), prefix_sources

    fields = {field for field, _label in CORE_METRICS.values()}
    records: dict[str, dict[tuple[int, int], float]] = {field: {} for field in fields}
    prefix_sources: list[dict[str, Any]] = []
    paths = sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl"))
    if not paths:
        return {}, prefix_sources
    for path in paths:
        data, source = frozen_prefix(path)
        prefix_sources.append(source)
        for raw_line in data.splitlines():
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if row.get("evaluation_kind") != RLEP_EVALUATION_KIND:
                continue
            step, draw = row.get("step"), row.get("draw_index")
            metrics = row.get("metrics")
            if (
                not isinstance(step, int)
                or draw not in range(RLEP_EXPECTED_DRAWS)
                or step < 0
                or step > target
                or step % interval
                or not isinstance(metrics, dict)
            ):
                continue
            for field in fields:
                value = metrics.get(field)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    identity = (step, int(draw))
                    if identity in records[field] and records[field][identity] != float(value):
                        raise RuntimeError(f"conflicting sampled evaluation: {path}:{identity}:{field}")
                    records[field][identity] = float(value)
    curves: dict[str, dict[int, float]] = {}
    for field, field_records in records.items():
        curve: dict[int, float] = {}
        for step in range(0, target + 1, interval):
            observed = [
                field_records[(step, draw)]
                for draw in range(RLEP_EXPECTED_DRAWS)
                if (step, draw) in field_records
            ]
            if len(observed) == RLEP_EXPECTED_DRAWS:
                curve[step] = sum(observed) / len(observed)
        curves[field] = curve
    _CORE_CURVE_CACHE[cache_key] = (curves, prefix_sources)
    return curves.get(source_field, {}), prefix_sources


def core_metric_panels(metric_key: str):
    """Build trajectory records from every available paired seed/checkpoint."""

    source_field, _label = CORE_METRICS[metric_key]
    records_by_scale: dict[str, dict[str, Any]] = {}
    telemetry_sources: list[dict[str, Any]] = []
    for scale, ledger_path in CORE_LEDGER_BY_SCALE.items():
        if not ledger_path.is_file():
            continue
        ledger = read(ledger_path)
        interval = int(ledger["checkpoint_interval_steps"])
        target = int(ledger["target_steps"])
        steps_per_pass = int(ledger["train_rows"])
        curves: dict[str, dict[str, dict[int, dict[int, float]]]] = {}
        for run in ledger["runs"]:
            domain = str(run["domain"])
            if domain not in DOMAIN_ORDER:
                continue
            arm = str(run["arm"])
            seed = int(run["seed"])
            curve, run_sources = core_metric_run_curve(
                Path(run["run_dir"]), interval=interval, target=target,
                source_field=source_field,
            )
            if curve:
                curves.setdefault(domain, {}).setdefault(arm, {})[seed] = curve
            telemetry_sources.extend({
                **source, "scale": scale, "domain": domain,
                "arm": arm, "seed": seed,
            } for source in run_sources)

        scale_records: dict[str, Any] = {}
        for domain in DOMAIN_ORDER:
            domain_curves = curves.get(domain, {})
            control = domain_curves.get("control", {})
            replay = domain_curves.get("replay", {})
            paired_seeds = sorted(set(control) & set(replay))
            summary: dict[str, Any] = {}
            for step in range(0, target + 1, interval):
                step_seeds = [
                    seed for seed in paired_seeds
                    if step in control[seed] and step in replay[seed]
                ]
                if not step_seeds:
                    continue
                control_values = [control[seed][step] for seed in step_seeds]
                replay_values = [replay[seed][step] for seed in step_seeds]
                summary[str(step / steps_per_pass)] = {
                    "paired_seeds": step_seeds,
                    "control_mean": sum(control_values) / len(control_values),
                    "replay_mean": sum(replay_values) / len(replay_values),
                    "replay_minus_control": (
                        sum(replay_values) / len(replay_values)
                        - sum(control_values) / len(control_values)
                    ),
                    "control_range": [min(control_values), max(control_values)],
                    "replay_range": [min(replay_values), max(replay_values)],
                }
            if summary:
                scale_records[domain] = {
                    "paired_summary_by_pass": summary,
                    "semantic_summary_by_arm": {},
                }
        records_by_scale[scale] = scale_records
    panels = {
        domain: [
            (scale, records_by_scale[scale][domain])
            for scale in MODEL_ORDER
            if domain in records_by_scale.get(scale, {})
        ]
        for domain in DOMAIN_ORDER
    }
    return panels, telemetry_sources


def render_core_metric(metric_key: str) -> Path:
    panels, telemetry_sources = core_metric_panels(metric_key)
    _source_field, label = CORE_METRICS[metric_key]
    telemetry_sources.extend(attach_available_plain_grpo(panels, metric_key))
    methods = ("grpo", "drgrpo", "replay_grpo")
    figure, axes = model_row_axes(
        f"GRPO controls and canonical verified replay — {label} over training",
        "Every available seed and checkpoint is shown; exact n is retained per method.",
        ylabel=label,
    )
    by_cell = {
        (scale, domain): record
        for domain, entries in panels.items() for scale, record in entries
    }
    missing_reasons = {}
    for row, scale in enumerate(MODEL_ORDER):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row][column]
            axis.set_ylim(-0.02, 1.02)
            record = by_cell.get((scale, domain))
            if record is None:
                missing_reasons[(scale, domain)] = (
                    "no sampled checkpoint available"
                )
            else:
                plot_record(axis, record, methods)
    handles, labels = method_handles(methods)
    output = OUTPUTS[f"core_{metric_key}"]
    save_model_rows(figure, output, handles, labels, ncol=3)
    write_provenance(
        output,
        figure_key=output.name,
        comparison="core_retention",
        evidence="all available trajectories; no minimum seed count",
        methods=methods,
        panels=panels,
        source_paths=(
            *CORE_LEDGER_BY_SCALE.values(),
            *registered_paths(PLAIN_GRPO_LEDGER_BY_SCALE),
        ),
        missing_reasons=missing_reasons,
        metric=f"{label} with paired-seed range",
        telemetry_sources=telemetry_sources,
    )
    return output


def render_core_pass8() -> Path:
    return render_core_metric("pass8")


def render_core_mean8() -> Path:
    return render_core_metric("mean8")


def render_adaptive() -> Path:
    source_paths = INPUTS["adaptive"]
    panels = {domain: [] for domain in DOMAIN_ORDER}
    for path in source_paths:
        payload = read(path)
        for domain, record in payload["records"].items():
            panels[domain].append((str(payload["scale"]), record))
    methods = ("drgrpo", "replay_grpo", "adaptive_semantic_replay")
    figure, axes = model_row_axes(
        "Adaptive semantic MaxEnt with replay across model scales",
        "Rows: Qwen2.5-0.5B · Falcon3-1B · Qwen2.5-3B; insufficient cells are blank.",
    )
    high, notes = y_max(panels, methods), {}
    by_cell = {
        (scale, domain): record
        for domain, entries in panels.items() for scale, record in entries
    }
    for row, scale in enumerate(MODEL_ORDER):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes[row][column]
            axis.set_ylim(0, high)
            record = by_cell.get((scale, domain))
            if record is None:
                notes[(scale, domain)] = (
                    "no balanced five-seed terminal adaptive comparison"
                )
            else:
                plot_record(axis, record, methods)
    handles, labels = method_handles(methods)
    output = OUTPUTS["adaptive"]
    save_model_rows(figure, output, handles, labels, ncol=3)
    write_provenance(
        output, figure_key=output.name, comparison="adaptive_semantic_replay",
        evidence="terminal five-seed where shown", methods=methods,
        panels=panels, source_paths=source_paths, missing_reasons=notes,
    )
    return output


def render_replay_dose() -> Path:
    source_paths = INPUTS["replay_dose"]
    scale, records = merge_family_sources(source_paths)
    methods = ("drgrpo", "replay_grpo", "adaptive_replay_grpo")
    panels = {domain: [(scale, records[domain])] for domain in DOMAIN_ORDER}
    # The seed count is the same in all five panels, and the badge sat at the
    # bottom right where the Dr.GRPO trace runs --- five copies of a constant,
    # each printed over the data. Hoist the common value into the subtitle and
    # badge only a panel that departs from it.
    panel_counts = {
        domain: seed_count(panels[domain][0][1], "adaptive_replay_grpo")
        for domain in DOMAIN_ORDER
    }
    shared_count = style.dominant_note(
        str(count) for count in panel_counts.values()
    )
    subtitle = (
        "Rows: Qwen2.5-0.5B · Falcon3-1B · Qwen2.5-3B; unregistered rows are blank."
    )
    if shared_count is not None:
        subtitle += f"  Paired n={shared_count} unless marked."
    figure, axes_grid = model_row_axes(
        "Replay-dose ablation — Qwen2.5-0.5B progress snapshot",
        subtitle,
    )
    axes = axes_grid[0]
    high = y_max(panels, methods)
    for axis, domain in zip(axes, DOMAIN_ORDER):
        axis.set_ylim(0, high)
        record = panels[domain][0][1]
        plot_record(axis, record, methods)
        if str(panel_counts[domain]) != shared_count:
            axis.text(
                0.97, 0.05,
                f"paired n={panel_counts[domain]}",
                transform=axis.transAxes, ha="right", va="bottom",
                fontsize=style.SMALL_FONT, color=style.MUTED,
            )
    handles, labels = method_handles(methods)
    output = OUTPUTS["replay_dose"]
    save_model_rows(figure, output, handles, labels, ncol=3)
    write_provenance(
        output, figure_key=output.name, comparison="replay_dose",
        evidence="frozen progress snapshot; not a five-seed estimand",
        methods=methods, panels=panels, source_paths=source_paths,
        missing_reasons={
            (scale_key, domain): "replay-dose comparison not registered at this scale"
            for scale_key in MODEL_ORDER if scale_key != "qwen05b"
            for domain in DOMAIN_ORDER
        },
    )
    return output


def _terminal_marker(run: dict[str, Any], *, target: int) -> Path | None:
    marker = Path(run["run_dir"]) / "TRAINING_COMPLETE.json"
    if not marker.is_file():
        return None
    payload = read(marker)
    step = payload.get("terminal_step")
    if not isinstance(step, int) or step < target:
        raise RuntimeError(f"{marker}: terminal step {step!r} is below {target}")
    return marker


def _run_map(
    ledger: dict[str, Any], *, arm: str | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for run in ledger["runs"]:
        if arm is not None and str(run.get("arm")) != arm:
            continue
        key = (str(run["domain"]), int(run["seed"]))
        if key in result:
            raise RuntimeError(f"duplicate direct-baseline run {key}")
        result[key] = run
    return result


def _merged_run_map(
    ledger_paths: Iterable[Path], *, arm: str | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    ledgers: list[dict[str, Any]] = []
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for path in ledger_paths:
        if not path.is_file():
            raise FileNotFoundError(f"registered ledger missing: {path}")
        ledger = read(path)
        current = _run_map(ledger, arm=arm)
        overlap = merged.keys() & current.keys()
        if overlap:
            raise RuntimeError(f"duplicate registered cells: {sorted(overlap)}")
        merged.update(current)
        ledgers.append(ledger)
    return ledgers, merged


def _terminal_method_summary(
    run_map: dict[tuple[str, int], dict[str, Any]],
    *,
    domain: str,
    seeds: Iterable[int],
    target: int,
    train_rows: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[Path]]:
    curves: dict[int, dict[int, dict[str, float]]] = {}
    prefix_sources: list[dict[str, Any]] = []
    markers: list[Path] = []
    for seed in seeds:
        run = run_map.get((domain, seed))
        if run is None:
            continue
        exclusion = approved_run_exclusion(Path(run["run_dir"]))
        if exclusion is not None:
            prefix_sources.append(exclusion)
            continue
        marker = _terminal_marker(run, target=target)
        if marker is None:
            continue
        result = rlep_run_curve(run)
        if result is None or target not in result["curve"]:
            raise RuntimeError(
                f"{domain}/seed {seed}: terminal run lacks terminal evaluation"
            )
        curves[seed] = {
            step: values
            for step, values in result["curve"].items()
            if 0 <= step <= target
        }
        prefix_sources.extend(result["prefix_sources"])
        markers.append(marker)
    if not curves:
        return None, prefix_sources, markers
    shared_steps = sorted(set.intersection(*(set(curve) for curve in curves.values())))
    if not shared_steps or shared_steps[-1] != target:
        raise RuntimeError(f"{domain}: terminal direct-baseline curves do not share pass 8")
    summaries: dict[str, Any] = {}
    for step in shared_steps:
        point: dict[str, Any] = {"training_pass": step / train_rows}
        for metric in RLEP_METRICS:
            per_seed = {
                str(seed): float(curves[seed][step][metric])
                for seed in sorted(curves)
            }
            values = list(per_seed.values())
            point[metric] = {
                "mean": sum(values) / len(values),
                "range": [min(values), max(values)],
                "per_seed": per_seed,
            }
        summaries[str(step)] = point
    return {
        "n": len(curves),
        "seeds": sorted(curves),
        "deepest_step": target,
        "deepest_training_pass": target / train_rows,
        "summaries": summaries,
    }, prefix_sources, markers


def _direct_terminal_snapshot() -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
    set[Path],
]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    trajectory_sources: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    for scale in MODEL_ORDER:
        core_path = CORE_LEDGER_BY_SCALE[scale]
        ucpo_paths = UCPO_LEDGER_BY_SCALE[scale]
        rlep_paths = RLEP_LEDGER_BY_SCALE[scale]
        for path in (core_path, *ucpo_paths, *rlep_paths):
            if not path.is_file():
                raise FileNotFoundError(
                    f"registered direct-baseline ledger missing: {path}"
                )
            source_paths.add(path)
        core = read(core_path)
        ucpo_ledgers, ucpo_runs = _merged_run_map(ucpo_paths)
        rlep_ledgers, rlep_runs = _merged_run_map(rlep_paths)
        target = int(core["target_steps"])
        train_rows = int(core["train_rows"])
        if any(
            int(ledger["target_steps"]) != target
            for ledger in (*ucpo_ledgers, *rlep_ledgers)
        ):
            raise RuntimeError(f"{scale}: direct-baseline target-step drift")
        seeds = tuple(int(seed) for seed in core["seeds"])
        maps = {
            "drgrpo": _run_map(core, arm="control"),
            "replay_grpo": _run_map(core, arm="replay"),
            "ucpo": ucpo_runs,
            "rlep_dr": rlep_runs,
        }
        registered_domains = {
            domain for domain, _seed in set(maps["ucpo"]) | set(maps["rlep_dr"])
        }
        for domain in DOMAIN_ORDER:
            if domain not in registered_domains:
                continue
            method_records: dict[str, Any] = {}
            for method, run_map in maps.items():
                summary, prefixes, markers = _terminal_method_summary(
                    run_map,
                    domain=domain,
                    seeds=seeds,
                    target=target,
                    train_rows=train_rows,
                )
                trajectory_sources.extend(
                    {**item, "scale": scale, "method": method}
                    for item in prefixes
                )
                source_paths.update(markers)
                if summary is not None:
                    method_records[method] = summary
            if not {"drgrpo", "replay_grpo"}.issubset(method_records):
                raise RuntimeError(
                    f"{scale}/{domain}: terminal matched baselines incomplete"
                )
            if not ({"ucpo", "rlep_dr"} & method_records.keys()):
                continue
            cells[(scale, domain)] = {
                "scale": scale,
                "scale_label": SCALE_LABEL[scale],
                "domain": domain,
                "methods": method_records,
            }
    return cells, trajectory_sources, source_paths


def render_ucpo(snapshot: dict[str, Any] | None = None) -> Path:
    if snapshot is None:
        cells, trajectory_sources, source_paths = _direct_terminal_snapshot()
    else:
        if (snapshot.get("schema") != "paper-aligned-domain-strip-v2"
                or snapshot.get("comparison") != "direct_baselines"):
            raise ValueError("expected a retained direct-baseline trajectory snapshot")
        cells = {tuple(key.split("/", 1)): cell for key, cell in snapshot["cells"].items()}
        trajectory_sources, source_paths = [], []
    methods = ("drgrpo", "replay_grpo", "ucpo", "rlep_dr")
    figure, axes_grid = model_row_axes(
        "Direct baselines across model scales — terminal evidence",
        "Exact terminal prefixes are shown; cells without a terminal comparator are blank.",
    )
    maximum = max(
        point["distinct8"]["range"][1]
        for cell in cells.values()
        for method in cell["methods"].values()
        for point in method["summaries"].values()
    )
    high = max(2.0, math.ceil(maximum * 2) / 2 + 0.25)
    short = {"drgrpo": "D", "replay_grpo": "V", "ucpo": "U", "rlep_dr": "R"}
    for row, scale in enumerate(MODEL_ORDER):
        for column, domain in enumerate(DOMAIN_ORDER):
            axis = axes_grid[row][column]
            axis.set_ylim(0, high)
            cell = cells.get((scale, domain))
            if cell is None:
                continue
            for method in methods:
                record = cell["methods"].get(method)
                if record is None:
                    continue
                visual = method_visuals.method_style(method)
                points = sorted(
                    record["summaries"].values(),
                    key=lambda point: float(point["training_pass"]),
                )
                x = [float(point["training_pass"]) for point in points]
                mean = [float(point["distinct8"]["mean"]) for point in points]
                low = [float(point["distinct8"]["range"][0]) for point in points]
                upper = [float(point["distinct8"]["range"][1]) for point in points]
                axis.fill_between(
                    x, low, upper, color=visual["color"],
                    alpha=style.BAND_ALPHA, linewidth=0, zorder=1,
                )
                axis.plot(
                    x, mean, color=visual["color"],
                    linestyle=visual["linestyle"], linewidth=style.MEAN_LW,
                    marker=visual["marker"], markevery=[len(x) - 1],
                    markersize=2.8, markeredgewidth=0.4, zorder=3,
                )
            counts = " · ".join(
                f"{short[method]} {cell['methods'][method]['n']}"
                for method in methods if method in cell["methods"]
            )
            axis.text(
                0.03, 0.96, f"n: {counts}", transform=axis.transAxes,
                ha="left", va="top", fontsize=5.1, color=style.MUTED,
            )
    handles, labels = method_handles(methods)
    output = OUTPUTS["ucpo"]
    save_model_rows(figure, output, handles, labels, ncol=4)
    if snapshot is not None:
        return output
    panel_payload = {}
    for domain in DOMAIN_ORDER:
        available = []
        for scale in MODEL_ORDER:
            cell = cells.get((scale, domain))
            if cell is None:
                continue
            available.append({
                "scale": scale,
                "scale_label": SCALE_LABEL[scale],
                "minimum_paired_seed_count_by_method": {
                    method: record["n"]
                    for method, record in cell["methods"].items()
                },
                "terminal_seeds_by_method": {
                    method: record["seeds"]
                    for method, record in cell["methods"].items()
                },
            })
        panel_payload[domain] = {
            "available": available,
            "cells": {
                scale: (
                    {"status": "available"}
                    if (scale, domain) in cells
                    else {"status": "blank", "reason": "no terminal registered direct comparator"}
                )
                for scale in MODEL_ORDER
            },
        }
    provenance = {
        "schema": "paper-aligned-domain-strip-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figure_key": output.name,
        "comparison": "direct_baselines",
        "layout": "three physical model rows by five static-domain columns",
        "environment_columns": list(DOMAIN_ORDER),
        "model_rows": list(MODEL_ORDER),
        "model_encoding": (
            "row 1 Qwen2.5-0.5B; row 2 Falcon3-1B; row 3 Qwen2.5-3B; "
            "cells without a terminal direct comparator are blank"
        ),
        "evidence": (
            "all terminal registered UCPO cells and every terminal sparse "
            "RLEP-Dr seed; exact n is recorded per method and domain"
        ),
        "metric": "mean distinct correct@8 with seed range",
        "methods": list(methods),
        "sources": sources(sorted(source_paths)),
        "trajectory_sources": trajectory_sources,
        "panels": panel_payload,
        "cells": {
            f"{scale}/{domain}": cell
            for (scale, domain), cell in cells.items()
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return output


def attach_available_plain_grpo(
    panels: dict[str, list[tuple[str, dict[str, Any]]]],
    metric_key: str,
) -> list[dict[str, Any]]:
    """Attach every sampled GRPO checkpoint, with no seed-count gate."""

    source_field, _label = CORE_METRICS[metric_key]
    records = {
        (scale, domain): record
        for domain, entries in panels.items()
        for scale, record in entries
    }
    telemetry_sources: list[dict[str, Any]] = []
    for scale, ledger_paths in PLAIN_GRPO_LEDGER_BY_SCALE.items():
        if not all(path.is_file() for path in ledger_paths):
            continue
        ledgers, run_map = _merged_run_map(ledger_paths)
        targets = {int(ledger["target_steps"]) for ledger in ledgers}
        intervals = {int(ledger["checkpoint_interval_steps"]) for ledger in ledgers}
        train_rows = {int(ledger["train_rows"]) for ledger in ledgers}
        if len(targets) != 1 or len(intervals) != 1 or len(train_rows) != 1:
            raise RuntimeError(f"{scale}: merged GRPO ledger contract drift")
        target = targets.pop()
        interval = intervals.pop()
        steps_per_pass = train_rows.pop()
        by_domain: dict[str, dict[int, dict[int, float]]] = {}
        for run in run_map.values():
            domain = str(run["domain"])
            if domain not in DOMAIN_ORDER:
                continue
            seed = int(run["seed"])
            curve, run_sources = core_metric_run_curve(
                Path(run["run_dir"]), interval=interval, target=target,
                source_field=source_field,
            )
            if curve:
                by_domain.setdefault(domain, {})[seed] = curve
            telemetry_sources.extend({
                **source, "scale": scale, "domain": domain,
                "arm": "grpo", "seed": seed,
            } for source in run_sources)
        for domain, seed_curves in by_domain.items():
            record = records.get((scale, domain))
            if record is None:
                continue
            for pass_key, point in record["paired_summary_by_pass"].items():
                step = round(float(pass_key) * steps_per_pass)
                per_seed = {
                    str(seed): float(curve[step])
                    for seed, curve in seed_curves.items() if step in curve
                }
                if not per_seed:
                    continue
                values = list(per_seed.values())
                point["grpo_mean"] = sum(values) / len(values)
                point["grpo_range"] = [min(values), max(values)]
                point["grpo_per_seed"] = per_seed
    return telemetry_sources


RENDERERS = {
    "fixed": render_fixed, "core_falcon": render_core_falcon,
    "core_pass8": render_core_pass8, "core_mean8": render_core_mean8,
    "adaptive": render_adaptive, "replay_dose": render_replay_dose,
    "ucpo": render_ucpo,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=(*RENDERERS, "all"), default="all")
    args = parser.parse_args()
    keys = RENDERERS if args.figure == "all" else (args.figure,)
    for key in keys:
        output = RENDERERS[key]()
        for suffix in (".pdf", ".png", ".json"):
            print(f"wrote {output.with_suffix(suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
