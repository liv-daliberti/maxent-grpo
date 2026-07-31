#!/usr/bin/env python3
"""Render the clean eight-environment 0.5B cohort surface.

The default rendering is intentionally empty. It never imports historical
results. A later audit-bound results payload can populate the same surface,
but the renderer refuses a terminal label unless all 80 training jobs and all
eight environment summaries are terminal and integrity-clean.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "paper/figures/e68_e58_vs_grpo_05b_12ep_terminal_provenance.png"
)
DEFAULT_SIDECAR = (
    ROOT / "var/artifacts/clean_05b_eight_environment_figure_provenance.json"
)
PROTOCOL = ROOT / "docs/clean_05b_maxent_vs_drgrpo_eight_environment_plan.md"
SCHEMA = "clean-05b-eight-environment-figure-v1"
EXPECTED_JOB_COUNT = 80
EXPECTED_SEEDS = (43, 44, 45, 46, 47)
ARMS = ("grpo_compute_matched", "verified_first_global_replay_canonical")


@dataclass(frozen=True)
class EnvironmentRow:
    key: str
    label: str
    model: str
    interface: str


ROWS = (
    EnvironmentRow("graph_coloring", "Graph coloring", "Qwen2.5 0.5B", "assignment"),
    EnvironmentRow("countdown", "Countdown", "Qwen2.5 0.5B", "expression"),
    EnvironmentRow("python_factor", "Python factors", "Qwen2.5 0.5B", "executed function"),
    EnvironmentRow("mathir", "MathIR", "Qwen2.5 0.5B", "executed action menu"),
    EnvironmentRow("constructive_code", "ConstructiveCode", "Coder 0.5B", "checker witness"),
    EnvironmentRow("pantry_plan", "PantryPlan", "Qwen2.5 0.5B", "ingredient support"),
    EnvironmentRow("point_maze", "PointMaze", "Qwen2.5 0.5B", "quantized force program"),
    EnvironmentRow("ant_maze", "AntMaze", "Qwen2.5 0.5B", "heading program + frozen controller"),
)

METRICS = (
    ("greedy", "Greedy success"),
    ("mean_at_8", "Mean@8"),
    ("pass_at_8", "Pass@8"),
    ("distinct_at_8", "Distinct@8"),
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("results payload must be an object")
    if payload.get("schema_version") != SCHEMA:
        raise ValueError("results payload schema does not match renderer")
    return payload


def _terminal_contract(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    violations: list[str] = []
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_JOB_COUNT:
        violations.append("expected exactly 80 jobs")
        jobs = []
    identities: set[tuple[str, str, int]] = set()
    for job in jobs:
        if not isinstance(job, Mapping):
            violations.append("malformed job record")
            continue
        try:
            identity = (
                str(job["environment"]),
                str(job["arm"]),
                int(job["seed"]),
            )
        except (KeyError, TypeError, ValueError):
            violations.append("job identity is incomplete")
            continue
        if identity in identities:
            violations.append(f"duplicate job identity: {identity!r}")
        identities.add(identity)
        if job.get("status") != "terminal":
            violations.append(f"nonterminal job: {identity!r}")
        if job.get("integrity_status") != "pass":
            violations.append(f"integrity failure: {identity!r}")
    expected = {
        (row.key, arm, seed)
        for row in ROWS
        for arm in ARMS
        for seed in EXPECTED_SEEDS
    }
    if identities != expected:
        violations.append("job identity set differs from frozen 8x2x5 design")

    summaries = payload.get("environments")
    if not isinstance(summaries, Mapping):
        violations.append("environment summaries missing")
        summaries = {}
    if set(summaries) != {row.key for row in ROWS}:
        violations.append("environment summary set differs from frozen rows")
    for row in ROWS:
        summary = summaries.get(row.key)
        if not isinstance(summary, Mapping):
            continue
        if summary.get("status") != "terminal":
            violations.append(f"nonterminal environment summary: {row.key}")
        if summary.get("integrity_status") != "pass":
            violations.append(f"environment integrity failure: {row.key}")
        metrics = summary.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != {
            key for key, _ in METRICS
        }:
            violations.append(f"metric set is incomplete: {row.key}")
    return not violations, violations


def _metric_curves(
    payload: Mapping[str, Any] | None,
    environment: str,
    metric: str,
) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    summaries = payload.get("environments")
    if not isinstance(summaries, Mapping):
        return None
    summary = summaries.get(environment)
    if not isinstance(summary, Mapping):
        return None
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    curves = metrics.get(metric)
    return curves if isinstance(curves, Mapping) else None


def _progress_state(
    payload: Mapping[str, Any] | None,
    environment: str,
) -> tuple[str, str]:
    if payload is None:
        return "pending", "0 / 10 training jobs terminal"
    summaries = payload.get("environments")
    if not isinstance(summaries, Mapping):
        return "pending", "0 / 10 training jobs terminal"
    summary = summaries.get(environment)
    if not isinstance(summary, Mapping):
        return "pending", "0 / 10 training jobs terminal"
    return str(summary.get("status", "pending")), str(summary.get("note", ""))


def _plot_curve(ax: Any, curve: Mapping[str, Any], *, color: str, label: str) -> None:
    passes = curve.get("passes")
    mean = curve.get("mean")
    if not isinstance(passes, list) or not isinstance(mean, list):
        raise ValueError(f"{label} curve is missing passes or mean")
    if len(passes) != len(mean) or not passes:
        raise ValueError(f"{label} curve has inconsistent lengths")
    ax.plot(passes, mean, color=color, linewidth=2.2, marker="o", markersize=3.8)
    seed_curves = curve.get("seeds", {})
    if isinstance(seed_curves, Mapping):
        for seed in EXPECTED_SEEDS:
            values = seed_curves.get(str(seed))
            if isinstance(values, list) and len(values) == len(passes):
                ax.plot(passes, values, color=color, linewidth=0.8, alpha=0.28)


def render(
    output: Path,
    sidecar: Path,
    payload: Mapping[str, Any] | None,
    payload_path: Path | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#64748B",
            "ytick.color": "#64748B",
        }
    )
    fig = plt.figure(figsize=(24, 16), facecolor="#F8FAFC")
    grid = fig.add_gridspec(
        len(ROWS),
        len(METRICS) + 1,
        width_ratios=[1.8, 3, 3, 3, 3],
        left=0.035,
        right=0.985,
        top=0.865,
        bottom=0.085,
        wspace=0.12,
        hspace=0.32,
    )

    fig.text(
        0.035,
        0.955,
        "Clean 0.5B language-policy comparison",
        fontsize=27,
        fontweight="bold",
        color="#0F172A",
        va="top",
    )
    fig.text(
        0.035,
        0.918,
        "Online verified MaxEnt vs compute-matched Dr.GRPO · 12 passes · seeds 43/44/45/46/47",
        fontsize=15,
        color="#475569",
        va="top",
    )
    fig.text(0.685, 0.955, "Dr.GRPO", color="#2563EB", fontsize=13, fontweight="bold")
    fig.text(0.77, 0.955, "Verified MaxEnt", color="#E11D48", fontsize=13, fontweight="bold")

    terminal = False
    violations: list[str] = []
    if payload is not None:
        terminal_candidate, terminal_violations = _terminal_contract(payload)
        if payload.get("status") == "terminal":
            if not terminal_candidate:
                raise ValueError(
                    "terminal payload violates contract: "
                    + "; ".join(terminal_violations)
                )
            terminal = True
        else:
            # Missing cells are expected during admission and training progress.
            # Only a payload claiming terminality can violate the terminal
            # completeness contract.
            violations = []

    for column, (_, label) in enumerate(METRICS, start=1):
        position = grid[0, column].get_position(fig)
        fig.text(
            (position.x0 + position.x1) / 2,
            0.883,
            label,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="#1E293B",
        )

    for row_index, row in enumerate(ROWS):
        label_ax = fig.add_subplot(grid[row_index, 0])
        label_ax.axis("off")
        label_ax.text(0.0, 0.69, row.label, fontsize=16, fontweight="bold", color="#0F172A")
        label_ax.text(0.0, 0.39, row.model, fontsize=10.5, color="#475569")
        label_ax.text(0.0, 0.15, row.interface, fontsize=9.5, color="#64748B")

        for column, (metric, _) in enumerate(METRICS, start=1):
            ax = fig.add_subplot(grid[row_index, column])
            ax.set_facecolor("#FFFFFF")
            ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
            curves = _metric_curves(payload, row.key, metric)
            if curves is None:
                progress_status, progress_note = _progress_state(payload, row.key)
                if progress_status == "ineligible":
                    headline, color = "INELIGIBLE", "#DC2626"
                elif progress_status == "training_running":
                    headline, color = "TRAINING RUNNING", "#D97706"
                elif progress_status in {"admission_passed", "viability_pending"}:
                    headline, color = "ADMISSION PASSED", "#059669"
                elif progress_status == "admission_running":
                    headline, color = "ADMISSION RUNNING", "#D97706"
                else:
                    headline, color = "AWAITING CLEAN COHORT", "#94A3B8"
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(
                    0.5,
                    0.56,
                    headline,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    fontweight="bold",
                    color=color,
                )
                ax.text(
                    0.5,
                    0.33,
                    progress_note or "0 / 10 main jobs terminal",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#64748B",
                    wrap=True,
                )
                continue
            for arm, color in ((ARMS[0], "#2563EB"), (ARMS[1], "#E11D48")):
                curve = curves.get(arm)
                if not isinstance(curve, Mapping):
                    raise ValueError(f"missing {row.key}/{metric}/{arm} curve")
                _plot_curve(ax, curve, color=color, label=arm)
            ax.set_xlim(0, 12)
            ax.set_xlabel("pass", fontsize=8)

    progress_states = [_progress_state(payload, row.key)[0] for row in ROWS]
    ineligible_count = progress_states.count("ineligible")
    running_count = sum(
        state in {"admission_running", "training_running"} for state in progress_states
    )
    passed_count = sum(
        state in {"admission_passed", "viability_pending", "terminal"}
        for state in progress_states
    )
    launched_count = int(
        payload.get("launched_job_count", len(payload.get("jobs", [])))
        if payload is not None
        else 0
    )
    if terminal:
        status = "TERMINAL · 80/80 integrity-clean"
        status_color = "#475569"
    elif payload is not None:
        status = (
            f"TRAINING + ADMISSION PROGRESS · {running_count} rows active · "
            f"{ineligible_count} stopped/ineligible · "
            f"{launched_count}/{EXPECTED_JOB_COUNT} main jobs launched"
        )
        status_color = "#B45309" if ineligible_count == 0 else "#B91C1C"
    else:
        status = "DESIGN SURFACE · 0/80 jobs launched"
        status_color = "#475569"
    fig.text(0.035, 0.035, status, fontsize=12, fontweight="bold", color=status_color)
    fig.text(
        0.985,
        0.035,
        "No historical values are copied into this cohort",
        ha="right",
        fontsize=10.5,
        color="#64748B",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output.suffix.lower() == ".png":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    provenance = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "terminal"
            if terminal
            else "admission_progress"
            if payload is not None
            else "design_empty"
        ),
        "expected_job_count": EXPECTED_JOB_COUNT,
        "observed_job_count": (
            int(payload.get("launched_job_count", len(payload.get("jobs", []))))
            if payload
            else 0
        ),
        "rows": [row.__dict__ for row in ROWS],
        "metrics": [key for key, _ in METRICS],
        "arms": list(ARMS),
        "seeds": list(EXPECTED_SEEDS),
        "historical_values_imported": False,
        "contract_violations": violations,
        "hashes": {
            "protocol_sha256": _sha256(PROTOCOL),
            "plot_source_sha256": _sha256(Path(__file__).resolve()),
            "results_payload_sha256": _sha256(payload_path) if payload_path else None,
            "figure_sha256": _sha256(output),
        },
        "output": str(output.relative_to(ROOT)),
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(sidecar)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--results", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_payload(args.results)
    payload_path = args.results.resolve() if args.results is not None else None
    render(args.output.resolve(), args.sidecar.resolve(), payload, payload_path)
    rendered_status = (
        "terminal"
        if payload and payload.get("status") == "terminal"
        else "admission_progress"
        if payload
        else "design_empty"
    )
    print(
        f"[clean-eight-environment-figure] "
        f"output={args.output} status={rendered_status}"
    )


if __name__ == "__main__":
    main()
