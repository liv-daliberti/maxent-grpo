#!/usr/bin/env python3
"""Render the compact five-domain E68/E69 live decision view."""

from __future__ import annotations

import json
import math
import copy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import plot_e61r1_e58_vs_grpo_12pass as historical


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/e68_e58_vs_grpo_05b_12ep_live"
AUDIT = ROOT / "var/artifacts/e69_gate2_compute_matched_screen_audit_latest.json"
R1_IDENTITY = ROOT / "var/artifacts/e69_gate2_r1_execution_repair_identity.json"
R2_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e69_gate2_r2_route_endpoint_bookkeeping_repair_identity.json"
)
R3_IDENTITY = (
    ROOT
    / "var/artifacts/"
    "e69_gate2_r3_math_dev_evaluation_split_repair_identity.json"
)
ROUTE_SNAPSHOTS = ROOT / "var/artifacts/e69_gate2_route_temporal_snapshots.json"
R1_ROUTE_SNAPSHOTS = (
    ROOT / "var/artifacts/e69_gate2_r1_route_temporal_snapshots.json"
)
R2_ROUTE_SNAPSHOTS = (
    ROOT / "var/artifacts/e69_gate2_r2_route_temporal_snapshots.json"
)

METRICS = (
    ("greedy", "neutral pass@1"),
    ("mean8", "neutral mean@8"),
    ("pass8", "neutral pass@8"),
    ("distinct8", "mean distinct correct@8"),
)
DOMAIN_SPECS = (
    (
        "Graph coloring",
        "graph_coloring",
        "gce61r1_e58_vs_grpo_05b_12ep",
        192,
    ),
    (
        "Countdown",
        "countdown",
        "cde61r1_e58_vs_grpo_05b_12ep",
        384,
    ),
    (
        "Python factors",
        "python_factor",
        "pye61r1_e58_vs_grpo_05b_12ep",
        384,
    ),
    (
        "MathIR action menu",
        "mathir",
        "mie61r1_e58_vs_grpo_05b_12ep",
        384,
    ),
    (
        "Held-out MATH-500\n(sealed)",
        None,
        "e64_math500_realism",
        384,
    ),
)
HISTORICAL_STYLE = {
    historical.CONTROL: ("#0057A8", "o", "matched Dr.GRPO, 3-seed mean"),
    historical.TREATMENT: ("#D55E00", "D", "historical E58, 3-seed mean"),
    historical.PLUMBING_CONTROL: (
        "#7A3E9D",
        "s",
        "E66 MathIR causal control, 3-seed mean",
    ),
    historical.REPAIR: (
        "#008A5A",
        "^",
        "terminal E68 MathIR, 3-seed mean",
    ),
}
E69_CONTROL_COLOR = "#5B6573"
E69_SUCCESSOR_COLOR = "#009E73"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _run_dir(run_stamp: str, job_id: int) -> Path | None:
    matches = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    return matches[0] if len(matches) == 1 else None


def _evaluation_curve(
    path: Path,
    *,
    steps_per_pass: int,
) -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    by_key: dict[tuple[int, str, int | None], dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("metrics"), dict):
            continue
        kind = str(row.get("evaluation_kind", ""))
        if kind not in {
            "deterministic_greedy_trace_neutral",
            "fixed_seed_sampled_k_neutral",
        }:
            continue
        step = int(row["step"])
        draw = row.get("draw_index")
        by_key[(step, kind, draw if isinstance(draw, int) else None)] = row
    result: dict[int, dict[str, float]] = {}
    for (step, kind, _draw), row in by_key.items():
        metrics = row["metrics"]
        values = result.setdefault(step // steps_per_pass, {})
        if kind == "deterministic_greedy_trace_neutral":
            values["greedy"] = float(metrics["any_correct_at_k"])
        else:
            values.update(
                {
                    "mean8": float(metrics["mean_at_k"]),
                    "pass8": float(metrics["any_correct_at_k"]),
                    "distinct8": float(
                        metrics["distinct_correct_modes_at_k"]
                    ),
                }
            )
    return {
        str(pass_index): values
        for pass_index, values in result.items()
        if set(values) >= {metric for metric, _ in METRICS}
    }


def _last_training_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - 1_048_576))
        tail = handle.read().decode("utf-8", errors="replace")
    latest: int | None = None
    for raw in tail.splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        value = row.get("trainer/global_step", row.get("misc/global_step"))
        if _finite(value):
            latest = int(value)
    return latest


def _refresh_live_state(audit: dict[str, Any]) -> dict[str, Any]:
    live = copy.deepcopy(audit)
    r1 = _read_json(R1_IDENTITY)
    mappings = {
        domain: list(rows)
        for domain, rows in r1.get("mappings", {}).items()
    }
    r2 = _read_json(R2_IDENTITY)
    r2_mapping = r2.get("mapping")
    if isinstance(r2_mapping, dict):
        mappings.setdefault("python_factor", [])
        mappings["python_factor"] = [
            row
            for row in mappings["python_factor"]
            if int(row.get("replacement_job_id", -1))
            != int(r2_mapping.get("invalid_job_id", -2))
        ]
        mappings["python_factor"].append(r2_mapping)
    for domain, rows in mappings.items():
        steps_per_pass = 192 if domain == "graph_coloring" else 384
        for mapping in rows:
            job_id = int(mapping["replacement_job_id"])
            run_dir = _run_dir(str(mapping["run_stamp"]), job_id)
            if run_dir is None:
                continue
            curve = _evaluation_curve(
                run_dir / "eval_mode_coverage_draws.jsonl",
                steps_per_pass=steps_per_pass,
            )
            if curve:
                live.setdefault("curves", {}).setdefault(domain, {})[
                    str(mapping["arm"])
                ] = curve
            for run in live.get("physical_runs", []):
                if int(run.get("job_id", -1)) != job_id:
                    continue
                step = _last_training_step(run_dir / "train_metrics.jsonl")
                if step is not None:
                    run.setdefault("training", {})["latest_step"] = step
    for run in live.get("physical_runs", []):
        if run.get("domain") != "math_dev" or not run.get("run_dir"):
            continue
        step = _last_training_step(
            Path(str(run["run_dir"])) / "train_metrics.jsonl"
        )
        if step is not None:
            run.setdefault("training", {})["latest_step"] = step
    return live


def _historical_points(
    label: str,
    prefix: str,
    steps_per_pass: int,
) -> list[dict[str, Any]]:
    if prefix == "e64_math500_realism":
        points = historical.load_e64_aggregate_points()
    else:
        points = historical._load_points(prefix, steps_per_pass)
    # E66 and E68 are retained only in the domain where their paired terminal
    # causal result is established. The incomplete exploratory domain traces
    # do not enter this decision view.
    if label == "MathIR action menu":
        points.extend(
            historical._load_points(
                "mie66_same_plumbing_control_05b_12ep",
                steps_per_pass,
                arm_alias=historical.PLUMBING_CONTROL,
            )
        )
        points.extend(
            historical._load_points(
                "mie68_separated_support_actuator_05b_12ep",
                steps_per_pass,
            )
        )
    paper_label = (
        historical.E64_DOMAIN
        if prefix == "e64_math500_realism"
        else label
    )
    return historical._paper_checkpoint_points(
        paper_label,
        points,
        steps_per_pass,
    )


def _style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 12)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7.5)
    axis.xaxis.set_major_locator(MaxNLocator(7, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(5))


def _plot_historical(
    axis: plt.Axes,
    points: list[dict[str, Any]],
    metric: str,
    *,
    mathir: bool,
) -> list[float]:
    values: list[float] = []
    arms = [historical.CONTROL, historical.TREATMENT]
    if mathir:
        arms.extend([historical.PLUMBING_CONTROL, historical.REPAIR])
    for arm in arms:
        xs, means, lows, highs = historical._complete_mean(points, arm, metric)
        if not xs:
            continue
        color, marker, _ = HISTORICAL_STYLE[arm]
        values.extend(lows)
        values.extend(highs)
        axis.fill_between(
            xs,
            lows,
            highs,
            color=color,
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            xs,
            means,
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=3.7,
            markerfacecolor=(
                color if arm == historical.CONTROL else "white"
            ),
            markeredgewidth=1.0,
            zorder=3,
        )
    return values


def _audit_curve(
    audit: dict[str, Any],
    domain: str,
    arm: str,
    *,
    allow: bool = True,
) -> dict[str, dict[str, float]]:
    if not allow:
        return {}
    curve = audit.get("curves", {}).get(domain, {}).get(arm, {})
    return curve if isinstance(curve, dict) else {}


def _plot_e69(
    axis: plt.Axes,
    audit: dict[str, Any],
    domain: str,
    metric: str,
) -> list[float]:
    r1_exists = R1_IDENTITY.is_file()
    allow_graph = domain != "graph_coloring" or r1_exists
    allow_successor = (
        allow_graph and (domain != "python_factor" or r1_exists)
    )
    curves = (
        (
            _audit_curve(
                audit,
                domain,
                "grpo",
                allow=allow_graph,
            ),
            E69_CONTROL_COLOR,
            "o",
        ),
        (
            _audit_curve(
                audit,
                domain,
                "verified_route_successor",
                allow=allow_successor,
            ),
            E69_SUCCESSOR_COLOR,
            "^",
        ),
    )
    values: list[float] = []
    for curve, color, marker in curves:
        rows = sorted(
            (
                (int(pass_index), float(metrics[metric]))
                for pass_index, metrics in curve.items()
                if isinstance(metrics, dict) and _finite(metrics.get(metric))
            ),
            key=lambda row: row[0],
        )
        if not rows:
            continue
        xs = [row[0] for row in rows]
        ys = [row[1] for row in rows]
        values.extend(ys)
        axis.plot(
            xs,
            ys,
            color=color,
            linestyle=(0, (4, 2)),
            linewidth=2.1,
            marker=marker,
            markersize=4.2,
            zorder=5,
        )
    return values


def _latest_step(
    audit: dict[str, Any],
    *,
    domain: str,
    arm: str,
) -> int | None:
    matches = [
        row
        for row in audit.get("physical_runs", [])
        if row.get("domain") == domain and row.get("arm") == arm
    ]
    if len(matches) != 1:
        return None
    value = matches[0].get("training", {}).get("latest_step")
    return int(value) if _finite(value) else None


def _route_pairs() -> dict[str, int]:
    path = (
        R2_ROUTE_SNAPSHOTS
        if R2_IDENTITY.is_file()
        else (
            R1_ROUTE_SNAPSHOTS
            if R1_IDENTITY.is_file()
            else ROUTE_SNAPSHOTS
        )
    )
    payload = _read_json(path)
    return {
        domain: int(
            payload.get("temporal_reproductions", {})
            .get(domain, {})
            .get("post_replay_neutral_reproduction_pairs", 0)
        )
        for domain in (
            "graph_coloring",
            "countdown",
            "python_factor",
            "mathir",
        )
    }


def _status_line(audit: dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    control_step = _latest_step(audit, domain="math_dev", arm="grpo")
    endpoint_step = _latest_step(
        audit,
        domain="math_dev",
        arm="verified_first_global_replay_canonical",
    )
    pairs = _route_pairs()
    r1 = _read_json(R1_IDENTITY)
    jobs = [
        int(row["replacement_job_id"])
        for rows in r1.get("mappings", {}).values()
        for row in rows
        if isinstance(row, dict) and _finite(row.get("replacement_job_id"))
    ]
    r2 = _read_json(R2_IDENTITY)
    r2_mapping = r2.get("mapping")
    if isinstance(r2_mapping, dict) and _finite(
        r2_mapping.get("replacement_job_id")
    ):
        repair_parts = ["R1 Graph", "R2 Python"]
    else:
        repair_parts = ["R1 Graph/Python"] if jobs else []
    r3 = _read_json(R3_IDENTITY)
    r3_jobs = [
        int(row["replacement_job_id"])
        for row in r3.get("mappings", [])
        if isinstance(row, dict) and _finite(row.get("replacement_job_id"))
    ]
    if r3_jobs:
        repair_parts.append("R3 MATH")
    repair_text = (
        "accepted repair chain " + " \N{RIGHTWARDS ARROW} ".join(repair_parts)
        if repair_parts
        else "repair freezing/submission pending"
    )
    return (
        f"{repair_text}  |  Gate-2 terminal "
        f"{summary.get('terminal_physical_runs', 0)}/18  |  "
        f"MATH-dev {control_step if control_step is not None else '—'}/2304 "
        f"and {endpoint_step if endpoint_step is not None else '—'}/2304  |  "
        "temporal route pairs "
        f"C={pairs['countdown']}, M={pairs['mathir']}, "
        f"G={pairs['graph_coloring']}, P={pairs['python_factor']}"
    )


def main() -> None:
    audit = _refresh_live_state(_read_json(AUDIT))
    figure, axes = plt.subplots(
        len(DOMAIN_SPECS),
        len(METRICS),
        figsize=(15.5, 13.0),
        sharex=True,
        squeeze=False,
    )

    for row, (label, audit_domain, prefix, steps_per_pass) in enumerate(
        DOMAIN_SPECS
    ):
        points = _historical_points(label, prefix, steps_per_pass)
        for column, (metric, title) in enumerate(METRICS):
            axis = axes[row, column]
            if row == 0:
                axis.set_title(title, fontsize=10, fontweight="bold")
            plotted = _plot_historical(
                axis,
                points,
                metric,
                mathir=label == "MathIR action menu",
            )
            if audit_domain is not None:
                plotted.extend(_plot_e69(axis, audit, audit_domain, metric))
            if not plotted:
                message = (
                    "N/A — one verified\nanswer class"
                    if audit_domain is None and metric == "distinct8"
                    else "awaiting clean evidence"
                )
                axis.text(
                    0.5,
                    0.5,
                    message,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
                    fontsize=8,
                )
            upper = max(plotted, default=1.0)
            axis.set_ylim(0, upper * 1.09 if upper > 0 else 1)
            _style_axis(axis)
            if column == 0:
                axis.set_ylabel(label, fontsize=9, fontweight="bold")
        if audit_domain == "graph_coloring" and not R1_IDENTITY.is_file():
            axes[row, -1].text(
                0.98,
                0.06,
                "E69 Graph excluded; fresh R1 pending",
                transform=axes[row, -1].transAxes,
                ha="right",
                fontsize=7,
                color="#666666",
            )
        if audit_domain == "python_factor" and not R1_IDENTITY.is_file():
            axes[row, -1].text(
                0.98,
                0.06,
                "E69 Python successor excluded; fresh R1 pending",
                transform=axes[row, -1].transAxes,
                ha="right",
                fontsize=7,
                color="#666666",
            )
        if audit_domain is None:
            axes[row, 0].text(
                0.98,
                0.94,
                "sealed historical evidence; no E69 evaluation",
                transform=axes[row, 0].transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#666666",
            )

    for axis in axes[-1]:
        axis.set_xlabel("training passes", fontsize=8)

    handles = [
        Line2D(
            [],
            [],
            color=color,
            linewidth=2.0,
            marker=marker,
            markerfacecolor=(
                color if arm == historical.CONTROL else "white"
            ),
            label=label,
        )
        for arm, (color, marker, label) in HISTORICAL_STYLE.items()
    ]
    handles.extend(
        [
            Line2D(
                [],
                [],
                color=E69_CONTROL_COLOR,
                linestyle=(0, (4, 2)),
                linewidth=2.1,
                marker="o",
                label="live E69 Gate2 Dr.GRPO, seed 43",
            ),
            Line2D(
                [],
                [],
                color=E69_SUCCESSOR_COLOR,
                linestyle=(0, (4, 2)),
                linewidth=2.1,
                marker="^",
                label="live E69 Gate2 route successor, seed 43",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=8.2,
    )
    figure.suptitle(
        "Five-domain outcome view — terminal E68 context and live E69 Gate 2",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.91,
        _status_line(audit),
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    figure.text(
        0.5,
        0.012,
        "Four frozen outcome metrics only. E66/E68 appear only for the "
        "terminal MathIR causal result. Superseded Graph/Python attempts and "
        "controller telemetry are omitted; MATH-500 remains sealed.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0.035, 0.035, 0.995, 0.885), h_pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"[e68-e69-live] wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
