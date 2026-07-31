#!/usr/bin/env python3
"""Render E70 in the compact historical E68 trajectory format.

Only identity-bound E70/Stage-B artifacts are eligible.  Historical E58,
E61, E66, E68, and E69 values are never imported.  During execution, every
available seed trajectory is shown without carry-forward and incomplete
environment row remains visibly marked as development/admission work.  The
paper surface is restricted to the four frozen outcome/discovery panels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "paper/figures/e68_e58_vs_grpo_05b_12ep_terminal_provenance.png"
)
DEFAULT_SIDECAR = (
    ROOT / "var/artifacts/clean_05b_eight_environment_figure_provenance.json"
)
WIDE_SIDECAR_ALIAS = (
    ROOT / "var/artifacts/e70_clean_05b_wide_live_provenance.json"
)
AUDIT = ROOT / "var/artifacts/e70_clean_stage_a_05b_audit_latest.json"
IDENTITY = ROOT / "var/artifacts/e70_clean_stage_a_05b_identity.json"
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e70_clean_verified_maxent_vs_compute_matched_drgrpo_stage_a_05b.md"
)

CONTROL = "grpo"
TREATMENT = "verified_first_global_replay_canonical"
BLUE = "#0057A8"
ORANGE = "#D55E00"
SEEDS = (43, 44, 45, 46, 47)
SEED_STYLES = {
    43: "-",
    44: (0, (5, 2)),
    45: (0, (1.5, 1.5)),
    46: "-.",
    47: (0, (7, 2, 1.5, 2)),
}
DOMAIN_SPECS = (
    (
        "Graph coloring",
        "graph_coloring",
        "gce70_clean_stage_a_05b_12pass",
        192,
    ),
    (
        "Countdown",
        "countdown",
        "cde70_clean_stage_a_05b_12pass",
        384,
    ),
    (
        "Python factors",
        "python_factor",
        "pye70_clean_stage_a_05b_12pass",
        384,
    ),
    (
        "MathIR action menu",
        "mathir",
        "mie70_clean_stage_a_05b_12pass",
        384,
    ),
    (
        "PointMaze geometry shift\n(replacement configuration)",
        "point_maze_geometry_shift",
        None,
        None,
    ),
    (
        "PantryPlan",
        "pantry_plan",
        "ppe70_clean_stage_b_05b_12pass",
        32,
    ),
    ("PointMaze", "point_maze", None, None),
    ("AntMaze", "ant_maze", None, None),
)
PANELS = (
    ("greedy", "neutral pass@1", False),
    ("pass8", "neutral pass@8", False),
    ("distinct8", "mean # distinct correct@8", True),
    (
        "online_canonical_tracked_outcomes",
        "cumulative verified discoveries",
        True,
    ),
)
QUALITY_METRICS = {"greedy", "pass8", "distinct8"}


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _curve_path(prefix: str) -> Path:
    return ROOT / f"var/artifacts/{prefix}_scaling_curve.json"


def _load_points(prefix: str | None, steps_per_pass: int | None) -> list[dict[str, Any]]:
    if prefix is None or steps_per_pass is None:
        return []
    payload = _read_json(_curve_path(prefix), [])
    if not isinstance(payload, list):
        return []
    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        arm = row.get("arm")
        seed = row.get("seed")
        step = row.get("step")
        if (
            arm not in {CONTROL, TREATMENT}
            or seed not in SEEDS
            or not _finite(step)
            or row.get("split") != "multi_answer"
        ):
            continue
        key = (str(arm), int(seed), int(step))
        point = merged.setdefault(
            key,
            {
                "arm": str(arm),
                "seed": int(seed),
                "step": int(step),
                "passes": float(step) / steps_per_pass,
            },
        )
        if _finite(row.get("training_passes")):
            point["passes"] = float(row["training_passes"])
        for metric, value in row.items():
            if _finite(value):
                point[str(metric)] = float(value)
    return [merged[key] for key in sorted(merged)]


def _load_maze_stage_b_points(domain: str) -> list[dict[str, Any]]:
    if domain == "point_maze":
        stem = "point_maze_stage_b_05b_12pass"
    elif domain == "point_maze_geometry_shift":
        stem = "point_maze_geometry_shift_stage_b_05b_12pass"
    elif domain == "ant_maze":
        stem = "ant_maze_stage_b_05b_12pass"
    else:
        return []
    merged: dict[tuple[str, int, float], dict[str, Any]] = {}
    for arm in (CONTROL, TREATMENT):
        for seed in SEEDS:
            path = ROOT / f"var/artifacts/{stem}_{arm}_s{seed}.metrics.jsonl"
            if not path.is_file():
                continue
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping) or not _finite(row.get("training_passes")):
                    continue
                passes = round(float(row["training_passes"]), 9)
                key = (arm, seed, passes)
                point = merged.setdefault(
                    key,
                    {"arm": arm, "seed": seed, "passes": passes},
                )
                for metric, value in row.items():
                    if _finite(value):
                        point[str(metric)] = float(value)
                if _finite(row.get("replay_active_modes")):
                    point["canonical_replay_available_modes"] = float(
                        row["replay_active_modes"]
                    )
                if _finite(row.get("canonical_tracked_outcomes")):
                    point["online_canonical_tracked_outcomes"] = float(
                        row["canonical_tracked_outcomes"]
                    )
                tracked_prompts = row.get("canonical_tracked_prompts")
                tracked_outcomes = row.get("canonical_tracked_outcomes")
                if (
                    _finite(tracked_prompts)
                    and float(tracked_prompts) > 0
                    and _finite(tracked_outcomes)
                ):
                    point["online_canonical_mean_support_per_prompt"] = (
                        float(tracked_outcomes) / float(tracked_prompts)
                    )
    return [merged[key] for key in sorted(merged)]


def _load_constructive_stage_b_points() -> list[dict[str, Any]]:
    identity = _read_json(
        ROOT / "var/artifacts/constructive_code_v8_stage_b_identity.json", {}
    )
    cells = identity.get("cells", {})
    if not isinstance(cells, Mapping):
        return []
    merged: dict[tuple[str, int, float], dict[str, Any]] = {}
    for arm in (CONTROL, TREATMENT):
        for seed in SEEDS:
            record = cells.get(f"{arm}/s{seed}")
            if not isinstance(record, Mapping):
                continue
            raw_path = record.get("metrics")
            if not isinstance(raw_path, str):
                continue
            path = ROOT / raw_path
            if not path.is_file():
                continue
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping):
                    continue
                raw_passes = row.get("misc/prompt_epoch")
                if not _finite(raw_passes):
                    continue
                passes = float(raw_passes)
                key = (arm, seed, passes)
                point = merged.setdefault(
                    key,
                    {
                        "arm": arm,
                        "seed": seed,
                        "step": int(round(passes * 4)),
                        "passes": passes,
                    },
                )
                for metric, value in row.items():
                    if _finite(value):
                        point[str(metric)] = float(value)
                for source, destination in (
                    ("eval/multi_answer/accuracy", "greedy"),
                    ("eval/multi_answer/sampled_mean_at_8", "mean8"),
                    ("eval/multi_answer/sampled_any_correct_at_8", "pass8"),
                    ("eval/multi_answer/sampled_distinct_correct_at_8", "distinct8"),
                ):
                    if _finite(row.get(source)):
                        point[destination] = float(row[source])
    return [merged[key] for key in sorted(merged)]


def _series(
    points: Iterable[Mapping[str, Any]],
    arm: str,
    seed: int,
    metric: str,
) -> list[tuple[float, float]]:
    by_pass: dict[float, float] = {}
    for point in points:
        if (
            point.get("arm") == arm
            and point.get("seed") == seed
            and _finite(point.get("passes"))
            and _finite(point.get(metric))
        ):
            by_pass[round(float(point["passes"]), 9)] = float(point[metric])
    return sorted(by_pass.items())


def _mean_series(
    points: Iterable[Mapping[str, Any]],
    arm: str,
    metric: str,
) -> tuple[list[float], list[float], list[float], list[float], list[int]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for seed in SEEDS:
        for pass_index, value in _series(points, arm, seed, metric):
            grouped[pass_index].append(value)
    xs = sorted(grouped)
    return (
        xs,
        [statistics.fmean(grouped[x]) for x in xs],
        [min(grouped[x]) for x in xs],
        [max(grouped[x]) for x in xs],
        [len(grouped[x]) for x in xs],
    )


def _plot_metric(
    axis: plt.Axes,
    points: list[dict[str, Any]],
    metric: str,
) -> tuple[list[float], int]:
    plotted: list[float] = []
    max_seed_count = 0
    for arm, color, marker in (
        (CONTROL, BLUE, "o"),
        (TREATMENT, ORANGE, "D"),
    ):
        for seed in SEEDS:
            rows = _series(points, arm, seed, metric)
            if not rows:
                continue
            axis.plot(
                [row[0] for row in rows],
                [row[1] for row in rows],
                color=color,
                linestyle=SEED_STYLES[seed],
                linewidth=1.15,
                marker=marker,
                markersize=2.8 if arm == CONTROL else 3.1,
                markerfacecolor=color if arm == CONTROL else "none",
                markeredgecolor=color,
                markeredgewidth=1.05,
                alpha=0.68,
                zorder=2,
            )
        xs, means, lows, highs, counts = _mean_series(points, arm, metric)
        if not xs:
            continue
        max_seed_count = max(max_seed_count, max(counts))
        plotted.extend(lows)
        plotted.extend(highs)
        axis.fill_between(
            xs,
            lows,
            highs,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            xs,
            means,
            color=color,
            linewidth=2.8,
            marker=marker,
            markersize=4.2 if arm == CONTROL else 4.6,
            markerfacecolor=color if arm == CONTROL else "none",
            markeredgecolor=color,
            markeredgewidth=1.35,
            zorder=4,
        )
        last_x = xs[-1]
        last_y = means[-1]
        last_n = counts[-1]
        axis.annotate(
            f"n={last_n}",
            (last_x, last_y),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=4.7,
            color=color,
            alpha=0.85,
            clip_on=True,
        )
    return plotted, max_seed_count


def _style_axis(axis: plt.Axes, metric: str, plotted: list[float], integer: bool) -> None:
    axis.set_xlim(0, 12)
    axis.grid(axis="y", color="#dddddd", linewidth=0.55, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.7, labelsize=7.5)
    axis.xaxis.set_major_locator(MaxNLocator(7))
    if not plotted:
        axis.set_ylim(0.0, 1.0)
    else:
        high = max(plotted)
        axis.set_ylim(0.0, high * 1.08 if high > 0 else 1.0)
    if integer:
        axis.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    else:
        axis.yaxis.set_major_locator(MaxNLocator(5))


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _status_for_pending_row(domain: str) -> tuple[str, str, str]:
    if domain == "point_maze_geometry_shift":
        stage_b_audit = _read_json(
            ROOT
            / "var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_audit.json",
            {},
        )
        if stage_b_audit.get("status") == "pass":
            return (
                "TEN REPLACEMENT RUNS PASSED AUDIT",
                "five paired seeds × 12 passes; configuration-level replacement",
                "#008A5A",
            )
        if stage_b_audit.get("status") == "fail":
            return (
                "TEN REPLACEMENT RUNS FAILED AUDIT",
                "result stopped fail-closed",
                "#B91C1C",
            )
        stage_b_identity = _read_json(
            ROOT
            / "var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_identity.json",
            {},
        )
        cells = stage_b_identity.get("jobs", stage_b_identity.get("cells", {}))
        if isinstance(cells, Mapping) and cells:
            return (
                "TEN REPLACEMENT RUNS LAUNCHED",
                f"{len(cells)}/10 identity-bound cells; seeds 43–47 × two arms",
                "#B36B00",
            )
        paired_audit = _read_json(
            ROOT / "var/artifacts/point_maze_geometry_shift_paired_smoke_v1_audit.json",
            {},
        )
        if paired_audit.get("status") == "pass":
            return (
                "PAIRED ONLINE SMOKE PASSED",
                "replacement Stage B is authorized next",
                "#008A5A",
            )
        if paired_audit.get("status") == "fail":
            return (
                "PAIRED ONLINE SMOKE FAILED",
                "replacement row stopped fail-closed",
                "#B91C1C",
            )
        paired_identity = _read_json(
            ROOT / "var/artifacts/point_maze_geometry_shift_paired_smoke_v1_identity.json",
            {},
        )
        if paired_identity:
            return (
                "PAIRED ONLINE SMOKE RUNNING",
                "two identity-bound development arms; terminal audit pending",
                "#B36B00",
            )
        viability = _read_json(
            ROOT / "var/artifacts/point_maze_geometry_shift_05b_viability_v1.json",
            {},
        )
        if viability.get("status") == "pass":
            summary = viability.get("summary", {})
            return (
                "0.5B GEOMETRY-SHIFT GATE PASSED",
                f"{summary.get('verified_completions', '?')} verified routes; "
                f"{summary.get('multimode_prompts', '?')}/4 multimode maps; paired smoke next",
                "#008A5A",
            )
        if viability.get("status") == "fail":
            summary = viability.get("summary", {})
            return (
                "0.5B GEOMETRY-SHIFT GATE FAILED",
                f"{summary.get('verified_completions', 0)} verified routes; "
                "replacement row stopped fail-closed",
                "#B91C1C",
            )
        identity = _read_json(
            ROOT / "var/artifacts/point_maze_geometry_shift_viability_v1_identity.json",
            {},
        )
        if identity:
            return (
                "0.5B GEOMETRY-SHIFT GATE RUNNING",
                f"4 unseen development maps × 64 trajectories; job "
                f"{identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        admission = _read_json(
            ROOT / "var/artifacts/point_maze_geometry_shift_v1_admission_audit.json",
            {},
        )
        if admission.get("status") == "pass":
            return (
                "GEOMETRY-SHIFT ADMISSION PASSED",
                "32/32 certified routes; frozen 0.5B viability gate next",
                "#008A5A",
            )
        return (
            "GEOMETRY-SHIFT ADMISSION PENDING",
            "prospective configuration-level replacement; no Stage-B result",
            "#777777",
        )
    if domain == "constructive_code":
        v8_viability = _read_json(
            ROOT / "var/artifacts/constructive_code_v8_coder_15b_viability.json", {}
        )
        if v8_viability.get("status") == "pass":
            summary = v8_viability.get("summary", {})
            return (
                "CODER-1.5B CAPACITY GATE PASSED",
                f"{summary.get('accepted_candidates', '?')} accepted candidates; "
                "separate 1.5B paired online qualification is next",
                "#008A5A",
            )
        if v8_viability.get("status") == "fail":
            summary = v8_viability.get("summary", {})
            return (
                "CODER-1.5B CAPACITY GATE FAILED",
                f"{summary.get('accepted_candidates', 0)}/192 accepted candidates; "
                "registered capacity ladder stopped",
                "#B91C1C",
            )
        v8_identity = _read_json(
            ROOT / "var/artifacts/constructive_code_v8_coder_15b_viability_identity.json",
            {},
        )
        if v8_identity:
            return (
                "CODER-1.5B CAPACITY GATE RUNNING",
                f"192 frozen development samples, job "
                f"{v8_identity.get('job_id', 'unknown')}; separate from 0.5B band",
                "#B36B00",
            )
        v7_viability = _read_json(
            ROOT / "var/artifacts/constructive_code_v7_post_sft_viability.json", {}
        )
        if v7_viability.get("status") == "pass":
            summary = v7_viability.get("summary", {})
            return (
                "CODER-0.5B V7 POST-SFT GATE PASSED",
                f"{summary.get('accepted_candidates', '?')} accepted candidates; "
                "paired online qualification is next",
                "#008A5A",
            )
        if v7_viability.get("status") == "fail":
            summary = v7_viability.get("summary", {})
            return (
                "CODER-0.5B V7 POST-SFT GATE FAILED",
                f"{summary.get('accepted_candidates', 0)}/192 accepted candidates; "
                "Stage B remains fail-closed",
                "#B91C1C",
            )
        v7_identity = _read_json(
            ROOT / "var/artifacts/constructive_code_v7_sft_gate_identity.json", {}
        )
        v7_sft = _read_json(
            ROOT / "var/artifacts/constructive_code_v7_sft.json", {}
        )
        if v7_sft.get("status") == "pass" and v7_identity:
            return (
                "CODER-0.5B V7 POST-SFT GATE RUNNING",
                f"32/32 train-only SFT updates passed; 192-request dev gate, "
                f"job {v7_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        if v7_identity:
            return (
                "CODER-0.5B V7 TRAIN-ONLY SFT RUNNING",
                f"64 examples, 32 frozen updates, job "
                f"{v7_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v6_stage_audit = _read_json(
            ROOT / "var/artifacts/constructive_code_v6_stage_b_audit.json", {}
        )
        if v6_stage_audit.get("status") == "pass":
            return (
                "TEN V6 STAGE-B RUNS PASSED",
                "five paired seeds × 12 passes; terminal audit passed",
                "#008A5A",
            )
        if v6_stage_audit.get("status") == "fail":
            return ("TEN V6 STAGE-B RUNS FAILED AUDIT", "result stopped", "#B91C1C")
        v6_stage_identity = _read_json(
            ROOT / "var/artifacts/constructive_code_v6_stage_b_identity.json", {}
        )
        v6_cells = v6_stage_identity.get("cells", {})
        if isinstance(v6_cells, Mapping) and v6_cells:
            return (
                "TEN V6 STAGE-B RUNS LAUNCHED",
                f"{len(v6_cells)}/10 identity-bound cells; seeds 43–47 × two arms",
                "#B36B00",
            )
        v6_paired_audit = _read_json(
            ROOT / "var/artifacts/constructive_code_v6_paired_smoke_audit.json", {}
        )
        if v6_paired_audit.get("status") == "pass":
            return (
                "V6 PAIRED ONLINE SMOKE PASSED",
                "three development updates per arm; final launch queued",
                "#008A5A",
            )
        if v6_paired_audit.get("status") == "fail":
            return ("V6 PAIRED ONLINE SMOKE FAILED", "Stage B stopped", "#B91C1C")
        v6_paired_identity = _read_json(
            ROOT / "var/artifacts/constructive_code_v6_paired_smoke_identity.json", {}
        )
        if v6_paired_identity:
            return (
                "V6 PAIRED ONLINE SMOKE LAUNCHED",
                "two identity-bound development arms; terminal audit pending",
                "#B36B00",
            )
        v6_viability = _read_json(
            ROOT / "var/artifacts/constructive_code_v6_coder_05b_viability.json", {}
        )
        if v6_viability.get("status") == "pass":
            return (
                "CODER-0.5B V6 VIABILITY PASSED",
                "paired online qualification is queued",
                "#008A5A",
            )
        if v6_viability.get("status") == "fail":
            return ("CODER-0.5B V6 VIABILITY FAILED", "Stage B stopped", "#B91C1C")
        v6_viability_identity = _read_json(
            ROOT / "var/artifacts/constructive_code_v6_coder_05b_viability_identity.json",
            {},
        )
        if v6_viability_identity:
            return (
                "CODER-0.5B V6 VIABILITY QUEUED",
                f"192 frozen development samples, job "
                f"{v6_viability_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v6_audit = _read_json(
            ROOT / "var/artifacts/constructive_code_v6_gate_audit.json", {}
        )
        if v6_audit.get("status") == "pass":
            return (
                "TEN-TASK V6 GATE PASSED",
                "960 selected-suite replays passed; Coder-0.5B probe next",
                "#008A5A",
            )
        if v6_audit.get("status") == "fail":
            return ("TEN-TASK V6 GATE FAILED", "Stage B stopped", "#B91C1C")
        v5_viability = _read_json(
            ROOT / "var/artifacts/constructive_code_v5_coder_05b_viability.json",
            {},
        )
        if v5_viability.get("status") == "pass":
            return (
                "CODER-0.5B V5 VIABILITY PASSED",
                "paired online qualification is next",
                "#008A5A",
            )
        if v5_viability.get("status") == "fail":
            return ("CODER-0.5B V5 VIABILITY FAILED", "Stage B stopped", "#B91C1C")
        v5_viability_identity = _read_json(
            ROOT / "var/artifacts/constructive_code_v5_coder_05b_viability_identity.json",
            {},
        )
        if v5_viability_identity:
            return (
                "CODER-0.5B V5 VIABILITY LAUNCHED",
                f"256 frozen development samples, job "
                f"{v5_viability_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v5_audit = _read_json(
            ROOT / "var/artifacts/constructive_code_v5_gate_audit.json", {}
        )
        if v5_audit.get("status") == "pass":
            return (
                "12-TASK V5 GATE PASSED",
                "2,304 checker replays passed; Coder-0.5B probe queued",
                "#008A5A",
            )
        if v5_audit.get("status") == "fail":
            return ("12-TASK V5 GATE FAILED", "Stage B stopped", "#B91C1C")
        v5_identity = _read_json(
            ROOT / "var/artifacts/constructive_code_v5_gate_identity.json", {}
        )
        if v5_identity:
            completed = _count_lines(
                ROOT / "var/artifacts/constructive_code_v5_replays.jsonl"
            )
            detail = (
                f"{min(completed, 2304)}/2304 frozen checker replays"
                if completed
                else f"full-source py3 materialization, job "
                f"{v5_identity.get('job_id', 'unknown')}"
            )
            return ("12-TASK V5 GATE RUNNING", detail, "#B36B00")
        v4_viability = _read_json(
            ROOT
            / "var/artifacts/constructive_code_v4_coder_05b_viability.json",
            {},
        )
        if v4_viability.get("status") == "pass":
            return (
                "CODER-0.5B V4 VIABILITY PASSED",
                "paired online qualification is next",
                "#008A5A",
            )
        if v4_viability.get("status") == "fail":
            return ("CODER-0.5B V4 VIABILITY FAILED", "Stage B stopped", "#B91C1C")
        v4_viability_identity = _read_json(
            ROOT
            / "var/artifacts/constructive_code_v4_coder_05b_viability_identity.json",
            {},
        )
        if v4_viability_identity:
            return (
                "CODER-0.5B V4 VIABILITY RUNNING",
                f"256 frozen development samples, job "
                f"{v4_viability_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v4_audit = _read_json(
            ROOT / "var/artifacts/constructive_code_v4_gate_audit.json", {}
        )
        if v4_audit.get("status") == "pass":
            return (
                "12-TASK V4 GATE PASSED",
                "3,072 checker replays passed; Coder-0.5B viability next",
                "#008A5A",
            )
        if v4_audit.get("status") == "fail":
            return ("12-TASK V4 GATE FAILED", "Stage B stopped", "#B91C1C")
        v4_failure_log = (
            ROOT
            / "var/artifacts/logs/constructive-code-v4-gate-30201021.err"
        )
        if v4_failure_log.is_file():
            failure_text = v4_failure_log.read_text(
                encoding="utf-8", errors="replace"
            )
            if "only 52 held-out Python-3 programs; 64 required" in failure_text:
                return (
                    "12-TASK V4 SOURCE GATE FAILED",
                    "52/64 held-out py3 programs; prospective v5 is next",
                    "#B91C1C",
                )
        v4_identity = _read_json(
            ROOT / "var/artifacts/constructive_code_v4_gate_identity.json", {}
        )
        if v4_identity:
            completed = _count_lines(
                ROOT / "var/artifacts/constructive_code_v4_replays.jsonl"
            )
            detail = (
                f"{min(completed, 3072)}/3072 frozen checker replays"
                if completed
                else f"full-source py3 materialization, job "
                f"{v4_identity.get('job_id', 'unknown')}"
            )
            return ("12-TASK V4 GATE RUNNING", detail, "#B36B00")
        v3_audit = _read_json(
            ROOT / "var/artifacts/constructive_code_v3_gate_audit.json", {}
        )
        if v3_audit.get("status") == "pass":
            return (
                "12-TASK V3 GATE PASSED",
                "problem-disjoint Coder-0.5B viability is next",
                "#008A5A",
            )
        if v3_audit.get("status") == "fail":
            return ("12-TASK V3 GATE FAILED", "Stage B stopped", "#B91C1C")
        v3_failure_log = (
            ROOT
            / "var/artifacts/logs/"
            "constructive-code-v3-gate-30187935.err"
        )
        if v3_failure_log.is_file():
            failure_text = v3_failure_log.read_text(
                encoding="utf-8", errors="replace"
            )
            if "only 31 held-out Python-3 programs; 100 required" in failure_text:
                return (
                    "12-TASK V3 SOURCE GATE FAILED",
                    "359B has 31/100 new correct replays; prospective v4 next",
                    "#B91C1C",
                )
        if (ROOT / "var/artifacts/constructive_code_v3_gate_identity.json").is_file():
            completed = _count_lines(
                ROOT / "var/artifacts/constructive_code_v3_replays.jsonl"
            )
            detail = (
                f"{min(completed, 4800)}/4800 frozen checker replays"
                if completed
                else "12-task source materialization, job 30187935"
            )
            return ("12-TASK V3 GATE RUNNING", detail, "#B36B00")
        audit = _read_json(
            ROOT / "var/artifacts/constructive_code_v2_gate_audit.json", {}
        )
        if audit.get("status") == "pass":
            return (
                "EXECUTABLE GATE PASSED",
                "Coder-0.5B viability is next; no main training cell yet",
                "#008A5A",
            )
        if audit.get("status") == "fail":
            return ("EXECUTABLE GATE FAILED", "Stage B stopped", "#B91C1C")
        completed = _count_lines(
            ROOT / "var/artifacts/constructive_code_v2_replays.jsonl"
        )
        return (
            "EXECUTABLE GATE RUNNING",
            f"{min(completed, 1600)}/1600 frozen checker replays",
            "#B36B00",
        )
    if domain == "pantry_plan":
        stage_b_audit = _read_json(
            ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json", {}
        )
        if stage_b_audit.get("status") == "pass":
            return (
                "TEN STAGE-B RUNS PASSED",
                "five paired seeds × 12 passes; terminal audit passed",
                "#008A5A",
            )
        if stage_b_audit.get("status") == "fail":
            return ("TEN STAGE-B RUNS FAILED AUDIT", "Pantry result stopped", "#B91C1C")
        stage_b_identity = _read_json(
            ROOT / "var/artifacts/pantry_stage_b_05b_12pass_identity.json", {}
        )
        if stage_b_identity:
            jobs = stage_b_identity.get("jobs", {})
            return (
                "TEN STAGE-B RUNS LAUNCHED",
                f"{len(jobs)}/10 identity-bound cells; seeds 43–47 × two arms",
                "#B36B00",
            )
        integration_v3 = _read_json(
            ROOT
            / "var/artifacts/"
            "pantry_support_mask_paired_integration_v3_audit.json",
            {},
        )
        if integration_v3.get("status") == "pass":
            return (
                "96-UPDATE BRIDGE V3 PASSED",
                "semantic + replay paths qualified; ten Stage-B jobs next",
                "#008A5A",
            )
        if integration_v3.get("status") == "fail":
            return (
                "96-UPDATE BRIDGE V3 FAILED",
                "new qualification stopped before Stage B",
                "#B91C1C",
            )
        integration = _read_json(
            ROOT
            / "var/artifacts/"
            "pantry_support_mask_paired_integration_v2_audit.json",
            {},
        )
        if integration.get("status") == "pass":
            return (
                "96-UPDATE BRIDGE PAIR PASSED",
                "semantic + replay paths qualified; ten Stage-B jobs next",
                "#008A5A",
            )
        if integration.get("status") == "fail":
            return (
                "96-UPDATE BRIDGE PAIR FAILED",
                "new qualification stopped before Stage B",
                "#B91C1C",
            )
        integration_identity = _read_json(
            ROOT
            / "var/artifacts/"
            "pantry_support_mask_paired_integration_v2_identity.json",
            {},
        )
        if integration_identity:
            jobs = integration_identity.get("jobs", {})
            return (
                "96-UPDATE BRIDGE PAIR QUEUED",
                f"jobs {jobs.get('grpo', '?')}/{jobs.get(TREATMENT, '?')}; "
                "warmup 64 will be crossed",
                "#B36B00",
            )
        paired_r1 = _read_json(
            ROOT / "var/artifacts/pantry_support_mask_paired_smoke_v1_r1_audit.json",
            {},
        )
        if paired_r1.get("status") == "fail":
            return (
                "V1-R1 MECHANISM PATH FAILED",
                "raw masks bypassed validator; task-bound bridge qualified next",
                "#B91C1C",
            )
        audit = _read_json(
            ROOT
            / "var/artifacts/"
            "pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit.json",
            {},
        )
        if audit.get("status") == "pass":
            return (
                "32-UPDATE CONTROL SMOKE PASSED",
                "positive reward + multimode support; paired MaxEnt smoke next",
                "#008A5A",
            )
        if audit.get("status") == "fail":
            return ("PLUMBING SMOKE FAILED", "Stage B stopped", "#B91C1C")
        if (
            ROOT
            / "var/artifacts/"
            "pantry_support_mask_drgrpo_smoke_v1_placement_amendment.json"
        ).is_file():
            return (
                "TRAINING SMOKE RUNNING",
                "six-bit Dr.GRPO plumbing on A5000, job 30187473",
                "#B36B00",
            )
        return (
            "TRAINING SMOKE QUEUED",
            "six-bit 0.5B gate passed; Dr.GRPO plumbing job 30187473",
            "#B36B00",
        )
    if domain == "point_maze":
        paired_audit = _read_json(
            ROOT
            / "var/artifacts/point_maze_interactive_paired_smoke_v1_audit.json",
            {},
        )
        if paired_audit.get("status") == "pass":
            return (
                "PAIRED ONLINE SMOKE PASSED",
                "interactive MaxEnt/control mechanism qualified; Stage B next",
                "#008A5A",
            )
        if paired_audit.get("status") == "fail":
            return ("PAIRED ONLINE SMOKE FAILED", "PointMaze Stage B stopped", "#B91C1C")
        paired_identity = _read_json(
            ROOT
            / "var/artifacts/point_maze_interactive_paired_smoke_v1_identity.json",
            {},
        )
        if paired_identity:
            jobs = paired_identity.get("jobs", {})
            return (
                "PAIRED ONLINE SMOKE LAUNCHED",
                f"development-only interactive jobs "
                f"{jobs.get('grpo', '?')}/{jobs.get(TREATMENT, '?')}",
                "#B36B00",
            )
        v3_gate = _read_json(
            ROOT
            / "var/artifacts/"
            "point_maze_interactive_05b_viability_warmstart_v3.json",
            {},
        )
        if v3_gate.get("status") == "pass":
            return (
                "MARKOV WARM START V3 PASSED",
                "paired online smoke is next",
                "#008A5A",
            )
        if v3_gate.get("status") == "fail":
            return (
                "MARKOV WARM START V3 FAILED",
                "public velocity exposed; recovery data is next",
                "#B91C1C",
            )
        v3_identity = _read_json(
            ROOT / "var/artifacts/point_maze_interactive_warmstart_v3_identity.json",
            {},
        )
        if v3_identity:
            return (
                "MARKOV WARM START V3 RUNNING",
                f"public-velocity SFT + unchanged dev gate, job "
                f"{v3_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v2_gate = _read_json(
            ROOT
            / "var/artifacts/"
            "point_maze_interactive_05b_viability_warmstart_v2.json",
            {},
        )
        if v2_gate.get("status") == "pass":
            return (
                "COMPACT WARM START V2 PASSED",
                "paired online smoke is next",
                "#008A5A",
            )
        if v2_gate.get("status") == "fail":
            return (
                "COMPACT WARM START V2 FAILED",
                "0/256 routes; recovery-policy data is next",
                "#B91C1C",
            )
        v2_identity = _read_json(
            ROOT / "var/artifacts/point_maze_interactive_warmstart_v2_identity.json",
            {},
        )
        if v2_identity:
            return (
                "COMPACT WARM START V2 RUNNING",
                f"train-only compact SFT + dev gate, job "
                f"{v2_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        gate = _read_json(
            ROOT
            / "var/artifacts/"
            "point_maze_interactive_05b_viability_warmstart_v1.json",
            {},
        )
        if gate.get("status") == "pass":
            return (
                "SHARED WARM START PASSED",
                "paired online smoke is next",
                "#008A5A",
            )
        if gate.get("status") == "fail":
            return ("WARM-START GATE FAILED", "Stage B stopped", "#B91C1C")
        return (
            "SHARED WARM START QUEUED",
            "train-only SFT + frozen development gate, job 30185642",
            "#B36B00",
            )
    if domain == "ant_maze":
        v13_viability = _read_json(
            ROOT / "var/artifacts/ant_maze_interactive_05b_viability_v13.json", {}
        )
        if v13_viability.get("status") == "pass":
            return (
                "CONSTRAINED POLICY V13 PASSED",
                "closed-loop compass interface; paired online smoke is next",
                "#008A5A",
            )
        if v13_viability.get("status") == "fail":
            summary = v13_viability.get("summary", {})
            return (
                "CONSTRAINED POLICY V13 FAILED",
                f"{summary.get('verified_completions', 0)}/256 routes; Stage B stopped",
                "#B91C1C",
            )
        v13_identity = _read_json(
            ROOT / "var/artifacts/ant_maze_interactive_warmstart_v13_identity.json",
            {},
        )
        if v13_identity:
            return (
                "CONSTRAINED POLICY V13 QUEUED",
                f"32-example SFT + 256-route gate, job "
                f"{v13_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v12_viability = _read_json(
            ROOT / "var/artifacts/ant_maze_05b_viability_v12.json", {}
        )
        if v12_viability.get("status") == "pass":
            return (
                "V12 FREE-FORM 0.5B GATE PASSED",
                "paired online smoke is next",
                "#008A5A",
            )
        if v12_viability.get("status") == "fail":
            summary = v12_viability.get("summary", {})
            return (
                "V12 FREE-FORM 0.5B GATE FAILED",
                f"{summary.get('verified_completions', 0)}/256 routes; "
                "constrained-policy interface required",
                "#B91C1C",
            )
        v12_viability_identity = _read_json(
            ROOT / "var/artifacts/ant_maze_05b_viability_v12_identity.json", {}
        )
        if v12_viability_identity:
            return (
                "V12 FREE-FORM 0.5B GATE RUNNING",
                f"256 development completions, job "
                f"{v12_viability_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v12_cross_node = _read_json(
            ROOT / "var/artifacts/ant_maze_v12_cross_node_v1_r1.json", {}
        )
        if v12_cross_node.get("status") == "pass":
            return (
                "V12 CROSS-NODE GATE PASSED",
                "216/216 exact simulator routes; 0.5B viability next",
                "#008A5A",
            )
        v12_route = _read_json(
            ROOT / "var/artifacts/ant_maze_modebench_v12_admission_audit.json",
            {},
        )
        if v12_route.get("status") == "pass":
            return (
                "V12 ROUTE GATE PASSED",
                "anchored unchanged slate; cross-node replay is next",
                "#008A5A",
            )
        if v12_route.get("status") == "fail":
            return (
                "V12 ROUTE GATE FAILED",
                "first frozen anchored-slate outcome; no substitution",
                "#B91C1C",
            )
        v12_r2_identity = _read_json(
            ROOT
            / "var/artifacts/"
            "ant_maze_modebench_v12_r2_generation_identity.json",
            {},
        )
        if v12_r2_identity:
            return (
                "V12 ROUTE GATE QUEUED",
                f"sealed anchored slate, job "
                f"{v12_r2_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v11_result = _read_json(
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v11.evaluation.json",
            {},
        )
        if v11_result.get("status") == "pass":
            v11_route = _read_json(
                ROOT / "var/artifacts/ant_maze_modebench_v11_admission_audit.json",
                {},
            )
            if v11_route.get("status") == "pass":
                return (
                    "V11 ROUTE GATE PASSED",
                    "three-node exact-slate replay is next",
                    "#008A5A",
                )
            if v11_route.get("status") == "fail":
                return ("V11 ROUTE GATE FAILED", "Stage B stopped", "#B91C1C")
            v11_route_identity = _read_json(
                ROOT / "var/artifacts/ant_maze_modebench_v11_generation_identity.json",
                {},
            )
            if v11_route_identity:
                return (
                    "V11 ROUTE GATE RUNNING",
                    f"unchanged unexecuted 11x11 slate, job "
                    f"{v11_route_identity.get('job_id', 'unknown')}",
                    "#B36B00",
                )
            return (
                "CONTROLLER V11 PASSED",
                "frozen controller-bound route gate is next",
                "#008A5A",
            )
        if v11_result:
            return ("CONTROLLER V11 FAILED", "Stage B stopped", "#B91C1C")
        v11_identity = _read_json(
            ROOT / "var/artifacts/ant_waypoint_controller_v11_identity.json",
            {},
        )
        if v11_identity:
            return (
                "CONTROLLER V11 TRAINING",
                f"SE/NE remediation on fresh 12x12 gate, job "
                f"{v11_identity.get('job_id', 30198291)}",
                "#B36B00",
            )
        v10_result = _read_json(
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v10.evaluation.json",
            {},
        )
        if v10_result.get("status") == "pass":
            v10_viability = _read_json(
                ROOT / "var/artifacts/ant_maze_05b_viability_v10.json",
                {},
            )
            if v10_viability.get("status") == "pass":
                return (
                    "0.5B VIABILITY V10 PASSED",
                    "paired online-training smoke is next",
                    "#008A5A",
                )
            if v10_viability.get("status") == "fail":
                return ("0.5B VIABILITY V10 FAILED", "Stage B stopped", "#B91C1C")
            v10_viability_identity = _read_json(
                ROOT / "var/artifacts/ant_maze_05b_viability_v10_identity.json",
                {},
            )
            if v10_viability_identity:
                return (
                    "0.5B VIABILITY V10 QUEUED",
                    f"development-only 0.5B sample, job "
                    f"{v10_viability_identity.get('job_id', 'unknown')}",
                    "#B36B00",
                )
            v10_cross_node = _read_json(
                ROOT / "var/artifacts/ant_maze_v10_cross_node_v1.json",
                {},
            )
            if v10_cross_node.get("status") == "pass":
                return (
                    "CROSS-NODE V10 PASSED",
                    "frozen 0.5B viability gate is next",
                    "#008A5A",
                )
            if v10_cross_node.get("status") == "fail":
                return ("CROSS-NODE V10 FAILED", "Stage B stopped", "#B91C1C")
            v10_cross_node_identity = _read_json(
                ROOT / "var/artifacts/ant_maze_v10_cross_node_v1_identity.json",
                {},
            )
            if v10_cross_node_identity:
                return (
                    "CROSS-NODE V10 RUNNING",
                    f"216 exact-slate executions, job "
                    f"{v10_cross_node_identity.get('job_id', 'unknown')}",
                    "#B36B00",
                )
            v10_route = _read_json(
                ROOT / "var/artifacts/ant_maze_modebench_v10_admission_audit.json",
                {},
            )
            if v10_route.get("status") == "pass":
                return (
                    "V10 ROUTE GATE PASSED",
                    "three-node exact-slate replay is next",
                    "#008A5A",
                )
            if v10_route.get("status") == "fail":
                return ("V10 ROUTE GATE FAILED", "Stage B stopped", "#B91C1C")
            v10_route_identity = _read_json(
                ROOT / "var/artifacts/ant_maze_modebench_v10_generation_identity.json",
                {},
            )
            if v10_route_identity:
                return (
                    "V10 ROUTE GATE RUNNING",
                    f"12 new 11x11 maps, job "
                    f"{v10_route_identity.get('job_id', 'unknown')}",
                    "#B36B00",
                )
            return (
                "CONTROLLER V10 PASSED",
                "fresh-map v10 route gate is next",
                "#008A5A",
            )
        if v10_result:
            return ("CONTROLLER V10 FAILED", "Stage B stopped", "#B91C1C")
        v10_identity = _read_json(
            ROOT / "var/artifacts/ant_waypoint_controller_v10_identity.json",
            {},
        )
        if v10_identity:
            return (
                "CONTROLLER V10 TRAINING",
                f"northeast remediation, job "
                f"{v10_identity.get('job_id', 30193111)}",
                "#B36B00",
            )
        viability = _read_json(
            ROOT / "var/artifacts/ant_maze_05b_viability_v9.json",
            {},
        )
        if viability.get("status") == "pass":
            return (
                "0.5B VIABILITY PASSED",
                "paired online-training smoke is next",
                "#008A5A",
            )
        if viability.get("status") == "fail":
            return ("0.5B VIABILITY FAILED", "Stage B stopped", "#B91C1C")
        viability_identity = _read_json(
            ROOT / "var/artifacts/ant_maze_05b_viability_v9_identity.json",
            {},
        )
        if viability_identity:
            return (
                "0.5B VIABILITY QUEUED",
                f"development-only 0.5B sample, job "
                f"{viability_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        cross_node = _read_json(
            ROOT / "var/artifacts/ant_maze_v9_cross_node_v1.json",
            {},
        )
        if cross_node.get("status") == "pass":
            return (
                "CROSS-NODE V9 PASSED",
                "frozen 0.5B viability gate is next",
                "#008A5A",
            )
        if cross_node.get("status") == "fail":
            return ("CROSS-NODE V9 FAILED", "Stage B stopped", "#B91C1C")
        cross_node_identity = _read_json(
            ROOT / "var/artifacts/ant_maze_v9_cross_node_v1_identity.json",
            {},
        )
        if cross_node_identity:
            return (
                "CROSS-NODE V9 RUNNING",
                f"216 exact-slate executions, job "
                f"{cross_node_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v9_route = _read_json(
            ROOT / "var/artifacts/ant_maze_modebench_v9_admission_audit.json",
            {},
        )
        if v9_route.get("status") == "pass":
            return (
                "V9 ROUTE GATE PASSED",
                "three-node exact-slate replay is next",
                "#008A5A",
            )
        if v9_route.get("status") == "fail":
            return ("V9 ROUTE GATE FAILED", "Stage B stopped", "#B91C1C")
        v9_route_identity = _read_json(
            ROOT / "var/artifacts/ant_maze_modebench_v9_generation_identity.json",
            {},
        )
        if v9_route_identity:
            return (
                "V9 ROUTE GATE RUNNING",
                f"12 new 9x9 maps, job "
                f"{v9_route_identity.get('job_id', 'unknown')}",
                "#B36B00",
            )
        v9_result = _read_json(
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v9.evaluation.json",
            {},
        )
        if v9_result.get("status") == "pass":
            return (
                "CONTROLLER V9 PASSED",
                "fresh-map route gate is next",
                "#008A5A",
            )
        if v9_result:
            return ("CONTROLLER V9 FAILED", "Stage B stopped", "#B91C1C")
        v9_identity = _read_json(
            ROOT / "var/artifacts/ant_waypoint_controller_v9_identity.json",
            {},
        )
        if v9_identity:
            return (
                "CONTROLLER V9 TRAINING",
                f"train-map local-edge continuation, job "
                f"{v9_identity.get('job_id', 30192731)}",
                "#B36B00",
            )
        v8_route = _read_json(
            ROOT / "var/artifacts/ant_maze_modebench_v8_admission_audit.json",
            {},
        )
        if v8_route.get("status") == "pass":
            return (
                "V8 ROUTE GATE PASSED",
                "cross-node determinism gate is next",
                "#008A5A",
            )
        if (
            ROOT / "var/artifacts/ant_maze_modebench_v8_generation_identity.json"
        ).is_file():
            return (
                "V8 ROUTE GATE FAILED",
                "first frozen fresh-map upper fixture; no substitution",
                "#B91C1C",
            )
        v8_result = _read_json(
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v8.evaluation.json",
            {},
        )
        if v8_result.get("status") == "pass":
            return (
                "CONTROLLER V8 PASSED",
                "fresh-map route gate is next",
                "#008A5A",
            )
        if v8_result:
            return ("CONTROLLER V8 FAILED", "Stage B stopped", "#B91C1C")
        if (ROOT / "var/artifacts/ant_waypoint_controller_v8_identity.json").is_file():
            return (
                "CONTROLLER V8 TRAINING",
                "fresh-seed conservative continuation, job 30187810",
                "#B36B00",
            )
        v7_result = _read_json(
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v7.evaluation.json",
            {},
        )
        if v7_result.get("status") == "pass":
            return (
                "CONTROLLER V7 PASSED",
                "fresh-map route gate is next",
                "#008A5A",
            )
        if v7_result:
            return ("CONTROLLER V7 FAILED", "v8 continuation not launched", "#B91C1C")
        return (
            "CONTROLLER V7 TRAINING",
            "maze-blind 5M-transition open-plane gate, job 30187375",
            "#B36B00",
        )
    return ("AWAITING STAGE B", "0/10 main jobs launched", "#777777")


def _audit_row_status(
    audit: Mapping[str, Any],
    domain: str,
) -> tuple[str, str]:
    if domain == "pantry_plan":
        pantry_audit = _read_json(
            ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json",
            {},
        )
        cells = pantry_audit.get("cells", {})
        materialized = len(cells) if isinstance(cells, Mapping) else 0
        terminal = materialized if pantry_audit.get("status") == "pass" else 0
        return f"{materialized}/10 materialized", f"{terminal}/10 terminal"
    if domain in {"point_maze", "point_maze_geometry_shift", "ant_maze"}:
        stem = (
            "point_maze_stage_b_05b_12pass"
            if domain == "point_maze"
            else (
                "point_maze_geometry_shift_stage_b_05b_12pass"
                if domain == "point_maze_geometry_shift"
                else "ant_maze_stage_b_05b_12pass"
            )
        )
        materialized = 0
        for arm in (CONTROL, TREATMENT):
            for seed in SEEDS:
                base = ROOT / f"var/artifacts/{stem}_{arm}_s{seed}"
                materialized += Path(str(base) + ".metrics.jsonl").is_file()
        stage_b_audit = _read_json(
            ROOT / f"var/artifacts/{stem}_audit.json", {}
        )
        audited_terminal = 10 if stage_b_audit.get("status") == "pass" else 0
        return (
            f"{materialized}/10 metric-bearing",
            f"{audited_terminal}/10 audited terminal",
        )
    runs = audit.get("domains", {}).get(domain, {}).get("runs", [])
    if not isinstance(runs, list):
        return "0/10 materialized", "0/10 terminal"
    materialized = sum(bool(row.get("run_dir")) for row in runs if isinstance(row, Mapping))
    terminal = sum(bool(row.get("terminal")) for row in runs if isinstance(row, Mapping))
    return f"{materialized}/10 materialized", f"{terminal}/10 terminal"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def render(output: Path, sidecar: Path) -> None:
    audit = _read_json(AUDIT, {})
    summary = audit.get("summary", {}) if isinstance(audit, Mapping) else {}
    pantry_audit = _read_json(
        ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json",
        {},
    )
    pantry_terminal = (
        len(pantry_audit.get("cells", {}))
        if pantry_audit.get("status") == "pass"
        and isinstance(pantry_audit.get("cells"), Mapping)
        else 0
    )
    point_audit = _read_json(
        ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit.json", {}
    )
    ant_audit = _read_json(
        ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_audit.json", {}
    )
    point_terminal = 10 if point_audit.get("status") == "pass" else 0
    ant_terminal = 10 if ant_audit.get("status") == "pass" else 0
    geometry_shift_audit = _read_json(
        ROOT
        / "var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_audit.json",
        {},
    )
    geometry_shift_terminal = 10 if geometry_shift_audit.get("status") == "pass" else 0
    total_terminal = (
        int(summary.get("terminal_runs", 0))
        + pantry_terminal
        + point_terminal
        + ant_terminal
        + geometry_shift_terminal
    )
    domain_points = {
        key: (
            _load_maze_stage_b_points(key)
            if key in {"point_maze", "point_maze_geometry_shift", "ant_maze"}
            else _load_points(prefix, steps)
        )
        for _label, key, prefix, steps in DOMAIN_SPECS
    }
    figure = plt.figure(figsize=(9.2, 12.0))
    outer_grid = figure.add_gridspec(
        4,
        2,
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.805,
        hspace=0.50,
        wspace=0.18,
    )
    domain_axes: dict[str, list[plt.Axes]] = {}
    for domain_index, (label, domain, _prefix, _steps) in enumerate(DOMAIN_SPECS):
        outer_cell = outer_grid[domain_index // 2, domain_index % 2]
        inner_grid = outer_cell.subgridspec(2, 2, hspace=0.44, wspace=0.34)
        panel_axes = [
            figure.add_subplot(inner_grid[panel_index // 2, panel_index % 2])
            for panel_index in range(len(PANELS))
        ]
        for axis in panel_axes:
            axis.set_box_aspect(0.80)
        domain_axes[domain] = panel_axes
        cell_box = outer_cell.get_position(figure)
        figure.text(
            cell_box.x0,
            cell_box.y1 + 0.011,
            label,
            ha="left",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#222222",
            linespacing=0.95,
        )

    for _domain_index, (label, domain, prefix, _steps) in enumerate(DOMAIN_SPECS):
        points = domain_points[domain]
        panel_axes = domain_axes[domain]
        if prefix is None and not points:
            headline, detail, color = _status_for_pending_row(domain)
            for panel_index, (_metric, title, _integer) in enumerate(PANELS):
                axis = panel_axes[panel_index]
                axis.set_title(
                    title,
                    fontsize=7.0,
                    pad=4.5 if _metric == "distinct8" else 2.5,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_color("#dddddd")
                axis.set_facecolor("#fafafa")
                axis.text(
                    0.5,
                    0.58,
                    headline,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    fontweight="bold",
                    color=color,
                    wrap=True,
                )
                axis.text(
                    0.5,
                    0.34,
                    detail,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=4.9,
                    color="#666666",
                    wrap=True,
                )
            continue

        materialized, terminal = _audit_row_status(audit, domain)
        for panel_index, (metric, title, integer) in enumerate(PANELS):
            axis = panel_axes[panel_index]
            plotted, max_seed_count = _plot_metric(axis, points, metric)
            if not plotted:
                pending = (
                    "awaiting evaluation"
                    if metric in QUALITY_METRICS
                    else "awaiting mechanism telemetry"
                )
                axis.text(
                    0.5,
                    0.5,
                    pending,
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#777777",
                    fontsize=5.8,
                )
            axis.set_title(
                title,
                fontsize=7.0,
                pad=4.5 if metric == "distinct8" else 2.5,
            )
            _style_axis(axis, metric, plotted, integer)
            if panel_index == 0:
                axis.text(
                    0.985,
                    0.05,
                    f"{materialized}; {terminal}",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=4.8,
                    color="#666666",
                )
            if panel_index < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel("training passes", fontsize=6.3, labelpad=1.5)
            if max_seed_count:
                axis.text(
                    0.015,
                    0.95,
                    f"latest mean n≤{max_seed_count}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=4.5,
                    color="#777777",
                )

    handles = [
        Line2D(
            [], [], color=BLUE, lw=2.8, marker="o",
            label="compute-matched Dr.GRPO",
        ),
        Line2D(
            [], [], color=ORANGE, lw=2.8, marker="D",
            markerfacecolor="white", markeredgewidth=1.2,
            label="online verified MaxEnt (E58 recipe)",
        ),
        Line2D(
            [], [], color="#333333", lw=2.4,
            label="available-seed mean; band = available-seed range",
        ),
    ]
    handles.extend(
        Line2D(
            [], [], color="#666666", lw=1.0,
            ls=SEED_STYLES[seed], label=f"seed {seed}",
        )
        for seed in SEEDS
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.865),
    )
    generated = datetime.now(timezone.utc)
    submitted = 40
    for identity_path in (
        ROOT / "var/artifacts/pantry_stage_b_05b_12pass_identity.json",
        ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_identity.json",
        ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_identity.json",
        ROOT / "var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_identity.json",
    ):
        identity_payload = _read_json(identity_path, {})
        jobs = identity_payload.get("jobs", {})
        if isinstance(jobs, Mapping):
            submitted += len(jobs)
    figure.suptitle(
        "Clean eight-environment 0.5B replacement campaign — historical E68 format\n"
        "Four frozen outcome/discovery panels; 5 seeds × 12 passes; "
        "PointMaze geometry shift is a labeled replacement configuration\n"
        f"Stage A: {summary.get('terminal_runs', 0)}/"
        f"{summary.get('expected_runs', 40)} terminal, "
        f"{summary.get('metric_runs', 0)} metric-bearing, "
        f"{summary.get('materialized_runs', 0)} materialized; "
        f"{total_terminal}/80 audited terminal overall\n"
        f"{submitted}/80 paper jobs submitted; remaining rows stay fail-closed; "
        "no historical values or carry-forward "
        f"({generated.strftime('%Y-%m-%d %H:%M UTC')})",
        fontsize=9.0,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Thin lines are exact E70 seed trajectories. Thick lines average only "
        "the seeds actually present at each x-coordinate, with n annotated;\n"
        "this is a live progress view, not a terminal five-seed estimate. "
        "Quality uses the frozen multi-answer evaluation split;\n"
        "cumulative discoveries use the registered online verified-support "
        "telemetry. PantryPlan, PointMaze, AntMaze, and PointMaze geometry "
        "shift show\n"
        "their newest identity-bound gate or Stage-B state; maze terminal "
        "counts remain zero until executable replay audit passes.\n"
        "ConstructiveCode remains a reported negative qualification (0/192 "
        "accepted at both 0.5B post-SFT and 1.5B) and is not counted in this roster.",
        ha="center",
        fontsize=6.2,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    figure.savefig(
        temporary_output,
        dpi=180,
        bbox_inches="tight",
        format=output.suffix.lstrip("."),
    )
    temporary_output.replace(output)
    if output.suffix.lower() == ".png":
        pdf_output = output.with_suffix(".pdf")
        temporary_pdf = pdf_output.with_name(f".{pdf_output.name}.tmp")
        figure.savefig(temporary_pdf, bbox_inches="tight", format="pdf")
        temporary_pdf.replace(pdf_output)
    plt.close(figure)

    inputs = {
        "audit": AUDIT,
        "constructive_v6_identity": (
            ROOT / "var/artifacts/constructive_code_v6_gate_identity.json"
        ),
        "constructive_v6_audit": (
            ROOT / "var/artifacts/constructive_code_v6_gate_audit.json"
        ),
        "constructive_v6_viability_identity": (
            ROOT / "var/artifacts/constructive_code_v6_coder_05b_viability_identity.json"
        ),
        "constructive_v6_viability": (
            ROOT / "var/artifacts/constructive_code_v6_coder_05b_viability.json"
        ),
        "constructive_v6_paired_identity": (
            ROOT / "var/artifacts/constructive_code_v6_paired_smoke_identity.json"
        ),
        "constructive_v6_paired_audit": (
            ROOT / "var/artifacts/constructive_code_v6_paired_smoke_audit.json"
        ),
        "constructive_v6_stage_b_identity": (
            ROOT / "var/artifacts/constructive_code_v6_stage_b_identity.json"
        ),
        "constructive_v6_stage_b_audit": (
            ROOT / "var/artifacts/constructive_code_v6_stage_b_audit.json"
        ),
        "constructive_v7_sft_gate_identity": (
            ROOT / "var/artifacts/constructive_code_v7_sft_gate_identity.json"
        ),
        "constructive_v7_sft_receipt": (
            ROOT / "var/artifacts/constructive_code_v7_sft.json"
        ),
        "constructive_v7_post_sft_viability": (
            ROOT / "var/artifacts/constructive_code_v7_post_sft_viability.json"
        ),
        "constructive_v7_repair_protocol": (
            ROOT
            / "paper/preregistration/"
            "constructive_code_v7_token_boundary_repair_r1_20260730.md"
        ),
        "constructive_v8_15b_identity": (
            ROOT / "var/artifacts/constructive_code_v8_coder_15b_viability_identity.json"
        ),
        "constructive_v8_15b_viability": (
            ROOT / "var/artifacts/constructive_code_v8_coder_15b_viability.json"
        ),
        "constructive_v8_15b_protocol": (
            ROOT
            / "paper/preregistration/"
            "constructive_code_v8_coder_15b_capacity_retry_20260730.md"
        ),
        "constructive_v5_identity": (
            ROOT / "var/artifacts/constructive_code_v5_gate_identity.json"
        ),
        "constructive_v5_audit": (
            ROOT / "var/artifacts/constructive_code_v5_gate_audit.json"
        ),
        "constructive_v5_viability_identity": (
            ROOT / "var/artifacts/constructive_code_v5_coder_05b_viability_identity.json"
        ),
        "constructive_v5_viability": (
            ROOT / "var/artifacts/constructive_code_v5_coder_05b_viability.json"
        ),
        "constructive_v5_coder_dependency_identity": (
            ROOT / "var/artifacts/constructive_code_v5_coder_dependency_identity.json"
        ),
        "constructive_v4_identity": (
            ROOT / "var/artifacts/constructive_code_v4_gate_identity.json"
        ),
        "constructive_v4_audit": (
            ROOT / "var/artifacts/constructive_code_v4_gate_audit.json"
        ),
        "constructive_v4_failure_log": (
            ROOT
            / "var/artifacts/logs/constructive-code-v4-gate-30201021.err"
        ),
        "constructive_v4_viability_identity": (
            ROOT / "var/artifacts/constructive_code_v4_coder_05b_viability_identity.json"
        ),
        "constructive_v4_viability": (
            ROOT / "var/artifacts/constructive_code_v4_coder_05b_viability.json"
        ),
        "pantry_bridge_v3_identity": (
            ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_identity.json"
        ),
        "pantry_bridge_v3_audit": (
            ROOT / "var/artifacts/pantry_support_mask_paired_integration_v3_audit.json"
        ),
        "pantry_stage_b_identity": (
            ROOT / "var/artifacts/pantry_stage_b_05b_12pass_identity.json"
        ),
        "pantry_stage_b_submission": (
            ROOT / "var/artifacts/pantry_stage_b_05b_12pass_submission.json"
        ),
        "pantry_stage_b_audit": (
            ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json"
        ),
        "point_paired_smoke_identity": (
            ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v1_identity.json"
        ),
        "point_paired_smoke_submission": (
            ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v1_submission.json"
        ),
        "point_paired_smoke_audit": (
            ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v1_audit.json"
        ),
        "point_paired_smoke_v3_identity": (
            ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v3_identity.json"
        ),
        "point_paired_smoke_v3_audit": (
            ROOT / "var/artifacts/point_maze_interactive_paired_smoke_v3_audit.json"
        ),
        "point_stage_b_identity": (
            ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_identity.json"
        ),
        "point_stage_b_audit": (
            ROOT / "var/artifacts/point_maze_stage_b_05b_12pass_audit.json"
        ),
        "point_maze_v3_viability": (
            ROOT / "var/artifacts/point_maze_interactive_05b_viability_warmstart_v3.json"
        ),
        "point_geometry_shift_protocol": (
            ROOT
            / "paper/preregistration/"
            "point_maze_geometry_shift_replacement_v1_20260730.md"
        ),
        "point_geometry_shift_admission": (
            ROOT / "var/artifacts/point_maze_geometry_shift_v1_admission_audit.json"
        ),
        "point_geometry_shift_viability_identity": (
            ROOT / "var/artifacts/point_maze_geometry_shift_viability_v1_identity.json"
        ),
        "point_geometry_shift_viability": (
            ROOT / "var/artifacts/point_maze_geometry_shift_05b_viability_v1.json"
        ),
        "point_geometry_shift_paired_identity": (
            ROOT / "var/artifacts/point_maze_geometry_shift_paired_smoke_v1_identity.json"
        ),
        "point_geometry_shift_paired_audit": (
            ROOT / "var/artifacts/point_maze_geometry_shift_paired_smoke_v1_audit.json"
        ),
        "point_geometry_shift_stage_b_identity": (
            ROOT
            / "var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_identity.json"
        ),
        "point_geometry_shift_stage_b_audit": (
            ROOT
            / "var/artifacts/point_maze_geometry_shift_stage_b_05b_12pass_audit.json"
        ),
        "ant_v12_cross_node_audit": (
            ROOT / "var/artifacts/ant_maze_v12_cross_node_v1_r1.json"
        ),
        "ant_v12_viability_identity": (
            ROOT / "var/artifacts/ant_maze_05b_viability_v12_identity.json"
        ),
        "ant_v12_viability": (
            ROOT / "var/artifacts/ant_maze_05b_viability_v12.json"
        ),
        "ant_v13_identity": (
            ROOT / "var/artifacts/ant_maze_interactive_warmstart_v13_identity.json"
        ),
        "ant_v13_viability": (
            ROOT / "var/artifacts/ant_maze_interactive_05b_viability_v13.json"
        ),
        "ant_paired_smoke_v13r2_identity": (
            ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13r2_identity.json"
        ),
        "ant_paired_smoke_v13r2_audit": (
            ROOT / "var/artifacts/ant_maze_interactive_paired_smoke_v13r2_audit.json"
        ),
        "ant_stage_b_identity": (
            ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_identity.json"
        ),
        "ant_stage_b_audit": (
            ROOT / "var/artifacts/ant_maze_stage_b_05b_12pass_audit.json"
        ),
        "identity": IDENTITY,
        "protocol": PROTOCOL,
        **{
            f"curve_{key}": _curve_path(prefix)
            for _label, key, prefix, _steps in DOMAIN_SPECS
            if prefix is not None
        },
        "constructive_gate": (
            ROOT / "var/artifacts/constructive_code_v3_gate_audit.json"
        ),
        "constructive_v3_identity": (
            ROOT / "var/artifacts/constructive_code_v3_gate_identity.json"
        ),
        "constructive_v3_failure_log": (
            ROOT
            / "var/artifacts/logs/"
            "constructive-code-v3-gate-30187935.err"
        ),
        "pantry_smoke": (
            ROOT
            / "var/artifacts/"
            "pantry_support_mask_drgrpo_smoke_v1_r2_r1_audit.json"
        ),
        "pantry_smoke_placement": (
            ROOT
            / "var/artifacts/"
            "pantry_support_mask_drgrpo_smoke_v1_placement_amendment.json"
        ),
        "point_warmstart_gate": (
            ROOT
            / "var/artifacts/"
            "point_maze_interactive_05b_viability_warmstart_v1.json"
        ),
        "point_warmstart_v2_identity": (
            ROOT / "var/artifacts/point_maze_interactive_warmstart_v2_identity.json"
        ),
        "point_warmstart_v2_gate": (
            ROOT
            / "var/artifacts/"
            "point_maze_interactive_05b_viability_warmstart_v2.json"
        ),
        "point_warmstart_v3_identity": (
            ROOT / "var/artifacts/point_maze_interactive_warmstart_v3_identity.json"
        ),
        "point_warmstart_v3_gate": (
            ROOT
            / "var/artifacts/"
            "point_maze_interactive_05b_viability_warmstart_v3.json"
        ),
        "ant_controller_v7": (
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v7.evaluation.json"
        ),
        "ant_controller_v8_identity": (
            ROOT / "var/artifacts/ant_waypoint_controller_v8_identity.json"
        ),
        "ant_controller_v8": (
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v8.evaluation.json"
        ),
        "ant_controller_v9_identity": (
            ROOT / "var/artifacts/ant_waypoint_controller_v9_identity.json"
        ),
        "ant_controller_v9": (
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v9.evaluation.json"
        ),
        "ant_controller_v10_identity": (
            ROOT / "var/artifacts/ant_waypoint_controller_v10_identity.json"
        ),
        "ant_controller_v10": (
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v10.evaluation.json"
        ),
        "ant_controller_v11_identity": (
            ROOT / "var/artifacts/ant_waypoint_controller_v11_identity.json"
        ),
        "ant_controller_v11": (
            ROOT
            / "var/maze_runtime/controllers/"
            "ant_waypoint_v11.evaluation.json"
        ),
        "ant_v11_route_identity": (
            ROOT / "var/artifacts/ant_maze_modebench_v11_generation_identity.json"
        ),
        "ant_v11_route_audit": (
            ROOT / "var/artifacts/ant_maze_modebench_v11_admission_audit.json"
        ),
        "ant_v12_r2_route_identity": (
            ROOT
            / "var/artifacts/"
            "ant_maze_modebench_v12_r2_generation_identity.json"
        ),
        "ant_v12_route_audit": (
            ROOT / "var/artifacts/ant_maze_modebench_v12_admission_audit.json"
        ),
        "ant_v10_route_identity": (
            ROOT / "var/artifacts/ant_maze_modebench_v10_generation_identity.json"
        ),
        "ant_v10_route_audit": (
            ROOT / "var/artifacts/ant_maze_modebench_v10_admission_audit.json"
        ),
        "ant_v10_cross_node_identity": (
            ROOT / "var/artifacts/ant_maze_v10_cross_node_v1_identity.json"
        ),
        "ant_v10_cross_node_audit": (
            ROOT / "var/artifacts/ant_maze_v10_cross_node_v1.json"
        ),
        "ant_v10_viability_identity": (
            ROOT / "var/artifacts/ant_maze_05b_viability_v10_identity.json"
        ),
        "ant_v10_viability": (
            ROOT / "var/artifacts/ant_maze_05b_viability_v10.json"
        ),
        "ant_v9_route_identity": (
            ROOT / "var/artifacts/ant_maze_modebench_v9_generation_identity.json"
        ),
        "ant_v9_route_audit": (
            ROOT / "var/artifacts/ant_maze_modebench_v9_admission_audit.json"
        ),
        "ant_v9_cross_node_identity": (
            ROOT / "var/artifacts/ant_maze_v9_cross_node_v1_identity.json"
        ),
        "ant_v9_cross_node_audit": (
            ROOT / "var/artifacts/ant_maze_v9_cross_node_v1.json"
        ),
        "ant_v9_viability_identity": (
            ROOT / "var/artifacts/ant_maze_05b_viability_v9_identity.json"
        ),
        "ant_v9_viability": (
            ROOT / "var/artifacts/ant_maze_05b_viability_v9.json"
        ),
        "ant_v8_route_identity": (
            ROOT / "var/artifacts/ant_maze_modebench_v8_generation_identity.json"
        ),
        "ant_v8_route_audit": (
            ROOT / "var/artifacts/ant_maze_modebench_v8_admission_audit.json"
        ),
    }
    provenance = {
        "schema": "e70-clean-05b-e68-four-panel-live-v1",
        "generated_at": generated.isoformat(),
        # Forty terminal Stage-A cells do not make the requested 80-cell
        # cohort terminal.
        "status": (
            "stage_a_terminal"
            if summary.get("terminal_runs") == 40
            else "live"
        ),
        "format_template": (
            "paper/figures/"
            "e68_e58_vs_grpo_05b_12ep_terminal_provenance_"
            "historical_20260728.png"
        ),
        "historical_values_imported": False,
        "carry_forward": False,
        "arms": [CONTROL, TREATMENT],
        "seeds": list(SEEDS),
        "rows": [key for _label, key, _prefix, _steps in DOMAIN_SPECS],
        "panels": [metric for metric, _title, _integer in PANELS],
        "layout": {
            "domain_card_grid": [4, 2],
            "panels_per_card": [2, 2],
            "figure_inches": [9.2, 12.0],
            "panel_box_aspect": 0.80,
        },
        "stage_a_summary": summary,
        "hashes": {
            key: _sha256(path)
            for key, path in {
                "plot_source": Path(__file__).resolve(),
                **inputs,
                "figure": output,
            }.items()
        },
        "output": str(output.relative_to(ROOT)),
    }
    _atomic_json(sidecar, provenance)
    if sidecar.resolve() != WIDE_SIDECAR_ALIAS.resolve():
        _atomic_json(WIDE_SIDECAR_ALIAS, provenance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.output.resolve(), args.sidecar.resolve())
    print(f"[e70-wide-plot] wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
