#!/usr/bin/env python3
"""Render every current E70 outcome/mechanism metric as readable square pages.

The July 28 historical figure used a thirteen-column overview.  This renderer
preserves the same metric inventory without shrinking the axes: the metrics
are split across four portrait pages, while every page uses the same current
identity-bound cohort rows and repair status cards.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_e70_primary_square_repairs_v6_20260730 as current


primary = current.primary
ROOT = primary.ROOT
FIGURES = ROOT / "paper/figures"
ARTIFACTS = ROOT / "var/artifacts"
REPORT_STEM = "e68_e58_vs_grpo_05b_12ep"
FINAL_PRIMARY_PNG = (
    FIGURES
    / "e68_e58_vs_grpo_05b_12ep_"
    "terminal_provenance_historical_20260730.png"
)
SIDECAR = (
    ARTIFACTS
    / "e68_e58_vs_grpo_05b_12ep_"
    "complete_square_pages_20260730_provenance.json"
)

OPEN_SET_ENTROPY = (
    "semantic_shannon_success_conditioned_signed_open_set_entropy_ema"
)
OPEN_SET_COEFFICIENT = (
    "semantic_shannon_success_conditioned_signed_open_set_next_coefficient"
)
MASS_COEFFICIENT = "canonical_replay_mass_next_alpha"
BALANCE_COEFFICIENT = "canonical_replay_next_alpha"

PAGES = (
    (
        "current_outcomes",
        "Clean 0.5B campaign — selected current cohorts and full status",
        (
            ("greedy", "Neutral\npass@1", False),
            ("mean8", "Neutral mean\ncorrect@8", False),
            ("pass8", "Neutral\npass@8", False),
            ("distinct8", "Mean distinct\ncorrect@8", True),
        ),
        (
            "The complete four-metric outcome surface; no outcome coordinate "
            "from the July 28 format is omitted."
        ),
    ),
    (
        "current_controller_diagnostics",
        "Clean 0.5B campaign — controller diagnostics",
        (
            (OPEN_SET_ENTROPY, "Open-set predictive\nentropy EMA", False),
            (OPEN_SET_COEFFICIENT, "Next semantic\ncoefficient β", False),
            (MASS_COEFFICIENT, "Next verified-mass\ncoefficient μ", False),
            (BALANCE_COEFFICIENT, "Next known-mode\ncoefficient α", False),
        ),
        (
            "Controller telemetry from the identity-bound runs; descriptive "
            "mechanism evidence, not additional efficacy endpoints."
        ),
    ),
    (
        "current_replay_diagnostics",
        "Clean 0.5B campaign — verified-replay diagnostics",
        (
            (
                "canonical_replay_balance_loss",
                "Verified replay\nKL",
                False,
            ),
            (
                "canonical_replay_available_modes",
                "Replayed verified\nmodes",
                True,
            ),
            (
                "online_canonical_new_outcome_row_fraction",
                "New verified\noutcome fraction",
                False,
            ),
        ),
        (
            "Replay-state telemetry from the same current cohort rows; "
            "missing values are never copied or carried forward."
        ),
    ),
    (
        "current_support_diagnostics",
        "Clean 0.5B campaign — verified-support diagnostics",
        (
            (
                "online_canonical_mean_support_per_prompt",
                "Mean verified support\nper prompt",
                False,
            ),
            (
                "online_canonical_tracked_outcomes",
                "Cumulative verified\ndiscoveries",
                True,
            ),
            (
                "online_canonical_support_at_least_two_prompt_fraction",
                "Tracked prompts\nwith ≥2 modes",
                False,
            ),
        ),
        (
            "Support/discovery telemetry from the same current cohort rows; "
            "the ≥2-mode fraction is an added current diagnostic."
        ),
    ),
)

ORIGINAL_SEEDS = (43, 44, 45, 46, 47)
PANTRY_REPAIR_SEEDS = (76411, 76412, 76413, 76414, 76415)
POINT_V6_SEEDS = (76631, 76632, 76633, 76634, 76635)
REPORT_ROWS = (
    (
        "Graph coloring",
        "Official 80-cell cohort",
        "graph_coloring",
        "gce70_clean_stage_a_05b_12pass",
        192,
        ORIGINAL_SEEDS,
    ),
    (
        "Countdown",
        "Official 80-cell cohort",
        "countdown",
        "cde70_clean_stage_a_05b_12pass",
        384,
        ORIGINAL_SEEDS,
    ),
    (
        "Python factors",
        "Official 80-cell cohort",
        "python_factor",
        "pye70_clean_stage_a_05b_12pass",
        384,
        ORIGINAL_SEEDS,
    ),
    (
        "MathIR action menu",
        "Official 80-cell cohort",
        "mathir",
        "mie70_clean_stage_a_05b_12pass",
        384,
        ORIGINAL_SEEDS,
    ),
    (
        "PointMaze geometry shift",
        "Official replacement configuration · 10/10 audited",
        "point_maze_geometry_shift",
        None,
        None,
        ORIGINAL_SEEDS,
    ),
    (
        "PointMaze",
        "Official 80-cell cohort · 10/10 audited",
        "point_maze",
        None,
        None,
        ORIGINAL_SEEDS,
    ),
    (
        "AntMaze",
        "Official 80-cell cohort · 10/10 audited",
        "ant_maze",
        None,
        None,
        ORIGINAL_SEEDS,
    ),
    (
        "PantryPlan",
        "Selected support-retention cohort · 10/10 audited · seeds 76411–76415",
        "pantry_support_retention_repair",
        "pprepair_support_retention_final_v1",
        32,
        PANTRY_REPAIR_SEEDS,
    ),
    (
        "PointMaze balanced v6",
        "Selected nontrivial repair cohort · gate 52/256 · five paired seeds",
        "point_maze_algorithm_repair",
        None,
        None,
        POINT_V6_SEEDS,
    ),
    (
        "AntMaze harder-task repair",
        "Secondary stable-handoff v18 controller gate",
        "ant_maze_harder_repair",
        None,
        None,
        (),
    ),
    (
        "ConstructiveCode",
        "Requested semantic domain · qualification ladder",
        "constructive_code_qualification",
        None,
        None,
        (),
    ),
)

# The square renderer indexes styles directly. Reuse the five registered
# original-cohort styles for the five independent Pantry repair seeds.
for repair_seed, original_seed in zip(
    PANTRY_REPAIR_SEEDS, ORIGINAL_SEEDS
):
    primary.SEED_STYLES[repair_seed] = primary.SEED_STYLES[original_seed]

for repair_seed, original_seed in zip(POINT_V6_SEEDS, ORIGINAL_SEEDS):
    primary.SEED_STYLES[repair_seed] = primary.SEED_STYLES[original_seed]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _current_rows() -> tuple[tuple[Any, ...], ...]:
    stage_a = _read_json(primary.base.AUDIT).get("summary", {})
    expected = int(stage_a.get("expected_runs", 40))
    terminal = int(stage_a.get("terminal_runs", 0))
    active = max(expected - terminal, 0)
    rows = []
    for label, note, domain, prefix, steps, seeds in REPORT_ROWS:
        if domain == "mathir":
            note = (
                "Official 80-cell cohort · audited"
                if active == 0
                else f"Official 80-cell cohort · {active} jobs still active"
            )
        rows.append((label, note, domain, prefix, steps, seeds))
    return tuple(rows)


def _load_curve(
    prefix: str,
    steps_per_pass: int,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    path = ARTIFACTS / f"{prefix}_scaling_curve.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in payload:
        if not isinstance(raw, Mapping):
            continue
        arm = raw.get("arm")
        seed = raw.get("seed")
        step = raw.get("step")
        if (
            arm not in {primary.CONTROL, primary.TREATMENT}
            or seed not in seeds
            or not primary._finite(step)
            or raw.get("split") != "multi_answer"
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
        if primary._finite(raw.get("training_passes")):
            point["passes"] = float(raw["training_passes"])
        for metric, value in raw.items():
            if primary._finite(value):
                point[str(metric)] = float(value)
    return [merged[key] for key in sorted(merged)]


def _load_point_v6_points() -> list[dict[str, Any]]:
    stem = "point_maze_balanced_v6_stage_b_05b_12pass"
    merged: dict[tuple[str, int, float], dict[str, Any]] = {}
    for arm in (primary.CONTROL, primary.TREATMENT):
        for seed in POINT_V6_SEEDS:
            path = ARTIFACTS / f"{stem}_{arm}_s{seed}.metrics.jsonl"
            if not path.is_file():
                continue
            for raw in path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping) or not primary._finite(
                    row.get("training_passes")
                ):
                    continue
                passes = round(float(row["training_passes"]), 9)
                key = (arm, seed, passes)
                point = merged.setdefault(
                    key, {"arm": arm, "seed": seed, "passes": passes}
                )
                for metric, value in row.items():
                    if primary._finite(value):
                        point[str(metric)] = float(value)
                if primary._finite(row.get("replay_active_modes")):
                    point["canonical_replay_available_modes"] = float(
                        row["replay_active_modes"]
                    )
                if primary._finite(row.get("canonical_tracked_outcomes")):
                    point["online_canonical_tracked_outcomes"] = float(
                        row["canonical_tracked_outcomes"]
                    )
                tracked_prompts = row.get("canonical_tracked_prompts")
                tracked_outcomes = row.get("canonical_tracked_outcomes")
                if (
                    primary._finite(tracked_prompts)
                    and float(tracked_prompts) > 0
                    and primary._finite(tracked_outcomes)
                ):
                    point["online_canonical_mean_support_per_prompt"] = (
                        float(tracked_outcomes) / float(tracked_prompts)
                    )
    return [merged[key] for key in sorted(merged)]


def _points(domain: str, prefix: str | None, steps: int | None):
    if domain == "point_maze_algorithm_repair":
        source = _load_point_v6_points()
    elif domain in {"point_maze", "ant_maze"}:
        source = primary.base._load_maze_stage_b_points(domain)
    elif domain == "pantry_plan":
        source = primary.base._load_points(prefix, steps)
    elif domain == "pantry_support_retention_repair":
        source = _load_curve(
            "pprepair_support_retention_final_v1",
            32,
            PANTRY_REPAIR_SEEDS,
        )
    else:
        source = primary._points(domain, prefix, steps)
    rows = [dict(row) for row in source]
    for row in rows:
        if (
            "online_canonical_support_at_least_two_prompt_fraction"
            not in row
            and "canonical_support_at_least_two_prompt_fraction" in row
        ):
            row[
                "online_canonical_support_at_least_two_prompt_fraction"
            ] = row["canonical_support_at_least_two_prompt_fraction"]
    return rows


def _status(domain: str) -> tuple[str, str, str]:
    if domain == "point_maze_algorithm_repair":
        qualification = _read_json(
            ARTIFACTS
            / "point_maze_balanced_short_warmstart_v6_qualification.json"
        )
        audit = _read_json(
            ARTIFACTS
            / "point_maze_balanced_v6_stage_b_05b_12pass_audit.json"
        )
        rate = qualification.get("summary", {}).get("verified_rate")
        rate_text = (
            f"{100.0 * float(rate):.2f}%"
            if isinstance(rate, (int, float))
            else "unknown"
        )
        if audit.get("status") == "pass":
            return (
                "POINTMAZE V6 · 10/10 TERMINAL AUDIT PASS",
                f"nontrivial development gate {rate_text}; paired five-seed "
                "Dr.GRPO/MaxEnt comparison on untouched evaluation rows",
                "#008A5A",
            )
        if qualification.get("status") == "pass":
            return (
                "POINTMAZE V6 GATE PASSED · 10 CELLS LAUNCHED",
                f"nontrivial verified rate {rate_text}; jobs 30206771–30206780 "
                "are managed by terminal replay audit 30206781",
                "#008A5A",
            )
        return (
            "POINTMAZE V6 QUALIFICATION PENDING",
            "the frozen 2–50% nontriviality gate controls all online cells",
            "#B36B00",
        )
    if domain == "constructive_code_qualification":
        capacity = _read_json(
            ARTIFACTS / "constructive_code_v8_coder_15b_viability.json"
        )
        if capacity.get("status") == "fail":
            return (
                "CONSTRUCTIVECODE STOPPED BEFORE FINAL CELLS",
                "0/10 paper jobs: executable gate passed, but the 0.5B, "
                "train-only-SFT, and registered 1.5B capacity gates failed",
                "#B91C1C",
            )
        return (
            "CONSTRUCTIVECODE QUALIFICATION INCOMPLETE",
            "No final paper cell may launch until the frozen capacity and "
            "paired-online gates pass",
            "#B36B00",
        )
    return primary._status(domain)


def _page_paths(slug: str) -> tuple[Path, Path]:
    png = FIGURES / f"{REPORT_STEM}_{slug}_20260730.png"
    return png, png.with_suffix(".pdf")


def _render_page(
    slug: str,
    title: str,
    metrics: tuple[tuple[str, str, bool], ...],
    footer: str,
    *,
    generated: datetime,
) -> dict[str, Any]:
    rows = _current_rows()
    column_count = len(metrics)
    figure_width = 16.0 if column_count == 4 else 12.7
    figure = plt.figure(
        figsize=(figure_width, max(24.0, 2.35 * len(rows) + 4.5)),
        facecolor="#F8FAFC",
    )
    grid = figure.add_gridspec(
        len(rows),
        column_count + 1,
        width_ratios=[1.55] + [3] * column_count,
        left=0.045,
        right=0.985,
        top=0.915,
        bottom=0.045,
        wspace=0.20,
        hspace=0.34,
    )
    repairs = primary.repair._repair_summary()
    point_v6_identity = _read_json(
        ARTIFACTS
        / "point_maze_balanced_v6_stage_b_05b_12pass_identity.json"
    )
    point_v6_jobs = len(point_v6_identity.get("jobs", {}))
    figure.text(
        0.045,
        0.982,
        title,
        fontsize=21,
        fontweight="bold",
        color="#0F172A",
        va="top",
    )
    figure.text(
        0.045,
        0.961,
        "Online verified MaxEnt vs compute-matched Dr.GRPO · "
        "12 passes · readable square panels",
        fontsize=11.5,
        color="#475569",
        va="top",
    )
    figure.text(
        0.045,
        0.943,
        f"Original estimator: {primary.repair._count_original_terminal()}/80 "
        "audited terminal · Repairs excluded · "
        f"Pantry {repairs['pantry_model_jobs_launched']}/10 · "
        f"Point v6 {point_v6_jobs}/10 launched · "
        f"Ant v18 {repairs['ant_controller_gate_status']} · "
        f"{generated.strftime('%Y-%m-%d %H:%M UTC')}",
        fontsize=9.2,
        color="#64748B",
        va="top",
    )
    for column, (_metric, metric_title, _integer) in enumerate(
        metrics, start=1
    ):
        position = grid[0, column].get_position(figure)
        figure.text(
            (position.x0 + position.x1) / 2,
            0.922,
            metric_title,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1E293B",
        )
    for row_index, (
        label,
        note,
        domain,
        prefix,
        steps,
        seeds,
    ) in enumerate(rows):
        label_axis = figure.add_subplot(grid[row_index, 0])
        label_axis.axis("off")
        label_axis.text(
            0.02,
            0.62,
            label,
            fontsize=12,
            fontweight="bold",
            color="#0F172A",
            va="center",
            wrap=True,
        )
        label_axis.text(
            0.02,
            0.32,
            note,
            fontsize=8.3,
            color="#64748B",
            va="center",
            wrap=True,
        )
        points = _points(domain, prefix, steps)
        if not points and domain in {
            "pantry_support_retention_repair",
            "point_maze_algorithm_repair",
            "ant_maze_harder_repair",
            "constructive_code_qualification",
        }:
            status_axis = figure.add_subplot(grid[row_index, 1:])
            headline, detail, color = _status(domain)
            status_axis.set_facecolor("#FFFFFF")
            status_axis.set_xticks([])
            status_axis.set_yticks([])
            for spine in status_axis.spines.values():
                spine.set_color("#CBD5E1")
                spine.set_linewidth(0.9)
            status_axis.text(
                0.5,
                0.60,
                headline,
                transform=status_axis.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )
            status_axis.text(
                0.5,
                0.39,
                detail,
                transform=status_axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#64748B",
                wrap=True,
            )
            continue
        for column, (metric, _metric_title, integer) in enumerate(
            metrics, start=1
        ):
            axis = figure.add_subplot(grid[row_index, column])
            axis.set_facecolor("#FFFFFF")
            primary._plot(axis, points, seeds, metric, integer)
            if row_index == len(rows) - 1:
                axis.set_xlabel("training passes", fontsize=8.5)
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=primary.BLUE,
                lw=2.5,
                marker="o",
                label="compute-matched Dr.GRPO",
            ),
            Line2D(
                [],
                [],
                color=primary.ORANGE,
                lw=2.5,
                marker="D",
                markerfacecolor="white",
                label="online verified MaxEnt",
            ),
            Line2D(
                [],
                [],
                color="#475569",
                lw=2.3,
                label="thick = available-seed mean; band = range",
            ),
        ],
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.985, 0.987),
    )
    figure.text(
        0.5,
        0.014,
        footer,
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    png, pdf = _page_paths(slug)
    png.parent.mkdir(parents=True, exist_ok=True)
    temporary_png = png.with_name(f".{png.name}.tmp")
    figure.savefig(
        temporary_png,
        format="png",
        dpi=220,
        facecolor=figure.get_facecolor(),
    )
    temporary_png.replace(png)
    temporary_pdf = pdf.with_name(f".{pdf.name}.tmp")
    figure.savefig(
        temporary_pdf,
        format="pdf",
        facecolor=figure.get_facecolor(),
    )
    temporary_pdf.replace(pdf)
    plt.close(figure)
    if slug == "current_outcomes":
        FINAL_PRIMARY_PNG.write_bytes(png.read_bytes())
    print(f"[e70-complete-square] wrote {png}")
    print(f"[e70-complete-square] wrote {pdf}")
    return {
        "slug": slug,
        "title": title,
        "png": str(png.relative_to(ROOT)),
        "png_sha256": _sha256(png),
        "pdf": str(pdf.relative_to(ROOT)),
        "pdf_sha256": _sha256(pdf),
        "metrics": [metric for metric, _title, _integer in metrics],
        "metric_count": len(metrics),
        "row_count": len(rows),
        "axes_box_aspect": 1,
    }


def render() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#64748B",
            "ytick.color": "#64748B",
        }
    )
    generated = datetime.now(timezone.utc)
    rendered = [
        _render_page(
            slug,
            title,
            metrics,
            footer,
            generated=generated,
        )
        for slug, title, metrics, footer in PAGES
    ]
    metric_inventory = [
        metric
        for _slug, _title, metrics, _footer in PAGES
        for metric, _metric_title, _integer in metrics
    ]
    _atomic_json(
        SIDECAR,
        {
            "schema": "e70-complete-readable-square-pages-v1",
            "generated_at": generated.isoformat(),
            "pages": rendered,
            "metric_inventory": metric_inventory,
            "july_28_metric_inventory_complete": set(
                metric_inventory
            ).issuperset(
                {
                    "greedy",
                    "mean8",
                    "pass8",
                    "distinct8",
                    OPEN_SET_ENTROPY,
                    OPEN_SET_COEFFICIENT,
                    MASS_COEFFICIENT,
                    BALANCE_COEFFICIENT,
                    "canonical_replay_balance_loss",
                    "canonical_replay_available_modes",
                    "online_canonical_new_outcome_row_fraction",
                    "online_canonical_mean_support_per_prompt",
                    "online_canonical_tracked_outcomes",
                }
            ),
            "additional_current_metric": (
                "online_canonical_support_at_least_two_prompt_fraction"
            ),
            "repair_excluded_from_original_estimator": True,
            "pantry_selection": {
                "displayed_cohort": "support_retention_final_v1",
                "displayed_seeds": list(PANTRY_REPAIR_SEEDS),
                "original_stage_b_displayed": False,
                "original_stage_b_retained_in_immutable_audit": True,
            },
            "point_maze_selection": {
                "displayed_cohort": "balanced_v6_stage_b_05b_12pass",
                "displayed_seeds": list(POINT_V6_SEEDS),
                "qualification_verified_rate": 52 / 256,
                "untouched_evaluation_split": True,
                "original_stage_b_retained_in_immutable_audit": True,
            },
            "historical_values_imported": False,
            "carry_forward": False,
            "primary_png_alias": str(FINAL_PRIMARY_PNG.relative_to(ROOT)),
            "primary_png_alias_sha256": _sha256(FINAL_PRIMARY_PNG),
        },
    )
    print(f"[e70-complete-square] wrote {SIDECAR}")


if __name__ == "__main__":
    render()
