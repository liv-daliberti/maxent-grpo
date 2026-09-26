#!/usr/bin/env python3
"""Render Level-2 admission plus the exact terminal Graph factorial block."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style

ADMISSION = ROOT / "var/results/modebench_level2_r5_frozen_repeat1/admission_fairness_report.json"
LEDGER = ROOT / "var/artifacts/e119_level2_qwen05b_factorial_jobs.json"
OUT = ROOT / "paper/figures/modebench_level_admission"
ORDER = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry")
LABELS = ("Graph coloring", "Countdown", "Python factors", "MathIR", "PantryPlan")
LEDGER_DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
TREATMENT_DOMAIN = "graph_coloring"
# Panel A encodes benchmark level; Panel B encodes algorithm. Keep their
# palettes disjoint so color never changes meaning across the figure.
LEVEL1_COLOR = "#475569"
LEVEL2_COLOR = "#0F766E"
METHODS = {
    "drgrpo": ("Dr.GRPO", "#7C3AED", "o", "none"),
    "replay_drgrpo": ("ReplayDr.GRPO (ours)", style.ADAPTIVE, "o", style.ADAPTIVE),
    "maxrl": ("MaxRL", "#D97706", "s", "none"),
    "replay_maxrl": ("ReplayMaxRL", "#DC2626", "s", "#DC2626"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_lines(path: Path, max_lines: int = 16, chunk_bytes: int = 1_048_576) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks: list[bytes] = []
        newlines = 0
        while position > 0 and newlines <= max_lines:
            take = min(chunk_bytes, position)
            position -= take
            handle.seek(position)
            chunk = handle.read(take)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    data = b"".join(reversed(chunks))
    return data.decode("utf-8", errors="replace").splitlines()[-max_lines:]


def endpoint(run: dict) -> dict[str, float] | None:
    draws: dict[int, dict] = {}
    for path in sorted(Path(run["run_dir"]).glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        for line in tail_lines(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("step", -1)) == 3072 and row.get("draw_index") is not None:
                draws[int(row["draw_index"])] = row
    if set(draws) != {0, 1, 2, 3}:
        return None
    return {
        "pass8": statistics.fmean(float(row["metrics"]["any_correct_at_k"]) for row in draws.values()),
        "distinct8": statistics.fmean(float(row["metrics"]["distinct_correct_modes_at_k"]) for row in draws.values()),
    }


def main() -> None:
    admission = json.loads(ADMISSION.read_text())
    ledger = json.loads(LEDGER.read_text())
    assert admission["status"] == "pass"
    assert ledger["schema"] == "e119_level2_qwen05b_factorial_jobs_v1"

    rows = []
    for domain, label in zip(ORDER, LABELS):
        record = admission["frozen_base_model_viability"][domain]["qwen-0.5b"]
        rows.append((label, float(record["level1_pass_at_8"]), float(record["level2_pass_at_8"])))

    by_domain: dict[str, dict[str, dict[int, dict[str, float]]]] = {
        domain: {arm: {} for arm in METHODS} for domain in LEDGER_DOMAINS
    }
    for run in ledger["runs"]:
        value = endpoint(run)
        if value is not None:
            by_domain[run["domain"]][run["arm"]][int(run["seed"])] = value
    by_arm = by_domain[TREATMENT_DOMAIN]
    matched = sorted(set.intersection(*(set(by_arm[arm]) for arm in METHODS)))
    if not matched:
        raise RuntimeError("Level-2 Graph has no terminal four-arm seed intersection")

    terminal_progress: dict[str, dict] = {}
    for domain in LEDGER_DOMAINS:
        arms = by_domain[domain]
        common_four = sorted(set.intersection(*(set(arms[arm]) for arm in METHODS)))
        contrasts = {}
        for control, replay, name in (
            ("drgrpo", "replay_drgrpo", "replay_drgrpo_minus_drgrpo"),
            ("maxrl", "replay_maxrl", "replay_maxrl_minus_maxrl"),
        ):
            common = sorted(set(arms[control]) & set(arms[replay]))
            contrasts[name] = {
                "matched_seeds": common,
                "n": len(common),
                "mean_effects": {
                    metric: statistics.fmean(
                        arms[replay][seed][metric] - arms[control][seed][metric]
                        for seed in common
                    )
                    for metric in ("pass8", "distinct8")
                } if common else {},
            }
        terminal_progress[domain] = {
            "terminal_seeds_by_arm": {
                arm: sorted(values) for arm, values in arms.items()
            },
            "four_arm_matched_seeds": common_four,
            "four_arm_n": len(common_four),
            "complete_block": len(common_four) == 5,
            "contrasts": contrasts,
        }

    figure, axes = plt.subplots(
        1, 2, figsize=(style.WIDTH, 2.12),
        gridspec_kw={"width_ratios": (1.18, 1.0), "wspace": .30},
    )
    left, right = axes
    left.set_facecolor(style.PANEL)
    for y, (_label, level1, level2) in enumerate(rows):
        left.plot([level2, level1], [y, y], color=style.GRID, lw=2, zorder=1)
        left.scatter(level1, y, s=32, facecolors="none", edgecolors=LEVEL1_COLOR, lw=1.3, zorder=3)
        left.scatter(level2, y, s=32, color=LEVEL2_COLOR, zorder=3)
        left.text(max(level1, level2) + .020, y, f"−{level1-level2:.3f}", va="center", fontsize=6.5, color=style.MUTED)
    left.set_yticks(range(5), [row[0] for row in rows], fontsize=7)
    left.invert_yaxis()
    left.set_xlim(0, 1.04)
    left.set_xlabel("frozen base-model pass@8", fontsize=7)
    left.set_title("A  Matched difficulty admission", loc="left", fontsize=8.2, pad=4, y=1.01)
    left.grid(color=style.GRID, lw=.6)
    left.spines[["top", "right", "left"]].set_visible(False)
    left.tick_params(axis="y", length=0)
    left.tick_params(axis="x", labelsize=7)
    left.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=LEVEL1_COLOR, label="Level 1"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=LEVEL2_COLOR, markeredgecolor=LEVEL2_COLOR, label="Level 2"),
        ],
        frameon=True, framealpha=1.0, facecolor="white", edgecolor=style.GRID,
        fancybox=False, borderpad=.35, ncol=2, loc="lower center",
        bbox_to_anchor=(.5, 1.14), fontsize=6.6,
    )

    right.set_facecolor(style.PANEL)
    metric_specs = (("pass8", "pass@8"), ("distinct8", "distinct@8"))
    offsets = {"drgrpo": -.24, "replay_drgrpo": -.08, "maxrl": .08, "replay_maxrl": .24}
    record_methods: dict[str, dict] = {}
    for arm, (label, color, marker, fill) in METHODS.items():
        values_by_metric = {metric: [by_arm[arm][seed][metric] for seed in matched] for metric, _ in metric_specs}
        record_methods[arm] = {"label": label, "per_seed": {str(seed): by_arm[arm][seed] for seed in matched}, "means": {metric: statistics.fmean(values) for metric, values in values_by_metric.items()}}
        for x, (metric, _metric_label) in enumerate(metric_specs):
            values = values_by_metric[metric]
            center = x + offsets[arm]
            jitter = [0.0] if len(values) == 1 else [(-.025 + .05 * i / (len(values)-1)) for i in range(len(values))]
            right.scatter([center+j for j in jitter], values, s=9, marker=marker, facecolors="none", edgecolors=color, lw=.55, alpha=.65, zorder=2)
            right.scatter(center, statistics.fmean(values), s=35, marker=marker, facecolors=fill, edgecolors=color, lw=1.2, zorder=4)
    right.set_xticks((0, 1), [label for _metric, label in metric_specs], fontsize=7)
    right.set_xlim(-.45, 1.45)
    right.set_ylim(0, 1.65)
    right.set_ylabel("terminal endpoint", fontsize=7)
    right.set_title(f"B  Level 2 Graph · terminal n={len(matched)} block", loc="left", fontsize=8.2, pad=4, y=1.01)
    right.grid(color=style.GRID, lw=.6)
    right.spines[["top", "right"]].set_visible(False)
    right.tick_params(axis="y", labelsize=7)
    right.legend(
        handles=[Line2D([0], [0], marker=marker, color="none", markerfacecolor=fill, markeredgecolor=color, label=label, markersize=5) for label, color, marker, fill in METHODS.values()],
        frameon=True, framealpha=1.0, facecolor="white", edgecolor=style.GRID,
        fancybox=False, borderpad=.35, ncol=2, loc="lower center",
        bbox_to_anchor=(.5, 1.14), fontsize=6.0,
    )

    figure.subplots_adjust(left=.16, right=.99, top=.70, bottom=.20)
    for extension in ("pdf", "png"):
        figure.savefig(OUT.with_suffix("." + extension), dpi=240, bbox_inches="tight", pad_inches=.02)
    payload = {
        "schema": "modebench-level-admission-and-e119-progress-v2",
        "target_step": 3072,
        "sources": {str(ADMISSION): digest(ADMISSION), str(LEDGER): digest(LEDGER)},
        "admission_rows": [{"domain": label, "level1_pass8": level1, "level2_pass8": level2} for label, level1, level2 in rows],
        "partial_treatment": {"domain": TREATMENT_DOMAIN, "matched_seeds": matched, "n": len(matched), "complete_block": len(matched) == 5, "methods": record_methods},
        "terminal_progress_by_domain": terminal_progress,
    }
    OUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
