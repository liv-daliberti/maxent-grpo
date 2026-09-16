#!/usr/bin/env python3
"""Plot terminal replay-bank and compute telemetry from the clean E78 cohort.

This is deliberately a mechanism diagnostic rather than an outcome figure.  It
reads every optimizer-update record from the 25 terminal ReplayDr.GRPO runs and
their 25 exact-zero, compute-matched controls.  No evaluation metric is used to
select a run, seed, domain, update, or displayed statistic.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402


DEFAULT_DATA = ROOT / "var/data"
DEFAULT_OUTPUT = ROOT / "paper/figures/replay_mechanism_telemetry_qwen05b"
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
DOMAIN_LABELS = {
    "graph_coloring": "Graph",
    "countdown": "Countdown",
    "python_factors": "Python",
    "mathir": "MathIR",
    "pantry_plan": "Pantry",
}
DOMAIN_COLORS = {
    "graph_coloring": style.METHOD,
    "countdown": style.CONTROL,
    "python_factors": style.ABLATION,
    "mathir": style.ADD_ON,
    "pantry_plan": style.INK,
}
DOMAIN_DASHES = {
    "graph_coloring": "solid",
    "countdown": (0, (5, 1.6)),
    "python_factors": (0, (1.6, 1.4)),
    "mathir": (0, (4, 1.2, 1, 1.2)),
    "pantry_plan": (0, (6, 1.2, 1, 1.2, 1, 1.2)),
}
SEEDS = (43, 44, 45, 46, 47)
BIN_WIDTH = 0.25
RUN_RE = re.compile(
    r"e78_replay_only_(graph|countdown|python|mathir|pantry)_"
    r"(control|replay)_s(4[3-7])$"
)
RUN_DOMAIN = {
    "graph": "graph_coloring",
    "countdown": "countdown",
    "python": "python_factors",
    "mathir": "mathir",
    "pantry": "pantry_plan",
}
FIELDS = {
    "available": "train/canonical_replay_available_modes",
    "retained": "train/canonical_replay_retained_modes",
    "realized_response": "train/canonical_replay_realized_response_tokens",
    "charged_response": "train/canonical_replay_charged_response_token_budget",
    "capacity": "train/canonical_replay_capacity",
    "compute_only": "train/canonical_replay_compute_only",
    "train_time": "train/total_time",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover(data_root: Path) -> dict[tuple[str, str, int], Path]:
    found: dict[tuple[str, str, int], Path] = {}
    for path in data_root.glob("*e78_replay_only_*/debug_job*/train_metrics.jsonl"):
        match = RUN_RE.search(path.parents[1].name)
        if match is None:
            continue
        run_domain, arm, seed_text = match.groups()
        domain = RUN_DOMAIN[run_domain]
        key = (domain, arm, int(seed_text))
        if key in found:
            raise RuntimeError(f"duplicate telemetry log for {key}: {found[key]} and {path}")
        found[key] = path.resolve()
    expected = {
        (domain, arm, seed)
        for domain in DOMAINS
        for arm in ("control", "replay")
        for seed in SEEDS
    }
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    if missing or extra:
        raise RuntimeError(f"E78 telemetry identity mismatch; missing={missing}, extra={extra}")
    return found


def _load(path: Path, *, arm: str) -> dict[str, Any]:
    by_step: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            # Evaluation summaries lack ``train/total_time``.  A handful of
            # early control updates have an empty bank and therefore omit the
            # retained/compute-only fields; those two structural zeros are
            # restored below, while every measured field remains required.
            required = {
                key
                for name, key in FIELDS.items()
                if name not in {"retained", "compute_only"}
            }
            if not all(key in row for key in required):
                continue
            step = int(float(row["misc/global_step"]))
            values = {
                name: float(
                    row.get(
                        key,
                        0.0 if name == "retained" else (1.0 if arm == "control" else 0.0),
                    )
                )
                for name, key in FIELDS.items()
            }
            if (
                (FIELDS["retained"] not in row or FIELDS["compute_only"] not in row)
                and values["available"] != 0.0
            ):
                raise RuntimeError(f"{path} omits replay fields for a nonempty bank")
            values["dataset_len"] = float(row["misc/prompt_dataset_len"])
            by_step[step] = values
    if sorted(by_step) != list(range(1, 3073)):
        raise RuntimeError(f"{path} does not contain the complete 3,072-update grid")
    rows = [by_step[step] for step in sorted(by_step)]
    dataset_lengths = {row["dataset_len"] for row in rows}
    capacities = {row["capacity"] for row in rows}
    charged = {row["charged_response"] for row in rows}
    compute_only = {row["compute_only"] for row in rows}
    expected_compute_only = {1.0 if arm == "control" else 0.0}
    if dataset_lengths != {384.0} or capacities != {16.0}:
        raise RuntimeError(f"{path} drifted from the registered dataset/capacity")
    if len(charged) != 1 or min(charged) <= 0 or compute_only != expected_compute_only:
        raise RuntimeError(f"{path} drifted from charged-budget/control semantics")
    return {
        "step": np.asarray(sorted(by_step), dtype=float),
        **{
            name: np.asarray([row[name] for row in rows], dtype=float)
            for name in FIELDS
        },
    }


def _run_summary(run: dict[str, Any]) -> dict[str, float | int]:
    late = run["step"] > 2304  # passes 6--8, fixed before reading outcomes
    warm = run["step"] > 192  # exclude first half-pass compilation/warm-up
    available = run["available"]
    retained = run["retained"]
    capacity = run["capacity"]
    return {
        "updates": int(len(run["step"])),
        "mean_available_modes": float(np.mean(available)),
        "late_mean_available_modes": float(np.mean(available[late])),
        "maximum_available_modes": int(np.max(available)),
        "capacity_hit_updates": int(np.sum(available >= capacity)),
        "capacity_hit_fraction": float(np.mean(available >= capacity)),
        "active_replay_fraction": float(np.mean(retained > 0)),
        "mean_retained_modes": float(np.mean(retained)),
        "mean_realized_response_tokens": float(np.mean(run["realized_response"])),
        "charged_response_token_budget": float(run["charged_response"][0]),
        "realized_over_charged_percent": float(
            100.0 * np.mean(run["realized_response"] / run["charged_response"])
        ),
        "median_optimizer_update_seconds_after_warmup": float(
            np.median(run["train_time"][warm])
        ),
    }


def _curve(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    passes = run["step"] / 384.0
    edges = np.arange(0.0, 8.0 + BIN_WIDTH, BIN_WIDTH)
    centers = (edges[:-1] + edges[1:]) / 2.0
    values = []
    for index in range(len(centers)):
        if index == len(centers) - 1:
            mask = (passes > edges[index]) & (passes <= edges[index + 1])
        else:
            mask = (passes > edges[index]) & (passes <= edges[index + 1])
        if not np.any(mask):
            raise RuntimeError("empty telemetry bin")
        values.append(float(np.mean(run["available"][mask])))
    return centers, np.asarray(values)


def _domain_payload(
    domain: str,
    runs: dict[tuple[str, str, int], dict[str, Any]],
    paths: dict[tuple[str, str, int], Path],
) -> dict[str, Any]:
    replay_summaries = {
        str(seed): _run_summary(runs[(domain, "replay", seed)]) for seed in SEEDS
    }
    control_summaries = {
        str(seed): _run_summary(runs[(domain, "control", seed)]) for seed in SEEDS
    }
    curves = [_curve(runs[(domain, "replay", seed)])[1] for seed in SEEDS]
    centers = _curve(runs[(domain, "replay", SEEDS[0])])[0]
    curve_array = np.stack(curves)
    timing_effects = []
    for seed in SEEDS:
        live = replay_summaries[str(seed)][
            "median_optimizer_update_seconds_after_warmup"
        ]
        zero = control_summaries[str(seed)][
            "median_optimizer_update_seconds_after_warmup"
        ]
        timing_effects.append(100.0 * (float(live) - float(zero)) / float(zero))
    late = [float(row["late_mean_available_modes"]) for row in replay_summaries.values()]
    dose = [float(row["realized_over_charged_percent"]) for row in replay_summaries.values()]
    return {
        "label": DOMAIN_LABELS[domain],
        "seeds": list(SEEDS),
        "replay_sources": {
            str(seed): {
                "path": str(paths[(domain, "replay", seed)]),
                "sha256": _sha256(paths[(domain, "replay", seed)]),
            }
            for seed in SEEDS
        },
        "control_sources": {
            str(seed): {
                "path": str(paths[(domain, "control", seed)]),
                "sha256": _sha256(paths[(domain, "control", seed)]),
            }
            for seed in SEEDS
        },
        "replay_seed_summaries": replay_summaries,
        "control_seed_summaries": control_summaries,
        "occupancy_curve": {
            "pass_bin_centers": centers.tolist(),
            "seed_values": {
                str(seed): curve.tolist() for seed, curve in zip(SEEDS, curves)
            },
            "mean": np.mean(curve_array, axis=0).tolist(),
            "minimum": np.min(curve_array, axis=0).tolist(),
            "maximum": np.max(curve_array, axis=0).tolist(),
        },
        "late_occupancy": {
            "window_passes": [6.0, 8.0],
            "seed_values": dict(zip(map(str, SEEDS), late)),
            "mean": float(np.mean(late)),
        },
        "realized_over_charged_percent": {
            "seed_values": dict(zip(map(str, SEEDS), dose)),
            "mean": float(np.mean(dose)),
        },
        "paired_optimizer_time_percent_effect": {
            "warmup_excluded_passes": 0.5,
            "seed_values": dict(zip(map(str, SEEDS), timing_effects)),
            "mean": float(np.mean(timing_effects)),
        },
    }


def _seed_dot_panel(
    axis,
    payload: dict[str, Any],
    field: str,
    xlabel: str,
    *,
    reference: float | None = None,
) -> None:
    style.style_axis(axis)
    y_positions = np.arange(len(DOMAINS) - 1, -1, -1)
    if reference is not None:
        axis.axvline(
            reference,
            color=style.MUTED,
            linewidth=0.9,
            linestyle=(0, (4, 1.5)),
            zorder=2,
        )
    for y, domain in zip(y_positions, DOMAINS):
        record = payload["domains"][domain][field]
        values = np.asarray(list(record["seed_values"].values()), dtype=float)
        jitter = np.linspace(-0.11, 0.11, len(values))
        axis.scatter(
            values,
            y + jitter,
            s=10,
            facecolor=style.WHITE,
            edgecolor=DOMAIN_COLORS[domain],
            linewidth=0.65,
            zorder=3,
        )
        axis.scatter(
            [float(record["mean"])],
            [y],
            s=27,
            marker="D",
            color=DOMAIN_COLORS[domain],
            edgecolor=style.WHITE,
            linewidth=0.5,
            zorder=4,
        )
    axis.set_yticks(y_positions, [DOMAIN_LABELS[domain] for domain in DOMAINS])
    axis.set_xlabel(xlabel, fontsize=style.LABEL_FONT)
    axis.set_ylim(-0.55, len(DOMAINS) - 0.45)


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(2, 2, figsize=(style.WIDTH, 5.0))
    axis = axes[0, 0]
    style.style_axis(axis, title="A  Scheduled-bank occupancy")
    for domain in DOMAINS:
        curve = payload["domains"][domain]["occupancy_curve"]
        x = np.asarray(curve["pass_bin_centers"], dtype=float)
        mean = np.asarray(curve["mean"], dtype=float)
        lower = np.asarray(curve["minimum"], dtype=float)
        upper = np.asarray(curve["maximum"], dtype=float)
        axis.fill_between(
            x,
            lower,
            upper,
            color=DOMAIN_COLORS[domain],
            alpha=0.06,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            mean,
            color=DOMAIN_COLORS[domain],
            linestyle=DOMAIN_DASHES[domain],
            linewidth=style.MEAN_LW,
            zorder=3,
        )
    axis.set_xlim(0.0, 8.0)
    axis.set_xlabel("training pass", fontsize=style.LABEL_FONT)
    axis.set_ylabel("available modes in scheduled bank", fontsize=style.LABEL_FONT)

    _seed_dot_panel(
        axes[0, 1],
        payload,
        "late_occupancy",
        "mean available modes, passes 6--8",
        reference=16.0,
    )
    axes[0, 1].set_title(
        "B  Late occupancy versus capacity", fontsize=style.TITLE_FONT, pad=3
    )
    axes[0, 1].set_xlim(0.0, 16.7)
    axes[0, 1].text(
        16.0,
        4.52,
        "capacity 16",
        ha="right",
        va="bottom",
        color=style.MUTED,
        fontsize=style.SMALL_FONT,
    )

    _seed_dot_panel(
        axes[1, 0],
        payload,
        "realized_over_charged_percent",
        "realized response tokens / charged budget (%)",
    )
    axes[1, 0].set_title(
        "C  Realized replay dose", fontsize=style.TITLE_FONT, pad=3
    )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlim(0.2, 40.0)
    axes[1, 0].set_xticks([0.3, 1.0, 3.0, 10.0, 30.0])
    axes[1, 0].set_xticklabels(["0.3", "1", "3", "10", "30"])

    _seed_dot_panel(
        axes[1, 1],
        payload,
        "paired_optimizer_time_percent_effect",
        "live replay minus exact-zero control (%)",
        reference=0.0,
    )
    axes[1, 1].set_title(
        "D  Median optimizer-update time", fontsize=style.TITLE_FONT, pad=3
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=DOMAIN_COLORS[domain],
            linestyle=DOMAIN_DASHES[domain],
            linewidth=style.MEAN_LW,
            label=DOMAIN_LABELS[domain],
        )
        for domain in DOMAINS
    ]
    style.bottom_legend(
        figure,
        handles,
        [handle.get_label() for handle in handles],
        y=0.005,
        ncol=5,
    )
    figure.suptitle(
        "What verified replay actually stores and computes",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.967,
        (
            "Qwen2.5-0.5B · five terminal seeds per domain · 3,072 updates per run; "
            "thin ranges/points are seeds"
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.995,
        top=0.91,
        bottom=0.13,
        hspace=0.43,
        wspace=0.30,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = _discover(args.data_root.resolve())
    runs = {
        key: _load(path, arm=key[1]) for key, path in sorted(paths.items())
    }
    domains = {
        domain: _domain_payload(domain, runs, paths) for domain in DOMAINS
    }
    replay_summaries = [
        domains[domain]["replay_seed_summaries"][str(seed)]
        for domain in DOMAINS
        for seed in SEEDS
    ]
    total_updates = sum(int(row["updates"]) for row in replay_summaries)
    total_capacity_hits = sum(int(row["capacity_hit_updates"]) for row in replay_summaries)
    payload = {
        "schema": "paper-replay-mechanism-telemetry-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "terminal observational mechanism telemetry",
        "cohort": "E78 Qwen2.5-0.5B ReplayDr.GRPO and exact-zero compute-matched control",
        "selection_rule": (
            "all five registered static domains, seeds 43--47, and all 3,072 "
            "optimizer updates; no evaluation outcome used for selection"
        ),
        "capacity": 16,
        "charged_response_token_budgets_per_update": sorted(
            {
                float(run["charged_response"][0])
                for key, run in runs.items()
                if key[1] == "replay"
            }
        ),
        "occupancy_curve_bin_width_passes": BIN_WIDTH,
        "late_occupancy_window_passes": [6.0, 8.0],
        "optimizer_time_warmup_excluded_passes": 0.5,
        "replay_runs": 25,
        "control_runs": 25,
        "replay_updates": total_updates,
        "capacity_hit_updates": total_capacity_hits,
        "capacity_hit_fraction": total_capacity_hits / total_updates,
        "domains": domains,
        "limitations": [
            "The terminal run directories retain aggregate bank telemetry but no bank-exemplar identity snapshots.",
            "Mode survival and later evaluation re-observation therefore cannot be reconstructed from these artifacts.",
            "Optimizer timing is descriptive wall-clock telemetry after a fixed half-pass warm-up, not a hardware-isolated benchmark.",
            "Charged response-token budget is the registered compute-accounting budget, not measured GPU token throughput.",
        ],
    }
    output = args.output.resolve()
    render(payload, output)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_suffix('.json')}")
    print(
        "capacity hits: "
        f"{total_capacity_hits}/{total_updates} "
        f"({100.0 * payload['capacity_hit_fraction']:.3f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
