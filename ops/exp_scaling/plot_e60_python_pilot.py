#!/usr/bin/env python3
"""Render the live E60 Python pilot against its matched Dr.GRPO seed."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
TREATMENT_CURVE = (
    ROOT / "var/artifacts/pye60_bootstrap_local_05b_5ep_pilot_scaling_curve.json"
)
CONTROL_CURVE = (
    ROOT
    / "var/artifacts/"
    "pye53_verified_replay_05b_50ep_sentinel_allcs_scaling_curve.json"
)
IDENTITY = ROOT / "var/artifacts/e60_bootstrap_local_python_pilot_identity.json"
AUDIT = ROOT / "var/artifacts/e60_python_pilot_audit_latest.json"
OUT = ROOT / "paper/figures/e60_python_bootstrap_local_pilot_live"
TREATMENT_ARM = "verified_first_bootstrap_local_canonical"
CONTROL_ARM = "grpo"
STEPS_PER_PASS = 384
ORANGE = "#D55E00"
BLUE = "#0072B2"
GREEN = "#009E73"
PURPLE = "#7A5195"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _eval_points(path: Path, arm: str) -> list[dict[str, float]]:
    rows = _load(path)
    by_step: dict[int, dict[str, float]] = {}
    for row in rows:
        if (
            row.get("arm") != arm
            or row.get("split") != "multi_answer"
            or not _finite(row.get("distinct8"))
        ):
            continue
        step = int(row["step"])
        by_step[step] = {
            "step": float(step),
            "passes": step / STEPS_PER_PASS,
            "distinct8": float(row["distinct8"]),
            "pass8": float(row["pass8"]),
            "mean8": float(row["mean8"]),
        }
    return [by_step[step] for step in sorted(by_step)]


def _find_metrics(identity: dict[str, Any]) -> Path | None:
    job_id = int(identity["job_id"])
    candidates = sorted(
        (ROOT / "var/data").glob(
            "*pye60_bootstrap_local_05b_5ep_pilot_"
            f"{TREATMENT_ARM}_s9010/debug_job{job_id}/train_metrics.jsonl"
        )
    )
    return candidates[0] if len(candidates) == 1 else None


def _training_points(path: Path | None) -> list[dict[str, float]]:
    if path is None or not path.is_file():
        return []
    by_step: dict[int, dict[str, float]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        step = int(row.get("trainer/global_step", -1))
        if step < 0:
            continue
        point = {
            "step": float(step),
            "passes": step / STEPS_PER_PASS,
        }
        mappings = {
            "support": "train/verified_discovery_mean_support_per_prompt",
            "entropy": (
                "train/semantic_shannon_success_conditioned_signed_"
                "open_set_entropy_ema"
            ),
            "alpha": (
                "train/semantic_shannon_success_conditioned_signed_"
                "open_set_next_coefficient"
            ),
            "bootstrap_updates": (
                "train/canonical_replay_global_bootstrap_updates"
            ),
            "used_local": (
                "train/canonical_replay_schedule_used_prompt_local"
            ),
        }
        for name, key in mappings.items():
            if _finite(row.get(key)):
                point[name] = float(row[key])
        by_step[step] = point
    return [by_step[step] for step in sorted(by_step)]


def _xy(rows: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if key in row]
    return (
        [row["passes"] for row in selected],
        [row[key] for row in selected],
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#dddddd", lw=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.7, labelsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def main() -> None:
    identity = _load(IDENTITY)
    audit = _load(AUDIT) if AUDIT.is_file() else {}
    treatment = _eval_points(TREATMENT_CURVE, TREATMENT_ARM)
    control = _eval_points(CONTROL_CURVE, CONTROL_ARM)
    training = _training_points(_find_metrics(identity))

    latest_pass = max(
        [row["passes"] for row in training]
        + [row["passes"] for row in treatment]
        + [0.0]
    )
    x_upper = max(0.5, latest_pass + max(0.08, latest_pass * 0.06))
    control = [row for row in control if row["passes"] <= x_upper + 1e-12]
    transition = next(
        (
            row["passes"]
            for row in training
            if row.get("used_local", 0.0) == 1.0
        ),
        None,
    )

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.0), sharex=True)
    metrics = (
        ("distinct8", "mean # distinct correct@8"),
        ("pass8", "pass@8"),
        ("mean8", "mean correctness@8"),
    )
    for ax, (metric, label) in zip(axes[0], metrics):
        cx, cy = _xy(control, metric)
        tx, ty = _xy(treatment, metric)
        ax.plot(
            cx,
            cy,
            color=BLUE,
            lw=1.8,
            ls=(0, (4, 2)),
            marker="o",
            ms=3.2,
            alpha=0.88,
            zorder=2,
        )
        ax.plot(
            tx,
            ty,
            color=ORANGE,
            lw=2.4,
            marker="D",
            ms=5.2,
            alpha=0.96,
            zorder=5,
        )
        if len(treatment) == 1 and latest_pass > treatment[0]["passes"]:
            ax.text(
                0.025,
                0.08,
                "E60 is present at pass 0\nnext eval: pass 0.25",
                transform=ax.transAxes,
                color=ORANGE,
                fontsize=7.2,
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": ORANGE,
                    "alpha": 0.92,
                    "linewidth": 0.8,
                },
                zorder=6,
            )
        ax.set_title(label, fontsize=9)
        ax.set_ylim(bottom=0)
        _style_axis(ax)

    ax = axes[1, 0]
    for rows, color, style, width in (
        (control, BLUE, (0, (4, 2)), 1.8),
        (treatment, ORANGE, "-", 2.4),
    ):
        xs = [row["passes"] for row in rows]
        ys = [row["distinct8"] - row["pass8"] for row in rows]
        marker = "D" if color == ORANGE else "o"
        ax.plot(
            xs,
            ys,
            color=color,
            ls=style,
            lw=width,
            marker=marker,
            ms=4.6 if color == ORANGE else 3.5,
            zorder=5 if color == ORANGE else 2,
        )
    ax.axhline(0.0, color="#777777", lw=0.8)
    ax.set_title("excess multiplicity: distinct@8 − pass@8", fontsize=9)
    ax.set_ylim(bottom=-0.01)
    _style_axis(ax)

    ax = axes[1, 1]
    sx, sy = _xy(training, "support")
    ax.plot(sx, sy, color=GREEN, lw=2.1)
    ax.axhline(1.0, color="#777777", lw=0.8, ls=(0, (2, 2)))
    ax.set_title("model-discovered mean support / solved prompt", fontsize=9)
    ax.set_ylim(bottom=0)
    _style_axis(ax)

    ax = axes[1, 2]
    ex, ey = _xy(training, "entropy")
    ax.plot(ex, ey, color=PURPLE, lw=2.0, label="entropy EMA")
    ax.set_ylim(bottom=0)
    ax.set_title("open-set entropy sensor and unbounded coefficient", fontsize=9)
    ax2 = ax.twinx()
    bx, by = _xy(training, "alpha")
    ax2.plot(bx, by, color=ORANGE, lw=1.6, label=r"next $\beta$")
    ax2.set_ylim(bottom=0)
    ax2.tick_params(length=2.5, width=0.7, labelsize=7.5, colors=ORANGE)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(ORANGE)
    _style_axis(ax)

    for ax in axes.flat:
        ax.set_xlim(0.0, x_upper)
        if transition is not None:
            ax.axvline(
                transition,
                color="#555555",
                lw=0.9,
                ls=(0, (2, 2)),
                alpha=0.8,
                zorder=1,
            )
    for ax in axes[1]:
        ax.set_xlabel("training passes", fontsize=8)

    handles = [
        Line2D(
            [], [], color=ORANGE, lw=2.4, marker="D",
            label="E60 bootstrap→local (seed 9010 pilot)",
        ),
        Line2D(
            [], [], color=BLUE, lw=1.8, ls=(0, (4, 2)), marker="o",
            label="matched Dr.GRPO (seed 9010)",
        ),
    ]
    if transition is not None:
        handles.append(
            Line2D(
                [], [], color="#555555", lw=0.9, ls=(0, (2, 2)),
                label="first prompt-local replay update",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.965),
    )
    status = str(audit.get("status", "starting"))
    step = int(audit.get("latest_step", -1))
    updates = int((audit.get("bootstrap") or {}).get("updates", 0))
    fig.suptitle(
        "E60 executable Python factors — finite verified bootstrap, then "
        "prompt-local canonical replay\n"
        f"live status={status}; optimizer step={step}/{1920}; "
        f"global bootstrap={updates}/64; one sentinel seed (not a 3-seed mean)",
        fontsize=11,
        y=1.025,
    )
    fig.text(
        0.5,
        0.005,
        "The x-axis expands with observed training data. Training and "
        "adaptation use no gold support, desired mode count, or evaluation "
        "feedback; all alpha coefficients are unprojected.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[e60-plot] wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
