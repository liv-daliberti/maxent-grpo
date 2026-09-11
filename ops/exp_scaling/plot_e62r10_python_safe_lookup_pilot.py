#!/usr/bin/env python3
"""Render E62-R10's live same-seed Python mechanism comparison."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
CURVE = (
    ROOT
    / "var/artifacts/"
    "pye62r10_safe_lookup_05b_1ep_pilot_scaling_curve.json"
)
IDENTITY = (
    ROOT / "var/artifacts/e62r10_safe_lookup_python_pilot_identity.json"
)
AUDIT = ROOT / "var/artifacts/e62r10_python_pilot_audit_latest.json"
OUT = ROOT / "paper/figures/e62r10_python_safe_lookup_pilot_live"
CONTROL = "verified_first_bootstrap_local_canonical"
TREATMENT = "verified_counterfactual_canonical"
SEED = 9011
STEPS_PER_PASS = 384
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _eval_points(arm: str) -> list[dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    for row in _load(CURVE, []):
        if (
            row.get("arm") != arm
            or row.get("split") != "multi_answer"
            or int(row.get("seed", -1)) != SEED
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


def _run_metrics(identity: dict[str, Any], arm: str) -> Path | None:
    try:
        job_id = int(identity["jobs"][arm])
    except (KeyError, TypeError, ValueError):
        return None
    candidates = sorted(
        (ROOT / "var/data").glob(
            "*pye62r10_safe_lookup_05b_1ep_pilot_"
            f"{arm}_s{SEED}/debug_job{job_id}/train_metrics.jsonl"
        )
    )
    return candidates[0] if len(candidates) == 1 else None


def _training_points(
    path: Path | None,
    *,
    treatment: bool,
) -> list[dict[str, float]]:
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
        point = {"passes": step / STEPS_PER_PASS, "step": float(step)}
        mappings = {
            "support": "train/verified_discovery_mean_support_per_prompt",
        }
        if treatment:
            mappings.update(
                {
                    "proposal_cumulative": (
                        "actor/counterfactual_proposal_"
                        "cumulative_new_outcomes"
                    ),
                    "proposal_group": (
                        "actor/counterfactual_proposal_groups_generated"
                    ),
                    "proposal_novel": (
                        "actor/counterfactual_proposal_"
                        "admitted_new_outcomes"
                    ),
                    "proposal_rows": (
                        "actor/counterfactual_proposal_rows_generated"
                    ),
                    "transform_success": (
                        "actor/counterfactual_proposal_transform_success"
                    ),
                    "transform_novel": (
                        "actor/counterfactual_proposal_"
                        "transform_novel_unique_outcomes"
                    ),
                }
            )
        for name, key in mappings.items():
            if _finite(row.get(key)):
                point[name] = float(row[key])
        by_step[step] = point
    return [by_step[step] for step in sorted(by_step)]


def _xy(
    rows: list[dict[str, float]],
    key: str,
) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if key in row]
    return (
        [row["passes"] for row in selected],
        [row[key] for row in selected],
    )


def _style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#dddddd", lw=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.7, labelsize=7.5)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def _atomic_savefig(fig: plt.Figure, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
    )
    try:
        fig.savefig(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    identity = _load(IDENTITY, {})
    audit = _load(AUDIT, {})
    control_eval = _eval_points(CONTROL)
    treatment_eval = _eval_points(TREATMENT)
    control_train = _training_points(
        _run_metrics(identity, CONTROL),
        treatment=False,
    )
    treatment_train = _training_points(
        _run_metrics(identity, TREATMENT),
        treatment=True,
    )
    latest_pass = max(
        [row["passes"] for row in control_train]
        + [row["passes"] for row in treatment_train]
        + [row["passes"] for row in control_eval]
        + [row["passes"] for row in treatment_eval]
        + [0.0]
    )
    x_upper = max(0.25, latest_pass + max(0.05, latest_pass * 0.06))

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.2), sharex=True)
    for ax, (metric, label) in zip(
        axes[0],
        (
            ("distinct8", "mean # distinct correct@8"),
            ("pass8", "pass@8"),
            ("mean8", "mean correctness@8"),
        ),
    ):
        for rows, color, marker, style, width, label_arm in (
            (
                control_eval,
                BLUE,
                "o",
                (0, (4, 2)),
                1.8,
                "E60-minus-proposal",
            ),
            (
                treatment_eval,
                ORANGE,
                "D",
                "-",
                2.5,
                "E62 safe lookup transforms",
            ),
        ):
            x, y = _xy(rows, metric)
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                ls=style,
                lw=width,
                ms=5.0 if marker == "D" else 3.5,
                label=label_arm,
                zorder=5 if marker == "D" else 2,
            )
        ax.set_title(label, fontsize=9)
        ax.set_ylim(bottom=0)
        _style(ax)

    ax = axes[1, 0]
    for rows, color, marker, style, width in (
        (control_eval, BLUE, "o", (0, (4, 2)), 1.8),
        (treatment_eval, ORANGE, "D", "-", 2.5),
    ):
        ax.plot(
            [row["passes"] for row in rows],
            [row["distinct8"] - row["pass8"] for row in rows],
            color=color,
            marker=marker,
            ls=style,
            lw=width,
            ms=4.8 if marker == "D" else 3.5,
        )
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_title("neutral excess multiplicity: distinct@8 − pass@8", fontsize=9)
    _style(ax)

    ax = axes[1, 1]
    cx, cy = _xy(control_train, "support")
    tx, ty = _xy(treatment_train, "support")
    ax.plot(cx, cy, color=BLUE, lw=1.7, ls=(0, (4, 2)))
    ax.plot(tx, ty, color=GREEN, lw=2.2)
    ax.axhline(1.0, color="#777777", lw=0.8, ls=(0, (2, 2)))
    ax.set_title("verified mean support / discovered prompt", fontsize=9)
    ax.set_ylim(bottom=0)
    _style(ax)

    ax = axes[1, 2]
    px, py = _xy(treatment_train, "proposal_cumulative")
    ax.plot(px, py, color=PURPLE, lw=2.2, label="cumulative new modes")
    ax.set_ylim(bottom=0)
    ax.set_title("E62 verified lookup discoveries and activity", fontsize=9)
    ax2 = ax.twinx()
    gx, gy = _xy(treatment_train, "transform_success")
    ax2.plot(
        gx,
        gy,
        color=ORANGE,
        lw=1.0,
        alpha=0.55,
        label="transform success this update",
    )
    ax2.set_ylim(-0.02, 1.05)
    ax2.tick_params(length=2.5, width=0.7, labelsize=7.5, colors=ORANGE)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(ORANGE)
    _style(ax)

    for ax in axes.flat:
        ax.set_xlim(0.0, x_upper)
    for ax in axes[1]:
        ax.set_xlabel(
            "training passes (axis expands with landed data)",
            fontsize=8,
        )
    handles = [
        Line2D(
            [],
            [],
            color=BLUE,
            lw=1.8,
            ls=(0, (4, 2)),
            marker="o",
            label="E60 mechanism control, seed 9011",
        ),
        Line2D(
            [],
            [],
            color=ORANGE,
            lw=2.5,
            marker="D",
            label="E62-R10 prompt-local verified lookup, seed 9011",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.965),
    )
    status = str(audit.get("status", "starting"))
    arms = audit.get("arms") if isinstance(audit.get("arms"), dict) else {}
    control_step = int((arms.get(CONTROL) or {}).get("latest_step", -1))
    treatment_step = int((arms.get(TREATMENT) or {}).get("latest_step", -1))
    cumulative = int(
        ((arms.get(TREATMENT) or {}).get("proposal") or {}).get(
            "cumulative_new_outcomes",
            0,
        )
    )
    fig.suptitle(
        "E62 executable Python factors — same-seed safe-lookup "
        "mechanism pilot\n"
        f"status={status}; control step={control_step}/384; "
        f"E62 step={treatment_step}/384; "
        f"verified transform discoveries={cumulative}",
        fontsize=11,
        y=1.025,
    )
    fig.text(
        0.5,
        0.006,
        "A model-generated verified output vector is serialized as a complete "
        "prompt-local conditional lookup and independently revalidated; "
        "training never sees gold support, a desired mode count, desired "
        "entropy, or evaluation feedback. Transform rows sent to PPO: "
        "exactly zero.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.98, 0.92))
    _atomic_savefig(
        fig,
        OUT.with_suffix(".png"),
        dpi=180,
        bbox_inches="tight",
    )
    _atomic_savefig(fig, OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[e62-plot] wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
