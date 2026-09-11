#!/usr/bin/env python3
"""Render E63's expanding one-seed cross-domain mechanism gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "var/artifacts/e63_cross_domain_transform_pilot_audit_latest.json"
OUT = ROOT / "paper/figures/e63_cross_domain_transform_pilot_live"
CONTROL = "verified_first_bootstrap_local_canonical"
TREATMENT = "verified_counterfactual_canonical"
DOMAINS = (
    ("graph_coloring", "Graph coloring"),
    ("countdown", "Countdown"),
    ("mathir", "MathIR action menu"),
)
BLUE = "#0072B2"
ORANGE = "#D55E00"


def _load() -> dict[str, Any]:
    try:
        value = json.loads(AUDIT.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _points(
    audit: dict[str, Any],
    domain: str,
    arm: str,
) -> list[dict[str, float]]:
    domain_result = (audit.get("domains") or {}).get(domain) or {}
    terminal_step = float(domain_result.get("terminal_step", 1))
    arm_result = (domain_result.get("arms") or {}).get(arm) or {}
    result: list[dict[str, float]] = []
    for row in arm_result.get("evaluations") or []:
        try:
            step = float(row["step"])
            distinct = float(row["distinct8"])
            passed = float(row["pass8"])
            mean = float(row["mean8"])
        except (KeyError, TypeError, ValueError):
            continue
        result.append(
            {
                "passes": step / terminal_step,
                "distinct": distinct,
                "excess": distinct - passed,
                "mean": mean,
            }
        )
    return result


def _style(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#dddddd", lw=0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.7, labelsize=7.5)
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.yaxis.set_major_locator(MaxNLocator(5))


def _atomic_savefig(
    figure: plt.Figure,
    path: Path,
    **kwargs: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
    )
    try:
        figure.savefig(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    audit = _load()
    figure, axes = plt.subplots(
        len(DOMAINS),
        3,
        figsize=(12.0, 8.3),
        sharex=True,
    )
    latest_pass = 0.0
    for row_index, (domain, label) in enumerate(DOMAINS):
        control = _points(audit, domain, CONTROL)
        treatment = _points(audit, domain, TREATMENT)
        latest_pass = max(
            [
                latest_pass,
                *(point["passes"] for point in control),
                *(point["passes"] for point in treatment),
            ]
        )
        for column, (metric, title) in enumerate(
            (
                ("distinct", "mean # distinct correct@8"),
                ("excess", "excess multiplicity: distinct@8 − pass@8"),
                ("mean", "mean correctness@8"),
            )
        ):
            axis = axes[row_index, column]
            for points, color, marker, line_style, width in (
                (control, BLUE, "o", (0, (4, 2)), 2.0),
                (treatment, ORANGE, "D", "-", 2.5),
            ):
                axis.plot(
                    [point["passes"] for point in points],
                    [point[metric] for point in points],
                    color=color,
                    marker=marker,
                    ls=line_style,
                    lw=width,
                    ms=4.2,
                )
            if metric == "excess":
                axis.axhline(0, color="#777777", lw=0.8)
            else:
                axis.set_ylim(bottom=0)
            if row_index == 0:
                axis.set_title(title, fontsize=9)
            if column == 0:
                axis.set_ylabel(label, fontsize=9, fontweight="bold")
            _style(axis)

    x_upper = max(0.25, latest_pass + max(0.05, 0.06 * latest_pass))
    for axis in axes.flat:
        axis.set_xlim(0.0, x_upper)
    for axis in axes[-1]:
        axis.set_xlabel(
            "training passes (axis expands with landed data)",
            fontsize=8,
        )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=BLUE,
                marker="o",
                ls=(0, (4, 2)),
                lw=2.0,
                label="same-seed mechanism control",
            ),
            Line2D(
                [],
                [],
                color=ORANGE,
                marker="D",
                lw=2.5,
                label="validator-preserving transform",
            ),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.96),
    )
    violations = len(audit.get("violations") or [])
    figure.suptitle(
        "E63 live — one seed per arm, Graph/Countdown/MathIR mechanism gate\n"
        f"status={audit.get('status', 'starting')}; "
        f"latest landed step={audit.get('latest_step', -1)}; "
        f"audit violations={violations}",
        fontsize=11,
        y=1.005,
    )
    figure.text(
        0.5,
        0.008,
        "Transformations use only model-generated verified responses and "
        "public constraints; no gold support, desired mode count, desired "
        "entropy, evaluation feedback, alpha cap, or transformed PPO rows.",
        ha="center",
        fontsize=7.4,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.98, 0.93))
    _atomic_savefig(
        figure,
        OUT.with_suffix(".png"),
        dpi=180,
        bbox_inches="tight",
    )
    _atomic_savefig(
        figure,
        OUT.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"[e63-plot] wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
