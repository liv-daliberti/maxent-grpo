#!/usr/bin/env python3
"""Render spaced comparison-family previews in the paper's visual language.

This is the replacement path for the all-in-one live Figure 4 wall. Each output
contains one model scale, one scientific comparison family, and at most three
domains. Outputs default to var/artifacts because incomplete live curves are
monitoring previews, not final manuscript evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as style  # noqa: E402
import paper_method_style as method_visuals  # noqa: E402
import cohorts as registry  # noqa: E402
import plot_figure4_with_falcon_preview as wall  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "var/artifacts/paper_comparison_previews"

import paper_matrix as matrix  # noqa: E402
import paper_figure_snapshot as filtered  # noqa: E402

@dataclass(frozen=True)
class ScaleSpec:
    key: str
    label: str
    family: str
    base_ledger: Path


@dataclass(frozen=True)
class Comparison:
    key: str
    title: str
    question: str
    methods: tuple[str, ...]
    attachments: dict[str, tuple[str, ...]]


SCALES = (
    ScaleSpec(
        "qwen05b",
        "Qwen2.5-0.5B",
        "Qwen2.5-0.5B",
        wall.E78_LEDGER,
    ),
    ScaleSpec(
        "falcon1b",
        "Falcon3-1B",
        "Falcon3-1B",
        wall.E79_LEDGER,
    ),
    ScaleSpec(
        "qwen3b",
        "Qwen2.5-3B",
        "Qwen2.5-3B",
        wall.E80R1_LEDGER,
    ),
)

COMPARISONS = (
    Comparison(
        "core_retention",
        "Canonical verified replay",
        "Does Re:Dr.GRPO retain more verified modes than matched Dr.GRPO?",
        ("drgrpo", "replay_grpo"),
        {},
    ),
    Comparison(
        "replay_dose",
        "Replay-dose ablation",
        "Does bank-normalized replay improve on the fixed replay dose?",
        ("replay_grpo", "adaptive_replay_grpo"),
        {"qwen05b": ("e90",)},
    ),
    Comparison(
        "fixed_semantic_factorial",
        "Fixed semantic-MaxEnt factorial",
        "What does fixed Semantic MaxEnt add with and without replay?",
        (
            "drgrpo",
            "semantic_maxent",
            "replay_grpo",
            "replay_semantic_maxent",
        ),
        {
            "qwen05b": ("e81", "e83"),
            "falcon1b": ("e82", "e86"),
            "qwen3b": ("e87",),
        },
    ),
    Comparison(
        "adaptive_semantic_replay",
        "Adaptive semantic MaxEnt with replay",
        "What does the registered adaptive semantic controller add to Re:Dr.GRPO?",
        ("replay_grpo", "adaptive_semantic_replay"),
        {
            "qwen05b": ("e89",),
            "falcon1b": ("e91",),
            "qwen3b": ("e92",),
        },
    ),
)

SCALE_BY_KEY = {item.key: item for item in SCALES}
COMPARISON_BY_KEY = {item.key: item for item in COMPARISONS}
MODEL_ORDER = ("qwen05b", "falcon1b", "qwen3b")

CROSS_SCALE_FACTORIAL_STEM = "fixed_semantic_factorial_cross_scale"
CROSS_SCALE_FACTORIAL_SCALES = MODEL_ORDER
STATIC_DOMAIN_COLUMNS = tuple(
    domain.key for domain in matrix.DOMAINS if domain.stratum == "static"
)

ARM_TO_METHOD = {
    "control": "drgrpo",
    "replay": "replay_grpo",
    "bank_normalized_replay": "adaptive_replay_grpo",
    "semantic_only": "semantic_maxent",
    "semantic": "replay_semantic_maxent",
    "adaptive_semantic": "adaptive_semantic_replay",
}


def _method_visual_tuple(method: str) -> tuple[Any, Any, str]:
    visual = method_visuals.method_style(method)
    return visual["color"], visual["linestyle"], visual["label"]


PANEL_ARM_STYLE = {
    arm: _method_visual_tuple(method) for arm, method in ARM_TO_METHOD.items()
}


def build_snapshot(
    scale: ScaleSpec,
    comparison: Comparison,
    allowed_domains: list[str] | None = None,
) -> dict[str, Any]:
    selected_domains = (
        list(STATIC_DOMAIN_COLUMNS) if allowed_domains is None else allowed_domains
    )
    snapshot = filtered.family_snapshot(scale.base_ledger, selected_domains)
    for tag in comparison.attachments.get(scale.key, ()):
        cohort = registry.by_tag(tag)
        if allowed_domains is None:
            wall._attach_semantic(
                snapshot,
                cohort.path(),
                arm=str(cohort.arm),
                comparator=str(cohort.comparator),
                cohort=cohort.tag,
            )
        else:
            filtered.attach_semantic(
                snapshot,
                cohort.path(),
                allowed_domains,
                arm=str(cohort.arm),
                comparator=str(cohort.comparator),
                cohort=cohort.tag,
            )
    return snapshot


def domain_chunks(
    snapshot: dict[str, Any],
    comparison: Comparison,
    limit: int,
) -> list[list[str]]:
    static = list(snapshot["domains"])
    chunks = [static[index : index + limit] for index in range(0, len(static), limit)]
    return [chunk for chunk in chunks if chunk]


def terminal_domains(
    cells: dict[matrix.CellKey, matrix.Cell],
    scale: ScaleSpec,
    comparison: Comparison,
) -> list[str]:
    """Return domains with any available training evidence.

    This deliberately has no minimum seed count and does not require every
    method in a comparison family to be present. Missing method curves remain
    absent inside an otherwise populated panel, while provenance records the
    exact seeds and checkpoint for every curve that is drawn.
    """

    seeds = matrix.SCALE_BY_KEY[scale.key].seeds
    out = []
    for domain in matrix.DOMAINS:
        keys = [
            matrix.CellKey(method, scale.key, domain.key, seed)
            for method in comparison.methods
            for seed in seeds
        ]
        if any(
            (cell := cells.get(key)) is not None
            and cell.status in {"terminal", "partial", "running"}
            for key in keys
        ):
            out.append(domain.key)
    return out


def fixed_semantic_domains(
    cells: dict[matrix.CellKey, matrix.Cell],
    scale: ScaleSpec,
) -> list[str]:
    """Return every fixed-factorial domain with any available evidence."""

    return terminal_domains(
        cells, scale, COMPARISON_BY_KEY["fixed_semantic_factorial"]
    )


def restrict_snapshot_to_seed(snapshot: dict[str, Any], seed: int) -> None:
    """Keep a semantic treatment and its parent curves seed matched."""

    for domain_curves in snapshot["curves"].values():
        for arm, seed_curves in domain_curves.items():
            domain_curves[arm] = {
                key: curve for key, curve in seed_curves.items() if int(key) == seed
            }
    for spec in wall._semantic_arms(snapshot):
        for domain, seed_curves in spec["curves"].items():
            spec["curves"][domain] = {
                key: curve for key, curve in seed_curves.items() if int(key) == seed
            }


def fixed_semantic_cross_scale_rows(
    cells: dict[matrix.CellKey, matrix.Cell],
) -> list[tuple[ScaleSpec, list[str]]]:
    """Return the three manuscript model rows in invariant order."""

    return [
        (SCALE_BY_KEY[key], fixed_semantic_domains(cells, SCALE_BY_KEY[key]))
        for key in MODEL_ORDER
    ]


def maximum_for(snapshot: dict[str, Any], domains: list[str]) -> float:
    maximum = 1.0
    for domain in domains:
        for seed_curves in snapshot["curves"].get(domain, {}).values():
            for curve in seed_curves.values():
                if curve:
                    maximum = max(maximum, *curve.values())
        for spec in wall._semantic_arms(snapshot):
            for curve in spec["curves"].get(domain, {}).values():
                if curve:
                    maximum = max(maximum, *curve.values())
    return max(2.0, math.ceil(maximum * 2) / 2 + 0.25)


def drawn_arms(snapshot: dict[str, Any], domains: list[str]) -> list[str]:
    arms = ["control", "replay"]
    for spec in wall._semantic_arms(snapshot):
        arm = str(spec["arm"])
        if any(spec["curves"].get(domain) for domain in domains):
            arms.append(arm)
    return arms


def render_chunk(
    snapshot: dict[str, Any],
    scale: ScaleSpec,
    comparison: Comparison,
    domains: list[str],
    output: Path,
    evidence: str,
) -> None:
    style.apply_rcparams()
    figure, axes = plt.subplots(
        1,
        len(domains),
        figsize=(style.WIDTH, 2.62),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    y_high = maximum_for(snapshot, domains)
    records: dict[str, Any] = {}
    for column, domain in enumerate(domains):
        axis = axes[0][column]
        axis.set_ylim(0, y_high)
        records[domain] = wall._panel(
            axis,
            snapshot=snapshot,
            domain=domain,
            row=0,
            total_rows=1,
            arm_styles=PANEL_ARM_STYLE,
        )
    axes[0][0].set_ylabel("mean distinct correct@8", fontsize=style.LABEL_FONT)

    handles = []
    labels = []
    methods = []
    for arm in drawn_arms(snapshot, domains):
        color, dash, label = PANEL_ARM_STYLE[arm]
        handles.append(Line2D([0], [0], color=color, linestyle=dash, lw=1.4))
        labels.append(label)
        methods.append(ARM_TO_METHOD[arm])
    handles.append(
        Line2D([0], [0], color=style.MUTED, lw=style.SEED_LW, alpha=0.55)
    )
    labels.append(
        "paired seed (thin)"
        if evidence in {"terminal", "progress"}
        else "available seed (thin)"
    )
    style.bottom_legend(
        figure,
        handles,
        labels,
        y=-0.025,
        ncol=min(len(labels), 4),
    )
    figure.suptitle(
        f"{comparison.title} — {scale.label}",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.955,
        comparison.question,
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.82,
        bottom=0.23,
        wspace=0.22,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)

    provenance = {
        "schema": "paper-comparison-family-figure-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": comparison.key,
        "question": comparison.question,
        "scale": scale.key,
        "domains": domains,
        "methods": methods,
        "status": {
            "terminal": "all available manuscript evidence; exact n recorded",
            "progress": (
                "frozen progress snapshot; exact available paired subsets"
            ),
            "preview": "live preview; not final manuscript evidence",
        }[evidence],
        "evidence": evidence,
        "metric": "mean distinct correct@8; thick curves use paired seeds only",
        "records": records,
    }
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def render_fixed_semantic_cross_scale(
    cells: dict[matrix.CellKey, matrix.Cell],
    output: Path,
) -> Path | None:
    """Render the invariant three-model by five-environment terminal grid."""

    comparison = COMPARISON_BY_KEY["fixed_semantic_factorial"]
    row_specs = fixed_semantic_cross_scale_rows(cells)
    if not any(domains for _scale, domains in row_specs):
        return None

    terminal_by_scale = {scale.key: domains for scale, domains in row_specs}
    snapshots = {}
    for scale, domains in row_specs:
        if not domains:
            continue
        snapshot = build_snapshot(scale, comparison, domains)
        snapshots[scale.key] = snapshot
    style.apply_rcparams()
    figure = plt.figure(figsize=(style.WIDTH, 7.0))
    grid = figure.add_gridspec(
        len(MODEL_ORDER),
        len(STATIC_DOMAIN_COLUMNS),
        left=0.105,
        right=0.995,
        top=0.89,
        bottom=0.13,
        hspace=0.30,
        wspace=0.24,
    )
    y_high = max(
        maximum_for(snapshots[scale.key], domains)
        for scale, domains in row_specs if domains
    )
    records_by_scale: dict[str, dict[str, Any]] = {
        scale.key: {} for scale, _domains in row_specs
    }
    for row_index, (scale, _domains) in enumerate(row_specs):
        for column, domain in enumerate(STATIC_DOMAIN_COLUMNS):
            axis = figure.add_subplot(grid[row_index, column])
            style.style_axis(
                axis,
                grid="both",
                title=(
                    matrix.DOMAIN_BY_KEY[domain].label
                    if row_index == 0 else None
                ),
            )
            axis.set_xlim(0, 8)
            axis.set_xticks((0, 2, 4, 6, 8))
            axis.set_ylim(0, y_high)
            if domain in terminal_by_scale[scale.key]:
                records_by_scale[scale.key][domain] = wall._panel(
                    axis,
                    snapshot=snapshots[scale.key],
                    domain=domain,
                    row=row_index,
                    total_rows=len(MODEL_ORDER),
                    arm_styles=PANEL_ARM_STYLE,
                )
                for note in list(axis.texts):
                    note.remove()
            if column == 0:
                axis.set_ylabel(
                    f"{scale.label}\nmean distinct correct@8",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)
            if row_index == len(MODEL_ORDER) - 1:
                axis.set_xlabel("training pass", fontsize=style.LABEL_FONT)

    provenance_rows = [
        {
            "row": row_index + 1,
            "scale": scale.key,
            "scale_label": scale.label,
            "environment_columns": list(STATIC_DOMAIN_COLUMNS),
            "domains": domains,
            "missing_domains": [
                domain for domain in STATIC_DOMAIN_COLUMNS if domain not in domains
            ],
            "seeds": list(matrix.SCALE_BY_KEY[scale.key].seeds),
            "row_evidence": "all available trajectories; exact n is stored per method and checkpoint",
            "records": records_by_scale[scale.key],
        }
        for row_index, (scale, domains) in enumerate(row_specs)
    ]

    method_to_arm = {method: arm for arm, method in ARM_TO_METHOD.items()}
    handles = []
    labels = []
    for method in comparison.methods:
        arm = method_to_arm[method]
        color, dash, label = PANEL_ARM_STYLE[arm]
        handles.append(Line2D([0], [0], color=color, linestyle=dash, lw=1.4))
        labels.append(label)
    style.bottom_legend(figure, handles, labels, y=0.003, ncol=4)
    figure.suptitle(
        "Fixed semantic-MaxEnt factorial across model scales",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.96,
        "All available runs are shown; methods and checkpoints retain their exact seed counts.",
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)

    provenance = {
        "schema": "paper-comparison-cross-scale-figure-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": comparison.key,
        "question": comparison.question,
        "methods": list(comparison.methods),
        "status": "all available fixed-factorial evidence; no minimum seed count",
        "evidence": "terminal",
        "metric": "mean distinct correct@8; thick curves use paired seeds only",
        "layout": "three physical model rows by five static-domain columns",
        "model_rows": list(MODEL_ORDER),
        "environment_columns": list(STATIC_DOMAIN_COLUMNS),
        "blank_rule": "a method is blank only when no sampled checkpoint is available",
        "rows": provenance_rows,
    }
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def render_comparison(
    scale: ScaleSpec,
    comparison: Comparison,
    output_dir: Path,
    *,
    panels_per_figure: int,
    evidence: str = "preview",
    allowed_domains: list[str] | None = None,
) -> list[Path]:
    snapshot = build_snapshot(scale, comparison, allowed_domains)
    if allowed_domains is not None:
        allowed = set(allowed_domains)
        snapshot["domains"] = [
            domain for domain in snapshot["domains"] if domain in allowed
        ]
    written = []
    for index, domains in enumerate(
        domain_chunks(snapshot, comparison, panels_per_figure),
        start=1,
    ):
        suffix = f"part{index}"
        output = output_dir / f"{comparison.key}_{scale.key}_{suffix}"
        render_chunk(snapshot, scale, comparison, domains, output, evidence)
        written.append(output)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        choices=(*COMPARISON_BY_KEY, "all"),
        default="all",
    )
    parser.add_argument(
        "--scale",
        choices=(*SCALE_BY_KEY, "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panels-per-figure", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--evidence",
        choices=("preview", "progress", "terminal"),
        default="preview",
        help="paper mode emits every domain with any available run evidence",
    )
    args = parser.parse_args()

    comparisons = (
        COMPARISONS
        if args.comparison == "all"
        else (COMPARISON_BY_KEY[args.comparison],)
    )
    scales = SCALES if args.scale == "all" else (SCALE_BY_KEY[args.scale],)
    cells = matrix.load_cells() if args.evidence == "terminal" else None
    written: list[Path] = []
    for comparison in comparisons:
        for scale in scales:
            # Do not emit a base-only figure under an ablation title when that
            # scale has no registered attachment for the comparison.
            if comparison.attachments and scale.key not in comparison.attachments:
                continue
            allowed_domains = None
            if cells is not None:
                allowed_domains = terminal_domains(cells, scale, comparison)
                if not allowed_domains:
                    print(
                        f"skip {comparison.key}/{scale.key}: no available run evidence"
                    )
                    continue
            written.extend(
                render_comparison(
                    scale,
                    comparison,
                    args.output_dir.resolve(),
                    panels_per_figure=args.panels_per_figure,
                    evidence=args.evidence,
                    allowed_domains=allowed_domains,
                )
            )
    if (
        cells is not None
        and args.scale == "all"
        and args.comparison in {"all", "fixed_semantic_factorial"}
    ):
        merged = render_fixed_semantic_cross_scale(
            cells,
            args.output_dir.resolve() / CROSS_SCALE_FACTORIAL_STEM,
        )
        if merged is not None:
            written.append(merged)
    for output in written:
        print(f"wrote {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
