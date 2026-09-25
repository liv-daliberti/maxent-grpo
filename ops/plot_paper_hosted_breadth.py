#!/usr/bin/env python3
"""Plot hosted correctness and diversity using explicit, complete display cohorts.

All original cells use the frozen formatting normalizer. Only Opus 5 Python
uses the complete separately collected direct-expression prompt cohort, under
that same normalizer. No prompt or draw is selected by correctness or provider
outcome. Opus 5's original-prompt Python cells are drawn beside the displayed
ones as hollow companion marks, so no reader has to take the substituted
condition on trust. The strict original cohorts and prompt-condition comparison
remain in the appendix; this figure is a descriptive display, not a prompting
experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "paper/results/frontier_comparison_20260911.json"
DEFAULT_OUTPUT = ROOT / "paper/figures/hosted_verified_breadth"
MODEL_ORDER = (
    "gpt-5.6-sol", "claude-opus-5", "gpt-5.4", "grok-4.3",
    "FW-Kimi-K3", "claude-opus-4-8", "DeepSeek-V4-Pro",
)
DOMAIN_ORDER = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
DOMAIN_LABELS = ("Graph", "Countdown", "Python", "MathIR", "Pantry")
LEVELS = (1, 2, 3)
PROMPTS = 128
DRAWS = 8
FIGSIZE = (6.4, 3.2)
LEVEL_COLORS = ("#00509E", "#C76A3A", "#7B1FA2")
LEVEL_MARKERS = ("o", "s", "^")


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}: expected an integer count")
    if not math.isfinite(value) or value != int(value) or value < 0:
        raise ValueError(f"{field}: invalid count {value}")
    return int(value)


def _assert_metric(value: Any, expected: float | None, field: str) -> None:
    if expected is None:
        if value is not None:
            raise ValueError(f"{field}: expected an undefined correct-pair metric")
    elif not isinstance(value, (int, float)) or not math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{field}: metric does not match complete-cohort counts")


#: Promptwise PCMD per hosted cell. It is joined rather than derived from the
#: pooled collision this builder already computes: one minus pooled collision
#: weights every correct pair equally across prompts, while \pmd{} averages the
#: per-prompt value, and the paper reports the promptwise form. The two agree to
#: about .003 in the median but differ by more than .05 on 19 of 70 cells, so
#: substituting one for the other would quietly change the estimand.
PMD_COHORT = ROOT / "paper/results/mode_diversity_hosted_cohort_20260917.json"


def pmd_cells() -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = json.loads(PMD_COHORT.read_text(encoding="utf-8"))
    if payload.get("schema") != "paper-mode-diversity-hosted-cohort-v1":
        raise ValueError("hosted cohort PCMD record schema drifted")
    return {(model["model"], cell["domain"], cell["level"]): cell
            for model in payload["models"] for cell in model["cells"]}


def _metrics(counts: dict[str, int]) -> dict[str, float | None]:
    if counts["prompts"] != PROMPTS or counts["responses"] != PROMPTS * DRAWS:
        raise ValueError("Every display cell must retain all 128 prompts and 1,024 responses")
    if counts["correct_responses"] > counts["responses"]:
        raise ValueError("Correct responses exceed the complete denominator")
    if counts["distinct_correct_modes"] > counts["correct_responses"]:
        raise ValueError("Distinct correct modes exceed correct responses")
    if counts["colliding_correct_pairs"] > counts["correct_pairs"]:
        raise ValueError("Colliding correct pairs exceed correct pairs")
    return {
        "accuracy": counts["correct_responses"] / counts["responses"],
        "distinct8": counts["distinct_correct_modes"] / counts["prompts"],
        "correct_pair_collision": (
            counts["colliding_correct_pairs"] / counts["correct_pairs"]
            if counts["correct_pairs"] else None
        ),
    }


def admitted_model_order(record: dict[str, Any]) -> tuple[str, ...]:
    """Keep known presentation order and include every additional admitted model."""
    admitted = [model["model"] for model in record["models"]]
    if len(admitted) != len(set(admitted)):
        raise ValueError("The admitted hosted deployments contain duplicate model identities")
    if not admitted or len(admitted) != record.get("completed_model_count"):
        raise ValueError("The admitted hosted deployments do not match completed_model_count")
    expected_responses = len(admitted) * len(DOMAIN_ORDER) * len(LEVELS) * PROMPTS * DRAWS
    if record.get("completed_response_count") != expected_responses:
        raise ValueError("The admitted hosted deployments do not match completed_response_count")
    return tuple(model for model in MODEL_ORDER if model in admitted) + tuple(
        model for model in admitted if model not in MODEL_ORDER
    )


def _original_cell(model_id: str, index: int, model: dict[str, Any],
                   domain: str, level: int) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
    """Read one complete original-prompt cell, with its own source binding."""
    key = f"level{level}/{domain}"
    source = model["normalized_secondary"]["cells"][key]
    if (source["complete_prompts"] != PROMPTS or source["expected_prompts"] != PROMPTS
            or source["missing_prompts"] != 0 or source["domain"] != domain
            or source["level"] != level):
        raise ValueError(f"{model_id} {key}: incomplete or misidentified original cohort")
    raw = source["counts"]
    counts = {
        "prompts": _integer(raw["prompts"], "prompts"),
        "responses": PROMPTS * DRAWS,
        "correct_responses": _integer(raw["correct_draws"], "correct_draws"),
        "distinct_correct_modes": _integer(raw["distinct_correct_modes"], "distinct_correct_modes"),
        "correct_pairs": _integer(raw["correct_pairs"], "correct_pairs"),
        "colliding_correct_pairs": _integer(raw["colliding_correct_pairs"], "colliding_correct_pairs"),
    }
    fields = f"/models/{index}/normalized_secondary/cells/{key.replace('/', '~1')}"
    source_names = {"accuracy": "pass1", "distinct8": "distinct8",
                    "correct_pair_collision": "correct_pair_collision"}
    supplied = {metric: source["metrics"][name]["estimate"] for metric, name in source_names.items()}
    provenance = {
        "condition": "original_benchmark_prompt",
        "cohort": model["run_directory"],
        "primary_samples": {"path": model["primary_samples_path"], "sha256": model["primary_samples_sha256"]},
        "summary_sha256": model["summary_sha256"],
        "completion_audit_sha256": model["completion_audit_sha256"],
        "source_cell": fields,
        "source_counts": deepcopy(raw),
        "source_metric_fields": {metric: f"{fields}/metrics/{name}/estimate"
                                 for metric, name in source_names.items()},
    }
    return counts, supplied, provenance


def build_display_data(record: dict[str, Any]) -> dict[str, Any]:
    """Select display cells without reading files, mutating data, or filtering draws."""
    pmd = pmd_cells()
    if record.get("schema") != "frontier-paper-comparison-v1":
        raise ValueError("Unexpected hosted comparison schema")
    if record.get("prompt_count_per_cell") != PROMPTS or record.get("draws_per_prompt") != DRAWS:
        raise ValueError("Hosted sampling denominator changed")
    model_order = admitted_model_order(record)
    indexed = {item["model"]: (index, item) for index, item in enumerate(record["models"])}
    sensitivity = record["python_prompt_sensitivity"]
    if sensitivity["model"] != "claude-opus-5":
        raise ValueError("The alternate prompt cohort belongs only to Claude Opus 5")
    conditions = sensitivity["conditions"]
    plain = conditions["plain"]
    if (plain["selected_domain"] != "python_factors"
            or plain["selected_prompts"] != PROMPTS * len(LEVELS)
            or plain["selected_responses"] != PROMPTS * DRAWS * len(LEVELS)
            or plain["source_cohort_responses"] != PROMPTS * DRAWS * len(LEVELS)):
        raise ValueError("The plain-prompt condition must retain the complete 3,072-response cohort")
    if plain["selected_rows_sha256"] != conditions["original"]["selected_rows_sha256"]:
        raise ValueError("The alternate prompt cohort must use the same task rows")
    if not sensitivity["validation"]["audited_primary_and_frozen_normalizer_hashes_authenticated"]:
        raise ValueError("The alternate cohort's frozen normalizer is not authenticated")
    graders = {
        (model["normalized_secondary"]["normalization_source_sha256"],
         model["normalized_secondary"]["frozen_grader_contract_sha256"])
        for _, model in indexed.values()
    }
    if len(graders) != 1:
        raise ValueError("All displayed deployments must use the same frozen normalized grading")
    normalizer_sha, grader_sha = next(iter(graders))
    expected_cells = {f"level{level}/{domain}" for level in LEVELS for domain in DOMAIN_ORDER}
    cells = []
    for model_id in model_order:
        index, model = indexed[model_id]
        normalized = model["normalized_secondary"]
        if set(normalized["cells"]) != expected_cells:
            raise ValueError(f"{model_id}: expected all five domains at all three levels")
        for domain in DOMAIN_ORDER:
            for level in LEVELS:
                key = f"level{level}/{domain}"
                alternative = model_id == "claude-opus-5" and domain == "python_factors"
                if alternative:
                    source = sensitivity["levels"][str(level)]["plain"]
                    raw = source["counts"]
                    counts = {
                        "prompts": _integer(raw["prompts"], "prompts"),
                        "responses": _integer(raw["responses"], "responses"),
                        "correct_responses": _integer(raw["normalized"]["correct_responses"], "correct_responses"),
                        "distinct_correct_modes": _integer(raw["normalized"]["distinct8"], "distinct8"),
                        "correct_pairs": _integer(raw["normalized"]["correct_pairs"], "correct_pairs"),
                        "colliding_correct_pairs": _integer(raw["normalized"]["colliding_correct_pairs"], "colliding_correct_pairs"),
                    }
                    fields = f"/python_prompt_sensitivity/levels/{level}/plain"
                    metric_fields = {
                        "accuracy": f"{fields}/metrics/normalized_accuracy",
                        "distinct8": f"{fields}/metrics/normalized_distinct8",
                        "correct_pair_collision": f"{fields}/metrics/normalized_correct_pair_collision",
                    }
                    supplied = {metric: source["metrics"][pointer.rsplit("/", 1)[-1]]
                                for metric, pointer in metric_fields.items()}
                    provenance = {
                        "condition": "plain_direct_expression_no_system",
                        "cohort": plain["directory"],
                        "condition_source": deepcopy(plain),
                        "paired_analysis": {"path": sensitivity["path"], "sha256": sensitivity["sha256"]},
                        "source_cell": fields,
                        "source_counts": deepcopy(raw),
                        "source_metric_fields": metric_fields,
                    }
                else:
                    counts, supplied, provenance = _original_cell(
                        model_id, index, model, domain, level)
                metrics = _metrics(counts)
                joined = pmd.get((model_id, domain, level))
                if joined and joined.get("reportable") and joined.get("pmd") is not None:
                    metrics["pmd"] = joined["pmd"]
                for metric, value in supplied.items():
                    _assert_metric(value, metrics[metric], f"{model_id} {key} {metric}")
                cells.append({
                    "model": model_id, "label": model["label"], "domain": domain, "level": level,
                    "grading": "frozen_formatting_normalized", "counts": counts, "metrics": metrics,
                    "provenance": provenance,
                })
    alternate_cells = [cell for cell in cells if cell["provenance"]["condition"] != "original_benchmark_prompt"]
    totals = sensitivity["totals"]["plain"]["counts"]
    for field in ("prompts", "responses"):
        if sum(cell["counts"][field] for cell in alternate_cells) != totals[field]:
            raise ValueError(f"Alternate prompt cohort {field} do not match its complete totals")
    for source_name, field in (("correct_responses", "correct_responses"),
                               ("distinct8", "distinct_correct_modes"),
                               ("correct_pairs", "correct_pairs"),
                               ("colliding_correct_pairs", "colliding_correct_pairs")):
        if sum(cell["counts"][field] for cell in alternate_cells) != totals["normalized"][source_name]:
            raise ValueError(f"Alternate prompt cohort {field} do not match its complete totals")
    # The substituted cells are the only place this figure departs from one
    # protocol, so the condition they replace is carried alongside them rather
    # than left to the appendix: every displayed point has its counterpart here.
    companion_cells = []
    for cell in alternate_cells:
        index, model = indexed[cell["model"]]
        counts, supplied, provenance = _original_cell(
            cell["model"], index, model, cell["domain"], cell["level"])
        metrics = _metrics(counts)
        for metric, value in supplied.items():
            _assert_metric(value, metrics[metric],
                           f"{cell['model']} companion level{cell['level']}/{cell['domain']} {metric}")
        companion_cells.append({
            "model": cell["model"], "label": cell["label"], "domain": cell["domain"],
            "level": cell["level"], "role": "replaced_condition",
            "grading": "frozen_formatting_normalized", "counts": counts,
            "metrics": metrics, "provenance": provenance,
        })
    if len(companion_cells) != len(alternate_cells):
        raise ValueError("Every substituted cell requires the condition it replaced")

    icons = record.get("model_icons", {})
    if not set(model_order).issubset(icons):
        raise ValueError("Every displayed deployment requires its source-bound model icon")
    return {
        "icons": {model: deepcopy(icons[model]) for model in model_order},
        "models": [{"model": model_id, "label": indexed[model_id][1]["label"]} for model_id in model_order],
        "domains": list(DOMAIN_ORDER), "levels": list(LEVELS),
        "grading": {"name": "frozen_formatting_normalized", "normalization_source_sha256": normalizer_sha,
                    "frozen_grader_contract_sha256": grader_sha,
                    "scope": "Applied to every displayed cell; strict original results retained in appendix."},
        "sampling": {"prompts_per_cell": PROMPTS, "draws_per_prompt": DRAWS,
                     "cell_count": len(cells), "displayed_responses": sum(cell["counts"]["responses"] for cell in cells),
                     "selection": "Complete cohorts; no response or prompt filtered by outcome.",
                     "alternate_cohort_responses": sum(cell["counts"]["responses"] for cell in alternate_cells)},
        "condition_disclosure": "Claude Opus 5 Python uses a separately collected direct-expression prompt without a system message; all other cells use the original benchmark prompt.",
        "companion_disclosure": "Every substituted cell is shown beside the original-prompt cell it replaces; companion cells are display marks only and enter no average.",
        "limits": ["This descriptive mixed-condition display is not a controlled comparison of prompting or deployment rankings.",
                   "Distinct@8 averages over all prompts, including those with no correct draw.",
                   "Benchmark levels contain different task populations; no monotonic difficulty claim is made.",
                   "Static concentration does not establish training-induced collapse."],
        "cells": cells,
        "companion_cells": companion_cells,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def build_record(source_path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source_path = Path(source_path)
    return {
        "schema": "hosted-verified-breadth-figure-v1",
        "source": {"path": _relative(source_path), "sha256": _sha256(source_path)},
        "renderer": {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "display": build_display_data(json.loads(source_path.read_text(encoding="utf-8"))),
        "figure": {"size_inches": list(FIGSIZE), "minimum_font_points": 8,
                   "layout": "Two metric rows by five domain columns; one row per admitted deployment, with three level marks each.",
                   "accuracy_axis_percent": [0, 100], "pmd_axis": [0, 1],
                   "marks": "Point estimates; source prompt-bootstrap intervals are reported in the appendix.",
                   "companion_marks": "Hollow marks repeat the original-prompt cell that each substituted cell replaces.",
                   "level_colors": list(LEVEL_COLORS), "level_markers": list(LEVEL_MARKERS)},
    }


def build_figure(display: dict[str, Any]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    cells = {(cell["model"], cell["domain"], cell["level"]): cell for cell in display["cells"]}
    companions = {(cell["model"], cell["domain"], cell["level"]): cell
                  for cell in display["companion_cells"]}
    model_order = tuple(model["model"] for model in display["models"])
    rc = {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
          "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
          "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": "#19324A",
          "axes.labelcolor": "#19324A", "xtick.color": "#607487", "ytick.color": "#19324A"}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 5, figsize=FIGSIZE)
        fig.subplots_adjust(left=.202, right=.967, bottom=.072, top=.82, wspace=.24, hspace=.64)
        labels = [model["label"].replace("Claude ", "") for model in display["models"]]
        for row, metric in enumerate(("accuracy", "pmd")):
            for column, domain in enumerate(DOMAIN_ORDER):
                ax = axes[row, column]
                for model_index in range(len(model_order)):
                    if model_index % 2 == 0:
                        ax.axhspan(model_index-.48, model_index+.48, color="#F1F5F8", zorder=0)
                for level, color, marker, offset in zip(LEVELS, LEVEL_COLORS, LEVEL_MARKERS, (-.25, 0, .25)):
                    # A cell with no \pmd{} is one whose successes are too rare
                    # to define it; it is left off the row rather than drawn at
                    # zero, so its position carries no reading either way.
                    drawn = [(index, cells[(model, domain, level)]["metrics"].get(metric))
                             for index, model in enumerate(model_order)]
                    drawn = [(index, value) for index, value in drawn if value is not None]
                    if metric == "accuracy":
                        drawn = [(index, 100 * value) for index, value in drawn]
                    # The replaced condition is drawn first and hollow, so a
                    # substituted point never hides the one it stands in for.
                    replaced = [(index, companions[(model, domain, level)]["metrics"][metric])
                                for index, model in enumerate(model_order)
                                if (model, domain, level) in companions
                                and companions[(model, domain, level)]["metrics"].get(metric) is not None]
                    if replaced:
                        scale = 100 if metric == "accuracy" else 1
                        ax.plot([scale * value for _, value in replaced],
                                [index + offset for index, _ in replaced], linestyle="none",
                                marker=marker, markerfacecolor="white", markeredgecolor=color,
                                markersize=3.4, markeredgewidth=.7, clip_on=False, zorder=2)
                    ax.plot([value for _, value in drawn],
                            [index + offset for index, _ in drawn], linestyle="none",
                            marker=marker, color=color, markersize=3.4, markeredgewidth=0,
                            clip_on=False, zorder=3)
                ax.set_ylim(len(model_order) - .45, -.55)
                ax.set_yticks(range(len(model_order)), labels if column == 0 else [])
                if column == 0:
                    for model_index, model in enumerate(model_order):
                        binding = display["icons"][model]
                        icon_path = ROOT / binding["path"]
                        if _sha256(icon_path) != binding["sha256"]:
                            raise ValueError(f"{model}: model icon differs from its source binding")
                        pixels = plt.imread(icon_path)
                        mark = OffsetImage(pixels, zoom=8 / max(pixels.shape[:2]))
                        ax.add_artist(AnnotationBbox(mark, (.025, model_index),
                                      xycoords=("figure fraction", "data"),
                                      frameon=False, pad=0, box_alignment=(.5, .5),
                                      annotation_clip=False))
                ax.tick_params(axis="y", length=0, pad=7)
                ax.tick_params(axis="x", length=2.5, width=.5, pad=2)
                ax.set_xlim(0, 100 if metric == "accuracy" else 1)
                ax.set_xticks((0, 50, 100) if metric == "accuracy" else (0, .5, 1))
                ax.grid(axis="x", color="#D8E2EA", linewidth=.5, zorder=1)
                for side in ("top", "right", "left"):
                    ax.spines[side].set_visible(False)
                ax.spines["bottom"].set_color("#AABAC7")
                ax.spines["bottom"].set_linewidth(.5)
                if row == 0:
                    ax.set_title(DOMAIN_LABELS[column], pad=7, fontweight="bold")
        fig.text(.018, .985, "(a) Accuracy (%)", va="top", fontsize=9, fontweight="bold")
        fig.text(.018, .432, "(b) Diversity over successes (PCMD)", va="bottom", fontsize=9, fontweight="bold")
        handles = [Line2D([], [], linestyle="none", marker=marker, color=color,
                          markersize=4, markeredgewidth=0, label=f"Level {level}")
                   for level, color, marker in zip(LEVELS, LEVEL_COLORS, LEVEL_MARKERS)]
        if companions:
            replaced_domains = sorted({DOMAIN_LABELS[DOMAIN_ORDER.index(domain)]
                                       for _, domain, _ in companions})
            replaced_labels = sorted({cell["label"].replace("Claude ", "")
                                      for cell in companions.values()})
            handles.append(Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                                  markeredgecolor="#607487", markersize=4, markeredgewidth=.7,
                                  label=", ".join(replaced_labels) + " "
                                        + "/".join(replaced_domains) + ", original prompt"))
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.99, 1.006),
                   ncol=len(handles), frameon=False, fontsize=8, handletextpad=.3,
                   columnspacing=1.0)
    return fig


def render(record: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(record["display"])
    try:
        from ops.paper_domain_figure_typography import apply_domain_typography
    except ModuleNotFoundError:
        from paper_domain_figure_typography import apply_domain_typography
    apply_domain_typography(fig)
    outputs = {}
    for suffix in (".pdf", ".png"):
        path = output.with_suffix(suffix)
        kwargs = {"dpi": 220} if suffix == ".png" else {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[suffix[1:]] = {"path": _relative(path), "sha256": _sha256(path)}
    plt.close(fig)
    result = deepcopy(record)
    result["outputs"] = outputs
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = render(build_record(args.source), args.output)
    print(json.dumps({"figure": str(args.output), "cells": len(result["display"]["cells"]),
                      "responses": result["display"]["sampling"]["displayed_responses"]}))


if __name__ == "__main__":
    main()
