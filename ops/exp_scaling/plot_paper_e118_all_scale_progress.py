#!/usr/bin/env python3
"""Render three-model replay evidence with audited paired-seed coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_style as style  # noqa: E402
from exp_scaling.build_paper_core_terminal_endpoints import sampled_endpoint

ENDPOINT_AUDIT: list[dict] = []

LEDGER = ROOT / "var/artifacts/e118_all_scales_maxrl_verified_replay_jobs.json"
BASE = ROOT / "paper/results/core_terminal_endpoints.json"
PRECHECK = ROOT / "paper/results/baseline_collapse_precheck.json"
PMD_SOURCE = ROOT / "paper/results/mode_diversity_training.json"
TRAJECTORY = ROOT / "paper/figures/direct_baseline_learning_curves_static_strip.json"
OUT = ROOT / "paper/figures/e118_all_scale_factorial_progress"
APPENDIX_OUT = ROOT / "paper/figures/e118_scale_extensions_appendix"
MODELS = (
    ("qwen05b", "Qwen2.5-0.5B", (43, 44, 45, 46, 47)),
    ("falcon1b", "Falcon3-1B", (55, 56, 57, 58, 59)),
    ("qwen3b", "Qwen2.5-3B", (70, 71, 72, 73, 74)),
)
DOMAINS = (
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan",
)
LABELS = ("Graph", "Countdown", "Python", "MathIR", "Pantry")
DOMAIN_SHORT = dict(zip(DOMAINS, LABELS))
DOMAIN_WASH = {
    "graph_coloring": "#E8F1FA", "countdown": "#E7FBF6",
    "python_factors": "#EFFBE7", "mathir": "#E7FBEE",
    "pantry_plan": "#E7ECFB",
}
METHODS = {
    "before_training": ("Untrained", "#6B7280", "D", "#6B7280"),
    "drgrpo": ("Dr.GRPO", style.CONTROL, "o", "none"),
    "replay_drgrpo": ("Re:Dr (ours)", style.ADAPTIVE, "o", style.ADAPTIVE),
    "maxrl": ("MaxRL", style.COMPARATOR, "s", "none"),
    "replay_maxrl": ("Re:Max (ours)", style.ABLATION, "s", style.ABLATION),
}
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_lines(path: Path, max_lines: int = 96, chunk_bytes: int = 1_048_576):
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks = []
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
    values = sampled_endpoint(Path(run["run_dir"]), step=3072, audit=ENDPOINT_AUDIT)
    return {metric: values[metric] for metric in ("pass8", "distinct8")} if values else None


def before_values(
    trajectory: dict, scale: str, domain: str, metric: str, seeds: list[int],
) -> list[float]:
    source = trajectory["cells"][f"{scale}/{domain}"]["methods"]["drgrpo"]
    per_seed = source["summaries"]["0"][metric]["per_seed"]
    return [float(per_seed[str(seed)]) for seed in seeds]


def initial_method(start: str) -> str:
    """Give each track an initial reference with its own paired seeds."""
    return "before_training_drgrpo" if start == "drgrpo" else "before_training"


def attach_reference_methods(
    cell: dict, *, base: dict, trajectory: dict, scale: str,
    model: str, domain: str,
) -> None:
    """Keep the MaxRL intersection and separately match the Dr.GRPO track."""
    common = cell["matched_seeds"]
    sources = base["models"][model]["domains"][domain]["methods"]
    dr_seeds = [
        seed for seed in common
        if all(str(seed) in sources[arm]["per_seed"] for arm in ("control", "replay"))
    ]
    cell["method_seeds"] = {
        "maxrl": list(common), "replay_maxrl": list(common),
        "before_training": list(common),
        "drgrpo": dr_seeds, "replay_drgrpo": dr_seeds,
        "before_training_drgrpo": dr_seeds,
    }
    for method, arm in (("drgrpo", "control"), ("replay_drgrpo", "replay")):
        cell["methods"][method] = {
            metric: [
                float(sources[arm]["per_seed"][str(seed)][metric])
                for seed in dr_seeds
            ]
            for metric in ("pass8", "distinct8")
        }
    if scale != "qwen3b":
        for method in ("before_training", "before_training_drgrpo"):
            cell["methods"][method] = {
                metric: before_values(
                    trajectory, scale, domain, metric, cell["method_seeds"][method],
                )
                for metric in ("pass8", "distinct8")
            }


MAIN_FIGURE_DESCRIPTION = (
    "three-model cross-domain pass@8 and PCMD with MaxRL/replay and "
    "Dr.GRPO/replay tracks; Qwen2.5-3B MaxRL uses descriptive equal-domain means "
    "of available within-domain paired seeds"
)
QWEN3B_DESCRIPTION = (
    "complete five-seed E80-R1 Dr.GRPO/replay cross-domain track and descriptive "
    "MaxRL/replay equal-domain means in the main model row; MaxRL pairs differ "
    "by domain, with exact domain counts and no pooled seed paths or intervals; "
    "all domains and both metrics in appendix panels"
)
AVAILABLE_DOMAIN_DEFINITION = "equal domain average of domain-specific paired-seed means"
PAIRED_DOMAIN_DEFINITION = "equal domain average within paired seed"


def complete_maxrl_scale(record: dict, scale: str) -> bool:
    """Require the registered cohort in every domain and both MaxRL arms."""
    expected = next(set(seeds) for key, _model, seeds in MODELS if key == scale)
    cells = record["cells"].get(scale, {})
    return all(
        set(cells.get(domain, {}).get("matched_seeds", [])) == expected
        and all(set(cells[domain].get("method_seeds", {}).get(method, [])) == expected
                for method in ("maxrl", "replay_maxrl"))
        for domain in DOMAINS
    )


def qwen3b_display_descriptions(record: dict) -> dict:
    if complete_maxrl_scale(record, "qwen3b"):
        return {
            "main_figure": (
                "three-model cross-domain pass@8 and PCMD with MaxRL/replay and "
                "Dr.GRPO/replay tracks; Qwen2.5-3B MaxRL uses five matched seeds "
                "across all five domains"
            ),
            "qwen3b_display": (
                "complete five-seed MaxRL/replay and E80-R1 Dr.GRPO/replay tracks; "
                "equal domain averages within each paired seed, drawn as marks without seed paths; "
                "all domains and both metrics in appendix panels"
            ),
        }
    return {"main_figure": MAIN_FIGURE_DESCRIPTION, "qwen3b_display": QWEN3B_DESCRIPTION}


def attach_complete_qwen3b_reference_track(
    record: dict, *, base: dict, precheck: dict,
) -> None:
    """Attach the completed E80-R1 track independently of E118 MaxRL progress."""
    seeds = [70, 71, 72, 73, 74]
    expected = {str(seed) for seed in seeds}
    domains = base["models"]["Qwen2.5-3B"]["domains"]
    initial = precheck["arms"]["drgrpo"]["scales"]["qwen3b"]
    if initial.get("target_step") != 3072:
        raise RuntimeError("Qwen3B initial-reference horizon drifted")
    for domain in DOMAINS:
        sources = domains[domain]["methods"]
        origin = initial["domains"][domain]["per_seed"]
        if set(origin) != expected or any(
            set(sources[arm]["per_seed"]) != expected for arm in ("control", "replay")
        ):
            raise RuntimeError(f"Qwen3B reference track requires five paired seeds: {domain}")
        cell = record["cells"]["qwen3b"][domain]
        for method, arm in (("drgrpo", "control"), ("replay_drgrpo", "replay")):
            cell["method_seeds"][method] = list(seeds)
            cell["methods"][method] = {
                metric: [float(sources[arm]["per_seed"][str(seed)][metric]) for seed in seeds]
                for metric in ("pass8", "distinct8")
            }
        cell["method_seeds"]["before_training_drgrpo"] = list(seeds)
        cell["methods"]["before_training_drgrpo"] = {
            metric: [float(origin[str(seed)]["pass0"][metric]) for seed in seeds]
            for metric in ("pass8", "distinct8")
        }
        maxrl_seeds = cell["method_seeds"]["maxrl"]
        cell["method_seeds"]["before_training"] = list(maxrl_seeds)
        cell["methods"]["before_training"] = {
            metric: [float(origin[str(seed)]["pass0"][metric]) for seed in maxrl_seeds]
            for metric in ("pass8", "distinct8")
        }
        for seed in seeds:
            for metric in ("pass8", "distinct8"):
                if origin[str(seed)]["pass8"][metric] != sources["control"]["per_seed"][str(seed)][metric]:
                    raise RuntimeError(f"Qwen3B initial/control reference mismatch: {domain}/{seed}")
        if any(
            not math.isfinite(value)
            for method in ("before_training_drgrpo", "drgrpo", "replay_drgrpo")
            for values in cell["methods"][method].values() for value in values
        ):
            raise RuntimeError(f"non-finite Qwen3B reference value: {domain}")


def add_qwen3b_main_figure_track(record: dict) -> None:
    """Extend a frozen figure record without refreshing ongoing E118 endpoints."""
    base = json.loads(BASE.read_text())
    precheck = json.loads(PRECHECK.read_text())
    source = base["sources"]["Qwen2.5-3B"]
    ledger = ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"
    registered = json.loads(ledger.read_text())
    if (
        Path(source["path"]).resolve() != ledger.resolve()
        or sha(ledger) != source["sha256"]
        or precheck["ledgers"][ledger.name]["sha256"] != source["sha256"]
        or registered["seeds"] != [70, 71, 72, 73, 74]
        or len(registered["runs"]) != 50
    ):
        raise RuntimeError("Qwen3B reference provenance does not match E80-R1")
    audits = [row for row in base["endpoint_audit"] if row.get("model") == "Qwen2.5-3B"]
    if len(audits) != 50 or any(
        row.get("status") != "admitted" or row.get("step") != 3072
        or row.get("observed_draws") != [0, 1, 2, 3]
        or row.get("conflicting_retry_selected")
        for row in audits
    ):
        raise RuntimeError("Qwen3B reference track lacks 50 admitted four-draw endpoints")
    attach_complete_qwen3b_reference_track(record, base=base, precheck=precheck)
    record["main_figure_scales"] = [scale for scale, _label, _seeds in MODELS]
    record["main_figure_tracks"] = {
        scale: ["maxrl", "drgrpo"] for scale, _label, _seeds in MODELS
    }
    record["main_cross_domain_tracks"] = {
        scale: ["maxrl", "drgrpo"] for scale, _label, _seeds in MODELS
    }
    record["main_domain_tracks"] = {}
    record["incomplete_scales"] = [
        scale for scale, _model, _seeds in MODELS
        if not complete_maxrl_scale(record, scale)
    ]
    record["source_sha256"][str(PRECHECK.relative_to(ROOT))] = sha(PRECHECK)
    record["qwen3b_reference_source"] = {
        "ledger": str(ledger.relative_to(ROOT)), "ledger_sha256": sha(ledger),
        "terminal_endpoints": str(BASE.relative_to(ROOT)),
        "initial_reference": str(PRECHECK.relative_to(ROOT)),
        "admitted_terminal_cells": 50, "seeds": [70, 71, 72, 73, 74],
        "aggregate_status": "post-hoc descriptive; no pooled inference",
    }
    if "display_contract" in record:
        record["display_contract"].update(qwen3b_display_descriptions(record))


def absolute_cross_domain_averages(record: dict) -> dict:
    """Average domains within the common seed set of each matched track."""
    averages = {}
    methods = (
        "before_training", "before_training_drgrpo", "drgrpo",
        "replay_drgrpo", "maxrl", "replay_maxrl",
    )
    for scale, _model, _seeds in MODELS:
        if scale not in record["cells"]:
            continue
        cells = record["cells"][scale]
        averages[scale] = {metric: {} for metric in ("pass8", "distinct8")}
        scale_methods = (("before_training_drgrpo", "drgrpo", "replay_drgrpo")
                         if scale == "qwen3b" and not complete_maxrl_scale(record, scale)
                         else methods)
        for method in scale_methods:
            seeds = sorted(set.intersection(*(
                set(cells[domain]["method_seeds"][method]) for domain in DOMAINS
            )))
            if not seeds:
                raise RuntimeError(f"no common cross-domain seeds for {scale}/{method}")
            for metric in ("pass8", "distinct8"):
                domain_values = {
                    domain: dict(zip(
                        cells[domain]["method_seeds"][method],
                        cells[domain]["methods"][method][metric], strict=True,
                    ))
                    for domain in DOMAINS
                }
                per_seed = {
                    str(seed): statistics.fmean(
                        domain_values[domain][seed] for domain in DOMAINS
                    )
                    for seed in seeds
                }
                averages[scale][metric][method] = {
                    "mean": statistics.fmean(per_seed.values()),
                    "per_seed": per_seed, "seeds": seeds, "n": len(seeds),
                    "definition": "equal domain average within paired seed",
                    "status": (
                        "post-hoc descriptive cross-domain average" if len(seeds) == 5
                        else "descriptive paired prefix; no uncertainty interval"
                    ),
                }
    return averages


def descriptive_available_domain_averages(record: dict) -> dict:
    """Balance domains while retaining each domain's own valid MaxRL pairs.

    This descriptive estimand has no shared cross-domain seed observations.
    Keep it separate from the common-seed macro averages and never attach a
    seed-level standard error, confidence interval, or inferred missing value.
    """
    scale = "qwen3b"
    if complete_maxrl_scale(record, scale):
        return {}
    cells = record["cells"].get(scale, {})
    if any(not cells.get(domain, {}).get("matched_seeds") for domain in DOMAINS):
        return {}
    methods = ("before_training", "maxrl", "replay_maxrl")
    output = {scale: {metric: {} for metric in ("pass8", "distinct8")}}
    for method in methods:
        for metric in ("pass8", "distinct8"):
            per_domain = {}
            for domain in DOMAINS:
                cell = cells[domain]
                seeds = cell["method_seeds"][method]
                if seeds != cell["matched_seeds"]:
                    raise RuntimeError(f"unmatched available-domain reference: {domain}/{method}")
                values = cell["methods"][method][metric]
                if any(not math.isfinite(value) for value in values):
                    raise RuntimeError(f"non-finite available-domain value: {domain}/{method}")
                per_domain[domain] = dict(zip(map(str, seeds), values, strict=True))
            domain_means = {
                domain: statistics.fmean(values.values())
                for domain, values in per_domain.items()
            }
            output[scale][metric][method] = {
                "mean": statistics.fmean(domain_means.values()),
                "n_domains": len(DOMAINS),
                "domain_means": domain_means,
                "per_domain_per_seed": per_domain,
                "domain_seed_counts": {
                    domain: len(values) for domain, values in per_domain.items()
                },
                "domain_weights": {domain: 1 / len(DOMAINS) for domain in DOMAINS},
                "definition": AVAILABLE_DOMAIN_DEFINITION,
                "status": "descriptive available-pair domain mean; no cross-domain seed inference",
                "uncertainty": "none; domain-specific seed sets are not pooled",
            }
    return output


PMD_DEFINITION = "equal domain average of domain-specific support-eligible seed means"
PMD_METHODS = (
    "before_training", "before_training_drgrpo", "drgrpo",
    "replay_drgrpo", "maxrl", "replay_maxrl",
)
PMD_START = {"before_training": "before", "before_training_drgrpo": "before"}


def pmd_domain_averages() -> dict:
    """Average PCMD over the domains where every arm clears the support bar.

    PCMD is undefined for a seed whose policy succeeds on too few prompts, so a
    common cross-domain seed cohort does not exist at every scale. Each domain
    therefore contributes the mean over its own support-eligible seeds, and the
    per-domain seed counts travel with the aggregate so a thin arm is visible
    rather than implied.
    """
    payload = json.loads(PMD_SOURCE.read_text(encoding="utf-8"))
    rows: dict[tuple[str, str, str], dict[int, dict]] = {}
    for row in payload["seeds"]:
        if row["level"] != "level1":
            continue
        rows.setdefault((row["scale"], row["domain"], row["method"]), {})[row["seed"]] = row

    output: dict[str, dict] = {}
    for scale, _model, _seeds in MODELS:
        arms = ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
        # Every marker in a row has to describe the same domains, so the
        # untrained reference must also be measurable, not just the two arms.
        eligible = [
            domain for domain in DOMAINS
            if all(
                any(entry["after"]["reportable"]
                    for entry in rows.get((scale, domain, arm), {}).values())
                for arm in arms
            ) and any(
                entry["before"]["reportable"] and entry["before"]["pmd"] is not None
                for entry in rows.get((scale, domain, "drgrpo"), {}).values()
            )
        ]
        if not eligible:
            continue
        scale_output: dict[str, dict] = {}
        for method in PMD_METHODS:
            checkpoint = PMD_START.get(method, "after")
            source_method = "drgrpo" if method.startswith("before_training") else method
            per_domain: dict[str, dict[str, float]] = {}
            for domain in eligible:
                values = {
                    str(seed): float(entry[checkpoint]["pmd"])
                    for seed, entry in sorted(rows.get((scale, domain, source_method), {}).items())
                    if entry[checkpoint]["reportable"] and entry[checkpoint]["pmd"] is not None
                }
                if values:
                    per_domain[domain] = values
            if len(per_domain) != len(eligible):
                continue
            domain_means = {
                domain: statistics.fmean(values.values())
                for domain, values in per_domain.items()
            }
            scale_output[method] = {
                "mean": statistics.fmean(domain_means.values()),
                "n_domains": len(eligible),
                "domains": list(eligible),
                "domain_means": domain_means,
                "per_domain_per_seed": per_domain,
                "domain_seed_counts": {
                    domain: len(values) for domain, values in per_domain.items()
                },
                "domain_weights": {domain: 1 / len(eligible) for domain in eligible},
                "definition": PMD_DEFINITION,
                "status": "descriptive support-eligible domain mean; no cross-domain seed inference",
                "uncertainty": "none; domain-specific eligible seed sets are not pooled",
            }
        if scale_output:
            output[scale] = {"pmd": scale_output}
    return output


def attach_pmd_cells(record: dict) -> None:
    """Add per-seed PCMD to every domain cell, on its own seed list.

    The appendix panels used to read distinct@8 as breadth, which moves with
    correctness -- the confound PCMD exists to remove. PCMD is not simply a
    third metric on the same rows, though: a seed whose policy succeeds on too
    few prompts reports no PCMD at all, so the eligible seeds differ per arm
    and per domain. They travel as ``method_seeds_pmd`` rather than reusing
    ``method_seeds``, so an arm thinned by the support bar prints its own n
    instead of inheriting the pass@8 pairing's.
    """
    rows: dict[tuple[str, str, str], dict[int, dict]] = {}
    for row in json.loads(PMD_SOURCE.read_text(encoding="utf-8"))["seeds"]:
        if row["level"] == "level1":
            rows.setdefault((row["scale"], row["domain"], row["method"]), {})[row["seed"]] = row
    for scale, cells in record["cells"].items():
        for domain, cell in cells.items():
            seeds_by_method = cell.setdefault("method_seeds_pmd", {})
            for method in PMD_METHODS:
                checkpoint = PMD_START.get(method, "after")
                source = "drgrpo" if method.startswith("before_training") else method
                entries = sorted(rows.get((scale, domain, source), {}).items())
                pairs = [(seed, float(entry[checkpoint]["pmd"]))
                         for seed, entry in entries
                         if entry[checkpoint]["reportable"]
                         and entry[checkpoint]["pmd"] is not None]
                # An arm with no eligible seed gets no PCMD key at all, so the
                # drawer skips its track instead of plotting an invented zero.
                if not pairs:
                    continue
                if method in cell["methods"]:
                    cell["methods"][method]["pmd"] = [value for _seed, value in pairs]
                    seeds_by_method[method] = [seed for seed, _value in pairs]


PAIR_TRACKS = (
    (
        "maxrl", "replay_maxrl", 0.18, "s",
        style.COMPARATOR, style.ABLATION,
    ),
    (
        "drgrpo", "replay_drgrpo", -0.18, "o",
        style.CONTROL, style.ADAPTIVE,
    ),
)


def pair_legend_handles(*, include_untrained: bool = False,
                        scale: float = 1.0) -> list[Line2D]:
    handles = []
    if include_untrained:
        handles.append(
            Line2D(
                [0], [0], marker="D", linestyle="none", markersize=4.0 * scale,
                markerfacecolor="#6B7280", markeredgecolor="#6B7280",
                label="Untrained",
            )
        )
    handles.extend([
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=4.3 * scale,
            markerfacecolor=style.WHITE, markeredgecolor=style.COMPARATOR,
            markeredgewidth=1.2, label="MaxRL",
        ),
        Line2D(
            [0], [0], marker="s", linestyle="none", markersize=4.5 * scale,
            markerfacecolor=style.ABLATION, markeredgecolor=style.ABLATION,
            label="Re:Max (ours)",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.5 * scale,
            markerfacecolor=style.WHITE, markeredgecolor=style.CONTROL,
            markeredgewidth=1.2, label="Dr.GRPO",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=4.7 * scale,
            markerfacecolor=style.ADAPTIVE, markeredgecolor=style.ADAPTIVE,
            label="Re:Dr (ours)",
        ),
    ])
    return handles


def draw_pair_panel(
    axis,
    *,
    record: dict,
    absolute_averages: dict,
    scale: str,
    metric: str,
    title: str,
    rows: tuple[tuple[str, str], ...],
    xlim: tuple[float, float],
    show_untrained: bool = False,
    tracks: tuple = PAIR_TRACKS,
    counts_in_labels: bool = False,
) -> None:
    positions = {
        key: len(rows) - 1 - index
        for index, (key, _label) in enumerate(rows)
    }
    axis.set_facecolor(style.WHITE)
    axis.set_title(title, loc="left", fontsize=9.2, fontweight="bold", pad=5)
    axis.set_xlim(*xlim)
    axis.set_ylim(-0.48, len(rows) - 0.52)
    axis.grid(axis="x", color=style.GRID, linewidth=0.65, zorder=1)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=7.7)
    axis.tick_params(axis="y", length=0, pad=6)

    for key, _label in rows:
        middle = positions[key]
        wash = "#F3F5F7" if key == "average" else DOMAIN_WASH[key]
        axis.axhspan(
            middle - 0.43, middle + 0.43,
            facecolor=wash, edgecolor="none", zorder=0,
        )
        if key == "average":
            values = {
                method: absolute_averages[scale][metric][method]["mean"]
                for method in (
                    (("before_training", "before_training_drgrpo") if show_untrained else ())
                    + ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
                )
                if method in absolute_averages[scale][metric]
            }
        else:
            cell = record["cells"][scale][key]
            values = {}
            for method in (
                (("before_training", "before_training_drgrpo") if show_untrained else ())
                + ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
            ):
                samples = cell["methods"].get(method, {}).get(metric, [])
                if samples:
                    values[method] = statistics.fmean(samples)

        drawn_tracks = 0
        for start, finish, offset, marker, start_color, finish_color in tracks:
            if start not in values or finish not in values:
                continue
            drawn_tracks += 1
            if key == "average":
                seeds = absolute_averages[scale][metric][start]["seeds"]
            elif metric == "pmd":
                # PCMD's eligible seeds are its own; see attach_pmd_cells.
                seeds = cell["method_seeds_pmd"][start]
            else:
                seeds = cell["method_seeds"][start]
            complete = len(seeds) == 5
            initial = initial_method(start)
            y = middle + (offset if len(tracks) > 1 else 0.0)
            line_style = "-" if complete else (0, (2.5, 1.8))
            if not complete and not counts_in_labels:
                axis.text(
                    xlim[1] - 0.015 * (xlim[1] - xlim[0]), y + 0.065,
                    f"n={len(seeds)}", fontsize=6.3, color=start_color,
                    ha="right", va="bottom", zorder=6,
                )
            if show_untrained and initial in values:
                axis.plot(
                    [values[initial], values[start]], [y, y],
                    color="#6B7280", linewidth=1.0, linestyle=line_style,
                    solid_capstyle="round", zorder=2,
                )
                axis.plot(
                    values[initial], y, marker="D", linestyle="none",
                    markersize=4.5, markerfacecolor="#6B7280",
                    markeredgecolor="#6B7280", markeredgewidth=0.9, zorder=4,
                )
            axis.annotate(
                "",
                xy=(values[finish], y),
                xytext=(values[start], y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": start_color,
                    "linewidth": 1.25,
                    "linestyle": line_style,
                    "mutation_scale": 7.5,
                    "shrinkA": 3.5,
                    "shrinkB": 4.5,
                },
                zorder=2,
            )
            axis.plot(
                values[start], y, marker=marker, linestyle="none",
                markersize=5.0, markerfacecolor=style.WHITE,
                markeredgecolor=start_color, markeredgewidth=1.2, zorder=3,
            )
            axis.plot(
                values[finish], y, marker=marker, linestyle="none",
                markersize=5.3, markerfacecolor=finish_color,
                markeredgecolor=finish_color, markeredgewidth=1.0, zorder=4,
            )
        # A row where neither pair could be drawn is a statement, not a hole in
        # the plate: one side of every pair succeeds on too few prompts for
        # PCMD to be defined. Saying so is the difference between a measurement
        # that does not exist and one that came out at zero.
        if not drawn_tracks:
            axis.text(
                0.5, middle, "below the support bar",
                transform=axis.get_yaxis_transform(), fontsize=6.6,
                color=style.MUTED, ha="center", va="center", style="italic",
                zorder=6,
            )

    axis.set_yticks(
        [positions[key] for key, _label in rows],
        [
            f"{label} (n={len(record['cells'][scale][key]['matched_seeds'])})"
            if counts_in_labels else label
            for key, label in rows
        ],
        fontsize=7.9,
    )
    if rows and rows[0][0] == "average":
        tick_labels = axis.get_yticklabels()
        if tick_labels:
            tick_labels[0].set_fontweight("bold")
    axis.set_xlabel(
        {"pass8": "pass@8", "distinct8": "distinct@8 (verified modes)",
         "pmd": "PCMD"}[metric],
        fontsize=8.2, labelpad=3,
    )


def cross_domain_seed_values(
    absolute_averages: dict,
    scale: str,
    metric: str,
    method: str,
    seeds: tuple[int, ...],
) -> list[float]:
    source = absolute_averages[scale][metric][method]
    if source.get("definition") != "equal domain average within paired seed":
        raise RuntimeError(f"non-comparable average for {scale}/{metric}/{method}")
    per_seed = source.get("per_seed", {})
    if set(per_seed) != {str(seed) for seed in seeds}:
        raise RuntimeError(f"incomplete seed set for {scale}/{metric}/{method}")
    return [float(per_seed[str(seed)]) for seed in seeds]


def draw_cross_domain_panel(
    axis,
    *,
    absolute_averages: dict,
    metric: str,
    title: str,
    xlim: tuple[float, float],
    descriptive_averages: dict | None = None,
) -> None:
    axis.set_facecolor(style.WHITE)
    axis.set_title(title, loc="left", fontsize=7.9, fontweight="bold", pad=4)
    axis.set_xlim(*xlim)
    completed = tuple(model for model in MODELS if model[0] in absolute_averages)
    positions = tuple(float(index) for index in reversed(range(len(completed))))
    axis.set_ylim(-0.48, len(completed) - 0.52)
    axis.grid(axis="x", color=style.GRID, linewidth=0.65, zorder=1)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="x", labelsize=7.7)
    axis.tick_params(axis="y", length=0, pad=7)
    for index, row in enumerate(positions):
        axis.axhspan(row - 0.45, row + 0.45, color=("#F3F6F8", "#FAFBFC")[index % 2], zorder=0)

    for row, (scale, _label, _registered_seeds) in zip(positions, completed):
        descriptive = (descriptive_averages or {}).get(scale, {}).get(metric, {})
        regular = absolute_averages[scale][metric]
        tracks = tuple(track for track in PAIR_TRACKS
                       if track[0] in regular or track[0] in descriptive)
        for start, finish, offset, marker, start_color, finish_color in tracks:
            y = row + (offset if len(tracks) > 1 else 0.0)
            if start in descriptive:
                source = descriptive[start]
                if source["definition"] not in (AVAILABLE_DOMAIN_DEFINITION, PMD_DEFINITION):
                    raise RuntimeError("unsupported descriptive domain aggregate")
                initial_mean, start_mean, finish_mean = (
                    descriptive[method]["mean"]
                    for method in (initial_method(start), start, finish)
                )
            else:
                seeds = tuple(regular[start]["seeds"])
                before, starts, finishes = (
                    cross_domain_seed_values(absolute_averages, scale, metric, method, seeds)
                    for method in (initial_method(start), start, finish)
                )
                initial_mean, start_mean, finish_mean = (
                    statistics.fmean(values) for values in (before, starts, finishes)
                )
            # Three marks per pair and nothing joining them. The seed traces,
            # the untrained-to-control paths and the arrows that used to
            # connect them were the part of the plate a reader had to work
            # through before the marks could be read, and the question the
            # panel asks -- where each arm ends up against the untrained
            # model -- is answered by position alone. Seed counts per pair
            # stay on the record and on the per-domain plate the caption
            # points to. Draw order settles a coincidence: the untrained
            # diamond goes down first and the replay mark last, so an arm
            # that lands on the untrained value still shows on top of it.
            axis.plot(
                initial_mean, y, marker="D", linestyle="none", markersize=3.4,
                markerfacecolor="#6B7280", markeredgecolor="#6B7280", zorder=4,
            )
            axis.plot(
                start_mean, y, marker=marker, linestyle="none", markersize=3.9,
                markerfacecolor=style.WHITE, markeredgecolor=start_color,
                markeredgewidth=1.1, zorder=5,
            )
            axis.plot(
                finish_mean, y, marker=marker, linestyle="none", markersize=4.1,
                markerfacecolor=finish_color, markeredgecolor=finish_color,
                markeredgewidth=0.8, zorder=6,
            )

    axis.set_yticks(
        positions,
        [model.replace("-", "-\n", 1) if "-" in model else model
         for _scale, model, _seeds in completed],
    )
    axis.set_xlabel(
        {"pass8": "P(a success in 8)", "distinct8": "verified modes (raw count)",
         "pmd": "P(two successes differ)"}[metric],
        fontsize=7.0, labelpad=2,
    )
    if metric == "pass8":
        # Five ticks crowd a panel this narrow; the ends and the midpoint are
        # what the marks are read against.
        axis.set_xticks([0.0, 0.5, 1.0])
    # The panels sit close together, so the two labels facing the gutter are
    # justified away from it instead of centred on their spines, which would
    # overlap. Everything else stays centred on its tick.
    figure_canvas = axis.get_figure()
    figure_canvas.canvas.draw()
    labels = axis.get_xticklabels()
    if labels:
        labels[-1 if metric == "pass8" else 0].set_horizontalalignment(
            "right" if metric == "pass8" else "left")


def distinct_axis_upper(record: dict, scales: tuple[str, ...]) -> float:
    """Keep every displayed absolute domain mean inside the shared metric axis."""
    maximum = max(
        statistics.fmean(values["distinct8"])
        for scale in scales for cell in record["cells"][scale].values()
        for values in cell["methods"].values() if values.get("distinct8")
    )
    return max(2.5, math.ceil(maximum * 1.04 * 10) / 10)


def pmd_axis_upper(record: dict, scales: tuple[str, ...]) -> float:
    """Window the PCMD column on the range it actually uses.

    PCMD can reach 1, but nothing here approaches it: the drawn means top out
    well under it, so a fixed zero-to-one axis would spend most of the column on
    empty space and squeeze the separations the panels exist to show. The
    window still starts at zero, so it is a zoom on the top end rather than a
    crop of the bottom.
    """
    means = [
        statistics.fmean(values["pmd"])
        for scale in scales for cell in record["cells"][scale].values()
        for values in cell["methods"].values() if values.get("pmd")
    ]
    return max(0.4, math.ceil(max(means) * 1.08 * 20) / 20) if means else 1.0


def render_main_figure(record: dict, absolute_averages: dict) -> None:
    # Sized to be wrapped, not set full width: the panels sit close together
    # and the scale labels take the only left gutter, because panel B hides its
    # tick labels.
    style.apply_rcparams(font_size=7.4)
    figure, axes = plt.subplots(
        1, 2, figsize=(3.62, 2.02),
        gridspec_kw={"wspace": 0.085, "width_ratios": [1.0, 1.12]},
    )
    descriptive = record["descriptive_available_domain_average"]
    draw_cross_domain_panel(
        axes[0], absolute_averages=absolute_averages, descriptive_averages=descriptive,
        metric="pass8", title="A  pass@8", xlim=(0.0, 1.0),
    )
    pmd_averages = pmd_domain_averages()
    maximum_pmd = max(method["mean"] for scale in pmd_averages.values()
                      for method in scale["pmd"].values())
    pmd_upper = math.ceil(maximum_pmd * 1.16 * 20) / 20
    record["display_contract"]["main_axis_limits"] = {
        "pass8": [0.0, 1.0], "pmd": [0.0, pmd_upper],
    }
    record["display_contract"]["pmd_domains"] = {
        scale: scale_record["pmd"]["drgrpo"]["domains"]
        for scale, scale_record in pmd_averages.items()
    }
    record["display_contract"]["pmd_seed_counts"] = {
        scale: {method: aggregate["domain_seed_counts"]
                for method, aggregate in scale_record["pmd"].items()}
        for scale, scale_record in pmd_averages.items()
    }
    draw_cross_domain_panel(
        axes[1], absolute_averages={scale: {"pmd": {}} for scale in pmd_averages},
        descriptive_averages=pmd_averages,
        metric="pmd", title="B  PCMD", xlim=(0.0, pmd_upper),
    )
    axes[1].tick_params(labelleft=False)
    figure.legend(
        handles=pair_legend_handles(include_untrained=True, scale=0.78), ncol=5,
        loc="upper center", bbox_to_anchor=(0.55, 1.01), frameon=False,
        fontsize=5.6, columnspacing=0.38, handletextpad=0.18,
    )
    maxrl_note = (
        "3B MaxRL: all five seeds across all five domains; equal domain weights within seed."
        if complete_maxrl_scale(record, "qwen3b") else
        "3B MaxRL: equal domain weights; domain-specific paired seeds; descriptive only.")
    pmd_note = "; ".join(
        f"{label}: " + ", ".join(
            DOMAIN_SHORT[domain] for domain in pmd_averages[scale]["pmd"]["drgrpo"]["domains"])
        for scale, label, _seeds in MODELS if scale in pmd_averages
    )
    # The two footnote rows moved out of the plate: they are prose, they set the
    # smallest type on the page, and the caption is where a reader looks for
    # scope. Both strings stay on the record so the caption and the appendix can
    # quote them without re-deriving the domain lists.
    record["display_notes"] = {
        "maxrl_seed_scope": maxrl_note,
        "panel_b_domain_averages": "B averages each scale's support-eligible domains --- "
                                   + pmd_note + ".",
    }
    figure.subplots_adjust(top=0.783, bottom=0.188, left=0.145, right=0.995)
    for extension in ("pdf", "png"):
        figure.savefig(
            OUT.with_suffix("." + extension), dpi=220,
            bbox_inches="tight", pad_inches=0.02,
        )
    plt.close(figure)


def render_appendix_figure(record: dict, absolute_averages: dict) -> None:
    style.apply_rcparams(font_size=8.8)
    figure, axes = plt.subplots(
        3, 2, figsize=(style.WIDTH, 7.25),
        gridspec_kw={"hspace": 0.43, "wspace": 0.17},
    )
    domain_rows = tuple(zip(DOMAINS, LABELS))
    # PCMD is a proportion, so the breadth column carries the same fixed window
    # at every scale instead of distinct@8's open, correctness-driven upper end.
    pmd_upper = pmd_axis_upper(record, tuple(scale for scale, _, _ in MODELS))
    record["display_contract"]["appendix_axis_limits"] = {
        "pass8": [-0.04, 1.04], "pmd": [-0.04 * pmd_upper, pmd_upper],
    }
    for row, (scale, label, _seeds) in enumerate(MODELS):
        for col, metric in enumerate(("pass8", "pmd")):
            letter = chr(ord("A") + 2 * row + col)
            metric_label = "pass@8" if metric == "pass8" else "PCMD"
            draw_pair_panel(
                axes[row, col], record=record, absolute_averages=absolute_averages,
                scale=scale, metric=metric, title=f"{letter}  {label} · {metric_label}",
                rows=domain_rows,
                xlim=(-0.04, 1.04) if metric == "pass8"
                else (-0.04 * pmd_upper, pmd_upper),
                show_untrained=True,
            )
        axes[row, 1].tick_params(labelleft=False)
    figure.legend(
        handles=pair_legend_handles(include_untrained=True, scale=0.78), ncol=5,
        loc="upper center", bbox_to_anchor=(0.55, 1.01), frameon=False,
        fontsize=5.6, columnspacing=0.38, handletextpad=0.18,
    )
    seed_legend = "Control-arm seeds: solid n=5; dashed n<5. Arm counts can differ."
    record["display_contract"]["appendix_seed_legend"] = seed_legend
    figure.text(
        0.99, 0.013, seed_legend,
        ha="right", fontsize=7.0, color=style.MUTED,
    )
    figure.subplots_adjust(top=0.94, bottom=0.07, left=0.17, right=0.99)
    style.apply_domain_typography(figure)
    for extension in ("pdf", "png"):
        figure.savefig(
            APPENDIX_OUT.with_suffix("." + extension), dpi=220,
            bbox_inches="tight", pad_inches=0.02,
        )
    plt.close(figure)


def write_qwen3b_python_table(record: dict) -> None:
    """Emit the completed domain block without aggregating incomplete scales."""
    cell = record["cells"]["qwen3b"]["python_factors"]
    expected = [70, 71, 72, 73, 74]
    methods = (("drgrpo", "Dr.GRPO"), ("replay_drgrpo", "Re:Dr"),
               ("maxrl", "MaxRL"), ("replay_maxrl", "Re:Max"))
    if any(cell["method_seeds"][method] != expected for method, _ in methods):
        raise RuntimeError("Qwen3B Python table requires five valid four-arm seeds")
    lines = []
    for method, label in methods:
        values = cell["methods"][method]
        lines.append(f"{label} & 5 & {statistics.fmean(values['pass8']):.3f} & "
                     f"{statistics.fmean(values['distinct8']):.3f} " + r"\\")
    path = ROOT / "paper/results/e118_qwen3b_python_terminal_table_body.tex"
    path.write_text("\n".join(lines) + "\n    \\bottomrule\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-audit", type=Path,
                        help="render a retained dated terminal census without rescanning live runs")
    args = parser.parse_args()
    ENDPOINT_AUDIT.clear()
    read_endpoint = endpoint
    snapshot_source = None
    if args.endpoint_audit:
        frozen = json.loads(args.endpoint_audit.read_text())
        frozen_rows = frozen["campaigns"]["e118"]["rows"]
        by_run = {row["run_dir"]: row for row in frozen_rows}
        if len(by_run) != 150:
            raise RuntimeError("frozen E118 census must contain all 150 registered runs")
        def read_endpoint(run):
            row = by_run[run["run_dir"]]
            if any(row[key] != run[key] for key in ("scale", "domain", "arm", "seed")):
                raise RuntimeError("frozen E118 registered science identity changed")
            ENDPOINT_AUDIT.append(row["integrity_audit"])
            value = row["endpoint"]
            return {metric: value[metric] for metric in ("pass8", "distinct8")} if value else None
        snapshot_source = {"path": str(args.endpoint_audit.resolve().relative_to(ROOT)),
                           "sha256": sha(args.endpoint_audit),
                           "collection_started_at_utc": frozen["collected_at_utc"],
                           "collection_finished_at_utc": frozen["finished_at_utc"]}
    base = json.loads(BASE.read_text())
    ledger = json.loads(LEDGER.read_text())
    trajectory = json.loads(TRAJECTORY.read_text())
    if trajectory.get("schema") != "paper-aligned-domain-strip-v2":
        raise RuntimeError("before-training trajectory schema drifted")
    runs = {
        (run["scale"], run["domain"], run["arm"], int(run["seed"])): run
        for run in ledger["runs"]
    }
    record = {
        "schema": "e118-all-scale-terminal-progress-v7",
        "target_step": 3072,
        "before_training_step": 0,
        "before_training_definition": (
            "shared initial checkpoint evaluated over each track's own "
            "admissible terminal seed intersection"
        ),
        "main_figure_scales": ["qwen05b", "falcon1b"],
        "appendix_figure_scales": [scale for scale, _label, _seeds in MODELS],
        "incomplete_scales": [],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (BASE, LEDGER, TRAJECTORY)
        },
        "cells": {},
    }
    # Keep the absolute arm values in the machine-readable record because
    # Figure 4 consumes them, but make the visual answer Figure 5's actual
    # question directly: what does replay add to MaxRL on matched seeds?
    for scale, model, seeds in MODELS:
        for domain in DOMAINS:
            fresh = {
                arm: {
                    seed: read_endpoint(runs[(scale, domain, arm, seed)])
                    for seed in seeds
                }
                for arm in ("maxrl", "replay_maxrl")
            }
            common = [
                seed for seed in seeds
                if fresh["maxrl"][seed] and fresh["replay_maxrl"][seed]
            ]
            cell = (
                record["cells"].setdefault(scale, {}).setdefault(
                    domain, {"matched_seeds": common, "methods": {}}
                )
            )
            for method in ("maxrl", "replay_maxrl"):
                cell["methods"][method] = {
                    metric: [fresh[method][seed][metric] for seed in common]
                    for metric in ("pass8", "distinct8")
                }
            attach_reference_methods(
                cell, base=base, trajectory=trajectory, scale=scale,
                model=model, domain=domain,
            )

            effect_by_seed = {
                str(seed): {
                    metric: (
                        float(fresh["replay_maxrl"][seed][metric])
                        - float(fresh["maxrl"][seed][metric])
                    )
                    for metric in ("pass8", "distinct8")
                }
                for seed in common
            }
            cell["replay_maxrl_minus_maxrl"] = {
                "per_seed": effect_by_seed,
                "summaries": {},
                "status": (
                    "complete paired block" if len(common) == 5
                    else "terminal paired prefix; descriptive only"
                ),
            }
            for metric in ("pass8", "distinct8"):
                values = [effect_by_seed[str(seed)][metric] for seed in common]
                summary = {"n": len(values)}
                if values:
                    summary["mean"] = statistics.fmean(values)
                if len(values) == 5:
                    half = 2.776445105 * statistics.stdev(values) / math.sqrt(5)
                    summary["student_t_95"] = [
                        summary["mean"] - half,
                        summary["mean"] + half,
                    ]
                cell["replay_maxrl_minus_maxrl"]["summaries"][metric] = summary

    record["endpoint_integrity_audit"] = ENDPOINT_AUDIT
    if snapshot_source is not None:
        record["endpoint_snapshot"] = snapshot_source
    qwen_cells = record["cells"]["qwen05b"]
    expected_qwen_seeds = {"43", "44", "45", "46", "47"}
    if any(
        set(qwen_cells[domain]["replay_maxrl_minus_maxrl"]["per_seed"])
        != expected_qwen_seeds
        for domain in DOMAINS
    ):
        raise RuntimeError("Qwen cross-domain average requires five common seeds")
    qwen_average_per_seed = {
        seed: {
            metric: statistics.fmean(
                qwen_cells[domain]["replay_maxrl_minus_maxrl"]["per_seed"][seed][metric]
                for domain in DOMAINS
            )
            for metric in ("pass8", "distinct8")
        }
        for seed in sorted(expected_qwen_seeds, key=int)
    }
    falcon_cells = record["cells"]["falcon1b"]
    falcon_expected_seeds = {55, 56, 57, 58, 59}
    falcon_complete = all(
        set(falcon_cells[domain]["matched_seeds"]) == falcon_expected_seeds
        for domain in DOMAINS
    )
    falcon_average_per_seed = {
        str(seed): {
            metric: statistics.fmean(
                falcon_cells[domain]["replay_maxrl_minus_maxrl"]
                ["per_seed"][str(seed)][metric]
                for domain in DOMAINS
            )
            for metric in ("pass8", "distinct8")
        }
        for seed in sorted(falcon_expected_seeds)
    } if falcon_complete else {}
    averages = {
        "qwen05b": {
            "definition": "equal domain average within paired seed",
            "per_seed": qwen_average_per_seed,
            "summaries": {},
            "status": "post-hoc descriptive cross-domain average",
        },
        "falcon1b": {
            "definition": (
                "equal domain average within paired seed" if falcon_complete
                else "equal average of the five displayed domain-prefix means"
            ),
            "per_seed": falcon_average_per_seed,
            "summaries": {},
            "status": (
                "post-hoc descriptive cross-domain average" if falcon_complete
                else "descriptive partial-prefix average; no uncertainty interval"
            ),
        },
    }
    for metric in ("pass8", "distinct8"):
        qwen_values = [row[metric] for row in qwen_average_per_seed.values()]
        qwen_mean = statistics.fmean(qwen_values)
        qwen_half = (
            2.776445105 * statistics.stdev(qwen_values) / math.sqrt(5)
        )
        averages["qwen05b"]["summaries"][metric] = {
            "n": 5,
            "mean": qwen_mean,
            "student_t_95": [qwen_mean - qwen_half, qwen_mean + qwen_half],
        }
        if falcon_complete:
            falcon_values = [
                row[metric] for row in falcon_average_per_seed.values()
            ]
            falcon_mean = statistics.fmean(falcon_values)
            falcon_half = (
                2.776445105 * statistics.stdev(falcon_values) / math.sqrt(5)
            )
            averages["falcon1b"]["summaries"][metric] = {
                "n": 5,
                "mean": falcon_mean,
                "student_t_95": [
                    falcon_mean - falcon_half,
                    falcon_mean + falcon_half,
                ],
            }
        else:
            averages["falcon1b"]["summaries"][metric] = {
                "n_domains": 5,
                "domain_seed_counts": {
                    domain: len(falcon_cells[domain]["matched_seeds"])
                    for domain in DOMAINS
                },
                "mean": statistics.fmean(
                    falcon_cells[domain]["replay_maxrl_minus_maxrl"]
                    ["summaries"][metric]["mean"]
                    for domain in DOMAINS
                ),
            }
    record["cross_domain_average"] = averages
    record["display_contract"] = {
        "estimand": "Re:Max minus MaxRL on matched terminal seeds",
        "filled_marker": "complete five-seed block",
        "open_marker": "terminal paired prefix; no interval",
        "intervals": "unadjusted descriptive 95% Student-t; n=5 only",
        "qwen_average": averages["qwen05b"]["definition"],
        "falcon_average": averages["falcon1b"]["definition"],
    }

    # E80-R1 and E118 each retain their own audited paired cohorts.
    add_qwen3b_main_figure_track(record)
    # Each track averages the same admissible seeds across every domain.
    absolute_averages = absolute_cross_domain_averages(record)
    record["absolute_cross_domain_average"] = absolute_averages
    record["descriptive_available_domain_average"] = descriptive_available_domain_averages(record)
    qwen3b_complete = complete_maxrl_scale(record, "qwen3b")
    if qwen3b_complete:
        per_seed = {
            str(seed): {
                metric: absolute_averages["qwen3b"][metric]["replay_maxrl"]["per_seed"][str(seed)]
                - absolute_averages["qwen3b"][metric]["maxrl"]["per_seed"][str(seed)]
                for metric in ("pass8", "distinct8")
            }
            for seed in MODELS[2][2]
        }
        summaries = {}
        for metric in ("pass8", "distinct8"):
            values = [row[metric] for row in per_seed.values()]
            mean = statistics.fmean(values)
            half = 2.776445105 * statistics.stdev(values) / math.sqrt(5)
            summaries[metric] = {"n": 5, "mean": mean,
                                 "student_t_95": [mean - half, mean + half]}
        record["cross_domain_average"]["qwen3b"] = {
            "definition": PAIRED_DOMAIN_DEFINITION, "per_seed": per_seed,
            "summaries": summaries, "status": "post-hoc descriptive cross-domain average",
        }
    record["display_contract"] = {
        **qwen3b_display_descriptions(record),
        "appendix_figure": (
            "all three models across five domains and both terminal metrics"
        ),
        "estimands": [
            "Re:Max minus MaxRL on matched terminal seeds",
            "Re:Dr minus Dr.GRPO on its admissible paired seed intersection",
        ],
        "tracks": {
            "upper": "Untrained to MaxRL to Re:Max",
            "lower": "Untrained to Dr.GRPO to Re:Dr",
        },
        "untrained_reference": (
            "shared frozen step-0 checkpoint restricted to each track's paired seeds; "
            "arms share initialization but are trained separately"
        ),
        "domain_backgrounds": (
            "exact Figure 2 domain-card tints; color is contextual, not data"
        ),
        "main_figure_marks": (
            "marks only: untrained diamond, open control, filled replay on one row; "
            "no connectors, seed traces or arrows"
        ),
        "solid_connector": "complete five-seed block",
        "dashed_connector": "partial or domain-specific paired seeds with exact counts and no interval",
        "numeric_annotations": (
            "complete five-seed tracks use solid connectors; partial tracks show exact n; "
            "all domain-specific counts and endpoints remain in JSON and appendix"
        ),
        "main_panels": {
            "A": "cross-domain pass@8", "B": "cross-domain PCMD",
        },
        "qwen_average": "equal domain average within paired seed",
        "falcon_average": averages["falcon1b"]["definition"],
        "qwen3b_maxrl_average": (PAIRED_DOMAIN_DEFINITION if qwen3b_complete
                                 else AVAILABLE_DOMAIN_DEFINITION),
        "qwen3b_maxrl_counts": {
            domain: len(record["cells"]["qwen3b"][domain]["matched_seeds"])
            for domain in DOMAINS
        },
        "qwen3b_maxrl_uncertainty": (
            "five paired seeds behind each mark; no seed paths or interval in the absolute-mean figure"
            if qwen3b_complete else "none; no pooled seed paths or cross-domain interval"
        ),
    }
    write_qwen3b_python_table(record)
    render_main_figure(record, absolute_averages)
    # The appendix panels read PCMD, which is not in the endpoint records the
    # rest of this builder assembles, so it is attached before they are drawn.
    attach_pmd_cells(record)
    render_appendix_figure(record, absolute_averages)
    for path in (
        OUT.with_suffix(".json"), APPENDIX_OUT.with_suffix(".json"),
    ):
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(OUT.with_suffix(".pdf"))
    print(APPENDIX_OUT.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
