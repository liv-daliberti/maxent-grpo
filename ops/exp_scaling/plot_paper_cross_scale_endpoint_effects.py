#!/usr/bin/env python3
"""Plot all available endpoint effects and the pass-8 paper frontier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402


DEFAULT_INPUT = ROOT / "paper/results/core_terminal_endpoints.json"
DEFAULT_PLAIN_GRPO_INPUT = (
    ROOT / "paper/results/e95_falcon_plain_grpo_reportable.json"
)
DEFAULT_DIRECT_COMPARATOR_INPUT = (
    ROOT / "paper/figures/direct_comparator_endpoint_effects.json"
)
DEFAULT_OUTPUT = ROOT / "paper/figures/cross_scale_terminal_endpoint_effects"
DEFAULT_FRONTIER_OUTPUT = ROOT / "paper/figures/terminal_pass8_distinct8_frontier"
MODELS = ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")
MODEL_SHORT = {
    "Qwen2.5-0.5B": "Qwen 0.5B",
    "Falcon3-1B": "Falcon 1B",
    "Qwen2.5-3B": "Qwen 3B",
}
DOMAIN_ORDER = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
DOMAIN_LABELS = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
T_CRIT_DF4 = 2.7764451051977987
FRONTIER_METHODS = (
    "drgrpo",
    "grpo",
    "replay_grpo",
    "ucpo",
    "rlep_dr",
)
# Redundant colour-and-shape encoding, so no comparison here depends on colour
# perception alone. Both channels come from the paper-wide registry rather than
# being restated locally. They used to be a separate Okabe-Ito set defined right
# here, which is how this figure ended up painting Re:Dr.GRPO orange while
# every other figure in the manuscript paints *matched Dr.GRPO* orange --- the
# reader met the method under the control's colour in the one figure that
# carries the headline result. paper_style.FRONTIER_FIVE records the validator
# run for the five identities that share these axes.
FRONTIER_VISUALS = {
    method: {
        "color": method_visuals.method_style(method)["color"],
        "marker": method_visuals.method_style(method)["marker"],
    }
    for method in FRONTIER_METHODS
}
FRONTIER_METHOD_LABELS = {
    "drgrpo": "matched Dr.GRPO",
    "grpo": "GRPO",
    "replay_grpo": "Re:Dr.GRPO (ours)",
    "ucpo": "UCPO",
    "rlep_dr": "RLEP-Dr",
}
FRONTIER_METHOD_SHORT = {
    "drgrpo": "Dr",
    "grpo": "GRPO",
    "replay_grpo": "Replay",
    "ucpo": "UCPO",
    "rlep_dr": "RLEP",
}
SCALE_TO_MODEL = {
    "qwen05b": "Qwen2.5-0.5B",
    "falcon1b": "Falcon3-1B",
    "qwen3b": "Qwen2.5-3B",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _effect_summary(values: list[float]) -> dict[str, Any]:
    """Summarize every available seed; reserve the preregistered CI for n=5."""

    if not values:
        raise RuntimeError("cannot summarize an empty endpoint block")
    mean = statistics.fmean(values)
    summary: dict[str, Any] = {
        "mean": mean,
        "n": len(values),
        "range": [min(values), max(values)],
    }
    if len(values) == 5:
        half = T_CRIT_DF4 * statistics.stdev(values) / math.sqrt(len(values))
        summary["student_t_95"] = [mean - half, mean + half]
    return summary


def _plain_grpo_endpoints(
    payload: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return every Falcon domain/seed with a sampled pass-8 endpoint."""

    if payload is None:
        return {}
    if (
        payload.get("schema") not in {
            "paper-e95-falcon-plain-grpo-reportable-v1",
            "paper-e95-falcon-plain-grpo-available-v2",
        }
        or payload.get("model") != "Falcon3-1B"
        or payload.get("scale") != "falcon1b"
    ):
        raise RuntimeError("plain-GRPO endpoint contract drifted")
    endpoints: dict[str, dict[str, dict[str, float]]] = {}
    for domain, domain_payload in payload.get("domains", {}).items():
        terminal = domain_payload.get("summary_by_pass", {}).get("8.0")
        if terminal is None:
            continue
        metrics = terminal["metrics"]
        seed_keys = sorted(
            set(metrics["pass8"]["per_seed"])
            & set(metrics["distinct8"]["per_seed"]),
            key=int,
        )
        if not seed_keys:
            continue
        endpoints[domain] = {
            seed: {
                "pass8": float(metrics["pass8"]["per_seed"][seed]),
                "distinct8": float(metrics["distinct8"]["per_seed"][seed]),
            }
            for seed in seed_keys
        }
    return endpoints


def _terminal_rows(
    payload: dict[str, Any],
    plain_grpo_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return every static-domain pass-8 row with at least one paired seed."""

    if payload.get("schema") != "paper-core-terminal-endpoints-v1":
        raise RuntimeError("current core terminal endpoint schema drifted")
    families = payload["models"]
    plain_grpo = _plain_grpo_endpoints(plain_grpo_payload)
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        domains = families.get(model, {}).get("domains", {})
        for domain in DOMAIN_ORDER:
            if domain not in domains:
                continue
            record = domains[domain]
            if float(record.get("training_pass", -1)) != 8.0:
                continue
            methods = record.get("methods", {})
            control = methods.get("control", {}).get("per_seed", {})
            replay = methods.get("replay", {}).get("per_seed", {})
            seeds = sorted(set(map(int, control)) & set(map(int, replay)))
            if not seeds:
                continue
            effects = {"pass8": [], "adjusted_breadth8": []}
            per_seed_effects: dict[str, Any] = {}
            grpo_effects = {"pass8": [], "adjusted_breadth8": []}
            grpo_per_seed_effects: dict[str, Any] = {}
            per_seed_endpoints: dict[str, Any] = {}
            grpo_endpoints = (
                plain_grpo.get(domain, {}) if model == "Falcon3-1B" else {}
            )
            for seed in sorted(set(map(int, control)) | set(map(int, replay))):
                key = str(seed)
                per_seed_endpoints[key] = {}
                if key in control:
                    per_seed_endpoints[key]["control"] = {
                        metric: float(control[key][metric])
                        for metric in ("pass8", "distinct8")
                    }
                if key in replay:
                    per_seed_endpoints[key]["replay"] = {
                        metric: float(replay[key][metric])
                        for metric in ("pass8", "distinct8")
                    }
                if key in grpo_endpoints and key in control:
                    per_seed_endpoints[key]["grpo"] = grpo_endpoints[key]
            for seed in seeds:
                key = str(seed)
                control_endpoint = control[key]
                replay_endpoint = replay[key]
                pass_effect = (
                    float(replay_endpoint["pass8"])
                    - float(control_endpoint["pass8"])
                )
                adjusted_effect = (
                    float(replay_endpoint["distinct8"])
                    - float(replay_endpoint["pass8"])
                    - float(control_endpoint["distinct8"])
                    + float(control_endpoint["pass8"])
                )
                effects["pass8"].append(pass_effect)
                effects["adjusted_breadth8"].append(adjusted_effect)
                per_seed_effects[key] = {
                    "pass8": pass_effect,
                    "adjusted_breadth8": adjusted_effect,
                }
                if key not in grpo_endpoints:
                    continue
                grpo = grpo_endpoints[key]
                grpo_pass_effect = (
                    float(grpo["pass8"]) - float(control_endpoint["pass8"])
                )
                grpo_adjusted_effect = (
                    float(grpo["distinct8"])
                    - float(grpo["pass8"])
                    - float(control_endpoint["distinct8"])
                    + float(control_endpoint["pass8"])
                )
                grpo_effects["pass8"].append(grpo_pass_effect)
                grpo_effects["adjusted_breadth8"].append(grpo_adjusted_effect)
                grpo_per_seed_effects[key] = {
                    "pass8": grpo_pass_effect,
                    "adjusted_breadth8": grpo_adjusted_effect,
                }

            summaries = {
                metric: _effect_summary(values)
                for metric, values in effects.items()
            }
            grpo_summaries = {
                metric: _effect_summary(values)
                for metric, values in grpo_effects.items() if values
            }
            endpoint_summaries: dict[str, Any] = {}
            for method in ("control", "replay", "grpo"):
                method_seeds = [
                    seed
                    for seed, endpoints in per_seed_endpoints.items()
                    if method in endpoints
                ]
                if not method_seeds:
                    continue
                endpoint_summaries[method] = {
                    metric: {
                        "mean": statistics.fmean(
                            per_seed_endpoints[seed][method][metric]
                            for seed in method_seeds
                        ),
                        "range": [
                            min(per_seed_endpoints[seed][method][metric] for seed in method_seeds),
                            max(per_seed_endpoints[seed][method][metric] for seed in method_seeds),
                        ],
                        "n": len(method_seeds),
                    }
                    for metric in ("pass8", "distinct8")
                }
            rows.append({
                "model": model,
                "domain": domain,
                "pass": 8.0,
                "seeds": seeds,
                "n": len(seeds),
                "per_seed_effects": per_seed_effects,
                "per_seed_endpoints": per_seed_endpoints,
                "summaries": summaries,
                "grpo_per_seed_effects": grpo_per_seed_effects,
                "grpo_summaries": grpo_summaries,
                "endpoint_summaries": endpoint_summaries,
            })
    return rows


def render(rows: list[dict[str, Any]], output: Path) -> None:
    """Render paired endpoint effects against matched Dr.GRPO controls."""

    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(MODELS),
        len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 3.25),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    metrics = (
        ("pass8", r"$\Delta P$"),
        ("adjusted_breadth8", r"$\Delta(D-P)$"),
    )
    contrasts = (
        (
            "grpo",
            "grpo_per_seed_effects",
            "grpo_summaries",
            method_visuals.method_style("grpo"),
            -0.075,
        ),
        (
            "replay",
            "per_seed_effects",
            "summaries",
            method_visuals.method_style("replay_grpo"),
            0.075,
        ),
    )
    x_positions = (0.0, 1.0)
    def seed_jitter(n: int) -> list[float]:
        if n <= 1:
            return [0.0] * n
        return [-0.04 + 0.08 * index / (n - 1) for index in range(n)]
    rows_by_cell = {(row["model"], row["domain"]): row for row in rows}
    display_bounds = [
        bound
        for row in rows
        for _name, _points_key, summary_key, _visual, _offset in contrasts
        if row.get(summary_key)
        for metric, _label in metrics
        for bound in row[summary_key][metric].get(
            "student_t_95", row[summary_key][metric]["range"]
        )
    ]
    y_low = math.floor((min(display_bounds) - 0.10) * 10) / 10
    y_high = math.ceil((max(display_bounds) + 0.10) * 10) / 10

    for model_index, model in enumerate(MODELS):
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            axis = axes[model_index][domain_index]
            style.style_axis(
                axis,
                grid="both",
                title=DOMAIN_LABELS[domain] if model_index == 0 else None,
            )
            axis.axhline(0.0, color=style.MUTED, lw=0.8, linestyle=(0, (2, 2)))
            row = rows_by_cell.get((model, domain))
            if row is not None:
                for x, (metric, _label) in zip(x_positions, metrics):
                    for (
                        _name,
                        points_key,
                        summary_key,
                        visual,
                        offset,
                    ) in contrasts:
                        if not row.get(summary_key):
                            continue
                        center = x + offset
                        values = [
                            row[points_key][str(seed)][metric]
                            for seed in row["seeds"]
                            if str(seed) in row[points_key]
                        ]
                        summary = row[summary_key][metric]
                        axis.scatter(
                            [center + seed_offset for seed_offset in seed_jitter(len(values))],
                            values,
                            s=9,
                            marker=visual["marker"],
                            facecolor=style.WHITE,
                            edgecolor=visual["color"],
                            linewidth=0.65,
                            zorder=3,
                        )
                        if "student_t_95" in summary:
                            low, high = summary["student_t_95"]
                            axis.vlines(
                                center,
                                low,
                                high,
                                color=visual["color"],
                                linewidth=1.35,
                                zorder=4,
                            )
                        axis.scatter(
                            [center],
                            [summary["mean"]],
                            s=25,
                            marker=visual["marker"],
                            color=visual["color"],
                            edgecolor=style.WHITE,
                            linewidth=0.5,
                            zorder=5,
                        )
            axis.set_xlim(-0.42, 1.42)
            axis.set_ylim(y_low, y_high)
            axis.set_xticks(x_positions, [item[1] for item in metrics])
            if domain_index == 0:
                axis.set_ylabel(
                    f"{MODEL_SHORT[model]}\neffect vs Dr.GRPO",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)

    handles = [
        Line2D(
            [0],
            [0],
            marker=visual["marker"],
            color=visual["color"],
            markerfacecolor=visual["color"],
            markeredgecolor=style.WHITE,
            markersize=4.6,
        )
        for _name, _points, _summary, visual, _offset in contrasts
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor=style.WHITE,
            markeredgecolor=style.MUTED,
            color="none",
            markersize=4,
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=style.MUTED,
            markeredgecolor=style.WHITE,
            markersize=4,
        ),
    ]
    style.bottom_legend(
        figure,
        handles,
        [
            "GRPO − Dr.GRPO",
            "Re:Dr.GRPO − Dr.GRPO",
            "paired seed",
            "mean; 95% interval at n=5",
        ],
        y=0.003,
        ncol=4,
    )
    figure.suptitle(
        "Terminal GRPO-family endpoint effects vs Dr.GRPO",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.88,
        bottom=0.19,
        hspace=0.18,
        wspace=0.20,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def _endpoint_method(
    method: str,
    per_seed: dict[str, dict[str, float]],
    *,
    evidence: str,
) -> dict[str, Any]:
    seeds = sorted(int(seed) for seed in per_seed)
    if set(per_seed) != {str(seed) for seed in seeds}:
        raise RuntimeError(f"{method}: malformed per-seed endpoint keys")
    record: dict[str, Any] = {
        "label": FRONTIER_METHOD_LABELS[method],
        "evidence": evidence,
        "n": len(seeds),
        "seeds": seeds,
        "per_seed": per_seed,
    }
    if seeds:
        record["summary"] = {
            metric: statistics.fmean(
                per_seed[str(seed)][metric] for seed in seeds
            )
            for metric in ("pass8", "distinct8")
        }
    return record




def _frontier_cells(
    rows: list[dict[str, Any]],
    direct_comparators: dict[str, Any],
) -> list[dict[str, Any]]:
    """Assemble every available pass-8 method endpoint with exact n."""

    if (
        direct_comparators.get("schema")
        != "paper-direct-comparator-endpoint-effects-v2"
    ):
        raise RuntimeError("direct-comparator endpoint contract drifted")

    cells = {
        (model, domain): {
            "model": model,
            "domain": domain,
            "pass": 8.0,
            "methods": {},
            "blank_methods": {},
        }
        for model in MODELS
        for domain in DOMAIN_ORDER
    }

    for row in rows:
        cell = cells[(row["model"], row["domain"])]
        for source_method, frontier_method in (
            ("control", "drgrpo"), ("replay", "replay_grpo")
        ):
            endpoints = {
                seed: methods[source_method]
                for seed, methods in row["per_seed_endpoints"].items()
                if source_method in methods
            }
            cell["methods"][frontier_method] = _endpoint_method(
                frontier_method, endpoints, evidence="all_available_terminal_seeds"
            )
    for source_cell in direct_comparators.get("cells", []):
        model = SCALE_TO_MODEL[source_cell["scale"]]
        domain = source_cell["domain"]
        key = (model, domain)
        if key not in cells:
            raise RuntimeError(
                f"unexpected direct-comparator cell {model}/{domain}"
            )
        cell = cells[key]
        baseline_endpoints = cell["methods"]["drgrpo"]["per_seed"]
        for method, record in source_cell.get("methods", {}).items():
            if method not in {"grpo", "ucpo", "rlep_dr"}:
                raise RuntimeError(
                    f"unexpected direct comparator {method} in {model}/{domain}"
                )
            endpoints: dict[str, dict[str, float]] = {}
            for seed, seed_record in record.get("per_seed", {}).items():
                baseline = {
                    metric: float(seed_record["baseline"][metric])
                    for metric in ("pass8", "distinct8")
                }
                if seed in baseline_endpoints and baseline != baseline_endpoints[seed]:
                    raise RuntimeError(
                        f"baseline endpoint drift for {model}/{domain}/seed {seed}"
                    )
                endpoints[seed] = {
                    metric: float(seed_record["comparator"][metric])
                    for metric in ("pass8", "distinct8")
                }
            if int(record.get("n", -1)) != len(endpoints):
                raise RuntimeError(
                    f"direct comparator n drift for {model}/{domain}/{method}"
                )
            cell["methods"][method] = _endpoint_method(
                method,
                endpoints,
                evidence="all_available_terminal_seeds",
            )

    for cell in cells.values():
        model, domain = cell["model"], cell["domain"]
        for method in FRONTIER_METHODS:
            if method in cell["methods"]:
                continue
            reason = (
                f"no terminal {FRONTIER_METHOD_LABELS[method]} pass-8 endpoint"
            )
            cell["blank_methods"][method] = reason
        if set(cell["methods"]) | set(cell["blank_methods"]) != set(
            FRONTIER_METHODS
        ):
            raise RuntimeError(f"frontier method gate incomplete for {model}/{domain}")
    return list(cells.values())


def render_frontier(cells: list[dict[str, Any]], output: Path) -> None:
    """Render pass@8 against distinct@8 for all requested endpoint methods."""

    style.apply_rcparams()
    figure, axes = plt.subplots(
        len(MODELS),
        len(DOMAIN_ORDER),
        figsize=(style.WIDTH, 4.65),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    cells_by_key = {
        (cell["model"], cell["domain"]): cell for cell in cells
    }
    visuals = FRONTIER_VISUALS
    maximum = max(
        endpoint["distinct8"]
        for cell in cells
        for record in cell["methods"].values()
        for endpoint in record["per_seed"].values()
    )
    y_high = max(2.0, math.ceil((maximum + 0.15) * 2) / 2)

    for model_index, model in enumerate(MODELS):
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            axis = axes[model_index][domain_index]
            style.style_axis(
                axis,
                grid="both",
                title=DOMAIN_LABELS[domain] if model_index == 0 else None,
            )
            axis.set_xlim(-0.03, 1.03)
            axis.set_ylim(-0.05, y_high)
            axis.set_xticks((0.0, 0.5, 1.0))
            cell = cells_by_key[(model, domain)]
            partial_labels = []
            for method in FRONTIER_METHODS:
                if method not in cell["methods"]:
                    continue
                record = cell["methods"][method]
                visual = visuals[method]
                endpoints = record["per_seed"].values()
                axis.scatter(
                    [point["pass8"] for point in endpoints],
                    [point["distinct8"] for point in endpoints],
                    s=21,
                    marker=visual["marker"],
                    facecolor=style.WHITE,
                    edgecolor=visual["color"],
                    linewidth=1.0,
                    zorder=3,
                )
                summary = record["summary"]
                axis.scatter(
                    [summary["pass8"]],
                    [summary["distinct8"]],
                    s=54,
                    marker=visual["marker"],
                    facecolor=visual["color"],
                    # A surface-coloured ring, not the old near-black one. The
                    # five method means land on top of each other in most cells
                    # here, and a white separator is what keeps two touching
                    # markers reading as two markers.
                    edgecolor=style.WHITE,
                    linewidth=0.8,
                    zorder=4,
                )
                if record["n"] != 5:
                    partial_labels.append(
                        f"{FRONTIER_METHOD_SHORT[method]}: n={record['n']}"
                    )
            if partial_labels:
                axis.text(
                    0.03,
                    0.97,
                    "\n".join(partial_labels),
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=style.SMALL_FONT - 0.4,
                    color=style.MUTED,
                )
            if domain_index == 0:
                axis.set_ylabel(
                    f"{MODEL_SHORT[model]}\ndistinct@8",
                    fontsize=style.LABEL_FONT,
                )
            else:
                axis.tick_params(labelleft=False)
            if model_index == len(MODELS) - 1:
                axis.set_xlabel("pass@8", fontsize=style.LABEL_FONT)

    method_order = FRONTIER_METHODS
    handles = [
        Line2D(
            [0],
            [0],
            marker=visuals[method]["marker"],
            color="none",
            markerfacecolor=visuals[method]["color"],
            markeredgecolor=style.WHITE,
            markeredgewidth=0.8,
            markersize=6.2,
        )
        for method in method_order
    ]
    style.bottom_legend(
        figure,
        handles,
        [FRONTIER_METHOD_LABELS[method] for method in method_order],
        y=0.003,
        ncol=5,
    )
    figure.suptitle(
        "Pass-8 correctness–breadth frontier",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.995,
    )
    figure.text(
        0.5,
        0.96,
        (
            "Color + shape identify methods; open = seed, filled = exact-n "
            "mean; upper right is better; unsupported cells are blank."
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.85,
        bottom=0.205,
        hspace=0.30,
        wspace=0.20,
    )
    style.save(figure, output, png=True, dpi=260)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--plain-grpo-input", type=Path, default=DEFAULT_PLAIN_GRPO_INPUT
    )
    parser.add_argument(
        "--direct-comparator-input",
        type=Path,
        default=DEFAULT_DIRECT_COMPARATOR_INPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--frontier-output", type=Path, default=DEFAULT_FRONTIER_OUTPUT
    )
    args = parser.parse_args()
    source = args.input.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    plain_grpo_source = args.plain_grpo_input.resolve()
    plain_grpo_payload = json.loads(
        plain_grpo_source.read_text(encoding="utf-8")
    )
    direct_comparator_source = args.direct_comparator_input.resolve()
    direct_comparator_payload = json.loads(
        direct_comparator_source.read_text(encoding="utf-8")
    )
    rows = _terminal_rows(payload, plain_grpo_payload)
    frontier_cells = _frontier_cells(rows, direct_comparator_payload)
    output = args.output.resolve()
    render(rows, output)
    frontier_output = args.frontier_output.resolve()
    render_frontier(frontier_cells, frontier_output)
    blank_rule = (
        "a method is blank only when no pass-8 seed endpoint exists for that cell"
    )
    excluded: dict[str, str] = {}
    frontier_excluded = {
        "dapo": "no standardized pass@8 or distinct@8 endpoint",
        "retired_semantic_maxent": (
            "retired-estimator treatments are preserved in the historical appendix"
        ),
    }
    common_provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "all available terminal pass-8 seeds with exact n",
        "source": str(source),
        "source_sha256": _sha256(source),
        "plain_grpo_source": str(plain_grpo_source),
        "plain_grpo_source_sha256": _sha256(plain_grpo_source),
        "model_rows": list(MODELS),
        "domain_order": list(DOMAIN_ORDER),
        "blank_rule": blank_rule,
        "rows": rows,
        "excluded": excluded,
    }
    provenance = {
        "schema": "paper-cross-scale-endpoint-effects-v2",
        **common_provenance,
        "methods": ["GRPO", "matched Dr.GRPO", "Re:Dr.GRPO"],
        "metrics": [
            "Re:Dr.GRPO minus matched Dr.GRPO pass@8",
            (
                "Re:Dr.GRPO minus matched Dr.GRPO "
                "(distinct@8 - pass@8)"
            ),
            "GRPO minus matched Dr.GRPO pass@8",
            "GRPO minus matched Dr.GRPO (distinct@8 - pass@8)",
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    frontier_provenance = {
        "schema": "paper-terminal-accuracy-breadth-frontier-v5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "current direct-method pass-8 endpoints with exact n; retired "
            "semantic treatments and endpoint-incomplete DAPO are excluded"
        ),
        "x_metric": "pass@8",
        "y_metric": "distinct@8",
        "methods": [
            FRONTIER_METHOD_LABELS[method] for method in FRONTIER_METHODS
        ],
        "method_keys": list(FRONTIER_METHODS),
        "model_rows": list(MODELS),
        "domain_order": list(DOMAIN_ORDER),
        "model_encoding": (
            "row 1 Qwen2.5-0.5B; row 2 Falcon3-1B; row 3 Qwen2.5-3B; "
            "unsupported method-cells are blank"
        ),
        "evidence_encoding": (
            "color and shape jointly identify methods; open markers are exact "
            "terminal seeds; filled markers of the same method-specific shape "
            "are descriptive means over the exact available n; non-five-seed "
            "blocks carry an explicit n label"
        ),
        "blank_rule": (
            "methods without the figure's stated pass-8 evidence remain blank"
        ),
        "cells": frontier_cells,
        "source": str(source),
        "source_sha256": _sha256(source),
        "plain_grpo_source": str(plain_grpo_source),
        "plain_grpo_source_sha256": _sha256(plain_grpo_source),
        "direct_comparator_source": str(direct_comparator_source),
        "direct_comparator_source_sha256": _sha256(direct_comparator_source),
        "excluded": frontier_excluded,
    }
    frontier_output.with_suffix(".json").write_text(
        json.dumps(frontier_provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.png')}")
    print(f"wrote {output.with_suffix('.json')}")
    print(f"wrote {frontier_output.with_suffix('.pdf')}")
    print(f"wrote {frontier_output.with_suffix('.png')}")
    print(f"wrote {frontier_output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
