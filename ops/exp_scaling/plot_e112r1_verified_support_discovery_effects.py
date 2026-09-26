#!/usr/bin/env python3
"""Render E112-R1 terminal and AUC forests in the tested E105 grammar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_e105_group_centered_semantic_endpoint_effects as e105_plot  # noqa: E402


DEFAULT_INPUT = ROOT / (
    "paper/results/e112r1_verified_support_discovery_three_scale.json"
)
DEFAULT_OUTPUT = ROOT / (
    "paper/figures/e112r1_verified_support_discovery_endpoint_effects"
)
DEFAULT_AUC_OUTPUT = ROOT / (
    "paper/figures/e112r1_verified_support_discovery_auc_effects"
)
DEFAULT_PRIMITIVE_OUTPUT = ROOT / (
    "paper/figures/e112r1_verified_support_discovery_primitive_endpoint_effects"
)
DEFAULT_PRIMITIVE_AUC_OUTPUT = ROOT / (
    "paper/figures/e112r1_verified_support_discovery_primitive_auc_effects"
)
SCHEMAS = {
    "terminal": "paper-e112r1-verified-support-endpoint-effects-v1",
    "auc": "paper-e112r1-verified-support-auc-effects-v1",
    "terminal_primitives": (
        "paper-e112r1-verified-support-primitive-endpoint-effects-v1"
    ),
    "auc_primitives": "paper-e112r1-verified-support-primitive-auc-effects-v1",
}
TITLES = {
    "terminal": ("E112-R1 bundle vs historical Re:Dr: paired terminal effects"),
    "auc": ("E112-R1 bundle vs historical Re:Dr: paired trajectory AUC"),
    "terminal_primitives": (
        "E112-R1 bundle vs historical Re:Dr: primitive terminal effects"
    ),
    "auc_primitives": (
        "E112-R1 bundle vs historical Re:Dr: primitive trajectory AUC"
    ),
}
BASE_EFFECT_KINDS = {
    "terminal": "terminal",
    "auc": "auc",
    "terminal_primitives": "terminal",
    "auc_primitives": "auc",
}

sha256 = e105_plot.sha256
render = e105_plot.render


def plot_payload(
    result: dict[str, Any], *, input_path: Path, effect_kind: str
) -> dict[str, Any]:
    if effect_kind not in SCHEMAS:
        raise RuntimeError(f"unknown E112-R1 effect kind: {effect_kind}")
    if result.get("schema") != "e112r1_verified_support_discovery_results_v1":
        raise RuntimeError("unexpected E112-R1 result schema")
    if result.get("pointmaze") != "excluded":
        raise RuntimeError("E112-R1 paired forest requires PointMaze exclusion")
    provenance = result.get("analysis_provenance", {})
    if provenance.get("analysis_plotter") != str(Path(__file__).resolve()):
        raise RuntimeError("E112-R1 result names another paired-effect plotter")
    if provenance.get("analysis_plotter_sha256") != sha256(Path(__file__)):
        raise RuntimeError("E112-R1 paired-effect plotter digest mismatch")
    estimand = result.get("estimand_scope", {})
    if (
        estimand.get("kind") != "bundled historical-comparator contrast"
        or estimand.get("isolated_semantic_v7_effect") is not False
        or estimand.get("confirmatory_blind") is not False
    ):
        raise RuntimeError("E112-R1 estimand disclosure drifted")

    # The compatibility view lets the existing extractor enforce the exact
    # 3-by-5, n=5, paired-summary contract. It is never serialized.
    compatibility = dict(result)
    compatibility.update(
        {
            "schema": "e105_group_centered_semantic_repair_results_v1",
            "analysis_plotter": str(Path(e105_plot.__file__).resolve()),
            "analysis_plotter_sha256": sha256(Path(e105_plot.__file__)),
        }
    )
    payload = e105_plot.plot_payload(
        compatibility,
        input_path=input_path,
        effect_kind=BASE_EFFECT_KINDS[effect_kind],
    )
    if effect_kind.endswith("_primitives"):
        prefix = "terminal" if effect_kind.startswith("terminal") else "auc"
        metrics = (
            (f"{prefix}_sampled_pass8", r"$\Delta P$"),
            (f"{prefix}_sampled_distinct8", r"$\Delta D$"),
        )
        for cell in payload["cells"]:
            family = result["families"][cell["scale"]][cell["domain"]]
            cell["per_seed_effects"] = {
                str(seed): {
                    metric: float(
                        family["paired_effects"][metric]["per_seed"][str(seed)]
                    )
                    for metric, _label in metrics
                }
                for seed in cell["seeds"]
            }
            cell["summaries"] = {
                metric: family["paired_effects"][metric] for metric, _label in metrics
            }
        payload["metric_order"] = [metric for metric, _label in metrics]
        payload["metric_labels"] = {metric: label for metric, label in metrics}
        payload["metrics"] = {
            metrics[0][0]: (
                "paired sampled pass@8 effect; primitive endpoint coordinate"
            ),
            metrics[1][0]: (
                "paired raw distinct correct modes@8 effect; primitive "
                "endpoint coordinate"
            ),
        }
        payload["registered_decision_view"] = False
        payload["supplementary_primitive_vector_view"] = True
    payload.update(
        {
            "schema": SCHEMAS[effect_kind],
            "plotter": str(Path(__file__).resolve()),
            "plotter_sha256": sha256(Path(__file__)),
            "treatment": "E112-R1 verified-support-discovery bundle",
            "title": TITLES[effect_kind],
            "estimand_scope": estimand,
            "metric_contract": result["metric_contract"],
            "registered_all_three_scales_criterion": result[
                "registered_all_three_scales_criterion"
            ],
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--auc-output", type=Path, default=DEFAULT_AUC_OUTPUT)
    parser.add_argument(
        "--primitive-output", type=Path, default=DEFAULT_PRIMITIVE_OUTPUT
    )
    parser.add_argument(
        "--primitive-auc-output", type=Path, default=DEFAULT_PRIMITIVE_AUC_OUTPUT
    )
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    terminal = plot_payload(result, input_path=args.input, effect_kind="terminal")
    auc = plot_payload(result, input_path=args.input, effect_kind="auc")
    terminal_primitives = plot_payload(
        result, input_path=args.input, effect_kind="terminal_primitives"
    )
    auc_primitives = plot_payload(
        result, input_path=args.input, effect_kind="auc_primitives"
    )
    render(terminal, args.output)
    render(auc, args.auc_output)
    render(terminal_primitives, args.primitive_output)
    render(auc_primitives, args.primitive_auc_output)
    print(
        f"wrote {args.output.with_suffix('.pdf')}, "
        f"{args.output.with_suffix('.png')}, and "
        f"{args.output.with_suffix('.json')}; "
        f"{args.auc_output.with_suffix('.pdf')}, "
        f"{args.auc_output.with_suffix('.png')}, and "
        f"{args.auc_output.with_suffix('.json')}; "
        f"{args.primitive_output.with_suffix('.pdf')}, "
        f"{args.primitive_output.with_suffix('.png')}, and "
        f"{args.primitive_output.with_suffix('.json')}; "
        f"{args.primitive_auc_output.with_suffix('.pdf')}, "
        f"{args.primitive_auc_output.with_suffix('.png')}, and "
        f"{args.primitive_auc_output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
