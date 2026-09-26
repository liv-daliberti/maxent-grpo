#!/usr/bin/env python3
"""Render the compiled supporting figures from retained numerical records only.

This deliberately bypasses live-data build() functions. New observations require
an explicit audited result refresh before this rendering step.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
FIGURES = ROOT / "paper/figures"
SPECS = {
    "baseline_collapse_precheck": (
        "plot_paper_baseline_collapse_precheck", "render",
        "paper/results/baseline_collapse_precheck.json"),
    "direct_comparator_endpoint_effects": (
        "plot_paper_direct_comparator_endpoint_effects", "render",
        "paper/figures/direct_comparator_endpoint_effects.json"),
    "direct_baseline_learning_curves_static_strip": (
        "plot_paper_aligned_domain_strips", "render_ucpo",
        "paper/figures/direct_baseline_learning_curves_static_strip.json"),
    "direct_baseline_learning_curves_pass8": (
        "plot_paper_aligned_domain_strips", "render_ucpo",
        "paper/figures/direct_baseline_learning_curves_static_strip.json"),
    "e121_fixed_bank_survival": (
        "build_paper_e121_survival", "plot",
        "paper/results/e121_fixed_bank_survival.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", *SPECS], default="all")
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = SPECS if args.figure == "all" else [args.figure]
    for key in keys:
        module_name, function, source = SPECS[key]
        path = ROOT / source
        before = path.read_bytes()
        payload = json.loads(before)
        module = importlib.import_module(module_name)
        output = args.output_dir / key
        if function == "render_ucpo":
            module.render_ucpo(
                snapshot=payload, output=output,
                metric="pass8" if key == "direct_baseline_learning_curves_pass8" else "distinct8",
            )
        elif function == "plot":
            module.FIGURE = output
            module.plot(payload)
        else:
            module.render(payload, output)
        if path.read_bytes() != before:
            raise RuntimeError(f"retained numerical source changed while rendering: {source}")
        print(f"{output.with_suffix('.pdf')} (input sha256 {hashlib.sha256(before).hexdigest()})",
              flush=True)


if __name__ == "__main__":
    main()
