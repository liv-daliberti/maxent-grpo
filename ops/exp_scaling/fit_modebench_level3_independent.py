#!/usr/bin/env python3
"""Fit unchanged difficulty gates using authenticated independent RNG receipts.

The existing mixture algorithm is retained exactly. This entry point adds the
v2 seed-contract checks before calling it, then records both fitter sources.
Legacy overlapping-seed receipts cannot enter this revision's development fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "ops", ROOT / "ops/exp_scaling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fit_modebench_level3 as mixture
from evaluate_modebench_level3_independent import validate_seed_receipt
from modebench_independent_seeds import POLICY

DEVELOPMENT_DRAW_LABELS = (6328000, 6328001, 6328002, 6328003)


def fit_recipe(baseline_path, score_paths, domain, output=None):
    paths = [Path(baseline_path), *map(Path, score_paths)]
    before = {str(path): mixture.file_sha(path) for path in paths}
    if len(paths) != 5 or len(before) != 5:
        raise ValueError("one baseline and four distinct candidate receipts required")
    for path in paths:
        receipt = json.loads(path.read_text())
        if (receipt.get("split") != "dev" or receipt.get("domain") != domain
                or receipt.get("identity", {}).get("seeds") != list(DEVELOPMENT_DRAW_LABELS)):
            raise ValueError("only registered v2 development receipts may enter fitting")
        rows = mixture.receipt_rows(receipt)
        validate_seed_receipt(receipt, rows=rows)
    result = mixture.fit_recipe(baseline_path, score_paths, domain,
                                seed=mixture.SELECTION_SEED)
    if before != {str(path): mixture.file_sha(path) for path in paths}:
        raise ValueError("development receipt changed during authenticated fitting")
    result["provenance"]["independent_fitter_source_sha256"] = mixture.file_sha(Path(__file__))
    result["sampling_validation"] = {
        "revision": 2,
        "policy": POLICY,
        "all_five_receipts_authenticated": True,
        "per_prompt_draw_seed_blocks_disjoint": True,
        "development_draw_labels": list(DEVELOPMENT_DRAW_LABELS),
        "legacy_receipts_used": False,
    }
    result["information_boundary"]["prior_v1_confirmation"] = (
        "Retained as diagnostic; its discovered RNG defect motivated this revision. "
        "No v1 confirmation scores enter mixture ranking or selected-row checks."
    )
    if output is not None:
        mixture.atomic_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--scores", required=True, nargs=4, type=Path)
    parser.add_argument("--domain", required=True, choices=mixture.DOMAINS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = fit_recipe(args.baseline, args.scores, args.domain, args.output)
    print(json.dumps({"domain": args.domain, "decision": result["decision"],
                      "weights": result["weights"], "development": result["development"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
