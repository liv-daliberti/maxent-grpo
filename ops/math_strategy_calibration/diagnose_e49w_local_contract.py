#!/usr/bin/env python3
"""Reproduce one E49W proposal and expose deterministic parser failures."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49w_bottom_up_route_calibration.py"
)


def _load_source() -> Any:
    spec = importlib.util.spec_from_file_location("e49w_diagnostic_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E49W source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=(
            ROOT
            / "var/artifacts/"
            "e49w_local_contract_diagnostic_v1.json"
        ),
    )
    args = parser.parse_args()

    source = _load_source()
    endpoint, model = source._endpoint(source.ENDPOINT_RECORD)
    candidates = source._load_candidates()
    if not 0 <= args.candidate_rank < len(candidates):
        raise RuntimeError("candidate rank is out of bounds")
    candidate = candidates[args.candidate_rank]
    original_contract = source._proposal_passes_local_contract
    source._proposal_passes_local_contract = (
        lambda proposal, sample_ids: True
    )
    proposal, generation = source._propose(
        endpoint=endpoint,
        model=model,
        candidate=candidate,
        timeout=args.timeout,
    )
    sample_ids = {
        str(row["sample_id"]) for row in candidate["exemplars"]
    }
    result = {
        "schema": "e49w_local_contract_diagnostic_v1",
        "candidate_rank": int(candidate["candidate_rank"]),
        "problem_id": str(candidate["problem_id"]),
        "row_id": str(candidate["row_id"]),
        "problem_sha256": source._sha256_text(candidate["problem"]),
        "selected_exemplar_ids": sorted(sample_ids),
        "generation": generation,
        "proposal": proposal,
        "local_contract_pass": original_contract(proposal, sample_ids),
        "local_contract_failures": (
            source._proposal_local_contract_failures(proposal, sample_ids)
        ),
        "diagnostic_source_sha256": source._sha256(
            pathlib.Path(__file__).resolve()
        ),
        "generator_source_sha256": source._sha256(SOURCE),
    }
    source._write_json(args.out.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
