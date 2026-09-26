#!/usr/bin/env python3
"""Qualify the frozen E75 PointMaze waypoint development gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


EXPECTED_FAMILIES = {"barrier_3mode", "barrier_4mode", "barrier_5mode"}
EXPECTED_MAPS = 32
EXPECTED_K = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def qualify(
    *,
    receipt: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    data_identity_sha256: str,
    metrics_sha256: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if receipt.get("status") != "complete":
        reasons.append("evaluation receipt is incomplete")
    if receipt.get("evaluation_only") is not True:
        reasons.append("development gate was not evaluation-only")
    if receipt.get("optimizer_updates") != 0:
        reasons.append("development gate performed optimizer updates")
    if receipt.get("evaluation_split") != "dev":
        reasons.append("development gate used the wrong split")
    if receipt.get("evaluation_prompt_count") != EXPECTED_MAPS:
        reasons.append("development gate did not cover all 32 maps")
    if receipt.get("evaluation_coordinates") != 1:
        reasons.append("development gate must contain exactly one coordinate")
    if receipt.get("data_identity_sha256") != data_identity_sha256:
        reasons.append("development receipt/data identity mismatch")
    if receipt.get("metrics_sha256") != metrics_sha256:
        reasons.append("development receipt/metrics hash mismatch")

    per_map = evaluation.get("per_map")
    if not isinstance(per_map, list) or len(per_map) != EXPECTED_MAPS:
        reasons.append("development metric does not contain 32 map rows")
        per_map = []
    if evaluation.get("schema") != "point-maze-waypoint-pilot-evaluation-v1":
        reasons.append("development metric schema changed")
    if evaluation.get("split") != "dev":
        reasons.append("development metric split changed")
    if evaluation.get("evaluation_prompt_count") != EXPECTED_MAPS:
        reasons.append("development metric prompt count changed")
    if evaluation.get("evaluation_trajectory_count") != EXPECTED_MAPS * EXPECTED_K:
        reasons.append("development trajectory count changed")

    map_ids = [str(row.get("map_id", "")) for row in per_map]
    if len(set(map_ids)) != len(map_ids) or any(not value for value in map_ids):
        reasons.append("development map IDs are empty or duplicated")
    successful_maps = sum(float(row.get("pass8", 0.0)) >= 1.0 for row in per_map)
    multimode_maps = sum(float(row.get("distinct8", 0.0)) >= 2.0 for row in per_map)
    mean8 = float(evaluation.get("mean8", -1.0))
    family_successes: dict[str, int] = {}
    for row in per_map:
        family = str(row.get("family", ""))
        family_successes[family] = family_successes.get(family, 0) + int(
            float(row.get("pass8", 0.0)) >= 1.0
        )
    observed_families = set(family_successes)
    if observed_families != EXPECTED_FAMILIES:
        reasons.append("development geometry strata changed")
    dead_families = sorted(
        family for family in EXPECTED_FAMILIES if family_successes.get(family, 0) == 0
    )

    if successful_maps < 24:
        reasons.append("fewer than 24/32 development maps have pass@8")
    if multimode_maps < 16:
        reasons.append("fewer than 16/32 development maps expose two routes")
    if dead_families:
        reasons.append("one or more geometry strata are entirely dead")
    if not 0.10 <= mean8 <= 0.70:
        reasons.append("aggregate development mean8 is outside [0.10, 0.70]")

    return {
        "schema": "e75-point-maze-waypoint-dev-qualification-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not reasons else "fail",
        "eligible_for_online_pilot": not reasons,
        "thresholds": {
            "maps_with_pass8_minimum": 24,
            "maps_with_two_routes_minimum": 16,
            "required_families": sorted(EXPECTED_FAMILIES),
            "mean8_interval": [0.10, 0.70],
        },
        "observed": {
            "map_count": len(per_map),
            "maps_with_pass8": successful_maps,
            "maps_with_two_routes": multimode_maps,
            "mean8": mean8,
            "family_successes": family_successes,
            "dead_families": dead_families,
        },
        "receipt_data_identity_sha256": receipt.get("data_identity_sha256"),
        "receipt_metrics_sha256": receipt.get("metrics_sha256"),
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--data-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh E75 qualification required: {args.output}")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    evaluations = []
    for line in args.metrics.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema") == "point-maze-waypoint-pilot-evaluation-v1":
            evaluations.append(row)
    if len(evaluations) != 1:
        raise ValueError("E75 development gate requires exactly one evaluation row")
    result = qualify(
        receipt=receipt,
        evaluation=evaluations[0],
        data_identity_sha256=sha256_file(args.data_identity),
        metrics_sha256=sha256_file(args.metrics),
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
