#!/usr/bin/env python3
"""Fail-closed terminal audit for the E75F1 PointMaze paper cohort."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
ARMS = ("grpo", "verified_first_global_replay_canonical")
SEEDS = (43, 44, 45, 46, 47)
TRAIN_UPDATES = 4_608
EVAL_ROUNDS = tuple(range(0, TRAIN_UPDATES + 1, 96))
EXPECTED_SPLITS = {"train": 384, "dev": 64, "eval": 128}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def stem(arm: str, seed: int) -> Path:
    return ROOT / f"var/artifacts/e75f1_point_maze_waypoint_{arm}_s{seed}"


def add(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-identity", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh E75F1 audit output required: {args.output}")

    errors: list[str] = []
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    data_identity = json.loads(args.data_identity.read_text(encoding="utf-8"))
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    add(errors, identity.get("schema") == "e75f1-point-maze-waypoint-05b-identity-v1", "identity schema mismatch")
    add(errors, identity.get("protocol_sha256") == sha256_file(args.protocol), "protocol hash mismatch")
    add(errors, identity.get("data_seed") == 88104, "data seed mismatch")
    add(errors, identity.get("seeds") == list(SEEDS), "seed set mismatch")
    add(errors, identity.get("arms") == list(ARMS), "arm set mismatch")
    add(errors, data_identity.get("split_counts") == EXPECTED_SPLITS, "data split shape mismatch")
    certification = data_identity.get("certification", [])
    fingerprints = [str(row.get("instance_fingerprint", "")) for row in certification]
    add(errors, len(certification) == 576, "data certification count mismatch")
    add(errors, len(set(fingerprints)) == 576 and all(fingerprints), "data fingerprints are not unique")
    exclusions = data_identity.get("excluded_prior_identities", [])
    add(errors, len(exclusions) == 4, "predecessor exclusion count mismatch")
    add(errors, sum(int(row.get("fingerprint_count", 0)) for row in exclusions) == 640, "predecessor fingerprint total mismatch")
    add(errors, qualification.get("status") == "pass", "development qualification did not pass")
    add(errors, qualification.get("eligible_for_full_cohort") is True, "development gate did not authorize full cohort")

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    online = {(row.get("arm"), int(row.get("seed") or -1)) for row in manifest if row.get("stage") == "online"}
    add(errors, online == {(arm, seed) for arm in ARMS for seed in SEEDS}, "manifest Cartesian product mismatch")
    jobs = identity.get("jobs", {})
    add(errors, set(jobs.get("online", {})) == {f"{arm}:s{seed}" for arm in ARMS for seed in SEEDS}, "identity job set mismatch")

    expected_eval_ids = {
        str(row.get("map_id"))
        for row in certification
        if row.get("split") == "eval"
    }
    summaries: dict[str, dict[str, Any]] = {}
    initial_by_seed: dict[int, dict[str, Any]] = {}
    source_identities = []
    for arm in ARMS:
        for seed in SEEDS:
            label = f"{arm}:s{seed}"
            base = stem(arm, seed)
            receipt_path = Path(str(base) + ".json")
            metrics_path = Path(str(base) + ".metrics.jsonl")
            replay_path = Path(str(base) + ".replay.jsonl")
            model_path = ROOT / f"var/models/e75f1_point_maze_waypoint_{arm}_s{seed}"
            missing = [str(path) for path in (receipt_path, metrics_path, replay_path, model_path) if not path.exists()]
            if missing:
                errors.append(f"{label}: missing outputs {missing}")
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            add(errors, receipt.get("status") == "complete", f"{label}: receipt incomplete")
            add(errors, receipt.get("arm") == arm and receipt.get("seed") == seed, f"{label}: receipt cell mismatch")
            add(errors, receipt.get("passes") == 12 and receipt.get("optimizer_updates") == TRAIN_UPDATES, f"{label}: training horizon mismatch")
            add(errors, receipt.get("evaluation_coordinates") == len(EVAL_ROUNDS), f"{label}: evaluation coordinate count mismatch")
            add(errors, receipt.get("evaluation_split") == "eval" and receipt.get("evaluation_prompt_count") == 128, f"{label}: evaluation split mismatch")
            add(errors, receipt.get("data_identity_sha256") == sha256_file(args.data_identity), f"{label}: data hash mismatch")
            add(errors, receipt.get("metrics_sha256") == sha256_file(metrics_path), f"{label}: metrics hash mismatch")
            add(errors, receipt.get("state_replay_sha256") == sha256_file(replay_path), f"{label}: replay hash mismatch")
            add(errors, receipt.get("output_model_tree_sha256") == tree_sha256(model_path), f"{label}: model tree hash mismatch")
            mechanism = receipt.get("mechanism", {})
            add(errors, mechanism.get("gold_support_feedback") is False, f"{label}: gold feedback flag changed")
            add(errors, mechanism.get("dynamic_legal_support") == "adjacent free cells only", f"{label}: legal support changed")
            add(errors, mechanism.get("verified_novelty") is (arm != "grpo"), f"{label}: novelty mechanism mismatch")
            add(errors, mechanism.get("verified_replay") is (arm != "grpo"), f"{label}: replay mechanism mismatch")
            source_identities.append(receipt.get("source_sha256"))

            rows = load_jsonl(metrics_path)
            training = [row for row in rows if row.get("schema") == "point-maze-waypoint-pilot-training-v1"]
            evaluation = [row for row in rows if row.get("schema") == "point-maze-waypoint-pilot-evaluation-v1"]
            add(errors, len(rows) == TRAIN_UPDATES + len(EVAL_ROUNDS), f"{label}: metrics row count mismatch")
            add(errors, len(training) == TRAIN_UPDATES, f"{label}: training row count mismatch")
            add(errors, len(evaluation) == len(EVAL_ROUNDS), f"{label}: evaluation row count mismatch")
            add(errors, finite_tree(rows), f"{label}: nonfinite metric")
            add(errors, [int(row.get("learning_round", -1)) for row in training] == list(range(1, TRAIN_UPDATES + 1)), f"{label}: training rounds changed")
            add(errors, [int(row.get("row_index", -1)) for row in training] == [index % 384 for index in range(TRAIN_UPDATES)], f"{label}: train order changed")
            add(errors, [int(row.get("learning_round", -1)) for row in evaluation] == list(EVAL_ROUNDS), f"{label}: evaluation cadence changed")
            add(errors, all(int(row.get("action_support_escapes", -1)) == 0 for row in training), f"{label}: action support escape")
            for row in evaluation:
                per_map = row.get("per_map", [])
                ids = [str(item.get("map_id", "")) for item in per_map]
                add(errors, row.get("evaluation_prompt_count") == 128 and row.get("evaluation_trajectory_count") == 1024, f"{label}: evaluation sample count changed")
                add(errors, len(ids) == 128 and set(ids) == expected_eval_ids, f"{label}: evaluation map set changed")
            if training:
                raw_replay = [float(row.get("replay_raw_score_gradient_l2", 0.0)) for row in training]
                applied_replay = [float(row.get("replay_applied_score_gradient_l2", 0.0)) for row in training]
                raw_exploration = [float(row.get("raw_exploration_advantage_rms", 0.0)) for row in training]
                applied_exploration = [float(row.get("applied_exploration_advantage_rms", 0.0)) for row in training]
                add(errors, any(value > 0.0 for value in raw_replay), f"{label}: no raw replay signal")
                add(errors, any(value > 0.0 for value in raw_exploration), f"{label}: no raw exploration signal")
                if arm == "grpo":
                    add(errors, all(value == 0.0 for value in applied_replay), f"{label}: control replay derivative nonzero")
                    add(errors, all(value == 0.0 for value in applied_exploration), f"{label}: control exploration derivative nonzero")
                    add(errors, all(float(row.get("replay_compute_only", 0.0)) == 1.0 for row in training), f"{label}: control compute-match flag changed")
                else:
                    add(errors, any(value > 0.0 for value in applied_replay), f"{label}: treatment replay never applied")
                    add(errors, any(value > 0.0 for value in applied_exploration), f"{label}: treatment exploration never applied")
                    add(errors, all(float(row.get("replay_compute_only", 1.0)) == 0.0 for row in training), f"{label}: treatment compute-match flag changed")
            if evaluation:
                initial = evaluation[0]
                if arm == "grpo":
                    initial_by_seed[seed] = initial
                else:
                    baseline = initial_by_seed.get(seed)
                    if baseline is not None:
                        keys = ("mean8", "pass8", "distinct8", "modes_per_success")
                        add(errors, all(initial.get(key) == baseline.get(key) for key in keys), f"{label}: paired initialization differs")
                terminal = evaluation[-1]
                summaries[label] = {key: terminal.get(key) for key in ("mean8", "pass8", "distinct8", "modes_per_success")}

    add(errors, len(source_identities) == 10 and all(value == source_identities[0] for value in source_identities), "source identity differs across cells")
    paired_distinct = []
    for seed in SEEDS:
        control = summaries.get(f"grpo:s{seed}", {}).get("distinct8")
        treatment = summaries.get(f"verified_first_global_replay_canonical:s{seed}", {}).get("distinct8")
        if control is not None and treatment is not None:
            paired_distinct.append(float(treatment) - float(control))
    result = {
        "schema": "e75f1-point-maze-waypoint-terminal-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "eligible_for_paper_integration": not errors,
        "efficacy_threshold_used": False,
        "cells_complete": len(summaries),
        "terminal": summaries,
        "paired_distinct8_deltas": paired_distinct,
        "mean_paired_distinct8_delta": (
            sum(paired_distinct) / len(paired_distinct)
            if paired_distinct
            else None
        ),
        "errors": errors,
        "identity_sha256": sha256_file(args.identity),
        "manifest_sha256": sha256_file(args.manifest),
        "data_identity_sha256": sha256_file(args.data_identity),
        "qualification_sha256": sha256_file(args.qualification),
        "protocol_sha256": sha256_file(args.protocol),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
