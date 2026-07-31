#!/usr/bin/env python3
"""Audit real AntMaze routes, topology perturbations, identity, and throughput."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

from datasets import load_from_disk


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[1]))
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.ant_maze_worker_v5 import RECEIPT_SHA256  # noqa: E402
from oat_drgrpo.maze_modebench import (  # noqa: E402
    MazeModeBenchError,
    parse_maze_action_program,
    parse_maze_action_spec,
    validate_maze_execution,
)
from oat_drgrpo.maze_modebench_process import MazeVerifierProcess  # noqa: E402


PERTURBATIONS_PER_ROUTE = 100
THROUGHPUT_FLOOR = 0.15
CONTROLLER_RECEIPT_SHA256 = RECEIPT_SHA256
DATA_SCHEMA = "ant-maze-modebench-data-v1"
AUDIT_SCHEMA = "ant-maze-modebench-admission-audit-v1"
DEFAULT_DATA_ROOT = ROOT / "var/data/ant_maze_modebench_v1"
DEFAULT_OUTPUT = ROOT / "var/artifacts/ant_maze_modebench_v1_admission_audit.json"
ANT_WORKER_SOURCE = "ant_maze_worker_v5.py"
DECISION = "admitted_for_0.5b_viability_sampling"
VERSION_LABEL = "ant-maze-audit"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=ROOT / "var/maze_runtime/venv/bin/python",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def _expect_rejection(callback, label: str) -> None:
    try:
        callback()
    except (MazeModeBenchError, ValueError):
        return
    raise AssertionError(f"AntMaze adversarial case was accepted: {label}")


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh AntMaze audit required: {args.output}")
    identity_path = args.data_root / "identity.json"
    identity = json.loads(identity_path.read_text())
    if identity.get("schema_version") != DATA_SCHEMA:
        raise ValueError("AntMaze data identity schema mismatch")
    if identity.get("controller_receipt_sha256") != CONTROLLER_RECEIPT_SHA256:
        raise ValueError("AntMaze data binds the wrong controller")

    rows_by_map = {}
    split_hashes = {}
    split_fingerprints = {}
    for split, dataset_split in (
        ("train", "train"),
        ("dev", "multi_answer"),
        ("eval", "multi_answer"),
    ):
        dataset = load_from_disk(str(args.data_root / split))
        if set(dataset) != {dataset_split}:
            raise ValueError(f"AntMaze {split} dataset split mismatch")
        rows = dataset[dataset_split].to_list()
        if len(rows) != 4:
            raise ValueError(f"AntMaze {split} must contain four rows")
        split_hashes[split] = _canonical_sha256(rows)
        split_fingerprints[split] = {
            str(row["instance_fingerprint"]) for row in rows
        }
        for row in rows:
            spec = json.loads(row["answer"])
            parsed = parse_maze_action_spec(spec)
            if parsed.spec_sha256 != row["instance_fingerprint"]:
                raise ValueError("AntMaze row fingerprint mismatch")
            if row["answer_mode_split"] != split:
                raise ValueError("AntMaze row split marker mismatch")
            if int(row["answer_mode_count"]) != 2:
                raise ValueError("AntMaze row does not declare two modes")
            rows_by_map[parsed.map_id] = (row, spec)
    if any(
        split_fingerprints[left] & split_fingerprints[right]
        for left, right in (("train", "dev"), ("train", "eval"), ("dev", "eval"))
    ):
        raise ValueError("AntMaze split fingerprints overlap")
    if split_hashes != identity.get("split_rows_sha256"):
        raise ValueError("AntMaze split row hashes differ from identity")

    certification = identity.get("certification")
    if not isinstance(certification, list) or len(certification) != 12:
        raise ValueError("AntMaze identity must certify 12 maps")
    real_records = []
    perturbation_count = 0
    verifier = MazeVerifierProcess(
        timeout_seconds=25.0,
        worker_python=args.worker_python,
    )
    started = time.monotonic()
    try:
        for map_record in certification:
            map_id = str(map_record["map_id"])
            _row, spec = rows_by_map[map_id]
            route_keys = set()
            for route in map_record["routes"]:
                program = str(route["program"])
                result = verifier.validate_with_execution(program, spec)
                if result is None:
                    raise ValueError(f"{map_id} real route replay failed")
                validation, execution = result
                if validation.canonical_key != route["canonical_key"]:
                    raise ValueError(f"{map_id} route key changed on real replay")
                route_keys.add(validation.canonical_key)
                real_records.append(
                    {
                        "map_id": map_id,
                        "route_name": route["route_name"],
                        "canonical_key": validation.canonical_key,
                        "simulator_steps": validation.simulator_steps,
                    }
                )
                stored_execution = route["execution"]
                rng = random.Random(
                    int(hashlib.sha256(
                        f"{map_id}:{route['route_name']}".encode("ascii")
                    ).hexdigest()[:16], 16)
                )
                for _ in range(PERTURBATIONS_PER_ROUTE):
                    perturbed = copy.deepcopy(stored_execution)
                    trajectory = perturbed["trajectory_xy"]
                    for point in trajectory[1:-1]:
                        point[0] += rng.uniform(-0.02, 0.02)
                        point[1] += rng.uniform(-0.02, 0.02)
                    replay = validate_maze_execution(program, spec, perturbed)
                    if replay.canonical_key != validation.canonical_key:
                        raise ValueError(
                            f"{map_id} topology key changed under perturbation"
                        )
                    perturbation_count += 1
            if len(route_keys) != 2:
                raise ValueError(f"{map_id} real route keys collided")
    finally:
        verifier.close()
    wall_seconds = time.monotonic() - started
    throughput = len(real_records) / wall_seconds
    if throughput < THROUGHPUT_FLOOR:
        raise ValueError(
            f"AntMaze throughput {throughput:.3f}/s is below {THROUGHPUT_FLOOR}/s"
        )
    if perturbation_count != 2_400:
        raise ValueError("AntMaze perturbation replay count mismatch")

    example = certification[0]["routes"][0]
    example_spec = rows_by_map[certification[0]["map_id"]][1]
    parse_maze_action_program(example["program"], parse_maze_action_spec(example_spec))
    _expect_rejection(
        lambda: parse_maze_action_program(
            example["program"] + " STOP",
            parse_maze_action_spec(example_spec),
        ),
        "removed STOP token",
    )
    for field in ("environment_sha256", "controller_sha256"):
        mutated = copy.deepcopy(example["execution"])
        mutated[field] = "0" * 64
        _expect_rejection(
            lambda value=mutated: validate_maze_execution(
                example["program"], example_spec, value
            ),
            f"{field} mutation",
        )
    near_miss = copy.deepcopy(example["execution"])
    near_miss["success"] = False
    near_miss["final_goal_distance"] = 0.500001
    _expect_rejection(
        lambda: validate_maze_execution(example["program"], example_spec, near_miss),
        "unsuccessful goal near miss",
    )

    payload = {
        "schema_version": AUDIT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": DECISION,
        "maps": len(certification),
        "split_rows": identity["split_rows"],
        "split_overlap_count": 0,
        "real_simulator_replays": len(real_records),
        "keys_per_map": 2,
        "perturbations_per_route": PERTURBATIONS_PER_ROUTE,
        "perturbation_replays": perturbation_count,
        "throughput": {
            "wall_seconds": wall_seconds,
            "executions_per_second": throughput,
            "floor": THROUGHPUT_FLOOR,
        },
        "controller_receipt_sha256": CONTROLLER_RECEIPT_SHA256,
        "dataset_identity_sha256": _canonical_sha256(identity),
        "source_hashes": {
            "audit": _sha256_file(Path(__file__).resolve()),
            "verifier": _sha256_file(SRC / "oat_drgrpo/maze_modebench.py"),
            "worker": _sha256_file(SRC / "oat_drgrpo/maze_modebench_worker.py"),
            "ant_worker": _sha256_file(SRC / f"oat_drgrpo/{ANT_WORKER_SOURCE}"),
        },
        "real_records": real_records,
        "information_boundary": {
            "language_model_sampled": False,
            "route_fixtures_in_prompts": False,
            "evaluation_outcomes_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        f"[{VERSION_LABEL}] status=pass maps=12 real=24 "
        f"perturbations=2400 throughput={throughput:.3f}/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
