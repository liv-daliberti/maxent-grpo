#!/usr/bin/env python3
"""Run the frozen PointMaze runtime, topology, perturbation, and speed audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import select
import subprocess
import sys
import time
from typing import Any

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.maze_modebench import (  # noqa: E402
    MazeModeBenchError,
    validate_maze_execution,
)


DEFAULT_DATA = ROOT / "var/data/point_maze_modebench_v1"
DEFAULT_WORKER_PYTHON = ROOT / "var/maze_runtime/venv/bin/python"
DEFAULT_OUTPUT = ROOT / "var/artifacts/point_maze_modebench_v1_admission_audit.json"
PERTURBATIONS_PER_ROUTE = 100
MIN_EXECUTIONS_PER_SECOND = 2.0


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_identity(worker_python: Path) -> dict[str, Any]:
    source = (
        "import json,sys;"
        "sys.path.insert(0,'src');"
        "from oat_drgrpo.maze_runtime_identity import maze_runtime_identity;"
        "print(json.dumps(maze_runtime_identity(),sort_keys=True))"
    )
    completed = subprocess.run(
        [str(worker_python), "-c", source],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.splitlines()[-1])


class _RawWorker:
    def __init__(self, worker_python: Path) -> None:
        source = (
            "import runpy,sys;"
            f"sys.path.insert(0,{str(SRC)!r});"
            "runpy.run_module('oat_drgrpo.maze_modebench_worker',run_name='__main__')"
        )
        environment = os.environ.copy()
        environment.setdefault("MUJOCO_GL", "egl")
        self.process = subprocess.Popen(
            [str(worker_python), "-I", "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
            env=environment,
        )

    def request(self, candidate: str, spec: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(
            json.dumps({"candidate": candidate, "spec": spec}, allow_nan=False) + "\n"
        )
        self.process.stdin.flush()
        readable, _, _ = select.select([self.process.stdout], [], [], 25.0)
        if not readable:
            raise TimeoutError("PointMaze worker exceeded 25 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise BrokenPipeError("PointMaze worker closed stdout")
        return json.loads(line)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=2)


def _load_rows(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": load_from_disk(str(data_root / "train"))["train"].to_list(),
        "dev": load_from_disk(str(data_root / "dev"))["multi_answer"].to_list(),
        "eval": load_from_disk(str(data_root / "eval"))["multi_answer"].to_list(),
    }


def _perturb_trajectory(
    trajectory: list[list[float]],
    *,
    seed: int,
) -> list[list[float]]:
    rng = random.Random(seed)
    perturbed = [list(point) for point in trajectory]
    for index in range(1, len(perturbed) - 1):
        perturbed[index][0] += rng.uniform(-0.001, 0.001)
        perturbed[index][1] += rng.uniform(-0.001, 0.001)
    if len(perturbed) > 4 and seed % 2:
        index = 1 + seed % (len(perturbed) - 2)
        perturbed[index] = [
            (perturbed[index - 1][0] + perturbed[index + 1][0]) / 2.0,
            (perturbed[index - 1][1] + perturbed[index + 1][1]) / 2.0,
        ]
    return perturbed


def _must_reject(candidate: str, spec: dict, execution: dict, label: str) -> None:
    try:
        validate_maze_execution(candidate, spec, execution)
    except MazeModeBenchError:
        return
    raise RuntimeError(f"{label} mutation did not fail closed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = json.loads((args.data_root / "identity.json").read_text())
    rows_by_split = _load_rows(args.data_root)
    runtime = _runtime_identity(args.worker_python)
    if runtime != identity["runtime_identity"]:
        raise RuntimeError("installed maze runtime differs from data identity")
    if _sha256_file(ROOT / "src/oat_drgrpo/maze_modebench_worker.py") != identity[
        "worker_source_sha256"
    ]:
        raise RuntimeError("maze worker source differs from data identity")
    if _sha256_file(ROOT / "src/oat_drgrpo/maze_modebench.py") != identity[
        "verifier_source_sha256"
    ]:
        raise RuntimeError("maze verifier source differs from data identity")
    for split, rows in rows_by_split.items():
        if len(rows) != identity["split_rows"][split]:
            raise RuntimeError(f"{split} row count differs from identity")
        if _canonical_sha256(rows) != identity["split_rows_sha256"][split]:
            raise RuntimeError(f"{split} row hash differs from identity")
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    row_by_map = {
        json.loads(row["answer"])["map_id"]: row for row in all_rows
    }
    if len(row_by_map) != len(all_rows):
        raise RuntimeError("PointMaze map IDs are not unique")

    worker = _RawWorker(args.worker_python)
    real_records = []
    start = time.monotonic()
    try:
        for certificate in identity["certification"]:
            row = row_by_map[certificate["map_id"]]
            spec = json.loads(row["answer"])
            observed_keys = set()
            for expected in certificate["routes"]:
                payload = worker.request(expected["program"], spec)
                if not payload.get("valid"):
                    raise RuntimeError(
                        f"{certificate['map_id']} real replay failed"
                    )
                if payload["canonical_key"] != expected["canonical_key"]:
                    raise RuntimeError(
                        f"{certificate['map_id']} real replay key changed"
                    )
                validation = validate_maze_execution(
                    expected["program"], spec, payload["execution"]
                )
                if validation.canonical_key != expected["canonical_key"]:
                    raise RuntimeError("local and external route keys disagree")
                observed_keys.add(validation.canonical_key)
                real_records.append(
                    {
                        "map_id": certificate["map_id"],
                        "split": certificate["split"],
                        "candidate": expected["program"],
                        "spec": spec,
                        "key": validation.canonical_key,
                        "execution": payload["execution"],
                    }
                )
            if len(observed_keys) != 2:
                raise RuntimeError(
                    f"{certificate['map_id']} route keys collide"
                )
    finally:
        worker.close()
    elapsed = time.monotonic() - start
    throughput = len(real_records) / elapsed
    if throughput < MIN_EXECUTIONS_PER_SECOND:
        raise RuntimeError("PointMaze worker throughput is below the frozen floor")

    perturbation_replays = 0
    for route_index, record in enumerate(real_records):
        for perturbation_index in range(PERTURBATIONS_PER_ROUTE):
            execution = dict(record["execution"])
            execution["trajectory_xy"] = _perturb_trajectory(
                execution["trajectory_xy"],
                seed=100_000 * route_index + perturbation_index,
            )
            validation = validate_maze_execution(
                record["candidate"], record["spec"], execution
            )
            if validation.canonical_key != record["key"]:
                raise RuntimeError("PointMaze perturbation changed route key")
            perturbation_replays += 1

    first = real_records[0]
    failed = dict(first["execution"])
    failed["success"] = False
    _must_reject(first["candidate"], first["spec"], failed, "unsuccessful")
    near_miss = dict(first["execution"])
    near_miss["final_goal_distance"] = first["spec"]["success_threshold"] + 0.01
    _must_reject(first["candidate"], first["spec"], near_miss, "near-miss")
    wrong_hash = dict(first["execution"])
    wrong_hash["environment_sha256"] = "0" * 64
    _must_reject(first["candidate"], first["spec"], wrong_hash, "hash-mismatch")

    receipt = {
        "schema_version": "point-maze-modebench-admission-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": "admitted_for_0.5b_viability_sampling",
        "real_simulator_replays": len(real_records),
        "maps": len(row_by_map),
        "families": len(identity["families"]),
        "keys_per_map": 2,
        "perturbations_per_route": PERTURBATIONS_PER_ROUTE,
        "perturbation_replays": perturbation_replays,
        "perturbation_failures": 0,
        "failure_mutations_rejected": [
            "unsuccessful",
            "near_miss",
            "environment_hash_mismatch",
        ],
        "throughput": {
            "elapsed_seconds": elapsed,
            "executions_per_second": throughput,
            "frozen_minimum_executions_per_second": MIN_EXECUTIONS_PER_SECOND,
        },
        "runtime_identity": runtime,
        "dataset_identity_sha256": _canonical_sha256(identity),
        "audit_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        "[point-maze-audit] "
        f"status=pass real={len(real_records)} perturbations={perturbation_replays} "
        f"throughput={throughput:.3f}/s"
    )


if __name__ == "__main__":
    main()
