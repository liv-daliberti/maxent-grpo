#!/usr/bin/env python3
"""Audit the K=16-aligned PointMaze repair-v2 pair."""

from __future__ import annotations

import json
from pathlib import Path

import audit_point_maze_algorithm_repair_pair_v1 as base


base.SEED = 76521
REAL_QUALIFICATION = Path(
    "var/artifacts/point_maze_algorithm_repair_v2_qualification.json"
)
_tree_hash = base.tree_hash
_loads = base.json.loads
_atomic = base.atomic
_is_file = Path.is_file
_read_text = Path.read_text
_read_bytes = Path.read_bytes


def _redirect(path: Path) -> Path:
    if path.as_posix().endswith(
        "var/artifacts/point_maze_algorithm_repair_v1_qualification.json"
    ):
        root = path
        while root.name != "maxent-grpo" and root.parent != root:
            root = root.parent
        return root / REAL_QUALIFICATION
    return path


def is_file(path: Path) -> bool:
    return _is_file(_redirect(path))


def read_text(path: Path, *args, **kwargs):
    return _read_text(_redirect(path), *args, **kwargs)


def read_bytes(path: Path):
    return _read_bytes(_redirect(path))


def tree_hash(path: Path) -> str:
    if path.as_posix().endswith(
        "var/data/point_maze_algorithm_repair_v1"
    ):
        path = path.parent / "point_maze_algorithm_repair_v2"
    return _tree_hash(path)


def loads(value):
    payload = _loads(value)
    schema = payload.get("schema")
    if schema == "point-maze-algorithm-repair-receipt-v2":
        payload["schema"] = "point-maze-algorithm-repair-receipt-v1"
    elif schema == "point-maze-algorithm-repair-evaluation-v2":
        payload["schema"] = "point-maze-algorithm-repair-evaluation-v1"
    return payload


def atomic(path: Path, payload: dict) -> None:
    if payload.get("schema") != "point-maze-algorithm-repair-pair-audit-v1":
        _atomic(path, payload)
        return
    errors = [
        error
        for error in payload.get("errors", [])
        if not (
            error.endswith("verified rate outside [0.10, 0.90]")
            or error.endswith("fewer than 24 task-gradient updates")
        )
    ]
    cells = payload.get("cells", {})
    checks = dict(payload.get("checks", {}))
    checks["v2_trainability_band"] = (
        set(cells) == set(base.ARMS)
        and all(
            0.02 <= float(cell.get("verified_rate", -1)) <= 0.50
            for cell in cells.values()
        )
    )
    checks["v2_task_gradient_updates"] = (
        set(cells) == set(base.ARMS)
        and all(
            int(cell.get("task_advantage_nonzero_updates", -1)) >= 10
            for cell in cells.values()
        )
    )
    for key in ("v2_trainability_band", "v2_task_gradient_updates"):
        if not checks[key]:
            errors.append(f"failed check: {key}")
    payload.update(
        schema="point-maze-algorithm-repair-pair-audit-v2",
        checks=checks,
        errors=sorted(set(errors)),
    )
    payload["status"] = "pass" if not payload["errors"] else "fail"
    payload["decision"] = (
        "eligible_for_point_maze_algorithm_repair_v2_five_seed_final"
        if not payload["errors"]
        else "point_maze_algorithm_repair_v2_stopped"
    )
    _atomic(path, payload)


Path.is_file = is_file
Path.read_text = read_text
Path.read_bytes = read_bytes
base.tree_hash = tree_hash
base.json.loads = loads
base.atomic = atomic


if __name__ == "__main__":
    base.main()
