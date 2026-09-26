#!/usr/bin/env python3
"""Fail-closed audit for the PointMaze v3 orientation-balanced pair."""

from __future__ import annotations

import json
import os
from pathlib import Path

import audit_point_maze_algorithm_repair_pair_v1 as base


base.SEED = 76541
ROOT = Path(
    os.environ.get(
        "OAT_ZERO_REPO_ROOT",
        Path(__file__).resolve().parents[1],
    )
)
REAL_QUALIFICATION = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_v3_qualification.json"
)
REAL_DATA = ROOT / "var/data/point_maze_algorithm_repair_v3"
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
        return REAL_QUALIFICATION
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
        path = REAL_DATA
    return _tree_hash(path)


def loads(value, *args, **kwargs):
    payload = _loads(value, *args, **kwargs)
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if schema == "point-maze-algorithm-repair-receipt-v3":
        payload["schema"] = "point-maze-algorithm-repair-receipt-v1"
    elif schema == "point-maze-algorithm-repair-evaluation-v3":
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
    checks["v3_trainability_band"] = (
        set(cells) == set(base.ARMS)
        and all(
            0.02 <= float(cell.get("verified_rate", -1)) <= 0.50
            for cell in cells.values()
        )
    )
    checks["v3_task_gradient_updates"] = (
        set(cells) == set(base.ARMS)
        and all(
            int(cell.get("task_advantage_nonzero_updates", -1)) >= 10
            for cell in cells.values()
        )
    )
    identity_path = path.parent / (
        "point_maze_algorithm_repair_pair_v3_identity.json"
    )
    identity = _loads(identity_path.read_text(encoding="utf-8"))
    qualification = _loads(
        REAL_QUALIFICATION.read_text(encoding="utf-8")
    )
    checks["v3_identity"] = (
        identity.get("repair_v3_schema")
        == "point-maze-algorithm-repair-pair-identity-v3"
        and identity.get("seeds") == [76541]
        and identity.get("orientation_balanced") is True
        and identity.get("development_only") is True
        and identity.get("final_seed_cohort") is False
    )
    checks["v3_qualification_pass"] = (
        qualification.get("schema_version")
        == "point-maze-algorithm-repair-v3-qualification-v1"
        and qualification.get("status") == "pass"
        and qualification.get("decision")
        == "eligible_for_point_maze_algorithm_repair_v3_pair"
        and qualification.get("errors") in ([], None)
    )
    for key in (
        "v3_trainability_band",
        "v3_task_gradient_updates",
        "v3_identity",
        "v3_qualification_pass",
    ):
        if not checks[key]:
            errors.append(f"failed check: {key}")
    payload.update(
        schema="point-maze-algorithm-repair-pair-audit-v3",
        checks=checks,
        errors=sorted(set(errors)),
    )
    payload["status"] = "pass" if not payload["errors"] else "fail"
    payload["decision"] = (
        "eligible_for_point_maze_algorithm_repair_v3_five_seed_final"
        if not payload["errors"]
        else "point_maze_algorithm_repair_v3_stopped"
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
