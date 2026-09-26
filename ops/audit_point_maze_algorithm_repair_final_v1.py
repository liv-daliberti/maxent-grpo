#!/usr/bin/env python3
"""Fail-closed ten-cell audit for the PointMaze repair-v2 final."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import datasets

import audit_point_maze_stage_b_05b_12pass as base


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (76531, 76532, 76533, 76534, 76535)
REAL_DATA = ROOT / "var/data/point_maze_algorithm_repair_v2"
REAL_QUALIFICATION = (
    ROOT
    / "var/artifacts/point_maze_algorithm_repair_pair_v2r5_audit.json"
)
IDENTITY = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_final_v1_identity.json"
)
MANIFEST = (
    ROOT / "var/artifacts/point_maze_algorithm_repair_final_v1_jobs.tsv"
)

base.SEEDS = SEEDS
base.FAMILIES = (
    "cross9_balanced",
    "upper_offset9_balanced",
    "block11_balanced",
    "bar11_balanced",
)

_tree_hash = base.tree_hash
_loads = base.json.loads
_atomic = base.atomic
_is_file = Path.is_file
_read_text = Path.read_text
_read_bytes = Path.read_bytes
_load_from_disk = datasets.load_from_disk


def _redirect(path: Path) -> Path:
    value = path.as_posix()
    if value.endswith(
        "var/artifacts/point_maze_interactive_paired_smoke_v3_audit.json"
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
    if path.as_posix().endswith("var/data/point_maze_modebench_v1"):
        path = REAL_DATA
    return _tree_hash(path)


def load_from_disk(path, *args, **kwargs):
    candidate = Path(path)
    marker = "var/data/point_maze_modebench_v1"
    value = candidate.as_posix()
    if marker in value:
        suffix = value.split(marker, 1)[1].lstrip("/")
        candidate = REAL_DATA / suffix
    return _load_from_disk(str(candidate), *args, **kwargs)


def loads(value, *args, **kwargs):
    payload = _loads(value, *args, **kwargs)
    if not isinstance(payload, dict):
        return payload
    schema = payload.get("schema")
    if schema == "point-maze-algorithm-repair-final-identity-v1":
        payload["schema"] = "point-maze-stage-b-05b-12pass-identity-v1"
    elif schema == "point-maze-algorithm-repair-final-submission-v1":
        payload["schema"] = "point-maze-stage-b-05b-12pass-submission-v1"
    elif schema == "point-maze-algorithm-repair-pair-audit-v2":
        payload.update(
            schema="point-maze-interactive-paired-smoke-audit-v3",
            decision="eligible_for_ten_point_maze_stage_b_jobs",
        )
    elif schema == "point-maze-algorithm-repair-receipt-v2":
        payload["schema"] = "point-maze-stage-b-05b-12pass-receipt-v1"
    elif schema == "point-maze-algorithm-repair-evaluation-v2":
        payload["schema"] = "point-maze-stage-b-evaluation-v1"
    return payload


def atomic(path: Path, payload: dict) -> None:
    if payload.get("schema") != "point-maze-stage-b-05b-12pass-audit-v1":
        _atomic(path, payload)
        return
    errors = list(payload.get("errors", []))
    checks = dict(payload.get("checks", {}))
    identity = _loads(IDENTITY.read_text(encoding="utf-8"))
    pair = _loads(REAL_QUALIFICATION.read_text(encoding="utf-8"))
    manifest_rows = list(csv.DictReader(MANIFEST.open(), delimiter="\t"))
    checks["repair_final_identity"] = (
        identity.get("schema")
        == "point-maze-algorithm-repair-final-identity-v1"
        and identity.get("seeds") == list(SEEDS)
        and identity.get("final_seed_cohort") is True
        and identity.get("development_only") is False
        and identity.get("evaluation_split") == "previously_untouched_eval"
        and identity.get("development_rows_loaded") is False
        and identity.get("secondary_post_outcome_repair") is True
        and identity.get("qualification_audit_sha256")
        == base.sha(REAL_QUALIFICATION)
    )
    checks["qualified_pair_pass"] = (
        pair.get("schema") == "point-maze-algorithm-repair-pair-audit-v2"
        and pair.get("status") == "pass"
        and pair.get("decision")
        == "eligible_for_point_maze_algorithm_repair_v2_five_seed_final"
        and pair.get("errors") in ([], None)
    )
    trainability = {}
    for row in manifest_rows:
        label = f"{row['arm']}/s{row['seed']}"
        metrics = [
            _loads(line)
            for line in (ROOT / row["metrics"]).read_text().splitlines()
            if line.strip()
        ]
        training = [
            item
            for item in metrics
            if item.get("schema") == "point-maze-stage-b-training-metric-v1"
        ]
        verified_rate = (
            sum(float(item.get("verified_episodes", 0)) for item in training)
            / (base.UPDATES * 16)
        )
        task_updates = sum(
            float(item.get("task_advantage_rms", 0)) > 0
            for item in training
        )
        trainability[label] = {
            "verified_rate": verified_rate,
            "task_advantage_nonzero_updates": task_updates,
        }
    checks["all_cells_trainable"] = (
        len(trainability) == 10
        and all(
            0.02 <= cell["verified_rate"] <= 0.50
            and cell["task_advantage_nonzero_updates"] >= 10
            for cell in trainability.values()
        )
    )
    for key in (
        "repair_final_identity",
        "qualified_pair_pass",
        "all_cells_trainable",
    ):
        if not checks[key]:
            errors.append(f"failed check: {key}")
    payload.update(
        schema="point-maze-algorithm-repair-final-audit-v1",
        checks=checks,
        trainability=trainability,
        errors=sorted(set(errors)),
    )
    payload["status"] = "pass" if not payload["errors"] else "fail"
    payload["decision"] = (
        "point_maze_algorithm_repair_final_terminal_eligible"
        if not payload["errors"]
        else "point_maze_algorithm_repair_final_stopped"
    )
    _atomic(path, payload)


Path.is_file = is_file
Path.read_text = read_text
Path.read_bytes = read_bytes
base.tree_hash = tree_hash
base.json.loads = loads
base.atomic = atomic
datasets.load_from_disk = load_from_disk


if __name__ == "__main__":
    base.main()
