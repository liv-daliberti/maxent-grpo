"""Hash-bound AntMaze executor for the admitted v11 waypoint controller."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import ant_maze_worker_v10 as base


ROOT = base.ROOT
MODEL_PATH = ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.zip"
RECEIPT_PATH = ROOT / "var/maze_runtime/controllers/ant_waypoint_v11.evaluation.json"
TRAINING_IDENTITY_PATH = ROOT / "var/artifacts/ant_waypoint_controller_v11_identity.json"
WAYPOINT_DISTANCE = base.WAYPOINT_DISTANCE
WAYPOINT_SUCCESS_THRESHOLD = base.WAYPOINT_SUCCESS_THRESHOLD


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def controller_receipt_sha256() -> str:
    controller_identity()
    return _sha256(RECEIPT_PATH)


def controller_identity() -> dict[str, Any]:
    receipt_sha256 = _sha256(RECEIPT_PATH)
    model_sha256 = _sha256(MODEL_PATH)
    training_identity_sha256 = _sha256(TRAINING_IDENTITY_PATH)
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    training_identity = json.loads(TRAINING_IDENTITY_PATH.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass" or receipt.get("decision") != "admitted_to_fresh_maze_route_gate_v11":
        raise RuntimeError("Ant v11 controller is not admitted")
    if receipt.get("seed") != 73011 or receipt.get("timesteps") != 2_000_000:
        raise RuntimeError("Ant v11 receipt differs from its frozen training identity")
    if training_identity.get("job_id") != 30198291 or training_identity.get("seed") != 73011:
        raise RuntimeError("Ant v11 training identity is not the frozen job")
    checks = receipt.get("checks", {})
    required_checks = {
        "episode_count", "success_rate_at_least_0p90",
        "minimum_heading_success_rate_at_least_0p75",
        "minimum_map_success_rate_at_least_0p75",
        "unhealthy_termination_rate_at_most_0p10",
        "median_success_steps_at_most_300", "all_metrics_finite",
    }
    if not isinstance(checks, dict) or not required_checks.issubset(checks) or not all(checks[key] is True for key in required_checks):
        raise RuntimeError("Ant v11 receipt contains a failed controller check")
    if receipt.get("hashes", {}).get("model_sha256") != model_sha256:
        raise RuntimeError("Ant v11 receipt does not bind the controller model")
    if receipt.get("training_waypoint_distances") != [WAYPOINT_DISTANCE] or receipt.get("waypoint_distance") != WAYPOINT_DISTANCE:
        raise RuntimeError("Ant v11 receipt has a different waypoint distance")
    if receipt.get("success_threshold") != WAYPOINT_SUCCESS_THRESHOLD:
        raise RuntimeError("Ant v11 receipt has a different waypoint threshold")
    route_identity_path = os.environ.get("OAT_ZERO_PROTOCOL_IDENTITY")
    if route_identity_path:
        route_identity = json.loads(Path(route_identity_path).read_text(encoding="utf-8"))
        expected = {
            "controller_receipt_sha256": receipt_sha256,
            "controller_model_sha256": model_sha256,
            "controller_training_identity_sha256": training_identity_sha256,
        }
        if route_identity.get("schema_version") != "ant-maze-v11-route-generation-identity-v1":
            raise RuntimeError("Ant v11 route identity schema mismatch")
        if any(route_identity.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Ant v11 route identity does not bind the controller")
    return receipt


base.MODEL_PATH = MODEL_PATH
base.RECEIPT_PATH = RECEIPT_PATH
base.TRAINING_IDENTITY_PATH = TRAINING_IDENTITY_PATH
base.controller_identity = controller_identity
base.controller_receipt_sha256 = controller_receipt_sha256
base._MODEL = None


def execute_ant_v11_raw(candidate: str, raw_spec: Mapping[str, Any]) -> dict[str, Any]:
    return base.execute_ant_v10_raw(candidate, raw_spec)


def execute_ant_v11(candidate: str, raw_spec: Mapping[str, Any]):
    return base.execute_ant_v10(candidate, raw_spec)
