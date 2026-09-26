"""V19 AntMaze executor bound to the continuing-task controller gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import ant_maze_worker_v18 as base


ROOT = Path(
    os.environ.get(
        "OAT_ZERO_REPO_ROOT",
        Path(__file__).resolve().parents[2],
    )
)
MODEL_PATH = (
    ROOT / "var/maze_runtime/controllers/ant_continuing_waypoint_v19.zip"
)
RECEIPT_PATH = (
    ROOT
    / "var/maze_runtime/controllers/ant_continuing_waypoint_v19.evaluation.json"
)
TRAINING_IDENTITY_PATH = (
    ROOT
    / "var/artifacts/ant_continuing_waypoint_controller_v19_identity.json"
)
WAYPOINT_DISTANCE = 4.0
WAYPOINT_SUCCESS_THRESHOLD = 0.45
STABLE_PLANAR_SPEED = 1.0
TARGETING_VERSION = "initial-grid-cumulative-stable-handoff-v19"
EXPECTED_CONTROLLER_JOB = 30259111


def _sha256(path: Path) -> str:
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


def _executor_identity() -> dict[str, Any]:
    return {
        "targeting_version": TARGETING_VERSION,
        "worker_source_sha256": _sha256(Path(__file__)),
        "controller_receipt_sha256": _sha256(RECEIPT_PATH),
        "controller_model_sha256": _sha256(MODEL_PATH),
        "controller_training_identity_sha256": _sha256(
            TRAINING_IDENTITY_PATH
        ),
        "waypoint_distance": WAYPOINT_DISTANCE,
        "waypoint_success_threshold": WAYPOINT_SUCCESS_THRESHOLD,
        "stable_planar_speed": STABLE_PLANAR_SPEED,
        "continuing_task_controller_gate": True,
        "explicit_ant_health_gate": True,
    }


def controller_receipt_sha256() -> str:
    controller_identity()
    return _canonical_sha256(_executor_identity())


def controller_identity() -> dict[str, Any]:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    training_identity = json.loads(
        TRAINING_IDENTITY_PATH.read_text(encoding="utf-8")
    )
    if (
        receipt.get("schema_version")
        != "ant-waypoint-controller-v19-evaluation-v1"
        or receipt.get("status") != "pass"
        or receipt.get("decision")
        != "admitted_to_fresh_maze_route_gate_v19"
        or receipt.get("seed") != 73019
        or receipt.get("timesteps") != 6_000_000
        or receipt.get("workers") != 8
        or receipt.get("learning_rate") != 5e-7
    ):
        raise RuntimeError("Ant v19 worker requires the passing frozen v19 gate")
    if (
        training_identity.get("schema_version")
        != "ant-continuing-waypoint-controller-v19-identity-v1"
        or training_identity.get("job_id") != EXPECTED_CONTROLLER_JOB
        or training_identity.get("seed") != 73019
        or training_identity.get("timesteps") != 6_000_000
        or training_identity.get("development_episode_count") != 96
        or training_identity.get("stable_planar_speed")
        != STABLE_PLANAR_SPEED
        or training_identity.get("continuing_task") is not True
        or training_identity.get("explicit_ant_health") is not True
    ):
        raise RuntimeError("Ant v19 training identity drift")
    checks = receipt.get("checks", {})
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("Ant v19 worker received a failed controller check")
    summary = receipt.get("evaluation", {}).get("summary", {})
    maximum_arrival_speed = summary.get("maximum_arrival_speed")
    if (
        summary.get("episodes") != 96
        or summary.get("maze_task_termination_count") != 0
        or maximum_arrival_speed is None
        or float(maximum_arrival_speed) > STABLE_PLANAR_SPEED + 1e-6
    ):
        raise RuntimeError("Ant v19 continuing stable-arrival gate drift")
    hashes = receipt.get("hashes", {})
    if hashes.get("model_sha256") != _sha256(MODEL_PATH):
        raise RuntimeError("Ant v19 receipt does not bind its model")
    if (
        hashes.get("training_source_sha256")
        != training_identity.get("trainer_sha256")
        or hashes.get("initial_model_sha256")
        != training_identity.get("initial_model_sha256")
    ):
        raise RuntimeError("Ant v19 receipt does not bind its frozen training")
    if (
        receipt.get("training_waypoint_distances") != [WAYPOINT_DISTANCE]
        or receipt.get("waypoint_distance") != WAYPOINT_DISTANCE
        or receipt.get("success_threshold") != WAYPOINT_SUCCESS_THRESHOLD
    ):
        raise RuntimeError("Ant v19 waypoint contract drift")
    route_identity_path = os.environ.get("OAT_ZERO_PROTOCOL_IDENTITY")
    if route_identity_path:
        route_identity = json.loads(
            Path(route_identity_path).read_text(encoding="utf-8")
        )
        expected = _executor_identity()
        if route_identity.get("schema_version") != (
            "ant-maze-v19-stable-route-generation-identity-v1"
        ):
            raise RuntimeError("Ant v19 route identity schema mismatch")
        for key, value in expected.items():
            if route_identity.get(key) != value:
                raise RuntimeError(
                    f"Ant v19 route identity does not bind {key}"
                )
    return receipt


base.MODEL_PATH = MODEL_PATH
base.RECEIPT_PATH = RECEIPT_PATH
base.TRAINING_IDENTITY_PATH = TRAINING_IDENTITY_PATH
base.WAYPOINT_DISTANCE = WAYPOINT_DISTANCE
base.WAYPOINT_SUCCESS_THRESHOLD = WAYPOINT_SUCCESS_THRESHOLD
base.STABLE_PLANAR_SPEED = STABLE_PLANAR_SPEED
base.TARGETING_VERSION = TARGETING_VERSION
base._MODEL = None
base._executor_identity = _executor_identity
base.controller_identity = controller_identity
base.controller_receipt_sha256 = controller_receipt_sha256


def execute_ant_v19_raw(
    candidate: str,
    raw_spec: Mapping[str, Any],
) -> dict[str, Any]:
    return base.execute_ant_v18_raw(candidate, raw_spec)


def execute_ant_v19(candidate: str, raw_spec: Mapping[str, Any]):
    return base.execute_ant_v18(candidate, raw_spec)
