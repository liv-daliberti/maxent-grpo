"""Grid-anchored AntMaze executor for the admitted sequential v17 controller."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import ant_maze_worker_v10 as executor
from . import ant_maze_worker_v12 as v12


ROOT = executor.ROOT
MODEL_PATH = (
    ROOT / "var/maze_runtime/controllers/ant_sequential_waypoint_v17.zip"
)
RECEIPT_PATH = (
    ROOT
    / "var/maze_runtime/controllers/"
    "ant_sequential_waypoint_v17.evaluation.json"
)
TRAINING_IDENTITY_PATH = (
    ROOT / "var/artifacts/ant_sequential_waypoint_controller_v17_identity.json"
)
WAYPOINT_DISTANCE = 4.0
WAYPOINT_SUCCESS_THRESHOLD = 0.45
TARGETING_VERSION = "initial-grid-cumulative-v17"


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
        receipt.get("status") != "pass"
        or receipt.get("decision")
        != "admitted_to_fresh_maze_route_gate_v17"
        or receipt.get("seed") != 73017
        or receipt.get("timesteps") != 4_000_000
    ):
        raise RuntimeError("Ant v17 worker requires the passing v17 gate")
    if (
        training_identity.get("job_id") != 30205033
        or training_identity.get("seed") != 73017
        or training_identity.get("development_episode_count") != 96
    ):
        raise RuntimeError("Ant v17 training identity drift")
    checks = receipt.get("checks", {})
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("Ant v17 worker received a failed controller check")
    if receipt.get("hashes", {}).get("model_sha256") != _sha256(MODEL_PATH):
        raise RuntimeError("Ant v17 receipt does not bind its model")
    if (
        receipt.get("training_waypoint_distances") != [WAYPOINT_DISTANCE]
        or receipt.get("waypoint_distance") != WAYPOINT_DISTANCE
        or receipt.get("success_threshold") != WAYPOINT_SUCCESS_THRESHOLD
    ):
        raise RuntimeError("Ant v17 waypoint contract drift")
    route_identity_path = os.environ.get("OAT_ZERO_PROTOCOL_IDENTITY")
    if route_identity_path:
        route_identity = json.loads(
            Path(route_identity_path).read_text(encoding="utf-8")
        )
        expected = _executor_identity()
        if route_identity.get("schema_version") != (
            "ant-maze-v17-route-generation-identity-v1"
        ):
            raise RuntimeError("Ant v17 route identity schema mismatch")
        for key, value in expected.items():
            if route_identity.get(key) != value:
                raise RuntimeError(
                    f"Ant v17 route identity does not bind {key}"
                )
    return receipt


def _model():
    executor.MODEL_PATH = MODEL_PATH
    executor.RECEIPT_PATH = RECEIPT_PATH
    executor.TRAINING_IDENTITY_PATH = TRAINING_IDENTITY_PATH
    executor.controller_identity = controller_identity
    executor.controller_receipt_sha256 = controller_receipt_sha256
    executor._MODEL = None
    return executor._model()


# Reuse only v12's grid-cumulative execution algorithm. All controller,
# receipt, identity, target-version, and model lookups are rebound above.
v12.MODEL_PATH = MODEL_PATH
v12.RECEIPT_PATH = RECEIPT_PATH
v12.TRAINING_IDENTITY_PATH = TRAINING_IDENTITY_PATH
v12.TARGETING_VERSION = TARGETING_VERSION
v12.controller_identity = controller_identity
v12.controller_receipt_sha256 = controller_receipt_sha256
v12._model = _model


def execute_ant_v17_raw(candidate, raw_spec):
    return v12.execute_ant_v12_raw(candidate, raw_spec)


def execute_ant_v17(candidate, raw_spec):
    return v12.execute_ant_v12(candidate, raw_spec)
