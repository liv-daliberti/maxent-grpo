from __future__ import annotations

import hashlib
import json

import pytest

from oat_drgrpo import ant_maze_worker_v18 as worker


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _frozen_fixture(tmp_path, monkeypatch):
    model = tmp_path / "controller.zip"
    model.write_bytes(b"stable-v18-controller")
    identity_path = tmp_path / "training-identity.json"
    receipt_path = tmp_path / "receipt.json"
    trainer_sha = "1" * 64
    initial_sha = "2" * 64
    _write_json(
        identity_path,
        {
            "schema_version": (
                "ant-stable-handoff-controller-v18-identity-v1"
            ),
            "job_id": 30205570,
            "seed": 73018,
            "timesteps": 6_000_000,
            "development_episode_count": 96,
            "stable_planar_speed": 1.0,
            "trainer_sha256": trainer_sha,
            "initial_model_sha256": initial_sha,
        },
    )
    receipt = {
        "schema_version": "ant-waypoint-controller-v18-evaluation-v1",
        "status": "pass",
        "decision": "admitted_to_fresh_maze_route_gate_v18",
        "seed": 73018,
        "timesteps": 6_000_000,
        "workers": 8,
        "learning_rate": 2e-7,
        "training_waypoint_distances": [4.0],
        "waypoint_distance": 4.0,
        "success_threshold": 0.45,
        "checks": {"all_recorded_arrivals_stable": True},
        "evaluation": {
            "summary": {
                "episodes": 96,
                "maximum_arrival_speed": 1.0,
            }
        },
        "hashes": {
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "training_source_sha256": trainer_sha,
            "initial_model_sha256": initial_sha,
        },
    }
    _write_json(receipt_path, receipt)
    monkeypatch.setattr(worker, "MODEL_PATH", model)
    monkeypatch.setattr(worker, "RECEIPT_PATH", receipt_path)
    monkeypatch.setattr(worker, "TRAINING_IDENTITY_PATH", identity_path)
    return receipt_path, receipt


def test_controller_identity_accepts_exact_stable_gate(tmp_path, monkeypatch):
    _frozen_fixture(tmp_path, monkeypatch)
    receipt = worker.controller_identity()
    assert receipt["status"] == "pass"
    identity = worker._executor_identity()
    assert identity["stable_planar_speed"] == 1.0
    assert identity["waypoint_success_threshold"] == 0.45
    assert worker.controller_receipt_sha256() == worker._canonical_sha256(
        identity
    )


def test_controller_identity_rejects_unstable_arrival(tmp_path, monkeypatch):
    receipt_path, receipt = _frozen_fixture(tmp_path, monkeypatch)
    receipt["evaluation"]["summary"]["maximum_arrival_speed"] = 1.00001
    _write_json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="stable-arrival"):
        worker.controller_identity()


def test_route_identity_must_bind_speed_threshold(
    tmp_path,
    monkeypatch,
):
    _frozen_fixture(tmp_path, monkeypatch)
    route_identity = {
        "schema_version": (
            "ant-maze-v18-stable-route-generation-identity-v1"
        ),
        **worker._executor_identity(),
    }
    route_identity["stable_planar_speed"] = 1.1
    route_identity_path = tmp_path / "route-identity.json"
    _write_json(route_identity_path, route_identity)
    monkeypatch.setenv(
        "OAT_ZERO_PROTOCOL_IDENTITY",
        str(route_identity_path),
    )
    with pytest.raises(RuntimeError, match="stable_planar_speed"):
        worker.controller_identity()
