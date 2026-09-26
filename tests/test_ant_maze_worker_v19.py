from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oat_drgrpo import ant_maze_worker_v19 as worker


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _frozen_fixture(tmp_path, monkeypatch):
    model = tmp_path / "controller.zip"
    model.write_bytes(b"continuing-v19-controller")
    identity_path = tmp_path / "training-identity.json"
    receipt_path = tmp_path / "receipt.json"
    trainer_sha = "1" * 64
    initial_sha = "2" * 64
    _write_json(
        identity_path,
        {
            "schema_version": (
                "ant-continuing-waypoint-controller-v19-identity-v1"
            ),
            "job_id": 30259111,
            "seed": 73019,
            "timesteps": 6_000_000,
            "development_episode_count": 96,
            "stable_planar_speed": 1.0,
            "continuing_task": True,
            "explicit_ant_health": True,
            "trainer_sha256": trainer_sha,
            "initial_model_sha256": initial_sha,
        },
    )
    receipt = {
        "schema_version": "ant-waypoint-controller-v19-evaluation-v1",
        "status": "pass",
        "decision": "admitted_to_fresh_maze_route_gate_v19",
        "seed": 73019,
        "timesteps": 6_000_000,
        "workers": 8,
        "learning_rate": 5e-7,
        "training_waypoint_distances": [4.0],
        "waypoint_distance": 4.0,
        "success_threshold": 0.45,
        "checks": {
            "all_recorded_arrivals_stable": True,
            "maze_task_never_terminated_controller_gate": True,
        },
        "evaluation": {
            "summary": {
                "episodes": 96,
                "maximum_arrival_speed": 1.0,
                "maze_task_termination_count": 0,
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


def test_controller_identity_accepts_exact_v19_gate(tmp_path, monkeypatch):
    _frozen_fixture(tmp_path, monkeypatch)
    receipt = worker.controller_identity()
    assert receipt["status"] == "pass"
    identity = worker._executor_identity()
    assert identity["continuing_task_controller_gate"] is True
    assert identity["explicit_ant_health_gate"] is True
    assert worker.controller_receipt_sha256() == worker._canonical_sha256(
        identity
    )


def test_controller_identity_rejects_maze_task_termination(
    tmp_path, monkeypatch
):
    receipt_path, receipt = _frozen_fixture(tmp_path, monkeypatch)
    receipt["evaluation"]["summary"]["maze_task_termination_count"] = 1
    _write_json(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="continuing stable-arrival"):
        worker.controller_identity()


def test_v19_route_protocol_and_dependency_forbid_early_lm_or_substitution():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "ant_maze_v15_controller_v19_admission_20260804.md"
    ).read_text(encoding="utf-8")
    scheduler = (
        ROOT
        / "ops/exp_scaling/"
        "schedule_ant_maze_v15_controller_v19.py"
    ).read_text(encoding="utf-8")
    assert "FROZEN WHILE CONTROLLER JOB 30259111 WAS RUNNING" in protocol
    assert "gate samples no language model" in protocol
    assert "afterok:{CONTROLLER_JOB}" in scheduler
    assert '"language_model_sampled": False' in scheduler
    assert '"post_outcome_map_or_route_substitution": False' in scheduler
