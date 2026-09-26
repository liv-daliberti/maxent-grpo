from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load_module():
    sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(
        "status_e105_under_test", EXP / "status_e105.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def status():
    return load_module()


def valid_ledger(status, tmp_path: Path) -> Path:
    repaired_runs = []
    repaired_index = {}
    for scale, seeds in status.SCALE_SEEDS.items():
        for seed in seeds:
            run = {
                "scale": scale,
                "domain": "python_factors",
                "seed": seed,
                "arm": "replay",
                "job_id": 900_000 + seed,
                "run_stamp": f"e109_{scale}_python_s{seed}",
                "run_dir": str(tmp_path / f"e109_{scale}_python_s{seed}"),
            }
            repaired_runs.append(run)
            repaired_index[(scale, "python_factors", seed)] = run
    repaired_path = (
        tmp_path / "e109_repaired_python_replay_comparators_jobs.json"
    )
    repaired = {
        "schema": "e109_repaired_python_replay_comparators_jobs_v1",
        "released": True,
        "pointmaze": "excluded",
        "domain": "python_factors",
        "semantic_coefficient": 0.0,
        "snapshot_sha256": "snapshot",
        "parser_surface_version": "surface",
        "target_steps": status.TARGET_STEPS,
        "checkpoint_interval_steps": status.CHECKPOINT_INTERVAL,
        "qwen3_a6000_seeds": [73, 74],
        "qwen3_paired_placement_artifact": str(tmp_path / "placement.json"),
        "qwen3_paired_placement_artifact_sha256": "placement-hash",
        "runs": repaired_runs,
    }
    repaired_path.write_text(json.dumps(repaired), encoding="utf-8")
    status.launch.REPAIRED_PYTHON_COMPARATOR_LEDGER = str(repaired_path)

    runs = []
    job_id = 1_000
    for scale, seeds in status.SCALE_SEEDS.items():
        for domain in status.DOMAINS:
            for seed in seeds:
                job_id += 1
                stamp = f"e105_{scale}_{domain}_s{seed}"
                paired = (
                    repaired_index[(scale, domain, seed)]
                    if domain == "python_factors"
                    else {
                        "job_id": job_id + 10_000,
                        "run_stamp": f"replay_{scale}_{domain}_s{seed}",
                        "run_dir": str(
                            tmp_path / f"replay_{scale}_{domain}_s{seed}"
                        ),
                    }
                )
                runs.append(
                    {
                        "scale": scale,
                        "domain": domain,
                        "seed": seed,
                        "arm": status.ARM,
                        "job_id": job_id,
                        "run_stamp": stamp,
                        "run_dir": str(tmp_path / stamp),
                        "paired_replay": {
                            field: paired[field]
                            for field in ("job_id", "run_stamp", "run_dir")
                        },
                    }
                )
    payload = {
        "schema": status.SCHEMA,
        "released": True,
        "pointmaze": "excluded",
        "models": list(status.SCALE_SEEDS),
        "domains": list(status.DOMAINS),
        "seeds": {
            scale: list(seeds) for scale, seeds in status.SCALE_SEEDS.items()
        },
        "arms": [status.ARM],
        "target_steps": status.TARGET_STEPS,
        "train_rows": status.TRAIN_ROWS,
        "passes": status.PASSES,
        "checkpoint_interval_steps": status.CHECKPOINT_INTERVAL,
        "python_comparator_repaired": True,
        "snapshot_sha256": "snapshot",
        "python_response_surface_version": "surface",
        "repaired_python_comparator_ledger": str(repaired_path),
        "repaired_python_comparator_ledger_sha256": status.launch.e104.digest(
            repaired_path
        ),
        "qwen3_paired_placement_artifact": str(tmp_path / "placement.json"),
        "qwen3_paired_placement_artifact_sha256": "placement-hash",
        "runs": runs,
    }
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rejects_any_pointmaze_or_grid_drift(status, tmp_path):
    path = valid_ledger(status, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pointmaze"] = "included"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="pointmaze_excluded"):
        status.load_and_validate_ledger(path)


def test_rejects_historical_or_semantic_active_python_comparator(
    status, tmp_path
):
    path = valid_ledger(status, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    python_run = next(
        run for run in payload["runs"] if run["domain"] == "python_factors"
    )
    repaired_pair = dict(python_run["paired_replay"])
    python_run["paired_replay"] = {
        "job_id": 123_456,
        "run_stamp": "historical_python_replay",
        "run_dir": str(tmp_path / "historical-python-replay"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match E109"):
        status.load_and_validate_ledger(path)

    python_run["paired_replay"] = repaired_pair
    repaired_path = Path(payload["repaired_python_comparator_ledger"])
    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
    repaired["semantic_coefficient"] = 0.1
    repaired_path.write_text(json.dumps(repaired), encoding="utf-8")
    payload["repaired_python_comparator_ledger_sha256"] = (
        status.launch.e104.digest(repaired_path)
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="semantic_disabled"):
        status.load_and_validate_ledger(path)


def test_rejects_e109_e105_qwen3_placement_digest_mismatch(status, tmp_path):
    path = valid_ledger(status, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    repaired_path = Path(payload["repaired_python_comparator_ledger"])
    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
    repaired["qwen3_paired_placement_artifact_sha256"] = "other-placement"
    repaired_path.write_text(json.dumps(repaired), encoding="utf-8")
    payload["repaired_python_comparator_ledger_sha256"] = (
        status.launch.e104.digest(repaired_path)
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="qwen3_placement_digest"):
        status.load_and_validate_ledger(path)


def test_monitor_grid_is_the_launcher_grid(status):
    assert status.SCALE_SEEDS is status.launch.SCALE_SEEDS
    assert status.DOMAINS is status.launch.DOMAINS
    assert status.SCALE_SEEDS["falcon1b"] == (55, 56, 57, 58, 59)
    assert status.ARM == status.launch.ARM
    assert status.TARGET_STEPS == status.launch.TARGET_STEPS == 3_072


def test_outcome_blind_snapshot_classifies_completion_and_failure(
    status, tmp_path, monkeypatch
):
    ledger = status.load_and_validate_ledger(valid_ledger(status, tmp_path))
    first, second = ledger["runs"][:2]
    states = {int(run["job_id"]): "PENDING" for run in ledger["runs"]}
    states.update(
        {
            int(run["paired_replay"]["job_id"]): "PENDING"
            for run in ledger["runs"]
        }
    )
    states[int(first["job_id"])] = "COMPLETED"
    states[int(second["job_id"])] = "FAILED"

    def receipt_step(run_dir):
        return status.TARGET_STEPS if run_dir == Path(first["run_dir"]) else 0

    monkeypatch.setattr(status.shared, "receipt_step", receipt_step)
    monkeypatch.setattr(status.shared, "run_step", lambda _run_dir: 0)
    monkeypatch.setattr(status.shared, "checkpoint_step", lambda _run_dir: 0)
    report = status.snapshot(ledger, scheduler_states=states)

    assert report["outcomes_read"] is False
    assert report["pointmaze"] == "excluded"
    assert report["completed"] == 1
    assert report["issue_cells"] == 1
    assert report["paired_comparator_completed"] == 0
    assert report["paired_comparator_issue_cells"] == 0
    assert report["ready_for_registered_analysis"] is False
    assert report["rows"][0]["state"] == "COMPLETED"
    assert report["rows"][1]["issues"] == ["scheduler_failed"]


def test_scheduler_completion_without_receipt_is_not_scientific_completion(
    status, tmp_path, monkeypatch
):
    ledger = status.load_and_validate_ledger(valid_ledger(status, tmp_path))
    states = {int(run["job_id"]): "PENDING" for run in ledger["runs"]}
    states.update(
        {
            int(run["paired_replay"]["job_id"]): "PENDING"
            for run in ledger["runs"]
        }
    )
    first = ledger["runs"][0]
    states[int(first["job_id"])] = "COMPLETED"
    monkeypatch.setattr(status.shared, "receipt_step", lambda _run_dir: 0)
    monkeypatch.setattr(status.shared, "run_step", lambda _run_dir: 0)
    monkeypatch.setattr(status.shared, "checkpoint_step", lambda _run_dir: 0)

    report = status.snapshot(ledger, scheduler_states=states)

    assert report["completed"] == 0
    assert report["rows"][0]["issues"] == [
        "scheduler_complete_without_terminal_training_receipt"
    ]


def test_incomplete_paired_comparator_prevents_analysis_readiness(
    status, tmp_path, monkeypatch
):
    ledger = status.load_and_validate_ledger(valid_ledger(status, tmp_path))
    missing = Path(ledger["runs"][0]["paired_replay"]["run_dir"])
    all_job_ids = {
        int(run[key]["job_id"] if key else run["job_id"])
        for run in ledger["runs"]
        for key in (None, "paired_replay")
    }
    monkeypatch.setattr(
        status.shared,
        "receipt_step",
        lambda run_dir: 0 if run_dir == missing else status.TARGET_STEPS,
    )
    monkeypatch.setattr(status.shared, "run_step", lambda _run_dir: 0)
    monkeypatch.setattr(status.shared, "checkpoint_step", lambda _run_dir: 0)

    report = status.snapshot(
        ledger,
        scheduler_states={job_id: "RUNNING" for job_id in all_job_ids},
    )

    assert report["completed"] == 75
    assert report["paired_comparator_completed"] == 74
    assert report["ready_for_registered_analysis"] is False
    assert report["outcomes_read"] is False
