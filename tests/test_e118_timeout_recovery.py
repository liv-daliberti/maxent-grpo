"""Deferred E118 allocations must never overlap writers or restart from zero."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
SPEC = importlib.util.spec_from_file_location(
    "e118_timeout_recovery", ROOT / "ops/exp_scaling/recover_e118_timeouts_20260906.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


@pytest.fixture
def setup_successor(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    audit = tmp_path / "transaction.json"
    run = tmp_path / "run"
    run.mkdir()
    frozen = tmp_path / "snapshot/ops/slurm/train.slurm"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("#!/bin/bash\nexit 0\n")
    row = dict(job_id=101, domain="pantry_plan", arm="maxrl", seed=73,
               run_dir=str(run), run_stamp="fixture", previous_job_ids=[100])
    item = dict(row, old_job_id=101, new_job_id=102, kind="successor",
                frozen_launcher=str(frozen),
                frozen_launcher_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
                held_scheduler_record="fixture")
    ledger.write_text(json.dumps({"runs": [row]}))
    audit.write_text(json.dumps({"rows": [item], "events": []}))
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "LOCK", tmp_path / "lock")
    monkeypatch.setattr(controller, "AUDIT", audit)
    monkeypatch.setattr(controller.base, "LEDGER", ledger)
    monkeypatch.setattr(controller.base, "queue", lambda: {})
    monkeypatch.setattr(controller, "accounting", lambda _: "TIMEOUT")
    monkeypatch.setenv("SLURM_JOB_ID", "102")
    calls = []

    def command(args):
        calls.append(args)
        return "/valid/checkpoints/step_01920\n" if "--select-under" in args else ""

    executions = []
    monkeypatch.setattr(controller, "command", command)
    monkeypatch.setattr(controller.os, "chdir", lambda _: None)
    monkeypatch.setattr(controller.os, "execv", lambda *args: executions.append(args))
    return ledger, audit, run, calls, executions


def test_refuses_live_predecessor_without_ledger_mutation(setup_successor, monkeypatch):
    ledger, _, _, _, executions = setup_successor
    before = ledger.read_bytes()
    monkeypatch.setattr(controller.base, "queue", lambda: {101: "RUNNING"})
    with pytest.raises(RuntimeError, match="still active"):
        controller.start_successor()
    assert ledger.read_bytes() == before
    assert not executions


def test_completed_receipt_skips_training_and_preserves_predecessor(setup_successor):
    ledger, _, run, calls, executions = setup_successor
    before = ledger.read_bytes()
    (run / "TRAINING_COMPLETE.json").write_text(json.dumps({
        "schema": "oat_zero_training_complete_v1", "terminal_step": 3073}))
    controller.start_successor()
    assert ledger.read_bytes() == before
    assert not calls
    assert not executions


def test_missing_or_rejected_checkpoint_refuses_promotion(setup_successor, monkeypatch):
    ledger, _, _, _, executions = setup_successor
    before = ledger.read_bytes()
    monkeypatch.setattr(controller, "command", lambda _: "")
    with pytest.raises(RuntimeError, match="lacks a valid checkpoint"):
        controller.start_successor()
    assert ledger.read_bytes() == before
    assert not executions


def test_invalid_completion_receipt_does_not_restart_cell(setup_successor):
    ledger, _, run, _, executions = setup_successor
    before = ledger.read_bytes()
    (run / "TRAINING_COMPLETE.json").write_text(json.dumps({
        "schema": "oat_zero_training_complete_v1", "terminal_step": 1920}))
    with pytest.raises(RuntimeError, match="invalid completion receipt"):
        controller.start_successor()
    assert ledger.read_bytes() == before
    assert not executions


def test_promotes_once_and_rebuilds_aggregate_on_same_id_restart(setup_successor):
    ledger, _, _, calls, executions = setup_successor
    controller.start_successor()
    first = json.loads(ledger.read_text())
    assert first["runs"][0]["job_id"] == 102
    assert first["runs"][0]["previous_job_ids"] == [100, 101]
    assert len(first["repair_history"]) == 1
    controller.start_successor()
    assert json.loads(ledger.read_text()) == first
    assert len(executions) == 2
    assert sum("build_e118_aggregate_ledger.py" in str(args) for args in calls) == 2
