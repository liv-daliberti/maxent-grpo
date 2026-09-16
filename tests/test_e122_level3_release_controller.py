from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ops/exp_scaling/control_e122_level3_release.py"
SPEC = importlib.util.spec_from_file_location("e122_release_test", SOURCE)
assert SPEC and SPEC.loader
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def campaign_fixture(tmp_path, choice="3b"):
    cells = [dict(domain=f"domain{index // 20}", dataset_domain=f"data{index // 20}",
                  arm=f"arm{index % 20 // 5}", seed=43 + index % 5,
                  run_stamp=f"e122_cell{index}", run_dir=str(tmp_path / f"run{index}"),
                  environment={"MODEL": choice}, command=["sbatch", "--hold"])
             for index in range(100)]
    peak, terminal = controller.PROFILE_BYTES[choice]
    proof = {"status": "matched_fixed_reference", "path": "admission.json", "sha256": "a" * 64}
    plan = {"schema": "e122_level3_factorial_plan_v1", "model_choice": choice,
            "dataset_identity_sha256": controller.DATASET_IDENTITY_SHA256,
            "admission_proof": proof, "cells": cells,
            "storage_profile": {"model_choice": choice, "peak_bytes": peak,
                                "terminal_bytes": terminal, "measurement_verified": True,
                                "evidence": ["measured original model/checkpoint sizes"]}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    rows = [dict(**{key: cell[key] for key in controller.CELL_FIELDS}, job_id=1000 + i,
                 held_scheduler_record=f"JobId={1000 + i} JobState=PENDING Reason=JobHeldUser")
            for i, cell in enumerate(cells)]
    ledger = {"schema": "e122_level3_factorial_jobs_v1", "model_choice": choice,
              "model_choice_pending": False, "status": "held_audited", "released": False,
              "plan_path": str(plan_path), "plan_sha256": controller.digest(plan_path),
              "admission_proof": proof, "runs": rows,
              "planned_runs": [{key: cell[key] for key in controller.CELL_FIELDS} for cell in cells]}
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    args = argparse.Namespace(plan=plan_path, plan_sha256=controller.digest(plan_path),
                              held_ledger=ledger_path, held_ledger_sha256=controller.digest(ledger_path),
                              model_choice=choice)
    launcher = SimpleNamespace(verify_plan=lambda **_: plan,
                               audit_held_record=lambda record, job_id, cell: record,
                               audit_held=lambda job_id, cell: f"audited={job_id}")
    return args, launcher, plan, ledger


def snapshot(ids, *, states=None, missing=()):
    states = states or {}
    queue, account = [], []
    for job_id in ids:
        state, reason, exit_code = states.get(job_id, ("PENDING", "JobHeldUser", "0:0"))
        if state not in controller.TERMINAL_STATES:
            queue.append(f"{job_id}|{state}|{reason}")
        if job_id not in missing:
            account.append(f"{job_id}|{state}|{exit_code}")
    return controller.parse_scheduler(ids, "\n".join(queue), "\n".join(account))


def entry(returncode=0):
    return {"intent": {}, "result": {"returncode": returncode, "error": None}}


def test_storage_exact_boundary_and_all_queued_jobs_consume_capacity():
    peak, terminal = controller.PROFILE_BYTES["3b"]
    required = controller.SHARED_HEADROOM_BYTES + 100 * terminal + peak
    below = controller.storage_decision(free_bytes=required - 1, unfinished=100, active=0,
                                       peak_bytes=peak, terminal_bytes=terminal)
    exact = controller.storage_decision(free_bytes=required, unfinished=100, active=0,
                                       peak_bytes=peak, terminal_bytes=terminal)
    assert below["can_release"] is False and exact["can_release"] is True
    assert required == int(773 * controller.GIB)
    assert not controller.storage_decision(free_bytes=10**15, unfinished=100, active=4,
                                          peak_bytes=peak, terminal_bytes=terminal)["can_release"]


def test_missing_accounting_and_steps_never_prove_completion(tmp_path):
    args, launcher, _, _ = campaign_fixture(tmp_path)
    campaign = controller.load_campaign(args, launcher)
    ids = [job["job_id"] for job in campaign["jobs"]]
    observations = snapshot(ids, missing={ids[0]})
    status = controller.evaluate(campaign, observations, {ids[0]: entry()}, {"free_bytes": 10**15})
    assert status["terminal"] == 0 and status["released_nonterminal"] == 1
    assert status["blocked_reason"] == "unknown_scheduler_state"
    step_only = controller.parse_scheduler(["1000"], "", "1000.batch|COMPLETED|0:0\n")
    assert step_only["jobs"]["1000"]["unknown"]
    assert not step_only["jobs"]["1000"]["terminal"]


def test_queued_external_and_failed_jobs_keep_slots_and_both_storage_reserves(tmp_path):
    args, launcher, _, _ = campaign_fixture(tmp_path)
    campaign = controller.load_campaign(args, launcher)
    ids = [job["job_id"] for job in campaign["jobs"]]
    states = {job_id: ("PENDING", "Resources", "0:0") for job_id in ids[:4]}
    states[ids[4]] = ("FAILED", "None", "1:0")
    entries = {job_id: entry() for job_id in ids[:5]}
    status = controller.evaluate(campaign, snapshot(ids, states=states), entries, {"free_bytes": 10**15})
    assert status["released_nonterminal"] == 4 and status["terminal"] == 1
    assert status["unfinished_terminal_exports"] == 100
    assert status["blocked_reason"] == "needs_operator_review"
    assert status["reserved_unfinished_slots"] == 5
    assert status["storage"]["active_peak_reserve_bytes"] == 5 * 84 * controller.GIB
    del entries[ids[0]]
    status = controller.evaluate(campaign, snapshot(ids, states=states), entries, {"free_bytes": 10**15})
    assert status["released_nonterminal"] == 4
    assert status["blocked_reason"] == "ambiguous_or_external_activity"


def test_export_reserve_shrinks_only_for_successful_matching_terminal_export(tmp_path):
    args, launcher, _, _ = campaign_fixture(tmp_path)
    campaign = controller.load_campaign(args, launcher)
    ids = [job["job_id"] for job in campaign["jobs"]]
    observations = snapshot(ids, states={ids[0]: ("COMPLETED", "None", "0:0")})
    run_dir = Path(campaign["jobs"][0]["row"]["run_dir"])
    export = run_dir / "debug_job1000/saved_models/step_03073"
    export.mkdir(parents=True)
    marker = {"schema": "oat_zero_training_complete_v1", "terminal_step": 3073,
              "terminal_export": str(export)}
    (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps(marker))
    status = controller.evaluate(campaign, observations, {ids[0]: entry()}, {"free_bytes": 10**15})
    assert status["unfinished_terminal_exports"] == 100
    (export / "model.safetensors").write_bytes(b"nonempty weight export")
    status = controller.evaluate(campaign, observations, {ids[0]: entry()}, {"free_bytes": 10**15})
    assert status["unfinished_terminal_exports"] == 99
    assert status["endpoint_evidence"][ids[0]]["terminal_step"] == 3073
    observations["jobs"][ids[0]]["exit_code"] = "1:0"
    assert not controller.successful_endpoint(campaign["jobs"][0]["row"], observations["jobs"][ids[0]])


@pytest.mark.parametrize("mutation", ["missing_pin", "wrong_pin", "pending_model", "wrong_model", "wrong_profile", "unverified_profile", "wrong_identity", "row_drift", "missing_cell"])
def test_explicit_pins_held_identity_and_model_profile_fail_closed(tmp_path, mutation):
    args, launcher, plan, ledger = campaign_fixture(tmp_path)
    if mutation == "missing_pin":
        args.plan_sha256 = ""
    elif mutation == "wrong_pin":
        args.held_ledger_sha256 = "b" * 64
    elif mutation == "pending_model":
        ledger["model_choice_pending"] = True
    elif mutation == "wrong_model":
        args.model_choice = "05b"
    elif mutation == "wrong_profile":
        plan["storage_profile"]["peak_bytes"] -= 1
    elif mutation == "unverified_profile":
        plan["storage_profile"]["measurement_verified"] = False
    elif mutation == "wrong_identity":
        plan["dataset_identity_sha256"] = "b" * 64
    elif mutation == "row_drift":
        ledger["runs"][0]["run_dir"] = str(tmp_path / "wrong")
    elif mutation == "missing_cell":
        ledger["runs"].pop()
    if mutation not in {"missing_pin", "wrong_pin"}:
        args.plan.write_text(json.dumps(plan))
        args.plan_sha256 = ledger["plan_sha256"] = controller.digest(args.plan)
        args.held_ledger.write_text(json.dumps(ledger))
        args.held_ledger_sha256 = controller.digest(args.held_ledger)
    with pytest.raises(ValueError):
        controller.load_campaign(args, launcher)


@pytest.mark.parametrize("choice", ["05b", "3b"])
def test_verified_model_profiles_are_bound_to_explicit_choice(tmp_path, choice):
    args, launcher, _, _ = campaign_fixture(tmp_path, choice)
    campaign = controller.load_campaign(args, launcher)
    assert campaign["profile"]["peak_bytes"] == controller.PROFILE_BYTES[choice][0]


def test_advance_has_once_only_intent_and_blocks_ambiguous_result(tmp_path, monkeypatch):
    args, launcher, _, _ = campaign_fixture(tmp_path)
    root = tmp_path / "controller"
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "disk_space", lambda path: {"free_bytes": 10**15, "device": path.stat().st_dev})
    monkeypatch.setattr(controller, "scheduler_snapshot", lambda ids: snapshot(ids))
    calls = []
    def release(argv):
        assert (root / "jobs/1000.intent.json").is_file()
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 45)
    monkeypatch.setattr(controller, "command", release)
    first = controller.advance_once(args, launcher, root)
    second = controller.advance_once(args, launcher, root)
    assert calls == [["scontrol", "release", "1000"]]
    assert first["blocked_reason"] == second["blocked_reason"] == "ambiguous_or_external_activity"
    assert second["released_nonterminal"] == 1
    assert len(list((root / "jobs").glob("*.intent.json"))) == 1
    with pytest.raises(FileExistsError):
        controller.immutable_json(root / "jobs/1000.intent.json", {})


def test_live_held_identity_audit_failure_prevents_intent_or_release(tmp_path, monkeypatch):
    args, launcher, _, _ = campaign_fixture(tmp_path)
    root = tmp_path / "controller"
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "disk_space", lambda path: {"free_bytes": 10**15, "device": path.stat().st_dev})
    monkeypatch.setattr(controller, "scheduler_snapshot", lambda ids: snapshot(ids))
    def reject(*_):
        raise ValueError("held environment differs")
    launcher.audit_held = reject
    monkeypatch.setattr(controller, "command", lambda _: pytest.fail("release should not execute"))
    with pytest.raises(ValueError, match="held environment"):
        controller.advance_once(args, launcher, root)
    assert not list((root / "jobs").glob("*.intent.json"))


def test_read_only_status_creates_no_files_and_missing_result_stops_progress(tmp_path, monkeypatch):
    args, launcher, _, _ = campaign_fixture(tmp_path)
    campaign = controller.load_campaign(args, launcher)
    root = tmp_path / "absent_controller"
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "scheduler_snapshot", lambda ids: snapshot(ids))
    monkeypatch.setattr(controller, "disk_space", lambda path: {"free_bytes": 10**15, "device": path.stat().st_dev})
    assert controller.status(campaign, root)["staged_held"] == 100
    assert not root.exists()
    ids = [job["job_id"] for job in campaign["jobs"]]
    result = controller.evaluate(campaign, snapshot(ids), {ids[0]: {"intent": {}, "result": None}}, {"free_bytes": 10**15})
    assert result["next_job_id"] is None
    assert result["released_nonterminal"] == 1


def test_exclusive_lock_prevents_overlapping_controllers(tmp_path):
    with controller.locked(tmp_path / "controller"):
        with pytest.raises(BlockingIOError):
            with controller.locked(tmp_path / "controller"):
                pytest.fail("second exclusive flock unexpectedly acquired")


def test_prospective_matrix_order_is_independent_of_release_order(tmp_path):
    args, launcher, _, ledger = campaign_fixture(tmp_path)
    ledger["planned_runs"].reverse()
    args.held_ledger.write_text(json.dumps(ledger))
    args.held_ledger_sha256 = controller.digest(args.held_ledger)
    assert len(controller.load_campaign(args, launcher)["jobs"]) == 100


def test_fresh_disk_measurement_after_held_audit_can_block_release(tmp_path, monkeypatch):
    args, launcher, _, _ = campaign_fixture(tmp_path)
    root = tmp_path / "controller"
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "scheduler_snapshot", lambda ids: snapshot(ids))
    readings = iter([10**15, 1])
    monkeypatch.setattr(controller, "disk_space", lambda path: {"free_bytes": next(readings), "device": path.stat().st_dev})
    monkeypatch.setattr(controller, "command", lambda _: pytest.fail("disk became insufficient"))
    result = controller.advance_once(args, launcher, root)
    assert result["blocked_reason"] == "waiting_disk"
    assert not list((root / "jobs").glob("*.intent.json"))


def test_successful_release_is_not_reissued_and_four_queued_jobs_stop_advancement(tmp_path, monkeypatch):
    args, launcher, _, _ = campaign_fixture(tmp_path, "05b")
    root = tmp_path / "controller"
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "disk_space", lambda path: {"free_bytes": 10**15, "device": path.stat().st_dev})
    states = {}
    monkeypatch.setattr(controller, "scheduler_snapshot", lambda ids: snapshot(ids, states=states))
    calls = []
    def release(argv):
        job_id = argv[-1]
        assert (root / f"jobs/{job_id}.intent.json").is_file()
        calls.append(argv)
        states[job_id] = ("PENDING", "Resources", "0:0")
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(controller, "command", release)
    for _ in range(5):
        result = controller.advance_once(args, launcher, root)
    assert [argv[-1] for argv in calls] == ["1000", "1001", "1002", "1003"]
    assert result["reserved_unfinished_slots"] == result["released_nonterminal"] == 4
    assert result["running"] == 0 and result["staged_held"] == 96
    assert result["blocked_reason"] == "concurrency_cap"
