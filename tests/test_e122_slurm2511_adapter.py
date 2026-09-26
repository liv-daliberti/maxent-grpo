"""Regression and provenance checks for the sole E122 Slurm display amendment."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e122_slurm2511_adapter_test", ROOT / "ops/exp_scaling/e122_slurm2511_adapter.py")
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


@pytest.fixture(scope="module")
def actual_held_fixture():
    review = json.loads((ROOT / "var/artifacts/e122_level3_factorial/slurm_2511_first_job_review.json").read_text())
    plan = json.loads(adapter.PLAN_PATH.read_text())
    original = adapter.load_private_module(adapter.LAUNCHER_PATH, "e122_adapter_real_launcher_test")
    return review["raw_scheduler_record"], review["job_id"], plan["cells"][0], original


def test_real_slurm_2511_record_passes_original_audit_only_after_exact_display_change(actual_held_fixture):
    raw, job_id, cell, original = actual_held_fixture
    with pytest.raises(ValueError):
        original.audit_held_record(raw, job_id, cell)
    normalized = adapter.normalize_num_nodes(raw)
    assert normalized == raw.replace(" NumNodes=1-1 ", " NumNodes=1 ", 1)
    assert original.audit_held_record(normalized, job_id, cell) == normalized
    proxy = adapter.LauncherAdapter(original)
    assert proxy.audit_held_record(raw, job_id, cell) == raw
    assert proxy.audit_held_record(normalized, job_id, cell) == normalized
    assert proxy.verify_plan is original.verify_plan
    assert proxy.submit_one_held is original.submit_one_held


@pytest.mark.parametrize("value", ["0", "1-2", "2", "0-1", "01", "1-01", "1-1-1", "1-1,", "1-1x", ""])
def test_any_other_node_request_fails_before_delegation(value):
    calls = []
    original = SimpleNamespace(audit_held_record=lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="one isolated"):
        adapter.LauncherAdapter(original).audit_held_record(f"JobId=1 NumNodes={value} NumTasks=1", 1, {})
    assert not calls


@pytest.mark.parametrize("raw", ["JobId=1", "xNumNodes=1-1", "x=NumNodes=1-1", "NumNodes=1 NumNodes=1-1", "NumNodes=1-1 NumNodes=1-1"])
def test_missing_nonisolated_or_duplicate_fields_fail_closed(raw):
    with pytest.raises(ValueError, match="one isolated"):
        adapter.normalize_num_nodes(raw)


def test_normalization_preserves_all_unrelated_bytes_and_whitespace():
    raw = "\tJobId=1\nNumNodes=1-1\tTag=containsNumNodes=1-1\n"
    assert adapter.normalize_num_nodes(raw) == raw.replace("\nNumNodes=1-1\t", "\nNumNodes=1\t")
    assert adapter.normalize_num_nodes("NumNodes=1") == "NumNodes=1"
    assert adapter.normalize_num_nodes("NumNodes=1-1") == "NumNodes=1"


@pytest.mark.parametrize("before,after", [
    ("NumCPUs=8", "NumCPUs=9"), ("JobHeldUser", "Priority"),
    ("NumTasks=1", "NumTasks=2"), ("gres/gpu=1", "gres/gpu=2"),
    ("OAT_ZERO_LEARNING_RATE=2e-07", "OAT_ZERO_LEARNING_RATE=1e-07"),
    ("Requeue=1", "Requeue=0"), ("Nice=0", "Nice=-1"),
])
def test_unrelated_actual_record_drift_still_fails_original_strict_audit(actual_held_fixture, before, after):
    raw, job_id, cell, original = actual_held_fixture
    assert before in raw
    with pytest.raises(ValueError):
        adapter.LauncherAdapter(original).audit_held_record(raw.replace(before, after), job_id, cell)


def test_live_audit_fetches_exact_job_and_returns_raw_without_scheduler_mutation(monkeypatch, actual_held_fixture):
    raw, job_id, cell, original = actual_held_fixture
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=raw, stderr="")
    monkeypatch.setattr(adapter.subprocess, "run", run)
    assert adapter.LauncherAdapter(original).audit_held(job_id, cell) == raw
    assert calls == [(["scontrol", "show", "job", "-dd", "-o", str(job_id)],
                      dict(capture_output=True, text=True, check=False, timeout=60))]
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout=raw, stderr="unavailable"))
    with pytest.raises(ValueError, match="unavailable"):
        adapter.LauncherAdapter(original).audit_held(job_id, cell)


@pytest.fixture
def amendment_fixture(tmp_path, monkeypatch):
    names = {key: tmp_path / name for key, name in {
        "SOURCE": "adapter.py", "TEST_SOURCE": "test_adapter.py", "PLAN_PATH": "plan.json",
        "LAUNCHER_PATH": "launcher.py", "CONTROLLER_PATH": "controller.py",
    }.items()}
    for key, path in names.items():
        path.write_text(key)
        monkeypatch.setattr(adapter, key, path)
    frozen = {str(names[key]): adapter.digest(names[key]) for key in ("LAUNCHER_PATH", "CONTROLLER_PATH")}
    monkeypatch.setattr(adapter, "FROZEN_SOURCES", frozen)
    plan = {"model_choice": "05b", "files_sha256": frozen}
    names["PLAN_PATH"].write_text(json.dumps(plan))
    monkeypatch.setattr(adapter, "PLAN_SHA256", adapter.digest(names["PLAN_PATH"]))
    extra = tmp_path / "first_submission_evidence.json"
    extra.write_text("immutable raw scheduler evidence")
    amendment = {
        "schema": "e122_slurm_2511_display_amendment_v1",
        "normalization": deepcopy(adapter.NORMALIZATION), "model_choice": "05b",
        "plan_path": str(names["PLAN_PATH"]), "plan_sha256": adapter.PLAN_SHA256,
        "files_sha256": {str(path): adapter.digest(path) for path in [*names.values(), extra]},
    }
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment))
    return path, amendment, names, extra


def test_amendment_authenticates_all_new_original_and_extra_pins_without_proof_or_scheduler_io(amendment_fixture, monkeypatch):
    path, amendment, _, extra = amendment_fixture
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: pytest.fail("no scheduler I/O in authentication"))
    assert adapter.authenticate_amendment(path, adapter.digest(path)) == amendment
    extra.write_text("changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        adapter.authenticate_amendment(path, adapter.digest(path))


@pytest.mark.parametrize("mutation", ["wrong_amendment_pin", "bad_schema", "changed_normalization", "extra_normalization", "wrong_plan", "wrong_model", "missing_new_pin", "missing_old_pin", "wrong_old_pin", "noncanonical_path", "source_changed", "plan_changed"])
def test_amendment_rejects_drift_and_broader_scope(amendment_fixture, mutation):
    path, amendment, names, _ = amendment_fixture
    expected = None
    if mutation == "wrong_amendment_pin":
        expected = "0" * 64
    elif mutation == "bad_schema":
        amendment["schema"] += "wrong"
    elif mutation == "changed_normalization":
        amendment["normalization"]["from"] = "1-2"
    elif mutation == "extra_normalization":
        amendment["normalization"]["also"] = "NumTasks"
    elif mutation == "wrong_plan":
        amendment["plan_sha256"] = "0" * 64
    elif mutation == "wrong_model":
        amendment["model_choice"] = "3b"
    elif mutation == "missing_new_pin":
        amendment["files_sha256"].pop(str(names["TEST_SOURCE"]))
    elif mutation == "missing_old_pin":
        amendment["files_sha256"].pop(str(names["CONTROLLER_PATH"]))
    elif mutation == "wrong_old_pin":
        amendment["files_sha256"][str(names["LAUNCHER_PATH"])] = "0" * 64
    elif mutation == "noncanonical_path":
        amendment["files_sha256"]["relative.py"] = "0" * 64
    elif mutation == "source_changed":
        names["SOURCE"].write_text("changed adapter")
    elif mutation == "plan_changed":
        names["PLAN_PATH"].write_text("changed frozen plan")
    path.write_text(json.dumps(amendment))
    with pytest.raises(ValueError):
        adapter.authenticate_amendment(path, expected or adapter.digest(path))


def test_controller_binding_authenticates_fresh_before_and_after_each_original_load(amendment_fixture, monkeypatch):
    path, amendment, _, extra = amendment_fixture
    campaign = {"binding": {"controller_sha256": "original"}, "jobs": []}
    calls = []
    def original_load(args, launcher):
        calls.append(launcher)
        return deepcopy(campaign)
    controller = SimpleNamespace(load_campaign=original_load, advance_once=object(), status=object())
    original_advance, original_status = controller.advance_once, controller.status
    auth = adapter.authenticate_amendment
    authentications = []
    def observed_auth(*args):
        authentications.append(args)
        return auth(*args)
    monkeypatch.setattr(adapter, "authenticate_amendment", observed_auth)
    pinned = adapter.digest(path)
    assert adapter.bind_controller(controller, path, pinned) is controller
    args = argparse.Namespace(plan=amendment["plan_path"], plan_sha256=amendment["plan_sha256"], model_choice="05b")
    sentinel = object()
    for _ in range(2):
        result = controller.load_campaign(args, sentinel)
        assert result["binding"] == dict(campaign["binding"], slurm2511_amendment_path=str(path), slurm2511_amendment_sha256=pinned)
    assert calls == [sentinel, sentinel] and len(authentications) == 5
    assert campaign["binding"] == {"controller_sha256": "original"}
    assert controller.advance_once is original_advance and controller.status is original_status
    extra.write_text("changed between advances")
    with pytest.raises(ValueError):
        controller.load_campaign(args, sentinel)
    assert calls == [sentinel, sentinel]


def test_changed_amendment_pin_cannot_reuse_existing_controller_journal(amendment_fixture):
    path, amendment, _, _ = amendment_fixture
    frozen = adapter.load_private_module(ROOT / "ops/exp_scaling/control_e122_level3_release.py", "adapter_journal_test")
    journal = path.parent / "journal"
    binding = {"slurm2511_amendment_path": str(path), "slurm2511_amendment_sha256": adapter.digest(path)}
    frozen.immutable_json(journal / "context.json", {"binding": binding})
    assert frozen.read_journals(journal, binding, set()) == {}
    with pytest.raises(ValueError, match="journal context differs"):
        frozen.read_journals(journal, dict(binding, slurm2511_amendment_sha256="0" * 64), set())
