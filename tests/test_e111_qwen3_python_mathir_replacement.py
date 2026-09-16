from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"
LAUNCHER = EXP / (
    "launch_e111_qwen3_python_mathir_replacement_after_purged_timeout.py"
)
AUDITOR = EXP / "audit_e111_qwen3_python_mathir_replacement.py"
STATS = EXP / "campaign_stats.py"
LEDGER = ROOT / "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"
RECORD = ROOT / "var/artifacts/e111_qwen3_python_mathir_replacement_jobs.json"
SECOND_RECORD = (
    ROOT / "var/artifacts/e111_qwen3_python_second_continuation_jobs.json"
)
PANTRY_LAUNCHER = EXP / (
    "launch_e111_qwen3_pantry_continuation_after_purged_timeout.py"
)
PANTRY_AUDITOR = EXP / "audit_e111_qwen3_pantry_continuation.py"
PANTRY_RECORD = ROOT / "var/artifacts/e111_qwen3_pantry_continuation_jobs.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e111_purged_timeout_continuations_are_exact_two_cells():
    launch = _load(LAUNCHER, "e111_timeout_continuation_launch")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    snapshot = Path(ledger["snapshot_root"])
    runs = launch.original_runs(ledger)
    assert [run["domain"] for run in runs] == ["python_factors", "mathir"]
    assert [run["job_id"] for run in runs] == [30674760, 30674761]
    for run in runs:
        command, env, _template = launch.continuation_command(run, snapshot)
        assert "--hold" in command
        assert "--partition=lowprio" in command
        assert f"--nodelist={launch.NODELIST}" in command
        assert "--gres=gpu:a6000:1" in command
        assert "--time=00:45:00" in command
        assert env["SAVE_PATH"] == run["run_dir"]
        assert env["RUN_STAMP"] == run["run_stamp"]
        assert env["OAT_ZERO_SEED"] == "70"
        assert env["OAT_ZERO_MAX_TRAIN"] == "8"
        assert env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "8"
        assert env["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == "64"
        assert env["OAT_ZERO_SAVE_STEPS"] == "2"
        assert env["OAT_ZERO_SAVE_FROM"] == "2"
        assert env["OAT_ZERO_RESUME_STEPS"] == "2"
        assert env["OAT_ZERO_RESUME_FROM"] == "2"
        assert "point" not in run["domain"].lower()

    pantry_launch = _load(PANTRY_LAUNCHER, "e111_pantry_continuation_launch")
    pantry_run = pantry_launch.original_run(ledger)
    assert pantry_run["job_id"] == 30674762
    assert pantry_run["domain"] == "pantry_plan"
    command, env, _template = pantry_launch.shared.continuation_command(
        pantry_run, snapshot
    )
    assert "--hold" in command
    assert "--partition=lowprio" in command
    assert f"--nodelist={pantry_launch.shared.NODELIST}" in command
    assert "--gres=gpu:a6000:1" in command
    assert "--time=00:45:00" in command
    assert env["SAVE_PATH"] == pantry_run["run_dir"]
    assert env["RUN_STAMP"] == pantry_run["run_stamp"]
    assert env["OAT_ZERO_SAVE_STEPS"] == "2"
    assert env["OAT_ZERO_RESUME_FROM"] == "2"


def test_e111_purged_timeout_continuation_record_passes_independent_audit():
    audit = _load(AUDITOR, "e111_timeout_continuation_audit")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    report, violations = audit.validate(ledger, ledger["runs"])
    assert violations == []
    assert report["passed"] is True
    assert report["released"] is True
    assert report["installed"] is True
    assert report["outcomes_inspected"] is False
    assert report["pointmaze"] == "excluded"
    assert set(report["continuation_by_original_job_id"]) == {
        "30674760",
        "30674761",
    }
    second = json.loads(SECOND_RECORD.read_text(encoding="utf-8"))
    assert report["second_continuation_job_id"] == second["continuation_job_id"]
    assert report["continuation_by_original_job_id"]["30674760"][
        "continuation_job_id"
    ] == second["continuation_job_id"]
    assert report["continuation_chains"]["30674760"] == [
        30674760,
        30739797,
        second["continuation_job_id"],
    ]

    pantry_audit = _load(PANTRY_AUDITOR, "e111_pantry_continuation_audit")
    pantry_report, pantry_violations = pantry_audit.validate(
        ledger, ledger["runs"]
    )
    pantry = json.loads(PANTRY_RECORD.read_text(encoding="utf-8"))
    assert pantry_violations == []
    assert pantry_report["passed"] is True
    assert pantry_report["outcomes_inspected"] is False
    assert pantry_report["pointmaze"] == "excluded"
    assert pantry_report["continuation_chains"]["30674762"] == [
        30674762,
        pantry["continuation_job_id"],
    ]


def test_e111_purged_timeout_audit_rejects_mapping_drift(tmp_path, monkeypatch):
    audit = _load(AUDITOR, "e111_timeout_continuation_tamper")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    payload = json.loads(RECORD.read_text(encoding="utf-8"))
    payload["continuations"][1]["original_job_id"] = 30674760
    tampered = tmp_path / "replacement.json"
    tampered.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(audit, "RECORD", tampered)
    report, violations = audit.validate(ledger, ledger["runs"])
    assert report["passed"] is False
    assert "duplicate original continuation identity 30674760" in violations
    assert "E111 Qwen-3B replacement original-job set mismatch" in violations

    monkeypatch.setattr(audit, "RECORD", RECORD)
    second_payload = json.loads(SECOND_RECORD.read_text(encoding="utf-8"))
    second_payload["prior_continuation_job_id"] = 30739798
    tampered_second = tmp_path / "second-replacement.json"
    tampered_second.write_text(
        json.dumps(second_payload) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(audit, "SECOND_RECORD", tampered_second)
    second_report, second_violations = audit.validate(ledger, ledger["runs"])
    assert second_report["passed"] is False
    assert (
        "E111 Qwen-3B Python second continuation has invalid "
        "prior_continuation_job_id"
    ) in second_violations
    assert second_report["continuation_by_original_job_id"]["30674760"][
        "continuation_job_id"
    ] == 30739797


def test_campaign_stats_uses_effective_continuation_job_ids(
    tmp_path, monkeypatch
):
    stats = _load(STATS, "campaign_stats_e111_continuations")
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    mapping = stats.e111_continuation_jobs(LEDGER)
    assert set(mapping) == {30674760, 30674761, 30674762}
    assert len(set(mapping.values())) == 3
    second = json.loads(SECOND_RECORD.read_text(encoding="utf-8"))
    pantry = json.loads(PANTRY_RECORD.read_text(encoding="utf-8"))
    assert mapping[30674760] == second["continuation_job_id"]
    assert mapping[30674762] == pantry["continuation_job_id"]
    captured: list[int] = []

    def states(job_ids):
        captured.extend(job_ids)
        return {job_id: "PENDING" for job_id in job_ids}

    monkeypatch.setattr(stats.shared, "scheduler_states", states)
    monkeypatch.setattr(stats.shared, "run_step", lambda _path: 0)
    monkeypatch.setattr(
        stats.shared,
        "is_complete",
        lambda _path, _step, _target: False,
    )
    snapshot = stats.load_smoke_snapshot(LEDGER)
    assert len(snapshot["rows"]) == 15
    assert all(job_id not in captured for job_id in (30674760, 30674761, 30674762))
    assert all(mapping[job_id] in captured for job_id in mapping)
    rows = {row["original_job_id"]: row for row in snapshot["rows"]}
    assert rows[30674760]["effective_job_id"] == mapping[30674760]
    assert rows[30674761]["effective_job_id"] == mapping[30674761]
    assert rows[30674762]["effective_job_id"] == mapping[30674762]

    tampered = dict(second)
    tampered["prior_continuation_job_id"] = 30739798
    tampered_path = tmp_path / "second-continuation.json"
    tampered_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    monkeypatch.setattr(stats, "E111_SECOND_CONTINUATION", tampered_path)
    fallback = stats.e111_continuation_jobs(LEDGER)
    assert fallback[30674760] == 30739797
