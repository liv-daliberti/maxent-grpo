import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "campaign_stats_live_allocations_under_test",
    ROOT / "ops/exp_scaling/campaign_stats.py",
)
assert SPEC is not None and SPEC.loader is not None
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def test_plain_report_lists_every_live_job_and_latest_restart() -> None:
    jobs = [
        {
            "cohort": "E118",
            "job_id": 200,
            "original_job_id": 200,
            "previous_job_ids": [100],
            "state": "RUNNING",
            "elapsed": "1:02",
            "node": "node205",
            "scale": "qwen3b",
            "domain": "python_factors",
            "arm": "maxrl",
            "seed": 73,
            "step": 1920,
            "target": 3072,
        },
        {
            "cohort": "E119",
            "job_id": 201,
            "original_job_id": 101,
            "previous_job_ids": [101],
            "state": "RUNNING",
            "elapsed": "0:31",
            "node": "node105",
            "scale": "qwen05b",
            "domain": "countdown",
            "arm": "drgrpo",
            "seed": 44,
            "step": 1728,
            "target": 3072,
        },
    ]
    rows = [
        {
            "label": "E118 test",
            "cells": 2,
            "terminal": 0,
            "running": 2,
            "pending": 0,
            "failed": 0,
            "realized": 3648,
            "total": 6144,
            "depth": 4.75,
            "passes": 8,
            "active_jobs": jobs,
        }
    ]
    recovery = (
        {
            "cohort": "e119",
            "old_job_id": 101,
            "job_id": 201,
            "scale": "qwen05b",
            "domain": "countdown",
            "arm": "drgrpo",
            "seed": 44,
            "checkpoint": 1728,
            "state": "RUNNING",
            "node": "node105",
        },
    )

    rendered = stats.render(
        rows,
        markdown=False,
        recovery_rows=recovery,
    )

    assert "101->201" in rendered
    assert "200" in rendered and "201" in rendered
    assert "2 listed; aggregate running count 2" in rendered
    assert "restart<-100" in rendered
    assert "restart<-101" in rendered


def _write_recovery_lineages(tmp_path, monkeypatch):
    import hashlib
    import json

    e118_runs, e119_runs, replacements = [], [], []
    for index in range(6):
        old, failed, current = 100 + index, 200 + index, 300 + index
        cohort = 'e118' if index < 5 else 'e119'
        run_dir = tmp_path / f'run_{index}'
        run_dir.mkdir()
        run = {
            'domain': 'python_factors' if index < 5 else 'countdown',
            'arm': 'maxrl' if index < 5 else 'drgrpo',
            'scale': 'qwen3b' if index < 5 else 'qwen05b',
            'seed': 70 + index if index < 5 else 44,
            'run_dir': str(run_dir),
            'run_stamp': f'run_{index}',
            'job_id': current if index < 5 else old,
        }
        if index < 5:
            run['previous_job_ids'] = [old, failed]
            e118_runs.append(run)
        else:
            e119_runs.append(run)
        replacements.append({
            **{key: run[key] for key in ('domain', 'arm', 'scale', 'seed', 'run_dir')},
            'cohort': cohort,
            'old_job_id': old,
            'new_job_id': failed,
            'resume_checkpoint': str(run_dir / 'debug_old/checkpoints/step_01920'),
        })
    e118_path, e119_path = tmp_path / 'e118.json', tmp_path / 'e119.json'
    e118_path.write_text(json.dumps({'runs': e118_runs, 'target_steps': 3072}))
    e119_path.write_text(json.dumps({'runs': e119_runs, 'target_steps': 3072}))
    recovery_path = tmp_path / 'recovery.json'
    recovery_path.write_text(json.dumps({
        'schema': 'e118-e119-overnight-failure-recovery-v1',
        'released': True,
        'replacements': replacements,
    }))
    continuation_path = tmp_path / 'continuations.json'
    continuation_path.write_text(json.dumps({
        'schema': 'e119_level2_continuation_jobs_v1',
        'original_ledger': str(e119_path),
        'original_ledger_sha256': hashlib.sha256(e119_path.read_bytes()).hexdigest(),
        'same_scientific_cells': True,
        'same_run_directories': True,
        'optimizer_update_changed': False,
        'treatment_changed': False,
        'released': True,
        'installed': True,
        'outcomes_inspected': False,
        'continuations': [{
            **e119_runs[0],
            'original_job_id': 105,
            'continuation_job_id': 305,
            'previous_continuation_job_ids': [205, 255],
        }],
    }))
    monkeypatch.setattr(stats, 'E118_LEDGER', e118_path)
    monkeypatch.setattr(stats, 'E119_LEDGER', e119_path)
    monkeypatch.setattr(stats, 'E119_CONTINUATIONS', continuation_path)
    monkeypatch.setattr(stats, 'LATEST_RECOVERY', recovery_path)
    return e118_runs + e119_runs


def test_recovery_report_resolves_successors_and_completed_receipts(tmp_path, monkeypatch):
    import json

    runs = _write_recovery_lineages(tmp_path, monkeypatch)
    for run in runs[:-1]:
        (Path(run['run_dir']) / 'TRAINING_COMPLETE.json').write_text(json.dumps({
            'schema': 'oat_zero_training_complete_v1', 'terminal_step': 3073,
        }))
    queried = []

    def states(job_ids):
        queried.extend(job_ids)
        # Completed science cells remain terminal even if scheduler history is stale.
        return {job: 'FAILED' for job in job_ids}

    monkeypatch.setattr(stats.shared, 'scheduler_states', states)
    monkeypatch.setattr(stats, 'scheduler_details', lambda ids: {
        305: {'state': 'RUNNING', 'node': 'node105'},
    })
    rows = stats.latest_recovery_rows()

    assert queried == list(range(300, 306))
    assert [row['job_id'] for row in rows] == list(range(300, 306))
    assert [row['new_job_id'] for row in rows] == list(range(200, 206))
    assert [row['state'] for row in rows] == ['COMPLETED'] * 5 + ['RUNNING']
    for markdown in (False, True):
        rendered = stats.render([], markdown=markdown, recovery_rows=tuple(rows))
        assert 'current successors' in rendered
        assert 'original resume' in rendered
        assert 'failed' not in rendered.split('current successors', 1)[1]
        assert '300' not in rendered and '305' in rendered
        history = stats.render([], markdown=markdown, recovery_rows=tuple(rows), include_history=True)
        assert '300' in history and '305' in history
        completed = stats.render([], markdown=markdown, recovery_rows=tuple(rows[:-1]))
        assert 'current successors' not in completed
        assert '200' not in rendered and '205' not in rendered


def test_recovery_report_rejects_unregistered_same_path_successor(tmp_path, monkeypatch):
    import json

    _write_recovery_lineages(tmp_path, monkeypatch)
    ledger = json.loads(stats.E118_LEDGER.read_text())
    ledger['runs'][0]['previous_job_ids'] = [999]
    stats.E118_LEDGER.write_text(json.dumps(ledger))
    monkeypatch.setattr(stats.shared, 'scheduler_states', lambda ids: {})
    monkeypatch.setattr(stats, 'scheduler_details', lambda ids: {})

    assert stats.latest_recovery_rows() == []



def _planned_factorial_payload(tmp_path):
    domains = ["graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan"]
    arms = ["drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl"]
    seeds = [43, 44, 45, 46, 47]
    return {
        "released": False,
        "model": None,
        "model_choice_pending": True,
        "domains": domains,
        "arms": arms,
        "seeds": seeds,
        "target_steps": 3072,
        "passes": 8,
        "runs": [],
        "admission": {"reason": "Fresh confirmation and model selection"},
        "planned_runs": [
            {
                "domain": domain,
                "arm": arm,
                "seed": seed,
                "run_dir": str(tmp_path / domain / arm / f"s{seed}"),
            }
            for domain in domains
            for arm in arms
            for seed in seeds
        ],
    }


def test_planned_factorial_reports_budget_without_querying_jobs(tmp_path, monkeypatch):
    import json

    path = tmp_path / "planned_jobs.json"
    path.write_text(json.dumps(_planned_factorial_payload(tmp_path)))

    def unexpected_query(*args, **kwargs):
        raise AssertionError("an unsubmitted campaign must not query jobs or run metrics")

    monkeypatch.setattr(stats.shared, "scheduler_states", unexpected_query)
    monkeypatch.setattr(stats.shared, "run_step", unexpected_query)
    row = stats.cohort_row("E122 Level-3 factorial", path, "smoke")
    assert row is not None
    assert row["status"] == "AWAITING_ADMISSION"
    assert row["cells"] == row["planned_cells"] == 100
    assert row["total"] == 307200
    assert row["passes"] == 8
    for field in ("terminal", "running", "pending", "failed", "realized", "depth"):
        assert row[field] == 0
    assert "active_jobs" not in row
    for markdown in (False, True):
        rendered = stats.render([row], markdown=markdown)
        assert "AWAITING_ADMISSION" in rendered
        assert "100 planned cells; no jobs submitted" in rendered
        assert "Fresh confirmation and model selection" in rendered
        assert "307,200" in rendered
        assert "Live registered science allocations" not in rendered
    assert "0.00/8p" in stats.render([row], markdown=False)


def test_planned_factorial_rejects_invented_jobs_and_invalid_matrix(tmp_path):
    import copy
    import pytest

    original = _planned_factorial_payload(tmp_path)
    malformed = []
    payload = copy.deepcopy(original)
    payload["planned_runs"][0]["job_id"] = 123
    malformed.append(payload)
    payload = copy.deepcopy(original)
    payload["planned_runs"].pop()
    malformed.append(payload)
    payload = copy.deepcopy(original)
    payload["planned_runs"][1] = payload["planned_runs"][0]
    malformed.append(payload)
    payload = copy.deepcopy(original)
    payload["released"] = True
    malformed.append(payload)
    for payload in malformed:
        with pytest.raises(ValueError):
            stats.planned_cohort_row("E122", payload)


def test_submitted_factorial_uses_real_jobs_even_before_release(tmp_path, monkeypatch):
    import json

    payload = _planned_factorial_payload(tmp_path)
    payload["runs"] = [
        {**run, "job_id": index + 1}
        for index, run in enumerate(payload["planned_runs"])
    ]
    path = tmp_path / "held_jobs.json"
    path.write_text(json.dumps(payload))
    queries = []

    def scheduler_states(job_ids):
        queries.append(job_ids)
        return {job_id: "PENDING" for job_id in job_ids}

    monkeypatch.setattr(stats.shared, "scheduler_states", scheduler_states)
    monkeypatch.setattr(stats.shared, "run_step", lambda run_dir: 0)
    row = stats.cohort_row("E122 Level-3 factorial", path, "smoke")
    assert queries == [list(range(1, 101))]
    assert row["cells"] == row["pending"] == 100
    assert row["running"] == row["terminal"] == row["failed"] == 0
    assert "planned_cells" not in row
    assert "AWAITING_ADMISSION" not in stats.render([row], markdown=False)


def test_e122_registry_entry_keeps_the_level2_campaign_separate():
    e119 = stats.registry.by_tag("e119")
    e122 = stats.registry.by_tag("e122")
    assert e119.ledger == "e119_level2_qwen05b_factorial_jobs.json"
    assert "Qwen-0.5B Level-2" in e119.label
    assert e122.ledger == "e122_level3_factorial_jobs.json"
    assert e122.resolved_reader() == "smoke"
    assert "Qwen-0.5B Level-3" in e122.label
    assert (e122.label, e122.ledger, "smoke", ()) in stats.COHORTS



def _e122_release_snapshot(tmp_path, monkeypatch):
    import hashlib
    import json

    payload = _planned_factorial_payload(tmp_path)
    payload.update(
        schema="e122_level3_factorial_jobs_v1",
        plan_path=str(tmp_path / "plan.json"),
        plan_sha256="a" * 64,
        model_choice="05b",
    )
    payload["runs"] = [
        {**run, "job_id": index + 1}
        for index, run in enumerate(payload["planned_runs"])
    ]
    ledger = tmp_path / "e122_jobs.json"
    ledger.write_text(json.dumps(payload))
    status_dir = tmp_path / "release_status"
    status_dir.mkdir()
    monkeypatch.setattr(stats, "E122_RELEASE_STATUS_DIR", status_dir)
    snapshot = {
        "schema": "e122_level3_release_status_v1",
        "observed_at": "2026-09-09T15:00:00+00:00",
        "binding": {
            "held_ledger_path": str(ledger),
            "held_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            "plan_path": payload["plan_path"],
            "plan_sha256": payload["plan_sha256"],
            "model_choice": "05b",
        },
        "staged_held": 96,
        "released_nonterminal": 4,
        "reserved_unfinished_slots": 4,
        "running": 0,
        "successful_endpoints": 0,
        "unknown": 0,
        "max_released_nonterminal": 4,
        "blocked_reason": "concurrency_cap",
        "storage": {"free_bytes": 300 * 1024**3, "remaining_headroom_bytes": 111 * 1024**3},
    }
    return ledger, payload, status_dir, snapshot


def test_e122_release_note_distinguishes_held_jobs_without_changing_totals(tmp_path, monkeypatch):
    import json

    ledger, payload, status_dir, snapshot = _e122_release_snapshot(tmp_path, monkeypatch)
    monkeypatch.setattr(stats.shared, "scheduler_states", lambda ids: {job: "PENDING" for job in ids})
    monkeypatch.setattr(stats.shared, "run_step", lambda path: 0)
    for reason in ("waiting_disk", "concurrency_cap", "needs_operator_review"):
        snapshot["blocked_reason"] = reason
        (status_dir / "20260909T150000.json").write_text(json.dumps(snapshot))
        row = stats.cohort_row("E122 Qwen-0.5B Level-3 factorial", ledger, "smoke")
        assert row["pending"] == row["cells"] == 100
        assert row["running"] == row["terminal"] == row["failed"] == 0
        assert row["release_status"]["staged_held"] == 96
        for markdown in (False, True):
            rendered = stats.render([row], markdown=markdown)
            assert "release snapshot (2026-09-09T15:00:00+00:00)" in rendered
            assert "96 staged/held; 4 released and nonterminal (0 running)" in rendered
            assert "4/4 reserved slots" in rendered
            assert reason.replace("_", " ") in rendered
            assert "disk free 300.0 GiB, remaining headroom 111.0 GiB" in rendered


def test_e122_release_note_ignores_missing_malformed_or_mismatched_snapshots(tmp_path, monkeypatch):
    import copy
    import json

    ledger, payload, status_dir, snapshot = _e122_release_snapshot(tmp_path, monkeypatch)
    assert stats.e122_release_status(ledger, payload) is None
    path = status_dir / "20260909T150000.json"
    path.write_text("{partial")
    assert stats.e122_release_status(ledger, payload) is None
    for field in ("held_ledger_sha256", "plan_sha256", "model_choice"):
        invalid = copy.deepcopy(snapshot)
        invalid["binding"][field] = "mismatch"
        path.write_text(json.dumps(invalid))
        assert stats.e122_release_status(ledger, payload) is None
    path.write_text(json.dumps(snapshot))
    assert stats.e122_release_status(ledger, payload)["staged_held"] == 96
    ledger.write_text(json.dumps(payload) + "\n")
    assert stats.e122_release_status(ledger, payload) is None


def test_e123_registry_preserves_separate_model_and_difficulty_campaigns():
    e119 = stats.registry.by_tag("e119")
    e122 = stats.registry.by_tag("e122")
    e123 = stats.registry.by_tag("e123")
    assert len({cohort.ledger for cohort in (e119, e122, e123)}) == 3
    assert e123.ledger == "e123_level3_factorial_jobs.json"
    assert "Qwen-3B Level-3" in e123.label
    assert e123.resolved_reader() == "smoke"
    assert (e123.label, e123.ledger, "smoke", ()) in stats.COHORTS


def _e123_factorial_payload(tmp_path):
    payload = _planned_factorial_payload(tmp_path)
    payload.update(
        schema="e123_level3_factorial_jobs_v1",
        model="Qwen2.5-3B-Instruct",
        model_choice="3b",
        model_choice_pending=False,
        seeds=[70, 71, 72, 73, 74],
        admission={"reason": "A100 systems benchmark awaiting measured production profile"},
    )
    for run in payload["planned_runs"]:
        run["seed"] += 27
        run["run_dir"] = str(tmp_path / run["domain"] / run["arm"] / f"s{run['seed']}")
    return payload


def test_e123_planned_budget_excludes_systems_benchmark_jobs(tmp_path, monkeypatch):
    import json

    payload = _e123_factorial_payload(tmp_path)
    payload["systems_benchmark"] = {"job_ids": [901, 902], "status": "RUNNING"}
    ledger = tmp_path / "e123_jobs.json"
    ledger.write_text(json.dumps(payload))

    def unexpected_query(*args, **kwargs):
        raise AssertionError("benchmark jobs must not be counted as E123 training")

    monkeypatch.setattr(stats.shared, "scheduler_states", unexpected_query)
    monkeypatch.setattr(stats.shared, "run_step", unexpected_query)
    row = stats.cohort_row(stats.registry.by_tag("e123").label, ledger, "smoke")
    assert row["cells"] == row["planned_cells"] == 100
    assert row["total"] == 307200
    assert row["running"] == row["pending"] == row["realized"] == 0
    for markdown in (False, True):
        rendered = stats.render([row], markdown=markdown)
        assert "100 planned cells; no jobs submitted" in rendered
        assert "A100 systems benchmark awaiting measured production profile" in rendered
        assert "E123 Qwen-3B Level-3" in rendered


def _e123_release_snapshot(tmp_path, monkeypatch):
    import hashlib
    import json

    payload = _e123_factorial_payload(tmp_path)
    payload.update(plan_path=str(tmp_path / "e123_plan.json"), plan_sha256="b" * 64)
    payload["runs"] = [
        {**run, "job_id": 1001 + index}
        for index, run in enumerate(payload["planned_runs"])
    ]
    ledger = tmp_path / "e123_jobs.json"
    ledger.write_text(json.dumps(payload))
    status_dir = tmp_path / "e123_release_status"
    status_dir.mkdir()
    monkeypatch.setattr(stats, "E123_RELEASE_STATUS_DIR", status_dir)
    snapshot = {
        "schema": "e123_level3_release_status_v1",
        "observed_at": "2026-09-09T20:00:00+00:00",
        "binding": {
            "held_ledger_path": str(ledger),
            "held_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            "plan_path": payload["plan_path"],
            "plan_sha256": payload["plan_sha256"],
            "model_choice": "3b",
        },
        "staged_held": 92,
        "released_nonterminal": 8,
        "reserved_unfinished_slots": 8,
        "running": 2,
        "successful_endpoints": 0,
        "unknown": 0,
        "max_released_nonterminal": 8,
        "blocked_reason": "concurrency_cap",
        "storage": {"free_bytes": 1600 * 1024**3, "remaining_headroom_bytes": 239 * 1024**3},
    }
    return ledger, payload, status_dir, snapshot


def test_e123_release_snapshot_and_running_jobs_use_the_science_ledger(tmp_path, monkeypatch):
    import json

    ledger, payload, status_dir, snapshot = _e123_release_snapshot(tmp_path, monkeypatch)
    (status_dir / "20260909T200000.json").write_text(json.dumps(snapshot))
    queries = []

    def scheduler_states(ids):
        queries.append(ids)
        return {job: "RUNNING" if job in (1001, 1002) else "PENDING" for job in ids}

    monkeypatch.setattr(stats.shared, "scheduler_states", scheduler_states)
    monkeypatch.setattr(stats.shared, "run_step", lambda path: 0)
    row = stats.cohort_row(stats.registry.by_tag("e123").label, ledger, "smoke")
    assert queries == [list(range(1001, 1101))]
    assert row["cells"] == 100 and row["running"] == 2 and row["pending"] == 98
    assert row["terminal"] == row["failed"] == row["realized"] == 0
    assert {job["job_id"] for job in row["active_jobs"]} == {1001, 1002}
    assert {job["scale"] for job in row["active_jobs"]} == {"qwen3b"}
    assert {job["cohort"] for job in row["active_jobs"]} == {"E123"}
    for markdown in (False, True):
        rendered = stats.render([row], markdown=markdown)
        assert "92 staged/held; 8 released and nonterminal (2 running)" in rendered
        assert "8/8 reserved slots" in rendered
        assert "E123 release snapshot" in rendered


def test_e123_release_snapshot_rejects_other_campaigns_and_changed_binding(tmp_path, monkeypatch):
    import copy
    import json

    ledger, payload, status_dir, snapshot = _e123_release_snapshot(tmp_path, monkeypatch)
    path = status_dir / "20260909T200000.json"
    assert stats.level3_release_status(ledger, payload) is None
    invalid_cases = []
    cross_campaign = copy.deepcopy(snapshot)
    cross_campaign["schema"] = "e122_level3_release_status_v1"
    invalid_cases.append(cross_campaign)
    for key in ("held_ledger_path", "held_ledger_sha256", "plan_path", "plan_sha256", "model_choice"):
        invalid = copy.deepcopy(snapshot)
        invalid["binding"][key] = "wrong"
        invalid_cases.append(invalid)
    for invalid in invalid_cases:
        path.write_text(json.dumps(invalid))
        assert stats.level3_release_status(ledger, payload) is None
    path.write_text(json.dumps(snapshot))
    assert stats.level3_release_status(ledger, payload)["staged_held"] == 92
    assert stats.e122_release_status(ledger, payload) is None
    ledger.write_text(json.dumps(payload) + "\n")
    assert stats.level3_release_status(ledger, payload) is None
