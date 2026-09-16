"""The original E122 Python jobs remain provenance after two fresh migrations."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "campaign_stats_e122_migrations_under_test", ROOT / "ops/exp_scaling/campaign_stats.py"
)
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    runs = []
    for i in range(100):
        domain = "python_factors" if i >= 80 else "graph_coloring"
        runs.append({
            "job_id": 100 + i, "domain": domain, "dataset_domain": domain,
            "arm": ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")[i // 5 % 4],
            "seed": 43 + i % 5, "run_dir": str(tmp_path / f"original/{i}"),
            "run_stamp": f"original_{i}", "target_steps": 3072,
        })
    payload = {
        "schema": "e122_level3_factorial_jobs_v1", "runs": runs,
        "target_steps": 3072, "passes": 8, "train_rows": 384,
        "checkpoint_interval_steps": 192, "plan_path": str(tmp_path / "launch.json"),
        "plan_sha256": "launch-pin", "model_choice": "05b",
    }
    ledger = tmp_path / "original.json"
    ledger_sha = write(ledger, payload)
    binding = {
        "held_ledger_path": str(ledger), "held_ledger_sha256": ledger_sha,
        "plan_path": payload["plan_path"], "plan_sha256": payload["plan_sha256"],
        "model_choice": "05b",
    }
    stages = []
    prior = runs[80:]
    for stage, prefix in enumerate(("migration", "cli_migration")):
        directory = tmp_path / prefix
        plan_name = "plan.json"
        replacements = []
        for i, old in enumerate(prior):
            cell = {key: value for key, value in old.items() if key != "job_id"}
            cell.update(run_dir=str(tmp_path / f"stage{stage}/{i}"), run_stamp=f"stage{stage}_{i}")
            replacements.append({
                "campaign": "e122", "old_job_id": str(old["job_id"]),
                "job_id": str(300 + 100 * stage + i), "old_cell": old, "cell": cell,
            })
        plan = {"rows": replacements}
        if stage == 0:
            original_binding = directory / "e122_original_binding.json"
            plan["files_sha256"] = {str(original_binding): write(original_binding, binding)}
        else:
            plan["parent_migration_sha256"] = stages[0][3]
        plan_sha = write(directory / plan_name, plan)
        commit_sha = write(directory / "committed.json", {
            "plan_sha256": plan_sha, "replacements": replacements,
        })
        stages.append((prefix, directory, plan_name, plan_sha, commit_sha))
        prior = [{**row["cell"], "job_id": int(row["job_id"])} for row in replacements]
    monkeypatch.setattr(stats, "E122_LEDGER", ledger)
    monkeypatch.setattr(stats, "E122_MIGRATIONS", tuple(stages))
    return ledger, payload, stages, binding


def test_current_ids_paths_and_lineage_preserve_the_hundred_cells(campaign, monkeypatch):
    ledger, payload, _, _ = campaign
    old_bytes = ledger.read_bytes()
    queried = []

    def scheduler(ids):
        queried.extend(ids)
        return {
            jid: "COMPLETED" if jid < 111 else "RUNNING" if jid < 125 or 400 <= jid < 404 else "PENDING"
            for jid in ids
        }

    def step(path):
        # Old Python namespaces have misleading output and must not contribute.
        if "original" in str(path):
            i = int(path.name)
            return 3072 if i < 11 or i >= 80 else 100 if i < 25 else 0
        return 50 if int(path.name) < 4 else 0

    monkeypatch.setattr(stats.shared, "scheduler_states", scheduler)
    monkeypatch.setattr(stats.shared, "run_step", step)
    monkeypatch.setattr(stats.shared, "receipt_step", lambda path: 0)
    monkeypatch.setattr(stats.shared, "checkpoint_step", lambda path: 0)
    monkeypatch.setattr(stats.shared, "is_complete", lambda path, value, target: value >= target)
    snapshot = stats.load_static_snapshot(ledger)
    assert queried == list(range(100, 180)) + list(range(400, 420))
    assert len(snapshot["rows"]) == 100
    first = snapshot["rows"][80]
    assert first["original_job_id"] == 180
    assert first["job_id"] == first["effective_job_id"] == 400
    assert first["previous_job_ids"] == [180, 300]
    assert first["step"] == 50 and "stage1" in first["run_dir"]
    cohort = stats.registry.by_tag("e122")
    row = stats.cohort_row(cohort.label, ledger, cohort.resolved_reader(), cohort.excluded_domains)
    assert {key: row[key] for key in ("cells", "terminal", "running", "pending", "failed")} == {
        "cells": 100, "terminal": 11, "running": 18, "pending": 71, "failed": 0,
    }
    assert row["realized"] == 11 * 3072 + 14 * 100 + 4 * 50
    python_job = next(job for job in row["active_jobs"] if job["job_id"] == 400)
    assert python_job["previous_job_ids"] == [180, 300]
    assert ledger.read_bytes() == old_bytes


@pytest.mark.parametrize("name", ["plan.json", "committed.json"])
def test_changed_migration_bytes_are_not_trusted(campaign, name):
    ledger, payload, stages, _ = campaign
    path = stages[-1][1] / name
    path.write_text(path.read_text() + " ")
    assert stats.e122_migrated_campaign(ledger, payload) is None


@pytest.mark.parametrize("invalid", ["duplicate_job", "wrong_cell", "old_directory"])
def test_invalid_successor_identity_is_rejected_even_with_matching_hashes(campaign, monkeypatch, invalid):
    ledger, payload, stages, _ = campaign
    prefix, directory, plan_name, plan_sha, _ = stages[-1]
    committed = json.loads((directory / "committed.json").read_text())
    rows = committed["replacements"]
    if invalid == "duplicate_job":
        rows[1]["job_id"] = rows[0]["job_id"]
    elif invalid == "wrong_cell":
        rows[0]["cell"]["seed"] += 1
    else:
        rows[0]["cell"]["run_dir"] = rows[0]["old_cell"]["run_dir"]
    commit_sha = write(directory / "committed.json", committed)
    monkeypatch.setattr(stats, "E122_MIGRATIONS", (stages[0], (prefix, directory, plan_name, plan_sha, commit_sha)))
    assert stats.e122_migrated_campaign(ledger, payload) is None


def test_uncommitted_second_migration_retains_the_first(campaign):
    ledger, payload, stages, _ = campaign
    (stages[-1][1] / "committed.json").unlink()
    migrated = stats.e122_migrated_campaign(ledger, payload)
    assert migrated["runs"][80]["job_id"] == 300
    assert migrated["runs"][80]["previous_job_ids"] == [180]
    assert migrated["status_dir"] == stages[0][1] / "e122_release_controller/status"


def test_original_ledger_must_match_its_registered_binding(campaign):
    ledger, payload, _, _ = campaign
    payload["runs"][0]["run_dir"] += "_unregistered"
    write(ledger, payload)
    assert stats.e122_migrated_campaign(ledger, payload) is None


def test_release_snapshot_uses_current_migration_binding(campaign):
    ledger, payload, stages, binding = campaign
    migrated = stats.e122_migrated_campaign(ledger, payload)
    snapshot = {
        "schema": "e122_level3_release_status_v1",
        "binding": {**binding, **migrated["binding"]},
        "observed_at": "2026-09-12T17:27:00Z", "blocked_reason": "concurrency_cap",
        "staged_held": 71, "released_nonterminal": 18, "reserved_unfinished_slots": 18,
        "running": 18, "successful_endpoints": 11, "unknown": 0,
        "max_released_nonterminal": 4,
        "storage": {"free_bytes": 1000, "remaining_headroom_bytes": 500},
    }
    path = migrated["status_dir"] / "latest.json"
    write(path, snapshot)
    observed = stats.level3_release_status(ledger, payload)
    assert observed["running"] == 18 and observed["source"] == str(path)
    snapshot["binding"].pop("cli_migration_commit_sha256")
    write(path, snapshot)
    assert stats.level3_release_status(ledger, payload) is None
