#!/usr/bin/env python3
"""Report cells, terminal counts, and realized depth for every active cohort.

Read-only. Progress comes from realized optimizer steps in each run's metrics
log; the scheduler query only explains which cells are running or waiting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
import status_e78 as shared  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "var/artifacts"
E111_LEDGER = ARTIFACTS / "e111_verified_support_discovery_mechanism_gate_jobs.json"
E111_CONTINUATIONS = ARTIFACTS / "e111_qwen3_python_mathir_replacement_jobs.json"
E111_SECOND_CONTINUATION = ARTIFACTS / "e111_qwen3_python_second_continuation_jobs.json"
E111_PANTRY_CONTINUATION = ARTIFACTS / "e111_qwen3_pantry_continuation_jobs.json"
E109_LEDGER = ARTIFACTS / "e109_repaired_python_replay_comparators_jobs.json"
E109_CONTINUATIONS = ARTIFACTS / "e109r1_qwen3_python_continuation_jobs.json"
E117_STAGE1_LEDGER = ARTIFACTS / "e117_stage1_development_jobs.json"
E117_STAGE1_CONTINUATIONS = ARTIFACTS / "e117_stage1_continuation_jobs.json"
E118_LEDGER = ARTIFACTS / "e118_all_scales_maxrl_verified_replay_jobs.json"
E119_LEDGER = ARTIFACTS / "e119_level2_qwen05b_factorial_jobs.json"
E119_CONTINUATIONS = ARTIFACTS / "e119_level2_continuation_jobs.json"
E120_LEDGER = ARTIFACTS / "e120r1_frequency_weighted_replay_jobs.json"
E120_CONTINUATIONS = ARTIFACTS / "e120r1_scheduler_continuation_jobs.json"
E122_LEDGER = ARTIFACTS / "e122_level3_factorial_jobs.json"
E122_RELEASE_STATUS_DIR = ARTIFACTS / "e122_level3_factorial/release_controller/status"
# Immutable, committed migrations: the original held ledger remains provenance.
# Pin both the admitted plan and its exact scheduler successor assignments.
E122_MIGRATIONS = (
    (
        "migration",
        ARTIFACTS / "python_level3_neutral_migration_v5_20260911",
        "plan.json",
        "3b5c13cdab5a4d7cd0ca71437736b1f8189358e945b7103a4791470c0b47b903",
        "22714737f9bd89c3649174bc5d8dbe34f718088d97bef11015f77daa661d7921",
    ),
    (
        "cli_migration",
        ARTIFACTS / "python_level3_cli_recovery_20260912",
        "migration_plan.json",
        "af966347f733140f055ba5f07c43cca445303d980c5a394d3c77e839a01d2854",
        "2f1eb0ed77b5681b7b814a72a1ba17914788f0cc3a63b153c50375b1b1d6c523",
    ),
)
E123_RELEASE_STATUS_DIR = ARTIFACTS / "e123_level3_factorial/release_controller/status"
LATEST_RECOVERY = (
    ARTIFACTS / "e118_e119_overnight_failure_recovery_20260904.json"
)
E105_LEDGER_NAME = "e105_group_centered_semantic_repair_full_three_scale_jobs.json"
E105_RETIREMENT = ARTIFACTS / "e105_superseded_v6_retirement_jobs.json"
E112_LEDGER_NAME = "e112_verified_support_discovery_full_three_scale_jobs.json"
E112_RETIREMENT = ARTIFACTS / "e112_sampler_contract_failure_retirement.json"
E113_R3_LEDGER = ARTIFACTS / "e113r3_dapo_full_relaunch_jobs.json"
E113_R3_RETIREMENT = ARTIFACTS / "e113r3_retirement_for_official_dapo.json"


import cohorts as registry  # noqa: E402

# Derived from the single cohort registry, so a launched cohort cannot be
# missing from this table without also failing test_cohort_registry.py.
COHORTS = tuple(
    (c.label, c.ledger, c.resolved_reader(), c.excluded_domains)
    for c in registry.REGISTRY
)
SCIENCE_FAILURE_STATES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")
SCIENCE_RUNNING_STATES = ("RUNNING", "CONFIGURING", "COMPLETING")


def e105_retirement() -> dict[str, object] | None:
    if not E105_RETIREMENT.is_file():
        return None
    ledger_path = ARTIFACTS / E105_LEDGER_NAME
    try:
        payload = json.loads(E105_RETIREMENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "schema": "e105_superseded_v6_retirement_jobs_v1",
        "ledger": str(ledger_path),
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "exact_job_ids": list(range(30659732, 30659807)),
        "active_after_count": 0,
        "installed": True,
        "e109_excluded": True,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "auxiliary_job_ids": [30660073],
        "auxiliary_active_after": False,
        "auxiliary_canceled": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    return payload


def e112_retirement() -> dict[str, object] | None:
    if not E112_RETIREMENT.is_file():
        return None
    ledger_path = ARTIFACTS / E112_LEDGER_NAME
    try:
        payload = json.loads(E112_RETIREMENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not ledger_path.is_file():
        return None
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        exact_ids = [int(run["job_id"]) for run in ledger.get("runs", [])]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    expected = {
        "schema": "e112_sampler_contract_failure_retirement_v1",
        "installed": True,
        "original_ledger": str(ledger_path),
        "original_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "exact_job_ids": exact_ids,
        "active_after_job_ids": [],
        "active_after_count": 0,
        "original_e112_retired": True,
        "original_artifacts_preserved": True,
        "pool_with_replacement": False,
        "efficacy_outcomes_inspected": False,
    }
    if len(exact_ids) != 75 or len(set(exact_ids)) != 75:
        return None
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    return payload


def e113_r3_release(path: Path = E113_R3_LEDGER) -> dict[str, object] | None:
    """Return a valid full successor that retires failed E113 placeholders."""

    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != "e113r3_dapo_full_relaunch_jobs_v1":
        return None
    if payload.get("released") is not True:
        return None
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != 50:
        return None
    expected = {
        (family, domain, int(seed))
        for family in ("qwen05b", "falcon1b")
        for domain in (
            "graph_coloring",
            "countdown",
            "python_factors",
            "mathir",
            "pantry_plan",
        )
        for seed in (
            (43, 44, 45, 46, 47) if family == "qwen05b" else (55, 56, 57, 58, 59)
        )
    }
    observed = {
        (
            str(run.get("model_family")),
            str(run.get("domain")),
            int(run.get("seed", -1)),
        )
        for run in runs
    }
    return payload if observed == expected else None


def e113_r3_retirement() -> dict[str, object] | None:
    if not E113_R3_RETIREMENT.is_file():
        return None
    try:
        payload = json.loads(E113_R3_RETIREMENT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    valid = (
        payload.get("schema") == "e113r3_retirement_for_official_dapo_v1"
        and payload.get("registered_job_ids") == list(range(30790925, 30790975))
        and payload.get("all_registered_jobs_canceled") is True
        and payload.get("exact_jobs_remaining_in_queue") == []
        and payload.get("r3_eligible_for_named_dapo_efficacy") is False
        and payload.get("failure_reaper_job_id") == 30791185
    )
    return payload if valid else None


def e111_continuation_jobs(ledger_path: Path) -> dict[int, int]:
    """Resolve scheduler-only continuations while preserving 15 science cells."""

    if (
        ledger_path.resolve() != E111_LEDGER.resolve()
        or not E111_CONTINUATIONS.is_file()
    ):
        return {}
    try:
        payload = json.loads(E111_CONTINUATIONS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    expected = {
        "schema": "e111_qwen3_python_mathir_replacement_jobs_v1",
        "ledger": str(E111_LEDGER),
        "ledger_sha256": hashlib.sha256(E111_LEDGER.read_bytes()).hexdigest(),
        "same_scientific_cells": True,
        "same_run_directories": True,
        "released": True,
        "installed": True,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return {}
    rows = payload.get("continuations")
    if not isinstance(rows, list) or len(rows) != 2:
        return {}
    try:
        mapping = {
            int(row["original_job_id"]): int(row["continuation_job_id"]) for row in rows
        }
    except (KeyError, TypeError, ValueError):
        return {}
    if len(mapping) != 2 or len(set(mapping.values())) != 2:
        return {}
    if not E111_SECOND_CONTINUATION.is_file():
        return mapping
    try:
        second = json.loads(E111_SECOND_CONTINUATION.read_text(encoding="utf-8"))
        ledger = json.loads(E111_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return mapping
    original_id = 30674760
    original = next(
        (
            run
            for run in ledger.get("runs", [])
            if int(run.get("job_id", -1)) == original_id
        ),
        None,
    )
    expected_second = {
        "schema": "e111_qwen3_python_second_continuation_jobs_v1",
        "ledger": str(E111_LEDGER),
        "ledger_sha256": hashlib.sha256(E111_LEDGER.read_bytes()).hexdigest(),
        "first_continuation_record": str(E111_CONTINUATIONS),
        "first_continuation_record_sha256": hashlib.sha256(
            E111_CONTINUATIONS.read_bytes()
        ).hexdigest(),
        "original_job_id": original_id,
        "prior_continuation_job_id": mapping.get(original_id),
        "domain": "python_factors",
        "scale": "qwen3b",
        "seed": 70,
        "run_dir": original.get("run_dir") if original else None,
        "run_stamp": original.get("run_stamp") if original else None,
        "same_scientific_cell": True,
        "same_run_directory": True,
        "state_reset": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "released": True,
        "installed": True,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
    }
    if any(second.get(key) != value for key, value in expected_second.items()):
        return mapping
    try:
        second_id = int(second["continuation_job_id"])
    except (KeyError, TypeError, ValueError):
        return mapping
    if second_id in mapping or second_id in mapping.values():
        return mapping
    mapping[original_id] = second_id
    pantry_original_id = 30674762
    original = next(
        (
            run
            for run in ledger.get("runs", [])
            if int(run.get("job_id", -1)) == pantry_original_id
        ),
        None,
    )
    if not E111_PANTRY_CONTINUATION.is_file():
        return mapping
    try:
        pantry = json.loads(E111_PANTRY_CONTINUATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return mapping
    expected_pantry = {
        "schema": "e111_qwen3_pantry_continuation_jobs_v1",
        "ledger": str(E111_LEDGER),
        "ledger_sha256": hashlib.sha256(E111_LEDGER.read_bytes()).hexdigest(),
        "original_job_id": pantry_original_id,
        "prior_job_id": pantry_original_id,
        "domain": "pantry_plan",
        "scale": "qwen3b",
        "seed": 70,
        "run_dir": original.get("run_dir") if original else None,
        "run_stamp": original.get("run_stamp") if original else None,
        "same_scientific_cell": True,
        "same_run_directory": True,
        "state_reset": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "released": True,
        "installed": True,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
    }
    if any(pantry.get(key) != value for key, value in expected_pantry.items()):
        return mapping
    try:
        pantry_id = int(pantry["continuation_job_id"])
    except (KeyError, TypeError, ValueError):
        return mapping
    if pantry_id in mapping or pantry_id in mapping.values():
        return mapping
    mapping[pantry_original_id] = pantry_id
    return mapping


def e109_continuation_jobs(ledger_path: Path) -> dict[int, int]:
    """Resolve the two same-directory E109 Qwen-3B continuations."""

    if (
        ledger_path.resolve() != E109_LEDGER.resolve()
        or not E109_CONTINUATIONS.is_file()
    ):
        return {}
    try:
        payload = json.loads(E109_CONTINUATIONS.read_text(encoding="utf-8"))
        ledger = json.loads(E109_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    expected = {
        "schema": "e109r1_qwen3_python_continuation_jobs_v1",
        "original_ledger": str(E109_LEDGER),
        "original_ledger_sha256": hashlib.sha256(E109_LEDGER.read_bytes()).hexdigest(),
        "exact_original_job_ids": [30659554, 30659555],
        "exact_seeds": [73, 74],
        "same_scientific_cells": True,
        "same_run_directories": True,
        "same_a6000_hardware_class": True,
        "scientific_environment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "released": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return {}
    originals = {int(run["job_id"]): run for run in ledger.get("runs", [])}
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2:
        return {}
    mapping: dict[int, int] = {}
    try:
        for record in records:
            original_id = int(record["original_job_id"])
            continuation_id = int(record["continuation_job_id"])
            original = originals[original_id]
            if (
                int(record["seed"]) != int(original["seed"])
                or record["run_dir"] != original["run_dir"]
                or record["run_stamp"] != original["run_stamp"]
            ):
                return {}
            mapping[original_id] = continuation_id
    except (KeyError, TypeError, ValueError):
        return {}
    if set(mapping) != {30659554, 30659555} or len(set(mapping.values())) != 2:
        return {}
    return mapping


def e117_stage1_continuation_jobs(ledger_path: Path) -> dict[int, int]:
    """Resolve exact-resume scheduler attempts for E117 Stage 1 cells."""

    if (
        ledger_path.resolve() != E117_STAGE1_LEDGER.resolve()
        or not E117_STAGE1_CONTINUATIONS.is_file()
    ):
        return {}
    try:
        payload = json.loads(E117_STAGE1_CONTINUATIONS.read_text(encoding="utf-8"))
        ledger = json.loads(E117_STAGE1_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    expected = {
        "schema": "e117_stage1_continuation_jobs_v1",
        "original_ledger": str(E117_STAGE1_LEDGER),
        "original_ledger_sha256": hashlib.sha256(
            E117_STAGE1_LEDGER.read_bytes()
        ).hexdigest(),
        "same_scientific_cells": True,
        "same_run_directories": True,
        "state_reset": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "released": True,
        "installed": True,
        "outcomes_inspected": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return {}
    records = payload.get("continuations")
    if not isinstance(records, list) or len(records) != 3:
        return {}
    originals = {int(run["job_id"]): run for run in ledger.get("runs", [])}
    mapping: dict[int, int] = {}
    try:
        for record in records:
            original_id = int(record["original_job_id"])
            continuation_id = int(record["continuation_job_id"])
            original = originals[original_id]
            if any(
                record.get(key) != original.get(key)
                for key in ("domain", "arm", "seed", "run_dir", "run_stamp")
            ):
                return {}
            mapping[original_id] = continuation_id
    except (KeyError, TypeError, ValueError):
        return {}
    expected_originals = {30980499, 30980502, 30980505}
    if set(mapping) != expected_originals or len(set(mapping.values())) != 3:
        return {}
    return mapping



def e119_continuation_jobs(ledger_path: Path) -> dict[int, int]:
    """Resolve audited scheduler-only continuations for E119 cells."""
    if ledger_path.resolve() != E119_LEDGER.resolve() or not E119_CONTINUATIONS.is_file():
        return {}
    try:
        payload = json.loads(E119_CONTINUATIONS.read_text(encoding="utf-8"))
        ledger = json.loads(E119_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    expected = {
        "schema": "e119_level2_continuation_jobs_v1",
        "original_ledger": str(E119_LEDGER),
        "original_ledger_sha256": hashlib.sha256(E119_LEDGER.read_bytes()).hexdigest(),
        "same_scientific_cells": True,
        "same_run_directories": True,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "released": True,
        "installed": True,
        "outcomes_inspected": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return {}
    records = payload.get("continuations")
    if not isinstance(records, list) or not records:
        return {}
    originals = {int(run["job_id"]): run for run in ledger.get("runs", [])}
    mapping: dict[int, int] = {}
    try:
        for record in records:
            original_id = int(record["original_job_id"])
            continuation_id = int(record["continuation_job_id"])
            original = originals[original_id]
            if any(record.get(key) != original.get(key) for key in ("domain", "arm", "seed", "run_dir", "run_stamp")):
                return {}
            mapping[original_id] = continuation_id
    except (KeyError, TypeError, ValueError):
        return {}
    if len(mapping) != len(records) or len(set(mapping.values())) != len(records):
        return {}
    return mapping


def e120_continuation_jobs(ledger_path: Path) -> dict[int, int]:
    """Resolve E120 continuations with validated resource-only amendments."""

    if (
        ledger_path.resolve() != E120_LEDGER.resolve()
        or not E120_CONTINUATIONS.is_file()
    ):
        return {}
    try:
        payload = json.loads(E120_CONTINUATIONS.read_text(encoding="utf-8"))
        ledger = json.loads(E120_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    expected = {
        "schema": "e120r1_scheduler_continuation_jobs_v1",
        "original_ledger": str(E120_LEDGER),
        "original_ledger_sha256": hashlib.sha256(
            E120_LEDGER.read_bytes()
        ).hexdigest(),
        "same_scientific_cells": True,
        "same_run_directories": True,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "pvl_excluded": True,
        "released": True,
        "installed": True,
        "outcomes_inspected": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return {}
    records = payload.get("continuations")
    if not isinstance(records, list) or not records:
        return {}
    originals = {int(run["job_id"]): run for run in ledger.get("runs", [])}
    runtime_only = (
        payload.get("scheduler_only") is False
        and payload.get("runtime_allocation_only") is True
    )
    if payload.get("scheduler_only") is not True and not runtime_only:
        return {}
    allowed_runtime_change = {
        "OAT_ZERO_VLLM_GPU_RATIO": {"before": "0.25", "after": "0.40"}
    }
    runtime_records = 0
    mapping: dict[int, int] = {}
    try:
        for record in records:
            changes = record.get("runtime_changes")
            if changes:
                if (
                    not runtime_only
                    or changes != allowed_runtime_change
                    or record.get("optimizer_update_changed") is not False
                    or record.get("treatment_changed") is not False
                ):
                    return {}
                runtime_records += 1
            original_id = int(record["original_job_id"])
            continuation_id = int(record["continuation_job_id"])
            original = originals[original_id]
            if any(
                record.get(key) != original.get(key)
                for key in ("domain", "model_key", "seed", "run_dir", "run_stamp")
            ):
                return {}
            mapping[original_id] = continuation_id
    except (KeyError, TypeError, ValueError):
        return {}
    if runtime_only and not runtime_records:
        return {}
    if len(mapping) != len(records) or len(set(mapping.values())) != len(records):
        return {}
    return mapping


def e122_migrated_campaign(
    ledger_path: Path, ledger: dict[str, object]
) -> dict[str, object] | None:
    """Follow committed Python replacements without changing the 100-cell ledger.

    The neutral migration and subsequent CLI repair use fresh output namespaces.
    Reading only their new scheduler IDs would still lose realized progress; both
    the current cell and its complete scheduler lineage must follow the commit.
    """

    if (
        ledger_path.resolve() != E122_LEDGER.resolve()
        or ledger.get("schema") != "e122_level3_factorial_jobs_v1"
    ):
        return None
    runs = [dict(run) for run in ledger["runs"]]
    binding: dict[str, str] = {}
    result = None
    fields = ("domain", "dataset_domain", "arm", "seed", "run_stamp", "run_dir")
    try:
        for prefix, directory, plan_name, plan_sha, commit_sha in E122_MIGRATIONS:
            commit_path = directory / "committed.json"
            if not commit_path.is_file():
                break
            plan_bytes = (directory / plan_name).read_bytes()
            commit_bytes = commit_path.read_bytes()
            if (
                hashlib.sha256(plan_bytes).hexdigest() != plan_sha
                or hashlib.sha256(commit_bytes).hexdigest() != commit_sha
            ):
                return None
            plan = json.loads(plan_bytes)
            committed = json.loads(commit_bytes)
            if committed["plan_sha256"] != plan_sha:
                return None
            if not binding:
                original_binding_path = directory / "e122_original_binding.json"
                original_binding_bytes = original_binding_path.read_bytes()
                if hashlib.sha256(original_binding_bytes).hexdigest() != plan["files_sha256"][str(original_binding_path)]:
                    return None
                original_binding = json.loads(original_binding_bytes)
                if (
                    original_binding["held_ledger_path"] != str(ledger_path.resolve())
                    or original_binding["held_ledger_sha256"] != hashlib.sha256(ledger_path.read_bytes()).hexdigest()
                ):
                    return None
            elif plan["parent_migration_sha256"] != binding["migration_plan_sha256"]:
                return None
            replacements = [
                row for row in committed["replacements"] if row["campaign"] == "e122"
            ]
            by_id = {int(run["job_id"]): run for run in runs}
            old_ids = {int(row["old_job_id"]) for row in replacements}
            new_ids = {int(row["job_id"]) for row in replacements}
            python_ids = {int(run["job_id"]) for run in runs if run["domain"] == "python_factors"}
            if (
                len(runs) != 100 or len(by_id) != 100
                or len(replacements) != 20 or old_ids != python_ids
                or len(new_ids) != 20 or new_ids.intersection(by_id)
            ):
                return None
            for replacement in replacements:
                old_id = int(replacement["old_job_id"])
                run = by_id[old_id]
                old_cell, cell = replacement["old_cell"], replacement["cell"]
                if (
                    any(run[key] != old_cell[key] for key in fields)
                    or any(cell[key] != old_cell[key] for key in fields[:4])
                    or int(cell["target_steps"]) != int(ledger["target_steps"])
                    or cell["run_dir"] == old_cell["run_dir"]
                ):
                    return None
                original_id = int(run.get("original_job_id", old_id))
                previous_ids = [*run.get("previous_job_ids", []), old_id]
                run.update(cell)
                run.update(
                    job_id=int(replacement["job_id"]),
                    original_job_id=original_id,
                    previous_job_ids=previous_ids,
                )
                # A predecessor console log cannot provide successor progress.
                run.pop("log_path", None)
            binding.update({f"{prefix}_plan_sha256": plan_sha, f"{prefix}_commit_sha256": commit_sha})
            result = {
                "runs": runs,
                "binding": dict(binding),
                "status_dir": directory / "e122_release_controller/status",
            }
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return result


def load_static_snapshot(ledger_path: Path) -> dict[str, object]:
    """Use registered run paths with any validated scheduler continuation IDs."""

    continuations = e109_continuation_jobs(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    migrated = e122_migrated_campaign(ledger_path, ledger)
    if not continuations and migrated is None:
        return shared.load_snapshot(ledger_path)
    target = int(ledger["target_steps"])
    runs = migrated["runs"] if migrated is not None else ledger["runs"]
    effective_ids = [
        continuations.get(int(run["job_id"]), int(run["job_id"])) for run in runs
    ]
    states = shared.scheduler_states(effective_ids)
    rows = []
    for run, job_id in zip(runs, effective_ids):
        run_dir = Path(str(run["run_dir"]))
        step = min(
            max(shared.run_step(run_dir), shared.receipt_step(run_dir)),
            target,
        )
        checkpoint = min(shared.checkpoint_step(run_dir), target)
        state = states.get(job_id, "NOT_IN_QUEUE")
        if shared.is_complete(run_dir, step, target):
            state = "COMPLETED"
        rows.append(
            {
                "arm": run["arm"],
                "checkpoint": checkpoint,
                "domain": run["domain"],
                "original_job_id": int(run.get("original_job_id", run["job_id"])),
                "previous_job_ids": list(run.get("previous_job_ids", [])),
                "run_dir": str(run_dir),
                "job_id": job_id,
                "effective_job_id": job_id,
                "seed": int(run["seed"]),
                "state": state,
                "step": step,
            }
        )
    return {
        "arms": ledger.get("arms") or sorted({str(run["arm"]) for run in runs}),
        "checkpoint_interval": int(ledger["checkpoint_interval_steps"]),
        "domains": ledger.get("domains")
        or sorted({str(run["domain"]) for run in runs}),
        "passes": int(ledger["passes"]),
        "rows": rows,
        "steps_per_pass": int(ledger["train_rows"]),
        "target": target,
    }


def load_smoke_snapshot(
    ledger_path: Path,
    *,
    excluded_domains: tuple[str, ...] = (),
) -> dict[str, object]:
    """Read minimal mechanism-gate ledgers without mutating their schemas."""

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    target = int(ledger["target_steps"])
    passes = int(ledger.get("passes") or 1)
    if target % passes:
        raise ValueError(
            f"smoke ledger target_steps={target} is not divisible by passes={passes}"
        )
    migrated = e122_migrated_campaign(ledger_path, ledger)
    runs = [
        run
        for run in (migrated["runs"] if migrated is not None else ledger.get("runs", []))
        if str(run.get("domain")) not in set(excluded_domains)
    ]
    continuations = e111_continuation_jobs(ledger_path)
    continuations.update(e117_stage1_continuation_jobs(ledger_path))
    continuations.update(e119_continuation_jobs(ledger_path))
    continuations.update(e120_continuation_jobs(ledger_path))
    effective_ids = [
        continuations.get(int(run["job_id"]), int(run["job_id"])) for run in runs
    ]
    states = shared.scheduler_states(effective_ids)
    rows: list[dict[str, object]] = []
    for run in runs:
        registered_job_id = int(run["job_id"])
        original_job_id = int(run.get("original_job_id", registered_job_id))
        job_id = continuations.get(registered_job_id, registered_job_id)
        run_dir = Path(str(run["run_dir"]))
        step = min(shared.run_step(run_dir), target)
        state = states.get(job_id, "NOT_IN_QUEUE")
        if shared.is_complete(run_dir, step, target):
            state = "COMPLETED"
            step = target
        rows.append(
            {
                "arm": str(
                    run.get("arm", ledger.get("key_weighting", "unknown"))
                ),
                "domain": str(run["domain"]),
                "original_job_id": original_job_id,
                "previous_job_ids": list(run.get("previous_job_ids", [])),
                "run_dir": str(run_dir),
                "job_id": job_id,
                "effective_job_id": job_id,
                "seed": int(run["seed"]),
                "state": state,
                "step": step,
            }
        )
    return {
        "rows": rows,
        "target": target,
        # Mechanism gates usually use one pass, but E108 deliberately cycles
        # eight training rows for eight passes. Treating the full 64-update
        # target as one pass made a completed E108 cohort print ``1.00/8p``.
        "steps_per_pass": target // passes,
        "passes": passes,
    }


def load_launch_gates(ledger_path: Path) -> dict[str, int] | None:
    """Summarize optional smoke gates without counting them as science cells."""

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    raw_gates = ledger.get("smokes")
    if not isinstance(raw_gates, dict) or not raw_gates:
        return None
    gates = [gate for gate in raw_gates.values() if isinstance(gate, dict)]
    states = shared.scheduler_states([int(gate["job_id"]) for gate in gates])
    rows: list[dict[str, object]] = []
    for gate in gates:
        job_id = int(gate["job_id"])
        target = int(gate["max_train"])
        run_dir = Path(str(gate["run_dir"]))
        log_step = (
            shared.verl_console_step(Path(str(gate["log_path"])))
            if gate.get("log_path")
            else 0
        )
        step = min(
            max(shared.run_step(run_dir), shared.receipt_step(run_dir), log_step),
            target,
        )
        state = states.get(job_id, "NOT_IN_QUEUE")
        if shared.is_complete(run_dir, step, target):
            state = "COMPLETED"
            step = target
        rows.append({"state": state, "step": step, "target": target})
    counts = shared.state_counts(rows)
    return {
        "total": len(rows),
        "terminal": sum(int(row["step"]) >= int(row["target"]) for row in rows),
        "running": sum(counts[state] for state in SCIENCE_RUNNING_STATES),
        "pending": counts["PENDING"],
        "failed": sum(counts[state] for state in SCIENCE_FAILURE_STATES),
    }


def progress_summary(
    rows: list[dict[str, object]],
    *,
    target: int,
    steps_per_pass: int,
    passes: int,
) -> dict[str, object]:
    counts = shared.state_counts(rows)
    realized = sum(int(row["step"]) for row in rows)
    return {
        "cells": len(rows),
        "terminal": sum(int(row["step"]) >= target for row in rows),
        "running": sum(counts[state] for state in SCIENCE_RUNNING_STATES),
        "pending": counts["PENDING"],
        "failed": sum(counts[state] for state in SCIENCE_FAILURE_STATES),
        "realized": realized,
        "total": len(rows) * target,
        "depth": realized / (len(rows) * steps_per_pass) if rows else 0.0,
        "passes": passes,
    }


def registered_scale(run: dict[str, object]) -> str | None:
    for key in ("scale", "model_family", "family", "model_key"):
        value = run.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def display_scale(
    payload: dict[str, object], run: dict[str, object]
) -> str:
    scale = registered_scale(run) or registered_scale(payload)
    if scale is not None:
        return scale
    model = str(payload.get("model", "")).lower()
    if "0.5b" in model or "0p5b" in model:
        return "qwen05b"
    if "3b" in model:
        return "qwen3b"
    if "falcon" in model:
        return "falcon1b"
    return "-"


def scheduler_details(job_ids: list[int]) -> dict[int, dict[str, str]]:
    """Return a live scheduler record for each requested job still in squeue."""

    if not job_ids:
        return {}
    result = subprocess.run(
        [
            "squeue",
            "-h",
            "-j",
            ",".join(str(job_id) for job_id in sorted(set(job_ids))),
            "-o",
            "%i|%T|%M|%P|%R|%N|%j",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return {}
    details: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|", 6)
        if len(parts) != 7 or not parts[0].isdigit():
            continue
        job_id = int(parts[0])
        details[job_id] = {
            "state": shared.normalize_state(parts[1]),
            "elapsed": parts[2],
            "partition": parts[3],
            "reason": parts[4],
            "node": parts[5] or "-",
            "job_name": parts[6],
        }
    return details


def enrich_live_jobs(rows: list[dict[str, object]]) -> None:
    """Attach node and elapsed-time data to every registered running cell."""

    jobs = [
        job
        for row in rows
        if row.get("retired") is not True
        for job in row.get("active_jobs", [])
        if isinstance(job, dict)
    ]
    details = scheduler_details([int(job["job_id"]) for job in jobs])
    for job in jobs:
        job.update(details.get(int(job["job_id"]), {}))


def latest_recovery_rows() -> list[dict[str, object]]:
    """Show current successors of audited recoveries without counting new cells."""

    if not LATEST_RECOVERY.is_file():
        return []
    try:
        payload = json.loads(LATEST_RECOVERY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if (
        payload.get("schema")
        != "e118-e119-overnight-failure-recovery-v1"
        or payload.get("released") is not True
    ):
        return []
    records = payload.get("replacements")
    if not isinstance(records, list) or len(records) != 6:
        return []
    rows: list[dict[str, object]] = []
    try:
        ledgers = {
            "e118": json.loads(E118_LEDGER.read_text(encoding="utf-8")),
            "e119": json.loads(E119_LEDGER.read_text(encoding="utf-8")),
        }
        continuations = e119_continuation_jobs(E119_LEDGER)
        for record in records:
            ledger = ledgers[str(record["cohort"])]
            matches = [
                run for run in ledger["runs"]
                if all(
                    run.get(key) == record.get(key)
                    for key in ("run_dir", "domain", "arm", "seed")
                )
                and display_scale(ledger, run) == record["scale"]
            ]
            if len(matches) != 1:
                return []
            run = matches[0]
            registered_id = int(run["job_id"])
            lineage = {registered_id, *map(int, run.get("previous_job_ids", []))}
            if int(record["old_job_id"]) not in lineage:
                return []
            job_id = (
                continuations.get(registered_id, registered_id)
                if record["cohort"] == "e119" else registered_id
            )
            checkpoint_name = Path(str(record["resume_checkpoint"])).name
            checkpoint = int(checkpoint_name.removeprefix("step_"))
            rows.append(
                {
                    **record,
                    "job_id": job_id,
                    "checkpoint": checkpoint,
                    "target": int(ledger["target_steps"]),
                }
            )
    except (OSError, KeyError, TypeError, ValueError):
        return []
    job_ids = [int(row["job_id"]) for row in rows]
    details = scheduler_details(job_ids)
    states = shared.scheduler_states(job_ids)
    for row in rows:
        job_id = int(row["job_id"])
        row.update({"state": states.get(job_id, "NOT_IN_QUEUE"), **details.get(job_id, {})})
        run_dir = Path(str(row["run_dir"]))
        step = max(shared.run_step(run_dir), shared.receipt_step(run_dir))
        if shared.is_complete(run_dir, step, int(row["target"])):
            row["state"] = "COMPLETED"
    return rows



def level3_release_status(
    ledger_path: Path, payload: dict[str, object]
) -> dict[str, object] | None:
    """Read a controller snapshot bound to this exact E122 or E123 cohort."""

    sources = {
        "e122_level3_factorial_jobs_v1": (E122_RELEASE_STATUS_DIR, "e122_level3_release_status_v1"),
        "e123_level3_factorial_jobs_v1": (E123_RELEASE_STATUS_DIR, "e123_level3_release_status_v1"),
    }
    source = sources.get(payload.get("schema"))
    if source is None or not payload.get("runs"):
        return None
    status_dir, status_schema = source
    migrated = e122_migrated_campaign(ledger_path, payload)
    if migrated is not None:
        status_dir = migrated["status_dir"]
    paths = sorted(status_dir.glob("*.json"), reverse=True)
    if not paths:
        return None
    expected = {
        "held_ledger_path": str(ledger_path.resolve()),
        "held_ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "plan_path": payload.get("plan_path"),
        "plan_sha256": payload.get("plan_sha256"),
        "model_choice": payload.get("model_choice"),
    }
    if migrated is not None:
        expected.update(migrated["binding"])
    if any(value is None for value in expected.values()):
        return None
    count_fields = (
        "staged_held", "released_nonterminal", "reserved_unfinished_slots",
        "running", "successful_endpoints", "unknown", "max_released_nonterminal",
    )
    for path in paths:
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if snapshot.get("schema") != status_schema:
                continue
            binding = snapshot["binding"]
            if any(binding.get(key) != value for key, value in expected.items()):
                continue
            if not isinstance(snapshot["observed_at"], str):
                continue
            if any(
                type(snapshot[key]) is not int or not 0 <= snapshot[key] <= len(payload["runs"])
                for key in count_fields
            ):
                continue
            if snapshot["blocked_reason"] is not None and not isinstance(snapshot["blocked_reason"], str):
                continue
            storage = snapshot["storage"]
            if any(type(storage[key]) is not int for key in ("free_bytes", "remaining_headroom_bytes")):
                continue
            return {
                **{key: snapshot[key] for key in count_fields},
                "observed_at": snapshot["observed_at"],
                "blocked_reason": snapshot["blocked_reason"],
                "free_bytes": storage["free_bytes"],
                "remaining_headroom_bytes": storage["remaining_headroom_bytes"],
                "source": str(path),
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return None


def e122_release_status(
    ledger_path: Path, payload: dict[str, object]
) -> dict[str, object] | None:
    """Keep the existing E122-only reader available to operational callers."""

    if payload.get("schema") != "e122_level3_factorial_jobs_v1":
        return None
    return level3_release_status(ledger_path, payload)


def release_status_note(row: dict[str, object]) -> str:
    snapshot = row["release_status"]
    reason = (snapshot["blocked_reason"] or "release eligible").replace("_", " ")
    return (
        f"{str(row['label']).split()[0]} release snapshot ({snapshot['observed_at']}): "
        f"{snapshot['staged_held']} staged/held; "
        f"{snapshot['released_nonterminal']} released and nonterminal "
        f"({snapshot['running']} running); "
        f"{snapshot['reserved_unfinished_slots']}/{snapshot['max_released_nonterminal']} "
        f"reserved slots; {snapshot['successful_endpoints']} successful endpoints; "
        f"{snapshot['unknown']} unknown; {reason}; "
        f"disk free {snapshot['free_bytes'] / 1024**3:.1f} GiB, "
        f"remaining headroom {snapshot['remaining_headroom_bytes'] / 1024**3:.1f} GiB"
    )


def planned_cohort_row(label: str, payload: dict[str, object]) -> dict[str, object]:
    """Expose a prospective science budget without inventing scheduler jobs."""

    if payload.get("released") is not False or payload.get("runs") != []:
        raise ValueError("planned campaigns must be unreleased with no submitted runs")
    planned = payload["planned_runs"]
    if not isinstance(planned, list) or not planned:
        raise ValueError("planned campaigns require a nonempty cell matrix")
    if any(not isinstance(run, dict) or "job_id" in run for run in planned):
        raise ValueError("planned cells cannot contain scheduler job IDs")
    identities = {
        (str(run["domain"]), str(run["arm"]), int(run["seed"]))
        for run in planned
    }
    expected = {
        (str(domain), str(arm), int(seed))
        for domain in payload["domains"]
        for arm in payload["arms"]
        for seed in payload["seeds"]
    }
    if len(identities) != len(planned) or identities != expected:
        raise ValueError("planned cells must match the complete declared factorial")
    target = int(payload["target_steps"])
    passes = int(payload["passes"])
    if target <= 0 or passes <= 0 or target % passes:
        raise ValueError("planned campaign target must contain positive whole passes")
    admission = payload.get("admission", {})
    reason = str(admission.get("reason", "Awaiting campaign admission"))
    return {
        "label": label,
        **progress_summary(
            [{"state": "AWAITING_ADMISSION", "step": 0} for _ in planned],
            target=target,
            steps_per_pass=target // passes,
            passes=passes,
        ),
        "status": "AWAITING_ADMISSION",
        "planned_cells": len(planned),
        "admission_reason": reason,
    }


def cohort_row(
    label: str,
    ledger: Path,
    reader: str,
    excluded_domains: tuple[str, ...] = (),
) -> dict[str, object] | None:
    if not ledger.is_file():
        return None
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    if "planned_runs" in payload and not payload.get("runs"):
        return planned_cohort_row(label, payload)
    r4_recovery = payload.get("vllm_scheduler_recovery")
    if (
        isinstance(r4_recovery, dict)
        and r4_recovery.get("science_replacements_submitted") is False
    ):
        # R4-R1's 50 jobs were canceled by the failed operational gate before
        # allocation. During the preregistered two-stage R4-R2 recovery, those
        # immutable records are provenance rather than 50 DAPO efficacy
        # failures. Keep the frozen denominator visible as awaiting its smoke
        # gate; load_launch_gates below reports the two live smokes separately.
        frozen_runs = r4_recovery.get("superseded_runs_pending_replacement", [])
        target = int(payload["target_steps"])
        passes = int(payload.get("passes") or 1)
        snapshot = {
            "rows": [{"state": "PENDING", "step": 0} for _ in frozen_runs],
            "target": target,
            "steps_per_pass": target // passes,
            "passes": passes,
        }
    elif payload.get("runs") == [] and isinstance(payload.get("smokes"), dict):
        # A smoke-only operational recovery has no scientific run matrix and
        # therefore no train_rows field for the static reader. Keep a zero-cell
        # row so its gate remains visible without adding it to science totals.
        target = int(payload.get("target_steps", 0))
        passes = int(payload.get("passes", 1))
        snapshot = {
            "rows": [],
            "target": target,
            "steps_per_pass": target // passes if target else 1,
            "passes": passes,
        }
    elif reader == "point":
        snapshot = shared.load_point_snapshot(ledger)
    elif reader == "static":
        snapshot = load_static_snapshot(ledger)
    elif reader == "smoke":
        snapshot = load_smoke_snapshot(
            ledger,
            excluded_domains=excluded_domains,
        )
    else:
        raise ValueError(f"unknown campaign progress reader: {reader!r}")
    rows = snapshot["rows"]
    target = int(snapshot["target"])
    steps_per_pass = int(snapshot["steps_per_pass"])
    passes = int(snapshot["passes"])
    scale_by_job_id = {
        int(run["job_id"]): scale
        for run in payload.get("runs", [])
        if isinstance(run, dict)
        and "job_id" in run
        and (scale := registered_scale(run)) is not None
    }
    by_scale: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        raw_job_id = row.get("original_job_id", row.get("job_id"))
        try:
            scale = scale_by_job_id.get(int(raw_job_id))
        except (TypeError, ValueError):
            scale = None
        if scale is not None:
            by_scale.setdefault(scale, []).append(row)
    result: dict[str, object] = {
        "label": label,
        **progress_summary(
            rows,
            target=target,
            steps_per_pass=steps_per_pass,
            passes=passes,
        ),
    }
    run_by_id = {
        int(run["job_id"]): run
        for run in payload.get("runs", [])
        if isinstance(run, dict) and "job_id" in run
    }
    active_jobs: list[dict[str, object]] = []
    for row in rows:
        if (
            str(row.get("state")) not in SCIENCE_RUNNING_STATES
            or row.get("job_id") is None
        ):
            continue
        original_job_id = int(row.get("original_job_id", row["job_id"]))
        effective_job_id = int(row.get("effective_job_id", row["job_id"]))
        run = run_by_id.get(original_job_id, {})
        previous_job_ids = [
            int(job_id) for job_id in row.get("previous_job_ids", run.get("previous_job_ids", []))
        ]
        if effective_job_id != original_job_id and original_job_id not in previous_job_ids:
            previous_job_ids.append(original_job_id)
        active_jobs.append(
            {
                "cohort": label.split(maxsplit=1)[0],
                "cohort_label": label,
                "job_id": effective_job_id,
                "original_job_id": original_job_id,
                "previous_job_ids": list(dict.fromkeys(previous_job_ids)),
                "state": str(row["state"]),
                "scale": display_scale(payload, run),
                "domain": str(row.get("domain", run.get("domain", "-"))),
                "arm": str(row.get("arm", run.get("arm", "-"))),
                "seed": row.get("seed", run.get("seed", "-")),
                "step": int(row["step"]),
                "target": target,
            }
        )
    if active_jobs:
        result["active_jobs"] = active_jobs

    if len(by_scale) > 1:
        result["scale_breakdown"] = {
            scale: progress_summary(
                scale_rows,
                target=target,
                steps_per_pass=steps_per_pass,
                passes=passes,
            )
            for scale, scale_rows in sorted(by_scale.items())
        }
    release_snapshot = level3_release_status(ledger, payload)
    if release_snapshot is not None:
        result["release_status"] = release_snapshot
    gates = load_launch_gates(ledger)
    if gates is not None:
        result["launch_gates"] = gates
    return result


def render(
    rows: list[dict[str, object]],
    *,
    markdown: bool,
    held: tuple[str, ...] = (),
    include_history: bool = False,
    recovery_rows: tuple[dict[str, object], ...] = (),
) -> str:
    active_rows = [
        row
        for row in rows
        if row.get("retired") is not True and (include_history or int(row["cells"]) > 0)
    ]
    retired_rows = (
        [row for row in rows if row.get("retired") is True] if include_history else []
    )
    planned_rows = [row for row in active_rows if int(row.get("planned_cells", 0))]
    release_notes = [release_status_note(row) for row in active_rows if "release_status" in row]
    cells = sum(int(r["cells"]) for r in active_rows)
    terminal = sum(int(r["terminal"]) for r in active_rows)
    running = sum(int(r["running"]) for r in active_rows)
    pending = sum(int(r["pending"]) for r in active_rows)
    failed = sum(int(r.get("failed", 0)) for r in active_rows)
    realized = sum(int(r["realized"]) for r in active_rows)
    total = sum(int(r["total"]) for r in active_rows)
    # Keep incomplete or failed release gates visible so a blocked science row
    # is explained.  Once every gate is terminal and successful, the science
    # row itself is the useful live status and the operational gate is quiet.
    launch_gates = [
        (str(row["label"]), row["launch_gates"])
        for row in active_rows
        if "launch_gates" in row
        and int(row["launch_gates"]["terminal"])
        < int(row["launch_gates"]["total"])
    ]
    # These stopped cohorts retain their registered cells and audit history.
    # Their scale details are quiet by default, unless training resumes.
    stopped_scale_labels = {
        registry.by_tag("e117s1").label,
        registry.by_tag("e113r4").label,
    }
    incomplete_scale_breakdowns = []
    for row in active_rows:
        if "scale_breakdown" not in row or int(row["terminal"]) >= int(row["cells"]):
            continue
        if (
            not include_history
            and row["label"] in stopped_scale_labels
            and int(row["running"]) == 0
        ):
            continue
        breakdown = {
            scale: summary
            for scale, summary in row["scale_breakdown"].items()
            if include_history or int(summary["terminal"]) < int(summary["cells"])
        }
        if breakdown:
            incomplete_scale_breakdowns.append((str(row["label"]), breakdown))
    recovery_rows = tuple(
        row for row in recovery_rows
        if include_history or str(row["state"]).upper() != "COMPLETED"
    )

    live_jobs = sorted(
        (
            job
            for row in active_rows
            for job in row.get("active_jobs", [])
            if isinstance(job, dict)
        ),
        key=lambda job: (
            str(job.get("cohort")),
            str(job.get("node", "-")),
            int(job["job_id"]),
        ),
    )

    if markdown:
        out = [
            "| cohort | cells | terminal | running | pending | failed | steps | depth |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in active_rows:
            out.append(
                f"| {r['label']} | {r['cells']} | {r['terminal']} | {r['running']} "
                f"| {r['pending']} | {r.get('failed', 0)} "
                f"| {r['realized']:,}/{r['total']:,} "
                f"| {r['depth']:.2f}p |"
            )
        out.append(
            f"| **total** | **{cells}** | **{terminal}** | {running} | {pending} "
            f"| {failed} "
            f"| {realized:,}/{total:,} | "
            f"**{100 * realized / total if total else 0.0:.1f}%** |"
        )
        if release_notes:
            out.extend(["", *[f"- {note}" for note in release_notes]])
        if planned_rows:
            out.extend(
                [
                    "",
                    "Planned campaigns awaiting admission (budgets included above):",
                    *[
                        f"- {row['label']}: AWAITING_ADMISSION; "
                        f"{row['planned_cells']} planned cells; no jobs submitted; "
                        f"{row['admission_reason']}"
                        for row in planned_rows
                    ],
                ]
            )
        if launch_gates:
            out.extend(
                [
                    "",
                    "Launch smoke gates (excluded from scientific cell totals):",
                    *[
                        f"- {label}: {gate['terminal']}/{gate['total']} terminal; "
                        f"{gate['running']} running; {gate['pending']} pending; "
                        f"{gate['failed']} failed"
                        for label, gate in launch_gates
                    ],
                ]
            )
        if incomplete_scale_breakdowns:
            out.extend(
                [
                    "",
                    "Scale coverage for incomplete multi-scale cohorts:",
                    *[
                        f"- {label}: "
                        + "; ".join(
                            f"{scale} {summary['terminal']}/{summary['cells']} "
                            f"terminal, {summary['running']} running, "
                            f"{summary['pending']} pending, "
                            f"{summary['depth']:.2f}/{summary['passes']}p"
                            for scale, summary in breakdown.items()
                        )
                        for label, breakdown in incomplete_scale_breakdowns
                    ],
                ]
            )
        if recovery_rows:
            out.extend(
                [
                    "",
                    "Audited recovery lineages (current successors, already included in cohort counts):",
                    "| cohort | lineage | cell | original resume | state | placement |",
                    "|---|---|---|---|---|---|",
                    *[
                        f"| {row['cohort'].upper()} "
                        f"| {row['old_job_id']} -> {row['job_id']} "
                        f"| {row['scale']}/{row['domain']}/{row['arm']}/s{row['seed']} "
                        f"| step {row['checkpoint']} "
                        f"| {str(row['state']).lower()} "
                        f"| {row.get('node') if row.get('node') not in (None, '', '-') else row.get('reason', '-')} |"
                        for row in recovery_rows
                    ],
                ]
            )
        if live_jobs:
            out.extend(
                [
                    "",
                    f"Live registered science allocations ({len(live_jobs)} listed; "
                    f"aggregate running count {running}):",
                    "| job | cohort | cell | step | state | elapsed | node | lineage |",
                    "|---|---|---|---|---|---|---|---|",
                    *[
                        f"| {job['job_id']} | {job['cohort']} "
                        f"| {job['scale']}/{job['domain']}/{job['arm']}/s{job['seed']} "
                        f"| {job['step']}/{job['target']} "
                        f"| {str(job.get('state', '-')).lower()} "
                        f"| {job.get('elapsed', '-')} | {job.get('node', '-')} "
                        f"| {','.join(str(old) for old in job['previous_job_ids']) or '-'} |"
                        for job in live_jobs
                    ],
                ]
            )

        if retired_rows:
            out.extend(
                [
                    "",
                    "Retired campaigns (excluded from active/scientific totals):",
                    *[
                        f"- {row['label']}: {row['terminal']}/{row['cells']} "
                        f"terminal before retirement; {row['realized']:,} "
                        "historical steps; 0 active jobs"
                        for row in retired_rows
                    ],
                ]
            )
        if held:
            out.extend(
                [
                    "",
                    "Registered but not released (no job ledger; excluded from totals):",
                    *[f"- {label}" for label in held],
                ]
            )
        return "\n".join(out)

    # Width follows the labels rather than a constant, so naming a cohort
    # accurately can never be discouraged by the table losing its alignment.
    width = max(
        [len("cohort"), len("TOTAL"), *(len(str(r["label"])) for r in active_rows)]
    )
    rule = "-" * (width + 56)
    out = [
        time.strftime("campaign status  %Y-%m-%d %H:%M:%S %Z"),
        "",
        f"{'cohort':<{width}} {'cells':>5} {'term':>5} {'run':>4} {'pend':>5} "
        f"{'fail':>5} "
        f"{'steps':>19} {'depth':>7}",
        rule,
    ]
    for r in active_rows:
        out.append(
            f"{r['label']:<{width}} {r['cells']:>5} {r['terminal']:>5} "
            f"{r['running']:>4} {r['pending']:>5} {r.get('failed', 0):>5} "
            f"{r['realized']:>9,}/{r['total']:<9,} "
            f"{r['depth']:>5.2f}/{r['passes']}p"
        )
    out.append(rule)
    out.append(
        f"{'TOTAL':<{width}} {cells:>5} {terminal:>5} {running:>4} {pending:>5} "
        f"{failed:>5} "
        f"{realized:>9,}/{total:<9,} "
        f"{100 * realized / total if total else 0.0:>6.1f}%"
    )
    if release_notes:
        out.extend(["", *[f"  {note}" for note in release_notes]])

    if planned_rows:
        out.extend(
            [
                "",
                "planned campaigns awaiting admission (budgets included above)",
                *[
                    f"  PLAN  {row['label']}: AWAITING_ADMISSION; "
                    f"{row['planned_cells']} planned cells; no jobs submitted; "
                    f"{row['admission_reason']}"
                    for row in planned_rows
                ],
            ]
        )

    if launch_gates:
        out.extend(
            [
                "",
                "launch smoke gates (excluded from scientific cell totals)",
                *[
                    f"  GATE  {label}: {gate['terminal']}/{gate['total']} terminal; "
                    f"{gate['running']} running; {gate['pending']} pending; "
                    f"{gate['failed']} failed"
                    for label, gate in launch_gates
                ],
            ]
        )
    if incomplete_scale_breakdowns:
        out.extend(
            [
                "",
                "scale coverage for incomplete multi-scale cohorts",
                *[
                    f"  SCALE  {label}: "
                    + "; ".join(
                        f"{scale} {summary['terminal']}/{summary['cells']} "
                        f"terminal, {summary['running']} running, "
                        f"{summary['pending']} pending, "
                        f"{summary['depth']:.2f}/{summary['passes']}p"
                        for scale, summary in breakdown.items()
                    )
                    for label, breakdown in incomplete_scale_breakdowns
                ],
            ]
        )
    if recovery_rows:
        out.extend(
            [
                "",
                "audited recovery lineages (current successors, included in cohort counts)",
                "  cohort  lineage               state       original resume  placement  cell",
                *[
                    f"  {str(row['cohort']).upper():<7} "
                    f"{str(row['old_job_id']) + '->' + str(row['job_id']):<21} "
                    f"{str(row['state']).lower():<11} "
                    f"{str(row['checkpoint']) + '/3072':>9}  "
                    f"{str(row.get('node') if row.get('node') not in (None, '', '-') else row.get('reason', '-')):<10} "
                    f"{row['scale']}/{row['domain']}/{row['arm']}/s{row['seed']}"
                    for row in recovery_rows
                ],
            ]
        )
    if live_jobs:
        out.extend(
            [
                "",
                f"live registered science allocations ({len(live_jobs)} listed; "
                f"aggregate running count {running})",
                "  job       cohort  state       elapsed    node      step       cell / lineage",
                *[
                    f"  {int(job['job_id']):<9} "
                    f"{str(job['cohort']):<7} "
                    f"{str(job.get('state', '-')).lower():<11} "
                    f"{str(job.get('elapsed', '-')):>9}  "
                    f"{str(job.get('node', '-')):<8} "
                    f"{str(job['step']) + '/' + str(job['target']):>9}  "
                    f"{job['scale']}/{job['domain']}/{job['arm']}/s{job['seed']}"
                    + (
                        "  restart<-" + ",".join(
                            str(old) for old in job["previous_job_ids"]
                        )
                        if job["previous_job_ids"]
                        else ""
                    )
                    for job in live_jobs
                ],
            ]
        )

    if retired_rows:
        out.extend(
            [
                "",
                "retired campaigns (excluded from active/scientific totals)",
                *[
                    f"  RETIRED  {row['label']}: {row['terminal']}/{row['cells']} "
                    f"terminal before retirement; {row['realized']:,} "
                    "historical steps; 0 active jobs"
                    for row in retired_rows
                ],
            ]
        )
    if held:
        out.extend(
            [
                "",
                "registered but not released (no job ledger; excluded from totals)",
                *[f"  HELD  {label}" for label in held],
            ]
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    parser.add_argument("--json", action="store_true", help="emit the raw rows")
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="include retired cohorts, stopped/completed scale details, completed recoveries, and zero-science-only launch gates",
    )
    args = parser.parse_args()

    rows = [
        row
        for label, name, reader, excluded_domains in COHORTS
        if (
            row := cohort_row(
                label,
                ARTIFACTS / name,
                reader,
                excluded_domains,
            )
        )
        is not None
    ]
    retirement = e105_retirement()
    if retirement is not None:
        for row in rows:
            if str(row["label"]).startswith("E105 "):
                row["retired"] = True
                row["retirement_record"] = str(E105_RETIREMENT)
    e112_retired = e112_retirement()
    if e112_retired is not None:
        for row in rows:
            if row.get("label") == registry.by_tag("e112").label:
                row["retired"] = True
                row["retirement_record"] = str(E112_RETIREMENT)
    r3_release = e113_r3_release()
    if r3_release is not None:
        for row in rows:
            if row.get("label") == registry.by_tag("e113").label:
                row["retired"] = True
                row["retirement_record"] = str(E113_R3_LEDGER)
    r3_retired = e113_r3_retirement()
    if r3_retired is not None:
        for row in rows:
            if row.get("label") == registry.by_tag("e113r3").label:
                row["retired"] = True
                row["retirement_record"] = str(E113_R3_RETIREMENT)
    enrich_live_jobs(rows)
    recovery_rows = tuple(latest_recovery_rows())
    held = tuple(
        label
        for label, name, _reader, _excluded_domains in COHORTS
        if not (ARTIFACTS / name).is_file()
    )
    if args.json:
        visible_rows = [
            row
            for row in rows
            if args.include_history
            or (row.get("retired") is not True and int(row["cells"]) > 0)
        ]
        print(json.dumps(visible_rows, indent=2))
        return 0
    print(
        render(
            rows,
            markdown=args.markdown,
            held=held,
            include_history=args.include_history,
            recovery_rows=recovery_rows,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
