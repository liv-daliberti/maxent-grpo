"""Continuation identity must survive documented allocation changes only."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "campaign_stats_e120_runtime_under_test", ROOT / "ops/exp_scaling/campaign_stats.py"
)
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    original = {"job_id": 100, "domain": "graph_coloring", "model_key": "qwen3b",
                "seed": 73, "run_dir": "/registered/run", "run_stamp": "registered"}
    ledger = tmp_path / "primary.json"
    ledger.write_text(json.dumps({"runs": [original]}))
    path = tmp_path / "continuations.json"
    row = {key: value for key, value in original.items() if key != "job_id"}
    row.update(original_job_id=100, continuation_job_id=200)
    payload = {
        "schema": "e120r1_scheduler_continuation_jobs_v1",
        "original_ledger": str(ledger),
        "original_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "same_scientific_cells": True, "same_run_directories": True,
        "optimizer_update_changed": False, "treatment_changed": False,
        "scheduler_only": True, "pvl_excluded": True, "released": True,
        "installed": True, "outcomes_inspected": False, "continuations": [row],
    }
    monkeypatch.setattr(stats, "E120_LEDGER", ledger)
    monkeypatch.setattr(stats, "E120_CONTINUATIONS", path)
    def resolve():
        path.write_text(json.dumps(payload))
        return stats.e120_continuation_jobs(ledger)
    return payload, row, ledger, resolve


def amend(payload, row):
    payload.update(scheduler_only=False, runtime_allocation_only=True)
    row.update(runtime_changes={"OAT_ZERO_VLLM_GPU_RATIO": {"before": "0.25", "after": "0.40"}},
               optimizer_update_changed=False, treatment_changed=False)


def test_existing_scheduler_only_mapping_is_preserved(continuation):
    _, _, _, resolve = continuation
    assert resolve() == {100: 200}


def test_documented_cache_allocation_preserves_registered_identity(continuation):
    payload, row, _, resolve = continuation
    amend(payload, row)
    assert resolve() == {100: 200}


@pytest.mark.parametrize("mutation", ["treatment", "batch", "ratio", "identity", "hash", "unmarked", "empty"])
def test_unvalidated_or_scientific_changes_fail_closed(continuation, mutation):
    payload, row, ledger, resolve = continuation
    amend(payload, row)
    if mutation == "treatment":
        row["treatment_changed"] = True
    elif mutation == "batch":
        row["runtime_changes"]["OAT_ZERO_TRAIN_BATCH_SIZE"] = {"before": "16", "after": "8"}
    elif mutation == "ratio":
        row["runtime_changes"]["OAT_ZERO_VLLM_GPU_RATIO"]["after"] = "0.99"
    elif mutation == "identity":
        row["seed"] = 74
    elif mutation == "hash":
        ledger.write_text(ledger.read_text() + " ")
    elif mutation == "unmarked":
        payload["scheduler_only"] = True
    elif mutation == "empty":
        row.pop("runtime_changes")
    assert resolve() == {}
