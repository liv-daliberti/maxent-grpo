from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops/exp_scaling/apply_e104_qwen3_a6000_placement_amendment.py"
PROTOCOL = ROOT / (
    "paper/preregistration/e104_qwen3_a6000_placement_amendment_20260817.md"
)


def _load():
    spec = importlib.util.spec_from_file_location("e104_q3_amendment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_targets_only_five_static_qwen3_cells() -> None:
    module = _load()
    assert module.SCALE == "qwen3b"
    assert module.TARGET_JOB_IDS == tuple(range(30637795, 30637800))
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "No PointMaze\nrun is included" in text
    assert "does not authorize any E105 placement change" in text


def test_update_changes_only_scheduler_placement() -> None:
    module = _load()
    command = module.update_command(30637795, amended=True)
    assert command == [
        "scontrol",
        "update",
        "JobId=30637795",
        "Partition=lowprio",
        "Account=mltheory",
        "NodeList=node[103-104,205-208,805]",
        "Gres=gpu:a6000:1",
        "TimeLimit=02:00:00",
    ]
    assert not any("OAT_ZERO_" in token for token in command)


def test_preflight_gate_rejects_outcome_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    ledger = {"job_id": 17}
    audit = {
        "job_id": 17,
        "complete": True,
        "passed": True,
        "scheduler_state": "COMPLETED",
        "outcome_metrics_inspected": True,
        "violations": [],
    }
    monkeypatch.setattr(
        module,
        "load",
        lambda path: ledger if path == module.PREFLIGHT_LEDGER else audit,
    )
    with pytest.raises(RuntimeError, match="outcome-blind"):
        module.validate_preflight()


def test_original_and_amended_records_require_pending_zero_runtime() -> None:
    module = _load()
    run = {"job_id": 30637795}
    scientific = (
        f" OAT_ZERO_VARIANT={module.e104.VARIANT}"
        " OAT_ZERO_MAX_TRAIN=64 OAT_ZERO_SEED=70"
        " OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1"
        " OAT_ZERO_ONLINE_CANONICAL_REPLAY=1"
    )
    original = (
        "JobState=PENDING RunTime=00:00:00 TimeLimit=02:00:00"
        " Partition=mltheory Account=mltheory ReqNodeList=node302"
        " TresPerNode=gres/gpu:a100:1" + scientific
    )
    amended = original.replace("Partition=mltheory", "Partition=lowprio").replace(
        "ReqNodeList=node302", "ReqNodeList=node[103-104,205-208,805]"
    ).replace("gres/gpu:a100:1", "gres/gpu:a6000:1")
    module.validate_original(run, original)
    module.validate_amended(run, amended)
    with pytest.raises(RuntimeError, match="RunTime"):
        module.validate_amended(run, amended.replace("RunTime=00:00:00", "RunTime=00:00:01"))
