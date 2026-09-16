from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e104_qwen3_update_only_capacity_retry.py"
AUDITOR = ROOT / "ops/exp_scaling/audit_e104_qwen3_update_only_capacity_retry.py"
PROTOCOL = ROOT / "paper/preregistration/e104_qwen3_update_only_capacity_retry_20260817.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retry_uses_valid_disabled_persistence_sentinels() -> None:
    launch = _load(LAUNCHER, "e104_q3_retry_launch")
    plan = launch.build_plan()
    env = plan["env"]
    assert env["OAT_ZERO_RESUME_STEPS"] == "-1"
    assert env["OAT_ZERO_EXPORT_STEPS"] == "-1"
    assert env["OAT_ZERO_SAVE_STEPS"] == "0"
    assert env["OAT_ZERO_CAPACITY_PREFLIGHT_SKIP_EVAL"] == "1"
    assert env[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"
    ] == "1"
    assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"


def test_retry_overlay_is_exact_bounded_repair() -> None:
    launch = _load(LAUNCHER, "e104_q3_retry_overlay")
    launch.verify_overlay()
    assert set(launch.OVERLAY_PATCH_SHA256) == {"run_experiment.sh", "train.sh"}
    run_text = (launch.OPS_OVERLAY / "run_experiment.sh").read_text()
    assert "export OAT_ZERO_RESUME_STEPS=-1" in run_text
    assert "EVAL_STEPS=0" in run_text


def test_retry_is_static_only_and_outcome_blind() -> None:
    launch = _load(LAUNCHER, "e104_q3_retry_scheduler")
    command = launch.build_plan()["command"]
    protocol = PROTOCOL.read_text(encoding="utf-8")
    auditor = AUDITOR.read_text(encoding="utf-8")
    assert "--partition=lowprio" in command
    assert "--gres=gpu:a6000:1" in command
    assert "includes no PointMaze run" in protocol
    assert "created no" in protocol and "evaluation artifact" in protocol
    for forbidden in ("pass@", "distinct@", "mean_correct", "eval/pass"):
        assert forbidden not in auditor
    assert '"outcome_metrics_inspected": False' in auditor
