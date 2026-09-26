from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e104_qwen3_update_only_capacity_preflight.py"
AUDITOR = ROOT / "ops/exp_scaling/audit_e104_qwen3_update_only_capacity_preflight.py"
PROTOCOL = ROOT / "paper/preregistration/e104_qwen3_update_only_capacity_preflight_20260817.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_update_only_preflight_keeps_exact_training_objective() -> None:
    launch = _load(LAUNCHER, "e104_q3_update_only_launch")
    plan = launch.build_plan()
    env = plan["env"]
    assert launch.SCALE == "qwen3b"
    assert launch.DOMAIN == "graph_coloring"
    assert launch.SEED == 70
    assert env["OAT_ZERO_MAX_TRAIN"] == "1"
    assert env["OAT_ZERO_CAPACITY_PREFLIGHT_SKIP_EVAL"] == "1"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_K"] == "0"
    assert env["OAT_ZERO_EXPORT_STEPS"] == "-1"
    assert env["OAT_ZERO_SAVE_STEPS"] == "0"
    assert env["OAT_ZERO_RESUME_STEPS"] == "0"
    assert env[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"
    ] == "1"
    assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"


def test_overlay_changes_exact_two_shell_files_and_has_bounded_hooks() -> None:
    launch = _load(LAUNCHER, "e104_q3_update_only_overlay")
    launch.verify_overlay()
    assert set(launch.OVERLAY_PATCH_SHA256) == {"run_experiment.sh", "train.sh"}
    run_text = (launch.OPS_OVERLAY / "run_experiment.sh").read_text()
    train_text = (launch.OPS_OVERLAY / "train.sh").read_text()
    assert 'EVAL_CADENCE_POLICY="capacity_preflight_none"' in run_text
    assert "cmd+=(--debug)" in train_text


def test_scheduler_and_protocol_are_static_only_and_capacity_only() -> None:
    launch = _load(LAUNCHER, "e104_q3_update_only_scheduler")
    command = launch.build_plan()["command"]
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "--partition=lowprio" in command
    assert "--gres=gpu:a6000:1" in command
    assert f"--nodelist={launch.NODE_LIST}" in command
    assert "includes no PointMaze run" in protocol
    assert "cannot replace an E104 outcome cell" in protocol


def test_auditor_reads_training_telemetry_but_no_evaluation_values() -> None:
    auditor = AUDITOR.read_text(encoding="utf-8")
    for forbidden in ("pass@", "distinct@", "mean_correct", "eval/pass"):
        assert forbidden not in auditor
    assert '"outcome_metrics_inspected": False' in auditor
    assert "evaluation_artifacts" in auditor
