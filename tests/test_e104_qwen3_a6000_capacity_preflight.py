from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e104_qwen3_a6000_capacity_preflight.py"
AUDITOR = ROOT / "ops/exp_scaling/audit_e104_qwen3_a6000_capacity_preflight.py"
PROTOCOL = ROOT / "paper/preregistration/e104_qwen3_a6000_capacity_preflight_20260817.md"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_is_one_update_qwen3_static_domain_and_no_pointmaze() -> None:
    launch = _load(LAUNCHER, "e104_q3_a6000_launch")
    assert launch.SCALE == "qwen3b"
    assert launch.DOMAIN == "graph_coloring"
    assert launch.SEED == 70
    assert launch.TRAIN_ROWS == 1
    assert "point" not in launch.DOMAIN


def test_preflight_plan_reuses_snapshot_objective_and_broadens_only_hardware() -> None:
    launch = _load(LAUNCHER, "e104_q3_a6000_plan")
    plan = launch.build_plan(ROOT)
    env = plan["env"]
    command = plan["command"]
    assert env["OAT_ZERO_MAX_TRAIN"] == "1"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_K"] == "0"
    assert env[
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE"
    ] == "1"
    assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
    assert "--partition=lowprio" in command
    assert "--account=mltheory" in command
    assert "--gres=gpu:a6000:1" in command
    assert f"--nodelist={launch.NODE_LIST}" in command
    assert str(plan["snapshot"]) == __import__("json").loads(
        (ROOT / launch.e104.LEDGER).read_text()
    )["snapshot_root"]


def test_preflight_auditor_is_outcome_blind_and_protocol_is_capacity_only() -> None:
    auditor = AUDITOR.read_text()
    protocol = PROTOCOL.read_text()
    for forbidden in ("pass@", "distinct@", "mean_correct", "eval/pass"):
        assert forbidden not in auditor
    assert '"outcome_metrics_inspected": False' in auditor
    assert "It is not an outcome experiment" in protocol
    assert "E105 retains its registered paired placements" in protocol


def test_preflight_auditor_accepts_one_finite_live_update(tmp_path: Path) -> None:
    audit = _load(AUDITOR, "e104_q3_a6000_audit")
    attempt = tmp_path / "debug_job1"
    attempt.mkdir()
    row = {
        "misc/global_step": 1,
        "train/semantic_shannon_success_conditioned_group_centered_advantage_active": 1.0,
        "train/semantic_shannon_success_conditioned_signed_advantage_active": 0.0,
        "train/semantic_rms_controller_active": 0.0,
        "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_mean": 0.0,
        "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_min": -0.02,
        "train/semantic_shannon_success_conditioned_group_centered_effective_advantage_max": 0.04,
        "train/canonical_replay_applied_score_gradient_l2": 0.1,
    }
    (attempt / "train_metrics.jsonl").write_text(json.dumps(row) + "\n")

    report, violations = audit.parse_run(tmp_path)

    assert violations == []
    assert report["last_step"] == 1
    assert report["replay_gradient_l2_max"] == 0.1
