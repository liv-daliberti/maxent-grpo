from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "paper/preregistration/e53_stage_a_execution_20260726.md"
LAUNCHER = ROOT / "ops/exp_scaling/launch_e53_stage_a.sh"
AUDITOR = ROOT / "ops/exp_scaling/audit_e53_stage_a.py"
VERIFIER = ROOT / "ops/exp_scaling/verify_e53_sentinel_approval.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE = _load("audit_e53_stage_a_contract", AUDITOR)
VERIFY = _load("verify_e53_stage_a_contract", VERIFIER)


def test_e53_stage_a_is_frozen_target_free_and_fresh():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "FROZEN DURING SENTINEL PASS 0",
        "All 27 jobs",
        "fresh seeds 43, 44,",
        "No sentinel checkpoint",
        "three-seed mean trajectory",
        "gold valid-answer",
    ):
        assert required in protocol
    for required in (
        "EXPECTED_JOBS_PER_DOMAIN=9",
        "EXPECTED_JOBS=27",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_ONLY_ARMS=grpo,maxent_inverse,maxent_inverse_canonical_replay",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16",
        "resume_from_sentinel",
        "scontrol release",
    ):
        assert required in launcher


def test_e53_stage_a_seed_mean_uses_all_three_seeds_and_e53_gate():
    def evaluations(offset: float):
        return [
            {
                "step": step,
                "distinct8": 1.0 + offset,
                "pass8": 0.5,
                "mean8": 0.4,
            }
            for step in range(1, 9)
        ]

    seeds = {
        seed: {
            "runs": {
                "grpo": {
                    "status": "complete",
                    "evaluations": evaluations(0.0),
                },
                STAGE.SENTINEL.REPLAY_ARM: {
                    "status": "complete",
                    "evaluations": evaluations(0.5),
                },
            }
        }
        for seed in STAGE.SEEDS
    }
    result = STAGE.seed_mean_behavioral_gate(seeds)
    assert result["status"] == "pass"
    assert result["paired_seed_mean_boundaries"] == 8
    assert result["checks"]["retains_75pct_of_own_best_rolling_eight"] is True


def test_e53_approval_verifier_rejects_live_nonterminal_audit(tmp_path):
    approval = tmp_path / "approval.json"
    approval.write_text(
        '{"schema":"e53_sentinel_audit_v2","status":"in_progress",'
        '"authorizes_stage_a":false,"violations":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(VERIFY.ApprovalError, match="unqualified pass"):
        VERIFY.verify_approval(approval_path=approval)


def test_e53_stage_a_monitor_and_auditor_are_identity_bound():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    auditor = AUDITOR.read_text(encoding="utf-8")
    for required in (
        "stage_a_auditor_sha256",
        "runtime_repair_protocol_sha256",
        "sentinel_runtime_auditor_sha256",
        "stage_a_watcher_sha256",
        "stage_a_watcher_slurm_sha256",
        "sentinel_approval_sha256",
        "sentinel_identity_sha256",
    ):
        assert required in launcher
        assert required in auditor
