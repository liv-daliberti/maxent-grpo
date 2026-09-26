from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def shell_case_body(name: str) -> str:
    text = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")
    match = re.search(rf"^  {re.escape(name)}\)\n(?P<body>.*?)^    ;;$", text, re.M | re.S)
    assert match is not None
    return match.group("body")


def test_e102_full_variant_keeps_semantics_outside_ppo() -> None:
    body = shell_case_body("open_bank_full_maxent_replay")
    required = (
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS=0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER",
    )
    for item in required:
        assert item in body


def test_e102_is_one_arm_paired_to_both_completed_e78_arms() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e102_full_open_bank_maxent_replay_05b.py",
        "e102_launch",
    )
    comparator = launch.check_e78_comparators(ROOT)
    assert comparator.name == "e78_verified_replay_only_05b_jobs.json"
    assert launch.ARM == "full_open_bank"
    assert len(launch.DOMAINS) * len(launch.SEEDS) == 25
    assert launch.TRAIN_ROWS == 384
    assert launch.PASSES == 8
    assert launch.TARGET_STEPS == 3072
    assert launch.SMOKE_TRAIN_ROWS == 4
    assert launch.SMOKE_PASSES == 8
    assert launch.SMOKE_TARGET_STEPS == 32


def test_e102_objective_is_safe_balance_plus_priority_mass() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e102_full_open_bank_maxent_replay_05b.py",
        "e102_objective",
    )
    objective = launch.fixed_objective()
    assert objective["OAT_ZERO_VARIANT"] == "open_bank_full_maxent_replay"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
    assert objective["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA"] == "0.0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "split_mass_balance_per_rollout"
    )
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA"] == "0.1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE"
    ] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "1"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT"
    ] == "1"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY"
    ] == "0"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS"
    ] == "0"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS"
    ] == "4"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER"
    ] == "4.0"


def test_e102_inherits_e78_schedule_and_uses_non_302_pool() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e102_full_open_bank_maxent_replay_05b.py",
        "e102_env",
    )
    source = launch.e78.references(ROOT)[0]
    env, target = launch.build_env(ROOT, source, Path("/frozen/snapshot"))
    command = launch.sbatch_command(ROOT, source, env)

    assert env["OAT_ZERO_MAX_TRAIN"] == "384"
    assert env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "8"
    assert env["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == "192"
    assert env["OAT_ZERO_NUM_SAMPLES"] == "16"
    assert env["OAT_ZERO_REPLICATED_FREEFORM_SAMPLING"] == "1"
    assert env["OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC"] == "1"
    assert "e102_full_open_bank" in str(target)
    assert "--partition=all" in command
    assert "--account=mltheory" in command
    assert "--gres=gpu:1" in command
    nodelist = next(part for part in command if part.startswith("--nodelist="))
    assert "node302" not in nodelist
    assert "node105" not in nodelist

    pantry = next(
        run for run in launch.e78.references(ROOT) if run["domain"] == "pantry_plan"
    )
    pantry_env, _ = launch.build_env(ROOT, pantry, Path("/frozen/snapshot"))
    assert pantry_env["OAT_ZERO_CANONICAL_ACTION_TASK"] == "pantry_support_mask"
    assert pantry_env["OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING"] == "1"
    assert pantry_env["OAT_ZERO_REPLICATED_FREEFORM_SAMPLING"] == "0"
    assert pantry_env["OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC"] == "0"
    assert pantry_env["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "1"


def test_e102_is_visible_to_base_campaign_stats() -> None:
    cohorts = load_module("ops/exp_scaling/cohorts.py", "e102_cohorts")
    selected = [cohort for cohort in cohorts.REGISTRY if cohort.tag == "e102"]
    assert len(selected) == 1
    cohort = selected[0]
    assert cohort.ledger == "e102_full_open_bank_maxent_replay_05b_jobs.json"
    assert cohort.comparator == "replay"
    assert cohort.arm == "full_open_bank"
    assert not cohort.plotted
    assert cohort.not_plotted_because


def test_e102_protocol_freezes_smoke_and_no_rerun_contract() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e102_full_open_bank_maxent_replay_05b.py",
        "e102_protocol",
    )
    protocol = (ROOT / launch.PROTOCOL).read_text(encoding="utf-8")
    words = " ".join(protocol.split())
    assert "does not rerun either E78 comparator" in words
    assert "Proposal rows are never appended to the PPO batch" in words
    assert "Priority affects only replay mass" in words
    assert "at least one discovered mode" in words


def test_e102_smoke_auditor_requires_safe_priority_actuation(tmp_path: Path) -> None:
    audit = load_module(
        "ops/exp_scaling/audit_e102_full_open_bank_maxent_replay.py",
        "e102_audit",
    )
    metrics = tmp_path / "debug_job1" / "train_metrics.jsonl"
    metrics.parent.mkdir()
    row = {
        "misc/global_step": 8,
        "train/loss": 0.1,
        "train/canonical_replay_actuator_groups": 1.0,
        "train/canonical_replay_eligible_groups": 1.0,
        "train/canonical_replay_retention_safe_balance": 1.0,
        "train/canonical_replay_balance_scale_min": 0.4,
        "train/canonical_replay_requested_positive_gradient_max": 0.2,
        "train/canonical_replay_applied_positive_gradient_max": 0.0,
        "train/canonical_replay_priority_modes": 1.0,
        "train/canonical_replay_priority_replay_groups_cumulative": 1.0,
        "train/canonical_replay_priority_replay_modes_cumulative": 1.0,
        "train/canonical_replay_mass_weight_max": 1.6,
        "actor/counterfactual_proposal_groups_generated": 1.0,
        "actor/counterfactual_proposal_rows_generated": 16.0,
        "actor/counterfactual_proposal_cumulative_new_outcomes": 1.0,
        "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
        "actor/counterfactual_proposal_objective_outcome_delta": 0.0,
        "actor/counterfactual_proposal_objective_support_separated": 1.0,
        "actor/counterfactual_proposal_transform_enabled": 0.0,
        "actor/counterfactual_proposal_exact_grammar_transform_enabled": 0.0,
        "actor/counterfactual_proposal_gold_support_feedback": 0.0,
    }
    metrics.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report, violations = audit.smoke_gate(tmp_path, 8)

    assert violations == []
    assert report["proposal_cumulative_admissions"] == 1.0
    assert report["priority_replay_groups_cumulative"] == 1.0
    assert report["applied_positive_gradient_max"] == 0.0
    assert report["balance_scale_min"] == 0.4
