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


def test_e103_variant_changes_only_the_explorer_budget_scheduler() -> None:
    body = shell_case_body("open_bank_starvation_fallback_maxent_replay")
    required = (
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK_MAX_ATTEMPTS",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS",
    )
    for item in required:
        assert item in body


def test_e103_full_objective_matches_e102_except_for_fallback() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e103_starvation_fallback_maxent_replay_05b.py",
        "e103_objective",
    )
    objective = launch.fixed_objective()
    baseline = launch.e102.fixed_objective()
    fallback_keys = {
        key for key in objective if "STARVATION" in key
    }
    assert {key: value for key, value in objective.items() if key not in fallback_keys} == (
        baseline | {"OAT_ZERO_VARIANT": launch.VARIANT}
    )
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK"
    ] == "1"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES"
    ] == "64"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK_MAX_ATTEMPTS"
    ] == "4"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_BURST_UPDATES"
    ] == "16"
    assert objective[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_COOLDOWN_UPDATES"
    ] == "48"


def test_e103_is_25_cells_paired_to_completed_e102_without_reruns() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e103_starvation_fallback_maxent_replay_05b.py",
        "e103_comparator",
    )
    comparator = launch.check_e102_comparator(ROOT)
    assert comparator.name == "e102_full_open_bank_maxent_replay_05b_jobs.json"
    assert len(launch.DOMAINS) * len(launch.SEEDS) == 25
    assert launch.TARGET_STEPS == 3072
    assert launch.PROPOSAL_MAX_ATTEMPTS == 1


def test_e103_full_and_smoke_envs_freeze_distinct_scheduler_timings() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e103_starvation_fallback_maxent_replay_05b.py",
        "e103_env",
    )
    source = launch.e102.e78.references(ROOT)[0]
    full_env, full_target = launch.build_env(
        ROOT, source, Path("/frozen/snapshot"), smoke=False
    )
    smoke_env, smoke_target = launch.build_env(
        ROOT, source, Path("/frozen/snapshot"), smoke=True
    )
    assert full_env["OAT_ZERO_MAX_TRAIN"] == "384"
    assert full_env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "8"
    assert full_env[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES"
    ] == "64"
    assert smoke_env["OAT_ZERO_MAX_TRAIN"] == "4"
    assert smoke_env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "8"
    assert smoke_env[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_PATIENCE_UPDATES"
    ] == "2"
    assert "e103_starvation_fallback" in str(full_target)
    assert str(smoke_target).endswith("_smoke")
    command = launch.sbatch_command(ROOT, source, full_env)
    assert "--partition=all" in command
    assert "--account=mltheory" in command
    assert "--mem=36G" in command
    assert "--nice=0" in command
    nodelist = next(item for item in command if item.startswith("--nodelist="))
    for excluded in ("node007", "node105", "node206", "node302"):
        assert excluded not in nodelist


def test_e103_is_visible_to_base_campaign_stats() -> None:
    cohorts = load_module("ops/exp_scaling/cohorts.py", "e103_cohorts")
    selected = [cohort for cohort in cohorts.REGISTRY if cohort.tag == "e103"]
    assert len(selected) == 1
    cohort = selected[0]
    assert cohort.ledger == "e103_starvation_fallback_maxent_replay_05b_jobs.json"
    assert cohort.arm == "starvation_fallback"
    assert not cohort.plotted
    assert cohort.not_plotted_because


def test_e103_protocol_freezes_target_free_no_rerun_contract() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e103_starvation_fallback_maxent_replay_05b.py",
        "e103_protocol",
    )
    words = " ".join((ROOT / launch.PROTOCOL).read_text(encoding="utf-8").split())
    assert "E103 does not rerun E102 or either E78 comparator" in words
    assert "Proposal rows are never appended to the PPO batch" in words
    assert "reset: only an actual new validator-positive bank admission" in words
    assert "does not inspect task accuracy or breadth" in words


def test_e103_smoke_auditor_requires_fallback_without_ppo_leakage(
    tmp_path: Path,
) -> None:
    audit = load_module(
        "ops/exp_scaling/audit_e103_starvation_fallback.py",
        "e103_audit",
    )
    metrics = tmp_path / "debug_job1" / "train_metrics.jsonl"
    metrics.parent.mkdir()
    row = {
        "misc/global_step": 32,
        "train/loss": 0.1,
        "train/canonical_replay_actuator_groups": 1.0,
        "train/canonical_replay_eligible_groups": 1.0,
        "train/canonical_replay_retention_safe_balance": 1.0,
        "train/canonical_replay_balance_scale_min": 0.4,
        "train/canonical_replay_applied_positive_gradient_max": 0.0,
        "actor/counterfactual_proposal_groups_generated": 4.0,
        "actor/counterfactual_proposal_rows_generated": 64.0,
        "actor/counterfactual_proposal_max_attempts": 4.0,
        "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
        "actor/counterfactual_proposal_objective_outcome_delta": 0.0,
        "actor/counterfactual_proposal_objective_support_separated": 1.0,
        "actor/counterfactual_proposal_transform_enabled": 0.0,
        "actor/counterfactual_proposal_exact_grammar_transform_enabled": 0.0,
        "actor/counterfactual_proposal_starvation_fallback_enabled": 1.0,
        "actor/counterfactual_proposal_starvation_eligible_update": 1.0,
        "actor/counterfactual_proposal_starvation_fallback_active": 1.0,
        "actor/counterfactual_proposal_starvation_fallback_extra_groups": 3.0,
        "actor/counterfactual_proposal_starvation_fallback_activations_cumulative": 1.0,
        "actor/counterfactual_proposal_starvation_fallback_updates_cumulative": 1.0,
        "actor/counterfactual_proposal_starvation_gold_support_feedback": 0.0,
        "actor/counterfactual_proposal_starvation_desired_mode_count_feedback": 0.0,
        "actor/counterfactual_proposal_starvation_eval_feedback": 0.0,
    }
    metrics.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report, violations = audit.smoke_gate(tmp_path, 32)

    assert violations == []
    assert report["fallback_activations"] == 1.0
    assert report["fallback_extra_groups"] == 3.0
    assert report["proposal_rows_to_ppo_max"] == 0.0
    assert report["applied_positive_gradient_max"] == 0.0
