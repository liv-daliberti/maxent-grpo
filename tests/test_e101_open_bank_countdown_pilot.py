from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shell_case_body(name: str) -> str:
    text = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")
    match = re.search(rf"^  {re.escape(name)}\)\n(?P<body>.*?)^    ;;$", text, re.M | re.S)
    assert match is not None
    return match.group("body")


def test_score_toy_separates_discovery_mass_and_balance() -> None:
    toy = load_module("ops/exp_scaling/open_bank_maxent_toy.py", "e101_toy")
    singleton = toy.split_score_gradients(
        [-0.2], mass_weight=0.1, balance_weight=0.1
    )
    assert singleton["balance"] == [0.0]

    report = toy.build_report(steps=12, learning_rate=1.0)
    first = report["first_step_after_admission"]["gradients"]
    assert math.isclose(first["weighted"][0], -0.0119202922, abs_tol=1e-9)
    assert math.isclose(first["weighted"][1], -0.0880797078, abs_tol=1e-9)
    assert math.isclose(report["mass_only"]["final_score_gap"], 2.0, abs_tol=1e-12)
    assert report["mass_plus_balance"]["final_score_gap"] < 2.0


def test_retention_safe_projection_preserves_mass_without_antitraining() -> None:
    toy = load_module("ops/exp_scaling/open_bank_maxent_toy.py", "e101_safe_toy")
    pair = toy.retention_safe_score_gradients(
        [-0.2, -2.2], mass_weight=0.1, balance_weight=0.1
    )
    assert all(
        math.isclose(raw, safe, abs_tol=1e-12)
        for raw, safe in zip(pair["weighted"], pair["retention_safe"])
    )

    four_scores = [math.log(0.9), *[math.log(0.1 / 3.0)] * 3]
    four = toy.retention_safe_score_gradients(
        four_scores,
        mass_weight=0.1,
        balance_weight=0.1,
    )
    assert four["weighted"][0] > 0.0
    assert four["retention_safe"][0] == 0.0
    assert all(value <= 0.0 for value in four["retention_safe"])
    assert math.isclose(sum(four["retention_safe"]), -0.1, abs_tol=1e-12)
    assert all(
        math.isclose(value, -1.0 / 30.0, abs_tol=1e-12)
        for value in four["retention_safe"][1:]
    )


def test_bank_local_balance_cap_is_maximal_and_retention_safe() -> None:
    toy = load_module("ops/exp_scaling/open_bank_maxent_toy.py", "e101_cap_toy")
    pair = toy.retention_safe_balance_gradients(
        [-0.2, -2.2], mass_weight=0.1, balance_weight=0.1
    )
    assert math.isclose(pair["safe_balance_weight"], 0.1, abs_tol=1e-12)
    assert all(
        math.isclose(raw, safe, abs_tol=1e-12)
        for raw, safe in zip(pair["weighted"], pair["retention_safe"])
    )

    four_scores = [math.log(0.9), *[math.log(0.1 / 3.0)] * 3]
    four = toy.retention_safe_balance_gradients(
        four_scores,
        mass_weight=0.1,
        balance_weight=0.1,
    )
    assert math.isclose(
        four["safe_balance_weight"], 0.1 / 2.6, abs_tol=1e-12
    )
    assert abs(four["retention_safe"][0]) <= 1e-12
    assert all(value <= 1e-12 for value in four["retention_safe"])
    assert math.isclose(sum(four["retention_safe"]), -0.1, abs_tol=1e-12)

    five_probabilities = [0.5, 0.41, 0.03, 0.03, 0.03]
    five = toy.retention_safe_balance_gradients(
        [math.log(value) for value in five_probabilities],
        mass_weight=0.1,
        balance_weight=0.1,
    )
    assert five["weighted"][0] > 0.0
    assert five["weighted"][1] > 0.0
    assert math.isclose(five["safe_balance_weight"], 1.0 / 15.0, abs_tol=1e-12)
    assert abs(five["retention_safe"][0]) <= 1e-12
    assert five["retention_safe"][1] < 0.0


def test_open_variant_has_no_advantage_entropy_channel() -> None:
    body = shell_case_body("open_bank_maxent_replay")
    required = (
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS=0",
    )
    for item in required:
        assert item in body


def test_three_arm_ladder_differs_only_as_registered() -> None:
    launch = load_module(
        "ops/exp_scaling/launch_e101_open_bank_countdown_pilot.py",
        "e101_launch",
    )
    mass = launch.fixed_objective("mass")
    balance = launch.fixed_objective("balance")
    opened = launch.fixed_objective("open")

    assert mass["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == "verified_likelihood_per_rollout"
    assert balance["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == "split_mass_balance_per_rollout"
    assert opened["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == "split_mass_balance_per_rollout"
    assert balance["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "0"
    assert opened["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "1"
    assert opened["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS"] == "0"
    run_source = (ROOT / "src/oat_drgrpo/learner/run.py").read_text()
    assert "if transform_proposals" in run_source
    for arm in (mass, balance, opened):
        assert arm["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
        assert arm["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "0"
        assert arm["OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA"] == "0.0"
        assert arm["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
        assert arm["OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA"] == "0.1"

    source = launch.reference(ROOT)
    snapshot_root = Path(
        __import__("json").loads(
            (ROOT / "var/artifacts/e101_open_bank_countdown_pilot_jobs.json").read_text()
        )["snapshot_root"]
    )
    for arm_name in launch.ARMS:
        env, _ = launch.build_env(ROOT, source, arm_name, snapshot_root)
        assert env["OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE"] == "1"
        assert env["OAT_ZERO_NUM_GPUS_PER_ACTOR"] == "1"
        assert env["OAT_ZERO_REPLICATED_FREEFORM_SAMPLING"] == "1"
        assert env["OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC"] == "1"
        assert env["OAT_ZERO_VLLM_SLEEP_LEVEL"] == "1"


def test_e101m_is_a_narrow_open_arm_mechanism_smoke() -> None:
    smoke = load_module(
        "ops/exp_scaling/launch_e101m_open_bank_mechanism_smoke.py",
        "e101m_launch",
    )
    parent = __import__("json").loads(
        (ROOT / "var/artifacts/e101_open_bank_countdown_pilot_jobs.json").read_text()
    )
    env, target = smoke.smoke_env(ROOT, Path(parent["snapshot_root"]))
    command = smoke.sbatch_command(ROOT, env)

    assert smoke.TRAIN_ROWS == 8
    assert smoke.TARGET_STEPS == 8
    assert smoke.TIME_LIMIT == "00:30:00"
    assert smoke.NODES == "node[007,020,022-023,101,103,202,204-206,302,403,805]"
    assert smoke.PARTITION == "all"
    assert smoke.ACCOUNT == "mltheory"
    assert smoke.MEMORY == "22G"
    assert "mechanism_smoke" in str(target)
    assert "mechanism_smoke_r1" in smoke.RUN_STAMP
    assert env["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
    assert env["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "0"
    assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "split_mass_balance_per_rollout"
    )
    assert env["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "1"
    assert env[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT"
    ] == "1"
    assert env["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS"] == "0"
    assert env["OAT_ZERO_REPLICATED_FREEFORM_SAMPLING"] == "1"
    assert env["OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC"] == "1"
    assert "--gres=gpu:1" in command
    protocol = (ROOT / smoke.PROTOCOL).read_text(encoding="utf-8")
    assert "not an additional E101 comparison arm" in protocol
    assert "proposal rows sent to PPO and feedback remain exactly zero" in protocol


def test_e101_auditor_counts_admission_and_same_step_replay_actuation(
    tmp_path: Path,
) -> None:
    audit = load_module(
        "ops/exp_scaling/audit_e101_open_bank_countdown_pilot.py",
        "e101_audit_success",
    )
    records = [
        {
            "trainer/global_step": 1,
            "train/loss": 0.1,
            "train/canonical_replay_actuator_groups": 1.0,
            "train/canonical_replay_actuator_modes": 2.0,
            "train/canonical_replay_eligible_groups": 1.0,
            "train/canonical_replay_retained_modes": 2.0,
            "train/canonical_replay_realized_response_tokens": 7.0,
            "actor/counterfactual_proposal_singleton_only_active": 1.0,
            "actor/counterfactual_proposal_original_prompt_groups": 1.0,
            "actor/counterfactual_proposal_admitted_new_outcomes": 1.0,
            "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
            "actor/counterfactual_proposal_objective_outcome_delta": 0.0,
            "actor/counterfactual_proposal_gold_support_feedback": 0.0,
        },
        {
            "trainer/global_step": 2,
            "train/loss": 0.1,
            "train/canonical_replay_actuator_groups": 1.0,
            "train/canonical_replay_actuator_modes": 2.0,
            "train/canonical_replay_eligible_groups": 1.0,
            "train/canonical_replay_retained_modes": 2.0,
            "train/canonical_replay_realized_response_tokens": 7.0,
        },
    ]
    metrics_path = tmp_path / "train_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    metrics, violations = audit.parse_metrics(metrics_path)

    assert violations == []
    mechanism = metrics["mechanism"]
    assert mechanism["proposal_admissions"] == 1.0
    assert mechanism["proposal_groups_generated"] == 0.0
    assert mechanism["proposal_rows_generated"] == 0.0
    assert mechanism["first_admission_step"] == 1
    assert mechanism["replay_actuation_updates_at_or_after_first_admission"] == 2
    assert mechanism["proposal_rows_to_ppo_max_abs"] == 0.0
    assert mechanism["proposal_objective_outcome_delta_max_abs"] == 0.0
    assert mechanism["forbidden_feedback_max_abs"] == 0.0


def test_e101_auditor_rejects_proposal_leakage(tmp_path: Path) -> None:
    audit = load_module(
        "ops/exp_scaling/audit_e101_open_bank_countdown_pilot.py",
        "e101_audit_leakage",
    )
    record = {
        "trainer/global_step": 1,
        "train/loss": 0.1,
        "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 1.0,
        "actor/counterfactual_proposal_objective_outcome_delta": 1.0,
        "actor/counterfactual_proposal_gold_support_feedback": 1.0,
    }
    metrics_path = tmp_path / "train_metrics.jsonl"
    metrics_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    _, violations = audit.parse_metrics(metrics_path)

    assert any("proposal rows reached PPO" in value for value in violations)
    assert any("proposal changed neutral objective support" in value for value in violations)
    assert any("forbidden training feedback" in value for value in violations)


def test_e101m2_expands_only_discovery_opportunity() -> None:
    discovery = load_module(
        "ops/exp_scaling/launch_e101m2_open_bank_discovery_smoke.py",
        "e101m2_launch",
    )
    parent = json.loads(
        (ROOT / "var/artifacts/e101_open_bank_countdown_pilot_jobs.json").read_text()
    )
    env, target = discovery.discovery_env(ROOT, Path(parent["snapshot_root"]))
    command = discovery.sbatch_command(ROOT, env)

    assert discovery.TRAIN_ROWS == 32
    assert discovery.PASSES == 1
    assert discovery.TARGET_STEPS == 32
    assert discovery.MAX_ATTEMPTS == 3
    assert discovery.TIME_LIMIT == "00:30:00"
    assert "e101m2_open_bank_discovery" in str(target)
    assert env["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
    assert env["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "0"
    assert env["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "split_mass_balance_per_rollout"
    )
    assert env["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS"] == "1"
    assert env["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS"] == "3"
    assert env["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS"] == "0"
    assert env["OAT_ZERO_REPLICATED_FREEFORM_SAMPLING"] == "1"
    assert env["OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC"] == "1"
    assert "--account=mltheory" in command
    assert "--gres=gpu:1" in command
    protocol = (ROOT / discovery.PROTOCOL).read_text(encoding="utf-8")
    assert "changes only discovery opportunity" in protocol
    assert "not a performance arm" in " ".join(protocol.split())


def test_e101m3_enables_only_the_exact_grammar_actuator() -> None:
    exact = load_module(
        "ops/exp_scaling/launch_e101m3_exact_grammar_open_bank_smoke.py",
        "e101m3_launch",
    )
    parent = json.loads(
        (ROOT / "var/artifacts/e101_open_bank_countdown_pilot_jobs.json").read_text()
    )
    env, target = exact.exact_env(ROOT, Path(parent["snapshot_root"]))
    command = exact.sbatch_command(ROOT, env)

    assert exact.SEED == 102
    assert exact.TRAIN_ROWS == exact.TARGET_STEPS == 32
    assert exact.MAX_ATTEMPTS == 1
    assert "e101m3_exact_grammar_open_bank" in str(target)
    assert env["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
    assert env["OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE"] == "0"
    assert env["OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA"] == "0.0"
    assert env["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS"] == "0"
    assert env[
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS"
    ] == "1"
    assert "--account=mltheory" in command
    assert "--gres=gpu:1" in command


def test_e101_parser_separately_gates_exact_grammar_transform_telemetry(
    tmp_path: Path,
) -> None:
    audit = load_module(
        "ops/exp_scaling/audit_e101_open_bank_countdown_pilot.py",
        "e101_audit_exact_grammar",
    )
    record = {
        "trainer/global_step": 1,
        "train/loss": 0.1,
        "train/canonical_replay_actuator_groups": 1.0,
        "train/canonical_replay_actuator_modes": 2.0,
        "actor/counterfactual_proposal_singleton_only_active": 1.0,
        "actor/counterfactual_proposal_transform_enabled": 0.0,
        "actor/counterfactual_proposal_exact_grammar_transform_enabled": 1.0,
        "actor/counterfactual_proposal_transform_candidate_surfaces": 7.0,
        "actor/counterfactual_proposal_transform_validator_positive": 1.0,
        "actor/counterfactual_proposal_transform_novel_unique_outcomes": 1.0,
        "actor/counterfactual_proposal_transform_success": 1.0,
        "actor/counterfactual_proposal_admitted_new_outcomes": 1.0,
        "actor/counterfactual_proposal_groups_generated": 0.0,
        "actor/counterfactual_proposal_transform_rows_sent_to_ppo": 0.0,
        "actor/counterfactual_proposal_objective_outcome_delta": 0.0,
        "actor/counterfactual_proposal_transform_gold_support_feedback": 0.0,
    }
    path = tmp_path / "train_metrics.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    _, default_violations = audit.parse_metrics(path)
    metrics, exact_violations = audit.parse_metrics(
        path,
        allow_exact_grammar_transforms=True,
    )

    assert any("exact-grammar" in value for value in default_violations)
    assert exact_violations == []
    mechanism = metrics["mechanism"]
    assert mechanism["proposal_exact_grammar_enabled_max"] == 1.0
    assert mechanism["proposal_transform_novel_outcomes"] == 1.0
    assert mechanism["proposal_admissions"] == 1.0
    assert mechanism["replay_actuation_updates_at_or_after_first_admission"] == 1
