from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402


def test_e117_is_exactly_four_paired_three_arm_sentinels():
    assert e117.SEED == 117
    assert e117.TRAIN_ROWS == 64
    assert e117.PASSES == 1
    assert e117.TARGET_STEPS == 64
    assert e117.CHECKPOINT_INTERVAL == 32
    assert e117.ARMS == ("c", "p", "f")
    assert len(e117.SENTINELS) == 4
    assert len(e117.SENTINELS) * len(e117.ARMS) == 12
    assert set(e117.SENTINEL_NODES) == set(e117.SENTINELS)


def test_e117_objective_has_only_two_arm_varying_fields():
    objectives = {arm: e117.objective_for_arm(arm) for arm in e117.ARMS}
    keys = set().union(*(objective for objective in objectives.values()))
    varying = {
        key for key in keys if len({objectives[arm].get(key) for arm in e117.ARMS}) > 1
    }
    assert varying == {
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF",
    }
    assert {
        arm: objectives[arm][
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY"
        ]
        for arm in e117.ARMS
    } == {"c": "1", "p": "0", "f": "0"}
    assert {
        arm: float(objectives[arm]["OAT_ZERO_SEMANTIC_SHANNON_COEF"])
        for arm in e117.ARMS
    } == {"c": 0.0, "p": 0.0, "f": 0.1}


def test_e117_keeps_proposal_compute_and_replay_identical():
    for arm in e117.ARMS:
        objective = e117.objective_for_arm(arm)
        assert (
            objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS"]
            == "1"
        )
        assert objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS"] == "1"
        assert (
            float(
                objective[
                    "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE"
                ]
            )
            == 1.2
        )
        assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
        assert float(objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) == 0.1
        assert (
            objective["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS"]
            == "0"
        )
        assert (
            objective["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY"]
            == "0"
        )
        assert (
            objective["OAT_ZERO_SEMANTIC_SHANNON_ALLOW_ZERO_COEFFICIENT_CONTROL"] == "1"
        )


def test_e117_snapshot_contract_contains_admission_boundary():
    for relative, needles in e117.SNAPSHOT_REQUIREMENTS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text


def test_e117r2_all_one_hour_amendment_is_scheduler_only():
    application = (
        ROOT / "ops/exp_scaling/" "apply_e117r2s1_all_one_hour_scheduler_amendment.py"
    ).read_text(encoding="utf-8")
    protocol = ROOT / (
        "paper/preregistration/" "e117r2s1_all_one_hour_scheduler_amendment_20260830.md"
    )
    required = (
        "COMPLETED_JOB_ID = 30970803",
        "PENDING_JOB_IDS = tuple(range(30970804, 30970815))",
        "AUDIT_JOB_ID = 30970815",
        'OLD_PARTITION = "cs"',
        'NEW_PARTITION = "all"',
        'OLD_ACCOUNT = "allcs"',
        'NEW_ACCOUNT = "mltheory"',
        'OLD_LIMIT = "02:00:00"',
        'NEW_LIMIT = "01:00:00"',
        'QOS = "none"',
        '"scientific_environment_changed": False',
        '"common_source_block_changed": False',
        '"outcomes_inspected_for_amendment": False',
        'run(["scontrol", "uhold", str(job_id)])',
    )
    assert protocol.is_file()
    assert all(contract in application for contract in required)


def test_e117r2_signal53_recovery_keeps_completed_reference_and_fresh_audit():
    application = (
        ROOT / "ops/exp_scaling/recover_e113r4_e117r2_signal53_wave.py"
    ).read_text(encoding="utf-8")
    required = (
        "E117_COMPLETED_ID = 30970803",
        "E117_PROCESS_FAILURE_IDS = (30970804,)",
        "E117_SIGNAL53_IDS = tuple(range(30970805, 30970815))",
        "E117_OLD_AUDIT_ID = 30970815",
        '"live_dependency_job_ids"',
        '"already_completed_job_ids": [E117_COMPLETED_ID]',
        '"restart_from_initial_model": True',
        '"Account=mltheory"',
        '"Partition=all"',
        '"TimeLimit=01:00:00"',
        '"scientific_environment_changed": False',
        '"source_snapshot_changed": False',
        '"efficacy_outcomes_inspected": False',
    )
    assert all(contract in application for contract in required)
