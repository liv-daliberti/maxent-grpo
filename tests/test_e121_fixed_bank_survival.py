from pathlib import Path
import pytest

from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_bank_rejects_new_membership_and_count_updates():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=16,
    )
    common = dict(
        prompt_token_ids=[[7, 8]] * 2,
        task_rewards=[1.0, 1.0],
        active_mask=[1, 1],
        response_token_ids=[[10], [11]],
        num_samples=2,
    )
    bank.score_and_update(outcome_keys=["a", "b"], **common)
    before = bank.state_dict()

    bank.score_and_update(
        outcome_keys=["a", "new-after-freeze"],
        update_bank=False,
        **common,
    )
    after = bank.state_dict()

    assert after["counts"] == before["counts"]
    assert after["exemplars"] == before["exemplars"]
    assert bank.verified_replay_support([7, 8]) == ("a", "b")
    with pytest.raises(ValueError, match="update_bank must be a boolean"):
        bank.score_and_update(outcome_keys=["a", "b"], update_bank=1, **common)


def test_freeze_step_configuration_fails_closed_without_replay():
    source = (ROOT / "src/oat_drgrpo/args.py").read_text(encoding="utf-8")
    assert "online_canonical_replay_bank_freeze_step < 0" in source
    assert (
        "online_canonical_replay_bank_freeze_step > 0 and not online_canonical_replay"
        in source
    )
    assert (
        "online_canonical_replay_bank_freeze_step requires canonical replay" in source
    )


def test_e121_contract_is_forwarded_registered_and_non_pvl():
    args = (ROOT / "src/oat_drgrpo/args.py").read_text(encoding="utf-8")
    learner = (ROOT / "src/oat_drgrpo/learner/grpo.py").read_text(encoding="utf-8")
    train = (ROOT / "ops/train.sh").read_text(encoding="utf-8")
    launcher = (
        ROOT / "ops/exp_scaling/launch_e121_fixed_bank_survival_telemetry.py"
    ).read_text(encoding="utf-8")
    registry = (ROOT / "ops/exp_scaling/cohorts.py").read_text(encoding="utf-8")

    assert "online_canonical_replay_bank_freeze_step: int = 0" in args
    assert "--online-canonical-replay-bank-freeze-step" in train
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP" in train
    assert "canonical_replay_exemplar_mean_logprob_row_" in learner
    assert "canonical_replay_exemplar_sequence_logprob_row_" in learner
    assert "canonical_replay_prompt_fingerprint_row_" in learner
    assert "FREEZE_STEP = 384" in launcher
    assert '"--dependency={dependency}"' in launcher
    assert "len(runs) != 45" in launcher
    assert "def barrier_command(" in launcher
    assert "len(prerequisite_ids) > 10" in launcher
    assert '"dependency": "transitive_afterok_all_45"' in launcher
    assert 'NODES = ("node202", "node203", "node204")' in launcher
    assert "refuse_pvl(record" in launcher
    assert 'Cohort("e121"' in registry
    assert '"e121_fixed_bank_survival_telemetry_jobs.json"' in registry


@pytest.fixture
def survival_auditor():
    import importlib.util

    path = ROOT / "ops/exp_scaling/audit_e121_fixed_bank_survival.py"
    spec = importlib.util.spec_from_file_location("e121_survival_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def survival_records():
    """Two immutable prompt banks, three identities, three visits each."""
    from copy import deepcopy

    records = []
    for step in range(1, 8):
        prompt = 10 if step % 2 == 0 else 20
        outcomes = [101, 102] if prompt == 10 else [201]
        row = {
            "trainer/step": step,
            "trainer/global_step": step,
            "trainer/policy_sgd_step": step,
            "misc/global_step": step,
            "train/online_canonical_tracked_prompts": 2,
            "train/online_canonical_tracked_outcomes": 3,
            "train/online_canonical_bank_size_before_mean": 1,
            "train/online_canonical_bank_size_after_mean": 1,
        }
        replay = {
            "bank_freeze_step": 2,
            "bank_membership_frozen": float(step >= 2),
            "global_groups_per_step": 1,
            "global_scheduler_active": 1,
            "schedule_used_global": 1,
            "verified_likelihood_active": 1,
            "frequency_count_fresh_only": 1,
            "key_weighting_frequency": 0,
            "compute_only_configured": 0,
            "global_bootstrap_steps": 0,
            "frequency_count_from_replay": 0,
            "frequency_count_from_proposals": 0,
            "prompt_fingerprint_group_00": prompt,
            "membership_fingerprint_group_00": prompt + 1000,
            "available_modes": len(outcomes),
            "actuator_modes": len(outcomes),
            "banked_modes": len(outcomes),
            "alpha_used": 0.1,
            "objective_scale": 1 / 16,
            "backward_scale": 16,
            "applied_positive_gradient_max": 0,
            "applied_score_gradient_sum": -0.1 / 16,
            "applied_score_gradient_l2": 0.1,
            "realized_response_tokens": 4 * len(outcomes),
        }
        for i, outcome in enumerate(outcomes):
            replay.update(
                {
                    f"prompt_fingerprint_row_{i:02d}": prompt,
                    f"outcome_fingerprint_row_{i:02d}": outcome,
                    f"exemplar_mean_logprob_row_{i:02d}": -1 - step / 100,
                    f"exemplar_sequence_logprob_row_{i:02d}": 4 * (-1 - step / 100),
                    f"fresh_count_row_{i:02d}": 2,
                    f"target_weight_row_{i:02d}": 1,
                }
            )
        row.update(
            {"train/canonical_replay_" + key: value for key, value in replay.items()}
        )
        records.append(row)
    terminal = deepcopy(records[-1])
    terminal["trainer/step"] = 8
    records.append(terminal)
    return records


def test_survival_audit_counts_actual_updates_and_complete_population(
    survival_auditor, survival_records
):
    summary, histories = survival_auditor.audit_records(
        survival_records, freeze_step=2, target_steps=7
    )
    assert summary["identity_count"] == 3
    assert summary["identity_observation_count"] == 9
    assert summary["postfreeze_records"] == 6
    assert summary["min_visits"] == summary["max_visits"] == 3
    assert summary["terminal_carried_forward_records_excluded"] == 1
    assert [row[0] for row in histories[(20, 201)]] == [3, 5, 7]


@pytest.mark.parametrize("missing_population", ["prompts", "outcomes"])
def test_survival_audit_rejects_never_observed_frozen_identity(
    survival_auditor, survival_records, missing_population
):
    # Every observed identity still has three perfectly finite visits. An
    # observed-only auditor would incorrectly report success for this fixture.
    for row in survival_records:
        row["train/online_canonical_tracked_" + missing_population] += 1
    with pytest.raises(ValueError, match="missing or colliding"):
        survival_auditor.audit_records(survival_records, freeze_step=2, target_steps=7)


def test_survival_audit_rejects_missing_scheduled_update(
    survival_auditor, survival_records
):
    del survival_records[3]
    with pytest.raises(ValueError, match="missing/duplicate updates"):
        survival_auditor.audit_records(survival_records, freeze_step=2, target_steps=7)


def test_survival_audit_rejects_changed_terminal_carry_forward(
    survival_auditor, survival_records
):
    survival_records[-1]["train/canonical_replay_exemplar_mean_logprob_row_00"] -= 0.1
    with pytest.raises(ValueError, match="not a carried-forward duplicate"):
        survival_auditor.audit_records(survival_records, freeze_step=2, target_steps=7)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("bank_membership_frozen", 0, "freeze flag boundary"),
        ("membership_fingerprint_group_00", 999, "membership fingerprint changed"),
        ("fresh_count_row_00", 3, "fresh count changed"),
        ("exemplar_sequence_logprob_row_00", float("nan"), "nonfinite telemetry"),
        ("prompt_fingerprint_group_00", 99, "row/group prompt mismatch"),
    ],
)
def test_survival_audit_rejects_corrupted_postfreeze_telemetry(
    survival_auditor, survival_records, field, value, error
):
    survival_records[3]["train/canonical_replay_" + field] = value
    with pytest.raises(ValueError, match=error):
        survival_auditor.audit_records(survival_records, freeze_step=2, target_steps=7)


def test_survival_audit_rejects_unaligned_row_fields(
    survival_auditor, survival_records
):
    del survival_records[3]["train/canonical_replay_exemplar_mean_logprob_row_01"]
    with pytest.raises(ValueError, match="row alignment"):
        survival_auditor.audit_records(survival_records, freeze_step=2, target_steps=7)
