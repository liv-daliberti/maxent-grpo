from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "ops/exp_scaling/launch_e60_python_pilot.sh"
AUDITOR = ROOT / "ops/exp_scaling/audit_e60_python_pilot.py"
PROTOCOL = (
    ROOT
    / "paper/preregistration/e60_bootstrap_then_local_canonical_05b.md"
)


def test_e60_variant_uses_finite_global_bootstrap_then_local_replay():
    runner = (ROOT / "ops/run_experiment.sh").read_text()
    start = runner.index("  verified_first_bootstrap_local_canonical)")
    end = runner.index("\n    ;;", start)
    branch = runner[start:end]

    assert "OAT_ZERO_POLICY_ENTROPY_COEF=0.0" in branch
    assert "OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=1" in branch
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1" in branch
    assert (
        'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS='
        '"$ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS"'
    ) in branch
    assert "split_mass_balance_per_rollout" in branch
    assert "VARIANT_TAG=\"verified_first_bootstrap_local_canonical\"" in branch


def test_e60_phase_is_checkpointed_and_does_not_use_gold_or_eval_feedback():
    bank = (ROOT / "src/oat_drgrpo/online_canonical_bank.py").read_text()
    learner = (ROOT / "src/oat_drgrpo/learner/grpo.py").read_text()
    args = (ROOT / "src/oat_drgrpo/args.py").read_text()

    assert "online_canonical_replay_global_bootstrap_steps" in args
    assert '"global_replay_updates": self._global_replay_updates' in bank
    assert "global_replay_bootstrap_active" in bank
    assert ".replay_groups(" in learner
    assert ".scheduled_global_replay_groups(" in learner
    assert "canonical_replay_prompt_local_phase_active" in learner
    assert "canonical_replay_schedule_used_global" in learner
    assert "canonical_replay_schedule_used_prompt_local" in learner

    branch_start = learner.index(
        "args.online_canonical_replay_global_bootstrap_steps"
    )
    phase_logic = learner[branch_start : branch_start + 1300].lower()
    assert "eval" not in phase_logic
    assert "gold" not in phase_logic
    assert "target" not in phase_logic


def test_e60_alpha_controllers_remain_unprojected():
    replay = (ROOT / "src/oat_drgrpo/canonical_replay.py").read_text()
    semantic = (ROOT / "src/oat_drgrpo/semantic_shannon.py").read_text()
    runner = (ROOT / "ops/run_experiment.sh").read_text()

    assert '"canonical_replay_projection_active": 0.0' in replay
    assert '"semantic_open_set_projection_active": 0.0' in semantic
    assert "verified_first_bootstrap_local_canonical" in runner


def test_e60_python_pilot_is_five_pass_exact_seed_and_snapshot_bound():
    launcher = PILOT.read_text()
    protocol = PROTOCOL.read_text()

    assert "OAT_ZERO_TRAIN_SEEDS=9010" in launcher
    assert "OAT_ZERO_MAX_TRAIN=384" in launcher
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=5" in launcher
    assert "OAT_ZERO_NUM_PROMPT_EPOCH=5" in launcher
    assert "OAT_ZERO_EVAL_PROMPT_INTERVAL=96" in launcher
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_K=8" in launcher
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4" in launcher
    assert 'OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"' in launcher
    assert "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" in launcher
    assert "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" in launcher
    assert "single-seed sentinel is not the final result" in protocol


def test_e60_python_live_auditor_checks_phase_and_behavioral_evidence():
    auditor = AUDITOR.read_text()

    for expected in (
        "canonical_replay_global_bootstrap_steps",
        "canonical_replay_global_bootstrap_updates",
        "canonical_replay_global_bootstrap_active",
        "canonical_replay_prompt_local_phase_active",
        "canonical_replay_schedule_used_global",
        "canonical_replay_schedule_used_prompt_local",
        "canonical_replay_gold_support_feedback",
        "effective_advantage_negative_fraction",
        "effective_advantage_positive_fraction",
        "distinct_correct_modes_at_k",
        "any_correct_at_k",
        "mean_excess_multiplicity",
        "positive_multiplicity_points",
    ):
        assert expected in auditor
