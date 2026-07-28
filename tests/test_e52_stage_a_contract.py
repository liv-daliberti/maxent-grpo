from pathlib import Path


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e52_stage_a.sh"
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e52_stage_a_execution_20260726.md"
)
WATCHER = ROOT / "ops/exp_scaling/watch_e52_sentinel.sh"
AUDITOR = ROOT / "ops/exp_scaling/audit_e52_stage_a.py"
STAGE_WATCHER = ROOT / "ops/exp_scaling/watch_e52_stage_a.sh"
STAGE_WATCHER_SLURM = ROOT / "ops/slurm/watch_e52_stage_a.slurm"


def test_stage_a_is_exact_fresh_three_seed_cartesian_product():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")

    for required in (
        "FROZEN BEFORE STAGE-A AUTHORIZATION",
        "exactly 27 fresh jobs",
        "seeds 43, 44, and 45",
        "No sentinel model checkpoint",
        "submitted held",
        "releasing the cohort atomically",
    ):
        assert required in protocol

    for required in (
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_ONLY_ARMS=grpo,maxent_inverse,maxent_inverse_canonical",
        "EXPECTED_JOBS_PER_DOMAIN=9",
        "EXPECTED_JOBS=27",
        "PROMPT_EPOCHS=50",
        "gce52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1",
        "cde52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs",
        "pye52_direct_inverse_entropy_canonical_05b_50ep_stage_a_v1_allcs",
        "for seed in (43, 44, 45)",
    ):
        assert required in launcher


def test_stage_a_replays_approval_before_any_submission():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    verifier_call = launcher.index('"$PYTHON_BIN" "$VERIFIER"')
    first_submission = launcher.index("submit_domain graph_coloring")
    assert verifier_call < first_submission
    for required in (
        "e52_sentinel_stage_a_approval.json",
        "--approval-sha256",
        "SOURCE_HASH=",
        "EXECUTION_HASH=",
        "e52_direct_inverse_entropy_${SOURCE_HASH}/src",
        "e52_direct_inverse_entropy_ops_${EXECUTION_HASH}/ops",
        'source "$OPS_ROOT/repo_env.sh"',
        '"resume_from_sentinel": False',
        '"sentinel_approval_sha256"',
        '"approval_verifier_sha256"',
        '"stage_a_auditor_sha256"',
        '"stage_a_watcher_sha256"',
        '"stage_a_watcher_slurm_sha256"',
    ):
        assert required in launcher


def test_stage_a_preserves_unbounded_target_free_treatment_and_held_audit():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean",
        "OAT_ZERO_MAXENT_ALPHA=0.000075",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION=1",
        "OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS=64",
        "OAT_ZERO_MAXENT_INVERSE_EMA_DECAY=0.90",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10",
        "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50",
        "OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0",
        "OAT_ZERO_SBATCH_HOLD=1",
        "Reason=JobHeldUser",
        'scontrol release "${job_ids[@]}"',
        'partial_job_ids=("${job_ids[@]}")',
        'partial_job_ids+=("${discovered_jobs[@]}")',
        'scancel "${partial_job_ids[@]}"',
    ):
        assert required in launcher
    assert "OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY" not in launcher
    assert "OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY" not in launcher


def test_live_watcher_can_only_autolaunch_from_terminal_approval_once():
    watcher = WATCHER.read_text(encoding="utf-8")

    for required in (
        "e52_sentinel_stage_a_approval.json",
        "e52_direct_inverse_entropy_canonical_05b_stage_a_v1_identity.json",
        "e52_stage_a_autolaunch.lock",
        'mkdir "$STAGE_A_LOCK"',
        "launch_e52_stage_a.sh stage_a",
        "Stage A launch failed closed",
    ):
        assert required in watcher
    assert watcher.index('[[ -f "$STAGE_A_APPROVAL"') < watcher.index(
        "launch_e52_stage_a.sh stage_a"
    )


def test_stage_a_audits_every_seed_and_the_seed_mean():
    auditor = AUDITOR.read_text(encoding="utf-8")

    for required in (
        "SEEDS = (43, 44, 45)",
        "seed_mean_behavioral_gate",
        "SENTINEL.audit_run",
        "SENTINEL.safety_gate",
        "SENTINEL.behavioral_gate",
        '"e52_stage_a_audit_v1"',
        '"stage_a_auditor_sha256"',
        '"stage_a_watcher_sha256"',
        '"stage_a_watcher_slurm_sha256"',
    ):
        assert required in auditor


def test_stage_a_releases_a_bound_four_day_audit_sidecar():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    watcher = STAGE_WATCHER.read_text(encoding="utf-8")
    slurm = STAGE_WATCHER_SLURM.read_text(encoding="utf-8")

    for required in (
        'sbatch --parsable --hold',
        "e52_stage_a_monitor",
        "TimeLimit=4-00:00:00",
        'scontrol release "${job_ids[@]}" "$monitor_job_id"',
        "cancelled held monitor",
    ):
        assert required in launcher
    assert "audit_e52_stage_a.py" in watcher
    assert "#SBATCH --time=4-00:00:00" in slurm
