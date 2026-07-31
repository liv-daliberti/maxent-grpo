import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "paper/preregistration/e44_online_growing_support_canonical_maxent_05b.md"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/launch_e44_online_growing_support_canonical_maxent_05b.sh"
)
ARGS = ROOT / "src/oat_drgrpo/args.py"
GRADER = ROOT / "src/oat_drgrpo/math_grader.py"
LEARNER = ROOT / "src/oat_drgrpo/learner/grpo.py"
RUN = ROOT / "src/oat_drgrpo/learner/run.py"
TRAIN = ROOT / "ops/train.sh"
RUN_EXPERIMENT = ROOT / "ops/run_experiment.sh"
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"
PLOT = ROOT / "ops/exp_scaling/plot_divergence.py"
REFRESH = ROOT / "ops/exp_scaling/refresh_latest_freeform_05b.py"
MONITOR = ROOT / "ops/exp_scaling/monitor_campaign.py"
MAKEFILE = ROOT / "Makefile"


def test_e44_ogs_protocol_freezes_executable_admission_not_math_text():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "**Status: FROZEN BEFORE LAUNCH (2026-07-23).**",
        "E44-OGS",
        "validated_modebench_outcome_key(y, reference)",
        "A_i_actor = A_i_task-DrGRPO + A_i_ent + A_i_new",
        "`alpha=0.10`",
        "`beta=0.50`",
        "Final-answer equivalence is not a proof or strategy verifier",
        "No gold catalogue of valid outcomes populates the bank",
        "`grpo`: ordinary Dr.GRPO",
        "`online_canonical_maxent`",
        "ten complete prompt-pool passes",
    ):
        assert literal in text


def test_e44_ogs_launcher_is_fresh_twelve_job_matched_cohort():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "EXPECTED_JOBS_PER_TASK=6",
        "EXPECTED_JOBS=12",
        "GRAPH_PREFIX=gce44_ogs_canonical_maxent_05b_v1",
        "COUNTDOWN_PREFIX=cde44_ogs_canonical_maxent_05b_v1",
        "export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_maxent",
        "export OAT_ZERO_INCLUDE_ONLINE_CANONICAL_MAXENT_ARM=1",
        "export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10",
        "export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50",
        "export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome",
        "submit_task graph_coloring",
        "submit_task countdown",
        "scontrol release \"${job_ids[@]}\"",
    ):
        assert literal in text


def test_e44_ogs_execution_stack_binds_validator_bank_and_resume():
    args = ARGS.read_text(encoding="utf-8")
    grader = GRADER.read_text(encoding="utf-8")
    learner = LEARNER.read_text(encoding="utf-8")
    run = RUN.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")
    run_experiment = RUN_EXPERIMENT.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")

    assert 'online_canonical_key_mode: Literal[' in args
    assert '"modebench_outcome"' in args
    assert "online canonical banks require executable ModeBench" in args
    assert "def validated_modebench_outcome_key(" in grader
    assert "key is derived from the exact AST object" in grader
    assert "_verify_graph_coloring_colors(colors, spec)" in grader
    assert "validated_modebench_outcome_key(" in learner
    assert "fail-closed intersection excludes them" in learner
    assert "validator_positive_actor_negative_rows" in learner
    assert "actor_positive_validator_negative_rows" in learner
    assert (
        "online canonical bank attempted to admit validator-negative"
        not in learner
    )
    assert "online_canonical_advantage_applied_after_task_centering" in learner
    assert "online_canonical_bank_state" in run
    assert "--online-canonical-bank-alpha" in train
    assert "online_canonical_maxent)" in run_experiment
    assert (
        "submit_arm online_canonical_maxent online_canonical_maxent"
        in submitter
    )


def test_verified_bank_live_figure_and_tracker_follow_current_cohort():
    # Long run-stamp prefixes wrap across lines in the plotting source, so
    # adjacent string literals are joined before searching.
    plot = re.sub(
        r'"\s*\n\s*"', "", PLOT.read_text(encoding="utf-8")
    )
    refresh = REFRESH.read_text(encoding="utf-8")
    monitor = MONITOR.read_text(encoding="utf-8")
    makefile = MAKEFILE.read_text(encoding="utf-8")

    assert "def render_online_canonical_maxent_05b()" in plot
    assert "ONLINE_CANONICAL_DIAGNOSTIC_METRICS" in plot
    assert "ONLINE_CANONICAL_ADVANTAGE_COMPONENTS" in plot
    assert "online_canonical_exploration_to_task_rms_ratio" in plot
    assert "online_canonical_mean_support_per_prompt" in plot
    assert "cumulative verified discoveries" in plot
    assert "new verified outcomes / batch" not in plot
    assert "online_canonical_bank_size_after_mean" not in plot
    # Curve paths are derived from the frontier prefixes rather than spelled
    # out, so the contract is the construction plus the current cohort.
    assert 'f"var/artifacts/{countdown_prefix}_scaling_curve.json"' in plot
    assert (
        "cde58_global_verified_replay_canonical_05b_50ep_sentinel_allcs"
        in plot
    )
    assert (
        "gce58_global_verified_replay_canonical_05b_50ep_sentinel" in plot
    )
    assert "e58_global_verified_replay_canonical_05b_live" in plot
    assert "--e44-ogs-only" in refresh
    assert "CURRENT_CANONICAL_CELLS" in refresh
    # The alias now takes an artifact-root argument across several lines.
    assert "def e44_ogs_specs(" in monitor
    assert "e44_ogs_only=True" not in monitor
    assert "--e44-ogs-only" in monitor
    assert (
        "monitor_campaign.py --current-canonical-only "
        "--figure-refresh-seconds 60"
        in makefile
    )
