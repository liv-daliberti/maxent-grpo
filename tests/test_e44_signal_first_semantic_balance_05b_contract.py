from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e44_signal_first_semantic_balance_05b.sh"
)
PROTOCOL = ROOT / "paper/preregistration/e44_signal_first_semantic_balance_05b.md"
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"
TRAIN = ROOT / "ops/train.sh"
ARGS = ROOT / "src/oat_drgrpo/args.py"
LEARNER = ROOT / "src/oat_drgrpo/learner/grpo.py"
RUN_EXPERIMENT = ROOT / "ops/run_experiment.sh"


def test_e44_protocol_freezes_two_channel_combination_before_launch():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "**Status: FROZEN BEFORE LAUNCH (2026-07-23).**",
        "`A_i_task = r_i - mean_j(r_j)`",
        "`U_i_task = A_i_task * T_i / T_max`",
        "`w_i = G * softmax_i(U_i_task / 0.05)`",
        "`A_i_actor = A_i_task + A_i_sem`",
        "xDr weights are computed only from the task advantage",
        "wrong, unparseable, and loss-inactive rows receive zero",
        "`signal_first_semantic_balance`",
        "`gce44_signal_first_semantic_balance_05b_v2`",
        "`cde44_signal_first_semantic_balance_05b_v2`",
        "`30072531--30072536`",
        "produced no training or",
        "MATH is not",
        "included: E39 showed no final-answer diversity effect",
        "ten complete prompt-pool passes",
    ):
        assert literal in text


def test_e44_launcher_is_six_job_modebench_only_frozen_cohort():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "EXPECTED_JOBS_PER_TASK=3",
        "EXPECTED_JOBS=6",
        "GRAPH_PREFIX=gce44_signal_first_semantic_balance_05b_v2",
        "COUNTDOWN_PREFIX=cde44_signal_first_semantic_balance_05b_v2",
        "export OAT_ZERO_ONLY_ARMS=signal_first_semantic_balance",
        "export OAT_ZERO_INCLUDE_SIGNAL_FIRST_SEMANTIC_BALANCE_ARM=1",
        "export OAT_ZERO_INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM=0",
        "export OAT_ZERO_SIGNAL_FIRST_XDR_TAU=0.05",
        "export OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS=1",
        "submit_task graph_coloring",
        "submit_task countdown",
        "'OAT_ZERO_VARIANT=signal_first_semantic_balance'",
        "'OAT_ZERO_XDR_TAU=0.05'",
        "'OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS=1'",
        "scontrol release \"${job_ids[@]}\"",
    ):
        assert literal in text
    assert "submit_task math" not in text
    assert "MATH_PREFIX=" not in text


def test_shared_stack_pins_combined_variant_and_false_paths():
    submitter = SUBMITTER.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")
    args = ARGS.read_text(encoding="utf-8")
    learner = LEARNER.read_text(encoding="utf-8")
    run_experiment = RUN_EXPERIMENT.read_text(encoding="utf-8")

    for literal in (
        'INCLUDE_SIGNAL_FIRST_SEMANTIC_BALANCE_ARM="${OAT_ZERO_INCLUDE_SIGNAL_FIRST_SEMANTIC_BALANCE_ARM:-0}"',
        'SIGNAL_FIRST_XDR_TAU="${OAT_ZERO_SIGNAL_FIRST_XDR_TAU:-0.05}"',
        'if [[ "$variant" == "signal_first_semantic_balance" ]]',
        'OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS=1',
        "submit_arm signal_first_semantic_balance signal_first_semantic_balance",
    ):
        assert literal in submitter
    for literal in (
        'XDR_TASK_ADVANTAGE_WEIGHTS="${OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS:-0}"',
        "--xdr-task-advantage-weights",
        "--no-xdr-task-advantage-weights",
    ):
        assert literal in train
    for literal in (
        "xdr_task_advantage_weights: bool = False",
        "xdr_task_advantage_weights requires the success-conditioned",
    ):
        assert literal in args
    for literal in (
        "task_advantages_for_xdr = advantages.detach()",
        "xdr_weight_advantages = (",
        "xdr_task_advantage_weights_active",
    ):
        assert literal in learner
    for literal in (
        "signal_first_semantic_balance)",
        "export OAT_ZERO_XDR_TAU=",
        "export OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS=1",
        'VARIANT_TAG="signal_first_semantic_balance"',
    ):
        assert literal in run_experiment


def test_e44_identity_records_task_only_weights_and_combined_actor_advantage():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        '"schema": "e44_signal_first_semantic_balance_05b_v2"',
        '"new_arms": ["signal_first_semantic_balance"]',
        '"xdr_tau": 0.05',
        '"xdr_weight_advantage": "task_only_before_semantic_augmentation"',
        '"xdr_task_advantage_weights": True',
        '"actor_advantage": "task_plus_success_conditioned_signed_semantic"',
    ):
        assert literal in text
