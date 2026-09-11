from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e38_semantic_shannon_05b.sh"
PROTOCOL = ROOT / "paper/preregistration/e38_semantic_shannon_05b.md"
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"
RUNNER = ROOT / "ops/run_experiment.sh"
TRAIN = ROOT / "ops/train.sh"


def test_e38_is_a_treatment_only_matched_ten_pass_extension():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "gce38_semantic_shannon_05b_v1",
        "cde38_semantic_shannon_05b_v1",
        "OAT_ZERO_ONLY_ARMS=semantic_shannon",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM=0",
        "OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM=1",
        "OAT_ZERO_OUTCOME_COLLISION_COEF=0",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10",
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0",
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=10",
        "OAT_ZERO_NUM_PROMPT_EPOCH=10",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K=8",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=370100",
        "OAT_ZERO_DIAYN_NUM_OPTIONS=0",
        "OAT_ZERO_CANONICAL_ACTION_TASK=none",
        "OAT_ZERO_SBATCH_HOLD=1",
        "EXPECTED_JOBS_PER_TASK=3",
    ):
        assert literal in text


def test_e38_freezes_source_execution_comparator_and_resume_identity():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        'cd "$tree"',
        "find . -type f -print0",
        "e38_semantic_shannon_05b_v1_identity.json",
        '"source_hash": sys.argv[4]',
        '"execution_surface_hash": sys.argv[5]',
        '"e37_comparators": {',
        '"identity_sha256": sys.argv[6]',
        'identity.get("schema") != "e37_outcome_collision_05b_v1"',
        'for arm in ("grpo", "outcome_collision")',
        '"step_zero_pairing_to_e37": "exact"',
        '"semantic_history_checkpointed": True',
        "Current source lacks the checkpointed E38 semantic-Shannon mechanism",
        "OAT_ZERO_MAX_RESUME_NUM=2",
        "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0",
        "OAT_ZERO_EXPORT_STEPS=0",
        "cleanup_partial_cohort()",
        'trap cleanup_partial_cohort EXIT',
        'scancel "${cleanup_ids[@]}"',
        'mv "$artifact" "${artifact}.failed_${failure_stamp}"',
        "COHORT_RELEASED=1",
        'scontrol update JobId="$job_id" Partition="$OAT_ZERO_TRAIN_PARTITION"',
        'scontrol release "${job_ids[@]}"',
    ):
        assert literal in text


def test_e38_held_job_audit_pins_only_semantic_shannon():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "'OAT_ZERO_VARIANT=semantic_shannon'",
        "'OAT_ZERO_OUTCOME_COLLISION_COEF=0.0'",
        "'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10'",
        "'OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0'",
        "'OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0'",
        "'OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed'",
        "'OAT_ZERO_DIAYN_NUM_OPTIONS=0'",
        "'OAT_ZERO_DIAYN_MI_BETA=0.0'",
    ):
        assert literal in text


def test_shared_shell_stack_pins_shannon_treatment_and_other_arms():
    submitter = SUBMITTER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")

    for literal in (
        'INCLUDE_SEMANTIC_SHANNON_ARM="${OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM:-0}"',
        'SEMANTIC_SHANNON_COEF="${OAT_ZERO_SEMANTIC_SHANNON_COEF:-0.1}"',
        'if [[ "$variant" == "semantic_shannon" ]]',
        'export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_COEF=${SEMANTIC_SHANNON_COEF}"',
        'export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0"',
        "submit_arm semantic_shannon semantic_shannon",
    ):
        assert literal in submitter

    for literal in (
        'SEMANTIC_SHANNON_COEF="${OAT_ZERO_SEMANTIC_SHANNON_COEF:-0.0}"',
        "export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "semantic_shannon)",
        'export OAT_ZERO_SEMANTIC_SHANNON_COEF="$SEMANTIC_SHANNON_COEF"',
    ):
        assert literal in runner

    for literal in (
        '--semantic-shannon-coef "$SEMANTIC_SHANNON_COEF"',
        '--semantic-shannon-surprisal-clip "$SEMANTIC_SHANNON_SURPRISAL_CLIP"',
        '--semantic-shannon-pseudocount "$SEMANTIC_SHANNON_PSEUDOCOUNT"',
        "Frozen source lacks semantic-Shannon support",
    ):
        assert literal in train


def test_e38_protocol_freezes_predictive_shannon_experiment():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "**Status: FROZEN BEFORE LAUNCH",
        "treatment-only extension",
        "one unseen",
        "`p_hat_i = 1 / D_i`",
        "`b_i = 0.10 * (s_i / 5.0 - 1)`",
        "`R_correct >= 0.90 > 0 >= R_incorrect`",
        "SHA256 digest of the unpadded prompt token IDs",
        "Training seeds: `43, 44, 45`",
        "ten complete prompt passes",
        "Draw seeds are `370100, 370101, 370102, 370103`",
        "every E38 step-zero evaluation metric and raw",
        "There is no latent-conditioned evaluation",
    ):
        assert literal in text
