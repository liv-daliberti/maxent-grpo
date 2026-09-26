from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/launch_e39_math12k_384_semantic_entropy_05b.sh"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/e39_math12k_384_semantic_entropy_05b.md"
)
RUNNER = ROOT / "ops/run_experiment.sh"
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"
LEARNER_RUN = ROOT / "src/oat_drgrpo/learner/run.py"


def test_e39_is_a_fresh_matched_three_arm_math12k_384_cohort():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "mte39_math12k_384_semantic_entropy_05b_v1",
        "var/data/math12k_384_math500",
        "materialize_e39_math12k_384.py",
        "OAT_ZERO_COMPARATIVE_TASK=math",
        "OAT_ZERO_ONLY_ARMS=grpo,outcome_collision,semantic_shannon",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM=1",
        "OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM=1",
        "OAT_ZERO_OUTCOME_COLLISION_COEF=0.10",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10",
        "OAT_ZERO_MAX_TRAIN=384",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=10",
        "OAT_ZERO_NUM_PROMPT_EPOCH=10",
        "EXPECTED_JOBS=9",
        "OAT_ZERO_SBATCH_HOLD=1",
        'for arm in ("grpo", "outcome_collision", "semantic_shannon")',
        'scontrol release "${job_ids[@]}"',
    ):
        assert literal in text


def test_e39_uses_authentic_math_grading_lengths_and_sparse_full_eval():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "OAT_ZERO_PROMPT_TEMPLATE=qwen_math",
        "OAT_ZERO_TEST_SPLIT=math",
        "OAT_ZERO_VERIFIER_VERSION=math_verify",
        "OAT_ZERO_PROMPT_MAX_LENGTH=1024",
        "OAT_ZERO_GENERATE_MAX_LENGTH=1024",
        "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024",
        "OAT_ZERO_MAX_MODEL_LEN=2048",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K=8",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1",
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=390100",
        "OAT_ZERO_ALLOW_SPARSE_EVAL=1",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=768",
        "OAT_ZERO_EVAL_STEPS=768",
        "OAT_ZERO_SAVE_STEPS=384",
        "OAT_ZERO_RESUME_STEPS=384",
        "OAT_ZERO_MAX_RESUME_NUM=2",
    ):
        assert literal in text


def test_e39_skips_the_duplicate_terminal_eval_of_unchanged_weights():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    learner = LEARNER_RUN.read_text(encoding="utf-8")
    assert '"count": 6' in launcher
    assert '"duplicate_terminal_policy_evaluation": "skip"' in launcher
    assert "_last_evaluated_global_step" in launcher
    assert "There are exactly six evaluations per run" in protocol
    assert "Skipping duplicate terminal evaluation" in learner


def test_e39_held_audit_separates_all_method_knobs():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "'OAT_ZERO_VARIANT=grpo'",
        "'OAT_ZERO_VARIANT=outcome_collision'",
        "'OAT_ZERO_VARIANT=semantic_shannon'",
        "'OAT_ZERO_OUTCOME_COLLISION_COEF=0.0'",
        "'OAT_ZERO_OUTCOME_COLLISION_COEF=0.10'",
        "'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0'",
        "'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10'",
        "'OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0'",
        "'OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0'",
        "cleanup_partial_cohort()",
        'scancel "${cleanup_ids[@]}"',
        'mv "$artifact" "${artifact}.failed_${failure_stamp}"',
    ):
        assert literal in text


def test_e39_held_audit_attests_all_frozen_shared_fields():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        '"RUN_STAMP=${run_stamp}"',
        '"OAT_ZERO_SEED=${seed}"',
        "'OAT_ZERO_MODEL=qwen2.5-0.5b-instruct'",
        '"OAT_ZERO_PRETRAIN=${MODEL_ROOT}"',
        '"OAT_ZERO_DATA_ROOT=${DATA_ROOT}"',
        "'OAT_ZERO_REQUIRE_EXISTING_DATA=1'",
        "'OAT_ZERO_MAX_QUERIES=100000000'",
        "'OAT_ZERO_TRAIN_BATCH_SIZE=16'",
        "'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1'",
        "'OAT_ZERO_ROLLOUT_BATCH_SIZE=1'",
        "'OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1'",
        "'OAT_ZERO_N_GPU=1'",
        "'OAT_ZERO_NUM_GPUS_PER_ACTOR=1'",
        "'OAT_ZERO_LEARNING_RATE=0.0000002'",
        "'OAT_ZERO_NUM_PPO_EPOCHS=1'",
        "'OAT_ZERO_MAX_NORM=1'",
        "'OAT_ZERO_BETA=0'",
        "'OAT_ZERO_IGNORE_NO_EOS=0'",
        "'OAT_ZERO_INPUT_KEY=problem'",
        "'OAT_ZERO_OUTPUT_KEY=answer'",
        "'OAT_ZERO_EVAL_INPUT_KEY=problem'",
        "'OAT_ZERO_EVAL_OUTPUT_KEY=answer'",
        "'OAT_ZERO_TEMPERATURE=1'",
        "'OAT_ZERO_TOP_P=1'",
        "'OAT_ZERO_EVAL_TEMPERATURE=0'",
        "'OAT_ZERO_EVAL_BATCH_SIZE=64'",
        "'OAT_ZERO_SYNC_PARAMS_EVERY=1'",
        "'OAT_ZERO_EVAL_STEPS=768'",
        "'OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1'",
        "'OAT_ZERO_USE_WB=0'",
        "'OAT_ZERO_CANONICAL_ACTION_TASK=none'",
        "'OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0'",
        "'OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0'",
        "'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0'",
        "'OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=0'",
        "'OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=0'",
        "'OAT_ZERO_ZERO_STAGE=2'",
        "'OAT_ZERO_VLLM_GPU_RATIO=0.25'",
        "'OAT_ZERO_ENABLE_FLASH_ATTN=0'",
        "'OAT_ZERO_ADAM_OFFLOAD=0'",
        "'OAT_ZERO_ACTIVATION_OFFLOADING=0'",
        "'OAT_ZERO_COLLOCATE=1'",
        "'OAT_ZERO_VLLM_SLEEP=1'",
        "'OAT_ZERO_VLLM_SLEEP_LEVEL=1'",
        "'VLLM_USE_V1=0'",
        "'HF_HUB_OFFLINE=1'",
        "'TRANSFORMERS_OFFLINE=1'",
        "'OAT_ZERO_RND_SEED=0'",
        "'OAT_ZERO_XDR_MODE_ADAPTIVE=0'",
        "'OAT_ZERO_XDR_TAU=inf'",
        "'OAT_ZERO_XDR_TAU_CONTROL_TARGET_RATIO=0.0'",
        "'OAT_ZERO_XDR_SAC_DUAL_TARGET_RATIO=0.0'",
        "'OAT_ZERO_MAXENT_ALPHA=0.0'",
        "'OAT_ZERO_MAXENT_OBJECTIVE=sequence'",
        "'OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO=0.0'",
        "'OAT_ZERO_MAXENT_DUAL_TARGET_RATIO=0.0'",
        "'OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY=0.0'",
        "'OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=0.0'",
        "'OAT_ZERO_XDR_TAU_CONTROL_RATIO=0.0'",
        "'OAT_ZERO_XDR_SAC_DUAL_RATIO=0.0'",
        "'OAT_ZERO_MAXENT_LENGTH_TARGET=0.0'",
        "'OAT_ZERO_POLICY_ENTROPY_COEF=0.0'",
        "'OAT_ZERO_SEED_ENTROPY_ALPHA=0.0'",
        "'OAT_ZERO_SAVE_CKPT=1'",
        "'OAT_ZERO_SAVE_STEPS=384'",
        "'OAT_ZERO_SAVE_FROM=384'",
        "'OAT_ZERO_MAX_SAVE_NUM=2'",
        "'OAT_ZERO_AUTO_RESUME=1'",
        "'OAT_ZERO_WATCHDOG_REQUEUE=1'",
        "'OAT_ZERO_WATCHDOG_MAX_RESTARTS=6'",
        "'OAT_ZERO_RESUME_STEPS=384'",
        "'OAT_ZERO_RESUME_FROM=384'",
        "'OAT_ZERO_MAX_RESUME_NUM=2'",
        "'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0'",
        "'OAT_ZERO_EXPORT_STEPS=0'",
        "'OAT_ZERO_EXPORT_FROM=0'",
        "'OAT_ZERO_MAX_EXPORT_NUM=1'",
        '"OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}"',
        '"OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}"',
        '"OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"',
        '"Partition=${OAT_ZERO_TRAIN_PARTITION}"',
        '"Account=${OAT_ZERO_TRAIN_ACCOUNT}"',
        '"--nodelist=${OAT_ZERO_TRAIN_NODELIST}"',
        '"gres/gpu:a5000=1"',
        '"NumNodes=1"',
        '"NumTasks=1"',
        '"NumCPUs=${OAT_ZERO_TRAIN_CPUS_PER_TASK}"',
        '"MinMemoryNode=${OAT_ZERO_TRAIN_MEMORY}"',
        '"--time=${OAT_ZERO_TRAIN_TIME_LIMIT}"',
    ):
        assert literal in text


def test_sparse_eval_is_explicitly_forwarded_and_fail_closed():
    runner = RUNNER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    for literal in (
        'ALLOW_SPARSE_EVAL="${OAT_ZERO_ALLOW_SPARSE_EVAL:-0}"',
        "OAT_ZERO_ALLOW_SPARSE_EVAL must be 0 or 1",
        "--allow-sparse-requested-interval",
        "requires OAT_ZERO_EVAL_PROMPT_INTERVAL",
        'EVAL_CADENCE_POLICY="explicit_sparse"',
    ):
        assert literal in runner
    assert (
        'export_vars+=",OAT_ZERO_ALLOW_SPARSE_EVAL='
        '${OAT_ZERO_ALLOW_SPARSE_EVAL:-0}"'
    ) in submitter
    assert (
        'export_vars+=",OAT_ZERO_EVAL_STEPS=${OAT_ZERO_EVAL_STEPS:-64}"'
        in submitter
    )


def test_e39_protocol_discloses_single_answer_interpretation():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "**Status: FROZEN BEFORE LAUNCH",
        "single-answer stress test",
        "mostly errors",
        "complete symbolic quotient",
        "rows 0 through 383",
        "all 500 rows",
        "Training seeds: `43, 44, 45`",
        "Ten complete passes",
        "passes 2, 4, 6, 8, and 10",
        "`390100`",
        "not interpreted or",
        "plotted for this row",
        "final-answer outcomes/errors",
        "than proof strategies",
    ):
        assert literal in text
