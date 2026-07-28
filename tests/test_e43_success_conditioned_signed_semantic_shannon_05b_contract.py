from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/launch_e43_success_conditioned_signed_semantic_shannon_05b.sh"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/e43_success_conditioned_signed_semantic_shannon_05b.md"
)
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"
RUNNER = ROOT / "ops/run_experiment.sh"
TRAIN = ROOT / "ops/train.sh"


def test_e43_is_treatment_only_three_domain_nine_run_cohort():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "cde43_success_conditioned_signed_semantic_shannon_05b_v1",
        "gce43_success_conditioned_signed_semantic_shannon_05b_v1",
        "mte43_math12k_384_success_conditioned_signed_semantic_shannon_05b_v1",
        "e43_success_conditioned_signed_semantic_shannon_05b_v1_identity.json",
        "OAT_ZERO_ONLY_ARMS=success_conditioned_signed_semantic_shannon",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM=0",
        "OAT_ZERO_INCLUDE_OUTCOME_COLLISION_OUTSIDE_CENTERING_ARM=0",
        "OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM=0",
        "OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM=0",
        "OAT_ZERO_INCLUDE_QUALITY_GATED_SEMANTIC_NOVELTY_ARM=0",
        "OAT_ZERO_INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM=1",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10",
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0",
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP=0.05",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=10",
        "OAT_ZERO_NUM_PROMPT_EPOCH=10",
        "EXPECTED_JOBS_PER_TASK=3",
        "EXPECTED_JOBS=9",
        "OAT_ZERO_SBATCH_HOLD=1",
        '("success_conditioned_signed_semantic_shannon", str(seed))',
    ):
        assert literal in text


def test_e43_audits_and_hashes_frozen_e37_e38_e39_e41_e42_comparators():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "e37_outcome_collision_05b_v1_identity.json",
        "gce37_outcome_collision_05b_v1_comparative_jobs.tsv",
        "cde37_outcome_collision_05b_v1_comparative_jobs.tsv",
        "e38_semantic_shannon_05b_v1_identity.json",
        "gce38_semantic_shannon_05b_v1_comparative_jobs.tsv",
        "cde38_semantic_shannon_05b_v1_comparative_jobs.tsv",
        "mte39_math12k_384_semantic_entropy_05b_v1_identity.json",
        "mte39_math12k_384_semantic_entropy_05b_v1_comparative_jobs.tsv",
        "e41_semantic_shannon_advantage_05b_v1_identity.json",
        "gce41_semantic_shannon_advantage_05b_v1_comparative_jobs.tsv",
        "cde41_semantic_shannon_advantage_05b_v1_comparative_jobs.tsv",
        "mte41_math12k_384_semantic_shannon_advantage_05b_v1_comparative_jobs.tsv",
        "e42_quality_gated_semantic_novelty_05b_v1_identity.json",
        "gce42_quality_gated_semantic_novelty_05b_v1_comparative_jobs.tsv",
        "cde42_quality_gated_semantic_novelty_05b_v1_comparative_jobs.tsv",
        "mte42_math12k_384_quality_gated_semantic_novelty_05b_v1_comparative_jobs.tsv",
        'e37.get("schema") != "e37_outcome_collision_05b_v1"',
        'e38.get("schema") != "e38_semantic_shannon_05b_v1"',
        'e39.get("schema") != "e39_math12k_384_semantic_entropy_05b_v1"',
        'e41.get("schema") != "e41_semantic_shannon_advantage_05b_v1"',
        'e42.get("schema") != "e42_quality_gated_semantic_novelty_05b_v1"',
        '"modebench_e37": {',
        '"modebench_e38": {',
        '"math": {',
        '"parent_e41": {',
        '"parent_e42": {',
        '"identity_sha256": sys.argv[6]',
        '"identity_sha256": sys.argv[9]',
        '"identity_sha256": sys.argv[12]',
        '"materialization_manifest_sha256": sys.argv[14]',
        '"identity_sha256": sys.argv[15]',
        '"identity_sha256": sys.argv[19]',
    ):
        assert literal in text


def test_e43_freezes_source_execution_data_and_held_release_gate():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        'cd "$tree"',
        "find . -type f -print0",
        "e43_success_conditioned_signed_semantic_shannon_05b_${SOURCE_HASH}",
        "e43_success_conditioned_signed_semantic_shannon_05b_ops_${EXECUTION_HASH}",
        "materialize_e39_math12k_384.py",
        "--audit-only",
        '"source_hash": sys.argv[4]',
        '"execution_surface_hash": sys.argv[5]',
        '"semantic_history_checkpointed": True',
        "cleanup_partial_cohort()",
        "trap cleanup_partial_cohort EXIT",
        'scancel "${cleanup_ids[@]}"',
        'mv "$artifact" "${artifact}.failed_${failure_stamp}"',
        'scontrol update JobId="$job_id" Partition="$TRAIN_PARTITION"',
        'scontrol release "${job_ids[@]}"',
        "COHORT_RELEASED=1",
    ):
        assert literal in text


def test_e43_matches_e41_e42_modebench_and_math_evaluation_contracts():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed",
        "OAT_ZERO_TEST_SPLIT=multi_answer",
        "OAT_ZERO_VERIFIER_VERSION=fast",
        "OAT_ZERO_PROMPT_MAX_LENGTH=256",
        "OAT_ZERO_GENERATE_MAX_LENGTH=192",
        "OAT_ZERO_MAX_MODEL_LEN=512",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=370100",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=48",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=96",
        "OAT_ZERO_PROMPT_TEMPLATE=qwen_math",
        "OAT_ZERO_TEST_SPLIT=math",
        "OAT_ZERO_VERIFIER_VERSION=math_verify",
        "OAT_ZERO_PROMPT_MAX_LENGTH=1024",
        "OAT_ZERO_GENERATE_MAX_LENGTH=1024",
        "OAT_ZERO_MAX_MODEL_LEN=2048",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1",
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=390100",
        "OAT_ZERO_ALLOW_SPARSE_EVAL=1",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=768",
        "OAT_ZERO_EVAL_STEPS=768",
        'MODEBENCH_CPUS="${OAT_ZERO_E43_MODEBENCH_TRAIN_CPUS_PER_TASK:-4}"',
        'MODEBENCH_MEMORY="${OAT_ZERO_E43_MODEBENCH_TRAIN_MEMORY:-32G}"',
        'MATH_CPUS="${OAT_ZERO_E43_MATH_TRAIN_CPUS_PER_TASK:-8}"',
        'MATH_MEMORY="${OAT_ZERO_E43_MATH_TRAIN_MEMORY:-64G}"',
    ):
        assert literal in text


def test_e43_held_audit_pins_signed_mode_and_excludes_other_mechanisms():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "'OAT_ZERO_VARIANT=success_conditioned_signed_semantic_shannon'",
        "'OAT_ZERO_OUTCOME_COLLISION_COEF=0.0'",
        "'OAT_ZERO_OUTCOME_COLLISION_OUTSIDE_CENTERING=0'",
        "'OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10'",
        "'OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0'",
        "'OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0'",
        "'OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1'",
        "'OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=0'",
        "'OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_CAP=0.05'",
        "'OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1'",
        "'OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP=0.05'",
        "'OAT_ZERO_DIAYN_NUM_OPTIONS=0'",
        "'OAT_ZERO_DIAYN_MI_BETA=0.0'",
        "'OAT_ZERO_MAXENT_ALPHA=0.0'",
        "'OAT_ZERO_POLICY_ENTROPY_COEF=0.0'",
        "'OAT_ZERO_SEED_ENTROPY_ALPHA=0.0'",
        "'OAT_ZERO_XDR_TAU=inf'",
        "'OAT_ZERO_CANONICAL_ACTION_TASK=none'",
        "'OAT_ZERO_SAVE_CKPT=1'",
        "'OAT_ZERO_MAX_RESUME_NUM=2'",
        "'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0'",
    ):
        assert literal in text


def test_e43_identity_records_success_only_support_and_signed_transform():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        '"quality_gated_advantage": False',
        '"success_conditioned_signed_advantage": True',
        '"success_conditioned_signed_cap": 0.05',
        '"semantic_shannon_tracker_v3_success_conditioned_signed"',
        '"raw_centering": "detached_row_predictive_distribution_expectation"',
        '"eligibility": "active_and_positive_task_reward_and_parseable"',
        '"effective_transform": (',
        '"eligible*max(-0.05,min(raw_advantage,0.05))"',
        '"application": "centered_task_advantage_plus_capped_signed_novelty"',
        '"task_reward_sent_to_centering": "unchanged"',
        '"predictive_support": "eligible_successes_only_plus_unseen_bucket"',
        '"history_rows": "eligible_successes_only"',
        '"invalid_outcome": "ineligible_and_excluded"',
    ):
        assert literal in text


def test_shared_shell_stack_has_e43_variant_and_explicit_false_paths():
    submitter = SUBMITTER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")

    for literal in (
        'INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM="${OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ADVANTAGE_ARM:-0}"',
        'INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM="${OAT_ZERO_INCLUDE_SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON_ARM:-0}"',
        'if [[ "$variant" == "semantic_shannon" ]]',
        'elif [[ "$variant" == "semantic_shannon_advantage" ]]',
        'elif [[ "$variant" == "success_conditioned_signed_semantic_shannon" ]]',
        'if [[ "$variant" == "success_conditioned_signed_semantic_shannon" ]]',
        'export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1"',
        'export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0"',
        'export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1"',
        'export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=0"',
        'export_vars+=",OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP=${SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP}"',
        "submit_arm success_conditioned_signed_semantic_shannon success_conditioned_signed_semantic_shannon",
    ):
        assert literal in submitter

    for literal in (
        'SEMANTIC_SHANNON_SEPARATE_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE:-0}"',
        'SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE:-0}"',
        'SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP:-0.05}"',
        "export OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0",
        "export OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=0",
        "semantic_shannon)",
        "success_conditioned_signed_semantic_shannon)",
        "export OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1",
        "export OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1",
        'export OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP="$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP"',
        'VARIANT_TAG="success_conditioned_signed_semantic_shannon"',
    ):
        assert literal in runner

    for literal in (
        'SEMANTIC_SHANNON_SEPARATE_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE:-0}"',
        'SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE:-0}"',
        'SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP="${OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP:-0.05}"',
        "grep -q 'semantic_shannon_separate_advantage'",
        "cmd+=(--semantic-shannon-separate-advantage)",
        "cmd+=(--no-semantic-shannon-separate-advantage)",
        "Frozen source lacks separate semantic-Shannon advantage support",
        "grep -q 'semantic_shannon_success_conditioned_signed_advantage'",
        'cmd+=(--semantic-shannon-success-conditioned-signed-cap "$SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP")',
        "cmd+=(--semantic-shannon-success-conditioned-signed-advantage)",
        "cmd+=(--no-semantic-shannon-success-conditioned-signed-advantage)",
        "Frozen source lacks success-conditioned signed semantic-Shannon advantage support",
    ):
        assert literal in train


def test_e43_protocol_freezes_success_conditioned_signed_capped_advantage():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "**Status: FROZEN BEFORE LAUNCH",
        "success-conditioned signed semantic-Shannon advantage",
        "eligible successful leave-one-out peer outcomes",
        "loss-active, has positive task reward, and has a parseable answer key",
        "`h_i = sum_a q_i(a) * min(-log q_i(a), 5.0)`",
        "`A_i_raw = (0.10 / 5.0) * (s_i - h_i)`",
        "`A_i_E43-sem = e_i * max(-0.05, min(A_i_raw, 0.05))`",
        "`A_i_E43 = A_i_task + A_i_E43-sem`",
        "at least `-0.05`",
        "cannot reward randomness among failures",
        "all-wrong",
        "No second empirical group centering",
        "`success_conditioned_signed_semantic_shannon`",
        "`OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1`",
        "`OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=0`",
        "`OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1`",
        "`OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP=0.05`",
        "seeds `43, 44, 45`",
        "ten complete prompt passes",
        "`370100, 370101, 370102, 370103`",
        "`390100`",
        "MATH-500",
    ):
        assert literal in text
