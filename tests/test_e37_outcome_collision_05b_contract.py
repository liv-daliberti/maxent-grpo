from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e37_outcome_collision_05b.sh"
PROTOCOL = ROOT / "paper/preregistration/e37_outcome_collision_05b.md"
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"
RUNNER = ROOT / "ops/run_experiment.sh"
TRAIN = ROOT / "ops/train.sh"


def test_e37_is_a_fresh_paired_ten_pass_cohort():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "gce37_outcome_collision_05b_v1",
        "cde37_outcome_collision_05b_v1",
        "OAT_ZERO_ONLY_ARMS=grpo,outcome_collision",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM=1",
        "OAT_ZERO_OUTCOME_COLLISION_COEF=0.10",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=10",
        "OAT_ZERO_NUM_PROMPT_EPOCH=10",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K=8",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=370100",
        "OAT_ZERO_DIAYN_NUM_OPTIONS=0",
        "OAT_ZERO_CANONICAL_ACTION_TASK=none",
        "OAT_ZERO_SBATCH_HOLD=1",
        "EXPECTED_JOBS_PER_TASK=6",
    ):
        assert literal in text


def test_e37_freezes_source_execution_and_resume_identity():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        'cd "$tree"',
        "find . -type f -print0",
        "e37_outcome_collision_05b_v1_identity.json",
        '"source_hash": sys.argv[4]',
        '"execution_surface_hash": sys.argv[5]',
        '"step_zero_pairing": "exact"',
        "OAT_ZERO_MAX_RESUME_NUM=2",
        "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0",
        "OAT_ZERO_EXPORT_STEPS=0",
        'scontrol update JobId="$job_id" Partition="$OAT_ZERO_TRAIN_PARTITION"',
        'scontrol release "${job_ids[@]}"',
    ):
        assert literal in text


def test_e37_held_job_audit_pins_only_the_collision_treatment():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "'OAT_ZERO_VARIANT=outcome_collision'" in text
    assert "'OAT_ZERO_OUTCOME_COLLISION_COEF=0.10'" in text
    assert "'OAT_ZERO_VARIANT=grpo'" in text
    assert "'OAT_ZERO_OUTCOME_COLLISION_COEF=0.0'" in text
    assert "'OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed'" in text
    assert "'OAT_ZERO_DIAYN_NUM_OPTIONS=0'" in text
    assert "'OAT_ZERO_DIAYN_MI_BETA=0.0'" in text


def test_shared_shell_stack_pins_control_and_treatment_coefficients():
    submitter = SUBMITTER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    train = TRAIN.read_text(encoding="utf-8")

    for literal in (
        'INCLUDE_OUTCOME_COLLISION_ARM="${OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM:-0}"',
        'OUTCOME_COLLISION_COEF="${OAT_ZERO_OUTCOME_COLLISION_COEF:-0.1}"',
        'if [[ "$variant" == "outcome_collision" ]]',
        'export_vars+=",OAT_ZERO_OUTCOME_COLLISION_COEF=${OUTCOME_COLLISION_COEF}"',
        'export_vars+=",OAT_ZERO_OUTCOME_COLLISION_COEF=0.0"',
        "submit_arm outcome_collision outcome_collision",
    ):
        assert literal in submitter

    for literal in (
        'OUTCOME_COLLISION_COEF="${OAT_ZERO_OUTCOME_COLLISION_COEF:-0.0}"',
        "export OAT_ZERO_OUTCOME_COLLISION_COEF=0.0",
        "outcome_collision)",
        'export OAT_ZERO_OUTCOME_COLLISION_COEF="$OUTCOME_COLLISION_COEF"',
    ):
        assert literal in runner

    assert 'cmd+=(--outcome-collision-coef "$OUTCOME_COLLISION_COEF")' in train
    assert "Frozen source lacks outcome-collision support" in train


def test_e37_protocol_freezes_dense_neutral_outcome_experiment():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "**Status: FROZEN BEFORE LAUNCH",
        "Every observed outcome participates",
        "parse failure maps to one shared `INVALID`",
        "b_i = -(0.10 / G)",
        "`R_correct >= 0.90625 > 0 >= R_incorrect`",
        "Training seeds: `43, 44, 45`",
        "ten complete prompt passes",
        "Draw seeds are `370100, 370101, 370102, 370103`",
        "every step-zero evaluation metric and raw draw must agree exactly",
        "There is no latent-conditioned evaluation",
    ):
        assert literal in text
