from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_e49b_protocol_freezes_matched_three_epoch_toy_and_full():
    protocol = (
        ROOT
        / "paper/preregistration/e49b_reasoned_math_strategy_haarnoja_05b.md"
    ).read_text(encoding="utf-8")
    for required in (
        "**Status: FROZEN BEFORE LAUNCH",
        "math_verify",
        "two independently permuted",
        "Any negative answer score",
        "`rho_x = H(q_x) / log |B_x^+|`",
        "target `rho*=0.80`",
        "exactly three prompt epochs",
        "Full MATH-500",
        "compute-matched Dr.GRPO",
    ):
        assert required in protocol


def test_e49b_launcher_pins_e46_and_exact_oat_split():
    launcher = (
        ROOT
        / "ops/exp_scaling/launch_e49_validator_bound_math_strategy_05b.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja",
        "OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_strategy_qwen72",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90",
        "e47w_pairwise_math_strategy_calibration_v1",
        "math_strategy_canonicalizer_pair_veto_v15",
        '"integrity_judge_passes": 2',
        '"boundary_partition_passes": 2',
        '"pairwise_veto_passes": 2',
        "CALIBRATED_CANONICALIZER_SHA256",
        "checkpoint_revision",
        "OAT_ZERO_NUM_PROMPT_EPOCH=3",
        "OAT_ZERO_TRAIN_GRES=gpu:a100:1",
        "var/data/e49b_math_strategy_toy",
        "var/data/math12k_384_math500",
    ):
        assert required in launcher


def test_e49b_runtime_preserves_exact_validator_graded_text_and_state():
    dataset = (ROOT / "src/oat_drgrpo/trajectory_dataset.py").read_text(
        encoding="utf-8"
    )
    learner = (ROOT / "src/oat_drgrpo/learner/grpo.py").read_text(
        encoding="utf-8"
    )
    run = (ROOT / "src/oat_drgrpo/learner/run.py").read_text(encoding="utf-8")
    assert '"prompt": item.prompt' in dataset
    assert '"response": item.response' in dataset
    assert 'trajectory.get("prompts")' in learner
    assert 'trajectory.get("responses")' in learner
    assert "invalid_admissions" in learner
    assert "rejected_integrity_rows" in learner
    assert "math_strategy_canonicalizer_state" in run
    canonicalizer = (
        ROOT / "src/oat_drgrpo/math_strategy_canonicalizer.py"
    ).read_text(encoding="utf-8")
    learner_init = (
        ROOT / "src/oat_drgrpo/learner/init.py"
    ).read_text(encoding="utf-8")
    assert '"type": "json_schema"' in canonicalizer
    assert '"maxItems": len(expected_ids)' in canonicalizer
    assert '"brief_check"' in canonicalizer
    assert '"math_strategy_canonicalizer_menu_bound_v18_trace"' in canonicalizer
    assert "novelty_rule=exact_precalibrated_menu_combo_v18" in learner_init
    assert "runtime_partition_passes=0" in learner_init
    assert "runtime_pairwise_veto_passes=0" in learner_init


def test_e49c_protocol_freezes_finite_action_execution_gate():
    protocol = (
        ROOT
        / "paper/preregistration/e49c_finite_action_math_haarnoja_05b.md"
    ).read_text(encoding="utf-8")
    for required in (
        "Finite-Action",
        "at least two",
        "Two independently prompted, temperature-zero audit",
        "`soundness_execution`",
        "`equivalence_attack`",
        "auditor-only reference answer",
        "<strategy_id>Sj</strategy_id>",
        "<action_combo>Aa>Ab>...</action_combo>",
        "task reward is set to zero",
        "same execution-gated task reward",
        "E46 normalized canonical-bank Haarnoja",
        "three prompt epochs",
        "MATH-500",
    ):
        assert required in protocol


def test_e49c_launcher_is_matched_and_uses_exact_reward_gate():
    launcher = (
        ROOT
        / "ops/exp_scaling/launch_e49c_finite_action_math_05b.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja",
        "OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=1",
        "OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_strategy_qwen72",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90",
        "OAT_ZERO_NUM_PROMPT_EPOCH=3",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_TRAIN_GRES=gpu:a100:1",
        "e49c_math_strategy_menu_toy",
        "e49c_math_strategy_menu_full",
        "math_strategy_canonicalizer_menu_bound_v17",
    ):
        assert required in launcher


def test_e49c_runtime_gates_both_reward_views_after_semantic_admission():
    learner = (ROOT / "src/oat_drgrpo/learner/grpo.py").read_text(
        encoding="utf-8"
    )
    menu = (ROOT / "src/oat_drgrpo/math_strategy_menu.py").read_text(
        encoding="utf-8"
    )
    canonicalizer = (
        ROOT / "src/oat_drgrpo/math_strategy_canonicalizer.py"
    ).read_text(encoding="utf-8")
    assert "apply_math_strategy_task_reward_gate" in learner
    assert "final_rewards * contract_mask" in learner
    assert "task_final_rewards * contract_mask" in learner
    assert "parse_strategy_execution" in canonicalizer
    assert "_canonicalize_menu_group_locked" in canonicalizer
    assert "rejected_contract_rows" in canonicalizer
    assert r"\A<strategy_id>" in menu


def test_e49d_protocol_freezes_maximal_certified_support_and_singletons():
    protocol = (
        ROOT
        / "paper/preregistration/e49d_maximal_support_math_haarnoja_05b.md"
    ).read_text(encoding="utf-8")
    for required in (
        "FROZEN BEFORE TRAINING; PREPROCESSING THROUGHPUT",
        "largest subset that is actually",
        "permits a singleton",
        "maximum clique",
        "`soundness_execution`",
        "`equivalence_attack`",
        "At least 20%",
        "exactly three prompt epochs",
        "384-row MATH training cohort",
        "500-row",
        "current E46",
        "support-recall amendment added exactly",
        "The 20% launch gate is unchanged",
    ):
        assert required in protocol


def test_e49d_launcher_and_materializer_are_fail_closed_and_matched():
    launcher = (
        ROOT
        / "ops/exp_scaling/launch_e49d_maximal_support_math_05b.sh"
    ).read_text(encoding="utf-8")
    materializer = (
        ROOT
        / "ops/math_strategy_calibration/materialize_e49d_strategy_menu_data.py"
    ).read_text(encoding="utf-8")
    judge_server = (
        ROOT / "ops/slurm/e47_qwen72_node105.slurm"
    ).read_text(encoding="utf-8")
    for required in (
        "OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja",
        "OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=1",
        "OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80",
        "OAT_ZERO_NUM_PROMPT_EPOCH=3",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_TRAIN_GRES=gpu:a100:1",
        "e49d_math_strategy_menu_toy",
        "e49d_math_strategy_menu_full",
        "all_retained_strategies_double_audited",
        'FROZEN_SOURCE_HASH="27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd"',
        '"structured_output_backend": "guidance"',
        '"$SOURCE_ROOT/oat_drgrpo/math_strategy_canonicalizer.py"',
    ):
        assert required in launcher
    for required in (
        'AUDIT_CONTRACT_VERSION = "maximal_certified_support_v4"',
        "def _maximal_certified_subset",
        "def _record_passes_contract",
        "collapsed_duplicate_strategy_count",
        "retained_original_strategy_ids",
        "reference_answer_sha256",
        "MAX_GENERATION_ATTEMPTS = 1",
        'SUPPORT_RECALL_VERSION = "one_extra_route_rescue_v1"',
        "MAX_RESCUE_ATTEMPTS = 2",
        "parse_strategy_menu(",
        '"enum": ["S1", "S2", "S3"]',
        "def _audit_assessment_is_well_formed",
        '"temperature": 0.2',
        '"structured_output_backend": "guidance"',
    ):
        assert required in materializer
    assert '"uniqueItems"' not in materializer
    assert "--guided-decoding-backend guidance" in judge_server
    assert '"structured_output_backend": "guidance"' in judge_server

    eval_audit = (
        ROOT / "ops/exp_scaling/audit_e49d_eval_contract.py"
    ).read_text(encoding="utf-8")
    eval_slurm = (
        ROOT / "ops/slurm/e49d_eval_audit_node302.slurm"
    ).read_text(encoding="utf-8")
    assert 'os.environ.get("OAT_ZERO_CAMPAIGN_SOURCE_ROOT"' in eval_audit
    assert "e49d_maximal_support_math_${SOURCE_HASH}/src" in eval_slurm
