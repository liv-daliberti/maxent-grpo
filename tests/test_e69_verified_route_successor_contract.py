from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _route_variant_branch() -> str:
    source = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")
    start = source.index("  verified_route_successor)")
    stop = source.index("  maxent_length_dual)", start)
    return source[start:stop]


def test_e69_variant_is_task_first_support_separated_and_fixed_budget():
    branch = _route_variant_branch()
    required = (
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=verified_route",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ENTROPY_GATE=0",
        'VARIANT_TAG="verified_route_successor"',
    )
    for literal in required:
        assert literal in branch


def test_e69_route_knobs_reach_the_typed_training_surface():
    train = (ROOT / "ops/train.sh").read_text(encoding="utf-8")
    for literal in (
        "--verified-route-replay-capacity-per-route",
        "--verified-route-recurring-min-neutral-prompts",
        "--verified-route-proposal-max-mean-logprob-drop",
        "--online-canonical-counterfactual-fixed-control-groups",
        "--online-canonical-replay-compute-only",
    ):
        assert literal in train
    assert "COUNTERFACTUAL_TEMPERATURE_STEP=0" in train

    submit = (ROOT / "ops/submit_countdown_comparative.sh").read_text(
        encoding="utf-8"
    )
    for literal in (
        "OAT_ZERO_INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM",
        "submit_arm verified_route_successor verified_route_successor",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS",
        "OAT_ZERO_DRGRPO_VARIANT",
    ):
        assert literal in submit


def test_e69_route_state_and_proposal_separation_are_checkpointed():
    run = (ROOT / "src/oat_drgrpo/learner/run.py").read_text(encoding="utf-8")
    grpo = (ROOT / "src/oat_drgrpo/learner/grpo.py").read_text(encoding="utf-8")
    for literal in (
        '"verified_route_library_state"',
        "route_library.state_dict()",
        "route_library.load_state_dict",
        "verified-route proposals changed the neutral objective ",
        "verified-route actuator may admit at most one proposal ",
        "counterfactual_fixed_control_rows_sent_to_ppo",
        "precomputed_proposal_groups",
    ):
        assert literal in run
    for literal in (
        "scheduled_cross_prompt_replay_groups",
        '"verified_route_proposal_rows_to_ppo"',
        '"verified_route_gold_support_feedback"',
        '"verified_route_eval_feedback"',
        '"canonical_replay_compute_only"',
        '"canonical_replay_charged_response_token_budget"',
        "torch.zeros_like(raw_score_gradients)",
    ):
        assert literal in grpo


def test_e69_compute_matched_drgrpo_runs_replay_with_zero_influence():
    source = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")
    start = source.index("  grpo_compute_matched)")
    stop = source.index("  grpo_entropy)", start)
    branch = source[start:stop]
    for literal in (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
        'VARIANT_TAG="grpo_compute_matched"',
    ):
        assert literal in branch


def test_e69_protocol_names_the_four_compute_matched_arms_and_five_areas():
    protocol = (
        ROOT / "paper/preregistration/e69_verified_route_successor_protocol_20260728.md"
    ).read_text(encoding="utf-8")
    protocol_flat = " ".join(protocol.split())
    for literal in (
        "1. Dr.GRPO;",
        "2. E66 endpoint-only replay;",
        "3. E68 separated-support endpoint proposals;",
        "4. the E69 hierarchical verified-route successor.",
        "Graph, Countdown, Python, MathIR",
        "MATH12K route-dev",
        "MATH-500 is the fifth panel area",
    ):
        assert literal in protocol_flat
