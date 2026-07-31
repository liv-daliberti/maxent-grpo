import json
from pathlib import Path

from ops.route_successor.audit_e69_gate2_screen import (
    _training_audit,
    evaluate_gate,
)
from ops.route_successor.audit_e69_gate3_confirmatory import (
    classify_internal_result,
    crossed_bootstrap_interval,
)
from ops.route_successor.analyze_e69_gate4_math500 import classify_final
from ops.route_successor.eval_e69_math500_checkpoint import _metrics


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


def test_e69_gate2_launcher_freezes_physical_arms_aliases_and_budget():
    launcher = (
        ROOT / "ops/route_successor/launch_e69_gate2_screen.sh"
    ).read_text(encoding="utf-8")
    protocol = (
        ROOT
        / "paper/preregistration/e69_gate2_compute_matched_screen_20260728.md"
    ).read_text(encoding="utf-8")
    for literal in (
        "OAT_ZERO_DRGRPO_VARIANT=grpo_compute_matched",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=6",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1",
        "physical_job_count",
        '"E68": "verified_first_global_replay_canonical"',
        '"E69": "verified_first_global_replay_canonical"',
        "MATH_DATA=\"$ROOT_DIR/var/data/math12k_384_route_dev128_v1\"",
    ):
        assert literal in launcher
    for literal in (
        "64 sampled rows per training prompt",
        "terminal pass 6",
        "post-replay reproduction counter",
        "MATH-500 remains absent",
    ):
        assert literal in protocol


def _passing_gate_curves():
    domains = (
        "graph_coloring",
        "countdown",
        "python_factor",
        "mathir",
        "math_dev",
    )
    curves = {}
    for domain in domains:
        treatment = (
            "verified_first_global_replay_canonical"
            if domain == "math_dev"
            else "verified_route_successor"
        )
        curves[domain] = {"grpo": {}, treatment: {}}
        for pass_index in range(7):
            curves[domain]["grpo"][pass_index] = {
                "greedy": 0.50,
                "mean8": 0.50,
                "pass8": 0.60,
                "distinct8": 1.0,
            }
            curves[domain][treatment][pass_index] = {
                "greedy": 0.51,
                "mean8": 0.51,
                "pass8": 0.61,
                "distinct8": 1.1,
            }
    route = {
        domain: {
            "post_replay_cross_prompt_neutral_reproductions": 1.0,
        }
        for domain in domains[:4]
    }
    return curves, route


def test_e69_gate2_gate_is_pure_strict_and_persistent():
    curves, route = _passing_gate_curves()
    result = evaluate_gate(curves, route)
    assert result["status"] == "pass"
    assert all(result["checks"].values())

    curves["mathir"]["verified_route_successor"][5]["pass8"] = 0.60
    result = evaluate_gate(curves, route)
    assert result["status"] == "fail"
    assert not result["checks"]["mathir_pass_and_distinct_gain_pass_5"]


def test_e69_gate2_gate_uses_prospective_temporal_route_observer():
    curves, route = _passing_gate_curves()
    temporal = {
        domain: {"post_replay_neutral_reproduction_pairs": 1}
        for domain in ("graph_coloring", "countdown", "python_factor", "mathir")
    }
    result = evaluate_gate(curves, route, temporal)
    assert result["status"] == "pass"
    assert (
        result["mechanism_observer"]
        == "prospective_checkpoint_temporal_lower_bound"
    )

    temporal["python_factor"]["post_replay_neutral_reproduction_pairs"] = 0
    temporal["graph_coloring"]["post_replay_neutral_reproduction_pairs"] = 0
    result = evaluate_gate(curves, route, temporal)
    assert result["status"] == "fail"
    assert not result["checks"]["post_replay_reuse_three_domains"]

    temporal["graph_coloring"]["post_replay_neutral_reproduction_pairs"] = 1
    result = evaluate_gate(
        curves,
        route,
        temporal,
        require_python_reproduction=True,
    )
    assert result["status"] == "fail"
    assert result["checks"]["post_replay_reuse_three_domains"]
    assert not result["checks"]["python_post_replay_neutral_reproduction"]

    temporal["python_factor"]["post_replay_neutral_reproduction_pairs"] = 1
    result = evaluate_gate(
        curves,
        route,
        temporal,
        require_python_reproduction=True,
    )
    assert result["status"] == "pass"
    assert result["checks"]["python_post_replay_neutral_reproduction"]


def test_e69_gate2_training_audit_allows_not_yet_started_run(tmp_path):
    training, violations = _training_audit(
        tmp_path / "train_metrics.jsonl",
        label="graph/e69/job0",
        arm="verified_route_successor",
        response_limit=192,
    )
    assert violations == []
    assert training["records"] == 0
    assert training["latest_step"] == -1
    assert training["route_terminal"] == {}


def test_e69_gate2_training_audit_selects_only_registered_latest_attempt(
    tmp_path,
):
    path = tmp_path / "train_metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"trainer/global_step": 0}',
                '{"trainer/global_step": 1}',
                '{"trainer/global_step": 0}',
                '{"trainer/global_step": 1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    training, violations = _training_audit(
        path,
        label="python/e69/job0",
        arm="verified_route_successor",
        response_limit=192,
        accepted_start_line=3,
    )
    assert violations == []
    assert training["latest_step"] == 1
    assert training["accepted_start_line"] == 3

    _, violations = _training_audit(
        path,
        label="python/e69/job0",
        arm="verified_route_successor",
        response_limit=192,
    )
    assert any("unregistered optimizer-step regression 1->0" in row for row in violations)


def test_e69_gate2_precheckpoint_requeue_repair_is_exact_and_outcome_blind():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e69_gate2_precheckpoint_requeue_attempt_repair_20260728.md"
    ).read_text(encoding="utf-8")
    protocol_flat = " ".join(protocol.split())
    for literal in (
        "while all Gate 2 runs were nonterminal",
        "Graph compute-matched Dr.GRPO job `30160592`",
        "Python verified-route successor job `30160205`",
        "Neither abandoned prefix produced an optimizer checkpoint",
        "byte-identical",
        "reject any later step regression",
        "Discarded pre-checkpoint attempts are not spliced",
    ):
        assert literal in protocol_flat

    identity = json.loads(
        (
            ROOT
            / "var/artifacts/"
            "e69_gate2_precheckpoint_requeue_attempt_repair_identity.json"
        ).read_text(encoding="utf-8")
    )
    assert identity["schema"] == (
        "e69_gate2_precheckpoint_requeue_attempt_repair_v1"
    )
    assert set(identity["jobs"]) == {"30160592", "30160205"}
    assert identity["terminal_outcomes_available_before_repair"] is False
    for row in identity["jobs"].values():
        assert row["accepted_start_line"] == (
            row["abandoned_prefix_last_line"] + 1
        )
        assert row["accepted_start_step"] == 0
        assert row["abandoned_prefix_checkpoint_count"] == 0
        assert row["abandoned_prefix_evaluation_steps"] == [0]
        assert (
            row["abandoned_step0_evaluation_sha256"]
            == row["accepted_step0_evaluation_sha256"]
        )


def test_e69_gate3_contract_freezes_reuse_compute_and_analysis_before_outcomes():
    protocol = (
        ROOT
        / "paper/preregistration/e69_gate3_confirmatory_execution_20260728.md"
    ).read_text(encoding="utf-8")
    protocol_flat = " ".join(protocol.split())
    for literal in (
        "before any terminal Gate 2 outcome was available",
        "unless the frozen Gate 2 audit returns `pass`",
        "The exact Gate 2 seed-43 control and successor jobs are reused.",
        "16 new executable-domain jobs and four new MATH jobs",
        "30 unique physical runs",
        "one neutral group and three discarded proposal-shaped control groups",
        "Pass 6 is the sole terminal checkpoint.",
        "10,000 deterministic crossed-bootstrap replicates with seed `690301`",
        "All three raw paired seed deltas",
        "MATH-500 once, regardless of the direction of the Gate 3 result",
    ):
        assert literal in protocol_flat

    launcher = (
        ROOT / "ops/route_successor/launch_e69_gate3_confirmatory.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        'if audit.get("status") != "pass":',
        "OAT_ZERO_TRAIN_SEEDS=44,45",
        'OAT_ZERO_DRGRPO_VARIANT=grpo_compute_matched',
        'OAT_ZERO_ONLY_ARMS="grpo,verified_route_successor"',
        'OAT_ZERO_ONLY_ARMS="grpo,verified_first_global_replay_canonical"',
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10",
        "OAT_ZERO_E69_GRAPH_NODELIST:-node105,node202,node203,node204",
        "OAT_ZERO_E69_GRAPH_GRES:-gpu:a5000:1",
        "OAT_ZERO_E69_A5000_NODELIST:-node105,node202,node203,node204",
        "OAT_ZERO_E69_A5000_GRES:-gpu:a5000:1",
        '"origin": "gate3_new"',
        'dict(row, origin="gate2_reuse")',
        '"auditor_sha256": digest(sys.argv[7])',
        '"reused_physical_jobs": 10',
        '"new_physical_jobs": 20',
        '"confirmatory_physical_jobs": 30',
        'scontrol release "${job_ids[@]}"',
    ):
        assert literal in launcher


def test_e69_gate3_crossed_bootstrap_preserves_constant_paired_delta():
    lower, upper = crossed_bootstrap_interval(
        {
            43: [0.125] * 8,
            44: [0.125] * 8,
            45: [0.125] * 8,
        },
        replicates=100,
    )
    assert lower == upper == 0.125


def test_e69_gate3_classification_is_pure_and_requires_persistence():
    aggregate = {}
    seeds = {}
    routes = {}
    for domain in (
        "graph_coloring",
        "countdown",
        "python_factor",
        "mathir",
        "math_dev",
    ):
        aggregate[domain] = {}
        seeds[domain] = {}
        for pass_index in range(7):
            aggregate[domain][pass_index] = {
                "greedy": 0.01,
                "mean8": 0.01,
                "pass8": 0.01,
                "distinct8": 0.01,
            }
        for seed in (43, 44, 45):
            seeds[domain][seed] = {}
            for pass_index in range(7):
                seeds[domain][seed][pass_index] = {
                    "greedy": 0.01,
                    "mean8": 0.01,
                    "pass8": 0.01,
                    "distinct8": 0.01,
                }
        if domain != "math_dev":
            routes[domain] = {
                seed: {
                    "post_replay_cross_prompt_neutral_reproductions": 1.0,
                }
                for seed in (43, 44, 45)
            }
    result = classify_internal_result(aggregate, seeds, routes)
    assert result["status"] == "success"
    assert all(result["checks"].values())

    aggregate["python_factor"][5]["distinct8"] = 0.0
    aggregate["countdown"][5]["distinct8"] = 0.0
    result = classify_internal_result(aggregate, seeds, routes)
    assert result["status"] == "mechanism_or_null"
    assert not result["checks"]["three_executable_domains_positive_support"]


def test_e69_gate4_contract_freezes_one_time_transfer_before_outcomes():
    protocol = (
        ROOT
        / "paper/preregistration/e69_gate4_math500_one_time_transfer_20260728.md"
    ).read_text(encoding="utf-8")
    protocol_flat = " ".join(protocol.split())
    for literal in (
        "before any terminal Gate 2 or Gate 3 outcome was available",
        "regardless of whether Gate 3's internal efficacy classification is positive",
        "`saved_models/step_02305`",
        "exactly all 500 rows",
        "seed `690401`",
        "ordinary full `math_verify` verifier",
        "Six evaluation jobs are submitted as one held cohort",
        "10,000-replicate crossed paired seed-and-prompt bootstrap",
        "seed `690402`",
        "**held-out MATH-500 transfer**",
        "Report every outcome without tuning",
    ):
        assert literal in protocol_flat

    evaluator = (
        ROOT / "ops/route_successor/eval_e69_math500_checkpoint.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "apply_qwen_math_template",
        "FullMathVerifierProcess",
        'reward_kind="boxed"',
        "SAMPLED_SEED = 690401",
        "MAX_TOKENS = 1024",
        "MAX_MODEL_LEN = 2048",
        "conflicting immutable Gate 4 output already exists",
        "immutable result already complete",
        "no regeneration",
        'checkpoint.name != "step_02305"',
    ):
        assert literal in evaluator

    launcher = (
        ROOT / "ops/route_successor/launch_e69_gate4_math500.sh"
    ).read_text(encoding="utf-8")
    assert launcher.index("Gate 4 requires clean, complete, sealed Gate 3") < (
        launcher.index('job_id="$(sbatch')
    )
    for literal in (
        "audit_e69_gate4_checkpoints.py",
        "analyze_e69_gate4_math500.py",
        "plot_e69_five_area_panel.py",
        "finalize_e69_gate4.sh",
        "e69_gate4_finalize.slurm",
        "sbatch --hold --parsable",
        '--dependency="afterany:${dependency}"',
        'scontrol release "${job_ids[@]}"',
        '"schema": "e69_gate4_finalizer_job_v1"',
        '"bootstrap": {"replicates": 10000, "seed": 690402}',
    ):
        assert literal in launcher


def test_e69_gate4_metric_recomputation_is_prompt_paired():
    rows = [
        [
            {
                "reward": reward,
                "response_token_count": 10,
                "verifier_info": {},
            }
            for reward in rewards
        ]
        for rewards in ((1.0, 0.0), (0.0, 0.0))
    ]
    metrics = _metrics(rows)
    assert metrics["mean_at_k"] == 0.25
    assert metrics["any_correct_at_k"] == 0.5
    assert metrics["mean_response_tokens"] == 10.0


def test_e69_gate4_final_classification_distinguishes_mechanism_result():
    executable = {
        domain: {"greedy": 0.0, "pass8": 0.0}
        for domain in ("graph_coloring", "countdown", "python_factor", "mathir")
    }
    support = {
        domain: True
        for domain in ("graph_coloring", "countdown", "python_factor", "mathir")
    }
    mechanism = dict(support)
    result = classify_final(
        executable,
        {"greedy": 0.0, "mean8": 0.0, "pass8": 0.0},
        support,
        mechanism,
    )
    assert result["status"] == "successful_exploration"
    assert all(result["checks"].values())

    executable["graph_coloring"]["greedy"] = -0.03
    result = classify_final(
        executable,
        {"greedy": 0.0, "mean8": 0.0, "pass8": 0.0},
        support,
        mechanism,
    )
    assert result["status"] == "mechanism_result"
    assert not result["checks"]["four_executable_areas_task_quality_noninferior"]


def test_e69_gate2_math_endpoint_repair_is_single_preoptimizer_correction():
    amendment = (
        ROOT
        / "paper/preregistration/e69_gate2_math_endpoint_startup_repair_20260728.md"
    ).read_text(encoding="utf-8")
    amendment_flat = " ".join(amendment.split())
    for literal in (
        "job `30159730`",
        "never entered optimization",
        "no optimizer metric record",
        "cancelled after five recorded retries",
        "`OAT_ZERO_SEMANTIC_SHANNON_COEF` from `0` to `0.10`",
        "does not enable route prompting",
        "Exactly one replacement job",
        "before any Gate 2 terminal outcome was available",
    ):
        assert literal in amendment_flat

    launcher = (
        ROOT / "ops/route_successor/repair_e69_gate2_math_endpoint.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "INVALID_JOB=30159730",
        "OAT_ZERO_ONLY_ARMS=verified_first_global_replay_canonical",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10",
        "OAT_ZERO_TRAIN_SEEDS=43",
        "E69 Gate 2 repair manifest must contain exactly one job",
        '"optimizer_records": 0',
        '"terminal_outcomes_observed_before_repair": False',
    ):
        assert literal in launcher

    gate2_audit = (
        ROOT / "ops/route_successor/audit_e69_gate2_screen.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "e69_gate2_math_endpoint_repair_identity.json",
        "excluded Gate 2 job unexpectedly has a run directory",
        'jobs["math_dev"][matches[0]] = replacement',
        '"repairs": repairs',
    ):
        assert literal in gate2_audit

    gate3_launcher = (
        ROOT / "ops/route_successor/launch_e69_gate3_confirmatory.sh"
    ).read_text(encoding="utf-8")
    assert 'for row in gate2_audit["physical_runs"]' in gate3_launcher
    assert '"run_stamp": row["run_stamp"]' in gate3_launcher


def test_e69_gate2_pending_placement_repair_is_matched_and_atomic():
    amendment = (
        ROOT
        / "paper/preregistration/e69_gate2_pending_placement_repair_20260728.md"
    ).read_text(encoding="utf-8")
    amendment_flat = " ".join(amendment.split())
    for literal in (
        "All 12 Gate 2 jobs",
        "zero runtime",
        "no run directory",
        "all four Graph arms run on `node101`, `gpu:a40:1`",
        "all four Countdown arms and all four Python arms run on `node105`",
        "submitted held and audited before the original 12 pending jobs are cancelled",
        "No Graph, Countdown, Python",
        "outcome informed it",
    ):
        assert literal in amendment_flat

    launcher = (
        ROOT / "ops/route_successor/repair_e69_gate2_pending_placements.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "30159713 30159714 30159715 30159716",
        "OAT_ZERO_TRAIN_NODELIST=node101",
        "OAT_ZERO_TRAIN_GRES=gpu:a40:1",
        "OAT_ZERO_TRAIN_NODELIST=node105",
        "OAT_ZERO_TRAIN_GRES=gpu:a5000:1",
        "OAT_ZERO_SBATCH_HOLD=1",
        'scancel "${ORIGINAL_JOBS[@]}"',
        'scontrol release "${replacement_jobs[@]}"',
        '"invalid_jobs_optimizer_records": 0',
        '"outcomes_observed_before_repair": False',
    ):
        assert literal in launcher
    assert launcher.index('scancel "${ORIGINAL_JOBS[@]}"') < launcher.index(
        'scontrol release "${replacement_jobs[@]}"'
    )


def test_e69_gate2_graph_preemption_repair_restarts_all_arms_atomically():
    amendment = (
        ROOT
        / "paper/preregistration/e69_gate2_graph_a5000_preemption_repair_20260728.md"
    ).read_text(encoding="utf-8")
    amendment_flat = " ".join(amendment.split())
    for literal in (
        "preempted",
        "optimizer step 106",
        "no valid resume point",
        "entire four-arm Graph cohort is therefore replaced together",
        "All partial attempts `30160181--30160184` are excluded wholesale",
        "No metric value, arm ranking, or outcome informed this repair",
        "`node105,node202,node203,node204`",
    ):
        assert literal in amendment_flat

    launcher = (
        ROOT / "ops/route_successor/repair_e69_gate2_pending_placements.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "graph-a5000-config|graph-a5000-full",
        "GRAPH_A5000_ORIGINAL_JOBS=(30160181 30160182 30160183 30160184)",
        "OAT_ZERO_TRAIN_NODELIST=node202",
        "OAT_ZERO_TRAIN_GRES=gpu:a5000:1",
        'scancel "${GRAPH_A5000_ORIGINAL_JOBS[@]}"',
        'scontrol release "${graph_replacements[@]}"',
        '"nonterminal_outcomes_available_before_repair": True',
        '"terminal_outcomes_observed_before_repair": False',
    ):
        assert literal in launcher
    assert launcher.index(
        'scancel "${GRAPH_A5000_ORIGINAL_JOBS[@]}"'
    ) < launcher.index('scontrol release "${graph_replacements[@]}"')

    audit = (
        ROOT / "ops/route_successor/audit_e69_gate2_screen.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "e69_gate2_graph_a5000_preemption_repair_identity.json",
        "E69 Gate 2 excluded Graph attempt",
        '"kind": "graph_preemption"',
    ):
        assert literal in audit


def test_e69_gate2_r1_replaces_exact_graph_python_cells_without_tuning():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e69_gate2_r1_execution_repair_20260728.md"
    ).read_text(encoding="utf-8")
    protocol_flat = " ".join(protocol.split())
    for literal in (
        "17 scheduler restarts",
        "replace the complete four-arm Graph cohort",
        "single corrupted Python route-successor cell",
        "`node105,node202,node203,node204`",
        "Terminal Graph outcomes and partial Python telemetry were available",
        "did not determine the replacement cells",
        "Python must supply both a clean terminal R1 result and nonzero temporal route reproduction",
        "MATH-500 remains sealed",
    ):
        assert literal in protocol_flat

    launcher = (
        ROOT
        / "ops/route_successor/"
        "launch_e69_gate2_r1_execution_repair.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "SOURCE_GRAPH_JOBS=(30160592 30160594 30160595 30160596)",
        "SOURCE_PYTHON_JOB=30160205",
        "OAT_ZERO_ONLY_ARMS=verified_route_successor",
        "OAT_ZERO_TRAIN_NODELIST=\"$A5000_POOL\"",
        "OAT_ZERO_TRAIN_PARTITION=all",
        "OAT_ZERO_SBATCH_HOLD=1",
        '"outcome_tuning": False',
        '"algorithm_or_gate_change": False',
        'scancel "$SOURCE_PYTHON_JOB" "$OLD_TRANSITION_JOB"',
        'scontrol release "${replacement_jobs[@]}"',
    ):
        assert literal in launcher
    assert launcher.index(
        'scancel "$SOURCE_PYTHON_JOB" "$OLD_TRANSITION_JOB"'
    ) < launcher.index('scontrol release "${replacement_jobs[@]}"')

    identity = json.loads(
        (
            ROOT
            / "var/artifacts/e69_gate2_r1_execution_repair_identity.json"
        ).read_text(encoding="utf-8")
    )
    assert identity["schema"] == "e69_gate2_r1_execution_repair_v1"
    assert identity["outcome_tuning"] is False
    assert identity["algorithm_or_gate_change"] is False
    assert len(identity["mappings"]["graph_coloring"]) == 4
    assert len(identity["mappings"]["python_factor"]) == 1
    assert identity["math_jobs_untouched"] == [30159729, 30160101]
    assert identity["math500_sealed"] is True

    audit = (
        ROOT / "ops/route_successor/audit_e69_gate2_screen.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "e69_gate2_r1_execution_repair_identity.json",
        '"kind": "execution_r1"',
        "frozen_scheduler_integrity_rule_not_metric_values",
        "e69_gate2_r1_route_temporal_snapshots.json",
    ):
        assert literal in audit

    snapshots = json.loads(
        (
            ROOT
            / "var/artifacts/e69_gate2_r1_route_temporal_snapshots.json"
        ).read_text(encoding="utf-8")
    )
    assert snapshots["schema"] == (
        "e69_gate2_r1_route_temporal_snapshots_v1"
    )
    assert snapshots["carried_parent_domains"] == ["countdown", "mathir"]
    assert snapshots["fresh_r1_domains"] == [
        "graph_coloring",
        "python_factor",
    ]
    for domain, job_id, steps in (
        ("graph_coloring", 30168830, {192, 384, 576, 768, 960, 1152}),
        ("python_factor", 30168831, {384, 768, 1152, 1536, 1920, 2304}),
    ):
        observed = snapshots["snapshots"][domain]
        assert {int(step) for step in observed}.issubset(steps)
        for row in observed.values():
            assert f"debug_job{job_id}" in row["checkpoint"]


def test_e69_gate2_r2_is_a_prospective_minimal_python_contract_repair():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e69_gate2_r2_route_endpoint_bookkeeping_repair_20260728.md"
    ).read_text(encoding="utf-8")
    protocol_flat = " ".join(protocol.split())
    for literal in (
        "failed at training step 302",
        "before its first step-384 checkpoint",
        "R1 therefore cannot pass",
        "one replay exemplar",
        "lexicographically smallest response-token tuple",
        "identical response-token tuple changes endpoint key, fail closed",
        "replaces only the failed Python route-successor cell",
        "MATH-500 remain sealed",
    ):
        assert literal in protocol_flat

    launcher = (
        ROOT
        / "ops/route_successor/"
        "launch_e69_gate2_r2_route_endpoint_repair.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "FAILED_R1_JOB=30168831",
        "OAT_ZERO_ONLY_ARMS=verified_route_successor",
        "OAT_ZERO_AUTO_RESUME=0",
        "changed_source_files",
        "oat_drgrpo/verified_route_library.py",
        '"repaired_outcomes_observed_before_freeze": False',
        '"implementation_contract_repair": True',
        '"outcome_tuning": False',
        'scancel "$OLD_TRANSITION_JOB"',
        'scontrol release "$replacement_job"',
    ):
        assert literal in launcher
    assert launcher.index('scancel "$OLD_TRANSITION_JOB"') < launcher.index(
        'scontrol release "$replacement_job"'
    )

    library = (
        ROOT / "src/oat_drgrpo/verified_route_library.py"
    ).read_text(encoding="utf-8")
    assert "stored_response_tokens == response_tokens" in library
    assert 'record["endpoint_key"] = endpoint_key' in library
    assert "verified route endpoint changed for one response" in library

    audit = (
        ROOT / "ops/route_successor/audit_e69_gate2_screen.py"
    ).read_text(encoding="utf-8")
    observer = (
        ROOT / "ops/route_successor/snapshot_e69_gate2_route_replay.py"
    ).read_text(encoding="utf-8")
    monitor = (
        ROOT / "ops/route_successor/monitor_e69_gate2_route_snapshots.sh"
    ).read_text(encoding="utf-8")
    for source in (audit, observer, monitor):
        assert "e69_gate2_r2_route" in source

    identity_path = (
        ROOT
        / "var/artifacts/"
        "e69_gate2_r2_route_endpoint_bookkeeping_repair_identity.json"
    )
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        assert identity["schema"] == (
            "e69_gate2_r2_route_endpoint_bookkeeping_repair_v1"
        )
        assert identity["mapping"]["invalid_job_id"] == 30168831
        assert identity["mapping"]["arm"] == "verified_route_successor"
        assert identity["outcome_tuning"] is False
        assert identity["implementation_contract_repair"] is True
        assert identity["math500_sealed"] is True


def test_e69_gate2_r3_repairs_the_paired_math_dev_evaluation_contract():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e69_gate2_r3_math_dev_evaluation_split_repair_20260729.md"
    ).read_text(encoding="utf-8")
    protocol_flat = " ".join(protocol.split())
    for literal in (
        "sole split `math_dev`",
        "`OAT_ZERO_TEST_SPLIT=math`",
        "no MATH-dev Gate 2 evaluation outcome was available",
        "complete two-arm MATH-dev cohort is replaced from initialization",
        "`OAT_ZERO_TEST_SPLIT=math` -> `OAT_ZERO_TEST_SPLIT=math_dev`",
        "online evaluation fails closed if split selection yields zero datasets",
        "never averaged with the replacements",
        "MATH-500 remains sealed",
    ):
        assert literal in protocol_flat

    launcher = (
        ROOT
        / "ops/route_successor/"
        "launch_e69_gate2_r3_math_dev_evaluation_split_repair.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "OLD_MATH_JOBS=(30159729 30160101)",
        "OAT_ZERO_TEST_SPLIT=math_dev",
        "submit_arm grpo 0 0",
        "submit_arm verified_first_global_replay_canonical 0.10 1",
        "OAT_ZERO_AUTO_RESUME=0",
        "TresPerNode=gres/gpu:a100:1",
        "startup_rejected_manifest_sha256",
        '"gate_outcomes_observed_before_freeze": False',
        '"training_telemetry_observed_before_freeze": True',
        '"outcome_tuning": False',
        '"algorithm_or_gate_change": False',
        'scancel "$OLD_TRANSITION_JOB"',
        'scontrol release "${replacement_jobs[@]}"',
    ):
        assert literal in launcher
    assert launcher.index('scancel "$OLD_TRANSITION_JOB"') < launcher.index(
        'scontrol release "${replacement_jobs[@]}"'
    )

    args_source = (ROOT / "src/oat_drgrpo/args.py").read_text(
        encoding="utf-8"
    )
    learner_init = (ROOT / "src/oat_drgrpo/learner/init.py").read_text(
        encoding="utf-8"
    )
    assert 'args.test_split not in {"math", "math_dev"}' in args_source
    assert "online evaluation selected zero datasets" in learner_init

    audit = (
        ROOT / "ops/route_successor/audit_e69_gate2_screen.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "e69_gate2_r3_math_dev_evaluation_split_repair_identity.json",
        '"kind": "implementation_r3_math_eval_split"',
        '{"oat_drgrpo/args.py", "oat_drgrpo/learner/init.py"}',
        '(run_dir / "eval_mode_coverage_draws.jsonl").exists()',
        "restarted the paired cohort as jobs",
        "both arms restart from initialization",
    ):
        assert literal in audit

    plotter = (
        ROOT / "ops/exp_scaling/plot_e68_e58_vs_grpo_12pass.py"
    ).read_text(encoding="utf-8")
    for literal in (
        "e69_gate2_r3_math_dev_evaluation_split_repair_identity.json",
        '"R3 MATH"',
        "accepted repair chain ",
    ):
        assert literal in plotter

    identity_path = (
        ROOT
        / "var/artifacts/"
        "e69_gate2_r3_math_dev_evaluation_split_repair_identity.json"
    )
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        assert identity["schema"] == (
            "e69_gate2_r3_math_dev_evaluation_split_repair_v1"
        )
        assert len(identity["mappings"]) == 2
        assert {
            row["invalid_job_id"] for row in identity["mappings"]
        } == {30159729, 30160101}
        assert identity["gate_outcomes_observed_before_freeze"] is False
        assert identity["outcome_tuning"] is False
        assert identity["math500_sealed"] is True


def test_e69_gate2_to_gate3_transition_fails_closed():
    transition = (
        ROOT / "ops/route_successor/advance_e69_gate2_to_gate3.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "audit_e69_gate2_screen.py",
        "snapshot_e69_gate2_route_replay.py",
        'audit.get("status") != "pass"',
        'summary.get("terminal_physical_runs") != 18',
        'summary.get("integrity_violations") != 0',
        'audit.get("math500_sealed") is not True',
        '"python_post_replay_neutral_reproduction"',
        'route_reproductions", {}).get("python_factor", 0)',
        "positive Python temporal route reproduction",
        "launch_e69_gate3_confirmatory.sh full",
    ):
        assert literal in transition

    slurm = (
        ROOT / "ops/slurm/e69_gate2_to_gate3.slurm"
    ).read_text(encoding="utf-8")
    for literal in (
        "#SBATCH --job-name=e69g2_to_g3",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=8G",
        "advance_e69_gate2_to_gate3.sh",
    ):
        assert literal in slurm


def test_e69_gate2_route_temporal_observer_is_prospective_and_read_only():
    amendment = (
        ROOT
        / "paper/preregistration/"
        "e69_gate2_route_temporal_observer_amendment_20260728.md"
    ).read_text(encoding="utf-8")
    amendment_flat = " ".join(amendment.split())
    for literal in (
        "before any pass-5 or pass-6 Gate 2 evaluation existed",
        "Training and its frozen source snapshot remain untouched",
        "strictly increases between those checkpoints",
        "conservative temporal lower bound",
        "every three minutes",
        "replaces only the structurally incapable in-process counter",
    ):
        assert literal in amendment_flat

    observer = (
        ROOT / "ops/route_successor/snapshot_e69_gate2_route_replay.py"
    ).read_text(encoding="utf-8")
    for literal in (
        'mmap=True',
        '"verified_route_library_state"',
        '"replayed_route_targets"',
        '"replayed_target_neutral_counts"',
        '"post_replay_neutral_reproduction_pairs"',
        "neutral count decreased",
        "replay-target set is not monotone",
    ):
        assert literal in observer

    monitor = (
        ROOT / "ops/route_successor/monitor_e69_gate2_route_snapshots.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "snapshot_e69_gate2_route_replay.py",
        'sleep 180',
        'observed_snapshots',
        '== 24',
    ):
        assert literal in monitor

    slurm = (
        ROOT / "ops/slurm/e69_gate2_route_observer.slurm"
    ).read_text(encoding="utf-8")
    for literal in (
        "#SBATCH --job-name=e69g2_route_obs",
        "#SBATCH --mem=8G",
        "#SBATCH --time=7-00:00:00",
        "#SBATCH --requeue",
        "monitor_e69_gate2_route_snapshots.sh",
    ):
        assert literal in slurm


def test_e69_gate3_to_gate4_transition_fails_closed():
    launcher = (
        ROOT / "ops/route_successor/launch_e69_gate3_confirmatory.sh"
    ).read_text(encoding="utf-8")
    assert launcher.index(
        '--dependency="afterany:${dependency}"'
    ) < launcher.index('scontrol release "${job_ids[@]}"')
    for literal in (
        "e69_gate3_to_gate4.slurm",
        'transition_job_id="$(\n  sbatch',
        "'Reason=Dependency'",
        '"schema": "e69_gate3_to_gate4_transition_job_v1"',
        'scontrol release "${job_ids[@]}"',
    ):
        assert literal in launcher

    transition = (
        ROOT / "ops/route_successor/advance_e69_gate3_to_gate4.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "audit_e69_gate3_confirmatory.py",
        '"auditor_sha256"',
        "E69 Gate 3 auditor changed after confirmatory launch",
        'audit.get("status") != "complete"',
        'summary.get("terminal_physical_runs") != 30',
        'summary.get("integrity_violations") != 0',
        'audit.get("math500_sealed") is not True',
        "launch_e69_gate4_math500.sh full",
    ):
        assert literal in transition

    slurm = (
        ROOT / "ops/slurm/e69_gate3_to_gate4.slurm"
    ).read_text(encoding="utf-8")
    for literal in (
        "#SBATCH --job-name=e69g3_to_g4",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=8G",
        "advance_e69_gate3_to_gate4.sh",
    ):
        assert literal in slurm


def test_e69_gate4_finalizer_requires_six_clean_results():
    finalizer = (
        ROOT / "ops/route_successor/finalize_e69_gate4.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "analyze_e69_gate4_math500.py",
        '"analyzer_sha256"',
        '"plotter_sha256"',
        '"finalizer_sha256"',
        '"finalizer_slurm_sha256"',
        "E69 Gate 4 analysis surface drift",
        'analysis.get("status") != "complete"',
        'summary.get("observed_results") != 6',
        'summary.get("integrity_violations") != 0',
        'analysis.get("final_classification") is None',
        "plot_e69_five_area_panel.py",
    ):
        assert literal in finalizer

    slurm = (
        ROOT / "ops/slurm/e69_gate4_finalize.slurm"
    ).read_text(encoding="utf-8")
    for literal in (
        "#SBATCH --job-name=e69g4_final",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=8G",
        "finalize_e69_gate4.sh",
    ):
        assert literal in slurm
