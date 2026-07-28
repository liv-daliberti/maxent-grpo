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


def test_e69_gate2_to_gate3_transition_fails_closed():
    transition = (
        ROOT / "ops/route_successor/advance_e69_gate2_to_gate3.sh"
    ).read_text(encoding="utf-8")
    for literal in (
        "audit_e69_gate2_screen.py",
        'audit.get("status") != "pass"',
        'summary.get("terminal_physical_runs") != 18',
        'summary.get("integrity_violations") != 0',
        'audit.get("math500_sealed") is not True',
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
