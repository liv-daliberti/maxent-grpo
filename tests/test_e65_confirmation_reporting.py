import csv
import importlib.util
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load(
    "audit_e65_confirmation",
    "ops/exp_scaling/audit_e65_entropy_gated_singleton_confirmation.py",
)
summary = _load(
    "summarize_e65_confirmation",
    "ops/exp_scaling/summarize_e65_five_domain_confirmation.py",
)
math_sensitivity = _load(
    "audit_e64_math500_verifier_sensitivity",
    "ops/exp_scaling/audit_e64_math500_verifier_sensitivity.py",
)
plumbing_audit = _load(
    "audit_e66_same_plumbing_actuator_ablation",
    "ops/exp_scaling/audit_e66_same_plumbing_actuator_ablation.py",
)
corrected_audit = _load(
    "audit_e68_separated_support_actuator_ablation",
    "ops/exp_scaling/audit_e68_separated_support_actuator_ablation.py",
)
equivalence_audit = _load(
    "audit_e68_preintervention_equivalence",
    "ops/exp_scaling/audit_e68_preintervention_equivalence.py",
)
cadence_audit = _load(
    "audit_e65_eval_cadence",
    "ops/exp_scaling/audit_e65_eval_cadence.py",
)
checkpoint_coverage_audit = _load(
    "audit_e65_fixed_checkpoint_coverage",
    "ops/exp_scaling/audit_e65_fixed_checkpoint_coverage.py",
)
readiness_audit = _load(
    "audit_e65_legitimate_result_readiness",
    "ops/exp_scaling/audit_e65_legitimate_result_readiness.py",
)


def _points(arm, passes, value, *, metric="pass8", steps_per_pass=1):
    return [
        {
            "arm": arm,
            "seed": seed,
            "step": training_pass * steps_per_pass,
            "passes": training_pass,
            metric: value,
        }
        for training_pass in passes
        for seed in summary.SEEDS
    ]


def test_e65_auditor_reads_raw_actor_and_normalized_train_metrics():
    assert audit._metric({"actor/example": 1}, "example") == 1
    assert audit._metric({"train/example": 2}, "example") == 2
    assert audit._metric({"unrelated": 3}, "example") is None


def test_eval_cadence_rejects_gaps_longer_than_one_epoch():
    assert cadence_audit._audit_steps(
        [0, 25, 50, 75, 100],
        latest_step=100,
        expected_step=1200,
        epoch_steps=100,
    ) == []
    violations = cadence_audit._audit_steps(
        [0, 101],
        latest_step=101,
        expected_step=1200,
        epoch_steps=100,
    )
    assert any("exceeding one epoch" in value for value in violations)


def test_fixed_checkpoint_coverage_requires_all_three_finite_seeds():
    complete = {"seeds": {"43": 0.1, "44": 0.2, "45": 0.3}}
    missing = {"seeds": {"43": 0.1, "44": 0.2}}
    nonfinite = {"seeds": {"43": 0.1, "44": 0.2, "45": float("nan")}}
    assert checkpoint_coverage_audit._complete_point(complete)
    assert not checkpoint_coverage_audit._complete_point(missing)
    assert not checkpoint_coverage_audit._complete_point(nonfinite)
    assert len(checkpoint_coverage_audit.MODEBENCH_PASSES) == 10
    assert len(checkpoint_coverage_audit.MATH_PASSES) <= 10


def test_legitimate_result_readiness_covers_the_full_claim_surface(tmp_path):
    artifact = tmp_path / "artifact.pdf"
    assert not readiness_audit._artifact_ready(artifact)
    artifact.write_bytes(b"%PDF")
    assert readiness_audit._artifact_ready(artifact)
    assert readiness_audit.EXPECTED_DOMAINS == {
        "Graph coloring",
        "Countdown",
        "Python factors",
        "MathIR action menu",
        "Held-out MATH-500 transfer",
    }
    assert set(readiness_audit.REQUIRED_GATES) == {
        "e58_cross_domain",
        "e68_repair",
        "plumbing_consistency",
        "math500_realism",
    }


def test_e65_auditor_narrowly_classifies_registered_rtx2080_interrupt(
    tmp_path,
):
    log = tmp_path / "attempt.log"
    log.write_text(
        "Bfloat16 is only supported on GPUs with compute capability of at "
        "least 8.0. Your NVIDIA GeForce RTX 2080 Ti GPU has compute "
        "capability 7.5.\n"
        "[rank0]: Traceback (most recent call last):\n",
        encoding="utf-8",
    )
    failures, interruptions = audit._scan_log(log)
    assert failures == []
    assert interruptions == 1

    log.write_text(
        "[rank0]: Traceback (most recent call last):\n",
        encoding="utf-8",
    )
    failures, interruptions = audit._scan_log(log)
    assert failures
    assert interruptions == 0


def _audited_intervention_row(*, inverse_multiplier=1.1):
    return {
        "trainer/global_step": 100,
        "train/dummy": 0.0,
        "actor/counterfactual_proposal_singleton_entropy_gate_enabled": 1.0,
        "actor/counterfactual_proposal_singleton_entropy_gate_available": 1.0,
        "actor/counterfactual_proposal_singleton_entropy_gate_known_support": 1.0,
        "actor/counterfactual_proposal_singleton_entropy_gate_active": 1.0,
        "actor/counterfactual_proposal_singleton_entropy_gate_warmup_complete": 1.0,
        "actor/counterfactual_proposal_singleton_entropy_gate_below_reference": 1.0,
        "actor/counterfactual_proposal_singleton_entropy_gate_inverse_multiplier": inverse_multiplier,
        "actor/counterfactual_proposal_anchor_available": 1.0,
        "actor/counterfactual_proposal_admitted_new_outcomes": 1.0,
        "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
        "actor/counterfactual_proposal_transform_rows_sent_to_ppo": 0.0,
        "train/semantic_shannon_success_conditioned_signed_open_set_projection_active": 0.0,
    }


def test_e65_auditor_records_the_full_registered_gate_conjunction(tmp_path):
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text(
        json.dumps(_audited_intervention_row()) + "\n",
        encoding="utf-8",
    )
    result, violations = audit._scan_metrics(metrics, label="valid")
    assert violations == []
    assert result["interventions"] == 1
    assert result["maximum_admitted_per_group"] == 1
    assert result["intervention_events"] == [
        {
            "step": 100,
            "line_number": 1,
            "admitted_new_outcomes": 1,
            "known_support_before_admission": 1,
            "gate_enabled": 1.0,
            "gate_available": 1.0,
            "warmup_complete": 1.0,
            "entropy_below_reference": 1.0,
            "inverse_multiplier": 1.1,
            "gate_active": 1.0,
            "conditioned_rows_sent_to_ppo": 0.0,
            "transform_rows_sent_to_ppo": 0.0,
            "coefficient_projection_active": 0.0,
        }
    ]


def test_e65_auditor_rejects_admission_without_inverse_activation(tmp_path):
    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text(
        json.dumps(_audited_intervention_row(inverse_multiplier=1.0)) + "\n",
        encoding="utf-8",
    )
    _, violations = audit._scan_metrics(metrics, label="invalid")
    assert any("inverse multiplier > 1" in violation for violation in violations)


def test_frozen_auc_is_withheld_until_every_checkpoint_lands():
    incomplete = summary._metric_summary(
        _points(summary.CONTROL, (0, 1), 0.5),
        arm=summary.CONTROL,
        metric="pass8",
        fixed_passes=(0, 1, 2),
        steps_per_pass=1,
    )
    assert incomplete["partial_auc"] == 0.5
    assert incomplete["auc"] is None

    complete = summary._metric_summary(
        _points(summary.CONTROL, (0, 1, 2), 0.5),
        arm=summary.CONTROL,
        metric="pass8",
        fixed_passes=(0, 1, 2),
        steps_per_pass=1,
    )
    assert complete["auc"] == 1.0


def test_math_sensitivity_recomputes_saved_raw_trace_scores():
    prompts = []
    for prompt_index in range(500):
        rewards = [1.0] + [0.0] * 7 if prompt_index < 100 else [0.0] * 8
        prompts.append(
            {
                "prompt": f"problem {prompt_index}",
                "reference": f"answer {prompt_index}",
                "responses": [
                    f"response {prompt_index}/{option}" for option in range(8)
                ],
                "rewards": rewards,
                "answer_keys": [None] * 8,
            }
        )
    trace = {
        "step": 0,
        "evaluation_kind": "fixed_seed_sampled_k_neutral",
        "metrics": {
            "mean_at_k": 100 / (500 * 8),
            "any_correct_at_k": 0.2,
        },
        "prompts": prompts,
    }
    result, violations, response_rewards = math_sensitivity._scan_trace(
        trace,
        label="synthetic",
    )
    assert violations == []
    assert result["recomputed_mean_at_k"] == 0.025
    assert result["recomputed_any_correct_at_k"] == 0.2
    assert len(response_rewards) == 4000


def test_fixed_checkpoint_csv_contains_every_seed_and_registered_metric():
    domains = {}
    for domain, *_ in summary.MODEBENCH:
        points = [
            {
                "arm": arm,
                "seed": seed,
                "step": 0,
                "passes": 0,
                "greedy": 0.1,
                "mean8": 0.2,
                "pass8": 0.3,
                "distinct8": 0.4,
            }
            for arm in (
                summary.CONTROL,
                summary.TREATMENT,
                summary.PLUMBING_CONTROL,
                summary.REPAIR,
            )
            for seed in summary.SEEDS
        ]
        domains[domain] = summary._domain_summary(
            points,
            arms=(
                summary.CONTROL,
                summary.TREATMENT,
                summary.PLUMBING_CONTROL,
                summary.REPAIR,
            ),
            fixed_passes=(0,),
            steps_per_pass=1,
        )
    math_points = [
        {
            "arm": arm,
            "seed": seed,
            "step": 0,
            "passes": 0,
            "greedy": 0.1,
            "mean8": 0.2,
            "pass8": 0.3,
        }
        for arm in (summary.CONTROL, summary.TREATMENT)
        for seed in summary.SEEDS
    ]
    domains[summary.MATH_DOMAIN] = summary._domain_summary(
        math_points,
        arms=(summary.CONTROL, summary.TREATMENT),
        fixed_passes=(0,),
        steps_per_pass=1,
        metrics=("greedy", "mean8", "pass8"),
    )
    rows = list(
        csv.DictReader(
            io.StringIO(summary._render_checkpoint_csv({"domains": domains}))
        )
    )
    assert len(rows) == 70
    assert all(row["is_registered_checkpoint"] == "1" for row in rows)
    assert all(
        row["seed_43"] and row["seed_44"] and row["seed_45"]
        for row in rows
    )


def test_same_plumbing_control_audit_rejects_any_proposal_activity(tmp_path):
    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        json.dumps(
            {
                "trainer/global_step": 1,
                "train/loss": 0.0,
                "train/canonical_replay_mass_projection_active": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result, violations = plumbing_audit._scan_metrics(clean, label="clean")
    assert violations == []
    assert result["proposal_activity_records"] == 0

    dirty = tmp_path / "dirty.jsonl"
    dirty.write_text(
        json.dumps(
            {
                "trainer/global_step": 1,
                "train/loss": 0.0,
                "actor/counterfactual_proposal_enabled": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result, violations = plumbing_audit._scan_metrics(dirty, label="dirty")
    assert result["proposal_activity_records"] == 1
    assert any("proposal activity" in violation for violation in violations)


def test_corrected_actuator_audit_requires_literal_e58_novelty_objective(
    tmp_path,
):
    clean = tmp_path / "clean.jsonl"
    clean.write_text(
        json.dumps(
            {
                "trainer/global_step": 1,
                "train/loss": 0.0,
                "train/online_canonical_new_outcome_count": 1.0,
                "train/online_canonical_advantage_applied_after_task_centering": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result, violations = corrected_audit._scan_metrics(
        clean,
        label="clean",
    )
    assert violations == []
    assert result["objective_contract_records"] == 1

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps(
            {
                "trainer/global_step": 1,
                "train/loss": 0.0,
                "train/online_canonical_new_outcome_count": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _, violations = corrected_audit._scan_metrics(
        invalid,
        label="invalid",
    )
    assert any(
        "E58 novelty objective not applied" in violation
        for violation in violations
    )


def test_e68_audit_requires_zero_proposal_effect_on_objective_support(
    tmp_path,
):
    valid = _audited_intervention_row()
    valid.update(
        {
            "actor/counterfactual_proposal_enabled": 1.0,
            "actor/counterfactual_proposal_objective_support_separated": 1.0,
            "actor/counterfactual_proposal_objective_outcome_delta": 0.0,
        }
    )
    metrics = tmp_path / "valid-separated.jsonl"
    metrics.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    result, violations = corrected_audit._scan_metrics(
        metrics,
        label="valid-separated",
    )
    assert violations == []
    assert result["separation_contract_records"] == 1

    for key, value in (
        ("actor/counterfactual_proposal_objective_support_separated", 0.0),
        ("actor/counterfactual_proposal_objective_outcome_delta", 1.0),
    ):
        invalid = dict(valid)
        invalid[key] = value
        metrics.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        _, violations = corrected_audit._scan_metrics(
            metrics,
            label="invalid-separated",
        )
        assert violations


def test_preintervention_equivalence_detects_objective_mismatch():
    row = {
        key: 1.0
        for key in equivalence_audit.REQUIRED_KEYS
    }
    assert equivalence_audit._compare_rows(
        row,
        dict(row),
        label="same",
    ) == []
    mismatched = dict(row)
    mismatched["train/online_canonical_novelty_advantage_mean"] = 0.0
    assert any(
        "online_canonical_novelty_advantage_mean" in violation
        for violation in equivalence_audit._compare_rows(
            row,
            mismatched,
            label="different",
        )
    )


def test_repair_gate_requires_python_mean_worst_seed_and_clean_mechanism():
    domains = {}
    for domain, *_ in summary.MODEBENCH:
        points = []
        comparator_values = {43: 0.8, 44: 0.8, 45: 0.2}
        repair_values = {43: 0.85, 44: 0.85, 45: 0.3}
        if domain != "Python factors":
            comparator_values = {43: 0.8, 44: 0.8, 45: 0.8}
            repair_values = {43: 0.76, 44: 0.8, 45: 0.8}
        for arm, values in (
            (summary.PLUMBING_CONTROL, comparator_values),
            (summary.REPAIR, repair_values),
        ):
            points.extend(
                {
                    "arm": arm,
                    "seed": seed,
                    "step": 12,
                    "passes": 12,
                    "pass8": value,
                }
                for seed, value in values.items()
            )
        domains[domain] = summary._domain_summary(
            points,
            arms=(summary.PLUMBING_CONTROL, summary.REPAIR),
            fixed_passes=(12,),
            steps_per_pass=1,
            metrics=("pass8",),
        )
    gate = summary._corrected_repair_gate(
        domains,
        repair_terminal_runs=12,
        comparator_terminal_runs=12,
        interventions=1,
        repair_audit_status="pass",
        comparator_audit_status="pass",
        equivalence_audit_status="pass",
        checkpoint_audit_status="pass",
    )
    assert gate["status"] == "pass"

    no_mechanism = summary._corrected_repair_gate(
        domains,
        repair_terminal_runs=12,
        comparator_terminal_runs=12,
        interventions=0,
        repair_audit_status="pass",
        comparator_audit_status="pass",
        equivalence_audit_status="pass",
        checkpoint_audit_status="pass",
    )
    assert no_mechanism["status"] == "fail"

    bad_checkpoint = summary._corrected_repair_gate(
        domains,
        repair_terminal_runs=12,
        comparator_terminal_runs=12,
        interventions=1,
        repair_audit_status="pass",
        comparator_audit_status="pass",
        equivalence_audit_status="pass",
        checkpoint_audit_status="fail",
    )
    assert bad_checkpoint["status"] == "fail"
