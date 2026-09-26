from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import audit_e117_same_plumbing_component_preflight as audit  # noqa: E402
from oat_drgrpo.admission_retention import AdmissionRetentionTracker  # noqa: E402


SEMANTIC_PREFIX = "train/semantic_shannon_success_conditioned_verified_support_"


def retention(prefix: str, tracked: float) -> dict[str, float]:
    result = {prefix + field: 0.0 for field in audit.RETENTION_REQUIRED_FIELDS}
    result[prefix + "tracking_enabled"] = 1.0
    result[prefix + "tracked_admissions"] = tracked
    return result


def mechanism_row(
    arm: str,
    *,
    step: int = 1,
    candidates: float = 1.0,
    admitted: float | None = None,
    cumulative: float | None = None,
    raw_rms: float | None = None,
    effective_rms: float | None = None,
    canonical_view: bool = False,
) -> dict[str, float]:
    if admitted is None:
        admitted = 0.0 if arm == "c" else candidates
    if cumulative is None:
        cumulative = admitted
    if raw_rms is None:
        raw_rms = 0.05 if arm == "f" else 0.0
    if effective_rms is None:
        effective_rms = 0.025 if arm == "f" and raw_rms > 0 else 0.0
    row = {
        "actor/counterfactual_fixed_control_rows_sent_to_ppo": 0.0,
        "actor/counterfactual_fixed_control_groups_generated": 1.0,
        "actor/counterfactual_fixed_control_rows_generated": 16.0,
        "actor/counterfactual_fixed_control_charged_response_token_budget": 3072.0,
        "actor/counterfactual_fixed_control_request_seed_min": float(20_000 + step),
        "actor/counterfactual_fixed_control_request_seed_max": float(20_000 + step),
        "actor/counterfactual_fixed_control_groups_consumed_by_explorer": 1.0,
        "actor/counterfactual_fixed_control_groups_discarded": 0.0,
        "actor/counterfactual_proposal_groups_generated": 1.0,
        "actor/counterfactual_proposal_rows_generated": 16.0,
        "actor/counterfactual_proposal_request_seed_min": float(20_000 + step),
        "actor/counterfactual_proposal_request_seed_max": float(20_000 + step),
        "actor/counterfactual_proposal_neutral_ppo_rows": 16.0,
        "actor/counterfactual_proposal_neutral_task_reward_positive_rows": 0.0,
        "actor/rewards": 0.0,
        "actor/counterfactual_proposal_conditioned_rows_sent_to_ppo": 0.0,
        "actor/counterfactual_proposal_transform_rows_sent_to_ppo": 0.0,
        "actor/counterfactual_proposal_objective_outcome_delta": 0.0,
        "actor/counterfactual_proposal_gold_support_feedback": 0.0,
        "actor/counterfactual_proposal_eval_feedback": 0.0,
        "actor/counterfactual_proposal_desired_mode_count_feedback": 0.0,
        "actor/counterfactual_proposal_admission_compute_only": (
            1.0 if arm == "c" else 0.0
        ),
        "actor/counterfactual_proposal_novel_unique_outcomes": candidates,
        "actor/counterfactual_proposal_admitted_new_outcomes": admitted,
        "actor/counterfactual_proposal_stored_exemplars": admitted,
        "actor/counterfactual_proposal_candidates_discarded": (
            candidates if arm == "c" else 0.0
        ),
        "actor/counterfactual_proposal_cumulative_new_outcomes": cumulative,
        "actor/sampling_request_seed": float(10_000 + step),
        SEMANTIC_PREFIX + "verified_support_at_least_two_eligible_fraction": 1.0,
        SEMANTIC_PREFIX + "raw_eligible_advantage_rms": raw_rms,
        SEMANTIC_PREFIX + "effective_advantage_rms": effective_rms,
        SEMANTIC_PREFIX
        + "open_set_coefficient_used": (0.10000000149011612 if arm == "f" else 0.0),
    }
    tracked = 0.0 if arm == "c" else cumulative
    row.update(retention(audit.RETENTION_PREFIXES[0], tracked))
    if canonical_view:
        row.update(retention(audit.RETENTION_PREFIXES[1], tracked))
    return row


@pytest.mark.parametrize("arm", audit.e117.ARMS)
def test_valid_arm_mechanism_rows_pass(arm):
    summary, failures = audit.audit_arm_rows(arm, {1: mechanism_row(arm)})
    assert failures == []
    assert summary["proposal_admissions"] == (0.0 if arm == "c" else 1.0)
    assert summary["terminal_tracked_admissions"] == (0.0 if arm == "c" else 1.0)


def test_replay_retention_view_is_optional_but_strict_when_present():
    row = mechanism_row("p", canonical_view=True)
    _, failures = audit.audit_arm_rows("p", {1: row})
    assert failures == []
    del row[audit.RETENTION_PREFIXES[1] + "tracking_enabled"]
    with pytest.raises(ValueError, match="required metric is absent"):
        audit.audit_arm_rows("p", {1: row})


def test_retention_subset_and_fraction_inconsistency_is_detected():
    row = mechanism_row("p")
    prefix = audit.RETENTION_PREFIXES[0]
    row[prefix + "rollout_converted_admissions"] = 1.0
    _, failures = audit.audit_arm_rows("p", {1: row})
    assert any("inconsistent retention subsets" in value for value in failures)

    row[prefix + "rollout_eligible_admissions"] = 1.0
    _, failures = audit.audit_arm_rows("p", {1: row})
    assert any("inconsistent retention fraction" in value for value in failures)


def test_required_retention_fields_equal_the_runtime_schema():
    runtime_fields = set(AdmissionRetentionTracker().diagnostics())
    assert set(audit.RETENTION_REQUIRED_FIELDS) == runtime_fields


def test_required_telemetry_is_fail_closed():
    row = mechanism_row("c")
    del row[audit.RETENTION_PREFIXES[0] + "tracking_enabled"]
    with pytest.raises(ValueError, match="required metric is absent"):
        audit.audit_arm_rows("c", {1: row})


def test_c_mutation_and_p_admission_failure_are_detected():
    c_row = mechanism_row("c")
    c_row["actor/counterfactual_proposal_admitted_new_outcomes"] = 1.0
    c_row["actor/counterfactual_proposal_stored_exemplars"] = 1.0
    c_row["actor/counterfactual_proposal_candidates_discarded"] = 0.0
    c_row["actor/counterfactual_proposal_cumulative_new_outcomes"] = 1.0
    c_row[audit.RETENTION_PREFIXES[0] + "tracked_admissions"] = 1.0
    _, c_failures = audit.audit_arm_rows("c", {1: c_row})
    assert any("C did not discard exactly" in value for value in c_failures)
    assert any("C proposal state mutated" in value for value in c_failures)
    assert any("C retention state mutated" in value for value in c_failures)

    p_row = mechanism_row("p", admitted=0.0, cumulative=0.0)
    _, p_failures = audit.audit_arm_rows("p", {1: p_row})
    assert any(
        "did not admit/store every novel candidate" in value for value in p_failures
    )


def test_semantic_actuator_and_coefficient_failures_are_detected():
    f_row = mechanism_row("f", effective_rms=0.0)
    _, f_failures = audit.audit_arm_rows("f", {1: f_row})
    assert any("actuable F update had zero" in value for value in f_failures)

    p_row = mechanism_row("p", effective_rms=0.1)
    p_row[SEMANTIC_PREFIX + "open_set_coefficient_used"] = 0.1
    _, p_failures = audit.audit_arm_rows("p", {1: p_row})
    assert any("semantic pressure was nonzero" in value for value in p_failures)
    assert any("semantic coefficient drifted" in value for value in p_failures)


def test_resume_counter_and_request_seed_discontinuities_are_detected():
    first = mechanism_row("p", step=1, candidates=1.0, cumulative=1.0)
    second = mechanism_row("p", step=2, candidates=0.0, admitted=0.0, cumulative=0.0)
    second["actor/sampling_request_seed"] = first["actor/sampling_request_seed"]
    second["actor/counterfactual_fixed_control_request_seed_min"] = first[
        "actor/counterfactual_fixed_control_request_seed_min"
    ]
    second["actor/counterfactual_fixed_control_request_seed_max"] = first[
        "actor/counterfactual_fixed_control_request_seed_max"
    ]
    second["actor/counterfactual_proposal_request_seed_min"] = first[
        "actor/counterfactual_proposal_request_seed_min"
    ]
    second["actor/counterfactual_proposal_request_seed_max"] = first[
        "actor/counterfactual_proposal_request_seed_max"
    ]
    _, failures = audit.audit_arm_rows("p", {1: first, 2: second})
    assert any("request seeds repeated" in value for value in failures)
    assert any("fixed-control request seeds repeated" in value for value in failures)
    assert any("proposal admission counter decreased" in value for value in failures)
    assert any("retention counter decreased" in value for value in failures)


def launch_ledger(
    snapshot: Path = Path("/immutable/e117-training-snapshot"),
    *,
    launch_layout: str = "original",
) -> tuple[dict, dict[int, dict]]:
    ledger = {
        "sentinels": [
            {"scale": scale, "domain": domain, "node": node}
            for (scale, domain), node in audit.e117.SENTINEL_NODES.items()
        ],
        "effective_nodes": {},
        "snapshot_root": str(snapshot),
        "runs": [],
    }
    accounting = {}
    job_id = 1
    for scale, domain in audit.e117.SENTINELS:
        node = audit.e117.SENTINEL_NODES[(scale, domain)]
        for arm in audit.e117.ARMS:
            environment_map = {
                "SAVE_PATH": f"/tmp/{scale}-{domain}-{arm}",
                "RUN_STAMP": f"{scale}-{domain}-{arm}",
                "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
                "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
                "OAT_ZERO_SEED": str(audit.e117.SEED),
                "OAT_ZERO_MAX_TRAIN": str(audit.e117.TRAIN_ROWS),
                "OAT_ZERO_NUM_PROMPT_EPOCH": str(audit.e117.PASSES),
                "OAT_ZERO_MAX_PROMPT_EPOCHS": str(audit.e117.PASSES),
                "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(audit.e117.TARGET_STEPS),
                "OAT_ZERO_SAVE_STEPS": str(audit.e117.CHECKPOINT_INTERVAL),
                "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(audit.e117.EVAL_DRAWS),
                "COMMON_SCIENCE": "frozen",
            }
            environment_map.update(audit.e117.objective_for_arm(arm))
            environment = "ALL," + ",".join(
                f"{key}={value}" for key, value in environment_map.items()
            )
            if launch_layout == "original":
                held_scheduler_record = (
                    f"SubmitLine=sbatch --export={environment} "
                    "--partition=lowprio --wrap=true"
                )
            elif launch_layout == "recovery":
                held_scheduler_record = (
                    "JobId=1 Command=/immutable/train.slurm "
                    "SubmitLine=sbatch --partition=all --account=allcs "
                    f"--time=01:00:00 --export={environment} "
                    "/immutable/train.slurm WorkDir=/immutable"
                )
            else:
                raise ValueError(f"unknown launch layout: {launch_layout}")
            ledger["runs"].append(
                {
                    "scale": scale,
                    "domain": domain,
                    "arm": arm,
                    "job_id": job_id,
                    "scientific_environment_sha256": hashlib.sha256(
                        environment.encode()
                    ).hexdigest(),
                    "held_scheduler_record": held_scheduler_record,
                }
            )
            accounting[job_id] = {"node_list": node}
            job_id += 1
    return ledger, accounting


def test_launch_blocks_verify_actual_nodes_and_only_two_science_differences():
    ledger, accounting = launch_ledger()
    assert audit.validate_launch_blocks(ledger, accounting) == []

    drifted = deepcopy(ledger)
    drifted["runs"][-1]["held_scheduler_record"] = drifted["runs"][-1][
        "held_scheduler_record"
    ].replace("COMMON_SCIENCE=frozen", "COMMON_SCIENCE=drifted")
    failures = audit.validate_launch_blocks(drifted, accounting)
    assert any("scientific exports differ" in value for value in failures)

    accounting[1]["node_list"] = "node999"
    failures = audit.validate_launch_blocks(ledger, accounting)
    assert any("actual node" in value for value in failures)


@pytest.mark.parametrize("launch_layout", ("original", "recovery"))
def test_launch_blocks_accept_original_and_recovery_argument_order(launch_layout):
    ledger, accounting = launch_ledger(launch_layout=launch_layout)
    assert audit.validate_launch_blocks(ledger, accounting) == []


@pytest.mark.parametrize(
    ("record", "expected"),
    (
        (
            "SubmitLine=sbatch --export=ALL,A=1,B=two --partition=all train.slurm",
            "ALL,A=1,B=two",
        ),
        (
            "JobId=1 SubmitLine=sbatch --partition=all --account=allcs "
            "--export=ALL,A=1,B=two /immutable/train.slurm WorkDir=/immutable",
            "ALL,A=1,B=two",
        ),
        (
            "SubmitLine=sbatch --partition=all --export 'ALL,A=1,B=two' train.slurm",
            "ALL,A=1,B=two",
        ),
    ),
)
def test_export_parser_is_token_aware(record, expected):
    assert audit.exported_environment_text(record) == expected


@pytest.mark.parametrize(
    ("record", "message"),
    (
        ("JobId=1 WorkDir=/tmp", "exactly one SubmitLine"),
        ("SubmitLine=sbatch --partition=all train.slurm", "exactly one --export"),
        ("SubmitLine=sbatch --export= train.slurm", "argument is empty"),
        ("SubmitLine=sbatch --export train.slurm", "malformed scheduler export"),
        (
            "SubmitLine=sbatch --export=ALL,A=1 --export=ALL,A=1 train.slurm",
            "exactly one --export",
        ),
        ("SubmitLine=sbatch --export='ALL,A=1 train.slurm", "tokenization failed"),
    ),
)
def test_export_parser_fails_closed_on_ambiguous_records(record, message):
    with pytest.raises(ValueError, match=message):
        audit.exported_environment(record)


def optimizer_metric_rows() -> list[dict[str, float]]:
    rows = []
    for step in range(1, audit.e117.TARGET_STEPS + 1):
        rows.append(
            {
                "misc/global_step": float(step),
                "trainer/step": float(step),
                "trainer/global_step": float(step),
                "misc/policy_sgd_step": float(step),
                "misc/query_step": float(step),
                "misc/prompt_consumed": float(step * 16),
                "misc/pi_beta_version": 0.0,
                "actor/sampling_request_seed": float(10_000 + step),
                "actor/sentinel_state": float(step),
                "train/sentinel_state": float(step),
            }
        )
    terminal = dict(rows[-1])
    terminal["trainer/step"] = float(audit.e117.TARGET_STEPS + 1)
    rows.append(terminal)
    return rows


def write_optimizer_metric_rows(tmp_path: Path, rows: list[dict[str, float]]) -> None:
    (tmp_path / "train_metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_training_rows_accepts_one_validated_terminal_sentinel(tmp_path):
    write_optimizer_metric_rows(tmp_path, optimizer_metric_rows())
    selected = audit.training_rows(tmp_path)
    assert list(selected) == list(range(1, audit.e117.TARGET_STEPS + 1))
    assert selected[audit.e117.TARGET_STEPS]["trainer/step"] == float(
        audit.e117.TARGET_STEPS
    )


def test_training_rows_rejects_duplicate_canonical_optimizer_row(tmp_path):
    rows = optimizer_metric_rows()
    rows.insert(-1, dict(rows[-2]))
    write_optimizer_metric_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="duplicate canonical"):
        audit.training_rows(tmp_path)


def test_training_rows_rejects_mutated_or_missing_terminal_sentinel(tmp_path):
    rows = optimizer_metric_rows()
    rows[-1]["train/sentinel_state"] += 1.0
    write_optimizer_metric_rows(tmp_path, rows)
    with pytest.raises(ValueError, match="mutated mechanism field"):
        audit.training_rows(tmp_path)

    write_optimizer_metric_rows(tmp_path, optimizer_metric_rows()[:-1])
    with pytest.raises(ValueError, match="exactly one validated terminal"):
        audit.training_rows(tmp_path)


def test_registered_design_and_step_zero_draw_identity(tmp_path):
    ledger = {
        "schema": "e117_same_plumbing_component_preflight_jobs_v1",
        "released": True,
        "seed": audit.e117.SEED,
        "arms": list(audit.e117.ARMS),
        "train_rows": audit.e117.TRAIN_ROWS,
        "passes": audit.e117.PASSES,
        "target_steps": audit.e117.TARGET_STEPS,
        "checkpoint_interval_steps": audit.e117.CHECKPOINT_INTERVAL,
        "evaluation_draws": audit.e117.EVAL_DRAWS,
        "proposal_fixed_control_groups": audit.e117.PROPOSAL_GROUPS,
        "proposal_max_attempts": audit.e117.e111.PROPOSAL_MAX_ATTEMPTS,
        "proposal_temperature": audit.e117.e111.PROPOSAL_TEMPERATURE,
        "replay_weight": audit.e117.e111.REPLAY_WEIGHT,
        "semantic_coefficient_f": audit.e117.SEMANTIC_COEFFICIENT,
        "efficacy_gate": False,
        "outcomes_inspected_for_release": False,
        "pointmaze": "excluded",
    }
    assert audit.validate_registered_design(ledger) == []
    ledger["target_steps"] = 65
    assert audit.validate_registered_design(ledger)

    greedy = {
        "step": 0,
        "draw_index": None,
        "evaluation_kind": "deterministic_greedy_trace_neutral",
        "metrics": {"pass": 0.0},
        "prompts": [{"prompt_index": 0}],
    }
    sampled = {
        "step": 0,
        "draw_index": 0,
        "evaluation_kind": "fixed_seed_sampled_k_neutral",
        "metrics": {"pass": 0.0},
        "prompts": [{"prompt_index": 0}],
    }
    terminal_greedy = dict(greedy, step=audit.e117.TARGET_STEPS)
    terminal_sampled = dict(sampled, step=audit.e117.TARGET_STEPS)
    serialized = (
        "\n".join(
            json.dumps(row)
            for row in (greedy, sampled, terminal_greedy, terminal_sampled)
        )
        + "\n"
    )
    path = tmp_path / "eval_mode_coverage_draws.jsonl"
    path.write_text(serialized, encoding="utf-8")
    assert audit.step_zero_draws(tmp_path) == [greedy, sampled]
    sampled["evaluation_kind"] = "wrong"
    serialized = (
        "\n".join(
            json.dumps(row)
            for row in (greedy, sampled, terminal_greedy, terminal_sampled)
        )
        + "\n"
    )
    path.write_text(serialized, encoding="utf-8")
    with pytest.raises(ValueError, match="sampled draws at step 0"):
        audit.step_zero_draws(tmp_path)


def test_exact_values_fail_when_all_arms_omit_the_same_metric():
    arm_rows = {arm: {1: {}} for arm in audit.e117.ARMS}
    failures = audit.exact_values(arm_rows, step=1, keys=("missing",))
    assert len(failures) == 3
    assert all("required metric is absent" in value for value in failures)


def test_conditional_identity_fields_allow_all_absent_but_not_partial_presence():
    arm_rows = {arm: {1: {}} for arm in audit.e117.ARMS}
    assert audit.conditional_exact_values(arm_rows, step=1, keys=("conditional",)) == []
    arm_rows["c"][1]["conditional"] = 1.0
    failures = audit.conditional_exact_values(arm_rows, step=1, keys=("conditional",))
    assert len(failures) == 1
    assert "presence" in failures[0]


def test_stage1_readiness_is_separate_from_identity_pass():
    blocks = [
        {
            "scale": "qwen05b",
            "domain": "countdown",
            "arms": {
                "p": {"proposal_admissions": 1.0},
                "f": {"semantic_active_updates": 2.0},
            },
        }
    ]
    assert audit.stage1_execution_readiness(blocks, identity_passed=True)["ready"]
    empty = audit.stage1_execution_readiness([], identity_passed=True)
    assert not empty["ready"]
    assert len(empty["blockers"]) == 2
    failed = audit.stage1_execution_readiness(blocks, identity_passed=False)
    assert not failed["ready"]
    assert "identity/leakage" in failed["blockers"][0]


def test_noncompleted_dependency_writes_durable_failed_audit(tmp_path, monkeypatch):
    runs = []
    job_id = 1
    for scale, domain in audit.e117.SENTINELS:
        for arm in audit.e117.ARMS:
            runs.append(
                {
                    "scale": scale,
                    "domain": domain,
                    "arm": arm,
                    "job_id": job_id,
                }
            )
            job_id += 1
    ledger = {
        "schema": "e117_same_plumbing_component_preflight_jobs_v1",
        "released": True,
        "seed": audit.e117.SEED,
        "arms": list(audit.e117.ARMS),
        "train_rows": audit.e117.TRAIN_ROWS,
        "passes": audit.e117.PASSES,
        "target_steps": audit.e117.TARGET_STEPS,
        "checkpoint_interval_steps": audit.e117.CHECKPOINT_INTERVAL,
        "evaluation_draws": audit.e117.EVAL_DRAWS,
        "proposal_fixed_control_groups": audit.e117.PROPOSAL_GROUPS,
        "proposal_max_attempts": audit.e117.e111.PROPOSAL_MAX_ATTEMPTS,
        "proposal_temperature": audit.e117.e111.PROPOSAL_TEMPERATURE,
        "replay_weight": audit.e117.e111.REPLAY_WEIGHT,
        "semantic_coefficient_f": audit.e117.SEMANTIC_COEFFICIENT,
        "efficacy_gate": False,
        "outcomes_inspected_for_release": False,
        "pointmaze": "excluded",
        "runs": runs,
    }
    ledger_path = tmp_path / "ledger.json"
    output_path = tmp_path / "audit.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(
        audit,
        "scheduler_status",
        lambda job_ids: {
            value: {
                "state": "FAILED",
                "exit_code": "1:0",
                "restarts": 0,
                "node_list": "node021",
            }
            for value in job_ids
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--ledger", str(ledger_path), "--output", str(output_path)],
    )
    assert audit.main() == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "e117_same_plumbing_component_preflight_audit_v3"
    assert payload["terminal"] is False
    assert payload["passed"] is False
    assert payload["stage1_execution_readiness"]["ready"] is False


def test_complete_synthetic_campaign_passes_full_audit(tmp_path, monkeypatch):
    snapshot = tmp_path / "training_snapshot"
    snapshot.mkdir()
    snapshot_identity = snapshot / "SNAPSHOT_IDENTITY.json"
    snapshot_identity.write_text('{"sha256":"frozen"}\n', encoding="utf-8")
    original_ledger = tmp_path / "original_ledger.json"
    original_ledger.write_text('{"released":true}\n', encoding="utf-8")

    ledger, accounting = launch_ledger(snapshot)
    ledger.update(
        {
            "schema": "e117_same_plumbing_component_preflight_jobs_v1",
            "released": True,
            "seed": audit.e117.SEED,
            "arms": list(audit.e117.ARMS),
            "train_rows": audit.e117.TRAIN_ROWS,
            "passes": audit.e117.PASSES,
            "target_steps": audit.e117.TARGET_STEPS,
            "checkpoint_interval_steps": audit.e117.CHECKPOINT_INTERVAL,
            "evaluation_draws": audit.e117.EVAL_DRAWS,
            "proposal_fixed_control_groups": audit.e117.PROPOSAL_GROUPS,
            "proposal_max_attempts": audit.e117.e111.PROPOSAL_MAX_ATTEMPTS,
            "proposal_temperature": audit.e117.e111.PROPOSAL_TEMPERATURE,
            "replay_weight": audit.e117.e111.REPLAY_WEIGHT,
            "semantic_coefficient_f": audit.e117.SEMANTIC_COEFFICIENT,
            "efficacy_gate": False,
            "outcomes_inspected_for_release": False,
            "pointmaze": "excluded",
            "snapshot_identity_sha256": audit.e117.digest(snapshot_identity),
            "original_ledger": str(original_ledger),
            "original_ledger_sha256": audit.e117.digest(original_ledger),
        }
    )
    for run in ledger["runs"]:
        arm = run["arm"]
        run_dir = tmp_path / f"run-{run['job_id']}"
        debug = run_dir / f"debug_job{run['job_id']}"
        debug.mkdir(parents=True)
        (run_dir / "TRAINING_COMPLETE.json").write_text("{}\n", encoding="utf-8")
        train_rows = []
        for step in range(1, audit.e117.TARGET_STEPS + 1):
            row = mechanism_row(
                arm,
                step=step,
                candidates=1.0,
                cumulative=(0.0 if arm == "c" else float(step)),
            )
            row.update(
                {
                    "misc/global_step": float(step),
                    "trainer/step": float(step),
                    "trainer/global_step": float(step),
                    "misc/policy_sgd_step": float(step),
                    "misc/query_step": float(step),
                    "misc/prompt_consumed": float(step * 16),
                    "misc/pi_beta_version": 0.0,
                    "actor/counterfactual_fixed_control_sampling_temperature_min": 1.2,
                    "actor/counterfactual_fixed_control_sampling_temperature_max": 1.2,
                    "actor/counterfactual_proposal_task_reward_positive_rows": 1.0,
                    "actor/counterfactual_proposal_validator_positive_rows": 1.0,
                    "actor/counterfactual_proposal_novel_candidate_rows": 1.0,
                }
            )
            train_rows.append(row)
        terminal_sentinel = dict(train_rows[-1])
        terminal_sentinel["trainer/step"] = float(audit.e117.TARGET_STEPS + 1)
        train_rows.append(terminal_sentinel)
        (debug / "train_metrics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in train_rows) + "\n",
            encoding="utf-8",
        )
        evaluation_rows = []
        for step in (0, audit.e117.TARGET_STEPS):
            evaluation_rows.extend(
                [
                    {
                        "schema_version": 1,
                        "evaluation_kind": "deterministic_greedy_trace_neutral",
                        "benchmark": run["domain"],
                        "step": step,
                        "draw_index": None,
                        "seed": 0,
                        "sample_count": 1,
                        "temperature": 0.0,
                        "metrics": {"pass": float(arm == "f" and step > 0)},
                        "prompts": [{"prompt_index": 0}],
                    },
                    {
                        "schema_version": 1,
                        "evaluation_kind": "fixed_seed_sampled_k_neutral",
                        "benchmark": run["domain"],
                        "step": step,
                        "draw_index": 0,
                        "seed": 610000,
                        "sample_count": 8,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "metrics": {"pass": float(arm == "f" and step > 0)},
                        "prompts": [{"prompt_index": 0}],
                    },
                ]
            )
        (debug / "eval_mode_coverage_draws.jsonl").write_text(
            "\n".join(json.dumps(row) for row in evaluation_rows) + "\n",
            encoding="utf-8",
        )
        run["run_dir"] = str(run_dir)
        accounting[int(run["job_id"])].update(
            {"state": "COMPLETED", "exit_code": "0:0", "restarts": 0}
        )

    ledger_path = tmp_path / "ledger.json"
    output_path = tmp_path / "audit.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(audit, "scheduler_status", lambda job_ids: accounting)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit", "--ledger", str(ledger_path), "--output", str(output_path)],
    )
    assert audit.main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "e117_same_plumbing_component_preflight_audit_v3"
    assert payload["terminal"] is True
    assert payload["passed"] is True
    assert payload["failures"] == []
    assert payload["stage1_execution_readiness"]["ready"] is True
    assert payload["resume_continuity"] == "unexercised"
