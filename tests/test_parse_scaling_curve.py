import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "ops" / "exp_scaling" / "parse_scaling_curve.py"
_SPEC = importlib.util.spec_from_file_location("parse_scaling_curve", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
parse_run = _MODULE.parse_run


def _record(step: int, coverage: float) -> dict:
    record = {
        "trainer/global_step": step,
        "misc/prompt_epoch": step / 10,
    }
    for split in ("multi_answer", "unique_answer"):
        record[f"eval/{split}/sampled_any_correct_at_8"] = coverage
        record[f"eval/{split}/sampled_mean_at_8"] = coverage
        record[f"eval/{split}/sampled_mode_coverage_at_8"] = coverage
        record[f"eval/{split}/sampled_distinct_correct_at_8"] = coverage
        record[f"eval/{split}/accuracy"] = coverage
    return record


def _math_record(step: int, score: float) -> dict:
    return {
        "trainer/global_step": step,
        "misc/prompt_epoch": step / 10,
        "eval/math/sampled_any_correct_at_8": score,
        "eval/math/sampled_mean_at_8": score / 2,
        "eval/math/sampled_distinct_correct_at_8": score,
        "eval/math/accuracy": score / 4,
    }


def _mathir_record(step: int, score: float) -> dict:
    return {
        "trainer/global_step": step,
        "misc/prompt_epoch": step / 10,
        "eval/multi_answer/sampled_any_correct_at_16": score,
        "eval/multi_answer/sampled_mean_at_16": score / 2,
        "eval/multi_answer/sampled_distinct_correct_at_16": score * 3,
        "eval/multi_answer/sampled_any_nonseed_correct_at_16": score / 4,
        "eval/multi_answer/sampled_distinct_nonseed_correct_at_16": score / 5,
        "eval/multi_answer/accuracy": score / 8,
    }


def _attempt(run_dir, name: str, points: list[tuple[int, float]]) -> None:
    debug_dir = run_dir / name
    debug_dir.mkdir()
    with (debug_dir / "train_metrics.jsonl").open("w") as handle:
        for step, coverage in points:
            handle.write(json.dumps(_record(step, coverage)) + "\n")


def _attempt_with_prompt_progress(
    run_dir,
    name: str,
    points: list[tuple[int, float, float]],
) -> None:
    debug_dir = run_dir / name
    debug_dir.mkdir()
    with (debug_dir / "train_metrics.jsonl").open("w") as handle:
        for step, prompt_consumed, coverage in points:
            record = _record(step, coverage)
            record["misc/prompt_consumed"] = prompt_consumed
            handle.write(json.dumps(record) + "\n")


def _multi_answer(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["split"] == "multi_answer"]


def test_parse_run_preserves_mathir_k16_and_nonseed_metrics(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(_mathir_record(10, 0.8)) + "\n",
        encoding="utf-8",
    )

    row = _multi_answer(parse_run(tmp_path))[0]

    assert row["pass16"] == 0.8
    assert row["mean16"] == 0.4
    assert abs(row["distinct16"] - 2.4) < 1e-12
    assert row["nonseed_pass16"] == 0.2
    assert row["nonseed_distinct16"] == 0.16
    assert row["pass8"] is None


def test_parse_run_keeps_predecessor_evaluation_at_duplicate_resume_boundary(
    tmp_path,
):
    _attempt(
        tmp_path,
        "debug_01",
        [(0, 0.0), (10, 0.1), (20, 0.2), (30, 0.3)],
    )
    _attempt(tmp_path, "debug_02", [(20, 2.0), (30, 3.0), (40, 4.0)])

    rows = _multi_answer(parse_run(tmp_path))

    assert [row["step"] for row in rows] == [0, 10, 20, 30, 40]
    assert [row["coverage8"] for row in rows] == [0.0, 0.1, 0.2, 3.0, 4.0]


def test_parse_run_retains_raw_evaluation_draws_and_uncertainty(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    record = _record(10, 0.25)
    metric_key = "eval/multi_answer/sampled_mode_coverage_at_8"
    record.update(
        {
            f"{metric_key}_draw_0": 0.20,
            f"{metric_key}_draw_1": 0.22,
            f"{metric_key}_draw_2": 0.27,
            f"{metric_key}_draw_3": 0.31,
            f"{metric_key}_draw_std": 0.05,
            f"{metric_key}_draw_se": 0.025,
            f"{metric_key}_draw_min": 0.20,
            f"{metric_key}_draw_max": 0.31,
        }
    )
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    row = _multi_answer(parse_run(tmp_path))[0]

    assert row["coverage8"] == 0.25
    assert row["coverage8_draws"] == [0.20, 0.22, 0.27, 0.31]
    assert row["coverage8_draw_se"] == 0.025
    assert row["coverage8_draw_min"] == 0.20
    assert row["coverage8_draw_max"] == 0.31


def test_parse_run_copies_namespaced_outcome_collision_telemetry(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    record = _record(10, 0.25)
    record.update(
        {
            "train/outcome_collision_rate": 0.375,
            "train/outcome_collision_distinct_fraction": 0.625,
            "train/outcome_collision_invalid_fraction": 0.125,
            "train/outcome_collision_parseable_fraction": 0.875,
            "train/outcome_collision_bonus_mean": -0.0375,
        }
    )
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    row = _multi_answer(parse_run(tmp_path))[0]

    assert row["outcome_collision_rate"] == 0.375
    assert row["outcome_collision_distinct_fraction"] == 0.625
    assert row["outcome_collision_invalid_fraction"] == 0.125
    assert row["outcome_collision_parseable_fraction"] == 0.875
    assert row["outcome_collision_bonus_mean"] == -0.0375


def test_parse_run_copies_namespaced_semantic_shannon_telemetry(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    record = _record(10, 0.25)
    record.update(
        {
            "train/semantic_shannon_normalized_surprisal_mean": 0.625,
            "train/semantic_shannon_entropy_mean": 1.75,
            "train/semantic_shannon_normalization_error_max": 2.2e-16,
            "train/semantic_shannon_unseen_fraction": 0.125,
            "train/semantic_shannon_parseable_fraction": 0.875,
            "train/semantic_shannon_bonus_mean": -0.0375,
        }
    )
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    row = _multi_answer(parse_run(tmp_path))[0]

    assert row["semantic_shannon_normalized_surprisal_mean"] == 0.625
    assert row["semantic_shannon_entropy_mean"] == 1.75
    assert row["semantic_shannon_normalization_error_max"] == 2.2e-16
    assert row["semantic_shannon_unseen_fraction"] == 0.125
    assert row["semantic_shannon_parseable_fraction"] == 0.875
    assert row["semantic_shannon_bonus_mean"] == -0.0375


def test_parse_run_binds_e44_online_canonical_optimizer_telemetry(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    telemetry = {
        "trainer/global_step": 9,
        "misc/prompt_consumed": 144,
        "train/online_canonical_advantage_applied_after_task_centering": 1.0,
        "train/online_canonical_entropy_estimate_mean": 1.25,
        "train/online_canonical_normalized_entropy_ratio_mean": 0.81,
        "train/online_canonical_normalized_entropy_ratio_eligible_fraction": 0.75,
        "train/online_canonical_entropy_alpha_used": 0.12,
        "train/online_canonical_dual_next_alpha": 0.121,
        "train/online_canonical_policy_entropy_observed": 0.75,
        "train/online_canonical_policy_entropy_ema": 0.80,
        "train/online_canonical_policy_entropy_reference": 1.00,
        "train/online_canonical_policy_entropy_normalized_score": 0.80,
        "train/online_canonical_policy_entropy_next_alpha": 0.08,
        "train/maxent_conditional_token_entropy": 1.5,
        "train/maxent_entropy_loss": -0.0001125,
        "train/maxent_alpha_used": 0.000075,
        "train/maxent_inverse_entropy_ema": 1.4,
        "train/maxent_inverse_multiplier": 1.1,
        "train/maxent_inverse_next_alpha": 0.0000825,
        "train/maxent_inverse_projection_active": 0.0,
        "train/maxent_inverse_reference_entropy": 1.54,
        "train/online_canonical_entropy_advantage_rms": 0.08,
        "train/online_canonical_novelty_advantage_rms": 0.20,
        "train/online_canonical_combined_advantage_rms": 0.22,
        "train/online_canonical_eligible_fraction": 0.75,
        "train/online_canonical_new_outcome_count": 3.0,
        "train/online_canonical_bank_size_after_mean": 4.5,
        "train/online_canonical_tracked_outcomes": 27.0,
        "train/online_canonical_tracked_prompts": 9.0,
        "train/online_canonical_separate_base_advantage_rms": 0.44,
        "train/math_strategy_rejected_integrity_rows": 2.0,
    }
    evaluation = _record(10, 0.25)
    evaluation["misc/prompt_consumed"] = 160
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(telemetry) + "\n" + json.dumps(evaluation) + "\n",
        encoding="utf-8",
    )

    row = _multi_answer(
        parse_run(tmp_path, prompt_pool_size=10, num_samples=16)
    )[0]

    assert row["training_passes"] == 1.0
    assert row["online_canonical_entropy_estimate_mean"] == 1.25
    assert row["online_canonical_normalized_entropy_ratio_mean"] == 0.81
    assert (
        row["online_canonical_normalized_entropy_ratio_eligible_fraction"]
        == 0.75
    )
    assert row["online_canonical_entropy_alpha_used"] == 0.12
    assert row["online_canonical_dual_next_alpha"] == 0.121
    assert row["online_canonical_policy_entropy_observed"] == 0.75
    assert row["online_canonical_policy_entropy_reference"] == 1.00
    assert row["online_canonical_policy_entropy_next_alpha"] == 0.08
    assert row["maxent_conditional_token_entropy"] == 1.5
    assert row["maxent_entropy_loss"] == -0.0001125
    assert row["maxent_alpha_used"] == 0.000075
    assert row["maxent_inverse_entropy_ema"] == 1.4
    assert row["maxent_inverse_multiplier"] == 1.1
    assert row["maxent_inverse_next_alpha"] == 0.0000825
    assert row["maxent_inverse_projection_active"] == 0.0
    assert row["maxent_inverse_reference_entropy"] == 1.54
    assert row["online_canonical_entropy_advantage_rms"] == 0.08
    assert row["online_canonical_novelty_advantage_rms"] == 0.20
    assert row["online_canonical_combined_advantage_rms"] == 0.22
    assert row["online_canonical_eligible_fraction"] == 0.75
    assert row["online_canonical_new_outcome_count"] == 3.0
    assert row["online_canonical_bank_size_after_mean"] == 4.5
    assert row["online_canonical_tracked_outcomes"] == 27.0
    assert row["online_canonical_mean_support_per_prompt"] == 3.0
    assert row["online_canonical_exploration_to_task_rms_ratio"] == 0.5
    assert row["math_strategy_rejected_integrity_rows"] == 2.0


def test_parse_run_binds_passive_drgrpo_verified_discovery_telemetry(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    telemetry = {
        "trainer/global_step": 9,
        "misc/prompt_consumed": 144,
        "train/verified_discovery_cumulative_outcomes": 27.0,
        "train/verified_discovery_tracked_prompts": 9.0,
        "train/verified_discovery_mean_support_per_prompt": 3.0,
    }
    evaluation = _record(10, 0.25)
    evaluation["misc/prompt_consumed"] = 160
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(telemetry) + "\n" + json.dumps(evaluation) + "\n",
        encoding="utf-8",
    )

    row = _multi_answer(
        parse_run(tmp_path, prompt_pool_size=10, num_samples=16)
    )[0]

    assert row["verified_discovery_cumulative_outcomes"] == 27.0
    assert row["verified_discovery_mean_support_per_prompt"] == 3.0
    # Existing live figures retain these aliases, so ordinary Dr.GRPO appears
    # immediately without changing historical treatment artifacts.
    assert row["online_canonical_tracked_outcomes"] == 27.0
    assert row["online_canonical_mean_support_per_prompt"] == 3.0


def test_parse_run_copies_e40_outside_centering_advantage_telemetry(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    record = _record(10, 0.25)
    record.update(
        {
            "train/outcome_collision_reward_sent_to_centering_mean": 0.5,
            "train/outcome_collision_bonus_zero_spread_group_fraction": 0.25,
            "train/outcome_collision_centered_bonus_rms": 0.03125,
            "train/outcome_collision_outside_centering_active": 1.0,
            "train/outcome_collision_outside_base_advantage_abs_mean": 0.4,
            "train/outcome_collision_outside_base_advantage_rms": 0.5,
            "train/outcome_collision_outside_semantic_advantage_mean": 0.0,
            "train/outcome_collision_outside_semantic_advantage_abs_mean": 0.125,
            "train/outcome_collision_outside_semantic_advantage_rms": 0.15625,
            "train/outcome_collision_outside_semantic_advantage_nonzero_fraction": 0.75,
            "train/outcome_collision_outside_combined_advantage_abs_mean": 0.45,
            "train/outcome_collision_outside_combined_advantage_rms": 0.5625,
        }
    )
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    row = _multi_answer(parse_run(tmp_path))[0]

    assert row["outcome_collision_reward_sent_to_centering_mean"] == 0.5
    assert row["outcome_collision_bonus_zero_spread_group_fraction"] == 0.25
    assert row["outcome_collision_centered_bonus_rms"] == 0.03125
    assert row["outcome_collision_outside_centering_active"] == 1.0
    assert row["outcome_collision_outside_base_advantage_abs_mean"] == 0.4
    assert row["outcome_collision_outside_base_advantage_rms"] == 0.5
    assert row["outcome_collision_outside_semantic_advantage_mean"] == 0.0
    assert (
        row["outcome_collision_outside_semantic_advantage_abs_mean"] == 0.125
    )
    assert row["outcome_collision_outside_semantic_advantage_rms"] == 0.15625
    assert (
        row[
            "outcome_collision_outside_semantic_advantage_nonzero_fraction"
        ]
        == 0.75
    )
    assert (
        row["outcome_collision_outside_combined_advantage_abs_mean"] == 0.45
    )
    assert row["outcome_collision_outside_combined_advantage_rms"] == 0.5625


def test_parse_run_copies_e41_separate_shannon_advantage_telemetry(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    telemetry_keys = (
        "semantic_shannon_separate_advantage_active",
        "semantic_shannon_reward_sent_to_centering_mean",
        "semantic_shannon_separate_predictive_baseline_mean",
        "semantic_shannon_separate_predictive_baseline_min",
        "semantic_shannon_separate_predictive_baseline_max",
        "semantic_shannon_separate_predictive_baseline_normalized_mean",
        "semantic_shannon_separate_predictive_centering_error_max",
        "semantic_shannon_separate_advantage_scale",
        "semantic_shannon_separate_base_advantage_mean",
        "semantic_shannon_separate_base_advantage_abs_mean",
        "semantic_shannon_separate_base_advantage_rms",
        "semantic_shannon_separate_base_advantage_nonzero_fraction",
        "semantic_shannon_separate_semantic_advantage_mean",
        "semantic_shannon_separate_semantic_advantage_min",
        "semantic_shannon_separate_semantic_advantage_max",
        "semantic_shannon_separate_semantic_advantage_abs_mean",
        "semantic_shannon_separate_semantic_advantage_rms",
        "semantic_shannon_separate_semantic_advantage_positive_fraction",
        "semantic_shannon_separate_semantic_advantage_negative_fraction",
        "semantic_shannon_separate_semantic_advantage_zero_fraction",
        "semantic_shannon_separate_semantic_advantage_nonzero_fraction",
        "semantic_shannon_separate_combined_advantage_mean",
        "semantic_shannon_separate_combined_advantage_abs_mean",
        "semantic_shannon_separate_combined_advantage_rms",
        "semantic_shannon_separate_combined_advantage_nonzero_fraction",
    )
    expected = {
        key: index / 100
        for index, key in enumerate(telemetry_keys, start=1)
    }
    record = _record(10, 0.25)
    record.update({f"train/{key}": value for key, value in expected.items()})
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    row = _multi_answer(parse_run(tmp_path))[0]

    assert {key: row[key] for key in telemetry_keys} == expected


def test_parse_run_copies_e42_quality_gated_semantic_telemetry(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    expected = {
        "semantic_shannon_quality_gated_advantage_active": 1.0,
        "semantic_shannon_quality_gated_effective_advantage_rms": 0.021,
        "semantic_shannon_quality_gated_eligible_fraction": 0.375,
        "semantic_shannon_quality_gated_active_fraction": 0.875,
        "semantic_shannon_quality_gated_reward_positive_fraction": 0.5,
        "semantic_shannon_quality_gated_cap_fraction": 0.125,
        "semantic_shannon_quality_gated_advantage_cap": 0.05,
        "semantic_shannon_quality_gated_history_rows_added": 6.0,
    }
    record = _record(10, 0.25)
    record.update({f"train/{key}": value for key, value in expected.items()})
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    row = _multi_answer(parse_run(tmp_path))[0]

    assert {key: row[key] for key in expected} == expected


def test_parse_run_copies_e43_success_conditioned_signed_telemetry(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    expected = {
        "semantic_shannon_success_conditioned_signed_advantage_active": 1.0,
        "semantic_shannon_success_conditioned_signed_effective_advantage_rms": 0.021,
        "semantic_shannon_success_conditioned_signed_eligible_fraction": 0.375,
        "semantic_shannon_success_conditioned_signed_effective_advantage_positive_fraction": 0.125,
        "semantic_shannon_success_conditioned_signed_effective_advantage_negative_fraction": 0.25,
        "semantic_shannon_success_conditioned_signed_positive_cap_fraction": 0.0625,
        "semantic_shannon_success_conditioned_signed_negative_cap_fraction": 0.03125,
        "semantic_shannon_success_conditioned_signed_advantage_cap": 0.05,
        "semantic_shannon_success_conditioned_signed_history_rows_added": 6.0,
    }
    record = _record(10, 0.25)
    record.update({f"train/{key}": value for key, value in expected.items()})
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    row = _multi_answer(parse_run(tmp_path))[0]

    assert {key: row[key] for key in expected} == expected
    assert "mechanism_only" not in row


def test_parse_run_retains_latest_e43_optimizer_telemetry_between_evals(
    tmp_path,
):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    eval_record = _record(0, 0.25)
    eval_record["misc/prompt_consumed"] = 0
    live_record = {
        "trainer/global_step": 3,
        "misc/prompt_epoch": 0.015625,
        "misc/prompt_consumed": 48,
        "train/semantic_shannon_success_conditioned_signed_advantage_active": 1.0,
        "train/semantic_shannon_success_conditioned_signed_effective_advantage_rms": 0.021,
        "train/semantic_shannon_success_conditioned_signed_eligible_fraction": 0.375,
        "train/semantic_shannon_success_conditioned_signed_effective_advantage_positive_fraction": 0.125,
        "train/semantic_shannon_success_conditioned_signed_effective_advantage_negative_fraction": 0.25,
        "train/semantic_shannon_success_conditioned_signed_positive_cap_fraction": 0.0625,
        "train/semantic_shannon_success_conditioned_signed_negative_cap_fraction": 0.03125,
    }
    (debug_dir / "train_metrics.jsonl").write_text(
        "\n".join((json.dumps(eval_record), json.dumps(live_record))) + "\n",
        encoding="utf-8",
    )

    rows = _multi_answer(
        parse_run(
            tmp_path,
            prompt_pool_size=192,
            num_samples=16,
        )
    )

    assert [row["step"] for row in rows] == [0, 3]
    assert rows[0]["coverage8"] == 0.25
    assert "mechanism_only" not in rows[0]
    live = rows[1]
    assert live["mechanism_only"] is True
    assert live["coverage8"] is None
    assert live["coverage8_draws"] == []
    assert live["training_passes"] == 0.015625
    assert (
        live[
            "semantic_shannon_success_conditioned_signed_effective_advantage_rms"
        ]
        == 0.021
    )
    assert (
        live[
            "semantic_shannon_success_conditioned_signed_eligible_fraction"
        ]
        == 0.375
    )
    assert (
        live[
            "semantic_shannon_success_conditioned_signed_effective_advantage_positive_fraction"
        ]
        == 0.125
    )
    assert (
        live[
            "semantic_shannon_success_conditioned_signed_effective_advantage_negative_fraction"
        ]
        == 0.25
    )
    assert (
        live[
            "semantic_shannon_success_conditioned_signed_positive_cap_fraction"
        ]
        == 0.0625
    )
    assert (
        live[
            "semantic_shannon_success_conditioned_signed_negative_cap_fraction"
        ]
        == 0.03125
    )


def test_parse_run_carries_e43_optimizer_telemetry_to_each_later_eval(
    tmp_path,
):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()

    def signed_record(step: int, rms: float) -> dict:
        return {
            "trainer/global_step": step,
            "misc/prompt_epoch": step / 10,
            "misc/prompt_consumed": step * 16,
            "train/semantic_shannon_success_conditioned_signed_advantage_active": 1.0,
            "train/semantic_shannon_success_conditioned_signed_effective_advantage_rms": rms,
        }

    records = (
        _record(0, 0.20),
        signed_record(4, 0.01),
        _record(5, 0.30),
        signed_record(9, 0.02),
        _record(10, 0.40),
        signed_record(11, 0.03),
    )
    (debug_dir / "train_metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    rows = _multi_answer(parse_run(tmp_path))

    assert [row["step"] for row in rows] == [0, 5, 10, 11]
    assert [
        row[
            "semantic_shannon_success_conditioned_signed_effective_advantage_rms"
        ]
        for row in rows
    ] == [None, 0.01, 0.02, 0.03]
    assert [row.get("mechanism_only", False) for row in rows] == [
        False,
        False,
        False,
        True,
    ]
    assert [row["coverage8"] for row in rows] == [0.20, 0.30, 0.40, None]


def test_parse_run_accepts_math_split_without_multi_answer_coverage(tmp_path):
    debug_dir = tmp_path / "debug_01"
    debug_dir.mkdir()
    record = _math_record(20, 0.5)
    record["misc/prompt_consumed"] = 384 * 16 * 2
    record["train/outcome_collision_rate"] = 0.375
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    rows = parse_run(
        tmp_path,
        prompt_pool_size=384,
        num_samples=16,
        eval_splits=("math",),
    )

    assert len(rows) == 1
    assert rows[0]["split"] == "math"
    assert rows[0]["pass8"] == 0.5
    assert rows[0]["coverage8"] is None
    assert rows[0]["training_passes"] == 2.0
    assert rows[0]["outcome_collision_rate"] == 0.375


def test_cli_eval_splits_writes_only_requested_math_rows(monkeypatch, tmp_path):
    stamp = "e39_math_split_contract"
    run_dir = tmp_path / f"xdr_test_{stamp}_semantic_shannon_s43"
    debug_dir = run_dir / "debug_01"
    debug_dir.mkdir(parents=True)
    (debug_dir / "train_metrics.jsonl").write_text(
        json.dumps(_math_record(0, 0.5)) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "curve.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_scaling_curve.py",
            "--stamp-prefix",
            stamp,
            "--run-data-root",
            str(tmp_path),
            "--out",
            str(out),
            "--prompt-pool-size",
            "384",
            "--num-samples",
            "16",
            "--max-training-passes",
            "10",
            "--eval-splits",
            "math",
        ],
    )

    _MODULE.main()

    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["split"] == "math"
    assert rows[0]["arm"] == "semantic_shannon"


def test_cli_passes_each_progress_argument_to_parse_run_once(
    monkeypatch,
    tmp_path,
):
    stamp = "e37_cli_contract"
    run_dir = tmp_path / f"xdr_test_{stamp}_outcome_collision_s43"
    run_dir.mkdir()
    _attempt_with_prompt_progress(
        run_dir,
        "debug_01",
        [(0, 0, 0.25)],
    )
    out = tmp_path / "curve.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parse_scaling_curve.py",
            "--stamp-prefix",
            stamp,
            "--run-data-root",
            str(tmp_path),
            "--out",
            str(out),
            "--prompt-pool-size",
            "384",
            "--num-samples",
            "16",
            "--max-training-passes",
            "10",
        ],
    )

    _MODULE.main()

    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert {row["arm"] for row in rows} == {"outcome_collision"}
    assert {row["training_passes"] for row in rows} == {0.0}


def test_parse_run_keeps_resume_boundary_when_predecessor_did_not_evaluate_it(
    tmp_path,
):
    _attempt(tmp_path, "debug_01", [(0, 0.0), (10, 0.1), (30, 0.3)])
    _attempt(tmp_path, "debug_02", [(20, 2.0), (30, 3.0), (40, 4.0)])

    rows = _multi_answer(parse_run(tmp_path))

    assert [row["step"] for row in rows] == [0, 10, 20, 30, 40]
    assert [row["coverage8"] for row in rows] == [0.0, 0.1, 2.0, 3.0, 4.0]


def test_parse_run_keeps_each_predecessor_boundary_across_multiple_resumes(
    tmp_path,
):
    _attempt(tmp_path, "debug_01", [(0, 0.0), (10, 0.1), (20, 0.2)])
    _attempt(tmp_path, "debug_02", [(20, 2.0), (30, 3.0), (40, 4.0)])
    _attempt(tmp_path, "debug_03", [(40, 40.0), (50, 5.0)])

    rows = _multi_answer(parse_run(tmp_path))

    assert [row["step"] for row in rows] == [0, 10, 20, 30, 40, 50]
    assert [row["coverage8"] for row in rows] == [0.0, 0.1, 0.2, 3.0, 4.0, 5.0]


def test_parse_run_does_not_mix_fresh_reruns(tmp_path):
    _attempt(tmp_path, "debug_01", [(0, 0.0), (10, 0.1), (20, 0.2)])
    _attempt(tmp_path, "debug_02", [(0, 1.0), (30, 3.0)])

    rows = _multi_answer(parse_run(tmp_path))

    assert [row["step"] for row in rows] == [0, 30]
    assert [row["coverage8"] for row in rows] == [1.0, 3.0]


def test_short_resume_does_not_hide_furthest_crashed_attempt(tmp_path):
    _attempt(tmp_path, "debug_01", [(0, 0.0), (10, 0.1), (20, 0.2)])
    _attempt(tmp_path, "debug_02", [(10, 1.0)])

    rows = _multi_answer(parse_run(tmp_path))

    assert [row["step"] for row in rows] == [0, 10, 20]
    assert [row["coverage8"] for row in rows] == [0.0, 0.1, 0.2]


def test_plot_horizon_prefers_later_terminal_eval_over_overrun(tmp_path):
    denominator = 384 * 32
    _attempt_with_prompt_progress(
        tmp_path,
        "debug_01",
        [
            (0, 0, 0.0),
            (768, 4 * denominator, 0.4),
            (1024, 16 * denominator / 3, 0.53),
        ],
    )
    _attempt_with_prompt_progress(
        tmp_path,
        "debug_02",
        [(0, 0, 1.0), (768, 4 * denominator, 1.4), (960, 5 * denominator, 1.5)],
    )

    rows = _multi_answer(
        parse_run(
            tmp_path,
            prompt_pool_size=384,
            num_samples=32,
            max_training_passes=5,
        )
    )

    assert [row["step"] for row in rows] == [0, 768, 960]
    assert [row["coverage8"] for row in rows] == [1.0, 1.4, 1.5]


def test_manifest_bound_attempt_ignores_further_canceled_job(tmp_path):
    _attempt(tmp_path, "debug_job100", [(0, 0.0), (100, 9.0)])
    _attempt(tmp_path, "debug_job200", [(0, 0.2), (10, 0.3)])

    rows = _multi_answer(
        parse_run(tmp_path, allowed_debug_dir="debug_job200")
    )

    assert [row["step"] for row in rows] == [0, 10]
    assert [row["coverage8"] for row in rows] == [0.2, 0.3]
