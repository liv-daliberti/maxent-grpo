from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

from exp_scaling.check_e14_c0 import (
    EXPECTED_ACTIONS,
    EXPECTED_ACTION_TOKEN_IDS,
    GateError,
    _inspect_update_row,
    _weight_identity,
    inspect_endpoint_audit,
    inspect_metrics,
)
from exp_scaling.check_e14_preflight import (
    EXPECTED_DATASET_IDENTITY,
    EXPECTED_RUNTIME_IDENTITY,
)


def _update_row(step: int, *, reward: float = 1.0) -> dict:
    return {
        "trainer/step": step,
        "trainer/global_step": step,
        "trainer/policy_sgd_step": step,
        "actor/canonical_action_count": 3.0,
        "actor/canonical_action_support_size": 3.0,
        "actor/canonical_behavior_q_norm_error_max": 1e-8,
        "actor/canonical_behavior_q_row_count": 48.0,
        "actor/canonical_behavior_q_support_max": 3.0,
        "actor/canonical_behavior_q_support_min": 3.0,
        "actor/canonical_finish_length_count": 16.0,
        "actor/canonical_finish_unexpected_count": 0.0,
        "actor/canonical_graph_actions": 1.0,
        "actor/canonical_invalid_count": 0.0,
        "actor/canonical_sampler_fixed_shape": 1.0,
        "actor/canonical_sampler_learner": 1.0,
        "actor/formatted": 1.0,
        "actor/generate_avg_str_len": 3.0,
        "actor/no_eos_count": 0.0,
        "actor/num_data": 16.0,
        "actor/response_tok_len": 3.0,
        "actor/rewards": reward,
        "actor/sampling_max_tokens": 3.0,
        "actor/sampling_temperature": 1.0,
        "misc/lr": 2e-7,
        "misc/prompt_consumed": float(step * 16),
        "misc/query_step": float(step * 16),
        "train/canonical_action_count": 3.0,
        "train/canonical_action_vocab_size": 3.0,
        "train/canonical_behavior_denominator_actor": 1.0,
        "train/canonical_behavior_kl_actor_learner_max": 0.0,
        "train/canonical_behavior_kl_actor_learner_mean": 0.0,
        "train/canonical_behavior_kl_learner_actor_max": 0.0,
        "train/canonical_behavior_kl_learner_actor_mean": 0.0,
        "train/canonical_behavior_prefix_ess_fraction_min": 1.0,
        "train/canonical_behavior_q_norm_error_max": 0.0,
        "train/canonical_behavior_q_row_count": 48.0,
        "train/canonical_behavior_q_support_max": 3.0,
        "train/canonical_behavior_q_support_min": 3.0,
        "train/canonical_behavior_ratio_max": 1.0,
        "train/canonical_behavior_ratio_min": 1.0,
        "train/canonical_behavior_sequence_ess_fraction": 1.0,
        "train/canonical_behavior_tv_max": 0.0,
        "train/canonical_behavior_tv_mean": 0.0,
        "train/canonical_sampled_prefix_entropy_ratio": 1.0,
        "train/canonical_sampled_prefix_entropy_sum": math.log(27.0),
        "train/canonical_token_entropy_mean": math.log(3.0),
        "train/entropy": math.log(3.0),
        "train/learning_round": float(step),
        "train/pg_loss": 0.0,
        "train/policy_grad_norm": 0.0,
    }


def _write_metrics(tmp_path: Path, rows: list[dict]) -> Path:
    run = tmp_path / "run"
    debug = run / "debug_1"
    debug.mkdir(parents=True)
    (debug / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    saved_models = debug / "saved_models"
    for step in (32, 64, 96, 128, 129):
        checkpoint = saved_models / f"step_{step:05d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"same endpoint weights")
    eval_results = debug / "eval_results"
    eval_results.mkdir()
    for step in (0, 32, 64, 96, 128, 129):
        (eval_results / f"{step}_multi_answer.json").write_text(
            "{}", encoding="utf-8"
        )
    return run


def _complete_rows() -> list[dict]:
    initial = {
        "trainer/step": 0,
        "trainer/global_step": 0,
        "trainer/policy_sgd_step": 0,
        "misc/prompt_dataset_len": 192.0,
        "misc/query_step": 0.0,
        "misc/prompt_consumed": 0.0,
        "eval/multi_answer/eval_count": 96.0,
        "eval/multi_answer/response_tok_len": 3.0,
    }
    updates = [_update_row(step) for step in range(1, 129)]
    for step in (32, 64, 96, 128):
        updates[step - 1].update(
            {
                "eval/multi_answer/eval_count": 96.0,
                "eval/multi_answer/response_tok_len": 3.0,
            }
        )
    terminal = dict(updates[-1])
    terminal.update(
        {
            "trainer/step": 129,
            "trainer/global_step": 128,
            "eval/multi_answer/eval_count": 96.0,
            "eval/multi_answer/response_tok_len": 3.0,
        }
    )
    return [initial, *updates, terminal]


def test_c0_metrics_gate_accepts_exact_contiguous_control(tmp_path):
    run = _write_metrics(tmp_path, _complete_rows())

    summary, _, checkpoint = inspect_metrics(run)

    assert summary["optimizer_updates"] == 128
    assert summary["contiguous_finite_tail"] == [97, 128]
    assert summary["final_32_mean_rollout_reward"] == 1.0
    assert summary["behavior_ratio_min"] == 1.0
    assert summary["behavior_ratio_max"] == 1.0
    assert checkpoint.name == "step_00128"


def test_c0_initial_gate_stays_full_schedule_only_and_archive_is_explicit(tmp_path):
    run = _write_metrics(tmp_path, _complete_rows())
    saved = next(run.glob("debug_*/saved_models"))
    for step in (32, 64, 96):
        shutil.rmtree(saved / f"step_{step:05d}")

    with pytest.raises(GateError, match="checkpoint schedule drifted"):
        inspect_metrics(run)

    summary, _, _ = inspect_metrics(
        run,
        archived_removed_steps=("step_00032", "step_00064", "step_00096"),
    )
    assert summary["optimizer_updates"] == 128

    with pytest.raises(GateError, match="checkpoint schedule drifted"):
        inspect_metrics(run, archived_removed_steps=("step_00032",))


def test_c0_kl_gate_accepts_only_negligible_negative_roundoff():
    row = _update_row(1)
    row["train/canonical_behavior_kl_actor_learner_mean"] = -8.511e-19
    row["train/canonical_behavior_kl_learner_actor_mean"] = -8.511e-19
    _inspect_update_row(row, step=1)

    row["train/canonical_behavior_kl_actor_learner_mean"] = -1.01e-12
    with pytest.raises(GateError, match="roundoff-aware range"):
        _inspect_update_row(row, step=1)

    row = _update_row(1)
    row["train/canonical_behavior_kl_actor_learner_max"] = 0.0100001
    with pytest.raises(GateError, match="roundoff-aware range"):
        _inspect_update_row(row, step=1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda rows: rows.__setitem__(112, rows[113]),
            "exact contiguous sequence",
        ),
        (
            lambda rows: rows[64].__setitem__(
                "actor/canonical_behavior_q_norm_error_max", 1e-3
            ),
            "normalization error",
        ),
        (
            lambda rows: rows[64].__setitem__(
                "train/canonical_behavior_ratio_max", 1.21
            ),
            "ratio range",
        ),
        (
            lambda rows: rows[64].__setitem__("train/maxent_alpha_used", 0.0),
            "forbidden treatment telemetry",
        ),
        (
            lambda rows: [
                row.__setitem__("actor/rewards", 0.0)
                for row in rows[97:129]
            ],
            "final-32 mean rollout reward",
        ),
    ),
)
def test_c0_metrics_gate_fails_closed(tmp_path, mutation, message):
    rows = _complete_rows()
    mutation(rows)
    run = _write_metrics(tmp_path, rows)

    with pytest.raises(GateError, match=message):
        inspect_metrics(run)


def _audit_payload(checkpoint: Path) -> dict:
    probability = 1.0 / 27.0
    log_probability = -math.log(27.0)
    selected = [-math.log(3.0)] * 3
    probability_sum = math.fsum([probability] * 27)
    entropy = -math.fsum([probability * log_probability] * 27)
    p_valid = 2.0 / 27.0
    h_valid = math.log(2.0)
    prompts = []
    for prompt_index in range(96):
        leaves = []
        for leaf_index, action in enumerate(EXPECTED_ACTIONS):
            reward = int(leaf_index < 2)
            leaves.append(
                {
                    "action": action,
                    "action_token_ids": [
                        EXPECTED_ACTION_TOKEN_IDS[int(digit) - 1] for digit in action
                    ],
                    "grader_reward": reward,
                    "leaf_log_probability": log_probability,
                    "leaf_probability": probability,
                    "q_plus": 0.5 if reward else 0.0,
                    "teacher_forced_selected_log_probabilities": selected,
                    "teacher_forced_sequence_log_probability": math.fsum(selected),
                }
            )
        prompts.append(
            {
                "prompt_index": prompt_index,
                "problem_sha256": hashlib.sha256(
                    str(prompt_index).encode()
                ).hexdigest(),
                "prompt_token_count": 8,
                "declared_valid_action_count": 2,
                "valid_action_count": 2,
                "probability_sum": probability_sum,
                "probability_sum_abs_error": abs(probability_sum - 1.0),
                "exact_action_entropy": entropy,
                "conditional_entropy": entropy,
                "entropy_identity_abs_error": 0.0,
                "p_valid": p_valid,
                "log_p_valid": math.log(p_valid),
                "h_valid": h_valid,
                "n_eff_valid": 2.0,
                "teacher_forced_vs_prefix_tree_per_token_max_abs_error": 0.0,
                "teacher_forced_vs_prefix_tree_sequence_max_abs_error": 0.0,
                "leaves": leaves,
            }
        )
    weight_hash, weight_files = _weight_identity(checkpoint)
    return {
        "schema": "e14_exact_endpoint_audit_v1",
        "status": "pass",
        "formulation": "canonical_graph_actions",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "oat_step_tag": 128,
            "optimizer_updates": 128,
            "role": "scheduled_update_boundary",
            "weights_manifest_sha256": weight_hash,
            "weight_files": weight_files,
        },
        "data": {
            "root": "/frozen/data",
            "combined_content_hash": EXPECTED_DATASET_IDENTITY[
                "combined_content_hash"
            ],
            "eval_rows": 96,
            "split": "multi_answer",
        },
        "runtime": {
            "device": "cpu",
            "model_dtype": "float32",
            "batch_size": 8,
            "torch_version": "test",
        },
        "policy": {
            "actions": ["1", "2", "3"],
            "action_token_ids": list(EXPECTED_ACTION_TOKEN_IDS),
            "horizon": 3,
            "leaf_count": 27,
            "max_action_entropy_nats": math.log(27.0),
            "tokenizer_class": "Qwen2TokenizerFast",
            "tokenizer_files_hash": EXPECTED_RUNTIME_IDENTITY[
                "tokenizer_files_hash"
            ],
            "tokenizer_path": "/frozen/tokenizer",
            "tokenizer_revision": EXPECTED_RUNTIME_IDENTITY[
                "tokenizer_revision"
            ],
            "tokenizer_vocab_hash": EXPECTED_RUNTIME_IDENTITY[
                "tokenizer_vocab_hash"
            ],
        },
        "tolerances": {
            "probability_sum_abs": 1e-5,
            "entropy_identity_abs_nats": 1e-5,
            "teacher_forced_log_probability_abs": 5e-3,
        },
        "aggregate": {
            "prompt_count": 96,
            "leaf_count": 96 * 27,
            "exact_action_entropy_mean": entropy,
            "conditional_entropy_mean": entropy,
            "p_valid_mean": p_valid,
            "p_valid_min": p_valid,
            "h_valid_mean": h_valid,
            "n_eff_valid_mean": 2.0,
            "valid_action_count_mean": 2.0,
            "valid_action_count_histogram": {"2": 96},
            "probability_sum_max_abs_error": abs(probability_sum - 1.0),
            "entropy_identity_max_abs_error": 0.0,
            "teacher_forced_vs_prefix_tree_per_token_max_abs_error": 0.0,
            "teacher_forced_vs_prefix_tree_sequence_max_abs_error": 0.0,
        },
        "prompts": prompts,
    }


def _write_audit(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return path


def test_c0_endpoint_gate_revalidates_exact_prompt_records_and_weights(tmp_path):
    checkpoint = tmp_path / "saved_models" / "step_00128"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"frozen weights")
    audit = _write_audit(tmp_path, _audit_payload(checkpoint))

    summary = inspect_endpoint_audit(audit, checkpoint=checkpoint)

    assert summary["prompt_count"] == 96
    assert summary["leaf_count"] == 2592
    assert summary["p_valid_mean"] == pytest.approx(2.0 / 27.0)


def test_c0_endpoint_gate_rejects_threshold_alias_and_checkpoint_mutation(tmp_path):
    checkpoint = tmp_path / "saved_models" / "step_00128"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"frozen weights")
    payload = _audit_payload(checkpoint)

    payload["checkpoint"]["role"] = "forced_terminal_alias"
    with pytest.raises(GateError, match="terminal alias"):
        inspect_endpoint_audit(_write_audit(tmp_path, payload), checkpoint=checkpoint)

    payload = _audit_payload(checkpoint)
    payload["aggregate"]["p_valid_mean"] = 0.04
    with pytest.raises(GateError, match="aggregate p_valid_mean"):
        inspect_endpoint_audit(_write_audit(tmp_path, payload), checkpoint=checkpoint)

    payload = _audit_payload(checkpoint)
    audit = _write_audit(tmp_path, payload)
    weights.write_bytes(b"mutated weights")
    with pytest.raises(GateError, match="weights changed"):
        inspect_endpoint_audit(audit, checkpoint=checkpoint)
