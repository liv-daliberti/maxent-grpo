from __future__ import annotations

import json

import pytest

from exp_scaling.check_e13_length_constrained_maxent import (
    ALPHA,
    ARMS,
    EMA_DECAY,
    LAMBDA_MAX,
    TARGET_LENGTH,
    T_MAX,
    inspect_all,
    stamp,
)


def _rows(
    *,
    dual_lr: float,
    entropy: float = 6.0,
    length: float = 8.0,
    no_eos: float = 0.0,
    accuracy: float = 0.5,
) -> list[dict]:
    ema = TARGET_LENGTH
    lambda_used = 0.0
    rows = []
    for step in range(1, 129):
        expected_length = length
        ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * expected_length
        violation = (ema - TARGET_LENGTH) / TARGET_LENGTH
        lambda_next = min(
            LAMBDA_MAX,
            max(0.0, lambda_used + dual_lr * violation),
        )
        rows.append(
            {
                "misc/global_step": step,
                "train/pg_loss": 0.1,
                "train/policy_grad_norm": 0.2,
                "train/maxent_alpha_used": ALPHA,
                "train/maxent_sequence_entropy": entropy,
                "train/maxent_sequence_entropy_per_tmax": entropy / T_MAX,
                "train/maxent_entropy_surrogate": entropy,
                "train/maxent_entropy_loss": -0.01,
                "train/maxent_prefix_ratio_mean": 1.0,
                "train/maxent_prefix_ratio_max": 1.1,
                "train/maxent_prefix_ratio_clipfrac": 0.0,
                "train/maxent_expected_length": expected_length,
                "train/maxent_sampled_prefix_length": length,
                "train/maxent_length_surrogate": length,
                "train/maxent_length_loss": lambda_used * length / T_MAX,
                "train/maxent_length_target": TARGET_LENGTH,
                "train/maxent_length_lambda_used": lambda_used,
                "train/maxent_length_lambda_next": lambda_next,
                "train/maxent_length_lambda_max": LAMBDA_MAX,
                "train/maxent_length_ema": ema,
                "train/maxent_length_ema_decay": EMA_DECAY,
                "train/maxent_length_relative_violation": violation,
                "train/maxent_length_dual_lr": dual_lr,
                "actor/response_tok_len": length,
                "actor/no_eos_count": no_eos,
                "actor/rewards": 1.0,
                "eval/average/accuracy": accuracy,
                "eval/average/sampled_mode_coverage_at_8": 0.4,
                "eval/multi_answer/sampled_any_correct_at_8": 0.6,
            }
        )
        lambda_used = lambda_next
    return rows


def _write(root, label: str, rows: list[dict]) -> None:
    debug = (
        root
        / f"run_{stamp(label)}_maxent_length_dual_s9005"
        / "debug_1"
    )
    debug.mkdir(parents=True, exist_ok=True)
    (debug / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _complete(root, *, entropies: dict[str, float] | None = None) -> None:
    entropies = entropies or {}
    for label, dual_lr in ARMS.items():
        _write(
            root,
            label,
            _rows(dual_lr=dual_lr, entropy=entropies.get(label, 6.0)),
        )


def test_e13_uses_slow_arm_when_tail_entropies_are_within_five_percent(tmp_path):
    _complete(tmp_path, entropies={"eta5em5": 6.0, "eta2em4": 6.1})

    summaries, preferred = inspect_all(tmp_path)

    assert all(row["viable"] for row in summaries)
    assert preferred is not None
    assert preferred["label"] == "eta5em5"


def test_e13_selects_higher_entropy_outside_tie_band(tmp_path):
    _complete(tmp_path, entropies={"eta5em5": 6.0, "eta2em4": 8.0})

    _, preferred = inspect_all(tmp_path)

    assert preferred is not None
    assert preferred["label"] == "eta2em4"


def test_e13_rejects_length_no_eos_and_pinned_controller(tmp_path):
    _complete(tmp_path)
    label = "eta2em4"
    _write(
        tmp_path,
        label,
        _rows(dual_lr=ARMS[label], length=192.0, no_eos=16.0),
    )

    summaries, preferred = inspect_all(tmp_path)

    rejected = next(row for row in summaries if row["label"] == label)
    assert not rejected["constraint_safe"]
    assert "tail actor length 192.00 > 18" in rejected["failures"]
    assert "tail max no-EOS 16 > 2" in rejected["failures"]
    assert "lambda pinned at maximum for final eight updates" in rejected["failures"]
    assert preferred is not None
    assert preferred["label"] == "eta5em5"


def test_e13_rejects_incorrect_projected_dual_transition(tmp_path):
    _complete(tmp_path)
    label = "eta5em5"
    rows = _rows(dual_lr=ARMS[label])
    rows[64]["train/maxent_length_lambda_next"] = 0.001
    _write(tmp_path, label, rows)

    with pytest.raises(RuntimeError, match="projected lambda update"):
        inspect_all(tmp_path)


def test_e13_rejects_broken_lambda_continuity(tmp_path):
    _complete(tmp_path)
    label = "eta5em5"
    rows = _rows(dual_lr=ARMS[label])
    rows[64]["train/maxent_length_lambda_used"] = 0.001
    rows[64]["train/maxent_length_lambda_next"] = 0.001
    _write(tmp_path, label, rows)

    with pytest.raises(RuntimeError, match="lambda continuity"):
        inspect_all(tmp_path)


def test_e13_requires_exact_step_128(tmp_path):
    _complete(tmp_path)
    label = "eta5em5"
    rows = _rows(dual_lr=ARMS[label])
    rows = [row for row in rows if row["misc/global_step"] != 128]
    extra = _rows(dual_lr=ARMS[label])[-1].copy()
    extra["misc/global_step"] = 129
    rows.append(extra)
    _write(tmp_path, label, rows)

    with pytest.raises(RuntimeError, match="lacks the frozen step-128"):
        inspect_all(tmp_path)


def test_e13_requires_contiguous_final_32_steps(tmp_path):
    _complete(tmp_path)
    label = "eta5em5"
    rows = [
        row
        for row in _rows(dual_lr=ARMS[label])
        if row["misc/global_step"] != 112
    ]
    _write(tmp_path, label, rows)

    with pytest.raises(RuntimeError, match="missing steps 112"):
        inspect_all(tmp_path)


def test_e13_rejects_sampled_length_or_entropy_unit_mismatch(tmp_path):
    _complete(tmp_path)
    label = "eta5em5"
    rows = _rows(dual_lr=ARMS[label])
    rows[64]["train/maxent_sampled_prefix_length"] = 9.0
    _write(tmp_path, label, rows)

    with pytest.raises(RuntimeError, match="sampled-prefix length"):
        inspect_all(tmp_path)

    rows[64]["train/maxent_sampled_prefix_length"] = 8.0
    rows[64]["train/maxent_sequence_entropy_per_tmax"] = 6.0
    _write(tmp_path, label, rows)
    with pytest.raises(RuntimeError, match="per-T_max entropy"):
        inspect_all(tmp_path)
