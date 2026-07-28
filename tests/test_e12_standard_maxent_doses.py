from __future__ import annotations

import json

import pytest

from exp_scaling.check_e12_standard_maxent_doses import (
    DOSES,
    T_MAX,
    inspect_all,
    stamp,
)


def _row(
    step: int,
    *,
    alpha: float,
    entropy: float = 6.0,
    length: float = 8.0,
    no_eos: float = 0.0,
    accuracy: float = 0.5,
) -> dict:
    return {
        "misc/global_step": step,
        "train/pg_loss": 0.1,
        "train/policy_grad_norm": 0.2,
        "train/maxent_alpha_used": alpha,
        "train/maxent_sequence_entropy": entropy,
        "train/maxent_sequence_entropy_per_tmax": entropy / T_MAX,
        "train/maxent_entropy_surrogate": entropy,
        "train/maxent_entropy_loss": -0.01,
        "train/maxent_prefix_ratio_mean": 1.0,
        "train/maxent_prefix_ratio_max": 1.1,
        "train/maxent_prefix_ratio_clipfrac": 0.0,
        "actor/response_tok_len": length,
        "actor/no_eos_count": no_eos,
        "actor/rewards": 1.0,
        "eval/average/accuracy": accuracy,
        "eval/average/sampled_mode_coverage_at_8": 0.4,
        "eval/multi_answer/sampled_any_correct_at_8": 0.6,
    }


def _write(root, label: str, rows: list[dict]) -> None:
    debug = root / f"run_{stamp(label)}_maxent_s9005" / "debug_1"
    debug.mkdir(parents=True, exist_ok=True)
    (debug / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _complete(root) -> None:
    for label, alpha in DOSES.items():
        _write(root, label, [_row(step, alpha=alpha) for step in range(129)])


def test_e12_selects_largest_safe_entropy_effective_dose(tmp_path):
    _complete(tmp_path)

    summaries, preferred = inspect_all(tmp_path)

    assert all(row["viable"] for row in summaries)
    assert preferred is not None
    assert preferred["alpha"] == pytest.approx(0.002)


def test_e12_rejects_tail_length_and_no_eos_runaway(tmp_path):
    _complete(tmp_path)
    label = "a0p0020"
    alpha = DOSES[label]
    rows = [_row(step, alpha=alpha) for step in range(113)]
    rows.extend(
        _row(step, alpha=alpha, length=70.0, no_eos=4.0)
        for step in range(113, 129)
    )
    _write(tmp_path, label, rows)

    summaries, preferred = inspect_all(tmp_path)

    rejected = next(row for row in summaries if row["label"] == label)
    assert not rejected["safe"]
    assert "tail max no-EOS 4 > 2" in rejected["failures"]
    assert preferred is not None
    assert preferred["alpha"] == pytest.approx(0.0015)


def test_e12_rejects_mismatched_raw_and_normalized_entropy(tmp_path):
    _complete(tmp_path)
    path = next(
        tmp_path.glob(
            f"*{stamp('a0p0005')}*/debug_1/train_metrics.jsonl"
        )
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["train/maxent_sequence_entropy_per_tmax"] = 6.0
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="raw/per-T_max"):
        inspect_all(tmp_path)


def test_e12_requires_exact_step_128_endpoint(tmp_path):
    _complete(tmp_path)
    label = "a0p0005"
    alpha = DOSES[label]
    rows = [_row(step, alpha=alpha) for step in range(128)]
    rows.append(_row(129, alpha=alpha))
    _write(tmp_path, label, rows)

    with pytest.raises(RuntimeError, match="lacks the frozen step-128"):
        inspect_all(tmp_path)


def test_e12_checks_alpha_on_every_row_and_rejects_controller_telemetry(tmp_path):
    _complete(tmp_path)
    label = "a0p0005"
    alpha = DOSES[label]
    rows = [_row(step, alpha=alpha) for step in range(129)]
    rows[64]["train/maxent_alpha_used"] = 0.5
    _write(tmp_path, label, rows)

    with pytest.raises(RuntimeError, match="used alpha=.*at step 64"):
        inspect_all(tmp_path)

    rows[64]["train/maxent_alpha_used"] = alpha
    rows[64]["train/maxent_dual_optimizer_steps"] = 1
    _write(tmp_path, label, rows)
    with pytest.raises(RuntimeError, match="adaptive-controller telemetry"):
        inspect_all(tmp_path)


def test_e12_requires_contiguous_final_32_steps(tmp_path):
    _complete(tmp_path)
    label = "a0p0005"
    alpha = DOSES[label]
    rows = [
        _row(step, alpha=alpha)
        for step in range(129)
        if step != 112
    ]
    _write(tmp_path, label, rows)

    with pytest.raises(RuntimeError, match="missing steps 112"):
        inspect_all(tmp_path)
