from __future__ import annotations

import json

import pytest

from exp_scaling.check_e9b_maxent_actuator_smoke import (
    ADAPTIVE_STAMP,
    CALIBRATION_STAMP,
    DOSES,
    stage_a,
    stage_b,
)


def _write_run(root, stamp, arm, rows):
    debug = root / f"run_{stamp}_{arm}_s9005" / "debug_1"
    debug.mkdir(parents=True)
    (debug / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _row(step, *, entropy, alpha, reward=1.0):
    return {
        "misc/global_step": step,
        "misc/lr": 0.0,
        "train/pg_loss": 0.1,
        "train/policy_grad_norm": 0.2,
        "train/maxent_alpha_used": alpha,
        "train/maxent_sequence_entropy": entropy,
        "train/maxent_entropy_surrogate": 0.1,
        "train/maxent_entropy_loss": -0.01,
        "actor/response_tok_len": 4.0,
        "actor/rewards": reward,
    }


def _complete_stage_a(root, *, dose_entropy=0.05):
    _write_run(
        root,
        CALIBRATION_STAMP,
        "maxent",
        [_row(step, entropy=0.1, alpha=0.05) for step in range(64)],
    )
    for stamp, alpha in DOSES.items():
        rows = [
            _row(step, entropy=dose_entropy, alpha=alpha) for step in range(128)
        ]
        for row in rows:
            row["misc/lr"] = 2e-7
        _write_run(root, stamp, "maxent", rows)


def test_stage_a_derives_frozen_target_and_accepts_viable_dose(tmp_path):
    _complete_stage_a(tmp_path)

    target, summaries = stage_a(tmp_path)

    assert target == pytest.approx(0.08)
    assert len(summaries) == 4
    assert all(row["target_fraction"] == pytest.approx(0.625) for row in summaries)


def test_stage_a_rejects_doses_that_cannot_actuate_entropy(tmp_path):
    _complete_stage_a(tmp_path, dose_entropy=0.01)

    with pytest.raises(RuntimeError, match="no fixed-alpha dose"):
        stage_a(tmp_path)


def test_stage_b_checks_immediate_controller_telemetry(tmp_path):
    _complete_stage_a(tmp_path)
    for arm in ("maxent_control", "maxent_dual"):
        rows = [_row(step, entropy=0.05, alpha=0.2) for step in range(128)]
        for step, row in enumerate(rows):
            row["misc/lr"] = 2e-7
            if arm == "maxent_control":
                row["train/maxent_control_target_entropy"] = 0.08
                row["train/maxent_control_relative_deficit"] = 0.375
            else:
                row["train/maxent_dual_target_entropy"] = 0.08
                row["train/maxent_dual_optimizer_steps"] = step + 1
        _write_run(tmp_path, ADAPTIVE_STAMP, arm, rows)

    target, summaries = stage_b(tmp_path)

    assert target == pytest.approx(0.08)
    assert len(summaries) == 2
