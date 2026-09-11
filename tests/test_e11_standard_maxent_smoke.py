from __future__ import annotations

import json

import pytest

from exp_scaling.check_e11_standard_maxent_smoke import (
    COMPARATIVE_STAMP,
    LITERAL_STAMP,
    RAW_TARGET,
    T_MAX,
    inspect_all,
    inspect_literal,
)


def _row(step: int, *, raw_entropy: float, alpha: float = 0.05) -> dict:
    return {
        "misc/global_step": step,
        "train/pg_loss": 0.1,
        "train/policy_grad_norm": 0.2,
        "train/maxent_alpha_used": alpha,
        "train/maxent_sequence_entropy": raw_entropy,
        "train/maxent_sequence_entropy_per_tmax": raw_entropy / T_MAX,
        "train/maxent_entropy_surrogate": raw_entropy,
        "train/maxent_entropy_loss": -0.01,
        "train/maxent_prefix_ratio_mean": 1.0,
        "train/maxent_prefix_ratio_max": 1.1,
        "train/maxent_prefix_ratio_clipfrac": 0.0,
        "actor/response_tok_len": 4.0,
        "actor/no_eos_count": 0.0,
        "actor/rewards": 1.0,
        "eval/average/accuracy": 0.5,
    }


def _write(root, stamp: str, arm: str, rows: list[dict]) -> None:
    debug = root / f"run_{stamp}_{arm}_s9005" / "debug_1"
    debug.mkdir(parents=True)
    (debug / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _complete(root) -> None:
    _write(
        root,
        LITERAL_STAMP,
        "maxent",
        [_row(step, raw_entropy=RAW_TARGET) for step in range(33)],
    )
    for arm in ("maxent", "maxent_control", "maxent_dual"):
        rows = [
            _row(
                step,
                raw_entropy=RAW_TARGET,
                alpha=0.05 if arm == "maxent" else 0.1,
            )
            for step in range(128)
        ]
        if arm != "maxent":
            target_key = (
                "train/maxent_control_target_entropy"
                if arm == "maxent_control"
                else "train/maxent_dual_target_entropy"
            )
            for row in rows:
                row[target_key] = RAW_TARGET
        _write(root, COMPARATIVE_STAMP, arm, rows)


def test_literal_gate_accepts_raw_entropy_and_normalized_diagnostic(tmp_path):
    _complete(tmp_path)

    summary = inspect_literal(tmp_path)

    assert summary["tail_entropy"] == pytest.approx(RAW_TARGET)


def test_literal_gate_rejects_old_normalized_entropy_units(tmp_path):
    _complete(tmp_path)
    path = next(tmp_path.glob(f"*{LITERAL_STAMP}*/debug_1/train_metrics.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["train/maxent_sequence_entropy_per_tmax"] *= T_MAX
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="raw/per-T_max"):
        inspect_literal(tmp_path)


def test_full_gate_requires_adaptive_raw_target_retention(tmp_path):
    _complete(tmp_path)
    path = next(
        tmp_path.glob(f"*{COMPARATIVE_STAMP}_maxent_control*/debug_1/*.jsonl")
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows[-16:]:
        row["train/maxent_sequence_entropy"] = 0.49 * RAW_TARGET
        row["train/maxent_sequence_entropy_per_tmax"] = (
            0.49 * RAW_TARGET / T_MAX
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="retained 0.490 of target"):
        inspect_all(tmp_path)
