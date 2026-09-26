from __future__ import annotations

import json

import pytest

from exp_scaling.check_e10_prefix_maxent_smoke import (
    COMPARATIVE_STAMP,
    STRESS_STAMP,
    TARGET_ENTROPY,
    discover_rows,
    inspect_all,
)


def _row(step: int, *, entropy: float, alpha: float) -> dict[str, float | int]:
    return {
        "misc/global_step": step,
        "train/pg_loss": 0.1,
        "train/policy_grad_norm": 0.2,
        "train/maxent_alpha_used": alpha,
        "train/maxent_sequence_entropy": entropy,
        "train/maxent_sampled_prefix_entropy": entropy,
        "train/maxent_entropy_surrogate": entropy,
        "train/maxent_entropy_loss": -0.01,
        "train/maxent_prefix_ratio_mean": 1.0,
        "train/maxent_prefix_ratio_max": 1.1,
        "train/maxent_prefix_ratio_clipfrac": 0.1,
        "actor/response_tok_len": 4.0,
        "actor/no_eos_count": 0.0,
        "actor/rewards": 1.0,
        "eval/average/accuracy": 0.5,
    }


def _write_run(root, stamp: str, arm: str, rows: list[dict]) -> None:
    debug = root / f"run_{stamp}_{arm}_s9005" / "debug_1"
    debug.mkdir(parents=True)
    (debug / "train_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_complete_grid(root, *, adaptive_entropy: float = TARGET_ENTROPY) -> None:
    for arm in ("maxent", "maxent_control", "maxent_dual"):
        alpha = 0.05 if arm == "maxent" else 0.2
        entropy = 0.01 if arm == "maxent" else adaptive_entropy
        rows = [_row(step, entropy=entropy, alpha=alpha) for step in range(128)]
        target_key = (
            "train/maxent_control_target_entropy"
            if arm == "maxent_control"
            else "train/maxent_dual_target_entropy"
        )
        if arm != "maxent":
            for row in rows:
                row[target_key] = TARGET_ENTROPY
        _write_run(root, COMPARATIVE_STAMP, arm, rows)
    _write_run(
        root,
        STRESS_STAMP,
        "maxent",
        [_row(step, entropy=0.02, alpha=0.5) for step in range(128)],
    )


def test_e10_gate_accepts_complete_finite_runs(tmp_path):
    _write_complete_grid(tmp_path)

    summaries = inspect_all(tmp_path)

    assert [row["arm"] for row in summaries] == [
        "maxent",
        "maxent_control",
        "maxent_dual",
        "maxent_stress_a0p50",
    ]


def test_e10_gate_rejects_adaptive_entropy_below_half_target(tmp_path):
    _write_complete_grid(tmp_path, adaptive_entropy=0.49 * TARGET_ENTROPY)

    with pytest.raises(RuntimeError, match="retained 0.490 of target"):
        inspect_all(tmp_path)


def test_e10_discovery_uses_furthest_attempt_and_deduplicates_steps(tmp_path):
    suffix = f"{COMPARATIVE_STAMP}_maxent_s9005"
    older = tmp_path / f"old_{suffix}" / "debug_1"
    newer = tmp_path / f"new_{suffix}" / "debug_2"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "train_metrics.jsonl").write_text(
        json.dumps(_row(8, entropy=0.01, alpha=0.05)) + "\n",
        encoding="utf-8",
    )
    duplicate = _row(9, entropy=0.02, alpha=0.05)
    replacement = _row(9, entropy=0.03, alpha=0.05)
    (newer / "train_metrics.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                _row(7, entropy=0.01, alpha=0.05),
                duplicate,
                replacement,
                _row(10, entropy=0.04, alpha=0.05),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    rows = discover_rows(tmp_path, COMPARATIVE_STAMP, "maxent")

    assert [row["misc/global_step"] for row in rows] == [7, 9, 10]
    assert rows[1]["train/maxent_sequence_entropy"] == pytest.approx(0.03)
