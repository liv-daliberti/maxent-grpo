from __future__ import annotations

import json

import pytest

from exp_scaling.check_e4_7b_smoke import discover_run, inspect_run


def _make_run(tmp_path, *, arm="xdr_tau_control", max_step=256):
    run_dir = tmp_path / f"xdr_model_{arm}_stamp_{arm}_s9001"
    debug_dir = run_dir / "debug_attempt"
    debug_dir.mkdir(parents=True)
    rows = [
        {
            "trainer/global_step": 64,
            "train/entropy": 0.25,
            "train/xdr_tau_used": 0.05,
            "train/xdr_tau_control_entropy_ema": 0.24,
            "train/xdr_tau_control_next_tau": 0.05,
            "train/xdr_tau_control_target_entropy": 0.20,
            "train/xdr_tau_control_observations": 64,
        },
        {
            "trainer/global_step": max_step,
            "train/entropy": 0.18,
            "train/xdr_tau_used": 0.04,
            "train/xdr_tau_control_entropy_ema": 0.19,
            "train/xdr_tau_control_next_tau": 0.04,
            "train/xdr_tau_control_target_entropy": 0.20,
            "train/xdr_tau_control_observations": max_step,
        },
    ]
    metrics = debug_dir / "train_metrics.jsonl"
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    checkpoint = debug_dir / "checkpoints" / f"step_{max_step:05d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"state")
    return run_dir


def test_smoke_check_accepts_controller_diagnostics_and_checkpoint(tmp_path):
    run_dir = _make_run(tmp_path)

    result = inspect_run(
        run_dir,
        arm="xdr_tau_control",
        expected_step=256,
        warmup_steps=64,
    )

    assert result["max_step"] == 256
    assert result["checkpoint_step"] == 256
    assert result["controller_observations"] == 256


def test_smoke_check_rejects_short_run(tmp_path):
    run_dir = _make_run(tmp_path, max_step=128)

    with pytest.raises(RuntimeError, match="expected at least 256"):
        inspect_run(
            run_dir,
            arm="xdr_tau_control",
            expected_step=256,
            warmup_steps=64,
        )


def test_discover_run_uses_exact_stamp_arm_seed_suffix(tmp_path):
    expected = tmp_path / "prefix_stamp_grpo_s9001"
    expected.mkdir()
    (tmp_path / "prefix_other_grpo_s9001").mkdir()

    assert (
        discover_run(tmp_path, stamp="stamp", arm="grpo", seed=9001) == expected
    )
