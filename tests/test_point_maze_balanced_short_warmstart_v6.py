from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v6_protocol_changes_only_sft_duration_after_v5_oversaturation():
    text = " ".join(
        (
            ROOT
            / "paper/preregistration/point_maze_balanced_short_warmstart_v6_20260730.md"
        ).read_text().split()
    )
    assert "BEFORE V6 MODEL UPDATE" in text
    assert "243 of 256" in text
    assert "exactly three epochs (69 AdamW optimizer steps)" in text
    assert "unchanged inclusive interval [0.02, 0.50]" in text


def test_v6_batch_uses_short_flag_and_frozen_seeds():
    text = (
        ROOT / "ops/slurm/train_point_maze_balanced_short_warmstart_v6.slurm"
    ).read_text()
    assert "--frozen-short-balanced-v6" in text
    assert "--seed 76621 --epochs 3" in text
    assert "--seed 76622" in text
    assert "point_maze_algorithm_repair_v3/dev" in text
