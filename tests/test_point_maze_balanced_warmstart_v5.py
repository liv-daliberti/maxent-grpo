from pathlib import Path

import make_point_maze_balanced_warmstart_v5_data as data


ROOT = Path(__file__).resolve().parents[1]


def test_v5_assignments_are_exactly_orientation_balanced_and_compute_matched():
    counts = {
        rotation: sum(item_rotation == rotation for _family, item_rotation in data.ASSIGNMENTS)
        for rotation in range(4)
    }
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2}
    assert sorted(family for family, _rotation in data.ASSIGNMENTS) == [
        "asymmetric_block9",
        "asymmetric_block9",
        "bar7",
        "bar7",
        "bar9",
        "bar9",
        "block9",
        "block9",
    ]


def test_v5_protocol_freezes_checkpoint_repair_before_sampling():
    text = (
        ROOT
        / "paper/preregistration/point_maze_balanced_warmstart_v5_20260730.md"
    ).read_text()
    assert "BEFORE V5" in text
    assert "exactly 276 AdamW steps" in text
    assert "unchanged inclusive interval [0.02, 0.50]" in text
    assert "no evaluation" in text


def test_v5_batch_keeps_public_markov_contract_and_fails_closed():
    text = (
        ROOT / "ops/slurm/train_point_maze_balanced_warmstart_v5.slurm"
    ).read_text()
    assert text.count("--policy-interface velocity_state_v3") == 2
    assert "--seed 76601" in text
    assert "--seed 76602" in text
    assert "qualify_point_maze_balanced_warmstart_v5.py" in text
