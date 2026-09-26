from pathlib import Path

import make_point_maze_geometry_shift_data as shifted
import make_point_maze_mode_data as original


ROOT = Path(__file__).resolve().parents[1]


def test_geometry_shift_families_are_frozen_and_new():
    families = shifted.geometry_shift_families()
    assert [row["family"] for row in families] == [
        "wide_block9_shift",
        "cross9_shift",
        "upper_offset9_shift",
        "lower_offset9_shift",
    ]
    assert all(len(row["maze_map"]) == 9 for row in families)
    assert all(row["reset"] == (4, 1) and row["goal"] == (4, 7) for row in families)
    original_maps = {tuple(map(tuple, row["maze_map"])) for row in original._base_families()}
    shifted_maps = {tuple(map(tuple, row["maze_map"])) for row in families}
    assert len(shifted_maps) == 4
    assert not shifted_maps & original_maps


def test_geometry_shift_protocol_is_prospective_and_honest():
    text = (ROOT / "paper/preregistration/point_maze_geometry_shift_replacement_v1_20260730.md").read_text()
    normalized = " ".join(text.split())
    assert "frozen before model sampling on any new map" in normalized
    assert "not an eighth independent semantic domain" in normalized
    assert "evaluation rows are not loaded" in normalized


def test_geometry_shift_viability_job_is_frozen():
    batch = (ROOT / "ops/slurm/evaluate_point_maze_geometry_shift_viability_v1.slurm").read_text()
    assert "--seed 75121" in batch
    assert "--sample-count 64 --prefix-count 16" in batch
    assert "--minimum-prefix-success-prompts 2 --minimum-multimode-prompts 1" in batch
    assert "point_maze_geometry_shift_v1/dev" in batch
    assert "point_maze_geometry_shift_v1/eval" not in batch
