from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_balanced_v6_final_has_five_shared_seeds_and_untouched_families():
    text = (
        ROOT / "ops/train_point_maze_balanced_v6_stage_b_05b_12pass.py"
    ).read_text()
    assert "SEEDS = (76631, 76632, 76633, 76634, 76635)" in text
    for family in (
        "medium_wide_block9_v3",
        "medium_lower_offset9_v3",
        "hard_wide_block11_v3",
        "hard_diamond11_v3",
    ):
        assert family in text


def test_balanced_v6_final_protocol_is_prospective_and_compute_matched():
    text = (
        ROOT
        / "paper/preregistration/point_maze_balanced_v6_stage_b_05b_12pass_20260730.md"
    ).read_text()
    assert "BEFORE THE V6 WARM-START QUALIFICATION OUTCOME" in text
    assert "Seeds: 76631, 76632, 76633, 76634, and 76635" in text
    assert "checkpoint-invariant common-random-number" in text
    assert "No efficacy threshold" in text


def test_balanced_v6_batches_bind_new_model_data_and_qualification():
    train_batch = (
        ROOT / "ops/slurm/train_point_maze_balanced_v6_stage_b_05b_12pass.slurm"
    ).read_text()
    audit_batch = (
        ROOT / "ops/slurm/audit_point_maze_balanced_v6_stage_b_05b_12pass.slurm"
    ).read_text()
    assert "point_maze_interactive_warmstart_v6_balanced_short" in train_batch
    assert "point_maze_algorithm_repair_v3/eval" in train_batch
    assert "point_maze_balanced_short_warmstart_v6_qualification.json" in train_batch
    assert "point-maze-balanced-short-warmstart-v6-qualification-v1" in (
        ROOT / "ops/train_point_maze_balanced_v6_stage_b_05b_12pass.py"
    ).read_text()
    assert "point_maze_balanced_v6_stage_b_05b_12pass_audit.json" in audit_batch
