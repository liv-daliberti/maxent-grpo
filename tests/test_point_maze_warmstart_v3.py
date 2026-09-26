from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v3_protocol_freezes_velocity_state_correction_before_sampling():
    text = (ROOT / "paper/preregistration/point_maze_interactive_warmstart_v3_20260730.md").read_text()
    assert "BEFORE V3 DATA MATERIALIZATION" in text
    assert "position alone is not Markov" in text
    assert "exactly 276 AdamW optimizer steps" in text
    assert "seed `75105`" in text


def test_v3_launcher_is_held_and_hash_binds_v2_failure():
    text = (ROOT / "ops/exp_scaling/launch_point_maze_interactive_warmstart_v3.sh").read_text()
    assert "sbatch --parsable --hold" in text
    assert '"v2_failure_receipt_sha256":d(sys.argv[13])' in text
    assert '"policy_interface":"velocity_state_v3"' in text
    assert '"sft_optimizer_steps":276' in text
    assert 'scontrol update "JobId=$job_id" Partition=all' in text


def test_v3_slurm_uses_velocity_interface_and_unchanged_gate():
    text = (ROOT / "ops/slurm/train_point_maze_interactive_warmstart_v3.slurm").read_text()
    assert text.count("--policy-interface velocity_state_v3") == 2
    assert "--epochs 12" in text
    assert "--minimum-prefix-success-prompts 2" in text
    assert "--minimum-multimode-prompts 1" in text
