from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_protocol_records_v1_failure_and_freezes_compact_repair():
    text = (
        ROOT
        / "paper/preregistration/point_maze_interactive_warmstart_v2_20260730.md"
    ).read_text()
    assert "0/256 verified routes" in text
    assert "24,328" in text
    assert "BEFORE V2 DATA MATERIALIZATION" in text
    assert "exactly 184 AdamW optimizer steps" in text
    assert "seed `75104`" in text


def test_v2_launcher_is_held_and_hash_binds_v1_failure():
    text = (
        ROOT / "ops/exp_scaling/launch_point_maze_interactive_warmstart_v2.sh"
    ).read_text()
    assert "sbatch --parsable --hold" in text
    assert "--partition=all --account=mltheory" in text
    assert '"v1_failure_receipt_sha256":digest(sys.argv[13])' in text
    assert '"policy_interface":"compact_state_v2"' in text
    assert '"sft_optimizer_steps":184' in text
    assert '"prelaunch_canceled_job_ids":[30197923,30197998]' in text
    assert 'scontrol update "JobId=$job_id" Partition=all' in text


def test_v2_slurm_uses_compact_interface_and_unchanged_dev_gate():
    text = (
        ROOT / "ops/slurm/train_point_maze_interactive_warmstart_v2.slurm"
    ).read_text()
    assert text.count("--policy-interface compact_state_v2") == 2
    assert "--epochs 8" in text
    assert "--minimum-prefix-success-prompts 2" in text
    assert "--minimum-multimode-prompts 1" in text
