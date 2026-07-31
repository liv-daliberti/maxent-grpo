from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r1_is_preexecution_binding_only_repair():
    protocol = (ROOT / "paper/preregistration/ant_maze_v12_cross_node_audit_r1_controller_binding_repair_20260730.md").read_text()
    launcher = (ROOT / "ops/exp_scaling/launch_ant_maze_cross_node_v12_r1.sh").read_text()
    assert "30200738" in protocol and "zero route replays" in protocol
    assert "216 total real-simulator executions" in protocol
    assert "executor_identity_sha256" in launcher
    assert '"scientific_change":False' in launcher


def test_r1_batch_preserves_three_node_exact_slate():
    text = (ROOT / "ops/slurm/audit_ant_maze_cross_node_v12_r1.slurm").read_text()
    assert "#SBATCH --nodes=3" in text and "#SBATCH --ntasks=3" in text
    assert "--repetitions 3" in text
    assert "bash --noprofile --norc" in text
