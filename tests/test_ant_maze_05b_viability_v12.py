from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ant_v12_viability_is_frozen_after_cross_node_pass():
    protocol = (ROOT / "paper/preregistration/ant_maze_05b_viability_v12_20260730.md").read_text()
    launcher = (ROOT / "ops/exp_scaling/launch_ant_maze_05b_viability_v12.sh").read_text()
    batch = (ROOT / "ops/slurm/evaluate_ant_maze_05b_viability_v12.slurm").read_text()
    assert "30200962" in protocol and "216/216" in protocol
    assert "seed 107312" in protocol and "first 16" in protocol
    assert "eligible_for_frozen_05b_viability_gate_v12" in launcher
    assert 'ant_maze_modebench_v12/dev"' in batch
    assert "ant_maze_modebench_v12/eval" not in batch
    assert "--sample-count 64" in batch and "--prefix-count 16" in batch
    assert "--seed 107312" in batch


def test_ant_v12_uses_only_the_audited_public_interface_reminder():
    protocol = (ROOT / "paper/preregistration/ant_maze_05b_viability_v12_20260730.md").read_text()
    batch = (ROOT / "ops/slurm/evaluate_ant_maze_05b_viability_v12.slurm").read_text()
    assert "already-audited v10 public" in protocol
    assert "--prompt-repair ant_maze_v10" in batch
