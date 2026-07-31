from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v12_cross_node_protocol_and_wrappers_are_exactly_bound():
    protocol = (ROOT / "paper/preregistration/ant_maze_v12_cross_node_audit_20260730.md").read_text()
    assert "30200585" in protocol and "216 total" in protocol and "three distinct" in protocol
    exporter = (ROOT / "ops/export_ant_maze_v12_cross_node_specs.py").read_text()
    auditor = (ROOT / "ops/audit_ant_maze_cross_node_v12.py").read_text()
    aggregate = (ROOT / "ops/aggregate_ant_maze_cross_node_v12.py").read_text()
    assert "ant-maze-modebench-data-v12" in exporter
    assert 'CONTROLLER_IDENTITY_FIELD = "executor_identity_sha256"' in exporter
    assert "execute_ant_v12_raw" in auditor
    assert "eligible_for_frozen_05b_viability_gate_v12" in aggregate


def test_v12_cross_node_batch_is_three_nodes_and_three_repetitions():
    text = (ROOT / "ops/slurm/audit_ant_maze_cross_node_v12.slurm").read_text()
    assert "#SBATCH --nodes=3" in text
    assert "#SBATCH --ntasks=3" in text
    assert "--repetitions 3" in text
