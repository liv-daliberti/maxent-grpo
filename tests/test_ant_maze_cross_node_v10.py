from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v10_cross_node_gate_is_conditional_and_exact_slate():
    protocol = (
        ROOT / "paper/preregistration/ant_maze_v10_cross_node_audit_20260729.md"
    ).read_text()
    launcher = (
        ROOT / "ops/exp_scaling/launch_ant_maze_cross_node_v10.sh"
    ).read_text()
    slurm = (ROOT / "ops/slurm/audit_ant_maze_cross_node_v10.slurm").read_text()
    assert "FROZEN DURING V10 CONTROLLER TRAINING" in protocol
    assert "216 total" in protocol
    assert "exact 12 admitted data rows" in protocol
    assert 'audit.get("status") != "pass"' in launcher
    assert "sbatch --parsable --hold" in launcher
    assert '"node_count":3' in launcher
    assert "#SBATCH --nodes=3" in slurm
    assert "#SBATCH --ntasks=3" in slurm
    assert 'OAT_ZERO_PROTOCOL_IDENTITY="$ROUTE_IDENTITY"' in slurm


def test_v10_cross_node_replays_exported_specs_and_expected_keys():
    exporter = (ROOT / "ops/export_ant_maze_v10_cross_node_specs.py").read_text()
    auditor = (ROOT / "ops/audit_ant_maze_cross_node_v10.py").read_text()
    aggregate = (ROOT / "ops/aggregate_ant_maze_cross_node_v10.py").read_text()
    assert "load_from_disk" in exporter
    assert "route_identity_sha256" in exporter
    assert "execute_ant_v10_raw" in auditor
    assert 'validation.canonical_key != route["canonical_key"]' in auditor
    assert 'len(records) == 72' in auditor
    assert "three distinct nodes" in aggregate
    assert '"eligible_for_frozen_05b_viability_gate_v10"' in aggregate
