from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r1_is_validator_only_and_binds_failed_jobs():
    protocol = (ROOT / "paper/preregistration/pantry_support_mask_paired_mechanism_smoke_v1_r1_validator_repair_20260730.md").read_text()
    launcher = (ROOT / "ops/exp_scaling/launch_pantry_support_mask_paired_smoke_v1_r1.py").read_text()
    assert "30200705--30200707" in protocol
    assert "argument-admission contract" in protocol
    assert "ten Pantry Stage-B jobs" in protocol
    assert "30200705.err" in launcher and "30200706.err" in launcher
    assert 'base.PREFIX = "ppsmoke_support_mask_paired_v1_r1"' in launcher


def test_r1_audit_uses_fresh_paths_and_unchanged_auditor():
    wrapper = (ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1_r1.py").read_text()
    batch = (ROOT / "ops/slurm/audit_pantry_support_mask_paired_smoke_v1_r1.slurm").read_text()
    assert "audit_pantry_support_mask_paired_smoke_v1 as base" in wrapper
    assert "pantry_support_mask_paired_smoke_v1_r1_identity.json" in batch
    assert "ppsmoke_support_mask_paired_v1_r1_comparative_jobs.tsv" in batch
