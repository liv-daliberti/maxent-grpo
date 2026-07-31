from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v2_freezes_bridge_and_crosses_replay_warmup():
    protocol = (ROOT / "paper/preregistration/pantry_support_mask_paired_integration_v2_20260730.md").read_text()
    launcher = (ROOT / "ops/exp_scaling/launch_pantry_support_mask_paired_integration_v2.py").read_text()
    assert "30200958/30200959" in protocol and "6--11 actor/verifier" in protocol
    assert "96 updates" in protocol and "warmup 64" in protocol
    assert 'base.OPTIMIZER_UPDATES = 96' in launcher
    assert 'OAT_ZERO_MAX_QUERIES="1488"' in launcher
    assert 'OAT_ZERO_NUM_PROMPT_EPOCH="3"' in launcher


def test_v2_audit_requires_exact_actor_verifier_agreement():
    audit = (ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1.py").read_text()
    wrapper = (ROOT / "ops/audit_pantry_support_mask_paired_integration_v2.py").read_text()
    assert "online_canonical_validator_task_disagreement_rows" in audit
    assert 'base.UPDATES = 96' in wrapper
