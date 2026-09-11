from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v3_repairs_only_the_query_budget_arithmetic():
    protocol = (ROOT / "paper/preregistration/pantry_support_mask_paired_integration_v3_budget_repair_20260730.md").read_text()
    launcher = (ROOT / "ops/exp_scaling/launch_pantry_support_mask_paired_integration_v3.py").read_text()
    assert "V2 remains failed and is not relabeled" in protocol
    assert "1520 = (96 - 1) * 16" in protocol
    assert 'base.EXPECTED_MAX_QUERIES = 1520' in launcher
    assert 'values["OAT_ZERO_MAX_QUERIES"] = "1520"' in launcher
    assert 'base.OPTIMIZER_UPDATES = 96' in launcher
    assert '"scientific_setting_change": False' in launcher


def test_v3_audit_is_bound_to_96_updates_and_new_prefix():
    wrapper = (ROOT / "ops/audit_pantry_support_mask_paired_integration_v3.py").read_text()
    assert 'base.UPDATES = 96' in wrapper
    assert 'base.PREFIX = "ppsmoke_support_mask_paired_integration_v3"' in wrapper
