from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r2_protocol_is_frozen_after_r1_and_before_execution():
    text = (
        ROOT
        / "paper/preregistration/pantry_support_mask_drgrpo_smoke_v1_r2_query_budget_repair_20260730.md"
    ).read_text()
    assert "FROZEN AFTER THE TERMINAL R1 AUDIT AND BEFORE R2" in text
    assert "30199460" in text and "30199464" in text
    assert "32 * 16 - 16 = 496" in text
    assert "R1 remains failed" in text


def test_r2_launcher_changes_only_query_budget_namespace_and_audit_contract():
    implementation = (
        ROOT
        / "ops/exp_scaling/launch_pantry_support_mask_drgrpo_smoke_v1_r1.py"
    ).read_text()
    entrypoint = (
        ROOT
        / "ops/exp_scaling/launch_pantry_support_mask_drgrpo_smoke_v1_r2.py"
    ).read_text()
    audit = (
        ROOT / "ops/audit_pantry_support_mask_drgrpo_smoke_v1.py"
    ).read_text()
    assert 'MAX_QUERIES = "496" if IS_R2 else "32"' in implementation
    assert 'os.environ["PANTRY_REPAIR_SUFFIX"] = "r2"' in entrypoint
    assert '"train/zero_pg_loss_count_inf"' in audit
    assert '"train/zero_pg_loss_count_nan"' in audit
    assert '"train/pg_loss_inf"' not in audit
    assert '"train/pg_loss_nan"' not in audit


def test_r2_keeps_science_and_recovery_settings_frozen():
    text = (
        ROOT
        / "ops/exp_scaling/launch_pantry_support_mask_drgrpo_smoke_v1_r1.py"
    ).read_text()
    for frozen in (
        '"OAT_ZERO_TRAIN_SEEDS": "76201"',
        '"OAT_ZERO_NUM_SAMPLES": "16"',
        '"OAT_ZERO_LEARNING_RATE": "0.0000002"',
        '"OAT_ZERO_MAX_TRAIN": "32"',
        '"OAT_ZERO_NUM_PROMPT_EPOCH": "1"',
        '"OAT_ZERO_NUM_PPO_EPOCHS": "1"',
        '"OAT_ZERO_AUTO_RESUME": "0"',
        '"OAT_ZERO_WATCHDOG_REQUEUE": "0"',
    ):
        assert frozen in text
    assert '"maxent_actuators_enabled": False' in text
    assert '"scientific_change": False' in text


def test_r2_audit_binds_failed_r1_and_fresh_namespace():
    wrapper = (
        ROOT / "ops/audit_pantry_support_mask_drgrpo_smoke_v1_r2.py"
    ).read_text()
    slurm = (
        ROOT / "ops/slurm/audit_pantry_support_mask_drgrpo_smoke_v1_r2.slurm"
    ).read_text()
    assert 'FAILED_JOB_ID = 30199460' in wrapper
    assert 'PREFIX = "ppsmoke_support_mask_drgrpo_v1_r2"' in wrapper
    assert 'identity.get("max_queries") == 496' in wrapper
    assert "PANTRY_TRAIN_JOB_ID" in slurm
    assert "pantry_support_mask_drgrpo_smoke_v1_r2_audit.json" in slurm
