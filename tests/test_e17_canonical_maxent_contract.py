"""Recovery contract for the frozen E17 canonical-MaxEnt cohort."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "ops/exp_scaling/launch_e17_canonical_maxent_3b.sh"


def test_e17_launcher_has_countdown_fixed_only_same_stamp_recovery():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "fixed-config" in text
    assert "fixed-retry" in text
    assert "export OAT_ZERO_ONLY_ARMS=maxent" in text
    assert "export OAT_ZERO_APPEND_MANIFEST=1" in text
    assert "manifest_lines_before" in text
    assert "released three Countdown fixed-MaxEnt recovery jobs" in text


def test_e17_launcher_can_stage_only_fixed_seed45_on_stable_capacity():
    text = LAUNCHER.read_text(encoding="utf-8")
    stage = text.split('if [[ "$PHASE" == fixed-seed45-stage ]]', 1)[1]
    stage = stage.split('if [[ "$PHASE" == fixed-retry ]]', 1)[0]

    assert "export OAT_ZERO_TRAIN_SEEDS=45" in stage
    assert "export OAT_ZERO_ONLY_ARMS=maxent" in stage
    assert "export OAT_ZERO_APPEND_MANIFEST=1" in stage
    assert "export OAT_ZERO_TRAIN_NODELIST=node302" in stage
    assert "export OAT_ZERO_TRAIN_GRES=gpu:a100:1" in stage
    assert "export OAT_ZERO_TRAIN_PARTITION=mltheory" in stage
    assert '$1 == "maxent" && $2 == "45"' in stage
    assert "Reason=JobHeldUser" in stage
    assert "staged one held Countdown fixed-MaxEnt seed-45 recovery job" in stage
    assert "scontrol release" not in stage


def test_e17_launcher_has_countdown_dual_only_same_stamp_recovery():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "dual-config" in text
    assert "dual-retry" in text
    assert "export OAT_ZERO_ONLY_ARMS=maxent_dual" in text
    assert '$1 == "maxent_dual"' in text
    assert "comm -12" in text
    assert 'squeue -h -u "$USER"' in text
    assert "scancel \"${old_live_ids[@]}\"" in text
    assert "released three Countdown dual-MaxEnt recovery jobs" in text


def test_e17_launcher_has_countdown_proportional_node302_recovery():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "control-config" in text
    assert "control-retry" in text
    assert "export OAT_ZERO_TRAIN_SEEDS=43,44" in text
    assert "export OAT_ZERO_ONLY_ARMS=maxent_control" in text
    assert '$1 == "maxent_control"' in text
    assert "ReqNodeList=node302" in text
    assert "TresPerNode=gres/gpu:a100:1" in text
    assert "released two Countdown proportional-MaxEnt node302 recovery jobs" in text
