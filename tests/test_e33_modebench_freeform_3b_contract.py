from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e33_modebench_freeform_3b_ema_10ep.sh"
PROTOCOL = ROOT / "paper/preregistration/e33_modebench_freeform_3b_ema_10ep.md"


def test_e33_is_a_fresh_matched_ten_epoch_3b_cohort():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "gce33_freeform_3b_ema_10ep_v3_a100",
        "cde33_freeform_3b_ema_10ep_v3_a100",
        "Qwen2.5-3B-Instruct",
        "OAT_ZERO_ONLY_ARMS=grpo,maxent_dual",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=10",
        "OAT_ZERO_NUM_PROMPT_EPOCH=10",
        "OAT_ZERO_MAXENT_DUAL_EMA_DECAY=0.7",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=1001",
        "OAT_ZERO_ADAM_OFFLOAD=1",
        "OAT_ZERO_ACTIVATION_OFFLOADING=1",
        "OAT_ZERO_E33_TRAIN_NODELIST:-node302",
        "OAT_ZERO_E33_TRAIN_GRES:-gpu:a100:1",
        "OAT_ZERO_E33_TRAIN_CPUS_PER_TASK:-16",
        "OAT_ZERO_E33_TRAIN_MEMORY:-96G",
        "OAT_ZERO_E33_TRAIN_PARTITION:-mltheory",
        "OAT_ZERO_SBATCH_HOLD=1",
    ):
        assert literal in text


def test_e33_protocol_is_prospective_and_excludes_old_trajectories():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "**Status: FROZEN" in text
    assert "four reproducible K=8" in text
    assert "mean, sample SD, SE, minimum, and maximum" in text
    assert "There is no smoothing" in text
    assert "synchronize its weights to every actor" in text
    assert "pre-synchronization resumed evaluation is rejected" in text
    assert "no reuse of the contaminated E25/E28 trajectories" in text
    assert "failed during actor initialization, before step 0" in text


def test_e33_retention_and_release_are_audited():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_SAVE_STEPS=384",
        "OAT_ZERO_MAX_SAVE_NUM=2",
        "OAT_ZERO_MAX_RESUME_NUM=2",
        "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0",
        "Reason=JobHeldUser",
        "scontrol release",
    ):
        assert literal in text


def test_latest_monitor_and_figures_route_to_e33():
    monitor = (ROOT / "ops/exp_scaling/monitor_campaign.py").read_text(encoding="utf-8")
    plotter = (ROOT / "ops/exp_scaling/plot_divergence.py").read_text(encoding="utf-8")
    refresher = (ROOT / "ops/exp_scaling/refresh_latest_freeform_05b.py").read_text(encoding="utf-8")
    for prefix in (
        "cde33_freeform_3b_ema_10ep_v3_a100",
        "gce33_freeform_3b_ema_10ep_v3_a100",
    ):
        assert prefix in monitor
        assert prefix in plotter
        assert prefix in refresher
