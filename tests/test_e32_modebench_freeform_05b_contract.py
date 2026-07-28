from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e32_modebench_freeform_05b_ema_10ep.sh"
PROTOCOL = ROOT / "paper/preregistration/e32_modebench_freeform_05b_ema_10ep.md"


def test_e32_is_a_fresh_matched_ten_epoch_cohort():
    text = LAUNCHER.read_text(encoding="utf-8")
    for literal in (
        "gce32_freeform_05b_ema_10ep_v5",
        "cde32_freeform_05b_ema_10ep_v5",
        "OAT_ZERO_ONLY_ARMS=grpo,maxent_dual",
        "OAT_ZERO_TRAIN_SEEDS=43,44,45",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=10",
        "OAT_ZERO_NUM_PROMPT_EPOCH=10",
        "OAT_ZERO_MAXENT_DUAL_EMA_DECAY=0.7",
        "OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.010",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K=8",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED=1001",
        "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0",
        "OAT_ZERO_EXPORT_STEPS=0",
        "OAT_ZERO_SBATCH_HOLD=1",
        "OAT_ZERO_E32_TRAIN_CPUS_PER_TASK:-4",
        "OAT_ZERO_E32_TRAIN_MEMORY:-32G",
        "OAT_ZERO_E32_TRAIN_PARTITION:-all",
    ):
        assert literal in text


def test_e32_protocol_freezes_raw_unsmoothed_evaluation_and_resume_semantics():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "**Status: FROZEN" in text
    assert "four reproducible K=8" in text
    assert "mean, sample SD, SE, minimum, and maximum" in text
    assert "There is no smoothing" in text
    assert "synchronize its weights to every actor" in text
    assert "pre-synchronization resumed evaluation is rejected" in text


def test_latest_monitor_and_chart_route_to_e32_and_e33():
    monitor = (ROOT / "ops/exp_scaling/monitor_campaign.py").read_text(encoding="utf-8")
    refresher = (ROOT / "ops/exp_scaling/refresh_latest_freeform_05b.py").read_text(encoding="utf-8")
    assert "LATEST_05B_PREFIXES" in monitor
    assert "LATEST_3B_PREFIXES" in monitor
    assert "latest_05b_only=not args.all_campaigns" in monitor
    assert "cde32_freeform_05b_ema_10ep_v4_preemptsafe" in refresher
    assert "gce32_freeform_05b_ema_10ep_v5" in refresher
    assert "cde33_freeform_3b_ema_10ep_v3_a100" in refresher
    assert "gce33_freeform_3b_ema_10ep_v3_a100" in refresher
    assert "render_all_divergence_figures()" in refresher


def test_e32_snapshot_hashes_are_location_independent():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'cd "$tree"' in text
    assert "find . -type f -print0" in text


def test_ten_epoch_cohort_explicitly_raises_the_runtime_ceiling():
    train = (ROOT / "ops/train.sh").read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    submitter = (ROOT / "ops/submit_countdown_comparative.sh").read_text(encoding="utf-8")
    assert 'MAX_PROMPT_EPOCHS="${OAT_ZERO_MAX_PROMPT_EPOCHS:-5}"' in train
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=10" in launcher
    assert 'OAT_ZERO_MAX_PROMPT_EPOCHS=${OAT_ZERO_MAX_PROMPT_EPOCHS:-5}' in submitter
    assert 'scontrol update JobId="$job_id" Partition="$OAT_ZERO_TRAIN_PARTITION"' in launcher
    assert 'E32 could not normalize held job' in launcher
