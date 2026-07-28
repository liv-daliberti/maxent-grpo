from oat_drgrpo.logging_utils import filter_wandb_logs


def test_filter_wandb_logs_hides_low_level_ppo_metrics(monkeypatch):
    monkeypatch.delenv("OAT_ZERO_WANDB_LOG_DEBUG_METRICS", raising=False)
    logs = {
        "train/logprobs_diff_max": 1.0,
        "train/agg_eff_rollouts": 15.0,
    }
    assert filter_wandb_logs(logs) == {"train/agg_eff_rollouts": 15.0}


def test_filter_wandb_logs_can_keep_debug_metrics(monkeypatch):
    monkeypatch.setenv("OAT_ZERO_WANDB_LOG_DEBUG_METRICS", "1")
    logs = {"train/logprobs_diff_max": 1.0}
    assert filter_wandb_logs(logs) == logs
