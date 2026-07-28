import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e58_global_replay_sentinel.sh"
)
AUDITOR = (
    ROOT / "ops/exp_scaling/audit_e58_global_replay_sentinel.py"
)
REFRESH = ROOT / "ops/exp_scaling/refresh_latest_freeform_05b.py"
PLOT = ROOT / "ops/exp_scaling/plot_divergence.py"
WATCH = ROOT / "ops/exp_scaling/watch_e55_sentinel.sh"
MONITOR = ROOT / "ops/exp_scaling/monitor_campaign.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location(
        "e58_global_replay_sentinel_test",
        AUDITOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(step: int, *, discovered: bool, mass_observations: int):
    row = {
        "trainer/global_step": float(step),
        "train/online_canonical_tracked_outcomes": float(discovered),
        "train/online_canonical_task_reward_mean": 0.0,
        "train/semantic_shannon_augmented_reward_mean": 0.0,
        "train/canonical_replay_mass_observations": float(
            mass_observations
        ),
        "train/policy_grad_norm": 0.0,
    }
    if discovered:
        row.update(
            {
                "train/canonical_replay_global_scheduler_active": 1.0,
                "train/canonical_replay_global_groups_per_step": 1.0,
                "train/canonical_replay_available_groups": 1.0,
                "train/canonical_replay_available_modes": 1.0,
                "train/canonical_replay_actuator_groups": 1.0,
            }
        )
    else:
        row["train/canonical_replay_available_groups"] = 0.0
    return row


def test_sentinel_launch_is_smoke_bound_target_free_and_global():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "requires a clean terminal Python smoke" in launcher
    assert "smoke log contains a crash signature" in launcher
    assert "e58_global_verified_replay_python_smoke_attempt3_identity" in (
        launcher
    )
    assert "OAT_ZERO_ONLY_ARMS=verified_first_global_replay_canonical" in (
        launcher
    )
    assert (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1"
        in launcher
    )
    assert "OAT_ZERO_MAXENT_ALPHA=0" in launcher
    assert '"desired_entropy": None' in launcher
    assert '"desired_mode_count": None' in launcher
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_K=8" in launcher
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=50" in launcher
    assert "OAT_ZERO_SBATCH_HOLD=1" in launcher


def test_sentinel_auditor_requires_checkpointed_global_scheduler():
    auditor = AUDITOR.read_text(encoding="utf-8")

    assert "global_replay_groups_per_step" in auditor
    assert "global_replay_cursor" in auditor
    assert "canonical_replay_global_scheduler_active" in auditor
    assert "global replay was not active on every post-discovery update" in (
        auditor
    )
    assert "smoke_audit.get(\"status\") != \"pass\"" in auditor
    assert "verified_first_global_replay_canonical" in auditor


def test_live_monitor_promotes_e58_and_keeps_the_x_axis_uncapped():
    refresh = REFRESH.read_text(encoding="utf-8")
    plot = PLOT.read_text(encoding="utf-8")
    watch = WATCH.read_text(encoding="utf-8")
    monitor = MONITOR.read_text(encoding="utf-8")

    assert "E58_CURRENT_CANONICAL_CELLS" in refresh
    assert "if E58_SENTINEL_IDENTITY.is_file()" in refresh
    assert "max_training_passes = (" in refresh
    assert "if args.current_canonical_only" in refresh
    assert "else cell[3]" in refresh
    assert "e58_active" in plot
    assert "verified_first_global_replay_canonical" in plot
    assert "ax.set_xlim(0.0, live_x_upper)" in plot
    assert "sentinel_trace = observed_seed_count < 3" in plot
    assert "not a three-seed mean; replication remains gated" in plot
    assert "audit_e58_sentinel.py" in watch
    assert "OAT_ZERO_E58_AUDIT_PYTHON" in watch
    assert "E58_SENTINEL_PREFIXES" in monitor
    assert "global verified-replay sentinel" in monitor


def test_scheduler_gate_accepts_zero_gradient_then_one_replay_per_update():
    auditor = _load_auditor()
    records = [
        _row(1, discovered=False, mass_observations=0),
        _row(2, discovered=False, mass_observations=0),
        _row(3, discovered=True, mass_observations=1),
        _row(4, discovered=True, mass_observations=2),
    ]

    result = auditor._scheduler_and_cold_start_gate(
        records,
        complete=False,
    )

    assert result["status"] == "running"
    assert result["violations"] == []
    assert result["first_discovery_step"] == 3
    assert result["pre_discovery_points"] == 2
    assert result["post_discovery_points"] == 2
    assert result["replay_activations"] == 2


def test_scheduler_gate_rejects_a_post_discovery_actuator_gap():
    auditor = _load_auditor()
    records = [
        _row(1, discovered=False, mass_observations=0),
        _row(2, discovered=True, mass_observations=1),
        _row(3, discovered=True, mass_observations=2),
    ]
    records[-1]["train/canonical_replay_available_groups"] = 0.0

    result = auditor._scheduler_and_cold_start_gate(
        records,
        complete=True,
    )

    assert result["status"] == "fail"
    assert any(
        "canonical_replay_available_groups" in violation
        for violation in result["violations"]
    )
    assert any(
        "global replay was not active" in violation
        for violation in result["violations"]
    )


def test_runtime_log_gate_rejects_hidden_crash_signature(
    tmp_path, monkeypatch
):
    auditor = _load_auditor()
    monkeypatch.setattr(auditor, "ROOT", tmp_path)
    log_root = tmp_path / "var/artifacts/logs"
    log_root.mkdir(parents=True)
    (log_root / "xdr_train-123.out").write_text(
        "Traceback (most recent call last)\n",
        encoding="utf-8",
    )

    result = auditor._runtime_log_gate(123, complete=True)

    assert result["status"] == "fail"
    assert result["matches"]
