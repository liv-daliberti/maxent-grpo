from pathlib import Path


ROOT = Path(__file__).parents[1]
PROTOCOL = (
    ROOT
    / "paper/preregistration/e50_python_factor_extension_20260724.md"
)
PLACEMENT_AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e50_python_factor_non_mltheory_placement_amendment_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/exp_scaling/launch_e50_python_factor_extension.sh"
)
SUBMITTER = ROOT / "ops/submit_countdown_comparative.sh"
MONITOR = ROOT / "ops/exp_scaling/monitor_campaign.py"
REFRESH = ROOT / "ops/exp_scaling/refresh_latest_freeform_05b.py"
PLOT = ROOT / "ops/exp_scaling/plot_divergence.py"


def test_e50_python_extension_is_frozen_and_matched():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    placement = PLACEMENT_AMENDMENT.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for required in (
        "**Status: FROZEN BEFORE LAUNCH",
        "post-launch domain extension",
        "exactly 50 complete 384-prompt passes",
        "`grpo`",
        "`online_canonical_haarnoja`",
        "Seeds: `43,44,45`",
        "no upper projection",
        "return vector",
    ):
        assert required in protocol
    for required in (
        "PREFIX=pye50_uncapped_normalized_canonical_haarnoja_05b_50ep_v3_allcs",
        "EXPECTED_JOBS=6",
        "PROMPT_POOL=384",
        "EVAL_INTERVAL=96",
        "PROMPT_EPOCHS=50",
        "export OAT_ZERO_COMPARATIVE_TASK=python_factor",
        "export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja",
        "export OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=inf",
        "export OAT_ZERO_SBATCH_HOLD=1",
        "scontrol release",
    ):
        assert required in launcher
    for required in (
        "**Status: FROZEN BEFORE RETRY",
        "No job ran",
        "Account: `allcs`",
        "Partition: `all`",
        "one `rtx_3090`",
        "No job ran",
        "QOS `none`",
    ):
        assert required in placement
    assert (
        'OAT_ZERO_E50_PYTHON_TRAIN_ACCOUNT:-allcs'
        in launcher
    )
    assert (
        'OAT_ZERO_E50_PYTHON_TRAIN_GRES:-gpu:rtx_3090:1'
        in launcher
    )


def test_python_factor_is_a_first_class_comparative_task():
    submitter = SUBMITTER.read_text(encoding="utf-8")

    assert "python_factor)" in submitter
    assert "var/data/python_factor_modebench_v1" in submitter
    assert "make_python_factor_mode_data.py" in submitter
    assert (
        "use countdown, graph_coloring, python_factor, or math"
        in submitter
    )


def test_python_extension_is_superseded_by_e51_in_current_surfaces():
    old_prefix = "pye50_uncapped_normalized_canonical_haarnoja_05b_50ep_v3_allcs"
    prefix = "pye51_policy_entropy_adaptive_canonical_05b_50ep_v2_allcs"
    monitor = MONITOR.read_text(encoding="utf-8")
    refresh = REFRESH.read_text(encoding="utf-8")
    plot = PLOT.read_text(encoding="utf-8")

    assert '"Python factors"' in monitor
    assert prefix in monitor
    assert prefix in refresh
    assert prefix in plot
    assert old_prefix not in monitor
    assert old_prefix not in refresh
    assert old_prefix not in plot
    assert "Python factors — E51 (live frontier)" in plot
