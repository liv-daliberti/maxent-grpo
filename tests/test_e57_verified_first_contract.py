from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "ops" / "run_experiment.sh"
SUBMITTER = ROOT / "ops" / "submit_countdown_comparative.sh"
PROTOCOL = (
    ROOT
    / "paper"
    / "preregistration"
    / "e57_verified_first_split_canonical_05b.md"
)
LAUNCHER = ROOT / "ops" / "exp_scaling" / "launch_e57_verified_first_smoke.sh"
GRAPH_LAUNCHER = ROOT / "ops" / "exp_scaling" / "launch_e57_graph_smoke.sh"


def test_verified_first_variant_has_no_direct_maxent_cold_start():
    source = RUN_EXPERIMENT.read_text(encoding="utf-8")
    start = source.index("  verified_first_split_canonical)")
    stop = source.index('    VARIANT_TAG="verified_first_split_canonical"', start)
    branch = source[start:stop]

    assert "OAT_ZERO_MAXENT_ALPHA=0.0" in branch
    assert "OAT_ZERO_MAXENT_INVERSE_ADAPTATION=0" in branch
    assert "OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=1" in branch
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1" in branch
    assert (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE="
        "split_mass_balance_per_rollout"
    ) in branch


def test_submitter_exports_verified_first_objective_isolation():
    source = SUBMITTER.read_text(encoding="utf-8")

    assert "OAT_ZERO_INCLUDE_VERIFIED_FIRST_SPLIT_CANONICAL_ARM" in source
    assert (
        "submit_arm verified_first_split_canonical "
        "verified_first_split_canonical"
    ) in source
    assert (
        '"$variant" == "open_set_split_canonical" || '
        '"$variant" == "verified_first_split_canonical"'
    ) in source
    assert (
        "maxent_inverse_canonical_replay|open_set_split_canonical)"
    ) in source


def test_protocol_freezes_unbounded_model_score_controllers():
    text = PROTOCOL.read_text(encoding="utf-8")

    assert "zero-gradient cold start" in text
    assert "no lower or upper projection" in text
    assert "desired mode count" in text
    assert "desired entropy" in text
    assert "direct MaxEnt telemetry and controller state are absent" in text


def test_smoke_launcher_replays_problematic_seed_without_direct_entropy():
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "OAT_ZERO_TRAIN_SEEDS=9010" in source
    assert "OAT_ZERO_ONLY_ARMS=verified_first_split_canonical" in source
    assert "OAT_ZERO_INCLUDE_VERIFIED_FIRST_SPLIT_CANONICAL_ARM=1" in source
    assert "OAT_ZERO_MAXENT_ALPHA=0" in source
    assert "OAT_ZERO_MAX_TRAIN=128" in source
    assert "OAT_ZERO_EVAL_PROMPT_INTERVAL=1000000" in source
    assert "OAT_ZERO_SBATCH_HOLD=1" in source
    assert 'scontrol update "JobId=$job_id" Requeue=0' in source


def test_graph_smoke_is_gated_on_terminal_python_smoke():
    source = GRAPH_LAUNCHER.read_text(encoding="utf-8")

    assert "E57 graph smoke requires a clean terminal Python smoke" in source
    assert "OAT_ZERO_TRAIN_SEEDS=9057" in source
    assert "OAT_ZERO_MAX_TRAIN=32" in source
    assert "OAT_ZERO_MAXENT_ALPHA=0" in source
    assert "OAT_ZERO_ONLY_ARMS=verified_first_split_canonical" in source
