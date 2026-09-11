from pathlib import Path

from oat_drgrpo.online_canonical_bank import OnlineCanonicalBank


ROOT = Path(__file__).resolve().parents[1]


def test_e58_protocol_is_target_free_and_global_replay_is_fixed_compute():
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e58_global_verified_replay_canonical_05b.md"
    ).read_text(encoding="utf-8")

    assert (
        "every\noptimizer update materializes exactly one previously "
        "model-discovered verified\nbank"
        in protocol
    )
    assert "deterministic round-robin" in protocol
    assert "never reads exhaustive support" in protocol
    assert "desired entropy" in protocol
    assert "gold mode count" in protocol
    assert "evaluation" in protocol
    assert "There is no coefficient projection" in protocol


def test_e58_variant_disables_direct_entropy_and_enables_one_global_group():
    runner = (ROOT / "ops/run_experiment.sh").read_text(encoding="utf-8")
    submitter = (
        ROOT / "ops/submit_countdown_comparative.sh"
    ).read_text(encoding="utf-8")
    trainer = (ROOT / "ops/train.sh").read_text(encoding="utf-8")

    assert "verified_first_global_replay_canonical)" in runner
    assert (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1"
        in runner
    )
    assert "OAT_ZERO_MAXENT_ALPHA=0.0" in runner
    assert "OAT_ZERO_MAXENT_INVERSE_ADAPTATION=0" in runner
    assert "INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM" in submitter
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1" in (
        submitter
    )
    assert "--online-canonical-replay-global-groups-per-step" in trainer


def test_e58_global_schedule_is_checkpointed_and_default_off():
    bank = OnlineCanonicalBank(
        entropy_alpha=0,
        novelty_beta=0,
        retain_exemplars=True,
    )
    state = bank.state_dict()

    assert state["global_replay_groups_per_step"] == 0
    assert state["global_replay_cursor"] == 0
    assert bank.scheduled_global_replay_groups() == []


def test_global_replay_requires_replay_objective():
    source = (ROOT / "src/oat_drgrpo/args.py").read_text(encoding="utf-8")

    assert "online_canonical_replay_global_groups_per_step < 0" in source
    assert (
        "online_canonical_replay_global_groups_per_step > 0"
        in source
    )
    assert (
        '"online_canonical_replay_global_groups_per_step requires replay"'
        in source
    )


def test_e58_smoke_is_exact_job_bound_and_checks_mechanism():
    launcher = (
        ROOT / "ops/exp_scaling/launch_e58_global_replay_smoke.sh"
    ).read_text(encoding="utf-8")
    auditor = (
        ROOT / "ops/exp_scaling/audit_e58_global_replay_smoke.py"
    ).read_text(encoding="utf-8")

    assert "debug_job" in auditor
    assert "post_discovery_records" in auditor
    assert "global replay was not active on every post-discovery update" in auditor
    assert "persistent global replay scheduler" in auditor
    assert "OAT_ZERO_SBATCH_HOLD=1" in launcher
    assert "scontrol release" in launcher
