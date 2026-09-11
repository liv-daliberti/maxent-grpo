from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e55_protocol_freezes_measure_and_information_boundary():
    protocol = (
        ROOT / "paper/preregistration/e55_per_rollout_verified_anchor_05b.md"
    ).read_text(encoding="utf-8")

    assert "FROZEN BEFORE E55 ENGINEERING SMOKE OR SENTINEL SUBMISSION" in protocol
    assert "`c_replay = 1 / num_samples = 1/16`" in protocol
    assert "alpha itself is\nnot bounded" in protocol
    assert "singleton replay update" in protocol
    assert "may not use gold answers or support counts" in protocol
    assert "cannot tune E55" in protocol


def test_e55_launcher_selects_scaled_singleton_objective_only():
    launcher = (
        ROOT
        / "ops/exp_scaling/launch_e55_per_rollout_verified_anchor_05b.sh"
    ).read_text(encoding="utf-8")

    assert "OAT_ZERO_ONLY_ARMS=maxent_inverse_canonical_replay" in launcher
    assert (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE="
        "verified_likelihood_per_rollout"
    ) in launcher
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10" in launcher
    assert "config|smoke_graph|smoke_python|sentinel" in launcher
    assert '"objective_scale": 0.0625' in launcher
    assert '"applied_score_gradient_sum": -0.0625' in launcher
    assert "e53_control_identity_sha256" in launcher
    assert "OAT_ZERO_SBATCH_HOLD=1" in launcher


def test_e55_auditor_distinguishes_actuator_from_entropy_eligibility():
    auditor = (
        ROOT / "ops/exp_scaling/audit_e55_sentinel.py"
    ).read_text(encoding="utf-8")

    for field in (
        "canonical_replay_actuator_groups",
        "canonical_replay_eligible_groups",
        "canonical_replay_objective_scale",
        "canonical_replay_applied_score_gradient_sum",
        "canonical_replay_observation_skipped",
    ):
        assert field in auditor
    assert "singleton anchor advanced/faked entropy control" in auditor
    assert "OBJECTIVE_SCALE = 1.0 / 16.0" in auditor
    assert "BASE.behavioral_gate(control, treatment)" in auditor


def test_e55_training_path_uses_singleton_groups_only_for_scaled_objective():
    learner = (
        ROOT / "src/oat_drgrpo/learner/grpo.py"
    ).read_text(encoding="utf-8")

    assert '"verified_likelihood_per_rollout"' in learner
    assert "1.0 / float(args.num_samples)" in learner
    assert "min_modes=(" in learner
    assert "canonical_replay_applied_score_gradient_sum" in learner
