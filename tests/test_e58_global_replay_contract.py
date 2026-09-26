import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARGS = ROOT / "src" / "oat_drgrpo" / "args.py"
BANK = ROOT / "src" / "oat_drgrpo" / "online_canonical_bank.py"
LEARNER = ROOT / "src" / "oat_drgrpo" / "learner" / "grpo.py"
TRAIN = ROOT / "ops" / "train.sh"
RUN = ROOT / "ops" / "run_experiment.sh"
SUBMIT = ROOT / "ops" / "submit_countdown_comparative.sh"
PROTOCOL = (
    ROOT
    / "paper"
    / "preregistration"
    / "e58_global_verified_replay_canonical_05b.md"
)
LAUNCHER = (
    ROOT / "ops" / "exp_scaling" / "launch_e58_global_replay_smoke.sh"
)
AUDITOR = (
    ROOT / "ops" / "exp_scaling" / "audit_e58_global_replay_smoke.py"
)


def _load_auditor():
    spec = importlib.util.spec_from_file_location("e58_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_replay_is_default_off_and_fixed_compute_budget_is_explicit():
    args = ARGS.read_text(encoding="utf-8")
    bank = BANK.read_text(encoding="utf-8")

    assert "online_canonical_replay_global_groups_per_step: int = 0" in args
    assert "scheduled_global_replay_groups" in bank
    assert "sorted(" in bank
    assert "global_replay_cursor" in bank
    assert "gold_support" not in bank


def test_global_replay_variant_routes_one_verified_bank_without_direct_entropy():
    run = RUN.read_text(encoding="utf-8")
    branch_start = run.index("  verified_first_global_replay_canonical)")
    branch_end = run.index("  maxent_length_dual)", branch_start)
    branch = run[branch_start:branch_end]

    assert "OAT_ZERO_MAXENT_ALPHA=0.0" in branch
    assert "OAT_ZERO_MAXENT_INVERSE_ADAPTATION=0" in branch
    assert (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1"
        in branch
    )
    assert 'VARIANT_TAG="verified_first_global_replay_canonical"' in branch


def test_global_scheduler_replaces_prompt_local_selection_only_when_enabled():
    learner = LEARNER.read_text(encoding="utf-8")
    assert (
        "args.online_canonical_replay_global_groups_per_step" in learner
    )
    assert "scheduled_global_replay_groups(" in learner
    assert "online_canonical_bank.replay_groups(" in learner
    assert "canonical_replay_global_scheduler_active" in learner


def test_training_and_submit_surfaces_propagate_global_replay_configuration():
    train = TRAIN.read_text(encoding="utf-8")
    submit = SUBMIT.read_text(encoding="utf-8")

    assert (
        "--online-canonical-replay-global-groups-per-step" in train
    )
    assert (
        "OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM"
        in submit
    )
    assert (
        "submit_arm verified_first_global_replay_canonical "
        "verified_first_global_replay_canonical"
        in submit
    )


def test_frozen_protocol_and_smoke_enforce_the_information_firewall():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    auditor = AUDITOR.read_text(encoding="utf-8")

    assert "FROZEN BEFORE E58 ENGINEERING SMOKE" in protocol
    assert "fixed one-bank budget" in protocol
    assert "There is no coefficient projection" in protocol
    assert "No ground-truth support size" in protocol
    assert "OAT_ZERO_ONLY_ARMS=verified_first_global_replay_canonical" in launcher
    assert (
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1"
        in launcher
    )
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_K=0" in launcher
    assert "OAT_ZERO_ALLOW_SPARSE_EVAL=1" in launcher
    assert "e58_global_verified_replay_smoke_attempt3_python_seed9010" in launcher
    assert '"attempt": 3' in launcher
    assert "global_replay_groups_per_step" in auditor
    assert "canonical_replay_gold_support_feedback" in auditor
    assert "direct MaxEnt telemetry present" in auditor


def test_auditor_coalesces_only_objective_identical_terminal_duplicates(
    tmp_path,
):
    module = _load_auditor()
    metrics = tmp_path / "train_metrics.jsonl"
    rows = [
        {
            "trainer/global_step": 256,
            "trainer/step": 256,
            "train/canonical_replay_mass_observations": 218,
            "actor/response_tok_len": 8.75,
        },
        {
            "trainer/global_step": 256,
            "trainer/step": 257,
            "train/canonical_replay_mass_observations": 218,
            "actor/response_tok_len": 8.75,
            "eval/multi_answer/accuracy": 0.1875,
        },
    ]
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    records = module._read_records(metrics)
    assert len(records) == 1
    assert records[0]["eval/multi_answer/accuracy"] == 0.1875

    rows[-1]["train/canonical_replay_mass_observations"] = 219
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicting duplicate step 256"):
        module._read_records(metrics)
