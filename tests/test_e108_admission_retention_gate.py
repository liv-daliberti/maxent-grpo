import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def _load(name: str, path: Path):
    sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_e108_freezes_paired_measurement_and_adaptive_arms():
    launch = _load(
        "e108_launch_under_test",
        EXP / "launch_e108_admission_retention_mechanism_gate.py",
    )
    assert launch.ARMS == ("retention_tracking", "adaptive_retention")
    assert len(launch.DOMAINS) == 5
    assert launch.SEED == 43
    assert launch.TARGET_STEPS == 64

    passive = launch.fixed_objective("retention_tracking")
    adaptive = launch.fixed_objective("adaptive_retention")
    differing = {key for key in passive if passive.get(key) != adaptive.get(key)}
    assert differing == {
        "OAT_ZERO_VARIANT",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY",
    }
    for objective in (passive, adaptive):
        assert objective["OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING"] == "1"
        assert (
            objective["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK"]
            == "0"
        )
        assert objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
        assert objective["OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA"] == "0.0"
        assert (
            objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"]
            == "split_mass_balance_per_rollout"
        )


def test_e108_protocol_and_campaign_registry_are_present():
    protocol = ROOT / (
        "paper/preregistration/" "e108_admission_retention_mechanism_gate_20260817.md"
    )
    text = protocol.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "Admission is therefore not treated as retention evidence" in normalized
    assert "outcome gates" in normalized
    assert "sequence log probability" in normalized

    cohorts = _load("e108_cohorts_under_test", EXP / "cohorts.py")
    cohort = cohorts.by_tag("e108")
    assert cohort.ledger == "e108_admission_retention_mechanism_gate_jobs.json"
    assert cohort.resolved_reader() == "smoke"


def test_e108_uses_all_partition_wrapper_and_common_retention_telemetry_path():
    launch = _load(
        "e108_launch_partition_under_test",
        EXP / "launch_e108_admission_retention_mechanism_gate.py",
    )
    assert launch.BATCH_SCRIPT == "ops/slurm/train_all_partition.slurm"
    command = launch.sbatch_command(
        ROOT,
        {"domain": launch.DOMAINS[0], "seed": launch.SEED},
        {},
        launch.ARMS[0],
    )
    assert command[-1] == str(ROOT / launch.BATCH_SCRIPT)

    learner_source = (ROOT / "src/oat_drgrpo/learner/grpo.py").read_text(
        encoding="utf-8"
    )
    marker = "online_canonical_proposal_retention_"
    assert marker in learner_source
    tree = ast.parse(learner_source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_source = ast.get_source_segment(learner_source, node.test) or ""
        if 'key_mode == "verified_route"' not in test_source:
            continue
        body_source = "\n".join(
            ast.get_source_segment(learner_source, child) or "" for child in node.body
        )
        assert marker not in body_source


def test_e108_auditor_checks_actuation_without_outcome_gate():
    source = (EXP / "audit_e108_admission_retention_mechanism_gate.py").read_text(
        encoding="utf-8"
    )
    assert '"outcomes_used_for_gate": False' in source
    assert "priority_visits_added_cumulative" in source
    assert "retention_forbidden_feedback_max" in source
    assert "proposal_rows_to_ppo" in (
        EXP / "audit_e102_full_open_bank_maxent_replay.py"
    ).read_text(encoding="utf-8")
