import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "ops"
    / "exp_scaling"
    / "launch_e58_global_replay_sentinel.sh"
)
AUDITOR = ROOT / "ops" / "exp_scaling" / "audit_e58_sentinel.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location("e58_sentinel_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sentinel_launcher_is_smoke_gated_and_three_domain_atomic():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "E58 sentinel requires a clean terminal Python smoke" in source
    assert "verified_first_global_replay_canonical" in source
    assert "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1" in source
    assert "submit_domain graph_coloring" in source
    assert "submit_domain countdown" in source
    assert "submit_domain python_factor" in source
    assert "OAT_ZERO_SBATCH_HOLD=1" in source
    assert 'scontrol release "${job_ids[@]}"' in source
    assert "OAT_ZERO_MAX_PROMPT_EPOCHS=50" in source
    assert "OAT_ZERO_EVAL_MODE_COVERAGE_K=8" in source


def test_global_scheduler_gate_requires_every_post_discovery_update():
    module = _load_auditor()
    records = []
    for step in range(1, 5):
        discovered = step >= 2
        records.append(
            {
                "trainer/global_step": step,
                "train/online_canonical_tracked_outcomes": int(discovered),
                "train/canonical_replay_global_scheduler_active": 1,
                "train/canonical_replay_global_groups_per_step": 1,
                "train/canonical_replay_available_groups": int(discovered),
                "train/canonical_replay_actuator_groups": int(discovered),
                "train/canonical_replay_mass_observations": (
                    step - 1 if discovered else 0
                ),
            }
        )

    gate = module._scheduler_gate(records, complete=False)
    assert gate["status"] == "running"
    assert gate["first_discovery_step"] == 2
    assert gate["post_discovery_records"] == 3
    assert gate["replay_activations"] == 3
    assert gate["violations"] == []

    records[2]["train/canonical_replay_available_groups"] = 0
    gate = module._scheduler_gate(records, complete=False)
    assert gate["status"] == "fail"
    assert any("expected one" in item for item in gate["violations"])
