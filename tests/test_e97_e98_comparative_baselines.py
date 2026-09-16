from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"


def load(name: str):
    sys.path.insert(0, str(EXP))
    path = EXP / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_e97_is_a_three_domain_seed_matched_ucpo_cohort():
    e97 = load("launch_e97_ucpo_05b")
    assert e97.DOMAINS == ("graph_coloring", "python_factors", "pantry_plan")
    assert e97.SEEDS == (43, 44, 45, 46, 47)
    assert len(e97.selected_references(ROOT)) == 15
    objective = e97.objective()
    assert objective["OAT_ZERO_VARIANT"] == "ucpo"
    assert objective["OAT_ZERO_UCPO_TAU"] == "0.2"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "1"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    assert objective["OAT_ZERO_RLEP_REPLAY_COUNT"] == "0"
    assert "src/oat_drgrpo/ucpo.py" in e97.PATCHED_FILES
    assert "src/oat_drgrpo/rlep.py" in e97.PATCHED_FILES


def test_e98_collection_uses_terminal_e78_control_and_training_prompt_view():
    e98 = load("launch_e98_rlep_05b")
    runs = e98.selected_references(ROOT)
    pairs = e98.selected_pairs(ROOT)
    run = next(
        row for row in runs
        if row["domain"] == "graph_coloring" and row["seed"] == 43
    )
    control = pairs[("graph_coloring", 43)]["control"]
    seed_export = e98.terminal_control_export(control)
    assert seed_export.name == "step_03073"
    collection_data = ROOT / "var/data/e98_rlep_collection_prompts/graph_coloring"
    env = e98.collection_env(
        ROOT,
        run,
        snapshot=ROOT / "frozen",
        seed_export=seed_export,
        collection_data=collection_data,
        target=ROOT / "pool",
    )
    assert env["OAT_ZERO_PRETRAIN"] == str(seed_export)
    assert env["OAT_ZERO_PROMPT_DATA"].endswith("graph_coloring_modebench_v2/train")
    assert env["OAT_ZERO_EVAL_DATA"] == str(collection_data)
    assert env["OAT_ZERO_TEST_SPLIT"] == "multi_answer"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_K"] == "16"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS"] == "4"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE"] == "0.7"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P"] == "0.95"


def test_e98_training_is_frequency_preserving_rlep_dr_not_canonical_replay():
    e98 = load("launch_e98_rlep_05b")
    pool = ROOT / "pool"
    objective = e98.training_objective(pool)
    assert objective["OAT_ZERO_VARIANT"] == "rlep"
    assert objective["OAT_ZERO_RLEP_EXPERIENCE_ROOT"] == str(pool)
    assert objective["OAT_ZERO_RLEP_REPLAY_COUNT"] == "2"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY"] == "0"
    assert objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert objective["OAT_ZERO_VERIFIED_DISCOVERY_TRACKING"] == "0"
    assert "ops/exp_scaling/audit_e98_rlep_pool.py" in e98.PATCHED_FILES


def test_preregistrations_freeze_smoke_and_pool_failure_rules():
    ucpo = (ROOT / "paper/preregistration/e97_ucpo_05b_20260812.md").read_text()
    rlep = (ROOT / "paper/preregistration/e98_rlep_dr_05b_20260812.md").read_text()
    rlep_words = " ".join(rlep.split())
    assert "tau=0.2" in ucpo
    assert "32-query" in ucpo
    assert "require at least two per prompt" in rlep
    assert "16 fresh rollouts" in rlep and "2 replayed successes" in rlep
    assert "Evaluation prompts are never" in rlep_words
    assert "RLEP-Dr" in rlep
