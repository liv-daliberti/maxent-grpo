from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

from oat_drgrpo.mathir import validate_mathir_algebra


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "paper/preregistration/e45_mathir_online_growing_support_05b.md"
LAUNCHER = (
    ROOT / "ops/exp_scaling/launch_e45_mathir_online_growing_support_05b.sh"
)
MATERIALIZER = ROOT / "ops/make_mathir_algebra_data.py"
DATA_ROOT = ROOT / "var/data/mathir_algebra_v0_probe"
PROBE = ROOT / "var/artifacts/e45_mathir_bootstrap_probe_v1.json"


def test_e45_protocol_freezes_restricted_executable_math_not_math_text():
    text = PROTOCOL.read_text(encoding="utf-8")

    for literal in (
        "**Status: FROZEN BEFORE LAUNCH (2026-07-23).**",
        "restricted synthetic linear-algebra benchmark",
        "`sub(b);div(a)`",
        "applied by the interpreter to both sides",
        "same successful interpreter execution",
        "bank still starts",
        "0 valid non-seed strategies",
        "does not use Haarnoja/SAC temperature adaptation",
        "sampled distinct non-seed valid strategies at 16",
        "unknown denominator",
    ):
        assert literal in text


def test_e45_launcher_is_six_job_one_a100_matched_cohort():
    text = LAUNCHER.read_text(encoding="utf-8")

    for literal in (
        "EXPECTED_JOBS=6",
        "PREFIX=mie45_ogs_mathir_05b_v2",
        "export OAT_ZERO_COMPARATIVE_TASK=math",
        "export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_maxent",
        "export OAT_ZERO_INCLUDE_ONLINE_CANONICAL_MAXENT_ARM=1",
        "export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10",
        "export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50",
        "export OAT_ZERO_TEMPERATURE=0.5",
        "export OAT_ZERO_TOP_P=0.9",
        "export OAT_ZERO_EVAL_MODE_COVERAGE_K=16",
        "export OAT_ZERO_TRAIN_NODELIST=node302",
        "export OAT_ZERO_TRAIN_GRES=gpu:a100:1",
        "scontrol release \"${job_ids[@]}\"",
    ):
        assert literal in text


def test_e45_materialized_data_is_open_support_disjoint_and_executable():
    identity = json.loads((DATA_ROOT / "identity.json").read_text(encoding="utf-8"))
    train = load_from_disk(str(DATA_ROOT / "train"))["train"]
    evaluation = load_from_disk(str(DATA_ROOT / "eval"))["multi_answer"]

    assert identity["train_rows"] == 384
    assert identity["eval_rows"] == 128
    assert identity["support"] == "open_growing"
    assert identity["certified_strategy_sets_are_exhaustive"] is False
    assert set(train["answer_mode_count"]) == {0}
    assert set(evaluation["answer_mode_count"]) == {0}

    def identities(dataset):
        result = set()
        for row in dataset:
            spec = json.loads(row["answer"])
            result.add((spec["family"], tuple(sorted(spec["bindings"].items()))))
        return result

    assert not (identities(train) & identities(evaluation))
    for row in list(train)[:12] + list(evaluation)[:12]:
        spec = json.loads(row["answer"])
        starter = spec["public_seed_program"]
        validation = validate_mathir_algebra(starter, spec)
        assert validation is not None
        assert validation.canonical_key == spec["public_seed_key"]
        assert f"\\boxed{{{starter}}}" in row["problem"]
        assert spec["support_is_open"] is True


def test_e45_materializer_and_bootstrap_gate_do_not_claim_exhaustive_support():
    materializer = MATERIALIZER.read_text(encoding="utf-8")
    probe = json.loads(PROBE.read_text(encoding="utf-8"))

    assert '"answer_mode_count": 0' in materializer
    assert '"certified_strategy_sets_are_exhaustive": False' in materializer
    assert probe["passed"] is True
    assert probe["total_attempts"] == 128
    assert probe["total_correct"] == 40
    assert probe["groups_any_correct_fraction"] == 1.0
    assert sum(
        family["nonseed_correct"]
        for family in probe["family_summary"].values()
    ) == 0
