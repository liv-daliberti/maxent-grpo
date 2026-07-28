"""Contracts for the frozen E47 hard-MATH strategy calibration pilot."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "ops/math_strategy_calibration/e47_calibration.py"
PROTOCOL = (
    ROOT / "paper/preregistration/e47_math_strategy_canonicalizer_calibration.md"
)


def load_pipeline():
    spec = importlib.util.spec_from_file_location("e47_calibration", PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_freezes_e46_controller_and_excludes_new_uncertainty_controller():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "**Status: FROZEN BEFORE LAUNCH" in text
    for needle in (
        "`rho_x = H(q_x) / log |B_x^+|`",
        "target `rho*=0.80`",
        "`alpha_0=alpha_min=0.10`, `alpha_max=0.50`",
        "learning rate `0.003`",
        "entropy EMA decay `0.90`",
        "novelty coefficient `0.50`",
        "policy-token-uncertainty controller is excluded",
    ):
        assert needle in text


def test_selection_is_exactly_50_level5_rows_with_frozen_subject_quotas(tmp_path):
    module = load_pipeline()
    module.prepare(tmp_path)
    problems = module._read_jsonl(tmp_path / "problems.jsonl")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert len(problems) == 50
    assert {row["level"] for row in problems} == {"5"}
    assert manifest["selection"]["quotas"] == module.QUOTAS
    assert manifest["selection"]["subject_counts"] == module.QUOTAS
    assert manifest["selection"]["ordered_problems_sha256"] == (
        "a378f9ca69924203002494e93e94a4653c6e9ac0044de031e9ab2d7ad2fecd3f"
    )


def test_all_blinded_injections_are_validator_positive_and_key_is_separate(tmp_path):
    module = load_pipeline()
    module.prepare(tmp_path)
    public = module._read_jsonl(tmp_path / "injections.blinded.jsonl")
    private = module._read_jsonl(tmp_path / "private/injection_key.jsonl")

    assert len(public) == 200
    assert len(private) == 200
    assert all("injection_kind" not in row for row in public)
    assert all(row["validator_reward"] == 1.0 for row in private)
    assert {row["injection_kind"] for row in private} == {
        "anchor",
        "exact_duplicate",
        "format_variant",
        "lexical_paraphrase",
    }
    assert {row["sample_id"] for row in public} == {
        row["sample_id"] for row in private
    }


def test_judge_partition_parser_fails_closed_on_missing_or_duplicate_ids():
    module = load_pipeline()
    valid = json.dumps(
        {
            "clusters": [
                {
                    "cluster_id": "c1",
                    "strategy": "route",
                    "member_ids": ["a", "b"],
                }
            ],
            "ambiguous_ids": ["c"],
        }
    )
    result = module._extract_judge_partition(valid, {"a", "b", "c"})
    assert result["ambiguous_ids"] == ["c"]

    missing = json.dumps(
        {
            "clusters": [{"cluster_id": "c1", "member_ids": ["a"]}],
            "ambiguous_ids": [],
        }
    )
    try:
        module._extract_judge_partition(missing, {"a", "b"})
    except ValueError as exc:
        assert "assignment mismatch" in str(exc)
    else:
        raise AssertionError("missing judge assignment did not fail closed")


def test_e46_replay_uses_exact_normalized_bank_controller():
    module = load_pipeline()
    valid = [
        {
            "problem_id": "p000",
            "sample_id": f"s{i}",
            "sample_index": i,
            "text": "",
        }
        for i in range(32)
    ]
    assignments = {
        "p000": {f"s{i}": ("c1" if i % 2 == 0 else "c2") for i in range(32)}
    }
    replay = module._replay_e46(valid, assignments)

    state = replay["controller_state"]
    assert state["entropy_units"] == (
        "verified_bank_entropy_over_log_support_v1"
    )
    assert state["target_ratio"] == 0.8
    assert state["base_alpha"] == 0.1
    assert state["min_alpha"] == 0.1
    assert state["max_alpha"] == 0.5
    assert state["alpha_lr"] == 0.003
    assert state["ema_decay"] == 0.9
