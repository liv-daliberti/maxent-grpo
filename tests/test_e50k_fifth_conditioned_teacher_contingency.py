from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50k_fifth_conditioned_teacher_contingency.py"
)
SPEC = importlib.util.spec_from_file_location("e50k_fifth", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fifth_stage_is_answer_blind_and_uses_new_fixed_seeds():
    excluded = (
        ("old one", "old two"),
        ("older one", "older two"),
        ("oldest one", "oldest two"),
        ("prior one", "prior two"),
    )
    payload = MODULE._proposal_payload(
        "judge", "problem only", 7, excluded
    )
    rendered = payload["messages"][1]["content"]
    assert "problem only" in rendered
    for left, right in excluded:
        assert f"{left} VERSUS {right}" in rendered
    assert "reference answer" not in rendered
    assert payload["seed"] == MODULE.PROPOSAL_SEED + 7
    assert MODULE.PROPOSAL_SEED not in {
        MODULE.F.PROPOSAL_SEED,
        MODULE.H.PROPOSAL_SEED,
        MODULE.I.PROPOSAL_SEED,
        MODULE.J.PROPOSAL_SEED,
    }


def test_contingency_is_final_bounded_corpus_only_and_preregistered():
    source = SOURCE.read_text(encoding="utf-8")
    protocol = MODULE.PROTOCOL.read_text(encoding="utf-8")
    assert "TRIGGER_THRESHOLD = 30" in source
    assert '"pass": False' in source
    assert "corpus_only_fifth_attempt_no_training_authority" in source
    assert "fifth_conditioned_records.partial.jsonl" in source
    assert "E50I 5/35" in protocol
    assert "at least 30" in protocol
    assert "final proposal stage" in protocol
