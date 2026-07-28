from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50i_third_conditioned_teacher_contingency.py"
)
SPEC = importlib.util.spec_from_file_location("e50i_third", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_third_attempt_is_answer_blind_and_uses_new_fixed_seeds():
    excluded = (
        ("old one", "old two"),
        ("older one", "older two"),
    )
    payload = MODULE._proposal_payload(
        "judge", "problem only", 7, excluded
    )
    rendered = payload["messages"][1]["content"]
    assert "problem only" in rendered
    assert "old one VERSUS old two" in rendered
    assert "older one VERSUS older two" in rendered
    assert "reference answer" not in rendered
    assert payload["seed"] == MODULE.PROPOSAL_SEED + 7
    assert MODULE.PROPOSAL_SEED not in {
        MODULE.F.PROPOSAL_SEED,
        MODULE.H.PROPOSAL_SEED,
    }


def test_contingency_is_bounded_corpus_only_and_preregistered():
    source = SOURCE.read_text(encoding="utf-8")
    protocol = MODULE.PROTOCOL.read_text(encoding="utf-8")
    assert "TRIGGER_THRESHOLD = 20" in source
    assert '"pass": False' in source
    assert "corpus_only_third_attempt_no_training_authority" in source
    assert "third_conditioned_records.partial.jsonl" in source
    assert "E50H 4/50" in protocol
    assert "at least 20" in protocol
    assert "only for each still" in protocol
