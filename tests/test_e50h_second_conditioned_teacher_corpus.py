from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50h_second_conditioned_teacher_corpus.py"
)
SPEC = importlib.util.spec_from_file_location("e50h_second", SOURCE)
assert SPEC is not None and SPEC.loader is not None
E50H = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E50H)


def test_second_attempt_is_answer_blind_concise_and_differently_seeded():
    payload = E50H._proposal_payload(
        "judge", "problem only", 4, ("old left", "old right")
    )
    rendered = payload["messages"][1]["content"]
    assert "problem only" in rendered
    assert "old left VERSUS old right" in rendered
    assert "reference answer" not in rendered
    assert payload["seed"] == E50H.PROPOSAL_SEED + 4
    assert E50H.PROPOSAL_SEED != E50H.BASE.PROPOSAL_SEED
    execution = E50H._execution_payload(
        model="judge",
        problem="problem only",
        method={"label": "method", "actions": ["a", "b"]},
        problem_order=4,
        route_index=0,
    )
    assert "at most 450 words" in execution["messages"][1]["content"]
    assert execution["n"] == 8
    assert execution["max_tokens"] == 1024


def test_second_attempt_is_corpus_only_and_checkpointed():
    source = SOURCE.read_text(encoding="utf-8")
    protocol = E50H.PROTOCOL.read_text(encoding="utf-8")
    assert '"pass": False' in source
    assert "corpus_only_second_attempt_no_training_authority" in source
    assert (
        "open_relation_judge_quarantined_after_frozen_"
        in source
    )
    assert "second_conditioned_records.partial.jsonl" in source
    assert "partial checkpoint identity drifted" in source
    assert '"repeated_excluded_label_pair"' in source
    assert "repeated the excluded first pair" not in source
    assert "E50F 12/50" in protocol
    assert "before E50H data" in protocol
