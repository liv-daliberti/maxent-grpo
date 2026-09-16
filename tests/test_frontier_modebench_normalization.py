from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
from frontier_modebench_normalization import normalize_and_grade, normalize_response


@pytest.mark.parametrize("raw, expected", [
    (r"\boxed{34+\frac{24}{28-16}}", r"\boxed{34+((24)/(28-16))}"),
    (r"\boxed{\dfrac{2}{\tfrac{3}{4}}}", r"\boxed{((2)/(((3)/(4))))}"),
    (r"\boxed{(7+2)(6-4)}", r"\boxed{(7+2)*(6-4)}"),
    (r"\boxed{2 (3+4)}", r"\boxed{2*(3+4)}"),
    (r"\boxed{(2+3)4}", r"\boxed{(2+3)*4}"),
    (r"\boxed{(6/2)(1+2)}", r"\boxed{(6/2)*(1+2)}"),
    (r"\boxed{\frac{6}{2}(1+2)}", r"\boxed{((6)/(2))*(1+2)}"),
    (r"\boxed{\left(2+3\right)4}", r"\boxed{(2+3)*4}"),
])
def test_countdown_typography_preserves_arithmetic_groups(raw, expected):
    result = normalize_response(2, "countdown", raw)
    assert result["normalized_text"] == expected
    assert result["transformations"]
    assert normalize_response(2, "countdown", expected)["normalized_text"] == expected


@pytest.mark.parametrize("raw", [
    r"\boxed{6/2(1+2)}",  # Ambiguous division/implicit-product precedence.
    r"\boxed{6/(2)(1+2)}",
    r"\boxed{\frac{2}{}}",
    r"\boxed{\frac{2}{3}",
    r"\boxed{2 3 + 4}",  # Never infer missing operators between numbers.
    r"\boxed{f(2)+3}",   # Never turn a function call into multiplication.
])
def test_ambiguous_or_malformed_countdown_is_not_repaired(raw):
    assert normalize_response(3, "countdown", raw)["normalized_text"] == raw


def test_python_and_pantry_escape_rules_are_domain_specific():
    code = r"\boxed{\lambda n: 2 if n \% 2 == 0 else 3}"
    result = normalize_response(2, "python_factors", code)
    assert result["normalized_text"] == r"\boxed{lambda n: 2 if n % 2 == 0 else 3}"
    assert result["transformations"] == ["python_lambda", "escaped_percent"]
    pantry = r"\boxed{pumpkin\_seeds=125;\ kale=50}"
    result = normalize_response(2, "pantry_plan", pantry)
    assert result["normalized_text"] == r"\boxed{pumpkin_seeds=125; kale=50}"
    assert normalize_response(1, "pantry_plan", pantry)["normalized_text"] == pantry
    assert normalize_response(1, "graph_coloring", code)["normalized_text"] == code
    assert normalize_response(3, "mathir", code)["normalized_text"] == code


def test_only_final_box_changes_and_surrounding_text_is_preserved():
    original = r"Earlier \frac{2}{3}. \boxed{4} Final: \boxed{(7+2)(6-4)} End."
    normalized = normalize_response(2, "countdown", original)["normalized_text"]
    assert normalized == r"Earlier \frac{2}{3}. \boxed{4} Final: \boxed{(7+2)*(6-4)} End."


def test_normalization_is_posthoc_and_cannot_rescue_wrong_arithmetic_or_operands():
    row = {"level": 2, "domain": "countdown", "answer": {"verifier": "countdown", "numbers": [7, 2, 6, 4], "target": 18}}
    text = r"\boxed{(7+2)(6-4)}"
    correct = normalize_and_grade(row, text)
    assert correct["verified"] and correct["posthoc"]
    assert correct["original_text"] == text
    assert not normalize_and_grade(row, r"\boxed{(7+2)(6+4)}")["verified"]
    assert not normalize_and_grade(row, r"\boxed{3(6)}")["verified"]


def test_original_success_keeps_its_canonical_key(monkeypatch):
    import frontier_modebench_normalization as module
    monkeypatch.setattr(module, "grade_response", lambda *args: pytest.fail("strict success was regraded"))
    row = {"level": 2, "domain": "countdown", "answer": {}}
    text = r"\boxed{(6/2)(1+2)}"
    strict = {"verified": True, "canonical_key": "original-key", "graded_text": text}
    result = normalize_and_grade(row, text, strict_grade=strict)
    assert result["canonical_key"] == "original-key"
    assert result["normalized_text"] == text
    assert result["transformations"] == []


def test_changed_answer_uses_injected_frozen_grader(monkeypatch):
    import frontier_modebench_normalization as module
    monkeypatch.setattr(module, "grade_response", lambda *args: pytest.fail("mutable default grader was used"))
    row = {"level": 2, "domain": "countdown", "answer": {}}
    strict = {"verified": False, "canonical_key": None, "graded_text": r"\boxed{2(3+4)}"}
    calls = []
    def frozen(level, domain, source, text):
        calls.append((level, domain, source, text))
        return {"verified": True, "canonical_key": "frozen-result", "graded_text": text}
    result = module.normalize_and_grade(row, strict["graded_text"], strict_grade=strict, grader=frozen)
    assert calls == [(2, "countdown", row, r"\boxed{2*(3+4)}")]
    assert result["canonical_key"] == "frozen-result"
    assert result["original_verified"] is False
