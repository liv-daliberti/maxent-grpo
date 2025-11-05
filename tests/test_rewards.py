from src import rewards as R


def test_canon_math_strips_wrappers_and_spaces():
    canon = R._canon_math  # type: ignore[attr-defined]
    assert canon(" { 42 } ") == "42"
    assert canon("( 3.0 )") == "3"
    assert canon("-0") == "0"
    assert canon("12.0") == "12"
    assert canon("  \\sqrt  { 4 }  ") == "\\sqrt{4}".replace(" ", "")


def test_pure_accuracy_reward_math_happy_and_mismatch():
    comp_ok = "<think>...</think><answer> 42 </answer>"
    comp_bad = "<think>...</think><answer> 41 </answer>"
    gold = ["42", "42"]
    out = R.pure_accuracy_reward_math([comp_ok, comp_bad], gold)
    assert out == [1.0, 0.0]


def test_pure_accuracy_reward_requires_format_tags():
    # Missing tags yields 0.0
    out = R.pure_accuracy_reward_math(["no tags here"], ["anything"])
    assert out == [0.0]

