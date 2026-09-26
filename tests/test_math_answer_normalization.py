from oat_drgrpo.math_answer_normalization import (
    NORMALIZATION_VERSION,
    audited_answer_candidates,
    audited_answer_matches,
)


def test_audited_answer_normalization_handles_units_and_unicode_radicals():
    assert NORMALIZATION_VERSION == "audited_math_answer_surface_v1"
    assert "4*sqrt(2)" in audited_answer_candidates("4√2 cm")
    assert audited_answer_matches("4√2 cm", r"4\sqrt{2}")
    assert audited_answer_matches("$40", r"\$40")


def test_audited_answer_normalization_does_not_change_values_or_variables():
    assert not audited_answer_matches("4√3 cm", r"4\sqrt{2}")
    assert not audited_answer_matches("20 dollars", r"\$40")
    assert "6z" in audited_answer_candidates("6z")
    assert audited_answer_matches("6z", "6z")
