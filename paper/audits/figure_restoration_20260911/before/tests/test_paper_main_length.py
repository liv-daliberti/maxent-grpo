"""Enforce main-body length independently of references and appendix length."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "ops/check_paper_main_length.py"
spec = importlib.util.spec_from_file_location("paper_main_length_test", SCRIPT)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def manuscript(*, main_end=9, conclusion=None, references=None, total_pages=40,
               label_overrides=None, caption_overrides=None, omit=()):
    """Synthetic aux/PDF pair with realistic review-mode caption and heading text."""
    conclusion = main_end if conclusion is None else conclusion
    references = main_end + 1 if references is None else references
    figure_pages = [2, 3, 5, min(7, main_end), main_end]
    labels = {
        name: (str(number), page)
        for number, (name, page) in enumerate(zip(checker.MAIN_FIGURE_LABELS, figure_pages), 1)
    }
    # Related Work can move after Results, and multiple logical sections can
    # share a page. The budget must not impose an obsolete narrative order.
    section_pages = [1, 2, 2, 2, 4, 5, 5, min(7, main_end), main_end - 1, conclusion]
    labels.update({name: (str(number), page) for number, (name, page) in
                   enumerate(zip(checker.MAIN_SECTION_LABELS, section_pages), 1)})
    labels[checker.MAIN_END_LABEL] = ("8", main_end)
    labels[checker.REFERENCES_LABEL] = ("8", references)
    labels.update(label_overrides or {})
    aux = "\\relax\n" + "\n".join(
        "\\newlabel{" + name + "}{{" + number + "}{" + str(page)
        + r"}{\textbf {A nested {caption} with \{literal\} braces}}{figure.2}{}}"
        for name, (number, page) in labels.items() if name not in omit
    )
    # Supporting labels do not consume the main budget, including labels whose
    # page syntax is intentionally not relevant to this Arabic main-page check.
    aux += "\n" + r"\newlabel{app:long-proof}{{A}{35}{Proof appendix}{appendix.1}{}}"
    aux += "\n" + r"\newlabel{unrelated@cref}{{[figure][20][]20}{[page][1][]35}}"
    pages = [f"Under review as a conference paper\n\nPage {n} body.\n" for n in range(1, total_pages + 1)]
    for number, name in enumerate(checker.MAIN_FIGURE_LABELS, 1):
        page = (caption_overrides or {}).get(number, labels[name][1])
        pages[page - 1] += f"{40 + number:03d}   Figure {number}: A scientific result.\n"
    pages[conclusion - 1] += "455   8     C ONCLUSION\nFinal scientific paragraph.\n"
    pages[references - 1] += "558   R EFERENCES\nA bibliography entry.\n"
    # Merely mentioning references or another figure in prose is not a heading
    # or caption, and must not shift the detected bibliography boundary.
    pages[0] += "043   References to Figure 6 are useful for interpreting the result.\n"
    pages[-1] += "References\nAppendix-only bibliography.\nFigure 20: A supporting result.\n"
    return aux, "\f".join(pages) + "\f"


def test_nine_pages_include_all_five_figures_while_total_pages_are_ignored():
    result = checker.validate_main_length(*manuscript(total_pages=80))
    assert result["main_pages"] == 9
    assert result["references_page"] == 10
    assert result["pdf_pages"] == 80
    assert result["figure_pages"]["fig:replay-level2-effects"] == 9
    assert result["figure_pages"]["fig:hosted-verified-breadth"] == 3
    assert len(result["figure_pages"]) == 5


def test_shorter_main_passes_without_forcing_exactly_nine_pages():
    result = checker.validate_main_length(*manuscript(main_end=8, total_pages=60))
    assert result["main_pages"] == 8
    assert result["references_page"] == 9


def test_last_figure_spill_after_conclusion_fails_even_when_conclusion_starts_nine():
    aux, text = manuscript(label_overrides={"fig:replay-level2-effects": ("5", 10)}, references=11)
    with pytest.raises(checker.MainLengthError, match="fig:replay-level2-effects is on page 10, exceeding"):
        checker.validate_main_length(aux, text)



def test_hosted_figure_spill_beyond_nine_pages_is_rejected():
    aux, text = manuscript(label_overrides={"fig:hosted-verified-breadth": ("2", 10)},
                           references=11)
    with pytest.raises(checker.MainLengthError,
                       match="fig:hosted-verified-breadth is on page 10, exceeding"):
        checker.validate_main_length(aux, text)


def test_hosted_figure_after_main_end_is_rejected_within_the_page_cap():
    aux, text = manuscript(main_end=8, references=10,
                           label_overrides={"fig:hosted-verified-breadth": ("2", 9)})
    with pytest.raises(checker.MainLengthError,
                       match="fig:hosted-verified-breadth is on page 9, after sec:main-end"):
        checker.validate_main_length(aux, text)


def test_hosted_caption_detects_stale_aux_that_hides_a_float_spill():
    with pytest.raises(checker.MainLengthError,
                       match=r"Figure 2 .* caption pages \[10\].*aux page 3") as error:
        checker.validate_main_length(*manuscript(caption_overrides={2: 10}))
    assert "Figure 2 caption on page 10 is outside the scientific main" in str(error.value)


def test_hosted_figure_cannot_renumber_the_existing_main_figures():
    aux, text = manuscript(label_overrides={"fig:replay-level2-effects": ("2", 9),
                                           "fig:hosted-verified-breadth": ("5", 3)})
    with pytest.raises(checker.MainLengthError, match="must identify Figure 5") as error:
        checker.validate_main_length(aux, text)
    assert "fig:hosted-verified-breadth must identify Figure 2" in str(error.value)


def test_missing_hosted_caption_is_rejected_even_with_a_valid_aux_label():
    aux, text = manuscript()
    text = text.replace("Figure 2: A scientific result.", "A result without its caption.")
    with pytest.raises(checker.MainLengthError, match=r"Figure 2 .* caption pages \[\]"):
        checker.validate_main_length(aux, text)


def test_duplicate_hosted_caption_in_appendix_is_rejected():
    aux, text = manuscript()
    text = text.replace("Figure 20: A supporting result.", "Figure 2: A duplicate main caption.")
    with pytest.raises(checker.MainLengthError,
                       match=r"Figure 2 .* caption pages \[3, 40\].*aux page 3"):
        checker.validate_main_length(aux, text)


def test_conclusion_starts_nine_but_last_paragraph_spills_to_ten():
    with pytest.raises(checker.MainLengthError, match="sec:main-end is on page 10, exceeding"):
        checker.validate_main_length(*manuscript(main_end=10, conclusion=9))


def test_main_float_on_reference_page_fails():
    aux, text = manuscript(label_overrides={"fig:replay-level2-effects": ("5", 10)})
    with pytest.raises(checker.MainLengthError, match="at or after References page 10"):
        checker.validate_main_length(aux, text)


def test_float_after_main_end_is_rejected_even_within_the_page_cap():
    aux, text = manuscript(main_end=8, references=10,
                           label_overrides={"fig:replay-level2-effects": ("5", 9)})
    with pytest.raises(checker.MainLengthError, match="after sec:main-end on page 8"):
        checker.validate_main_length(aux, text)


def test_actual_pdf_caption_detects_stale_aux_that_hides_a_float_spill():
    with pytest.raises(checker.MainLengthError, match=r"Figure 5 .* caption pages \[10\].*aux page 9"):
        checker.validate_main_length(*manuscript(caption_overrides={5: 10}))


@pytest.mark.parametrize("missing", ["fig:replay-level2-effects", "fig:hosted-verified-breadth",
                                     "sec:main-end", "sec:references", "sec:method"])
def test_missing_required_label_fails(missing):
    with pytest.raises(checker.MainLengthError, match="missing required aux labels: " + missing):
        checker.validate_main_length(*manuscript(omit=(missing,)))


def test_references_heading_must_match_aux_not_merely_occur_on_a_later_page():
    aux, text = manuscript(label_overrides={"sec:references": ("8", 11)})
    with pytest.raises(checker.MainLengthError, match="first PDF References heading is on page 10"):
        checker.validate_main_length(aux, text)


def test_references_need_a_separate_page_after_the_whole_main():
    with pytest.raises(checker.MainLengthError, match="on a separate page"):
        checker.validate_main_length(*manuscript(references=9))


def test_late_references_rejected_even_if_main_labels_fit():
    with pytest.raises(checker.MainLengthError, match="References starts on page 11"):
        checker.validate_main_length(*manuscript(references=11))


def test_prose_mentioning_references_is_not_a_references_heading():
    aux, text = manuscript()
    text = text.replace("558   R EFERENCES", "558   References are cited throughout.")
    text = text.replace("\nReferences\nAppendix-only", "\nReferences are cited here.\nAppendix-only")
    with pytest.raises(checker.MainLengthError, match="no standalone References heading"):
        checker.validate_main_length(aux, text)


def test_missing_caption_and_wrong_figure_number_are_not_silently_accepted():
    aux, text = manuscript(label_overrides={"fig:replay-level2-effects": ("7", 9)})
    text = text.replace("Figure 5: A scientific result.", "A result without its caption.")
    with pytest.raises(checker.MainLengthError, match="must identify Figure 5") as error:
        checker.validate_main_length(aux, text)
    assert "caption pages []" in str(error.value)


@pytest.mark.parametrize("page", [2, 3])
def test_first_main_figure_can_follow_measurement_on_a_later_page(page):
    result = checker.validate_main_length(*manuscript(
        label_overrides={"fig:concentration-story": ("1", page)}))
    assert result["figure_pages"]["fig:concentration-story"] == page


def test_conclusion_label_is_checked_against_the_rendered_heading():
    aux, text = manuscript(label_overrides={"sec:conclusion": ("8", 8)})
    with pytest.raises(checker.MainLengthError, match="no Conclusion heading on aux conclusion page 8"):
        checker.validate_main_length(aux, text)


@pytest.mark.parametrize("record, message", [
    (r"\newlabel{sec:main-end}{{8}{9}{duplicate}}", "duplicate required aux label"),
    (r"\newlabel{sec:main-end}{{8}{ix}{Roman}}", "invalid Arabic page number"),
    (r"\newlabel{sec:main-end}{{8}{9}", "malformed aux label"),
])
def test_malformed_or_ambiguous_required_aux_records_fail(record, message):
    aux, text = manuscript(omit=() if "duplicate" in message else ("sec:main-end",))
    with pytest.raises(checker.MainLengthError, match=message):
        checker.validate_main_length(aux + "\n" + record, text)


def test_custom_page_cap_is_honored():
    with pytest.raises(checker.MainLengthError, match="exceeding the 8-page main limit"):
        checker.validate_main_length(*manuscript(), max_pages=8)
    assert checker.validate_main_length(*manuscript(main_end=8), max_pages=8)["main_pages"] == 8


def test_blank_physical_pages_are_preserved_and_terminal_delimiter_is_ignored():
    assert checker.split_pdf_pages("First\f\fThird\f") == ["First", "", "Third"]


def test_cli_reads_artifacts_and_invokes_layout_extraction(tmp_path, monkeypatch, capsys):
    aux_text, pdf_text = manuscript(main_end=8)
    aux = tmp_path / "main.aux"
    pdf = tmp_path / "main.pdf"
    aux.write_text(aux_text)
    pdf.write_bytes(b"%PDF fixture; extraction is mocked")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=pdf_text, stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)
    checker.main(["--pdf", str(pdf), "--aux", str(aux), "--max-pages", "8"])
    assert calls[0][0] == ["pdftotext", "-layout", str(pdf), "-"]
    assert "8/8 main pages, all 5 figures included" in capsys.readouterr().out


def test_missing_build_or_failed_extraction_fails_closed(tmp_path, monkeypatch):
    pdf, aux = tmp_path / "main.pdf", tmp_path / "main.aux"
    with pytest.raises(checker.MainLengthError, match="missing built artifact"):
        checker.check_main_length(pdf, aux)
    pdf.write_bytes(b"invalid PDF")
    aux.write_text(manuscript()[0])
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 1, stdout="", stderr="Invalid PDF"))
    with pytest.raises(SystemExit, match="main-length contract failed: Invalid PDF"):
        checker.main(["--pdf", str(pdf), "--aux", str(aux)])


def test_key_weighting_figure_cannot_spill_out_of_the_main_budget():
    aux, text = manuscript(label_overrides={"fig:replay-key-weighting": ("4", 10)}, references=11)
    with pytest.raises(checker.MainLengthError, match="fig:replay-key-weighting is on page 10, exceeding"):
        checker.validate_main_length(aux, text)
