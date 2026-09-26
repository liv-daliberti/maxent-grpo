"""Check the main page budget with delayed floats and real TeX/PDF text shapes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "ops/check_paper_main_layout.py"
spec = importlib.util.spec_from_file_location("paper_main_layout_test", SCRIPT)
assert spec is not None and spec.loader is not None
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


@pytest.fixture
def manuscript():
    """A nine-page main, two-page bibliography, and arbitrarily long appendix."""
    label_pages = dict(zip(layout.MAIN_SECTIONS, (1, 2, 3, 3, 4, 5, 6, 9)))
    label_pages.update(zip(layout.MAIN_FIGURES, (1, 3, 4, 7, 8, 9)))
    label_pages[layout.MAIN_END] = 9

    def build(*, overrides=None, references=10, total=80, caption_overrides=None,
              omit=(), reference_heading="540   R EFERENCES"):
        current = label_pages | (overrides or {})
        aux = [r"\relax", r"\newlabel{app:unrelated}{{Z}{999}{Appendix}{appendix.90}{}}"]
        for name, page in current.items():
            if name in omit:
                continue
            number = str(layout.MAIN_FIGURES.index(name) + 1) if name in layout.MAIN_FIGURES else "8"
            # Real captions contain nested groups, escaped braces, and refs.
            title = r"\textbf{Nested {caption}} with \{escaped\} braces and \ref{app:data}"
            aux.append(r"\newlabel{" + name + "}{{" + number + "}{" + str(page) + "}{" + title + "}{figure.2}{}}")
        pages = ["Under review as a conference paper at ICLR 2027\n\nMain paragraph.\n"
                 for _ in range(total)]
        for page in range(references, total + 1):
            pages[page - 1] = "Under review as a conference paper at ICLR 2027\n\n"
            pages[page - 1] += (reference_heading + "\nAuthor. A cited paper.\n"
                                if page == references else "Appendix content or bibliography continuation.\n")
        for number, name in enumerate(layout.MAIN_FIGURES, 1):
            page = (caption_overrides or {}).get(number, current[name])
            if page is not None:
                pages[page - 1] += f"400   Figure {number}: Main figure caption with verified results.\n"
        return "\n".join(aux), "\f".join(pages) + "\f"

    return build


def test_main_budget_excludes_arbitrarily_long_references_and_appendix(manuscript):
    aux, pdf = manuscript(total=200)
    report = layout.validate_layout(aux, pdf)
    assert report["main_last_page"] == 9
    assert report["references_start_page"] == 10
    assert report["pdf_pages"] == 200
    assert report["figure_pages"]["6"] == 9


def test_fewer_than_nine_main_pages_is_valid(manuscript):
    aux, pdf = manuscript(overrides={
        "sec:conclusion": 7, layout.MAIN_END: 7,
        "fig:maxrl-factorial": 7, "fig:level2-admission": 7,
    }, references=8, total=15)
    report = layout.validate_layout(aux, pdf)
    assert report["main_last_page"] == 7
    assert report["references_start_page"] == 8


def test_figure_six_flushed_to_page_ten_after_conclusion_fails(manuscript):
    aux, pdf = manuscript(overrides={"fig:level2-admission": 10}, references=11)
    with pytest.raises(layout.LayoutError) as error:
        layout.validate_layout(aux, pdf)
    message = str(error.value)
    assert "fig:level2-admission is on page 10" in message
    assert "Figure 6 PDF caption on page 10" in message
    assert "References begins on page 11" in message


def test_conclusion_end_on_tenth_page_fails_even_when_heading_is_on_ninth(manuscript):
    aux, pdf = manuscript(overrides={layout.MAIN_END: 10}, references=11)
    with pytest.raises(layout.LayoutError, match="sec:main-end is on page 10"):
        layout.validate_layout(aux, pdf)


@pytest.mark.parametrize("missing", ["fig:maxrl-factorial", "sec:results", "sec:main-end"])
def test_missing_main_labels_fail(manuscript, missing):
    aux, pdf = manuscript(omit=(missing,))
    with pytest.raises(layout.LayoutError, match="missing required aux labels: " + missing):
        layout.validate_layout(aux, pdf)


def test_required_label_on_reference_page_fails_even_within_nine_pages(manuscript):
    aux, pdf = manuscript(references=9)
    with pytest.raises(layout.LayoutError, match="at or after References starts on page 9"):
        layout.validate_layout(aux, pdf)


def test_pdf_caption_must_match_aux_page(manuscript):
    aux, pdf = manuscript(caption_overrides={6: 10})
    with pytest.raises(layout.LayoutError) as error:
        layout.validate_layout(aux, pdf)
    assert "PDF caption is on page 10, but aux says 9" in str(error.value)
    assert "Figure 6 PDF caption on page 10 is outside the main body" in str(error.value)


def test_missing_or_duplicate_pdf_caption_fails(manuscript):
    aux, pdf = manuscript(caption_overrides={6: None})
    with pytest.raises(layout.LayoutError, match="Figure 6 .* needs one PDF caption; found 0"):
        layout.validate_layout(aux, pdf)
    aux, pdf = manuscript()
    pdf += "Appendix\nFigure 6: Accidentally repeated main figure.\f"
    with pytest.raises(layout.LayoutError, match="Figure 6 .* needs one PDF caption; found 2"):
        layout.validate_layout(aux, pdf)


def test_reference_heading_must_start_a_new_page(manuscript):
    aux, pdf = manuscript(reference_heading="Conclusion continues here.\n540   R EFERENCES")
    with pytest.raises(layout.LayoutError, match="does not begin its own page"):
        layout.validate_layout(aux, pdf)


def test_reference_heading_ignores_prose_mentions_and_later_headings(manuscript):
    aux, pdf = manuscript()
    pdf = pdf.replace("Main paragraph.", "References are discussed in the prose.\nFigure 6 shows the level comparison.", 1)
    pdf += "REFERENCES\nSupplementary reading list.\f"
    report = layout.validate_layout(aux, pdf)
    assert report["references_start_page"] == 10
    assert report["figure_pages"]["6"] == 9


def test_reference_heading_is_required(manuscript):
    aux, pdf = manuscript(reference_heading="Bibliographic entries without a heading.")
    with pytest.raises(layout.LayoutError, match="no standalone References heading"):
        layout.validate_layout(aux, pdf)


def test_aux_nested_captions_and_escaped_braces_are_read(manuscript):
    aux, _ = manuscript()
    labels = layout.read_required_labels(aux)
    assert labels["fig:level2-admission"] == ("6", 9)
    assert len(labels) == 15


def test_duplicate_aux_label_fails(manuscript):
    aux, pdf = manuscript()
    aux += "\n" + r"\newlabel{fig:level2-admission}{{6}{8}{Duplicate}{figure.99}{}}"
    with pytest.raises(layout.LayoutError, match="duplicate required aux label: fig:level2-admission"):
        layout.validate_layout(aux, pdf)


@pytest.mark.parametrize("bad_page", ["0", "iv", "-1", "??"])
def test_invalid_aux_page_fails(manuscript, bad_page):
    aux, pdf = manuscript()
    aux = aux.replace(r"\newlabel{fig:level2-admission}{{6}{9}",
                      r"\newlabel{fig:level2-admission}{{6}{" + bad_page + "}")
    with pytest.raises(layout.LayoutError, match="has invalid page"):
        layout.validate_layout(aux, pdf)


def test_section_order_and_main_end_marker_are_checked(manuscript):
    aux, pdf = manuscript(overrides={"sec:results": 3, layout.MAIN_END: 8})
    with pytest.raises(layout.LayoutError) as error:
        layout.validate_layout(aux, pdf)
    assert "out of manuscript order" in str(error.value)
    assert "sec:main-end precedes a main section" in str(error.value)


def test_form_feeds_preserve_blank_physical_pages():
    assert layout.split_pdf_pages("Main\f\fReferences\f") == ["Main", "", "References"]
    with pytest.raises(layout.LayoutError, match="PDF text is empty"):
        layout.split_pdf_pages("\f")


def test_cli_uses_supplied_build_artifacts_and_extracts_layout(manuscript, tmp_path, monkeypatch, capsys):
    aux_text, pdf_text = manuscript()
    aux = tmp_path / "build with spaces.aux"
    pdf = tmp_path / "build with spaces.pdf"
    aux.write_text(aux_text)
    pdf.write_bytes(b"%PDF-test-fixture")
    calls = []

    def extract(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout=pdf_text, stderr="")

    monkeypatch.setattr(layout.subprocess, "run", extract)
    assert layout.main(["--pdf", str(pdf), "--aux", str(aux)]) == 0
    assert calls[0][0] == ["pdftotext", "-layout", str(pdf), "-"]
    assert "all six figures" in capsys.readouterr().out
    aux.write_text(aux_text.replace("\\newlabel{sec:main-end}", "\\newlabel{sec:old-main-end}"))
    assert layout.main(["--pdf", str(pdf), "--aux", str(aux)]) == 1
    assert "missing required aux labels: sec:main-end" in capsys.readouterr().err
