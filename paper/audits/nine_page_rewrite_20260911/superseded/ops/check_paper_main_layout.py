#!/usr/bin/env python3
"""Gate the long paper's nine-page main body against its compiled PDF and aux.

The end-of-prose marker alone cannot account for delayed floats. Require each
main figure's label and its actual PDF caption to precede References, together
with all main sections and the marker placed after the last Conclusion paragraph.
The references and appendix may occupy any number of subsequent pages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAIN_PAGE_LIMIT = 9
MAIN_FIGURES = (
    "fig:story",
    "fig:modebench-examples",
    "fig:verified-support-story",
    "fig:cross-scale-terminal-effects",
    "fig:maxrl-factorial",
    "fig:level2-admission",
)
MAIN_SECTIONS = (
    "sec:introduction",
    "sec:related",
    "sec:collapse",
    "sec:modebench",
    "sec:method",
    "sec:experiments",
    "sec:results",
    "sec:conclusion",
)
MAIN_END = "sec:main-end"
REQUIRED_LABELS = MAIN_SECTIONS + MAIN_FIGURES + (MAIN_END,)


class LayoutError(ValueError):
    """A missing or inconsistent build artifact, or a main-layout violation."""


def _group(text: str, offset: int) -> tuple[str, int]:
    """Read a TeX braced group without treating escaped braces as delimiters."""
    while offset < len(text) and text[offset].isspace():
        offset += 1
    if offset >= len(text) or text[offset] != "{":
        raise LayoutError("expected a braced group in aux label")
    start, depth, index = offset + 1, 1, offset + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            # Escaped braces do not affect nesting; a doubled backslash leaves
            # the following brace unescaped, as TeX does.
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if not depth:
                return text[start:index], index + 1
        index += 1
    raise LayoutError("unterminated braced group in aux label")


def read_required_labels(aux_text: str) -> dict[str, tuple[str, int]]:
    """Return label number/page pairs, rejecting missing or ambiguous records."""
    labels: dict[str, tuple[str, int]] = {}
    for match in re.finditer(r"(?m)^\s*\\newlabel\s*", aux_text):
        name, offset = _group(aux_text, match.end())
        if name not in REQUIRED_LABELS:
            continue
        if name in labels:
            raise LayoutError(f"duplicate required aux label: {name}")
        try:
            payload, _ = _group(aux_text, offset)
            number, offset = _group(payload, 0)
            page_text, _ = _group(payload, offset)
        except LayoutError as error:
            raise LayoutError(f"malformed aux label {name}: {error}") from error
        if not re.fullmatch(r"[1-9]\d*", page_text.strip()):
            raise LayoutError(f"aux label {name} has invalid page {page_text!r}")
        labels[name] = (number.strip(), int(page_text))
    missing = [name for name in REQUIRED_LABELS if name not in labels]
    if missing:
        raise LayoutError("missing required aux labels: " + ", ".join(missing))
    return labels


def split_pdf_pages(pdf_text: str) -> list[str]:
    """Retain physical page boundaries from pdftotext, including blank pages."""
    pages = pdf_text.split("\f")
    if len(pages) > 1 and not pages[-1].strip():
        pages.pop()  # pdftotext terminates its last page with a form feed.
    if not any(page.strip() for page in pages):
        raise LayoutError("PDF text is empty")
    return pages


def _without_line_number(line: str) -> str:
    return re.sub(r"^\s*\d+\s+", "", line).strip()


def _is_page_furniture(line: str) -> bool:
    line = _without_line_number(line)
    return not line or line.isdigit() or bool(re.fullmatch(
        r"(?:Under review|Published) as a conference paper at ICLR\s+\d{4}", line,
        flags=re.IGNORECASE,
    ))


def reference_start(pages: list[str]) -> int:
    """Find the first standalone References heading, allowing spaced smallcaps."""
    for page_number, page in enumerate(pages, 1):
        lines = page.splitlines()
        for line_number, line in enumerate(lines):
            compact = re.sub(r"\s+", "", _without_line_number(line)).casefold()
            if compact != "references":
                continue
            preceding = [line.strip() for line in lines[:line_number]
                         if not _is_page_furniture(line)]
            if preceding:
                raise LayoutError(
                    f"References on page {page_number} does not begin its own page; "
                    f"content precedes the heading: {preceding[0]!r}"
                )
            return page_number
    raise LayoutError("no standalone References heading found in PDF text")


def figure_caption_pages(pages: list[str]) -> dict[int, list[int]]:
    """Locate caption starts, excluding prose citations such as 'Figure 6 shows'."""
    locations: dict[int, list[int]] = {number: [] for number in range(1, 7)}
    for page_number, page in enumerate(pages, 1):
        for line in page.splitlines():
            match = re.match(r"Figure\s+(\d+)\s*[:.]", _without_line_number(line),
                             flags=re.IGNORECASE)
            if match and int(match.group(1)) in locations:
                locations[int(match.group(1))].append(page_number)
    return locations


def validate_layout(aux_text: str, pdf_text: str) -> dict[str, object]:
    """Validate compiled artifacts; return a concise, JSON-compatible audit."""
    labels = read_required_labels(aux_text)
    pages = split_pdf_pages(pdf_text)
    refs = reference_start(pages)
    errors = []
    if refs > MAIN_PAGE_LIMIT + 1:
        errors.append(f"References begins on page {refs}, later than page {MAIN_PAGE_LIMIT + 1}")
    for name, (_, page) in labels.items():
        if page > MAIN_PAGE_LIMIT:
            errors.append(f"{name} is on page {page}, exceeding the {MAIN_PAGE_LIMIT}-page main limit")
        if page >= refs:
            errors.append(f"{name} is on page {page}, at or after References starts on page {refs}")
        if page > len(pages):
            errors.append(f"{name} is on page {page}, beyond the {len(pages)} PDF pages")

    section_pages = [labels[name][1] for name in MAIN_SECTIONS]
    if section_pages != sorted(section_pages):
        errors.append("main-section label pages are out of manuscript order")
    if max(section_pages) > labels[MAIN_END][1]:
        errors.append(f"{MAIN_END} precedes a main section instead of ending the Conclusion")

    captions = figure_caption_pages(pages)
    for number, name in enumerate(MAIN_FIGURES, 1):
        aux_number, aux_page = labels[name]
        if aux_number != str(number):
            errors.append(f"{name} is numbered {aux_number!r}; expected Figure {number}")
        actual = captions[number]
        if len(actual) != 1:
            errors.append(f"Figure {number} ({name}) needs one PDF caption; found {len(actual)} on pages {actual}")
        elif actual[0] != aux_page:
            errors.append(f"Figure {number} ({name}) PDF caption is on page {actual[0]}, but aux says {aux_page}; rebuild both artifacts")
        for page in actual:
            if page > MAIN_PAGE_LIMIT or page >= refs:
                errors.append(f"Figure {number} PDF caption on page {page} is outside the main body before References page {refs}")
    if errors:
        raise LayoutError("\n".join(errors))
    return {
        "main_last_page": max(page for _, page in labels.values()),
        "references_start_page": refs,
        "pdf_pages": len(pages),
        "label_pages": {name: page for name, (_, page) in labels.items()},
        "figure_pages": {str(number): locations[0] for number, locations in captions.items()},
    }


def extract_pdf_text(pdf: Path) -> str:
    try:
        return subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"], check=True,
            capture_output=True, text=True,
        ).stdout
    except FileNotFoundError as error:
        raise LayoutError("pdftotext is required to inspect the compiled PDF") from error
    except subprocess.CalledProcessError as error:
        raise LayoutError(f"pdftotext failed for {pdf}: {error.stderr.strip()}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "paper/main.pdf")
    parser.add_argument("--aux", type=Path, default=ROOT / "paper/main.aux")
    args = parser.parse_args(argv)
    try:
        if not args.pdf.is_file():
            raise LayoutError(f"compiled PDF does not exist: {args.pdf}")
        aux_text = args.aux.read_text()
        report = validate_layout(aux_text, extract_pdf_text(args.pdf))
    except (LayoutError, OSError) as error:
        print(f"Main-paper layout check failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Main-paper layout passed: all six figures, eight main sections, and the "
        f"Conclusion end fit within {report['main_last_page']} of {MAIN_PAGE_LIMIT} pages; "
        f"References starts on page {report['references_start_page']} "
        f"({report['pdf_pages']} total PDF pages)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
