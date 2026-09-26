#!/usr/bin/env python3
"""Validate main-paper length, including all figures and the hosted summary table.

The end-of-main marker must follow the final Conclusion paragraph. References
must start on a later page, with ``sec:references`` attached to their heading.
The PDF's References heading and display captions independently verify aux page
numbers; references, disclosures, and appendix pages do not consume the budget.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
MAIN_FIGURE_LABELS = (
    "fig:story",
    "fig:modebench-examples",
    "fig:level-construction",
    "fig:verified-support-story",
    "fig:cross-scale-terminal-effects",
    "fig:maxrl-factorial",
    "fig:level2-admission",
    "fig:gpt56-temperature-curve",
)
MAIN_TABLE_LABELS = ("tab:hosted-level-averages",)
MAIN_SECTION_LABELS = (
    "sec:introduction", "sec:related", "sec:collapse", "sec:modebench",
    "sec:method", "sec:experiments", "sec:results", "sec:conclusion",
)
MAIN_END_LABEL = "sec:main-end"
REFERENCES_LABEL = "sec:references"
REQUIRED_LABELS = (*MAIN_FIGURE_LABELS, *MAIN_TABLE_LABELS, *MAIN_SECTION_LABELS,
                   MAIN_END_LABEL, REFERENCES_LABEL)


class MainLengthError(ValueError):
    """The compiled manuscript violates the scientific-main page contract."""


def _group(text: str, start: int) -> tuple[str, int]:
    """Read one balanced TeX group, including nested captions and escaped braces."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start == len(text) or text[start] != "{":
        raise MainLengthError("expected a braced aux field")
    depth = 1
    pos = start + 1
    while pos < len(text):
        char = text[pos]
        if char == "\\":
            # An escaped brace cannot open/close a TeX group. Skipping the next
            # character also correctly handles a literal escaped backslash.
            pos += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:pos], pos + 1
        pos += 1
    raise MainLengthError("unterminated braced aux field")


def parse_required_labels(aux_text: str) -> dict[str, dict[str, str | int]]:
    """Read required newlabels without being confused by nested caption braces."""
    labels: dict[str, dict[str, str | int]] = {}
    # newlabel records are emitted on individual lines. Anchoring also avoids
    # mistaking a quoted command inside an unrelated caption for a real label.
    pattern = re.compile(r"^\s*\\newlabel\s*\{([^{}]+)\}", re.MULTILINE)
    for match in pattern.finditer(aux_text):
        name = match.group(1)
        if name not in REQUIRED_LABELS:
            continue
        if name in labels:
            raise MainLengthError(f"duplicate required aux label {name}")
        try:
            fields, _ = _group(aux_text, match.end())
            number, pos = _group(fields, 0)
            page, _ = _group(fields, pos)
        except MainLengthError as error:
            raise MainLengthError(f"malformed aux label {name}: {error}") from error
        if not re.fullmatch(r"[1-9]\d*", page.strip()):
            raise MainLengthError(f"{name} has an invalid Arabic page number: {page!r}")
        labels[name] = {"number": number, "page": int(page)}
    missing = [name for name in REQUIRED_LABELS if name not in labels]
    if missing:
        raise MainLengthError("missing required aux labels: " + ", ".join(missing))
    return labels


def split_pdf_pages(pdf_text: str) -> list[str]:
    """Preserve physical blank pages while discarding pdftotext's final delimiter."""
    pages = pdf_text.split("\f")
    if len(pages) > 1 and not pages[-1].strip():
        pages.pop()
    if not pages or not any(page.strip() for page in pages):
        raise MainLengthError("PDF text is empty")
    return pages


def _heading(line: str) -> str:
    # Review-mode line numbers and small-caps letter spacing are not heading
    # content. The remaining whole line must match, rather than prose mentioning
    # references somewhere on a page.
    line = re.sub(r"^(?:\s*\d+\s+)*", "", line)
    return re.sub(r"\s+", "", line).upper()


def validate_main_length(
    aux_text: str, pdf_text: str, max_pages: int = 9,
) -> dict[str, object]:
    """Validate aux records against extracted PDF pages; raise on any violation.

    This pure entry point accepts the full ``pdftotext -layout`` output, with
    form-feed page boundaries. It intentionally does not limit total PDF pages.
    """
    if not isinstance(max_pages, int) or isinstance(max_pages, bool) or max_pages < 1:
        raise MainLengthError("max-pages must be a positive integer")
    labels = parse_required_labels(aux_text)
    pages = split_pdf_pages(pdf_text)
    label_pages = {name: int(record["page"]) for name, record in labels.items()}
    issues: list[str] = []
    for name, page in label_pages.items():
        if page > len(pages):
            issues.append(f"{name} points to page {page}, beyond the {len(pages)}-page PDF")

    reference_pages = [
        number for number, page in enumerate(pages, 1)
        if any(_heading(line) == "REFERENCES" for line in page.splitlines())
    ]
    if not reference_pages:
        raise MainLengthError("no standalone References heading found in the PDF")
    references_page = reference_pages[0]
    main_end_page = label_pages[MAIN_END_LABEL]
    if label_pages[REFERENCES_LABEL] != references_page:
        issues.append(
            f"{REFERENCES_LABEL} says page {label_pages[REFERENCES_LABEL]}, "
            f"but the first PDF References heading is on page {references_page}"
        )
    if references_page > max_pages + 1:
        issues.append(f"References starts on page {references_page}; it must start by page {max_pages + 1}")
    if references_page <= main_end_page:
        issues.append(
            f"References page {references_page} must follow {MAIN_END_LABEL} "
            f"on page {main_end_page} on a separate page"
        )

    main_labels = (*MAIN_SECTION_LABELS, *MAIN_FIGURE_LABELS, *MAIN_TABLE_LABELS, MAIN_END_LABEL)
    for name in main_labels:
        page = label_pages[name]
        if page > max_pages:
            issues.append(f"{name} is on page {page}, exceeding the {max_pages}-page main limit")
        if page > main_end_page:
            issues.append(f"{name} is on page {page}, after {MAIN_END_LABEL} on page {main_end_page}")
        if page >= references_page:
            issues.append(f"{name} is on page {page}, at or after References page {references_page}")

    if label_pages["fig:story"] != 1:
        issues.append(f"fig:story must be on page 1, not page {label_pages['fig:story']}")

    # Caption prefixes are distinguished from ordinary in-text references by
    # their colon. Review-mode line numbers may precede either display kind.
    display_groups = (
        ("Figure", {name: str(number) for number, name in enumerate(MAIN_FIGURE_LABELS, 1)}),
        ("Table", {name: str(labels[name]["number"]) for name in MAIN_TABLE_LABELS}),
    )
    for kind, expected_numbers in display_groups:
        caption_pattern = re.compile(
            r"^\s*(?:\d+\s+)?" + kind + r"\s+(\d+)\s*:", re.MULTILINE | re.IGNORECASE)
        caption_pages: dict[str, list[int]] = {number: [] for number in expected_numbers.values()}
        for page_number, page in enumerate(pages, 1):
            for match in caption_pattern.finditer(page):
                number = match.group(1)
                if number in caption_pages:
                    caption_pages[number].append(page_number)
        for name, number in expected_numbers.items():
            if not re.fullmatch(r"[1-9]\d*", number):
                issues.append(f"{name} has an invalid Arabic {kind.lower()} number: {number!r}")
            if labels[name]["number"] != number:
                issues.append(f"{name} must identify {kind} {number}, not {labels[name]['number']!r}")
            actual = caption_pages[number]
            expected = label_pages[name]
            if actual != [expected]:
                issues.append(
                    f"{kind} {number} ({name}) caption pages {actual} do not match "
                    f"its single aux page {expected}"
                )
            for page in actual:
                if page > min(max_pages, main_end_page) or page >= references_page:
                    issues.append(f"{kind} {number} caption on page {page} is outside the scientific main")

    conclusion_page = label_pages["sec:conclusion"]
    if conclusion_page <= len(pages) and not any(
        _heading(line) == "CONCLUSION" for line in pages[conclusion_page - 1].splitlines()
    ):
        issues.append(f"no Conclusion heading on aux conclusion page {conclusion_page}")
    if issues:
        raise MainLengthError("\n".join(issues))
    return {
        "main_pages": main_end_page,
        "max_pages": max_pages,
        "references_page": references_page,
        "pdf_pages": len(pages),
        "figure_pages": {name: label_pages[name] for name in MAIN_FIGURE_LABELS},
        "table_pages": {name: label_pages[name] for name in MAIN_TABLE_LABELS},
        "section_pages": {name: label_pages[name] for name in MAIN_SECTION_LABELS},
    }


def extract_pdf_text(pdf: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise MainLengthError(result.stderr.strip() or "pdftotext failed")
    return result.stdout


def check_main_length(pdf: Path, aux: Path, max_pages: int = 9) -> dict[str, object]:
    """Read built artifacts and run the same validation exposed to unit tests."""
    for path in (pdf, aux):
        if not path.is_file():
            raise MainLengthError(f"missing built artifact: {path}")
    return validate_main_length(aux.read_text(encoding="utf-8"), extract_pdf_text(pdf), max_pages)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "paper/main.pdf")
    parser.add_argument("--aux", type=Path, default=ROOT / "paper/main.aux")
    parser.add_argument("--max-pages", type=int, default=9)
    args = parser.parse_args(argv)
    try:
        result = check_main_length(args.pdf, args.aux, args.max_pages)
    except (MainLengthError, OSError) as error:
        raise SystemExit(f"main-length contract failed: {error}") from error
    print(
        f"Main-length contract passed: {result['main_pages']}/{args.max_pages} main pages, "
        f"all {len(MAIN_FIGURE_LABELS)} figures and {len(MAIN_TABLE_LABELS)} table included; "
        f"References starts on page {result['references_page']} "
        f"({result['pdf_pages']} total PDF pages)."
    )


if __name__ == "__main__":
    main()
