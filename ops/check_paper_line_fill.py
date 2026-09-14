#!/usr/bin/env python3
"""Fail when a natural-prose paragraph ends on less than half its line.

The audit operates on the rendered PDF, so source reflow, macro expansion, and
bibliography generation cannot evade it. It checks multi-line prose blocks,
including captions and the prose portion of bibliography entries. A terminal
URL or DOI is treated as an indivisible structural identifier; the prose ending
immediately before it remains audited. Display mathematics, algorithms, tables,
headings, and verbatim prompt transcripts are structural rather than
natural-prose paragraphs and are excluded by geometry and terminal syntax.
"""
from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

MIN_FINAL_FRACTION = 0.50
MIN_REFERENCE_WIDTH_PT = 300.0
MIN_PROSE_BASELINE_GAP_PT = 9.5
MAX_PROSE_BASELINE_GAP_PT = 14.5


@dataclass
class Line:
    xmin: float
    xmax: float
    ymin: float
    words: list[str] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def text(self) -> str:
        return " ".join(word for word in self.words if word).strip()


@dataclass
class Block:
    page: int
    index: int
    lines: list[Line] = field(default_factory=list)


class BBoxParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page = 0
        self.block_index = -1
        self.block: Block | None = None
        self.line: Line | None = None
        self.in_word = False
        self.word_parts: list[str] = []
        self.blocks: list[Block] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "page":
            self.page += 1
            self.block_index = -1
        elif tag == "block":
            self.block_index += 1
            self.block = Block(self.page, self.block_index)
        elif tag == "line" and self.block is not None:
            self.line = Line(
                xmin=float(values["xmin"]),
                xmax=float(values["xmax"]),
                ymin=float(values["ymin"]),
            )
        elif tag == "word" and self.line is not None:
            self.in_word = True
            self.word_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_word:
            self.word_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "word" and self.in_word and self.line is not None:
            self.line.words.append("".join(self.word_parts).strip())
            self.in_word = False
            self.word_parts = []
        elif tag == "line" and self.line is not None and self.block is not None:
            if self.line.words:
                self.block.lines.append(self.line)
            self.line = None
        elif tag == "block" and self.block is not None:
            if self.block.lines:
                self.blocks.append(self.block)
            self.block = None


def without_terminal_identifier(block: Block) -> Block:
    """Return the prose portion when a citation ends in an atomic URL or DOI."""
    lines = list(block.lines)
    if lines and re.fullmatch(
        r"(?:https?://\S+|(?:doi:\s*)?10\.\d{4,9}/\S+)[.]?",
        lines[-1].text,
        flags=re.IGNORECASE,
    ):
        lines.pop()
    return Block(page=block.page, index=block.index, lines=lines)


def natural_prose(block: Block) -> bool:
    if len(block.lines) < 2:
        return False
    reference = max(line.width for line in block.lines)
    if reference < MIN_REFERENCE_WIDTH_PT:
        return False
    gaps = [block.lines[i + 1].ymin - block.lines[i].ymin
            for i in range(len(block.lines) - 1)]
    baseline_gap = statistics.median(gaps)
    if not (MIN_PROSE_BASELINE_GAP_PT <= baseline_gap <= MAX_PROSE_BASELINE_GAP_PT):
        return False
    text = " ".join(line.text for line in block.lines)
    if "<|im_" in text:
        return False
    # Paragraphs end in prose punctuation. Blocks flowing directly into a
    # display equation, headings, algorithms, and most table cells do not.
    if not re.search(r"""[.!?:;]["'”’)]*$""", block.lines[-1].text):
        return False
    return len(re.findall(r"[A-Za-z]+", text)) >= 8


def extract_blocks(pdf: Path) -> list[Block]:
    with tempfile.NamedTemporaryFile(suffix=".html") as handle:
        result = subprocess.run(
            ["pdftotext", "-bbox-layout", str(pdf), handle.name],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise SystemExit(result.stderr.strip() or "pdftotext failed")
        parser = BBoxParser()
        parser.feed(Path(handle.name).read_text(encoding="utf-8", errors="replace"))
    return parser.blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("paper/main.pdf"))
    parser.add_argument("--min-final-fraction", type=float,
                        default=MIN_FINAL_FRACTION)
    args = parser.parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"line-fill contract failed: missing PDF {args.pdf}")
    if not 0 < args.min_final_fraction < 1:
        raise SystemExit("line-fill fraction must lie strictly between zero and one")

    violations: list[tuple[Block, float, float]] = []
    checked = 0
    for rendered_block in extract_blocks(args.pdf):
        block = without_terminal_identifier(rendered_block)
        if not natural_prose(block):
            continue
        checked += 1
        reference = max(line.width for line in block.lines)
        ratio = block.lines[-1].width / reference
        if ratio + 1e-9 < args.min_final_fraction:
            violations.append((block, ratio, reference))

    if violations:
        details = []
        for block, ratio, reference in violations:
            details.append(
                f"page {block.page}, block {block.index}: final={ratio:.1%} "
                f"({block.lines[-1].width:.1f}/{reference:.1f} pt); "
                f"ending={block.lines[-1].text!r}"
            )
        raise SystemExit(
            "line-fill contract failed: natural-prose final lines must occupy "
            f">={args.min_final_fraction:.0%} of their measure; "
            f"{len(violations)}/{checked} checked blocks violate\n" +
            "\n".join(details)
        )
    print(
        f"Line-fill contract passed: {checked} natural-prose blocks, "
        f"minimum final-line fraction {args.min_final_fraction:.0%}."
    )


if __name__ == "__main__":
    main()
