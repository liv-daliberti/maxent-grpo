#!/usr/bin/env python3
"""Fail the build on wording the paper has decided against.

Two decisions are enforced here, both of which a regenerated results file or a
pasted paragraph can undo silently.

``breadth`` was the paper's word for the success-conditional axis. It is now
``diversity`` (or ``reasoning mode diversity``) everywhere a reader sees it.
Identifiers keep the old spelling on purpose -- a figure stem, a label, a
module path and a JSON key are not prose, and renaming them would break
bindings for a cosmetic gain -- so this check looks only at words standing on
their own.

The second decision is about attribution rather than vocabulary. The method's
accuracy gain comes from reusing verified successes, which is rejection
sampling fine-tuning with a deduplication rule; the diversity gain is what
uniform weighting buys, and the weighting ablation measures the correctness
effect as spanning zero. A sentence that credits the memory for both at once
re-conflates them, so the abstract's summary line is pinned to the split
wording rather than left to drift.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
MAIN = PAPER / 'main.tex'
# A word on its own: not part of a path, label, macro name or JSON key.
FORBIDDEN = re.compile(r"(?<![\w/_.-])[Bb]readth(?![\w/_.-])")
# Identifier-bearing constructs whose contents are not prose.
PROTECTED = re.compile(r"\\(?:label|ref|eqref|autoref|Cref|cref|includegraphics"
                       r"|input|path|url|href)(?:\[[^\]]*\])?\{[^}]*\}")


def prose(text: str) -> str:
    """The document with identifier arguments blanked out."""
    return PROTECTED.sub(lambda m: ' ' * len(m.group(0)), text)


def included(path: Path, seen: set[Path] | None = None) -> list[Path]:
    """main.tex and every .tex it pulls in, transitively."""
    seen = set() if seen is None else seen
    if path in seen or not path.is_file():
        return []
    seen.add(path)
    out = [path]
    for name in re.findall(r"\\input\{([^}]+)\}", path.read_text(encoding='utf-8')):
        target = PAPER / (name if name.endswith('.tex') else name + '.tex')
        out += included(target, seen)
    return out


def main() -> int:
    failures = []
    for path in included(MAIN):
        text = prose(path.read_text(encoding='utf-8'))
        for number, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN.search(line):
                failures.append(f"{path.relative_to(ROOT)}:{number}: "
                                f"{line.strip()[:96]}")
    # The abstract must keep the two gains apart.
    abstract = MAIN.read_text(encoding='utf-8').split(r'\begin{abstract}', 1)[-1]
    abstract = abstract.split(r'\end{abstract}', 1)[0]
    if 'rehearsing them uniformly' not in abstract:
        failures.append('paper/main.tex: the abstract no longer separates the '
                        'accuracy gain (reuse of verified successes) from the '
                        'diversity gain (uniform weighting); see '
                        'App. Uniform versus fresh-frequency replay')
    if failures:
        print(f'Paper vocabulary contract failed ({len(failures)}):')
        for line in failures[:40]:
            print('  ' + line)
        return 1
    print('Paper vocabulary contract passed: the success-conditional axis reads '
          'as diversity in every included file, and the abstract keeps the '
          'accuracy and diversity gains attributed separately.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
