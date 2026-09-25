"""Render benchmark domain names in monospace without restyling other text."""
from __future__ import annotations

import re

DOMAIN_NAMES = ("Graph", "Countdown", "Python", "MathIR", "PantryPlan", "Pantry")
DOMAIN_PATTERN = re.compile(r"\b(?:" + "|".join(DOMAIN_NAMES) + r")\b")
MONOSPACE_FAMILY = "DejaVu Sans Mono"


def domain_mathtext(text: str) -> str:
    """Style domain spans in mixed labels, leaving existing math unchanged."""
    pieces = re.split(r"(?<!\\)\$", text)
    for index in range(0, len(pieces), 2):
        pieces[index] = DOMAIN_PATTERN.sub(
            lambda match: r"$\mathtt{" + match.group(0) + r"}$", pieces[index])
    return "$".join(pieces)


def apply_domain_typography(figure) -> int:
    """Style visible domain labels, preserving sizes, colors, weights and data.

    Whole-name labels use the native monospace font. Mixed labels retain their
    surrounding font and use mathtext only for domain-name spans. Repeated
    calls are safe, including when the same figure is saved as PDF and PNG.
    """
    from matplotlib.text import Text

    changed = 0
    for artist in figure.findobj(match=Text):
        text = artist.get_text()
        if (not artist.get_visible() or not DOMAIN_PATTERN.search(text)
                or getattr(artist, "_paper_domain_typography", None) == text):
            continue
        if text.strip() in DOMAIN_NAMES:
            artist.set_fontfamily(MONOSPACE_FAMILY)
        else:
            artist.set_math_fontfamily("dejavusans")
            artist.set_text(domain_mathtext(text))
        artist._paper_domain_typography = artist.get_text()
        changed += 1
    return changed
