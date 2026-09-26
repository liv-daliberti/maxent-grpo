#!/usr/bin/env python3
"""Stable visual identities for the canonical ten-method paper program.

The base geometry, typography, palette, grids, and save behavior remain in
paper_style.py. This extension maps scientific method names onto that visual
language without forcing ten curves into one panel. Related methods share a
hue and differ redundantly by dash and marker.
"""

from __future__ import annotations

from typing import Any

import paper_style as base


METHOD_STYLE: dict[str, dict[str, Any]] = {
    "grpo": {
        "label": "GRPO",
        # Ordinary GRPO is a different estimator, not a dash-only variant of
        # matched Dr.GRPO. Keep the two visually distinct even in dense
        # endpoint panels. The hue is base.COMPARATOR rather than the older
        # #2E6FBB: that blue sat at dE 10.0 from the method teal under
        # all-pairs, making GRPO vs Re:Dr --- the comparison the paper
        # is about --- the least separable pair on the page.
        "color": base.COMPARATOR,
        "linestyle": base.ARM_DASH[base.COMPARATOR],
        "marker": "o",
    },
    "drgrpo": {
        "label": "Dr.GRPO",
        "color": base.CONTROL,
        "linestyle": base.ARM_DASH[base.CONTROL],
        "marker": "s",
    },
    "ucpo": {
        "label": "UCPO",
        "color": base.ABLATION,
        "linestyle": base.ARM_DASH[base.ABLATION],
        "marker": "^",
    },
    "rlep_dr": {
        "label": "RLEP-Dr",
        # Was base.MUTED, which is the spine and tick colour: it made a real
        # comparator read as chrome, and the validator puts that grey at
        # dE 1.4 (deutan) from the method teal --- indistinguishable from the
        # arm it is meant to be compared against. It borrows the ADD_ON hue,
        # which it never shares a panel with; see paper_style.FRONTIER_FIVE.
        "color": base.ADD_ON,
        "linestyle": (0, (6, 1.5)),
        "marker": "D",
    },
    "replay_grpo": {
        "label": "Re:Dr (ours)",
        "color": base.ADAPTIVE,
        "linestyle": base.ARM_DASH[base.ADAPTIVE],
        # A pentagon, not the circle this used to carry: GRPO is a circle, and
        # in the frontier scatter the two sit in the same axes, where shape is
        # the redundant channel that makes identity survive a greyscale print.
        "marker": "p",
    },
    "adaptive_replay_grpo": {
        "label": "Adaptive Re:Dr",
        "color": base.METHOD,
        "linestyle": base.METHOD_DOSE_DASH,
        "marker": "s",
    },
    "semantic_maxent": {
        "label": "Semantic MaxEnt",
        "color": base.ADD_ON,
        "linestyle": base.ARM_DASH[base.ADD_ON],
        "marker": "^",
    },
    "adaptive_semantic_maxent": {
        "label": "Adaptive Semantic MaxEnt",
        "color": base.ADAPTIVE,
        "linestyle": (0, (1.2, 1.2)),
        "marker": "v",
    },
    "replay_semantic_maxent": {
        "label": "Re:Dr + Semantic MaxEnt",
        "color": base.ABLATION,
        "linestyle": base.ARM_DASH[base.ABLATION],
        "marker": "P",
    },
    "adaptive_semantic_replay": {
        "label": "Adaptive Semantic MaxEnt + Re:Dr",
        "color": base.ADAPTIVE,
        "linestyle": base.ARM_DASH[base.ADAPTIVE],
        "marker": "X",
    },
}


def method_style(key: str) -> dict[str, Any]:
    """Return a copy so one plot cannot mutate the paper-wide identity."""

    try:
        return dict(METHOD_STYLE[key])
    except KeyError as error:
        raise KeyError(f"unknown paper method {key!r}") from error
