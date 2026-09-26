#!/usr/bin/env python3
"""Render the complete sealed legacy live PNG as a PDF report page."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT / "paper/figures/e68_e58_vs_grpo_05b_12ep_live.png"
)
OUTPUT = (
    ROOT
    / "paper/figures/"
    "e68_e58_vs_grpo_05b_12ep_live_historical_reference_20260730.pdf"
)


def render() -> None:
    image = mpimg.imread(SOURCE)
    height, width = image.shape[:2]
    figure = plt.figure(
        figsize=(16.0, 16.0 * height / width),
        facecolor="white",
    )
    axis = figure.add_axes([0, 0, 1, 1])
    axis.imshow(image)
    axis.axis("off")
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp")
    figure.savefig(
        temporary,
        format="pdf",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
    )
    temporary.replace(OUTPUT)
    plt.close(figure)
    print(f"[e68-live-reference] wrote {OUTPUT}")


if __name__ == "__main__":
    render()
