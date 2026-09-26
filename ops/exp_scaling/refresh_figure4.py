#!/usr/bin/env python3
"""Refresh Figure 4 end to end and report what the manuscript must now say.

The refresh is six coupled steps -- render the rolling preview, render the
dated snapshot the manuscript embeds, rebuild the four-metric table, re-pin the
snapshot timestamp in the contract, re-sync the table rows, rebuild the PDF --
and skipping any one of them leaves the paper inconsistent with its own
artifacts. Running them by hand is how the interim prose went stale while the
panel counts stayed right.

This runs all of them, then recomputes the directional claims the "Interim
directions so far" paragraph states and prints them, so a claim that moved is
visible instead of assumed unchanged. It does not edit prose: that paragraph is
English and stays a human decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIEW = HERE / "plot_figure4_with_falcon_preview.py"
TABLE_BUILDER = HERE / "build_figure4_interim_table.py"
ROW_RENDERER = HERE / "render_interim_replay_table.py"
CONTRACT = ROOT / "ops/check_paper_figure1_contract.py"
DATED_STEM = ROOT / "paper/figures/figure4_interim_20260806"
DATED_JSON = DATED_STEM.with_suffix(".json")
TABLE_JSON = ROOT / "paper/results/figure4_interim_20260806_table.json"
METRICS = ("pass1", "pass8", "mean8", "distinct8")


def run(command: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            f"step failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def snapshot_timestamp() -> str:
    return json.loads(DATED_JSON.read_text(encoding="utf-8"))["generated_at"]


def directional_claims() -> dict[str, object]:
    """Recompute exactly what the interim paragraph asserts."""

    payload = json.loads(TABLE_JSON.read_text(encoding="utf-8"))
    tally = {metric: [0, 0, 0] for metric in METRICS}
    exceptions: list[str] = []
    gaps: list[tuple[float, str, int]] = []
    panels = 0
    for family, record in payload["families"].items():
        for domain, entry in record["domains"].items():
            panels += 1
            control = entry["arms"]["control"]["means"]
            replay = entry["arms"]["replay"]["means"]
            for metric in METRICS:
                low, high = control.get(metric), replay.get(metric)
                if low is None or high is None:
                    continue
                index = 0 if high > low else (1 if high == low else 2)
                tally[metric][index] += 1
                if index:
                    exceptions.append(
                        f"{'tie  ' if index == 1 else 'lower'} "
                        f"{family}/{domain}/{metric}: {high - low:+.3f}"
                    )
            gaps.append(
                (
                    replay["distinct8"] - control["distinct8"],
                    f"{family}/{domain}",
                    len(entry["paired_seeds"]),
                )
            )
    gaps.sort()
    return {
        "panels": panels,
        "tally": tally,
        "exceptions": exceptions,
        "distinct8_min": gaps[0],
        "distinct8_max": gaps[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pdf", action="store_true", help="do not rebuild paper/main.pdf"
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    print("[1/6] rolling preview")
    run([args.python, str(PREVIEW)])
    print("[2/6] dated snapshot the manuscript embeds")
    run([args.python, str(PREVIEW), "--output", str(DATED_STEM)])
    print("[3/6] four-metric table")
    run([args.python, str(TABLE_BUILDER)])
    print("[4/6] verify snapshot identity")
    print("      ", snapshot_timestamp())
    print("[5/6] sync manuscript table rows")
    print("      ", run([args.python, str(ROW_RENDERER), "--write"]))
    print("[6/6] contract")
    print("      ", run([args.python, str(CONTRACT)]))

    if not args.skip_pdf:
        print("[pdf] latexmk")
        run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=ROOT / "paper",
        )

    claims = directional_claims()
    print(f"\ninterim paragraph must state {claims['panels']} evaluable panels")
    for metric, (high, tie, low) in claims["tally"].items():
        print(f"  {metric:<10} higher {high} / tie {tie} / lower {low}  (n={high+tie+low})")
    for line in claims["exceptions"]:
        print(f"    {line}")
    low_gap, low_name, low_pairs = claims["distinct8_min"]
    high_gap, high_name, _ = claims["distinct8_max"]
    print(
        f"  distinct@8 range {low_gap:+.3f} ({low_name}, {low_pairs} pairs)"
        f" .. {high_gap:+.3f} ({high_name})"
    )
    print("\nCheck these against the 'Interim directions so far' paragraph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
