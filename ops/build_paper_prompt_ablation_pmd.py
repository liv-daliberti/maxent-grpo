#!/usr/bin/env python3
"""Paired PCMD for the prompt-hint ablation, per checkpoint group and cell.

The hint tables reported pass@8 and distinct@8 only. Both fall when the model
simply succeeds less often, which is the confound PCMD removes, and the draws
needed to compute it were retained the whole time: every response carries the
canonical key its verifier assigned, or none when it was not verified.

The estimand is paired on prompts. PCMD needs two verified responses to a
prompt before it exists, and removing the hint changes how often that happens,
so each arm defines it on a different set; only prompts defined in both enter
the difference. The result is deliberately sparse, and the sparsity is the
finding: on this cohort most cells cannot support the measurement at all.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))
from mode_diversity import mode_diversity  # noqa: E402
SOURCE = ROOT / "artifacts/modebench_prompt_ablation_20260911/local/results_v2"
OUT = ROOT / "paper/results/prompt_ablation_pmd.json"
#: As everywhere else: two verified responses define a prompt, thirty defined
#: prompts make a cell reportable.
MIN_VERIFIED_PER_PROMPT = 2
MIN_PAIRED_PROMPTS = 30
DIRECTORY = re.compile(
    r"qwen05b_(?:E119_(?P<domain>.+?)_(?P<method>replay_drgrpo|drgrpo)_s(?P<seed>\d+)|initial)$")


def per_prompt(path: Path) -> dict[tuple, float]:
    """PCMD for each (arm, domain, level, prompt) with two verified responses."""
    groups: dict[tuple, list[str]] = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # An unverified draw carries no canonical key, so it contributes no mode.
        if row.get("canonical_key") is None:
            continue
        groups[(row["arm"], row["domain"], int(row["level"]),
                int(row["row_index"]))].append(row["canonical_key"])
    out = {}
    for key, keys in groups.items():
        if len(keys) < MIN_VERIFIED_PER_PROMPT:
            continue
        # The shared estimator, not the plug-in 1 - sum q^2: PCMD is the
        # unbiased pairwise form 1 - sum C(n,2)/C(K,2) everywhere else in the
        # paper, and the two differ most exactly where K is small, which is
        # where these cohorts live.
        out[key] = mode_diversity(collections.Counter(keys))
    return out


def draw_counts(path: Path) -> dict[tuple, dict[str, int]]:
    """Draws and verified draws per (arm, domain, level) in one checkpoint."""
    counts: dict[tuple, dict[str, int]] = collections.defaultdict(
        lambda: {"draws": 0, "verified": 0})
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cell = counts[(row["arm"], row["domain"], int(row["level"]))]
        cell["draws"] += 1
        if row.get("canonical_key") is not None:
            cell["verified"] += 1
    return counts


def build_record(source: Path = SOURCE) -> dict:
    """The whole record, so a checker can rebuild it and compare."""
    paired: dict[tuple, dict[str, list]] = collections.defaultdict(
        lambda: {"original": [], "neutral": [], "seeds": set()})
    # A seed whose checkpoint ran but never lands two verified draws on one
    # prompt in both arms contributes nothing to PCMD and silently leaves the
    # average. That is a property of the intervention, not a missing file, so
    # the draws behind the absence are recorded rather than inferred later.
    gaps: dict[tuple, list[dict]] = collections.defaultdict(list)
    sources = {}
    for directory in sorted(p for p in source.iterdir() if p.is_dir()):
        match = DIRECTORY.match(directory.name)
        responses = directory / "responses.jsonl"
        if not match or not responses.is_file():
            continue
        sources[directory.name] = hashlib.sha256(responses.read_bytes()).hexdigest()
        method = match.group("method") or "initial"
        values = per_prompt(responses)
        contributed = set()
        for (arm, domain, level, row), value in values.items():
            if arm != "original":
                continue
            twin = ("neutral", domain, level, row)
            if twin not in values:
                continue
            cell = paired[(method, domain, level)]
            cell["original"].append(value)
            cell["neutral"].append(values[twin])
            contributed.add((domain, level))
            if match.group("seed"):
                cell["seeds"].add(int(match.group("seed")))
        if match.group("seed"):
            counts = draw_counts(responses)
            for domain, level in sorted({(d, l) for _, d, l in counts}):
                if (domain, level) in contributed:
                    continue
                gaps[(method, domain, level)].append({
                    "seed": int(match.group("seed")),
                    "arms": {arm: dict(counts[(arm, domain, level)])
                             for arm in ("original", "neutral")
                             if (arm, domain, level) in counts},
                })

    cells = {}
    for (method, domain, level), value in sorted(paired.items()):
        n = len(value["original"])
        original = statistics.fmean(value["original"])
        neutral = statistics.fmean(value["neutral"])
        cells[f"{method}/{domain}/level{level}"] = {
            "method": method, "domain": domain, "level": level,
            "paired_prompts": n, "seeds": sorted(value["seeds"]),
            "seeds_without_support": sorted(gaps[(method, domain, level)],
                                            key=lambda item: item["seed"]),
            "reportable": n >= MIN_PAIRED_PROMPTS,
            # A cell can be reportable and still carry no information: MathIR
            # returns one key to every prompt in both arms, so PCMD is defined
            # and identically zero. That is a property of the construction, not
            # a null result about the hint.
            "degenerate": original == 0.0 and neutral == 0.0,
            "original": original, "neutral": neutral,
            "effect": statistics.fmean(b - a for a, b in
                                       zip(value["original"], value["neutral"])),
        }
    reportable = [c for c in cells.values() if c["reportable"]]
    return {
        "schema": "paper-prompt-ablation-pmd-v1",
        "estimand": "neutral minus original wording, terminal PCMD, averaged "
                    "over the prompts both arms define",
        "support": (f"a prompt enters with {MIN_VERIFIED_PER_PROMPT} verified "
                    f"responses in both arms; a cell is reportable from "
                    f"{MIN_PAIRED_PROMPTS} such prompts"),
        "coverage": {"cells": len(cells), "reportable": len(reportable),
                     "reportable_and_informative":
                         sum(1 for c in reportable if not c["degenerate"])},
        "builder": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                    "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "sources": sources,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    record = build_record(args.source)
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"event": "built", **record["coverage"]}))


if __name__ == "__main__":
    main()
