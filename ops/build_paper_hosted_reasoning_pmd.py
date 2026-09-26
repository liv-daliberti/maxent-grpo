#!/usr/bin/env python3
"""Paired PCMD for the reasoning-disabled control, per deployment.

The reasoning-off tables reported ``pass@8`` and ``distinct@8`` only, and said
so themselves: reduced success also reduces opportunities to observe distinct
successful modes, so a fall in ``distinct@8`` cannot be read as concentration.
PCMD is the endpoint that separates them, and it was recoverable the whole
time -- both arms retain a canonical key for every draw, so the per-prompt key
distribution can be rebuilt without re-running a single hosted call.

The estimand is paired on prompts, not on arms read separately. PCMD needs two
verified responses to a prompt before it is defined at all, and disabling
deliberation changes how often that happens, so the set of prompts each arm can
define is not the same set. Averaging each arm over its own defined prompts
compares different populations: it moved grok-4.3's baseline from .183 to .135
here. Only prompts both arms define enter the difference.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))
from mode_diversity import mode_diversity  # noqa: E402
SOURCE = ROOT / "artifacts/hosted_reasoning_off_32_20260911_v2"
OUT = ROOT / "paper/results/hosted_reasoning_pmd.json"
#: The reasoning-disabled arm, normalized; and the audited medium cohort it is
#: paired against, read under the same normalizer.
OFF_FILE, ON_FILE = "normalized_samples.jsonl", "paired_medium_samples.jsonl"
#: A prompt defines PCMD only with two verified responses, and a cell is
#: reported only from enough such prompts, as everywhere else in the paper.
MIN_VERIFIED_PER_PROMPT = 2
MIN_DEFINED_PROMPTS = 30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def per_prompt(path: Path, nested: bool, strict: bool = False) -> dict[tuple, float]:
    """PCMD for each prompt that returns two verified responses."""
    groups: dict[tuple, list[str]] = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = row["normalization"] if nested else row
        verified, key = record.get("verified"), record.get("canonical_key")
        if strict and isinstance(row.get("raw_strict_verified"), bool):
            # The medium cohort keeps both grades; read the strict one when the
            # arm it is paired against has no normalized grading at all.
            verified, key = row["raw_strict_verified"], row.get("raw_strict_canonical_key")
        if verified and key is not None:
            groups[(row["domain"], int(row["level"]), int(row["row_index"]))].append(
                record["canonical_key"])
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


def summarize(rows: list[tuple], on: dict[tuple, float],
              off: dict[tuple, float]) -> dict:
    """The paired reading over one group of prompts both arms define."""
    return {
        "paired_prompts": len(rows),
        "reportable": len(rows) >= MIN_DEFINED_PROMPTS,
        "reasoning_on": statistics.fmean(on[k] for k in rows) if rows else None,
        "reasoning_off": statistics.fmean(off[k] for k in rows) if rows else None,
        "effect": statistics.fmean(off[k] - on[k] for k in rows) if rows else None,
    }


def deployment(directory: Path) -> dict | None:
    off_path, on_path = directory / OFF_FILE, directory / ON_FILE
    if not (off_path.is_file() and on_path.is_file()):
        return None
    off, on = per_prompt(off_path, True), per_prompt(on_path, False)
    shared = sorted(set(off) & set(on))
    by_domain = {domain: summarize([k for k in shared if k[0] == domain], on, off)
                 for domain in sorted({key[0] for key in shared})}
    # The reasoning tables are read by level, so the paired estimand is carried
    # by level too. Each level is its own group of prompts and clears the
    # support bar on its own count; no level average is taken over the others.
    by_level = {str(level): summarize([k for k in shared if k[1] == level], on, off)
                for level in sorted({key[1] for key in shared})}
    return {
        **summarize(shared, on, off),
        "defined_reasoning_on_only": len(set(on) - set(off)),
        "defined_reasoning_off_only": len(set(off) - set(on)),
        "by_domain": by_domain,
        "by_level": by_level,
        "sources": {OFF_FILE: digest(off_path), ON_FILE: digest(on_path)},
    }


def build_record(source: Path = SOURCE) -> dict:
    """The whole record, so a checker can rebuild it and compare."""
    deployments = {}
    for directory in sorted(p for p in source.iterdir() if p.is_dir()):
        record = deployment(directory)
        if record is not None:
            deployments[directory.name] = record

    # DeepSeek V4 Pro failed complete-cohort admission by one response and is
    # not scored with the five. Its draws exist, so the measurement is made and
    # kept apart rather than discarded: both arms are read strictly, because the
    # formatting normalizer only ever ran on admitted cohorts, and one prompt
    # holds seven draws instead of eight. Neither is true of the five, so this
    # number is not comparable with them and is never averaged into them.
    excluded = {}
    partial = source / "deepseek_adapter_v1/deepseek_v4_pro"
    off_path, on_path = partial / "samples.jsonl", partial / ON_FILE
    if off_path.is_file() and on_path.is_file():
        off, on = per_prompt(off_path, False), per_prompt(on_path, False, strict=True)
        shared = sorted(set(off) & set(on))
        draws: dict[tuple, int] = collections.Counter()
        for line in off_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                draws[(row["domain"], int(row["level"]), int(row["row_index"]))] += 1
        short = {f"{k[0]}/level{k[1]}/row{k[2]}": n for k, n in draws.items() if n != 8}
        excluded["deepseek_v4_pro"] = {
            "admitted": False,
            "why_not_admitted": "one of 3,840 sample receipts was never recovered",
            "grading": "strict in both arms; the frozen normalizer was not run on "
                       "this cohort, so it is not on the same grading as the five",
            "prompts_without_eight_draws": short,
            "paired_prompts": len(shared),
            "reportable": len(shared) >= MIN_DEFINED_PROMPTS,
            "reasoning_on": statistics.fmean(on[k] for k in shared) if shared else None,
            "reasoning_off": statistics.fmean(off[k] for k in shared) if shared else None,
            "effect": statistics.fmean(off[k] - on[k] for k in shared) if shared else None,
            "sources": {"samples.jsonl": digest(off_path), ON_FILE: digest(on_path)},
        }
    if not deployments:
        raise SystemExit("no deployment retains both arms")
    return {
        "schema": "paper-hosted-reasoning-pmd-v2",
        "estimand": "reasoning disabled minus the matched medium cohort, terminal "
                    "PCMD, averaged over the prompts both arms define",
        "support": (f"a prompt enters only with {MIN_VERIFIED_PER_PROMPT} verified "
                    f"responses in both arms; a cell is reportable from "
                    f"{MIN_DEFINED_PROMPTS} such prompts"),
        "grading": "frozen formatting normalizer in both arms, as in the "
                   "reasoning-off tables",
        "builder": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                    "sha256": digest(Path(__file__))},
        "source_root": str(source.relative_to(ROOT)),
        "deployments": deployments,
        "excluded_deployments": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    record = build_record(args.source)
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"event": "built", "deployments": len(record["deployments"]),
                      "output": str(args.output.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
