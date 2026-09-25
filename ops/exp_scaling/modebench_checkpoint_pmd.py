#!/usr/bin/env python3
"""PCMD for one admitted evaluation checkpoint, read back from its own origins.

The frozen verified-sample archive that serves the manuscript's Level-1 and
Level-2 training claims is bound to its own census, and that census does not
carry every checkpoint the level figure now plots. This module computes the
same statistic the archive reports, from the same run-directory lines a
coverage snapshot already recorded, so a checkpoint outside the census is
measured identically rather than differently.

Nothing here selects a checkpoint or re-grades a response: it re-reads saved
canonical keys that are already gated by their verification reward.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

from load_paper_collision_samples import normalize_draw  # noqa: E402
from mode_diversity import mode_diversity  # noqa: E402

READERS = ('ops/mode_diversity.py', 'ops/exp_scaling/load_paper_collision_samples.py',
           'ops/exp_scaling/modebench_checkpoint_pmd.py')
DEFINITION = {
    'metric': 'pairwise correct-mode diversity (PCMD)',
    'estimator': "PCMD = 1 - sum_m n_m (n_m - 1) / (K (K - 1)) over a prompt's verified responses",
    'aggregation': "pooled over a checkpoint's four admitted draws, unweighted mean over defined prompts",
}


def read_draws(checkpoint: dict) -> dict[int, dict]:
    """Load the exact four draws a coverage snapshot admitted for this step."""
    wanted: dict[str, dict[int, int]] = defaultdict(dict)
    for draw in checkpoint['draws']:
        origin = draw['origins'][0]
        wanted[origin['path']][origin['line']] = draw['draw_index']
    draws = {}
    for path, lines in wanted.items():
        with open(path) as handle:
            for number, raw in enumerate(handle, 1):
                if number in lines:
                    draws[lines[number]] = normalize_draw(json.loads(raw))
    if set(draws) != {0, 1, 2, 3}:
        raise RuntimeError('admitted draws are not readable at their recorded origins')
    return draws


def checkpoint_pmd(checkpoint: dict, min_defined: int) -> dict:
    """PCMD over the checkpoint's prompts, pooling each prompt's four draws."""
    draws = read_draws(checkpoint)
    pooled: dict[str, Counter] = defaultdict(Counter)
    for index in range(4):
        for prompt in draws[index]['prompts']:
            pooled[prompt['prompt_id']].update(
                key for key in prompt['verified_keys'] if key is not None)
    values = [value for value in (mode_diversity(counts) for counts in pooled.values())
              if value is not None]
    return {
        'pmd': statistics.fmean(values) if values else None,
        'defined_prompts': len(values), 'prompts': len(pooled),
        'support': len(values) / len(pooled) if pooled else 0.0,
        'reportable': len(values) >= min_defined,
    }


def definition(min_defined: int, digest) -> dict:
    return {**DEFINITION, 'min_defined_prompts': min_defined,
            'readers': {path: digest((ROOT / path).read_bytes()) for path in READERS}}
