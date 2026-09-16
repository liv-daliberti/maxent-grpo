#!/usr/bin/env python3
"""Pantry at Levels 3 and 5, on a context wide enough to read every prompt.

PantryPlan trains with ``prompt_max_length`` 640 inside a 704-token context. Its
Level-1, Level-2 and Level-4 evaluation splits fit: the longest rendered prompt
is 589. Level 3 and Level 5 do not -- they render up to 737 and 741 tokens, so
the training surface cannot express 42 and 74 of their 128 prompts.

Replaying the training rule there (drop the over-long rows, as the learner does)
is faithful but measures Levels 3 and 5 on the short two-thirds and two-fifths of
their prompts while every other level uses all 128. For a comparison whose whole
point is to hold the prompt population fixed across the ladder, a population that
shifts with the level is worse than a wider context.

So these cells declare a wider surface and say so. Only the context and the
prompt budget move; the template, generation length, KV-cache fraction, action
support, sampling policy and checkpoint are the run's own. The widening is not
assumed harmless -- the manifest carries Level-1 cells on the same widened
surface whose prompts already fit under 640, so a greedy trace on them must
still reproduce the recorded one. If it does, widening changed no output it
could have changed; if it does not, the gate refuses the cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'var/artifacts/pmd_independent_resample_20260915'
# Each domain is sized from its own longest rendered prompt, measured across
# every template and level, with room for the ladder to grow. The context keeps
# the run's own generation headroom, so only the prompt budget really moves.
#
#   PantryPlan  longest 741 (Falcon3 template, Level 5)  -> 768 + 64
#   MathIR      longest 287 (Falcon3 template, Levels 3-4) -> 320 + 128
#
# MathIR overflows for Falcon only: the Falcon3 tokenizer renders the same
# prompts about thirty-five tokens longer than Qwen's, which is the whole margin
# at a 256 budget. The Qwen scales stay on their trained surface.
WIDENED = {
    'pantry_plan': {'prompt_max': 768, 'headroom': 64, 'scales': None},
    'mathir': {'prompt_max': 320, 'headroom': 128, 'scales': ('falcon1b',)},
}


def widen(cell: dict) -> dict:
    """A copy of the cell on the widened surface, with the change recorded."""
    spec = WIDENED[cell['domain']]
    wide_prompt = spec['prompt_max']
    wide_len = wide_prompt + spec['headroom']
    out = json.loads(json.dumps(cell))
    config = out['eval_config']
    out['context_widening'] = {
        'from_prompt_max_length': int(config['prompt_max_length']),
        'to_prompt_max_length': wide_prompt,
        'from_max_model_len': int(config['max_model_len']),
        'to_max_model_len': wide_len,
        'reason': 'prompts on the harder levels exceed the trained context; '
                  'widened so every level shares one prompt population',
        'unchanged': ['prompt_template', 'eval_generate_max_length',
                      'vllm_gpu_ratio', 'canonical_action_task', 'seed policy',
                      'checkpoint'],
    }
    config['prompt_max_length'] = wide_prompt
    config['max_model_len'] = wide_len
    return out


def in_scope(cell: dict) -> bool:
    """Only the domain/scale combinations whose prompts actually overflow."""
    spec = WIDENED.get(cell['domain'])
    if spec is None:
        return False
    return spec['scales'] is None or cell['scale'] in spec['scales']


def build(levels: list[str]) -> dict:
    cells: list[dict] = []
    # Level-1 saved-draws cells come first: they are what proves the surface.
    # Their prompts fit the original budget, so a greedy trace on the widened
    # engine must match the recorded one exactly.
    control = json.loads((RUN / 'cohort_manifest.json').read_text())
    by_template: dict[str, dict] = {}
    for cell in control['cells']:
        if not in_scope(cell):
            continue
        if cell.get('prompt_source', 'saved_draws') != 'saved_draws':
            continue
        key = (cell['gpu'], cell['eval_config']['prompt_template'])
        by_template.setdefault(key, cell)
    for cell in by_template.values():
        widened = widen(cell)
        widened['role'] = 'surface_control'
        cells.append(widened)
    control_count = len(cells)
    for level in levels:
        payload = json.loads((RUN / f'levels_cohort_manifest_{level}.json').read_text())
        for cell in payload['cells']:
            if not in_scope(cell):
                continue
            widened = widen(cell)
            widened['role'] = 'measurement'
            cells.append(widened)
    return {
        'schema': 'pmd-resample-cohort-v1',
        'cohort': 'wide_context',
        'levels': list(levels),
        'widened_surface': {d: {'prompt_max_length': v['prompt_max'],
                                'max_model_len': v['prompt_max'] + v['headroom'],
                                'scales': list(v['scales']) if v['scales'] else 'all'}
                            for d, v in WIDENED.items()},
        'surface_controls': control_count,
        'cell_count': len(cells),
        'cells': cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', action='append', default=None)
    parser.add_argument('--output', type=Path,
                        default=RUN / 'wide_context_cohort_manifest.json')
    args = parser.parse_args()
    payload = build(args.level or ['level3', 'level5'])
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'cells': payload['cell_count'],
                      'surface_controls': payload['surface_controls'],
                      'levels': payload['levels'],
                      'widened_surface': payload['widened_surface'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
