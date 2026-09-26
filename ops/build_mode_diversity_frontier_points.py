#!/usr/bin/env python3
"""Extract (pass@8, PCMD) for each hosted deployment cell, for the levels figure.

The hosted comparison already publishes correct-pair collision per
model/level/domain; PCMD is one minus that, on the same success-conditional
definition the local grid uses. Frontier models have no parameter count, so
they are emitted separately rather than being placed on the scale ramp.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'paper/results/frontier_comparison_20260911.json'
# Level 4 was collected separately, direct through OpenAI rather than Azure, on a
# dataset whose composite admission is not complete. Both facts ride with every
# point it contributes.
LEVEL4 = ROOT / 'artifacts/frontier_modebench_gpt56sol_level4_20260914/audited_primary_samples.jsonl'
LEVEL5 = ROOT / 'artifacts/frontier_modebench_gpt56sol_level5_20260915/audited_primary_samples.jsonl'
OUT = ROOT / 'paper/results/mode_diversity_frontier_points.json'
# the hosted comparison spells PantryPlan with an underscore
DOMAIN = {'graph_coloring': 'graph_coloring', 'countdown': 'countdown',
          'python_factors': 'python_factors', 'mathir': 'mathir',
          'pantry_plan': 'pantry', 'pantry': 'pantry'}
MIN_DEFINED = 30


def hosted_level_cells(path: Path, level: str, *, admitted: bool) -> list[dict]:
    """Per-domain points for one OpenAI-direct hosted level.

    ``pmd`` is pooled over verified pairs, matching how the Azure cells above
    are read out of their published collision estimate, so the two are the same
    statistic even though they come from different providers.
    """
    from collections import Counter, defaultdict

    # Prefer the normalized samples, so the level is graded the same way as the
    # levels it sits beside; fall back to strict only if normalization has not
    # been run yet.
    normalized = path.with_name('normalized_secondary_samples.jsonl')
    source = normalized if normalized.is_file() else path
    if not source.is_file():
        return []
    by_prompt: dict[tuple, Counter] = defaultdict(Counter)
    seen: set[tuple] = set()
    for line in source.read_text().splitlines():
        record = json.loads(line)
        key = (record['domain'], record['row_index'])
        seen.add(key)
        if record.get('verified') and record.get('canonical_key') is not None:
            by_prompt[key][record['canonical_key']] += 1
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for domain, row_index in seen:
        grouped[domain].append(by_prompt.get((domain, row_index), Counter()))
    cells = []
    for domain, counters in grouped.items():
        colliding = sum(sum(n * (n - 1) for n in c.values()) for c in counters)
        pairs = sum(sum(c.values()) * (sum(c.values()) - 1) for c in counters)
        passed = sum(1 for c in counters if sum(c.values()))
        cells.append({
            'model': 'GPT-5.6 Sol',
            'level': level,
            'domain': DOMAIN.get(domain, domain),
            'pass8': passed / len(counters),
            'pmd': 1.0 - colliding / pairs if pairs else None,
            'correct_pairs': pairs / 2,
            'reportable': bool(pairs and pairs / 2 >= MIN_DEFINED),
            'provider': 'openai_direct',
            'level_admitted': admitted,
            'grading': 'normalized' if source is normalized else 'strict',
        })
    return [c for c in cells if c['pmd'] is not None]


def _level4_normalization_note() -> dict:
    """Strict grading rejects LaTeX-escaped Python; say so where the data lives.

    The overlay reads strict cells at every level, so Level 4 is graded strictly
    too. In Python that is not a capability measurement: the deployment emits
    Python wrapped in LaTeX escapes, which the strict grader refuses, and the
    frozen formatting normalizer rescues most of it. The same artifact affects
    the Levels 1-3 Python overlay cells.
    """
    path = ROOT / 'artifacts/frontier_modebench_gpt56sol_level4_20260914/normalized_summary.json'
    if not path.is_file():
        return {}
    summary = json.loads(path.read_text())
    return {'analysis': summary['analysis'],
            'rescued_by_normalization': summary['rescued_by_normalization'],
            'per_domain': summary['per_domain'],
            'affects_python_most': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    raw = json.loads(args.source.read_text())

    cells = []
    for model in raw['models']:
        label = model.get('label') or model.get('model')
        # Read the formatting-normalized cells, which is the grading the
        # manuscript's own frontier display uses. Strict grading rejects
        # LaTeX-escaped Python outright, which is a presentation artifact rather
        # than a capability measurement and collapses one deployment's Python
        # cells to near zero while leaving the others untouched.
        normalized = ((model.get('normalized_secondary') or {}).get('cells')) or {}
        for key, strict_cell in model['cells'].items():
            cell = normalized.get(key) or strict_cell
            graded = 'normalized' if key in normalized else 'strict'
            metrics = cell.get('metrics', {})
            collision = (metrics.get('correct_pair_collision') or {}).get('estimate')
            pass8 = (metrics.get('pass8') or {}).get('estimate')
            if collision is None or pass8 is None:
                continue
            counts = cell.get('counts', {})
            pairs = counts.get('correct_pairs')
            cells.append({
                'model': label,
                'level': f"level{cell['level']}",
                'domain': DOMAIN.get(cell['domain'], cell['domain']),
                'pass8': pass8,
                'pmd': 1.0 - collision,
                'correct_pairs': pairs,
                # a hosted cell is reportable when enough prompts yielded a
                # verified pair, matching the local grid's support rule
                'reportable': bool(pairs and pairs >= MIN_DEFINED),
                'grading': graded,
            })
    extra = hosted_level_cells(LEVEL4, 'level4', admitted=False)
    # Level 5's evaluation splits carry admitted bindings, which Level 4's do
    # not; the flag travels with the cells so a reader of this payload cannot
    # confuse the two.
    extra_five = hosted_level_cells(LEVEL5, 'level5', admitted=True)
    cells.extend(extra)
    cells.extend(extra_five)
    payload = {
        'schema': 'mode-diversity-frontier-points-v1',
        'level4': {
            'present': bool(extra), 'cells': len(extra),
            'grading': 'strict, matching the Levels 1-3 overlay cells',
            'formatting_sensitivity': _level4_normalization_note(),
            'provider': 'openai_direct',
            'differs_from_levels_1_3_provider': True,
            'composite_admission_completed': False,
            'claims_not_made': [
                'No difficulty equivalence between Level 4 and any other level is claimed.',
                'Level 4 and Levels 1-3 hosted cells were served by different providers.',
            ],
        },
        'level5': {
            'present': bool(extra_five), 'cells': len(extra_five),
            'provider': 'openai_direct',
            'differs_from_levels_1_3_provider': True,
            'dataset_admitted': True,
            'claims_not_made': [
                'No difficulty equivalence between Level 5 and any other level is claimed.',
                'Level 5 and Levels 1-3 hosted cells were served by different providers.',
            ],
        },
        'definition': {'pmd': 'one minus correct-pair collision',
                       'grading': 'frozen formatting-normalized, matching the '
                                  'frontier display; strict grading scores '
                                  'LaTeX-escaped Python as failure',
                       'min_correct_pairs': MIN_DEFINED,
                       'note': 'inference-only deployments; no parameter count'},
        'source': {'path': str(args.source.relative_to(ROOT)),
                   'sha256': hashlib.sha256(args.source.read_bytes()).hexdigest()},
        'models': sorted({c['model'] for c in cells}),
        'cells': sorted(cells, key=lambda c: (c['model'], c['level'], c['domain'])),
    }
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + '\n')
    rep = sum(1 for c in cells if c['reportable'])
    print(json.dumps({'event': 'built', 'models': len(payload['models']),
                      'cells': len(cells), 'reportable': rep}))


if __name__ == '__main__':
    main()
