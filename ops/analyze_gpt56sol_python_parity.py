#!/usr/bin/env python3
"""Paired analysis of the two GPT-5.6 Sol Python wordings, offline.

No API calls and no grading happen here. The collected cohort is authenticated
against its own manifest before any statistic is computed, and every comparison
stays inside that one directory: both arms were served by the same OpenAI-direct
deployment, so neither is compared against the published Azure cells.

Two readings of the wording difference are reported, because they answer
different questions:

* **All prompts.** Accuracy and ``distinct@8`` average over all 128 prompts of a
  cell, including prompts with no verified draw, so a wording that simply
  succeeds more often moves them.
* **Jointly eligible prompts.** \\pmd{} conditions on verified draws, so an
  unrestricted difference between arms compares two different correct-pair
  populations. The joint reading restricts to prompts where *both* arms return
  at least two verified draws, which is the matched-population contrast.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'ops',):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_RUN = ROOT / 'artifacts/frontier_modebench_gpt56sol_python_parity_20260918'
SCHEMA = 'gpt56sol-python-wording-parity-analysis-v1'
ARMS = ('original', 'revised')
LEVELS = (1, 2, 3)
PROMPTS = 128
DRAWS = 8
GRADINGS = ('strict', 'normalized')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha(path: Path) -> str:
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(run: Path) -> dict:
    """Authenticate the collected cohort before a single statistic is computed."""
    manifest = json.loads((run / 'manifest.json').read_text())
    require(manifest['schema'] == 'gpt56sol-python-wording-parity-openai-direct-v1',
            'Unexpected collection schema')
    require(tuple(manifest['arms']) == ARMS and tuple(manifest['levels']) == LEVELS,
            'The collected arms or levels changed')
    expected = len(ARMS) * len(LEVELS) * PROMPTS * DRAWS
    require(manifest['requests'] == expected, f'Expected {expected} frozen requests')

    requests = {record['sample_id']: record for record in read_jsonl(run / 'requests.jsonl')}
    require(len(requests) == expected, 'Frozen request identities are not unique')

    strict = read_jsonl(run / 'audited_primary_samples.jsonl')
    require(len(strict) == expected, f'Expected {expected} graded responses, found {len(strict)}')
    require(len({record['sample_id'] for record in strict}) == expected,
            'Graded responses contain duplicate identities')
    for record in strict:
        frozen = requests.get(record['sample_id'])
        require(frozen is not None, f"{record['sample_id']}: graded a request that was not frozen")
        require(frozen['request_sha256'] == record['request_sha256']
                and frozen['row_sha256'] == record['row_sha256'],
                f"{record['sample_id']}: graded response does not bind its frozen request")
        require(record.get('status') == 'completed',
                f"{record['sample_id']}: response did not complete")

    normalized = {record['sample_id']: record
                  for record in read_jsonl(run / 'normalized_samples.jsonl')}
    require(set(normalized) == set(requests), 'Normalized analysis does not cover every response')
    for record in strict:
        secondary = normalized[record['sample_id']]
        require(secondary['strict_verified'] == record['verified'],
                f"{record['sample_id']}: normalized record disagrees with its strict grade")
        require(record['verified'] <= secondary['verified'],
                f"{record['sample_id']}: normalization withdrew a strict success")

    raw = {record['sample_id']: record['body']
           for record in read_jsonl(run / 'raw_responses.jsonl')}
    require(set(raw) >= set(requests), 'Native bodies do not cover every response')

    served = Counter(record.get('served_model') for record in strict)
    require(len(served) == 1, f'Responses report more than one served model: {dict(served)}')
    return {'manifest': manifest, 'strict': strict, 'normalized': normalized, 'raw': raw,
            'served_model': next(iter(served)),
            'bindings': {name: file_sha(run / name) for name in (
                'manifest.json', 'requests.jsonl', 'audited_primary_samples.jsonl',
                'normalized_samples.jsonl', 'raw_responses.jsonl')}}


def refusals(raw: dict, strict: list[dict]) -> dict:
    """Explicit provider-declared refusals, from the native bodies alone."""
    from audit_hosted_provider_outcomes import classify_native

    counts = defaultdict(int)
    for record in strict:
        verdict = classify_native(raw[record['sample_id']], 'responses')
        if verdict['refusal']:
            counts[(record['arm'], record['level'])] += 1
    return {f'{arm}/level{level}': counts[(arm, level)]
            for arm in ARMS for level in LEVELS}


#: The frozen grader parses Python, not LaTeX. This deployment sometimes wraps
#: its conditional in ``\text{...}``, which no rule in the frozen normalizer
#: undoes, so such a response cannot be scored whatever it says. The rate is
#: reported per arm because it is the mechanism behind most of the difference.
UNSCOREABLE_SURFACE = '\\text{'


def surface_taxonomy(run: Path, strict: list[dict], normalized: dict) -> dict:
    """Split each arm by a surface the frozen grader cannot parse."""
    text = {record['sample_id']: record['text']
            for record in read_jsonl(run / 'responses.jsonl') if 'text' in record}
    rows = defaultdict(lambda: Counter())
    for record in strict:
        body = text[record['sample_id']]
        bucket = 'unscoreable_surface' if UNSCOREABLE_SURFACE in body else 'parseable_surface'
        counts = rows[(record['arm'], bucket)]
        counts['responses'] += 1
        counts['verified'] += bool(normalized[record['sample_id']]['verified'])
    out = {}
    for (arm, bucket), counts in rows.items():
        out[f'{arm}/{bucket}'] = {
            'responses': counts['responses'], 'verified': counts['verified'],
            'rate': counts['verified'] / counts['responses'] if counts['responses'] else None}
    for arm in ARMS:
        unscoreable = out.get(f'{arm}/unscoreable_surface')
        require(unscoreable is None or unscoreable['verified'] == 0,
                f'{arm}: a response carrying the unscoreable surface was graded verified, '
                'so this split no longer explains what it claims to')
    return {'marker': UNSCOREABLE_SURFACE,
            'definition': 'A response whose text contains the LaTeX \\text{} macro. No rule in '
                          'the frozen normalizer removes it, so the frozen grader rejects the '
                          'response regardless of the expression inside it.',
            'reading': 'The parseable-surface rows are the arm comparison that is not confounded '
                       'by an unparseable surface. They are not a random subset of each arm, so '
                       'they bound rather than identify a content effect.',
            'buckets': out}


def _groups(records, key: str) -> dict:
    """Per (arm, level, row): the verified canonical keys of its eight draws."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record['arm'], record['level'], record['row_index'])].append(record)
    out = {}
    for identity, draws in grouped.items():
        require(len(draws) == DRAWS, f'{identity}: expected eight draws')
        out[identity] = [item[key] for item in draws if item['verified']]
    return out


def _cell(rows: list[list[str]]) -> dict:
    """Complete-cohort metrics over a set of eight-draw prompt groups."""
    correct = sum(len(keys) for keys in rows)
    distinct = sum(len(set(keys)) for keys in rows)
    pairs = sum(len(keys) * (len(keys) - 1) // 2 for keys in rows)
    colliding = sum(sum(n * (n - 1) // 2 for n in Counter(keys).values()) for keys in rows)
    return {'prompts': len(rows), 'responses': len(rows) * DRAWS,
            'correct_responses': correct, 'distinct_correct_modes': distinct,
            'correct_pairs': pairs, 'colliding_correct_pairs': colliding,
            'collision_eligible_prompts': sum(1 for keys in rows if len(keys) >= 2),
            'accuracy': correct / (len(rows) * DRAWS) if rows else None,
            'distinct8': distinct / len(rows) if rows else None,
            'pcmd': 1 - colliding / pairs if pairs else None}


def _bootstrap(original: list[list[str]], revised: list[list[str]],
               replicates: int, rng) -> dict:
    """Resample whole prompt groups, paired across arms by their frozen row."""
    require(len(original) == len(revised), 'Paired arms must cover the same prompts')
    n = len(original)
    draws = {metric: [] for metric in ('accuracy', 'distinct8', 'pcmd')}
    for _ in range(replicates):
        index = rng.integers(0, n, n)
        left, right = _cell([original[i] for i in index]), _cell([revised[i] for i in index])
        for metric in draws:
            if left[metric] is None or right[metric] is None:
                continue
            draws[metric].append(right[metric] - left[metric])
    out = {}
    for metric, values in draws.items():
        if len(values) < replicates * 0.95:
            # An interval built from replicates that mostly could not be defined
            # would understate its own uncertainty; it is withheld instead.
            out[metric] = {'estimate': None, 'ci95': None, 'defined_replicates': len(values)}
            continue
        array = np.asarray(values, dtype=float)
        out[metric] = {'estimate': None, 'ci95': [float(np.percentile(array, 2.5)),
                                                  float(np.percentile(array, 97.5))],
                       'defined_replicates': len(values)}
    return out


def analyze(run: Path, replicates: int, seed: int) -> dict:
    data = load(run)
    rng = np.random.default_rng(seed)
    cells, differences = {}, {}
    for grading in GRADINGS:
        records = (data['strict'] if grading == 'strict'
                   else [data['normalized'][r['sample_id']] for r in data['strict']])
        groups = _groups(records, 'canonical_key')
        for level in LEVELS:
            rows = {arm: [groups[(arm, level, index)] for index in range(PROMPTS)]
                    for arm in ARMS}
            for arm in ARMS:
                cells[f'{grading}/{arm}/level{level}'] = _cell(rows[arm])
            # Unrestricted difference: all 128 prompts, both arms.
            entry = {'all_prompts': _bootstrap(rows['original'], rows['revised'],
                                               replicates, rng)}
            # Matched-population difference: prompts where both arms clear two
            # verified draws, which is the only population PCMD can be paired on.
            joint = [index for index in range(PROMPTS)
                     if len(rows['original'][index]) >= 2 and len(rows['revised'][index]) >= 2]
            entry['jointly_eligible_prompts'] = len(joint)
            entry['jointly_eligible'] = (
                _bootstrap([rows['original'][i] for i in joint],
                           [rows['revised'][i] for i in joint], replicates, rng)
                if len(joint) >= 30 else None)
            for population in ('all_prompts', 'jointly_eligible'):
                if entry[population] is None:
                    continue
                source = (range(PROMPTS) if population == 'all_prompts' else joint)
                left = _cell([rows['original'][i] for i in source])
                right = _cell([rows['revised'][i] for i in source])
                for metric in ('accuracy', 'distinct8', 'pcmd'):
                    if left[metric] is not None and right[metric] is not None:
                        entry[population][metric]['estimate'] = right[metric] - left[metric]
            differences[f'{grading}/level{level}'] = entry

    return {
        'schema': SCHEMA, 'created_at_utc': now(),
        'analysis_source': {'path': str(Path(__file__).resolve().relative_to(ROOT)),
                            'sha256': file_sha(Path(__file__))},
        'run': {'directory': str(run.resolve().relative_to(ROOT)), 'sha256': data['bindings']},
        'deployment': {'model': data['manifest']['model'],
                       'served_model': data['served_model'],
                       'endpoint': data['manifest']['endpoint'],
                       'published_cells_provider': data['manifest']['published_cells_provider'],
                       'build_equality_with_published_cells': 'not established'},
        'population': {'domain': data['manifest']['domain'], 'levels': list(LEVELS),
                       'arms': list(ARMS), 'prompts_per_cell': PROMPTS,
                       'draws_per_prompt': DRAWS},
        'bootstrap': {'replicates': replicates, 'seed': seed,
                      'unit': 'Whole eight-draw prompt groups, paired across arms by frozen row.',
                      'interval': 'Pointwise percentile; no multiplicity adjustment; draws are '
                                  'not separately resampled.'},
        'native_refusals': refusals(data['raw'], data['strict']),
        'surface_taxonomy': surface_taxonomy(run, data['strict'], data['normalized']),
        'cells': cells,
        'differences': differences,
        'limitations': [
            'Task wording and system-message presence change together; neither is isolated.',
            'Both arms were collected here, after the published Azure cells, and replace none of them.',
            'Unrestricted PCMD differences compare different correct-pair populations; the '
            'jointly eligible reading is the matched-population contrast.',
            'A single deployment answers this condition; it does not generalize to the cohort.',
            'Most of the measured arm difference is a surface the frozen grader cannot parse; '
            'the reported drop is in scoreable breadth, not in demonstrated capability.',
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path,
                        default=DEFAULT_RUN / 'paired_wording_comparison.json')
    parser.add_argument('--replicates', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260918)
    args = parser.parse_args()
    record = analyze(args.run, args.replicates, args.seed)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output),
                      'served_model': record['deployment']['served_model'],
                      'native_refusals': record['native_refusals']}, indent=2))


if __name__ == '__main__':
    main()
