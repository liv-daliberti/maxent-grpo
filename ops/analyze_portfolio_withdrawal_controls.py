#!/usr/bin/env python3
"""Score the decoding grid against the frozen withdrawals: an inference-time baseline.

App.~\\ref{app:pantry-adaptation} answers "could you just decode differently?"
in PantryPlan alone, because its controls needed new generation. The E72 grid
answers the same question in all five domains with none: it already holds, for a
control and a replay arm at five seeds each, the same 128 evaluation prompts
sampled under eleven decoding settings -- six temperatures, a wider sampling
budget and two nucleus settings. Those prompts are hash-identical to the frozen
Level-1 prompts, so the withdrawal protocol applies unchanged and only the saved
outcomes differ.

The comparison is deliberately generous to the baseline. The control is credited
with the best setting in the whole grid, chosen on its own survival, while the
replay arm stays at the reference setting the paper reports. A gain that
survives that is not a decoding artefact.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))

from build_paper_decoding_objection import (  # noqa: E402
    ARMS, MANIFEST, REFERENCE_SETTING, SEEDS, SETTINGS, _draw_file)
from followup_metrics import atomic_new, file_sha, miss, sha  # noqa: E402
import portfolio_withdrawals as pw  # noqa: E402
from prepare_portfolio_withdrawals import first_line_specs  # noqa: E402

FROZEN = ROOT / 'artifacts/modebench_portfolio_withdrawals_20260917'
BASE = ROOT / 'artifacts/modebench_portfolio_withdrawal_controls_20260917'
CODE = ('ops/analyze_portfolio_withdrawal_controls.py', 'ops/portfolio_withdrawals.py',
        'ops/build_paper_decoding_objection.py')
SAMPLED = 'fixed_seed_sampled_k_neutral'
BUDGETS = (1, 2, 4)
REPLICATES = 20000
SEED = 20260917


def frozen_options():
    """Binding, feasible withdrawals per Level-1 prompt, from the frozen table."""
    inputs = json.loads((FROZEN / 'inputs.json').read_text())
    payload = (FROZEN / 'table.json.gz').read_bytes()
    if hashlib.sha256(payload).hexdigest() != inputs['table_sha256']:
        raise ValueError('frozen withdrawal table changed')
    table = json.loads(gzip.decompress(payload))
    options, census = {}, {}
    for row in table['rows']:
        if row['level'] != 'level1':
            continue
        key = (row['domain'], row['prompt_index'])
        census[key] = frozenset(row['support'])
        options[key] = [(tuple(o['option']), frozenset(o['surviving'])) for o in row['options']
                        if o['feasible'] and o['binding']]
    return inputs, options, census


def survival(path, domain, options, census, undecidable):
    """Per prompt and block: survival at each budget, from one decoding cell."""
    blocks = collections.defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # The greedy trace shares the file and is not one of the sampled draws.
        if record.get('evaluation_kind') != SAMPLED:
            continue
        for prompt in record['prompts']:
            verified = [key for key, reward in zip(prompt['answer_keys'], prompt['rewards'])
                        if key and reward and reward > 0]
            blocks[prompt['prompt_index']].append(verified)
    out = {}
    for index, drawn in blocks.items():
        chosen = options.get((domain, index))
        if not chosen:
            continue
        for position, keys in enumerate(drawn):
            record = {'drawn': len(keys), 'budget': {}}
            for budget in BUDGETS:
                if len(keys) < budget:
                    continue
                values = []
                for option, surviving in chosen:
                    inside = 0
                    for key in keys:
                        verdict = pw.survives(domain, option, key, surviving, census[(domain, index)])
                        if verdict is None:
                            undecidable[domain] += 1
                        elif verdict:
                            inside += 1
                    values.append(1.0 - miss(len(keys), inside, budget))
                record['budget'][budget] = statistics.fmean(values)
            out[(index, position)] = record
    return out


def bootstrap(by_prompt, replicates=REPLICATES, seed=SEED):
    ids = sorted(by_prompt)
    sums = np.array([by_prompt[i][0] for i in ids], dtype=float)
    counts = np.array([by_prompt[i][1] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
    estimates = np.sort(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
    return {'estimate': float(sums.sum() / counts.sum()),
            'ci95': [float(estimates[int(math.floor(0.025 * replicates))]),
                     float(estimates[int(math.ceil(0.975 * replicates)) - 1])],
            'prompts': len(ids), 'blocks': int(counts.sum())}


def paired(left, right, budget):
    """Prompt-clustered sums for a paired difference at one budget."""
    by_prompt = collections.defaultdict(lambda: [0.0, 0])
    for unit, record in left.items():
        other = right.get(unit)
        if other is None or budget not in record['budget'] or budget not in other['budget']:
            continue
        bucket = by_prompt[unit[0]]
        bucket[0] += record['budget'][budget] - other['budget'][budget]
        bucket[1] += 1
    return by_prompt


def analyze(rewrite):
    if rewrite:
        (BASE / 'results.json').unlink(missing_ok=True)
    inputs, options, census = frozen_options()
    manifest = json.loads(MANIFEST.read_text())
    started = time.time()
    identity = {}
    for domain in pw.DOMAINS:
        path = _draw_file(REFERENCE_SETTING[0], domain, 'drgrpo', SEEDS[0], REFERENCE_SETTING[1])
        if path is None:
            raise ValueError('reference decoding cell is absent for ' + domain)
        digest = sha(first_line_specs(path))
        expected = inputs['prompt_identity']['level1|' + domain]['references_sha256']
        if digest != expected:
            raise ValueError('decoding grid prompts differ from the frozen Level-1 prompts: ' + domain)
        identity[domain] = digest

    undecidable = collections.Counter()
    scored, sources = {}, {}
    for domain in pw.DOMAINS:
        for arm in ARMS:
            for stage, settings in SETTINGS.items():
                for setting, temperature, top_p, budget_k in settings:
                    for seed in SEEDS:
                        path = _draw_file(stage, domain, arm, seed, setting)
                        if path is None:
                            continue
                        cell = (domain, arm, setting, seed)
                        scored[cell] = survival(path, domain, options, census, undecidable)
                        sources[' / '.join(str(part) for part in cell)] = {
                            'path': str(path.relative_to(ROOT)), 'sha256': file_sha(path),
                            'temperature': temperature, 'top_p': top_p, 'k': budget_k, 'stage': stage}
    if undecidable:
        raise ValueError('undecidable outcomes: ' + json.dumps(dict(undecidable)))
    print(json.dumps({'stage': 'scored', 'cells': len(scored),
                      'seconds': round(time.time() - started, 1)}), flush=True)

    grid = []
    for domain in pw.DOMAINS:
        for arm in ARMS:
            for stage, settings in SETTINGS.items():
                for setting, temperature, top_p, budget_k in settings:
                    seeds = [seed for seed in SEEDS if (domain, arm, setting, seed) in scored]
                    if not seeds:
                        continue
                    entry = {'domain': domain, 'arm': ARMS[arm], 'setting': setting, 'stage': stage,
                             'temperature': temperature, 'top_p': top_p, 'k': budget_k,
                             'seeds': seeds, 'budget': {}}
                    for budget in BUDGETS:
                        values = [statistics.fmean(r['budget'][budget] for r in scored[(domain, arm, setting, seed)].values()
                                                   if budget in r['budget'])
                                  for seed in seeds
                                  if any(budget in r['budget'] for r in scored[(domain, arm, setting, seed)].values())]
                        if values:
                            entry['budget'][str(budget)] = statistics.fmean(values)
                    grid.append(entry)

    reference = REFERENCE_SETTING[1]
    contrasts = []
    for domain in pw.DOMAINS:
        control = [e for e in grid if e['domain'] == domain and e['arm'] == 'control'
                   and '4' in e['budget']]
        best = max(control, key=lambda e: e['budget']['4'])
        at_reference = next(e for e in control if e['setting'] == reference)
        replay = next(e for e in grid if e['domain'] == domain and e['arm'] == 'replay'
                      and e['setting'] == reference)
        row = {'domain': domain, 'control_best_setting': best['setting'],
               'control_best_temperature': best['temperature'], 'control_best_k': best['k'],
               'control_grid_range': {str(b): [min(e['budget'][str(b)] for e in control if str(b) in e['budget']),
                                               max(e['budget'][str(b)] for e in control if str(b) in e['budget'])]
                                      for b in BUDGETS},
               'control_reference': at_reference['budget'], 'control_best': best['budget'],
               'replay_reference': replay['budget'], 'gap_vs_best': {}, 'gap_vs_reference': {}}
        for budget in BUDGETS:
            for label, against in (('gap_vs_best', best['setting']),
                                   ('gap_vs_reference', reference)):
                by_prompt = collections.defaultdict(lambda: [0.0, 0])
                for seed in SEEDS:
                    left = scored.get((domain, 'xgrpo', reference, seed))
                    right = scored.get((domain, 'drgrpo', against, seed))
                    if not left or not right:
                        continue
                    for prompt, bucket in paired(left, right, budget).items():
                        target = by_prompt[prompt]
                        target[0] += bucket[0]
                        target[1] += bucket[1]
                if by_prompt:
                    row[label][str(budget)] = bootstrap(by_prompt)
        contrasts.append(row)

    results = {
        'schema': 'modebench-portfolio-withdrawal-controls-v1',
        'frozen_inputs_sha256': file_sha(FROZEN / 'inputs.json'),
        'frozen_table_sha256': inputs['table_sha256'],
        'decoding_manifest': {'path': str(MANIFEST.relative_to(ROOT)), 'sha256': file_sha(MANIFEST),
                              'model_tag': manifest['model_tag'], 'arms': manifest['arms'],
                              'resolved_runs': manifest['resolved_runs']},
        'cohort': 'E72 terminal checkpoints at step 4,609; the replay arm predates the Re:Dr '
                  'objective contract, and this is a separate cohort from the reported factorial.',
        'prompt_identity': identity, 'reference_setting': reference,
        'settings': {stage: [list(s) for s in settings] for stage, settings in SETTINGS.items()},
        'budgets': list(BUDGETS), 'bootstrap': {'replicates': REPLICATES, 'seed': SEED},
        'baseline_rule': 'The control is credited with the best whole-grid setting by its own '
                         'survival at four verified draws; the replay arm stays at the reference '
                         'setting. Selection favours the baseline.',
        'code_sha256': {name: file_sha(ROOT / name) for name in CODE},
        'grid': grid, 'contrasts': contrasts, 'sources': sources,
    }
    atomic_new(BASE / 'results.json', results)
    print(json.dumps({'stage': 'complete', 'grid_points': len(grid), 'domains': len(contrasts),
                      'seconds': round(time.time() - started, 1)}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rewrite', action='store_true')
    analyze(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
