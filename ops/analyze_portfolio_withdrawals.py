#!/usr/bin/env python3
"""Score saved portfolios against the frozen withdrawals; no generation.

Every number comes from outcome keys already in the frozen container, read at
the same terminal endpoints the paper's PCMD curves report. Three views of the
same portfolios are produced together so none is selected afterwards:

``raw``
    Survival of the eight saved draws as they stand. This moves with correctness:
    a policy that verifies more often has more saved answers to survive with.
``budget``
    Survival when the number of verified draws is held fixed at 1, 2 or 4, as an
    exact expectation over subsets of the saved correct draws -- the estimator
    the hosted mixture analysis already uses. Correctness is then equalized by
    construction, so a gain at budget 2 or 4 that is absent at budget 1 is a
    property of how the portfolio spreads, not of how often it is right.
``distinct``
    Expected distinct outcomes at the same fixed budget, which is the breadth
    that survival is supposed to be buying.
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

from followup_metrics import atomic_new, file_sha, miss  # noqa: E402
import portfolio_withdrawals as pw  # noqa: E402

BASE = ROOT / 'artifacts/modebench_portfolio_withdrawals_20260917'
RETENTION = ROOT / 'paper/results/mode_diversity_retention_matrix.json'
PAIRS = (('replay_drgrpo', 'drgrpo', 'Re:Dr'), ('replay_maxrl', 'maxrl', 'Re:Max'))
#: ``pair`` is two binding withdrawals at once -- the severity check that asks
#: whether the effect is an artefact of the mildest possible change.
VARIANTS = ('feasible', 'binding', 'pair')
#: Panel A of the retention matrix keys Level 2 apart from Level 1 by scale.
PMD_SCALE = {('level1', 'qwen05b'): 'qwen05b', ('level1', 'falcon1b'): 'falcon1b',
             ('level1', 'qwen3b'): 'qwen3b', ('level2', 'qwen05b'): 'qwen05b_level2'}


def load_frozen():
    inputs = json.loads((BASE / 'inputs.json').read_text())
    if inputs.get('outcomes_read') is not False:
        raise ValueError('frozen inputs must record an outcome-blind freeze')
    payload = (BASE / 'table.json.gz').read_bytes()
    if hashlib.sha256(payload).hexdigest() != inputs['table_sha256']:
        raise ValueError('withdrawal table changed since the freeze')
    for name, expected in inputs['code_sha256'].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError('protocol code changed since the freeze: ' + name)
    if file_sha(Path(inputs['container'])) != inputs['container_sha256']:
        raise ValueError('outcome container changed since the freeze')
    table = json.loads(gzip.decompress(payload))
    frozen = {}
    for row in table['rows']:
        options = {'feasible': [o for o in row['options'] if o['feasible']],
                   'binding': [o for o in row['options'] if o['feasible'] and o['binding']]}
        frozen[(row['level'], row['domain'], row['prompt_index'])] = {
            'census': frozenset(row['support']),
            'options': {variant: [(tuple(o['option']), frozenset(o['surviving']))
                                  for o in chosen] for variant, chosen in options.items()},
            # A pair carries its own census survivors only where intersecting the
            # two singles would be wrong; elsewhere the analyzer intersects the
            # resolved single-withdrawal sets.
            'pairs': [{'options': [tuple(o) for o in p['options']],
                       'surviving': frozenset(p['surviving']) if 'surviving' in p else None}
                      for p in row['pairs'] if p['feasible'] and p['binding']]}
    return inputs, frozen


def survival_lookup(frozen, observed):
    """Decide each observed outcome against each withdrawal, once per prompt.

    Survival is asked of the key, not of census membership, so a verified outcome
    the generator's census never enumerated is still scored. Where the withdrawal
    is not intrinsic to the key the census answers instead, and a key it does not
    hold is undecidable -- counted and reported rather than treated as lost.
    """
    lookup, outside = {}, collections.Counter()
    undecidable = collections.Counter()
    for prompt, entry in frozen.items():
        level, domain, index = prompt
        keys = observed.get(prompt, frozenset())
        outside[domain] += sum(1 for key in keys if key not in entry['census'])
        variants, single = {}, {}
        for variant, options in entry['options'].items():
            resolved = []
            for option, census_surviving in options:
                surviving = set()
                for key in keys:
                    verdict = pw.survives(domain, option, key, census_surviving, entry['census'])
                    if verdict is None:
                        undecidable[domain] += 1
                    elif verdict:
                        surviving.add(key)
                resolved.append(frozenset(surviving))
                single[option] = frozenset(surviving)
            variants[variant] = resolved
        pairs = []
        for pair in entry['pairs']:
            left, right = pair['options']
            if pair['surviving'] is None:
                pairs.append(single[left] & single[right])
                continue
            surviving = set()
            for key in keys:
                if key not in entry['census']:
                    undecidable[domain] += 1
                elif key in pair['surviving']:
                    surviving.add(key)
            pairs.append(frozenset(surviving))
        variants['pair'] = pairs
        lookup[prompt] = variants
    if undecidable:
        raise ValueError('undecidable outcomes outside the census: ' + json.dumps(dict(undecidable)))
    return lookup, {'distinct_keys_outside_census': dict(outside),
                    'undecidable_outcomes': 0}


def portfolios(container):
    """Terminal-step verified keys per cell, prompt and block of eight draws."""
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    endpoints = {}
    with gzip.open(container, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('record_kind') != 'source':
                continue
            draws = record['draws']
            if not draws:
                continue
            cell = (record['level'], record['scale'], record['domain'], record['method'], record['seed'])
            terminal = max(draw['step'] for draw in draws)
            previous = endpoints.get(cell)
            if previous is not None:
                if previous['step'] > terminal:
                    continue
                if previous['step'] < terminal:
                    cells[cell].clear()
            endpoints[cell] = {'step': terminal, 'path': record['path'], 'run_dir': record['run_dir']}
            for draw in draws:
                if draw['step'] != terminal:
                    continue
                for index, keys in draw['prompts'].items():
                    cells[cell][int(index)].append(tuple(keys))
    return {cell: dict(prompts) for cell, prompts in cells.items()}, endpoints


def endpoint_parity(endpoints, curves_path):
    """Require the paper's own terminal endpoint for every cell that is scored."""
    curve = {}
    for entry in json.loads(Path(curves_path).read_text())['curves']:
        if entry['level'] not in ('level1', 'level2'):
            continue
        point = max(entry['points'], key=lambda p: p['step'])
        curve[(entry['level'], entry['scale'], entry['domain'], entry['method'], int(entry['seed']))] = point['step']
    missing = sorted(str(cell) for cell in endpoints if cell not in curve)
    differing = sorted(str(cell) for cell in endpoints if cell in curve and curve[cell] != endpoints[cell]['step'])
    if missing or differing:
        raise ValueError(f'endpoint parity failed: {len(missing)} unregistered, {len(differing)} differing')
    return {'cells': len(endpoints), 'curve_cells': len(curve), 'unregistered': 0, 'differing': 0}


def units(cell, prompts, lookup, budgets):
    """One record per (prompt, block): survival and breadth under each view."""
    level, scale, domain, method, seed = cell
    out = {}
    for index, blocks in prompts.items():
        variants = lookup[(level, domain, index)]
        if not variants['feasible']:
            continue
        for position, block in enumerate(blocks):
            keys = [key for key in block if key]
            drawn = len(keys)
            counts = collections.Counter(keys)
            record = {'correct': drawn, 'observed_modes': len(counts), 'raw': {}, 'budget': {}, 'distinct': {}}
            for variant in VARIANTS:
                options = variants[variant]
                if not options:
                    continue
                record['raw'][variant] = (statistics.fmean(
                    1.0 if any(key in surviving for key in keys) else 0.0 for surviving in options)
                    if drawn else 0.0)
                for budget in budgets:
                    if drawn < budget:
                        continue
                    record['budget'][f'{variant}/{budget}'] = statistics.fmean(
                        1.0 - miss(drawn, sum(1 for key in keys if key in surviving), budget)
                        for surviving in options)
            for budget in budgets:
                if drawn >= budget:
                    record['distinct'][budget] = sum(1.0 - miss(drawn, n, budget) for n in counts.values())
            out[(index, position)] = record
    return out


def paired(a, b, field, key):
    """Mean paired difference over the blocks both arms define for this field."""
    shared = [unit for unit in a if unit in b and key in a[unit][field] and key in b[unit][field]]
    if not shared:
        return None, 0
    return statistics.fmean(a[unit][field][key] - b[unit][field][key] for unit in shared), len(shared)


def bootstrap(by_prompt, replicates, seed):
    """Paired whole-prompt bootstrap: every block and seed of a prompt moves together."""
    ids = sorted(by_prompt)
    sums = np.array([by_prompt[i][0] for i in ids], dtype=float)
    counts = np.array([by_prompt[i][1] for i in ids], dtype=float)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(replicates, len(ids)))
    estimates = np.sort(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
    lo = float(estimates[int(math.floor(0.025 * replicates))])
    hi = float(estimates[int(math.ceil(0.975 * replicates)) - 1])
    return {'estimate': point, 'ci95': [lo, hi], 'prompts': len(ids),
            'blocks': int(counts.sum())}


def correlate(x, y):
    def pearson(a, b):
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
        return num / den if den else None

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        position = 0
        while position < len(order):
            stop = position
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[position]]:
                stop += 1
            shared = (position + stop) / 2
            for index in order[position:stop + 1]:
                out[index] = shared
            position = stop + 1
        return out

    return {'pearson': pearson(x, y), 'spearman': pearson(ranks(x), ranks(y)), 'n': len(x)}


def analyze(replicates=None, rewrite=False):
    if rewrite:
        (BASE / 'results.json').unlink(missing_ok=True)
    inputs, frozen = load_frozen()
    budgets = tuple(inputs['budgets'])
    started = time.time()
    cells, endpoints = portfolios(Path(inputs['container']))
    parity = endpoint_parity(endpoints, inputs['curve_archive'])
    observed = collections.defaultdict(set)
    for cell, prompts in cells.items():
        for index, blocks in prompts.items():
            observed[(cell[0], cell[2], index)].update(key for block in blocks for key in block if key)
    lookup, census_note = survival_lookup(frozen, observed)
    scored = {cell: units(cell, prompts, lookup, budgets) for cell, prompts in cells.items()}
    print(json.dumps({'stage': 'scored', 'cells': len(scored),
                      'seconds': round(time.time() - started, 1)}), flush=True)

    summary = {}
    for cell, records in scored.items():
        blocks = list(records.values())
        entry = {'blocks': len(blocks), 'prompts': len({unit[0] for unit in records}),
                 'pass8': statistics.fmean(1.0 if b['correct'] else 0.0 for b in blocks),
                 'correct': statistics.fmean(b['correct'] for b in blocks),
                 'observed_modes': statistics.fmean(b['observed_modes'] for b in blocks),
                 'raw': {}, 'budget': {}, 'distinct': {}}
        for variant in VARIANTS:
            values = [b['raw'][variant] for b in blocks if variant in b['raw']]
            if values:
                entry['raw'][variant] = statistics.fmean(values)
            for budget in budgets:
                key = f'{variant}/{budget}'
                values = [b['budget'][key] for b in blocks if key in b['budget']]
                if values:
                    entry['budget'][key] = {'estimate': statistics.fmean(values), 'blocks': len(values)}
        for budget in budgets:
            values = [b['distinct'][budget] for b in blocks if budget in b['distinct']]
            if values:
                entry['distinct'][budget] = {'estimate': statistics.fmean(values), 'blocks': len(values)}
        summary['/'.join(str(part) for part in cell)] = entry

    contrasts = []
    for level, scale, domain in sorted({(c[0], c[1], c[2]) for c in scored}):
        for replay, fresh, label in PAIRS:
            seeds = sorted({c[4] for c in scored if c[:4] == (level, scale, domain, replay)}
                           & {c[4] for c in scored if c[:4] == (level, scale, domain, fresh)})
            if not seeds:
                continue
            row = {'level': level, 'scale': scale, 'domain': domain, 'arm': label,
                   'seeds': seeds, 'raw': {}, 'budget': {}, 'distinct': {}}
            for variant in VARIANTS:
                values = [paired(scored[(level, scale, domain, replay, s)],
                                 scored[(level, scale, domain, fresh, s)], 'raw', variant) for s in seeds]
                values = [v for v, n in values if v is not None]
                if values:
                    row['raw'][variant] = statistics.fmean(values)
                for budget in budgets:
                    key = f'{variant}/{budget}'
                    got = [paired(scored[(level, scale, domain, replay, s)],
                                  scored[(level, scale, domain, fresh, s)], 'budget', key) for s in seeds]
                    kept = [(v, n) for v, n in got if v is not None]
                    if kept:
                        row['budget'][key] = {'estimate': statistics.fmean(v for v, _ in kept),
                                              'blocks': sum(n for _, n in kept), 'seeds': len(kept)}
            for budget in budgets:
                got = [paired(scored[(level, scale, domain, replay, s)],
                              scored[(level, scale, domain, fresh, s)], 'distinct', budget) for s in seeds]
                kept = [(v, n) for v, n in got if v is not None]
                if kept:
                    row['distinct'][budget] = {'estimate': statistics.fmean(v for v, _ in kept),
                                               'blocks': sum(n for _, n in kept)}
            contrasts.append(row)

    replicates = replicates or inputs['bootstrap']['replicates']
    seed = inputs['bootstrap']['seed']
    pooled = []
    for domain in pw.DOMAINS:
        for replay, fresh, label in PAIRS:
            entry = {'domain': domain, 'arm': label, 'budget': {}, 'distinct': {}, 'raw': {}}
            for field, keys in (('raw', VARIANTS),
                                ('budget', [f'{v}/{b}' for v in VARIANTS for b in budgets]),
                                ('distinct', list(budgets))):
                for key in keys:
                    by_prompt = collections.defaultdict(lambda: [0.0, 0])
                    for cell in scored:
                        if cell[2] != domain or cell[3] != replay:
                            continue
                        control = (cell[0], cell[1], domain, fresh, cell[4])
                        if control not in scored:
                            continue
                        a, b = scored[cell], scored[control]
                        for unit in a:
                            if unit not in b or key not in a[unit][field] or key not in b[unit][field]:
                                continue
                            bucket = by_prompt[(cell[0], cell[1], unit[0])]
                            bucket[0] += a[unit][field][key] - b[unit][field][key]
                            bucket[1] += 1
                    if by_prompt:
                        entry[field][str(key)] = bootstrap(by_prompt, replicates, seed)
            pooled.append(entry)

    primary = {(row['domain'], row['arm']): row for row in pooled}
    pmd = json.loads(RETENTION.read_text())['panel_a']
    points = []
    for row in contrasts:
        scale_key = PMD_SCALE.get((row['level'], row['scale']))
        value = ((pmd.get(scale_key) or {}).get(row['arm']) or {}).get(row['domain'], {}).get('mean')
        survival = row['budget'].get('binding/4', {}).get('estimate')
        if value is None or survival is None:
            continue
        points.append({'level': row['level'], 'scale': row['scale'], 'domain': row['domain'],
                       'arm': row['arm'], 'pmd_gain': value, 'survival_gain_budget4': survival})
    correlation = correlate([p['pmd_gain'] for p in points],
                            [p['survival_gain_budget4'] for p in points])

    results = {
        'schema': 'modebench-portfolio-withdrawal-results-v1',
        'inputs_sha256': file_sha(BASE / 'inputs.json'),
        'protocol': {k: inputs[k] for k in ('budgets', 'bootstrap', 'withdrawal', 'certificate',
                                            'design_disclosure', 'cells', 'options',
                                            'feasible_options', 'binding_options',
                                            'pairs', 'feasible_pairs', 'binding_pairs',
                                            'pair_by_intersection')},
        'endpoint_parity': parity,
        'census_coverage': census_note,
        'inventory': inventory(),
        'cells': summary, 'contrasts': contrasts, 'pooled': pooled,
        'pmd_agreement': {'points': points, 'correlation': correlation},
        'headline': headline(contrasts, primary, budgets),
        'bootstrap_replicates': replicates,
    }
    atomic_new(BASE / 'results.json', results)
    print(json.dumps({'stage': 'complete', 'contrasts': len(contrasts), 'pooled': len(pooled),
                      'pmd_points': len(points), 'pearson': round(correlation['pearson'], 3),
                      'seconds': round(time.time() - started, 1)}), flush=True)
    return results


def inventory():
    """Per level and domain: prompts, support, and how binding the options are."""
    out = {}
    payload = json.loads(gzip.decompress((BASE / 'table.json.gz').read_bytes()))
    for row in payload['rows']:
        entry = out.setdefault(row['level'] + '|' + row['domain'],
                               {'prompts': 0, 'support': [], 'options': 0, 'feasible': 0,
                                'binding': 0, 'surviving_fraction': [], 'pairs': 0,
                                'binding_pairs': 0, 'pair_surviving_fraction': []})
        entry['prompts'] += 1
        entry['support'].append(len(row['support']))
        entry['options'] += len(row['options'])
        for option in row['options']:
            entry['feasible'] += int(option['feasible'])
            entry['binding'] += int(option['binding'] and option['feasible'])
            if option['feasible'] and option['binding']:
                entry['surviving_fraction'].append(len(option['surviving']) / len(row['support']))
        for pair in row['pairs']:
            entry['pairs'] += 1
            if pair['feasible'] and pair['binding']:
                entry['binding_pairs'] += 1
                entry['pair_surviving_fraction'].append(pair['surviving_count'] / len(row['support']))
    for entry in out.values():
        entry['mean_support'] = statistics.fmean(entry['support'])
        entry['mean_surviving_fraction'] = (statistics.fmean(entry['surviving_fraction'])
                                            if entry['surviving_fraction'] else None)
        entry['mean_pair_surviving_fraction'] = (statistics.fmean(entry['pair_surviving_fraction'])
                                                 if entry['pair_surviving_fraction'] else None)
        del entry['support'], entry['surviving_fraction'], entry['pair_surviving_fraction']
    return out


def headline(contrasts, primary, budgets):
    """The counts and ranges the paper states, computed once here."""
    out = {}
    for variant in VARIANTS:
        for budget in list(budgets) + ['raw']:
            if budget == 'raw':
                values = [row['raw'][variant] for row in contrasts if variant in row['raw']]
            else:
                key = f'{variant}/{budget}'
                values = [row['budget'][key]['estimate'] for row in contrasts if key in row['budget']]
            out[f'cells/{variant}/{budget}'] = {
                'cells': len(values), 'positive': sum(v > 0 for v in values),
                'mean': statistics.fmean(values) if values else None}
    positive = [row for row in primary.values()
                if (row['budget'].get('binding/4') or {}).get('ci95', [0, 0])[0] > 0]
    out['domains'] = len({row['domain'] for row in primary.values()})
    out['pooled_positive_ci_budget4'] = len(positive)
    out['pooled_domains_positive_ci_budget4'] = sorted({row['domain'] for row in positive})
    # A positive interval is not yet a material effect: MathIR clears zero by a
    # tenth of a point. The paper therefore states magnitudes, so the per-domain
    # best-arm estimate at each budget is recorded here and read from the record.
    out['per_domain'] = {}
    for domain in pw.DOMAINS:
        entry = {}
        for budget in budgets:
            arms = [primary[(domain, label)]['budget'].get(f'binding/{budget}')
                    for _, _, label in PAIRS if (domain, label) in primary]
            arms = [arm for arm in arms if arm]
            if arms:
                best = max(arms, key=lambda a: a['estimate'])
                entry[str(budget)] = {'best': best,
                                      'min': min(a['estimate'] for a in arms),
                                      'max': max(a['estimate'] for a in arms),
                                      'arms_positive_ci': sum(a['ci95'][0] > 0 for a in arms)}
        distinct = primary[(domain, 'Re:Dr')]['distinct'].get('4')
        entry['distinct4'] = distinct
        out['per_domain'][domain] = entry
    # ``broad`` is the set of domains whose portfolios are measurably broader at a
    # fixed budget; it is decided by the breadth interval, not by survival, so the
    # survival range the paper quotes is not selected on survival.
    broad = [domain for domain in pw.DOMAINS
             if (out['per_domain'][domain].get('distinct4') or {}).get('ci95', [0, 0])[0] > 0.05]
    out['broad_domains'] = broad
    out['broad_budget_range'] = {
        str(budget): {'min': min(out['per_domain'][d][str(budget)]['min'] for d in broad),
                      'max': max(out['per_domain'][d][str(budget)]['max'] for d in broad)}
        for budget in budgets if all(str(budget) in out['per_domain'][d] for d in broad)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=None)
    parser.add_argument('--rewrite', action='store_true')
    analyze(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
