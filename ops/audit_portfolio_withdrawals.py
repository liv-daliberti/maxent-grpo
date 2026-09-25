#!/usr/bin/env python3
"""Independently re-derive the withdrawal protocol's claims.

Each check reaches its target by a different route than the analyzer does:
supports are re-enumerated or re-verified from the frozen validators rather than
from the enumerator that built the table, the fixed-budget expectation is matched
against exhaustive subset enumeration, and every outcome key the cohort actually
produced is required to lie inside the frozen support it is scored against.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / 'ops', ROOT / 'src'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from followup_metrics import atomic_new, file_sha, miss, sha  # noqa: E402
import portfolio_withdrawals as pw  # noqa: E402
from analyze_portfolio_withdrawals import endpoint_parity, portfolios  # noqa: E402
from oat_drgrpo.math_grader import _verify_graph_coloring_colors  # noqa: E402
from oat_drgrpo.mathir import validate_mathir_action_menu  # noqa: E402
from oat_drgrpo.pantry_plan import validate_pantry_plan  # noqa: E402
from prepare_portfolio_withdrawals import first_line_specs  # noqa: E402

BASE = ROOT / 'artifacts/modebench_portfolio_withdrawals_20260917'
SAMPLE_PROMPTS = 6
SAMPLE_FILES = 24
SAMPLE_UNITS = 240


def frozen_table():
    inputs = json.loads((BASE / 'inputs.json').read_text())
    payload = (BASE / 'table.json.gz').read_bytes()
    if hashlib.sha256(payload).hexdigest() != inputs['table_sha256']:
        raise ValueError('withdrawal table changed since the freeze')
    return inputs, json.loads(gzip.decompress(payload))['rows']


def pick(items, count, tag, identity=str):
    """A deterministic sample. ``identity`` must not depend on set ordering."""
    return sorted(items, key=lambda item: sha([tag, identity(item)]))[:count]


def check_support(domain, spec, support):
    """Re-verify a frozen support without using the enumerator that built it."""
    keys = set(support)
    if domain == 'graph_coloring':
        # A second enumeration: all 3^n colourings, screened by the frozen
        # verifier, which also re-checks the prompt's fixed digits.
        n = int(spec['n'])
        rebuilt = {'graph_coloring:' + ''.join(str(c) for c in colours)
                   for colours in itertools.product(pw.GRAPH_COLORS, repeat=n)
                   if _verify_graph_coloring_colors(list(colours), spec)}
        return rebuilt == keys, len(rebuilt)
    if domain == 'countdown':
        # Each key is re-executed here, independently of the grader that produced
        # it, and checked for the exact target and operand multiset.
        ok = all(_countdown_functional_ok(key, spec) for key in keys)
        return ok and len(keys) == int(spec['num_expressions']), len(keys)
    if domain == 'python_factors':
        cases = [int(case) for case in spec['cases']]
        ok = True
        for key in keys:
            vector = [int(d) for d in key.split(':', 1)[1].split(',')]
            ok = ok and len(vector) == len(cases) and all(1 < d < n and n % d == 0
                                                          for d, n in zip(vector, cases))
        expected = math.prod(len([d for d in range(2, n) if n % d == 0]) for n in cases)
        return ok and len(keys) == expected, len(keys)
    if domain == 'mathir':
        # Execute explicit semicolon-separated action programs directly; every
        # frozen key must be reachable and nothing outside it may appear.
        actions = sorted(spec['actions'])
        produced = set()
        for length in range(1, int(spec['max_steps']) + 1):
            for program in itertools.product(actions, repeat=length):
                validation = validate_mathir_action_menu(';'.join(program), spec)
                if validation is not None:
                    produced.add(validation.canonical_key)
        return produced == keys, len(produced)
    if domain == 'pantry_plan':
        witnesses = pw.pantry_support(spec)
        ok = set(witnesses) == keys and all(validate_pantry_plan(candidate, spec) is not None
                                            for candidate in witnesses.values())
        return ok, len(witnesses)
    raise ValueError('unsupported domain ' + domain)


def _countdown_functional_ok(key, spec):
    """Evaluate the canonical functional key against the task directly."""
    from fractions import Fraction
    import ast

    def walk(node):
        if isinstance(node, ast.Constant):
            return Fraction(int(node.value)), [int(node.value)]
        name = node.func.id
        parts = [walk(argument) for argument in node.args]
        values = [value for value, _ in parts]
        numbers = [n for _, used in parts for n in used]
        if name == 'add':
            return values[0] + values[1], numbers
        if name == 'sub':
            return values[0] - values[1], numbers
        if name == 'mul':
            return values[0] * values[1], numbers
        if name == 'div':
            return values[0] / values[1], numbers
        if name == 'neg':
            return -values[0], numbers
        raise ValueError('unsupported operator')

    value, numbers = walk(ast.parse(key.split(':', 1)[1], mode='eval').body)
    return value == Fraction(int(spec['target'])) and \
        collections.Counter(numbers) == collections.Counter(int(v) for v in spec['numbers'])


def exhaustive_budget(keys, surviving, budget):
    """Survival probability by enumerating every subset of the saved draws."""
    subsets = list(itertools.combinations(range(len(keys)), budget))
    hit = sum(1 for subset in subsets if any(keys[i] in surviving for i in subset))
    return hit / len(subsets)


def audit():
    inputs, rows = frozen_table()
    report = {'schema': 'modebench-portfolio-withdrawal-audit-v1',
              'inputs_sha256': file_sha(BASE / 'inputs.json'), 'checks': {}}
    specs = {}
    for key, identity in inputs['prompt_identity'].items():
        specs[key] = first_line_specs(identity['spec_source'])

    # 1. Structural invariants on every frozen row.
    structural = {'rows': 0, 'options': 0, 'certified_reproduced': 0}
    for row in rows:
        support = set(row['support'])
        structural['rows'] += 1
        structural['certified_reproduced'] += int(row['certified_match'] and
                                                  row['certified'] == len(support))
        for option in row['options']:
            surviving = set(option['surviving'])
            if not surviving <= support:
                raise ValueError('a withdrawal produced an outcome outside the original support')
            if option['feasible'] != bool(surviving):
                raise ValueError('feasibility disagrees with the surviving set')
            if option['binding'] != (len(surviving) < len(support)):
                raise ValueError('binding flag disagrees with the surviving set')
            structural['options'] += 1
    if structural['certified_reproduced'] != structural['rows']:
        raise ValueError('a frozen support does not equal its certified mode count')
    report['checks']['structural'] = structural

    # 1b. The same invariants for pair withdrawals, plus the rule that decides
    # whether a pair may be recovered by intersecting its two singles.
    pairs = {'pairs': 0, 'binding': 0, 'exact_by_intersection': 0, 'strictly_smaller': 0,
             'by_domain': {}}
    for row in rows:
        support = set(row['support'])
        singles = {tuple(o['option']): set(o['surviving']) for o in row['options']}
        for pair in row['pairs']:
            left, right = (tuple(o) for o in pair['options'])
            intersection = singles[left] & singles[right]
            count = pair['surviving_count']
            if count > len(intersection):
                raise ValueError('a pair kept more outcomes than either single withdrawal')
            if pair['feasible'] != (count > 0) or pair['binding'] != (count < len(support)):
                raise ValueError('pair flags disagree with its surviving count')
            exact = pair['exact_by_intersection']
            if exact != (count == len(intersection)):
                raise ValueError('the intersection flag disagrees with the surviving count')
            if row['domain'] in pw.PAIR_BY_INTERSECTION and not exact:
                raise ValueError('pair survival is not the intersection for ' + row['domain'])
            if 'surviving' in pair and not set(pair['surviving']) <= intersection:
                raise ValueError('a stored pair survivor is outside the intersection')
            pairs['pairs'] += 1
            pairs['binding'] += int(pair['feasible'] and pair['binding'])
            pairs['exact_by_intersection'] += int(exact)
            pairs['strictly_smaller'] += int(not exact)
            counts = pairs['by_domain'].setdefault(row['domain'], {'pairs': 0, 'strictly_smaller': 0})
            counts['pairs'] += 1
            counts['strictly_smaller'] += int(not exact)
    report['checks']['pairs'] = pairs

    # 2. Re-derive a deterministic sample of supports by an independent route.
    resupport = {'checked': 0, 'domains': {}}
    index = {(row['level'], row['domain'], row['prompt_index']): row for row in rows}
    for key, references in sorted(specs.items()):
        level, domain = key.split('|')
        for prompt in pick(range(len(references)), SAMPLE_PROMPTS, 'withdrawal-audit-support'):
            spec = json.loads(references[prompt])
            row = index[(level, domain, prompt)]
            agreed, count = check_support(domain, spec, row['support'])
            if not agreed:
                raise ValueError(f'independent support re-derivation differs: {key} prompt {prompt}')
            resupport['checked'] += 1
            resupport['domains'].setdefault(domain, 0)
            resupport['domains'][domain] += 1
    report['checks']['independent_support'] = resupport

    # 2b. Re-decide a deterministic sample of pairs by an independent route: the
    # key predicate applied twice where the withdrawal is intrinsic, and an
    # explicit action program avoiding both actions for MathIR.
    repair = {'pairs': 0, 'by_domain': {}}
    for key, references in sorted(specs.items()):
        level, domain = key.split('|')
        for prompt in pick(range(len(references)), SAMPLE_PROMPTS, 'withdrawal-audit-pairs'):
            spec = json.loads(references[prompt])
            row = index[(level, domain, prompt)]
            support = set(row['support'])
            chosen = pick([p for p in row['pairs'] if p['feasible']], 4,
                          'withdrawal-audit-pair-choice',
                          lambda pair: json.dumps(pair['options'], sort_keys=True))
            for pair in chosen:
                left, right = (tuple(o) for o in pair['options'])
                if domain in pw.KEY_INTRINSIC:
                    rebuilt = {k for k in support
                               if pw.survives(domain, left, k, frozenset(), support)
                               and pw.survives(domain, right, k, frozenset(), support)}
                elif domain == 'pantry_plan':
                    revised = pw.pantry_revised_spec(pw.pantry_revised_spec(spec, left[1]), right[1])
                    rebuilt = {k for k, candidate in pw.pantry_support(spec).items()
                               if validate_pantry_plan(candidate, revised) is not None}
                else:
                    kept = [a for a in sorted(spec['actions']) if a not in (left[1], right[1])]
                    rebuilt = set()
                    for length in range(1, int(spec['max_steps']) + 1):
                        for program in itertools.product(kept, repeat=length):
                            validation = validate_mathir_action_menu(';'.join(program), spec)
                            if validation is not None and validation.canonical_key in support:
                                rebuilt.add(validation.canonical_key)
                if len(rebuilt) != pair['surviving_count']:
                    raise ValueError(f'independent pair re-derivation differs: {key} prompt {prompt}')
                if 'surviving' in pair and rebuilt != set(pair['surviving']):
                    raise ValueError(f'stored pair survivors differ: {key} prompt {prompt}')
                repair['pairs'] += 1
                repair['by_domain'][domain] = repair['by_domain'].get(domain, 0) + 1
    report['checks']['independent_pairs'] = repair

    # 3. Every outcome key the scored cohort produced lies in its frozen support.
    cells, endpoints = portfolios(Path(inputs['container']))
    report['checks']['endpoint_parity'] = endpoint_parity(endpoints, inputs['curve_archive'])
    # Every verified outcome must be decidable against every withdrawal it is
    # scored under. A key-intrinsic domain decides its own; the other two must
    # have the key in their census, which is what makes their scores complete.
    membership = {'keys': 0, 'outside_census': 0, 'undecidable': 0, 'by_domain': {}}
    for cell, prompts in cells.items():
        level, scale, domain, method, seed = cell
        for prompt, blocks in prompts.items():
            row = index[(level, domain, prompt)]
            support = frozenset(row['support'])
            options = [(tuple(o['option']), frozenset(o['surviving'])) for o in row['options']
                       if o['feasible']]
            for block in blocks:
                for observed in block:
                    if not observed:
                        continue
                    membership['keys'] += 1
                    inside = observed in support
                    membership['outside_census'] += int(not inside)
                    counts = membership['by_domain'].setdefault(domain, {'keys': 0, 'outside_census': 0})
                    counts['keys'] += 1
                    counts['outside_census'] += int(not inside)
                    if inside:
                        continue
                    for option, surviving in options:
                        if pw.survives(domain, option, observed, surviving, support) is None:
                            membership['undecidable'] += 1
    if membership['undecidable']:
        raise ValueError(f"{membership['undecidable']} verified outcomes cannot be decided")
    for domain, counts in membership['by_domain'].items():
        if domain not in pw.KEY_INTRINSIC and counts['outside_census']:
            raise ValueError('a census-decided domain produced an outcome outside its census: ' + domain)
    report['checks']['key_membership'] = membership

    # 4. Match the fixed-budget expectation against exhaustive subset enumeration.
    units = []
    for cell in pick(cells, 40, 'withdrawal-audit-cells'):
        level, scale, domain, method, seed = cell
        for prompt, blocks in sorted(cells[cell].items()):
            options = [sorted(o['surviving']) for o in index[(level, domain, prompt)]['options']
                       if o['feasible'] and o['binding']]
            if not options:
                continue
            for position, block in enumerate(blocks):
                keys = [key for key in block if key]
                if len(keys) >= 2:
                    units.append({'cell': cell, 'prompt': prompt, 'position': position,
                                  'keys': keys, 'options': [frozenset(o) for o in options]})
    # The sample is keyed by the unit's identity rather than its contents, because
    # a set's string form varies between processes and would resample the audit.
    units = pick(units, SAMPLE_UNITS, 'withdrawal-audit-units',
                 lambda unit: '/'.join(str(part) for part in
                                       (*unit['cell'], unit['prompt'], unit['position'])))
    exhaustive = {'units': 0, 'comparisons': 0, 'max_absolute_difference': 0.0}
    for unit in units:
        keys, options = unit['keys'], unit['options']
        for budget in inputs['budgets']:
            if len(keys) < budget:
                continue
            for surviving in options:
                formula = 1.0 - miss(len(keys), sum(1 for key in keys if key in surviving), budget)
                brute = exhaustive_budget(keys, surviving, budget)
                exhaustive['comparisons'] += 1
                exhaustive['max_absolute_difference'] = max(exhaustive['max_absolute_difference'],
                                                            abs(formula - brute))
        exhaustive['units'] += 1
    if exhaustive['max_absolute_difference'] > 1e-12:
        raise ValueError('fixed-budget expectation disagrees with exhaustive enumeration')
    report['checks']['exhaustive_budget'] = exhaustive

    # 5. Budget survival is monotone in the budget for every sampled unit.
    monotone = {'units': 0, 'violations': 0}
    for unit in units:
        keys, options = unit['keys'], unit['options']
        curve = []
        for budget in sorted(inputs['budgets']):
            if len(keys) < budget:
                continue
            curve.append(statistics.fmean(1.0 - miss(len(keys), sum(1 for key in keys if key in surviving), budget)
                                          for surviving in options))
        monotone['violations'] += sum(1 for a, b in zip(curve, curve[1:]) if b < a - 1e-12)
        monotone['units'] += 1
    if monotone['violations']:
        raise ValueError('budget survival is not monotone in the budget')
    report['checks']['monotone_budget'] = monotone

    # 6. Re-read a sample of source files and re-check the prompt-spec identity.
    files = pick([path for cell in endpoints.values() for path in [cell['path']]], SAMPLE_FILES,
                 'withdrawal-audit-files')
    identity = {'files': 0, 'mismatched': 0}
    for path in files:
        cell = next(key for key, value in endpoints.items() if value['path'] == path)
        expected = inputs['prompt_identity'][cell[0] + '|' + cell[2]]['references_sha256']
        identity['files'] += 1
        identity['mismatched'] += int(sha(first_line_specs(path)) != expected)
    if identity['mismatched']:
        raise ValueError('evaluation prompts differ between cells of one level and domain')
    report['checks']['prompt_identity'] = identity

    # 7. Record what the published analysis claims, beside the checks above.
    results = json.loads((BASE / 'results.json').read_text())
    points = {(p['domain'], p['arm']): p for p in results['pooled']}
    report['checks']['published_results'] = {
        'contrasts': len(results['contrasts']), 'pooled': len(points),
        'endpoint_parity': results['endpoint_parity'],
        'pmd_pearson': results['pmd_agreement']['correlation']['pearson'],
        'bootstrap_replicates': results['bootstrap_replicates']}
    atomic_new(BASE / 'independent_audit.json', report)
    print(json.dumps({'status': 'passed', **{k: (v if not isinstance(v, dict) else
                                                 {j: v[j] for j in list(v)[:3]})
                                             for k, v in report['checks'].items()}}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    audit()


if __name__ == '__main__':
    main()
