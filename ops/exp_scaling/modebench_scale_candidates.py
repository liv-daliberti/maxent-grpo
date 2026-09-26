"""Prospective exact-support candidate laws for Qwen 7B/14B calibration.

The four tiers are structural hypotheses, not calibrated model levels. Both
scale campaigns may fit mixtures of these same laws using fresh, disjoint
development pools. Requested quotas only stop independent cell streams; they
never tune the proposal law. Existing prompts and executable verifiers remain
the source of truth.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))

from modebench_level3_discrete import (
    _countdown_prompt, _graph_prompt, graph_completion_count, row_identity,
)
from modebench_level3_countdown_v2 import _target_statistics
import modebench_level3_constraints as constraints
import modebench_level3_python_v2 as python_broad
import modebench_level3_python_v3 as python_small

SCHEMA = 'modebench_scale_candidate_laws_v1'
DOMAINS = ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')
COUNTDOWN_PRESETS = ((2, 18, 72), (5, 64, None), (8, 99, None), (16, 192, None))
GRAPH_PRESETS = ((6, 4), (8, 5), (10, 6), (12, 7))
PYTHON_ROUTES = ((python_small, 0), (python_small, 2), (python_broad, 2), (python_broad, 3))
MAX_PROPOSALS_PER_ROW = 100_000
MAX_PANTRY_PROPOSALS_PER_ROW = 1000
PROFILES = {
    'countdown': [
        {'operands': 4, 'minimum': lo, 'maximum': hi, 'target_cap': cap,
         'target_weight': 'shallow_products_small_targets' if tier == 0 else 'division_product_log_target'}
        for tier, (lo, hi, cap) in enumerate(COUNTDOWN_PRESETS)
    ],
    'graph_coloring': [
        {'vertices': n, 'hidden_vertices': hidden, 'planted_colors': 3,
         'edge_probability_interval': [0.32, 0.65], 'isolated_hidden_vertices': False}
        for n, hidden in GRAPH_PRESETS
    ],
    'python_factors': [
        {'origin_module': module.__name__, 'origin_tier': tier, 'cases': 4,
         'case_limit': 1000, 'sampling': module.PROFILE}
        for module, tier in PYTHON_ROUTES
    ],
    'mathir': [
        {'origin_module': constraints.__name__, 'origin_tier': tier,
         'family': constraints._mathir_family(tier).name,
         'actions': 6, 'max_steps': 4, 'canonical_modes': 5}
        for tier in range(4)
    ],
    'pantry': [
        {'origin_module': constraints.__name__, 'origin_tier': tier,
         'menu_size': (6, 7, 8, 8)[tier], 'selected_ingredients': [2, 4],
         'sampling': 'independent_support_family_streams_original_candidate_formula'}
        for tier in range(4)
    ],
}


def source_paths():
    """Generator and direct implementation dependencies for provenance pins.

    Campaign seals should additionally include their usual grader/runtime
    dependency inventory; this is the additive generation dependency set.
    """
    modules = ('modebench_level3_discrete', 'modebench_level3_countdown_v2',
               'modebench_level3_constraints', 'modebench_level3_python_v2',
               'modebench_level3_python_v3', 'make_modebench_data',
               'make_python_factor_mode_data', 'make_mathir_action_menu_data',
               'make_pantry_plan_mode_data')
    return sorted({Path(__file__).resolve(), *(
        Path(sys.modules[name].__file__).resolve() for name in modules)})


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(
        [SCHEMA, *parts], separators=(',', ':'), sort_keys=True).encode()).digest(), 'big')


def identity(domain, row):
    if domain in ('countdown', 'graph_coloring', 'python_factors'):
        return row_identity(domain, row)
    if domain == 'pantry':
        return (domain, row['instance_fingerprint'])
    spec = json.loads(row['answer'])
    return (domain, spec['family'], tuple(sorted(spec['bindings'].items())))


def _annotate(row, domain, tier, index):
    row = dict(row)
    inherited = {key: row.pop(key) for key in list(row) if key.startswith('level3_')}
    row['scale_origin_metadata'] = json.dumps(inherited, sort_keys=True, separators=(',', ':'))
    row.update(scale_candidate_generator=SCHEMA, scale_candidate_tier=tier,
               scale_candidate_profile=json.dumps(PROFILES[domain][tier], sort_keys=True, separators=(',', ':')),
               scale_cell_index=index)
    return row


def _countdown_rows(target, excluded, seed, tag, tier):
    lower, upper, cap = COUNTDOWN_PRESETS[tier]
    blocked, rows = set(excluded), []
    for support, count in sorted(target.items()):
        if support not in range(2, 9):
            raise ValueError('Countdown requires exact supports 2..8')
        rng = random.Random(_seed(seed, tier, support, 'countdown'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                numbers = tuple(sorted(rng.sample(range(lower, upper + 1), 4)))
                choices = [item for item in _target_statistics(numbers)
                           if item[1] == support and (cap is None or item[0] <= cap)]
                if not choices:
                    continue
                def weight(item):
                    value, _, family, divisions, products = item
                    if tier == 0:
                        return (6., 3., 1.)[family] / (1 + abs(value - sum(numbers)) / upper)
                    return (1 + 3 * divisions + products) * (1 + math.log1p(value))
                value, _, family, _, _ = rng.choices(choices, weights=list(map(weight, choices)))[0]
                key = ('countdown', numbers, value)
                if key in blocked:
                    continue
                blocked.add(key)
                spec = {'verifier': 'countdown', 'numbers': list(numbers), 'target': value,
                        'source': SCHEMA, 'instance_id': f'{tag}-{seed}-m{support}-{index}',
                        'num_completions': support, 'num_expressions': support}
                row = {'problem': _countdown_prompt(list(numbers), value),
                       'answer': json.dumps(spec, sort_keys=True), 'modebench_task': 'countdown',
                       'answer_mode_count': support, 'answer_mode_split': tag,
                       'scale_countdown_target_family': ('paired_products', 'one_product', 'other')[family]}
                rows.append(_annotate(row, 'countdown', tier, index))
                break
            else:
                raise RuntimeError(f'Countdown tier {tier}/support {support} exhausted fixed proposal budget')
    return rows


def _graph_rows(target, excluded, seed, tag, tier):
    n, hidden_count = GRAPH_PRESETS[tier]
    blocked, rows = set(excluded), []
    for support, count in sorted(target.items()):
        if support not in {4, 5, 6, 8, 9, 12, 18}:
            raise ValueError('Graph requires registered supports 4,5,6,8,9,12,18')
        rng = random.Random(_seed(seed, tier, support, 'graph'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                planted = [rng.randint(1, 3) for _ in range(n)]
                hidden = set(rng.sample(range(n), hidden_count))
                partial = [None if v in hidden else color for v, color in enumerate(planted)]
                probability = rng.uniform(.32, .65)
                edges = [[u + 1, v + 1] for u, v in combinations(range(n), 2)
                         if planted[u] != planted[v] and rng.random() < probability]
                incident = {vertex for edge in edges for vertex in edge}
                if any(v + 1 not in incident for v in hidden):
                    continue
                if graph_completion_count(n, edges, partial, cap=support) != support:
                    continue
                key = ('graph_coloring', n, tuple(map(tuple, edges)),
                       ''.join('?' if c is None else str(c) for c in partial))
                if key in blocked:
                    continue
                blocked.add(key)
                spec = {'verifier': 'graph_coloring', 'n': n, 'edges': edges,
                        'partial_colors': partial, 'source': SCHEMA,
                        'instance_id': f'{tag}-{seed}-m{support}-{index}',
                        'num_completions': support,
                        'num_solutions': graph_completion_count(n, edges, [None] * n)}
                rows.append(_annotate({
                    'problem': _graph_prompt(n, edges, partial),
                    'answer': json.dumps(spec, sort_keys=True), 'modebench_task': 'graph_coloring',
                    'answer_mode_count': support, 'answer_mode_split': tag}, 'graph_coloring', tier, index))
                break
            else:
                raise RuntimeError(f'Graph tier {tier}/support {support} exhausted fixed proposal budget')
    return rows


def _pantry_rows(target, joint_target, excluded, seed, tag, tier):
    if joint_target is None:
        raise ValueError('Pantry requires an explicit exact support/family joint_target')
    marginal = Counter()
    for cell, count in joint_target.items():
        if not isinstance(cell, tuple) or len(cell) != 2:
            raise ValueError('Pantry joint_target keys must be (support, family) tuples')
        support, family = cell
        if type(support) is not int or support < 2 or type(count) is not int or count < 0:
            raise ValueError('Pantry joint_target requires integer supports and nonnegative quotas')
        if family not in constraints.FAMILY_POOLS:
            raise ValueError(f'unknown Pantry family: {family}')
        marginal[support] += count
    if +marginal != target:
        raise ValueError('Pantry joint_target marginal differs from requested support histogram')
    blocked, rows = set(excluded), []
    for (support, family), count in sorted(joint_target.items()):
        rng = random.Random(_seed(seed, tier, support, family, 'pantry'))
        for index in range(count):
            for _ in range(MAX_PANTRY_PROPOSALS_PER_ROW):
                row = constraints._pantry_candidate(family, support, seed,
                    f'{tag}-m{support}', index, tier, rng)
                if row is None or identity('pantry', row) in blocked:
                    continue
                blocked.add(identity('pantry', row))
                row['answer_mode_split'] = tag
                rows.append(_annotate(row, 'pantry', tier, index))
                break
            else:
                raise RuntimeError(f'Pantry tier {tier}/cell {(support, family)} exhausted fixed proposal budget')
    return rows


def build_pool(domain, target, excluded, seed, tag, tier, multiplier=1, *, joint_target=None):
    """Build exact quotas from prospective laws; no empirical match is claimed.

    All exclusions use the established domain semantic identity tuples. Pantry
    requires an explicit joint histogram, preserving its support and family
    composition exactly. Larger quotas retain each support/family stream prefix.
    """
    if domain not in DOMAINS or type(tier) is not int or tier not in range(4):
        raise ValueError('expected one of five domains and integer tier 0..3')
    if type(seed) is not int or seed < 0 or type(multiplier) is not int or multiplier < 1:
        raise ValueError('seed must be a nonnegative integer; multiplier must be a positive integer')
    if any(type(support) is not int or support < 2 or type(count) is not int or count < 0
           for support, count in target.items()):
        raise ValueError('supports must be integers >=2 and quotas nonnegative integers')
    required = Counter({support: count * multiplier for support, count in target.items() if count})
    if joint_target is not None and domain != 'pantry':
        raise ValueError('joint_target is only supported for Pantry')
    if not required:
        if joint_target and any(joint_target.values()):
            raise ValueError('nonempty joint_target cannot accompany empty target')
        return []
    if domain == 'countdown':
        rows = _countdown_rows(required, excluded, seed, tag, tier)
    elif domain == 'graph_coloring':
        rows = _graph_rows(required, excluded, seed, tag, tier)
    elif domain == 'pantry':
        # Validate before multiplication so bool/float quotas cannot turn into integers.
        if joint_target is not None and any(type(c) is not int or c < 0 for c in joint_target.values()):
            raise ValueError('Pantry joint_target quotas must be nonnegative integers')
        joint = None if joint_target is None else Counter({cell: count * multiplier
                                                          for cell, count in joint_target.items()})
        rows = _pantry_rows(required, joint, excluded, seed, tag, tier)
    else:
        if domain == 'python_factors':
            module, origin_tier = PYTHON_ROUTES[tier]
            rows = module.build_pool(domain, required, excluded, seed, tag, origin_tier, multiplier=1)
        else:
            rows = constraints.build_pool(domain, required, excluded, seed, tag, tier, multiplier=1)
        # Original builders shuffle display order independently. Restore the
        # index encoded by their generation stream rather than enumerate output.
        rows = [_annotate(row, domain, tier, int(row.get('level3_cell_index',
                 json.loads(row['answer'])['instance_id'].rsplit('-', 1)[-1]))) for row in rows]
    rows.sort(key=lambda row: _seed(seed, tier, 'display', identity(domain, row)))
    ids = {identity(domain, row) for row in rows}
    if len(ids) != len(rows) or ids & set(excluded):
        raise RuntimeError('candidate semantic identities overlap exclusions or each other')
    if Counter(int(row['answer_mode_count']) for row in rows) != required:
        raise RuntimeError('candidate support histogram differs from exact requested quotas')
    return rows



def _graph_witnesses(spec):
    """Enumerate the small accepted completion set by independent plain DFS."""
    colors = list(spec['partial_colors'])
    neighbors = [set() for _ in colors]
    for u, v in spec['edges']:
        neighbors[u - 1].add(v - 1)
        neighbors[v - 1].add(u - 1)
    def visit():
        if None not in colors:
            yield list(colors)
            return
        vertex = colors.index(None)
        for color in (1, 2, 3):
            if all(colors[neighbor] != color for neighbor in neighbors[vertex]):
                colors[vertex] = color
                yield from visit()
        colors[vertex] = None
    yield from visit()


@lru_cache(maxsize=4)
def _mathir_certificate(tier):
    from oat_drgrpo.mathir import enumerate_mathir_action_menu_validations
    family = constraints._mathir_family(tier)
    bindings = dict(a=7, b=11, c=13, d=-3)
    if tier:
        bindings.update(e=5, f=2)
    actions = dict(zip('ABCDEF', family.commands))
    spec = constraints._mathir_spec(family, bindings, actions, 'certificate', 0, 0)
    validations = enumerate_mathir_action_menu_validations(spec)
    keys = frozenset(result.canonical_key for result in validations)
    if len(keys) != 5:
        raise RuntimeError('unchanged MathIR template no longer has five canonical modes')
    return keys, tuple(tuple(actions[action] for action in result.action_ids) for result in validations)


def verify_rows(domain, rows):
    """Recheck exact canonical support and original-verifier witnesses.

    MathIR's exhaustive symbolic template is independent of numeric bindings;
    every row must attain all five template modes through the original verifier.
    Python's divisor product counts its Cartesian canonical output support; its
    existing constructor and this audit both check the two original programs.
    """
    if domain not in DOMAINS:
        raise ValueError(domain)
    witnesses = 0
    for row in rows:
        spec = json.loads(row['answer'])
        support = row['answer_mode_count']
        tier = row['scale_candidate_tier']
        if domain == 'countdown':
            from modebench_level3_discrete import countdown_modes
            from oat_drgrpo.math_grader import _verify_countdown_expression
            modes = countdown_modes(tuple(spec['numbers']))[spec['target']]
            correct = len(modes) == support and all(
                _verify_countdown_expression(expression, spec) for expression in modes.values())
            prompt = _countdown_prompt(spec['numbers'], spec['target'])
            witnesses += len(modes)
        elif domain == 'graph_coloring':
            from oat_drgrpo.math_grader import _verify_graph_coloring_colors
            completions = list(_graph_witnesses(spec))
            correct = len(completions) == support == graph_completion_count(
                spec['n'], spec['edges'], spec['partial_colors']) and all(
                    _verify_graph_coloring_colors(colors, spec) for colors in completions)
            prompt = _graph_prompt(spec['n'], spec['edges'], spec['partial_colors'])
            witnesses += len(completions)
        elif domain == 'python_factors':
            from make_python_factor_mode_data import _certified_programs, _prompt
            from oat_drgrpo.python_modebench import python_factor_mode_count
            from oat_drgrpo.python_modebench_process import validate_python_factor_function_external
            programs = _certified_programs(tuple(spec['cases']))
            validated = [validate_python_factor_function_external(program, spec) for program in programs]
            correct = support == python_factor_mode_count(spec['cases']) and all(validated)
            prompt = _prompt(tuple(spec['cases']))
            witnesses += len(programs)
        elif domain == 'mathir':
            from oat_drgrpo.mathir import validate_mathir_action_menu
            family = constraints._mathir_family(tier)
            expected, paths = _mathir_certificate(tier)
            commands = {command: action for action, command in spec['actions'].items()}
            validated = [validate_mathir_action_menu(';'.join(commands[c] for c in path), spec)
                         for path in paths]
            correct = support == 5 and all(validated) and {
                result.canonical_key for result in validated if result is not None} == expected
            correct = correct and spec['family'] == family.name and spec['max_steps'] == 4
            prompt = constraints.mathir_prompt(family, spec['bindings'], spec['actions'])
            witnesses += len(paths)
        else:
            from oat_drgrpo.pantry_plan import validate_pantry_plan
            modes = constraints.exact_pantry_supports(spec)
            correct = len(modes) == support and all(
                validate_pantry_plan(witness, spec) is not None for witness in modes.values())
            fingerprint = dict(spec)
            fingerprint.pop('instance_id')
            correct = correct and row['instance_fingerprint'] == constraints._canonical_sha256(fingerprint)
            # JSON serialization sorts keys, while the original prompt uses
            # the generator's fixed nutrition-attribute display order.
            prompt_spec = dict(spec)
            prompt_spec['targets'] = {attr: spec['targets'][attr] for attr in constraints.ATTRIBUTES
                                      if attr in spec['targets']}
            prompt = constraints.pantry_prompt(spec['family'], prompt_spec)
            witnesses += len(modes)
        if not correct or row['problem'] != prompt:
            raise RuntimeError(f'{domain} exact-support, original-prompt or witness audit failed')
    return {'exact_canonical_support': True, 'unchanged_original_prompt': True,
            'original_verifier_witnesses': True, 'rows_verified': len(rows),
            'witnesses_verified': witnesses}
