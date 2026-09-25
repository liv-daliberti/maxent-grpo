"""Exclusion-only option withdrawals for saved ModeBench portfolios.

Every ModeBench domain admits the same downstream question the PantryPlan
adaptation asks: a portfolio of verified answers is produced, then the task
loses one option, and a portfolio is worth something afterwards only if one of
its saved answers is still valid. The withdrawal families below extend that one
experiment to all five domains.

Three properties are shared by construction, and the audit re-checks each one:

* *Input only.* Options are enumerated from the task specification in a fixed
  order. No generated answer, mode count or metric takes part in choosing them.
* *Exclusion only.* A withdrawal can only remove answers from the original
  verified support, never add one. The complete original support therefore
  certifies the revised task: it is feasible exactly when some original answer
  survives, which is what ``surviving`` returns.
* *Key faithfulness.* Nothing here re-reads a response body. Each canonical key
  is a faithful encoding of the executed answer -- a colouring vector, an
  expression tree, a divisor vector, an executed state path, an ingredient
  support -- and survival is decided from that encoding, directly for three
  domains and against the enumerated support for the other two.

Three withdrawals are decided by the key alone -- a colouring's digit at the
vertex, the operations an expression executes, the values a divisor vector
returns -- so ``survives`` answers them for *any* verified key, including one the
generator's census never enumerated. That matters: the Countdown census is built
from an expression generator that never emits unary negation, while the reward
verifier accepts it, so the cohort does contain verified outcomes outside the
census. Deciding survival by the key rather than by census membership keeps those
outcomes scored instead of silently counted as lost.

The other two withdrawals are not intrinsic to the key. MathIR's key is the
executed state path, not the action sequence, so its withdrawal is resolved by
re-enumerating the menu with that action dropped: a key survives exactly when the
reduced menu can still realize it. PantryPlan's key is an ingredient support, but
the quantities that make it feasible live in its witness plan. Both therefore
answer only for keys the census holds, and ``survives`` returns ``None``
otherwise so the analyzer can count such keys rather than assume them away.
"""
from __future__ import annotations

import ast
import copy
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / 'src', ROOT / 'ops'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from make_pantry_plan_mode_data import enumerate_pantry_supports  # noqa: E402
from make_modebench_data import _countdown_expression_map  # noqa: E402
from oat_drgrpo.math_grader import (  # noqa: E402
    _canonical_countdown_expression_key,
    _verify_graph_coloring_colors,
)
from oat_drgrpo.mathir import enumerate_mathir_action_menu_keys  # noqa: E402
from oat_drgrpo.pantry_plan import validate_pantry_plan  # noqa: E402

DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
#: The frozen field each dataset row carries for its certified mode count. The
#: enumerations below must reproduce it exactly, which is how the support code is
#: checked against the published benchmark rather than against itself.
CERTIFIED_FIELD = {'graph_coloring': 'num_completions', 'countdown': 'num_expressions',
                   'python_factors': 'num_modes', 'mathir': 'num_completions',
                   'pantry_plan': 'certified_mode_count'}
GRAPH_COLORS = (1, 2, 3)
#: Countdown withdraws one of the four arithmetic operations. Unary ``neg`` is a
#: sign inside an operand, not an operation the task offers, so it is not a
#: withdrawable option.
COUNTDOWN_OPS = ('add', 'sub', 'mul', 'div')
WITHDRAWAL = {'graph_coloring': 'a colour becomes unavailable at one vertex',
              'countdown': 'one arithmetic operation becomes unavailable',
              'python_factors': 'one returned divisor value is rejected',
              'mathir': 'one menu action is withdrawn',
              'pantry_plan': 'one ingredient goes out of stock'}


# --------------------------------------------------------------- graph colouring
def _graph_hidden(spec):
    partial = spec.get('partial_colors') or [None] * int(spec['n'])
    return [i for i, colour in enumerate(partial) if colour is None]


def graph_support(spec):
    """Every proper colouring consistent with the prompt's fixed digits."""
    partial = spec.get('partial_colors') or [None] * int(spec['n'])
    hidden = _graph_hidden(spec)
    support = {}
    for fill in itertools.product(GRAPH_COLORS, repeat=len(hidden)):
        colours = [int(c) if c is not None else 0 for c in partial]
        for index, colour in zip(hidden, fill):
            colours[index] = colour
        if _verify_graph_coloring_colors(colours, spec):
            support['graph_coloring:' + ''.join(str(c) for c in colours)] = colours
    return support


def graph_options(spec):
    return tuple(('color_out', f'{index + 1}:{colour}')
                 for index in _graph_hidden(spec) for colour in GRAPH_COLORS)


def graph_surviving(spec, option, support):
    vertex, colour = (int(part) for part in option[1].split(':'))
    return [key for key in support if int(key.split(':', 1)[1][vertex - 1]) != colour]


# -------------------------------------------------------------------- Countdown
def countdown_key_operations(key):
    """The operations a canonical Countdown key executes."""
    used = set()

    def walk(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise ValueError('Countdown key constants must be integers')
            return
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError('malformed canonical Countdown key')
        used.add(node.func.id)
        for argument in node.args:
            walk(argument)

    walk(ast.parse(key.split(':', 1)[1], mode='eval').body)
    if not used <= set(COUNTDOWN_OPS) | {'neg'}:
        raise ValueError('unsupported Countdown operator in canonical key')
    return used


def countdown_support(spec):
    numbers = [int(value) for value in spec['numbers']]
    support = {}
    for expression in _countdown_expression_map(numbers).get(int(spec['target']), set()):
        key = _canonical_countdown_expression_key(expression, spec)
        if key is not None:
            support[key] = expression
    return support


def countdown_options(spec):
    return tuple(('op_out', operation) for operation in COUNTDOWN_OPS)


def countdown_surviving(spec, option, support):
    return [key for key in support if option[1] not in countdown_key_operations(key)]


# --------------------------------------------------------------- Python factors
def _proper_divisors(value):
    return [d for d in range(2, value) if value % d == 0]


def python_support(spec):
    per_case = [_proper_divisors(int(case)) for case in spec['cases']]
    return {'python_factor:' + ','.join(str(d) for d in vector): list(vector)
            for vector in itertools.product(*per_case)}


def python_options(spec):
    values = sorted({d for case in spec['cases'] for d in _proper_divisors(int(case))})
    return tuple(('value_out', str(value)) for value in values)


def python_surviving(spec, option, support):
    rejected = int(option[1])
    return [key for key in support
            if rejected not in [int(d) for d in key.split(':', 1)[1].split(',')]]


# ----------------------------------------------------------------------- MathIR
def mathir_support(spec):
    return {key: None for key in enumerate_mathir_action_menu_keys(spec)}


def mathir_options(spec):
    return tuple(('action_out', action) for action in sorted(spec['actions']))


def mathir_surviving(spec, option, support):
    """Keys the menu still realizes once ``option`` is withdrawn.

    The menu grammar requires contiguous action ids, so the surviving actions
    are relabelled in their original order. Mode identity is the executed state
    path, which no relabelling can change, so the reduced menu's key set is the
    exact set of original outcomes that remain reachable.
    """
    kept = [action for action in sorted(spec['actions']) if action != option[1]]
    if len(kept) < 2:
        return []
    revised = dict(spec)
    revised['actions'] = {chr(ord('A') + position): spec['actions'][action]
                          for position, action in enumerate(kept)}
    return sorted(set(enumerate_mathir_action_menu_keys(revised)) & set(support))


# ------------------------------------------------------------------ PantryPlan
def pantry_support(spec):
    """Support key to one verified witness plan, as the frozen census records."""
    return {'pantry_plan:pantry-v1:' + '+'.join(sorted(support)): candidate
            for support, candidate in enumerate_pantry_supports(spec).items()}


def pantry_options(spec):
    return tuple(('outage', ingredient['id']) for ingredient in spec['ingredients'])


def pantry_revised_spec(spec, ingredient_id):
    """The same outage encoding the frozen Pantry adaptation protocol uses."""
    revised = copy.deepcopy(spec)
    for ingredient in revised['ingredients']:
        if ingredient['id'] == ingredient_id:
            ingredient['tags'] = sorted(set(ingredient['tags']) | {'outage'})
    revised['forbidden_tags'] = sorted(set(revised['forbidden_tags']) | {'outage'})
    return revised


def pantry_surviving(spec, option, support):
    revised = pantry_revised_spec(spec, option[1])
    return [key for key, candidate in support.items()
            if validate_pantry_plan(candidate, revised) is not None]


#: Withdrawals decided by the outcome key alone, for any verified key.
KEY_INTRINSIC = ('graph_coloring', 'countdown', 'python_factors')


def survives(domain, option, key, census_surviving, census):
    """Whether one verified outcome is still valid after the withdrawal.

    ``census_surviving`` is the frozen census subset that survives this option
    and ``census`` the whole original support. The first three domains never
    consult either: their answer is read off the key. The other two are decided
    by the census, and return ``None`` for a key the census does not hold.
    """
    if domain == 'graph_coloring':
        vertex, colour = (int(part) for part in option[1].split(':'))
        return int(key.split(':', 1)[1][vertex - 1]) != colour
    if domain == 'countdown':
        return option[1] not in countdown_key_operations(key)
    if domain == 'python_factors':
        return int(option[1]) not in [int(d) for d in key.split(':', 1)[1].split(',')]
    if domain in ('mathir', 'pantry_plan'):
        return key in census_surviving if key in census else None
    raise ValueError('unsupported domain ' + domain)


FAMILY = {'graph_coloring': (graph_support, graph_options, graph_surviving),
          'countdown': (countdown_support, countdown_options, countdown_surviving),
          'python_factors': (python_support, python_options, python_surviving),
          'mathir': (mathir_support, mathir_options, mathir_surviving),
          'pantry_plan': (pantry_support, pantry_options, pantry_surviving)}


#: Domains whose pair survival is exactly the intersection of its two single
#: withdrawals, because each withdrawal is an independent exclusion on the
#: outcome itself. MathIR is not among them: dropping two menu actions can
#: remove a state path that either action alone could still realize.
PAIR_BY_INTERSECTION = ('graph_coloring', 'countdown', 'python_factors', 'pantry_plan')


def pair_surviving(domain, spec, support, first, second):
    """Outcomes surviving both withdrawals, re-derived for the joint task."""
    if domain == 'mathir':
        kept = [action for action in sorted(spec['actions'])
                if action not in (first[1], second[1])]
        if len(kept) < 2:
            return []
        revised = dict(spec)
        revised['actions'] = {chr(ord('A') + position): spec['actions'][action]
                              for position, action in enumerate(kept)}
        return sorted(set(enumerate_mathir_action_menu_keys(revised)) & set(support))
    if domain == 'pantry_plan':
        revised = pantry_revised_spec(pantry_revised_spec(spec, first[1]), second[1])
        return sorted(key for key, candidate in support.items()
                      if validate_pantry_plan(candidate, revised) is not None)
    surviving_fn = FAMILY[domain][2]
    return sorted(set(surviving_fn(spec, first, support)) & set(surviving_fn(spec, second, support)))


def prompt_row(domain, spec):
    """Support size, certified-count agreement and every option's survivors."""
    support_fn, options_fn, surviving_fn = FAMILY[domain]
    support = support_fn(spec)
    certified = spec.get(CERTIFIED_FIELD[domain])
    options = []
    for option in options_fn(spec):
        surviving = sorted(surviving_fn(spec, option, support))
        if not set(surviving) <= set(support):
            raise ValueError('withdrawal added an outcome outside the original support')
        if domain in KEY_INTRINSIC:
            # The census route and the key predicate must agree on the census, so
            # that scoring an out-of-census key by the predicate stays consistent
            # with the feasibility the census certifies.
            predicted = {key for key in support
                         if survives(domain, option, key, frozenset(surviving), frozenset(support))}
            if predicted != set(surviving):
                raise ValueError('key predicate disagrees with the census survivors')
        options.append({'option': list(option), 'feasible': bool(surviving),
                        'binding': len(surviving) < len(support), 'surviving': surviving})
    binding = [option for option in options if option['feasible'] and option['binding']]
    pairs = []
    for position, first in enumerate(binding):
        for second in binding[position + 1:]:
            left, right = tuple(first['option']), tuple(second['option'])
            surviving = pair_surviving(domain, spec, support, left, right)
            intersection = set(first['surviving']) & set(second['surviving'])
            if not set(surviving) <= intersection:
                raise ValueError('a withdrawal pair kept an outcome neither single one keeps')
            exact = set(surviving) == intersection
            if domain in PAIR_BY_INTERSECTION and not exact:
                raise ValueError('pair survival is not the intersection for ' + domain)
            entry = {'options': [list(left), list(right)], 'feasible': bool(surviving),
                     'binding': len(surviving) < len(support), 'surviving_count': len(surviving),
                     'exact_by_intersection': exact}
            # Only the census-decided domain that cannot be recovered by
            # intersection carries its key list; the others are re-derived.
            if domain not in PAIR_BY_INTERSECTION:
                entry['surviving'] = surviving
            pairs.append(entry)
    return {'support': sorted(support), 'certified': None if certified is None else int(certified),
            'certified_match': certified is not None and int(certified) == len(support),
            'options': options, 'pairs': pairs}
