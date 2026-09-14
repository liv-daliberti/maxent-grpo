"""Three- and four-operand Countdown anchors with quota-invariant sampling.

Tiers 0/1 uniformly select admissible exact-support targets after proposing
three distinct operands, except for the predeclared support-7 four-operand
case required by the exact finite inventory. Tiers 2/3 retain the frozen target weights of Countdown
v2 tiers 1/3. Every support cell has an independent random stream; requested
counts and model outcomes never determine candidate ranks or target weights.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations
from math import comb
import hashlib
import json
from numbers import Integral
from pathlib import Path
import random
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from modebench_level3_discrete import _countdown_prompt, countdown_modes, row_identity
from modebench_level3_countdown_v2 import _target_statistics as four_target_statistics, proposal_weight as four_proposal_weight

SCHEMA = 'modebench_level3_countdown_candidate_v3'
# Operand count, lower/upper operand bounds, target cap, inherited weight tier.
PRESETS = ((3, 2, 32, None, None), (3, 8, 99, None, None),
           (4, 2, 18, 72, 1), (4, 5, 64, None, 3))
SUPPORTS = frozenset(range(2, 9))
SAMPLING_LAW = 'independent_support_rng_uniform_operand_proposals_uniform_three_operand_targets_fixed_four_operand_target_weights_without_replacement'
MAX_PROPOSALS_PER_ROW = 100_000
DEVELOPMENT_SEED_OFFSET = 400_000
# The originally proposed 2..24 anchor had only eight fresh support-seven IDs,
# fewer than the 14 needed across the reference splits. This pre-model capacity
# revision enlarges only the numeric range; the proposal law stays uniform.
MAX_EXACT_CAPACITY_OPERAND_SETS = 5000
SUPPORT7_EXCEPTION = {
    'applies_to_tiers': [0, 1], 'support': 7,
    'preset': [4, 2, 18, 72, 1],
    'reason': 'Three-operand support-7 inventory cannot supply the fixed split quotas.',
    'tier0_three_operand_total_identities': 23,
    'tier0_fresh_after_historical_and_prior_pool_exclusions': 8,
    'tier1_three_operand_total_identities': 1,
    'tier1_only_three_operand_identity': {'numbers': [20, 32, 48], 'target': 4},
    'development_support7_rows_required_per_tier': 5,
    'fixed_before_pool_generation': True,
    'quota_or_depletion_fallback': False,
}


def preset_for_support(support: int, difficulty: int) -> tuple:
    if support == 7 and difficulty in (0, 1):
        return tuple(SUPPORT7_EXCEPTION['preset'])
    return PRESETS[difficulty]


@lru_cache(maxsize=8192)
def three_target_statistics(numbers: tuple[int, ...]) -> tuple[tuple[int, int, int, int, int], ...]:
    """Exact support for positive targets other than an input operand."""
    if len(numbers) != 3:
        raise ValueError('three operands are required')
    records = []
    for target, expressions in countdown_modes(numbers).items():
        if target <= 0 or target in numbers or len(expressions) not in SUPPORTS:
            continue
        records.append((target, len(expressions), 3,
                        min(key.count('div(') for key in expressions),
                        min(key.count('mul(') for key in expressions)))
    return tuple(sorted(records))


def target_choices(numbers: tuple[int, ...], support: int, difficulty: int) -> tuple:
    count, _lower, _upper, cap, _profile = preset_for_support(support, difficulty)
    if len(numbers) != count:
        raise ValueError('operand count differs from the preset')
    statistics = three_target_statistics(numbers) if count == 3 else four_target_statistics(numbers)
    return tuple(record for record in statistics if record[1] == support and (cap is None or record[0] <= cap))


@lru_cache(maxsize=8)
def _small_region_catalog(preset: tuple) -> tuple[tuple[int, tuple[int, ...], int], ...] | None:
    operand_count, lower, upper, cap, _profile = preset
    if comb(upper - lower + 1, operand_count) > MAX_EXACT_CAPACITY_OPERAND_SETS:
        return None
    records = []
    for numbers in combinations(range(lower, upper + 1), operand_count):
        statistics = three_target_statistics(numbers) if operand_count == 3 else four_target_statistics(numbers)
        records.extend((support, numbers, target) for target, support, *_ in statistics
                       if cap is None or target <= cap)
    return tuple(records)


def available_capacity(difficulty: int, excluded: set) -> Counter | None:
    """Exact fresh capacity in finite small regions; None means not enumerated.

    This checks feasibility only. The catalog never selects or ranks proposals.
    Larger regions retain the same fixed per-row rejection limit.
    """
    catalog = _small_region_catalog(PRESETS[difficulty])
    if catalog is None:
        return None
    return Counter(support for support, numbers, value in catalog
                   if ('countdown', numbers, value) not in excluded)


def proposal_weight(numbers: tuple[int, ...], statistic: tuple, difficulty: int) -> float:
    inherited_profile = preset_for_support(statistic[1], difficulty)[4]
    return 1.0 if inherited_profile is None else four_proposal_weight(numbers, statistic, inherited_profile)


def _stream_seed(seed: int, difficulty: int, support: int) -> int:
    return int.from_bytes(hashlib.sha256(f'{SCHEMA}:{seed}:{difficulty}:{support}'.encode()).digest()[:8], 'big')


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def build_pool(domain: str, target: Counter, excluded: set, seed: int, tag: str,
               difficulty: int, multiplier: int = 4) -> list[dict[str, Any]]:
    """Draw exact support quotas without changing the per-cell proposal law."""
    if domain != 'countdown':
        raise ValueError('Countdown v3 only supports countdown')
    difficulty = _nonnegative_integer(difficulty, 'difficulty')
    if difficulty not in range(4):
        raise ValueError('difficulty must be 0..3')
    seed = _nonnegative_integer(seed, 'seed')
    multiplier = _nonnegative_integer(multiplier, 'multiplier')
    if not multiplier:
        raise ValueError('multiplier must be positive')
    required = Counter()
    for support, count in target.items():
        support, count = _nonnegative_integer(support, 'support'), _nonnegative_integer(count, 'count')
        if support not in SUPPORTS:
            raise ValueError('Countdown v3 requires support cells 2..8')
        if count:
            required[support] = count * multiplier
    if not required:
        return []
    blocked = set(excluded)
    rows = []
    for support, needed in sorted(required.items()):
        operand_count, lower, upper, _cap, _profile = preset_for_support(support, difficulty)
        rng = random.Random(_stream_seed(seed, difficulty, support))
        for accepted in range(needed):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                numbers = tuple(sorted(rng.sample(range(lower, upper + 1), operand_count)))
                choices = target_choices(numbers, support, difficulty)
                if not choices:
                    continue
                if operand_count == 3:
                    statistic = rng.choice(choices)
                else:
                    statistic = rng.choices(choices, weights=[proposal_weight(numbers, item, difficulty)
                                                              for item in choices], k=1)[0]
                value, _count, family, _divisions, _products = statistic
                identity = (domain, numbers, value)
                if identity in blocked:
                    continue
                blocked.add(identity)
                spec = {'verifier': domain, 'numbers': list(numbers), 'target': value,
                        'source': SCHEMA, 'instance_id': f'{tag}-{seed}-m{support}-{accepted}',
                        'num_completions': support, 'num_expressions': support}
                rows.append({'problem': _countdown_prompt(list(numbers), value),
                             'answer': json.dumps(spec, sort_keys=True), 'modebench_task': domain,
                             'answer_mode_count': support, 'answer_mode_split': tag,
                             'level3_difficulty': difficulty, 'level3_generator': SCHEMA,
                             'level3_sampling_law': SAMPLING_LAW,
                             'level3_support_sampling_index': accepted,
                             'level3_countdown_support7_exception': support == 7 and difficulty in (0, 1),
                             'level3_countdown_target_family': ('paired_products', 'one_product', 'other', 'three_operand')[family]})
                break
            else:
                raise RuntimeError(f'Countdown v3 tier {difficulty}, support {support}: fixed proposal budget '
                                   f'exhausted after {accepted}/{needed} rows; no proposal-law fallback is permitted')
    random.Random(seed).shuffle(rows)
    identities = {row_identity(domain, row) for row in rows}
    if len(identities) != len(rows) or identities & excluded:
        raise RuntimeError('Countdown v3 semantic identity disjointness failed')
    if Counter(row['answer_mode_count'] for row in rows) != required:
        raise RuntimeError('Countdown v3 support histogram differs from target')
    return rows


def verify_semantics(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Certify original prompts, exact support, and every canonical witness."""
    from oat_drgrpo.math_grader import _verify_countdown_expression
    for row in rows:
        spec = json.loads(row['answer'])
        numbers, target = tuple(spec['numbers']), spec['target']
        expressions = countdown_modes(numbers)[target]
        if len(expressions) != row['answer_mode_count'] or not all(
                _verify_countdown_expression(expression, spec) for expression in expressions.values()):
            raise RuntimeError('Countdown v3 original-verifier canonical witness check failed')
        if row['problem'] != _countdown_prompt(list(numbers), target):
            raise RuntimeError('Countdown v3 original prompt drift')
    return {'exact_canonical_support': True, 'original_prompt': True,
            'all_canonical_witnesses_original_verifier': True}


def materialize_development_pools(output_root: Path) -> list[dict[str, Any]]:
    """Write all four 128-row pools using this builder, regardless of root routing."""
    import shutil
    import tempfile
    from materialize_modebench_level3 import ROOT, SEEDS, historical_ids, identity_set, reference_rows, row_hash, verify_rows
    destination = Path(output_root).resolve() / 'pools/countdown'
    if destination.exists():
        raise FileExistsError(f'fresh Countdown pool directory required: {destination}')
    reference = reference_rows('countdown', 'dev')
    if len(reference) != 128:
        raise ValueError('expected the 128-row Level 2 development reference')
    target = Counter(int(row['answer_mode_count']) for row in reference)
    blocked = historical_ids('countdown')
    for directory in sorted((ROOT / 'var/data').glob('modebench_level3_calibration*/pools/countdown')):
        for path in sorted(directory.glob('*.jsonl')):
            blocked |= identity_set('countdown', [json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.countdown.', dir=destination.parent))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = []
    try:
        for difficulty in range(4):
            seed = SEEDS['countdown'] + DEVELOPMENT_SEED_OFFSET + 1000 * difficulty
            rows = build_pool('countdown', target, blocked, seed, 'level3_development_pool', difficulty, multiplier=1)
            checks = verify_rows('countdown', rows, reference, target, blocked)
            checks.update(verify_semantics(rows))
            path = staging / f'difficulty_{difficulty}.jsonl'
            with path.open('x') as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + '\n')
            record = {'schema': 'modebench_level3_development_pool_v1', 'domain': 'countdown',
                      'difficulty': difficulty, 'seed': seed, 'rows': len(rows), 'rows_sha256': row_hash(rows),
                      'checks': checks, 'support_histogram': dict(sorted(target.items())),
                      'source_sha256': source_hash, 'source_path': str(Path(__file__).resolve()),
                      'generation_profile': SCHEMA, 'preset': list(PRESETS[difficulty]),
                      'support7_exception': SUPPORT7_EXCEPTION,
                      'information_boundary': 'development candidates only; no confirmation model outcomes used'}
            with path.with_suffix('.identity.json').open('x') as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write('\n')
            blocked |= identity_set('countdown', rows)
            records.append(record)
            print(json.dumps({'event': 'pool_built', 'difficulty': difficulty, 'rows': len(rows), 'seed': seed}), flush=True)
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
        return records
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'pools': materialize_development_pools(args.output_root)}, sort_keys=True))


if __name__ == '__main__':
    main()
