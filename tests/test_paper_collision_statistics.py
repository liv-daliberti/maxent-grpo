"""Independent checks for conditional-concentration estimands (synthetic data only)."""
from collections import Counter
from fractions import Fraction
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations, product
from math import sqrt
from pathlib import Path
import statistics
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'ops/exp_scaling/analyze_paper_conditional_concentration.py'
spec = spec_from_file_location('paper_conditional_concentration_statistics_tests', SOURCE)
m = module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def padded(keys):
    assert len(keys) <= 8
    return list(keys) + [None] * (8 - len(keys))


def four_draws(first, second=(), third=(), fourth=()):
    return [padded(x) for x in (first, second, third, fourth)]


def pair_oracle(keys):
    """Directly enumerate unordered valid pairs, independently of count algebra."""
    valid = [x for x in keys if x is not None]
    pairs = list(combinations(valid, 2))
    if not pairs:
        return None
    return Fraction(sum(a == b for a, b in pairs), len(pairs))


def test_collision_requires_two_valid_observations_and_preserves_extremes():
    assert m.collision({}) is None
    assert m.collision({'a': 1}) is None
    assert m.collision({'a': 2}) == 1
    assert m.collision({'a': 1, 'b': 1}) == 0
    assert m.collision({'a': 3, 'b': 2, 'c': 1}) == pytest.approx(4 / 15)


def test_prompt_metrics_retain_budget_units_for_synthetic_independent_draws():
    draws = four_draws(['a', 'a', 'b'], ['c'], (), ['a', 'b', 'c', 'd'])
    result = m.prompt_metrics(draws)
    assert result['correct'] == 8
    assert result['total'] == 32
    assert result['correct_pairs'] == 28
    assert result['colliding_pairs'] == 5
    assert result['collision'] == pytest.approx(5 / 28)
    assert result['mean8'] == pytest.approx(1 / 4)
    assert result['pass8'] == pytest.approx(3 / 4)
    assert result['distinct8'] == pytest.approx(7 / 4)
    assert result['extra8'] == pytest.approx(1)
    # For genuinely independent synthetic streams, neither small group alone
    # identifies collision, but the pooled law does. Actual experiment streams
    # require source-certified deduplication before this pooling is admissible.
    separate = four_draws(['a'], ['a'])
    assert all(pair_oracle(draw) is None for draw in separate)
    pooled = m.prompt_metrics(separate)
    assert pooled['collision'] == 1
    assert pooled['correct_pairs'] == 1


def test_empty_correct_samples_are_undefined_not_zero_collision():
    result = m.prompt_metrics(four_draws(()))
    assert result['collision'] is None
    assert result['correct'] == 0
    assert result['correct_pairs'] == result['colliding_pairs'] == 0
    assert result['pass8'] == result['distinct8'] == result['extra8'] == 0


def test_pair_seed_uses_common_eligibility_and_equal_prompt_weights():
    a = {'x': four_draws(['a', 'b']),
         'y': four_draws(['same'] * 8),
         'only_b': four_draws(['a']),
         'neither': four_draws(())}
    b = {'x': four_draws(['a', 'a']),
         'y': four_draws(list('abcdefgh')),
         'only_b': four_draws(['a'] * 8),
         'neither': four_draws(())}
    result = m.pair_seed(a, b)
    assert result['n_total'] == 4
    assert result['n_eligible'] == 2
    assert set(result['eligible_ids']) == {'x', 'y'}
    assert result['a']['collision'] == pytest.approx(.5)
    assert result['b']['collision'] == pytest.approx(.5)
    assert result['delta']['collision'] == pytest.approx(0)
    assert result['pooled_collision_a'] == pytest.approx(28 / 29)
    assert result['pooled_collision_b'] == pytest.approx(1 / 29)
    # The prompt-weighted effect is zero despite a large pair-pooled change.
    assert result['pooled_collision_b'] != result['pooled_collision_a']


def test_explicit_prompt_population_changes_neither_pairing_nor_weights():
    a = {'x': four_draws(['a', 'b']), 'y': four_draws(['a', 'a'])}
    b = {'x': four_draws(['a', 'a']), 'y': four_draws(['a', 'b'])}
    result = m.pair_seed(a, b, prompt_ids=['x'])
    assert result['n_eligible'] == 1
    assert set(result['eligible_ids']) == {'x'}
    assert result['delta']['collision'] == pytest.approx(1)


def test_disjoint_draw_selection_has_orientation_specific_eligibility():
    a = {'x': four_draws(['a'], ['a'], ['a'], ['b']),
         'only_orientation_zero': four_draws(['a'], ['b'])}
    b = {'x': four_draws(['a'], ['b'], ['a'], ['a']),
         'only_orientation_zero': four_draws((), (), ['a'], ['a'])}
    zero = m.pair_seed(a, b, draws_a=(0, 1), draws_b=(2, 3))
    one = m.pair_seed(a, b, draws_a=(2, 3), draws_b=(0, 1))
    assert set(zero['eligible_ids']) == {'x', 'only_orientation_zero'}
    assert set(one['eligible_ids']) == {'x'}
    assert zero['n_eligible'] == 2
    assert one['n_eligible'] == 1
    assert zero['delta']['collision'] == pytest.approx(.5)
    assert one['delta']['collision'] == pytest.approx(0)


def test_empty_pairwise_population_is_undefined():
    result = m.pair_seed({'x': four_draws(['a'])}, {'x': four_draws(['a', 'a'])})
    assert result['n_eligible'] == 0
    assert result['eligible_ids'] == []
    assert result['delta']['collision'] is None


def test_five_seed_t_interval_matches_independent_hand_calculation():
    values = {'s1': -.2, 's2': -.1, 's3': 0., 's4': .1, 's5': .2}
    result = m.summarize_seeds(values)
    half = 2.7764451051977987 * statistics.stdev(values.values()) / sqrt(5)
    assert result['n'] == 5
    assert result['mean'] == pytest.approx(0, abs=1e-15)
    assert result['ci95'] == pytest.approx([-half, half])
    assert result['values'] == values


@pytest.mark.parametrize('values', [{}, {'s1': .1}, {'s1': .1, 's2': .2, 's3': .3, 's4': .4}])
def test_partial_or_empty_seed_sets_have_no_five_seed_interval(values):
    result = m.summarize_seeds(values)
    assert result['n'] == len(values)
    assert result['ci95'] is None
    if not values:
        assert result['mean'] is None
    else:
        assert result['mean'] == pytest.approx(statistics.mean(values.values()))


def test_collision_is_unbiased_at_each_fixed_valid_count_by_exact_enumeration():
    # P=3/5, q=(1/3,2/3), so C(q)=5/9 independently of valid count.
    categories, probabilities = ('a', 'b', None), (Fraction(1, 5), Fraction(2, 5), Fraction(2, 5))
    for budget in (2, 3, 4, 5, 6):
        denominator, numerator = Counter(), Counter()
        for draws in product(range(3), repeat=budget):
            keys = [categories[i] for i in draws]
            probability = Fraction(1)
            for i in draws:
                probability *= probabilities[i]
            correct = sum(x is not None for x in keys)
            oracle = pair_oracle(keys)
            actual = m.collision(Counter(x for x in keys if x is not None))
            if correct < 2:
                assert actual is None
                continue
            assert actual == pytest.approx(float(oracle))
            denominator[correct] += probability
            numerator[correct] += probability * oracle
        for correct in range(2, budget + 1):
            assert numerator[correct] / denominator[correct] == Fraction(5, 9)


def test_shared_rng_common_eligibility_can_bias_a_paired_conditional_contrast():
    # Uniform midpoints reproduce the exact interval probabilities.
    uniforms = [Fraction(2 * i + 1, 16) for i in range(8)]
    def keys(u, p):
        return 'a' if u < p / 2 else ('b' if u < p else None)
    total, sum_a, sum_b = 0, Fraction(0), Fraction(0)
    for u in product(uniforms, repeat=2):
        a = [keys(v, Fraction(1, 2)) for v in u]
        b = [keys(v, Fraction(3, 4)) for v in u]
        ca, cb = pair_oracle(a), pair_oracle(b)
        if ca is None or cb is None:
            continue
        total += 1
        sum_a += ca
        sum_b += cb
        assert m.collision(Counter(a)) == float(ca)
        assert m.collision(Counter(b)) == float(cb)
    assert sum_a / total == Fraction(1, 2)
    assert sum_b / total == Fraction(5, 8)
    assert (sum_b - sum_a) / total == Fraction(1, 8)
    # Both unconditional policies nevertheless have q=(1/2,1/2).
    for p in (Fraction(1, 2), Fraction(3, 4)):
        correct_keys = [keys(v, p) for v in uniforms if keys(v, p) is not None]
        proportions = [Fraction(n, len(correct_keys)) for n in Counter(correct_keys).values()]
        assert sum(x*x for x in proportions) == Fraction(1, 2)


def test_independent_streams_remove_the_shared_rng_counterexample_bias():
    uniforms = [Fraction(2 * i + 1, 16) for i in range(8)]
    def keys(u, p):
        return 'a' if u < p / 2 else ('b' if u < p else None)
    total, difference = 0, Fraction(0)
    for u in product(uniforms, repeat=4):
        ca = pair_oracle([keys(v, Fraction(1, 2)) for v in u[:2]])
        cb = pair_oracle([keys(v, Fraction(3, 4)) for v in u[2:]])
        if ca is not None and cb is not None:
            total += 1
            difference += cb-ca
    assert total > 0
    assert difference / total == 0


def test_collision_is_exact_lyapunov_quantity_for_categorical_mean_flow():
    # Check exact directional derivative using a central difference of the
    # quadratic C, independently from its variance representation.
    epsilon = Fraction(1, 100)
    for numerators in product(range(5), repeat=4):
        if sum(numerators) != 4:
            continue
        q = [Fraction(v, 4) for v in numerators]
        concentration = sum(x*x for x in q)
        velocity = [x*(x-concentration) for x in q]
        plus = sum((x+epsilon*v)**2 for x,v in zip(q, velocity))
        minus = sum((x-epsilon*v)**2 for x,v in zip(q, velocity))
        derivative = (plus-minus)/(2*epsilon)
        variance_form = 2*(sum(x**3 for x in q)-concentration**2)
        assert derivative == variance_form
        assert derivative >= 0
        positive = {x for x in q if x > 0}
        assert (derivative == 0) == (len(positive) == 1)


@pytest.mark.parametrize('counts', [{'a': -1}, {'a': 1.5}, {'a': True}])
def test_collision_rejects_invalid_integer_counts(counts):
    with pytest.raises(ValueError):
        m.collision(counts)


@pytest.mark.parametrize('draws', [[], [[None] * 7], [[''] + [None] * 7], [[False] + [None] * 7]])
def test_prompt_metrics_rejects_corrupt_draw_groups_or_keys(draws):
    with pytest.raises(ValueError):
        m.prompt_metrics(draws)


def test_pair_seed_rejects_different_prompt_populations_and_reused_draw_indices():
    a = {'x': four_draws(['a', 'a'])}
    with pytest.raises(ValueError):
        m.pair_seed(a, {'y': four_draws(['a', 'a'])})
    with pytest.raises(ValueError):
        m.pair_seed(a, a, prompt_ids=['missing'])
    with pytest.raises(ValueError):
        m.pair_seed(a, a, draws_a=(0, 0))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), None])
def test_seed_summary_rejects_nonfinite_or_implicitly_missing_estimates(value):
    with pytest.raises(ValueError):
        m.summarize_seeds({'s1': value})


def test_reused_streams_bias_naive_32_collision_even_with_identical_marginals():
    stream_ids = [draw + option for draw in range(4) for option in range(8)]
    multiplicities = Counter(stream_ids)
    assert len(multiplicities) == 11
    assert list(multiplicities.values()) == [1, 2, 3, 4, 4, 4, 4, 4, 3, 2, 1]
    duplicate_pairs = sum(n * (n-1) // 2 for n in multiplicities.values())
    assert duplicate_pairs == 38
    # Eleven independent fair binary modes; each occurrence of a stream uses
    # the same outcome. Enumerating all 2^11 possibilities gives exact means.
    naive, distinct = Fraction(0), Fraction(0)
    for outcomes in product(('a', 'b'), repeat=11):
        full = [outcomes[stream] for stream in stream_ids]
        counts_full, counts_unique = Counter(full), Counter(outcomes)
        raw_full = Fraction(sum(n*(n-1) for n in counts_full.values()), 32*31)
        raw_unique = Fraction(sum(n*(n-1) for n in counts_unique.values()), 11*10)
        assert m.collision(counts_full) == pytest.approx(float(raw_full))
        assert m.collision(counts_unique) == pytest.approx(float(raw_unique))
        naive += raw_full
        distinct += raw_unique
    assert distinct / (2**11) == Fraction(1, 2)
    assert naive / (2**11) == Fraction(267, 496)
    assert naive / (2**11) == Fraction(1, 2) + Fraction(1, 2)*Fraction(19, 248)


def test_cross_prompt_rng_reuse_can_bias_random_eligible_mean_but_not_fixed_score():
    # Conditions A/B have independent streams. Within each condition the two
    # prompts reuse the same uniforms. True C_X=.5 and C_Y=1 in both models.
    # X always verifies; Y verifies with P_A=.5 versus P_B=1.
    selected_mean, fixed_score = Fraction(0), Fraction(0)
    trials = 0
    for outcomes in product((0, 1), repeat=4):
        a_u, b_u = outcomes[:2], outcomes[2:]
        xa, xb = [('a' if u == 0 else 'b') for u in a_u], [('a' if u == 0 else 'b') for u in b_u]
        ya, yb = [('a' if u == 0 else None) for u in a_u], ['a', 'a']
        dx = pair_oracle(xb) - pair_oracle(xa)
        cy_a, cy_b = pair_oracle(ya), pair_oracle(yb)
        eligible = [dx]
        if cy_a is not None:
            eligible.append(cy_b - cy_a)
        selected_mean += sum(eligible) / len(eligible)
        fixed_score += sum(eligible) / 2  # Fixed total prompt count, not |E|.
        trials += 1
        for keys in (xa, xb, ya, yb):
            actual = m.collision(Counter(k for k in keys if k is not None))
            oracle = pair_oracle(keys)
            assert actual == (None if oracle is None else float(oracle))
    assert selected_mean / trials == Fraction(1, 16)
    assert fixed_score / trials == 0


# The fixtures below model the audited V0 parent+i branch. They intentionally
# carry no experiment outputs and do not assert statistical independence merely
# because nominal IDs differ.
DEFAULT_PARENT_SEEDS = [100, 101, 102, 103]


def checkpoint_from_stream_keys(prompt_stream_keys, parent_seeds=None):
    seeds = DEFAULT_PARENT_SEEDS if parent_seeds is None else parent_seeds
    return {
        'sampling_certificate': {
            'draw_seeds': list(seeds),
            'same_prompt_identities': True,
            'same_recorded_decoder_fields': True,
        },
        'prompts': {
            pid: {
                'draws': [[mapping.get(seed + option) for option in range(8)] for seed in seeds],
                'option_ids_by_draw': [None] * 4,
                'request_seeds_by_draw': [None] * 4,
            }
            for pid, mapping in prompt_stream_keys.items()
        },
    }


def prepared_from_stream_keys(prompt_stream_keys, parent_seeds=None):
    return m.prepare_checkpoint(checkpoint_from_stream_keys(prompt_stream_keys, parent_seeds))


def constant_streams(key='a'):
    return {sid: key for sid in range(100, 111)}


def test_stream_representatives_select_earliest_positions_not_all_32_outputs():
    cp = checkpoint_from_stream_keys({'x': {i: str(i) for i in range(100, 111)}})
    result = m.stream_representatives(cp['prompts']['x']['draws'], DEFAULT_PARENT_SEEDS)
    assert result['n_saved'] == 32
    assert result['n_streams'] == 11
    assert result['repeated_occurrences'] == 21
    assert result['duplicate_key_disagreements'] == 0
    positions = [(r['stream_id'], r['draw_index'], r['option_index']) for r in result['representatives']]
    assert positions == [(100+i, 0, i) for i in range(8)] + [(108, 1, 7), (109, 2, 7), (110, 3, 7)]
    assert [r['key'] for r in result['representatives']] == [str(i) for i in range(100, 111)]


def test_stream_selection_is_index_based_even_when_saved_keys_disagree():
    draws = [[None] * 8 for _ in range(4)]
    # Stream 103 first fails; its three later appearances verify. Selection
    # cannot prefer success or majority agreement. Stream 107 first succeeds.
    draws[0][7] = 'first'
    draws[1][2] = draws[2][1] = draws[3][0] = 'later'
    draws[1][6] = 'different'
    result = m.stream_representatives(draws, DEFAULT_PARENT_SEEDS)
    by_id = {r['stream_id']: r for r in result['representatives']}
    assert by_id[103]['key'] is None
    assert by_id[107]['key'] == 'first'
    # Three disagreements for each named stream (including later missing keys).
    assert result['duplicate_key_disagreements'] == 6
    assert result['n_streams'] == 11
    assert result['repeated_occurrences'] == 21


def test_first_representative_uses_recorded_draw_order_not_numerical_seed_order():
    seeds = [103, 100, 102, 101]
    draws = [[f'draw-{d}-option-{i}' for i in range(8)] for d in range(4)]
    result = m.stream_representatives(draws, seeds)
    by_id = {r['stream_id']: r for r in result['representatives']}
    assert list(by_id) == list(range(100, 111))
    assert by_id[103]['draw_index'] == 0
    assert by_id[103]['option_index'] == 0
    assert by_id[103]['key'] == 'draw-0-option-0'
    assert by_id[100]['draw_index'] == 1
    assert by_id[100]['option_index'] == 0


@pytest.mark.parametrize('draws,seeds', [([[None]*8], []), ([[None]*7], [100]), ([[None]*8], [True])])
def test_stream_representatives_reject_uncertified_shape_or_seed_type(draws, seeds):
    with pytest.raises(ValueError):
        m.stream_representatives(draws, seeds)


def test_disjoint_splits_partition_only_common_ids_with_odd_budget_and_orientation_flip():
    a = [999, *reversed(range(100, 111)), 103]
    b = [998, *range(100, 111)]
    lower, upper = m.split_stream_ids(a, b)
    reverse_a, reverse_b = m.split_stream_ids(a, b, orientation=1)
    assert lower == list(range(100, 105))
    assert upper == list(range(105, 111))
    assert reverse_a == upper and reverse_b == lower
    assert not set(lower) & set(upper)
    assert set(lower) | set(upper) == set(a) & set(b)


@pytest.mark.parametrize('common,expected', [([], ([], [])), ([3], ([], [3])), ([3, 5], ([3], [5]))])
def test_small_common_stream_sets_preserve_empty_halves(common, expected):
    assert m.split_stream_ids(common, common) == expected


def test_invalid_stream_orientation_is_rejected():
    with pytest.raises(ValueError):
        m.split_stream_ids([1, 2], [1, 2], orientation=2)


def test_checkpoint_preparation_deduplicates_within_each_prompt_not_across_prompts():
    source = checkpoint_from_stream_keys({'x': constant_streams('a'), 'y': constant_streams('b')})
    prepared = m.prepare_checkpoint(source)
    assert set(prepared) == {'x', 'y'}
    for pid, key in [('x', 'a'), ('y', 'b')]:
        assert len(prepared[pid]['streams']) == 11
        assert set(prepared[pid]['streams'].values()) == {key}
        assert prepared[pid]['stream_audit']['n_saved'] == 32
        assert prepared[pid]['primary_metrics'] == m.prompt_metrics(source['prompts'][pid]['draws'])
    assert m.prepare_checkpoint(None) is None


@pytest.mark.parametrize('field', ['same_prompt_identities', 'same_recorded_decoder_fields'])
def test_checkpoint_preparation_requires_recorded_sampling_certificates(field):
    cp = checkpoint_from_stream_keys({'x': constant_streams()})
    cp['sampling_certificate'][field] = False
    with pytest.raises(ValueError):
        m.prepare_checkpoint(cp)


@pytest.mark.parametrize('field,value', [('option_ids_by_draw', [[0, 1, 2, 3, 4, 5, 6, 7]] * 4),
                                         ('request_seeds_by_draw', [[100] * 8] * 4)])
def test_checkpoint_preparation_rejects_unsupported_request_branches(field, value):
    cp = checkpoint_from_stream_keys({'x': constant_streams()})
    cp['prompts']['x'][field] = value
    with pytest.raises(ValueError):
        m.prepare_checkpoint(cp)


def test_endpoint_representatives_are_chosen_independently_without_success_filtering():
    a = checkpoint_from_stream_keys({'x': constant_streams('a')})
    b = checkpoint_from_stream_keys({'x': constant_streams('b')})
    a['prompts']['x']['draws'][0][3] = None
    b['prompts']['x']['draws'][0][4] = None
    pa, pb = m.prepare_checkpoint(a)['x'], m.prepare_checkpoint(b)['x']
    assert pa['streams'][103] is None and pb['streams'][103] == 'b'
    assert pb['streams'][104] is None and pa['streams'][104] == 'a'
    assert pa['stream_audit']['duplicate_key_disagreements'] == 3
    assert pb['stream_audit']['duplicate_key_disagreements'] == 3
    assert set(pa['streams']) == set(pb['streams']) == set(range(100, 111))


def test_paired_streams_use_11_primary_and_5_6_disjoint_budgets():
    a = prepared_from_stream_keys({'x': {i: str(i % 2) for i in range(100, 111)}})
    b = prepared_from_stream_keys({'x': constant_streams()})
    primary = m.paired_streams(a, b)
    zero = m.paired_streams(a, b, orientation=0)
    one = m.paired_streams(a, b, orientation=1)
    assert primary['selected_total_a'] == primary['selected_total_b'] == [11]
    assert primary['stream_selection']['shared_stream_counts'] == [11]
    assert not primary['stream_selection']['disjoint_in_every_prompt']
    assert zero['selected_total_a'] == one['selected_total_b'] == [5]
    assert zero['selected_total_b'] == one['selected_total_a'] == [6]
    for result in (zero, one):
        assert result['stream_selection']['shared_stream_counts'] == [0]
        assert result['stream_selection']['disjoint_in_every_prompt']
    assert primary['a']['collision'] == pytest.approx(float(pair_oracle(list(a['x']['streams'].values()))))
    assert zero['a']['collision'] == pytest.approx(float(pair_oracle([a['x']['streams'][i] for i in range(100, 105)])))
    assert one['a']['collision'] == pytest.approx(float(pair_oracle([a['x']['streams'][i] for i in range(105, 111)])))
    # Draw-budget metrics keep their original intact K8 definition even when
    # collision uses only one nominal representative or one disjoint subset.
    for result in (primary, zero, one):
        for metric in ('mean8', 'pass8', 'distinct8', 'extra8'):
            assert result['a'][metric] == a['x']['primary_metrics'][metric]
            assert result['b'][metric] == b['x']['primary_metrics'][metric]


def test_disjoint_stream_eligibility_is_separate_for_each_orientation():
    a = prepared_from_stream_keys({'both': constant_streams(), 'zero': {100: 'a', 101: 'b'}, 'one': {105: 'a', 106: 'b'}})
    b = prepared_from_stream_keys({'both': constant_streams(), 'zero': {105: 'a', 106: 'a'}, 'one': {100: 'a', 101: 'a'}})
    zero, one = m.paired_streams(a, b, orientation=0), m.paired_streams(a, b, orientation=1)
    assert zero['eligible_ids'] == ['both', 'zero']
    assert one['eligible_ids'] == ['both', 'one']
    assert zero['n_eligible'] == one['n_eligible'] == 2
    assert zero['coverage'] == one['coverage'] == pytest.approx(2/3)
    assert zero['delta']['collision'] == one['delta']['collision'] == pytest.approx(.5)
    assert m.paired_streams(a, b)['eligible_ids'] == ['both', 'one', 'zero']


def test_explicit_stream_sensitivity_population_is_respected():
    a = prepared_from_stream_keys({'x': {100: 'a', 101: 'b'}, 'y': {100: 'a', 101: 'a'}})
    b = prepared_from_stream_keys({'x': {100: 'a', 101: 'a'}, 'y': {100: 'a', 101: 'b'}})
    assert m.paired_streams(a, b)['delta']['collision'] == pytest.approx(0)
    selected = m.paired_streams(a, b, prompt_ids=['x'])
    assert selected['n_total'] == selected['n_eligible'] == 1
    assert selected['delta']['collision'] == pytest.approx(1)
    empty = m.paired_streams(a, b, prompt_ids=[])
    assert empty['n_total'] == empty['n_eligible'] == 0
    assert empty['delta']['collision'] is None and empty['coverage'] is None


def test_paired_streams_reject_unmatched_prompt_populations_and_missing_streams():
    a = prepared_from_stream_keys({'x': constant_streams()})
    b = prepared_from_stream_keys({'y': constant_streams()})
    with pytest.raises(ValueError):
        m.paired_streams(a, b)
    with pytest.raises(ValueError):
        m.paired_streams(a, a, prompt_ids=['z'])
    with pytest.raises(ValueError):
        m.selected_metrics(a['x'], stream_ids=[999])


def test_disjoint_stream_comparison_handles_no_common_nominal_ids_as_ineligible():
    a = prepared_from_stream_keys({'x': constant_streams()})
    b = prepared_from_stream_keys({'x': {i: 'b' for i in range(200, 211)}}, parent_seeds=[200, 201, 202, 203])
    # Primary is explicitly descriptive if endpoints have different seed sets.
    assert m.paired_streams(a, b)['n_eligible'] == 1
    for orientation in (0, 1):
        result = m.paired_streams(a, b, orientation=orientation)
        assert result['n_eligible'] == 0
        assert result['delta']['collision'] is None
        assert result['selected_total_a'] == result['selected_total_b'] == [0]


def test_four_condition_change_uses_one_common_population_and_correct_sign():
    cases = [
        {'x': {100: 'a', 101: 'b'}, 'excluded': {100: 'a'}},
        {'x': {100: 'a', 101: 'a'}, 'excluded': {100: 'a', 101: 'a'}},
        {'x': {100: 'b', 101: 'b'}, 'excluded': {100: 'a', 101: 'a'}},
        {'x': {100: 'a', 101: 'b'}, 'excluded': {100: 'a', 101: 'b'}},
    ]
    prepared = [prepared_from_stream_keys(v) for v in cases]
    result = m.four_condition_change(*prepared)
    assert result['eligible_ids'] == ['x']
    assert result['n_total'] == 2 and result['n_eligible'] == 1
    assert result['delta'] == pytest.approx(-2)  # (0-1)-(1-0)


def test_intact_k8_sensitivity_equal_weights_prompts_after_within_prompt_draws():
    # x contributes only one eligible draw with effect +1; y contributes four
    # eligible draws with effect -1. Equal prompt averaging is zero, while
    # averaging the five eligible prompt/draw units directly would give -3/5.
    ca = checkpoint_from_stream_keys({'x': {}, 'y': {}})
    cb = checkpoint_from_stream_keys({'x': {}, 'y': {}})
    ca['prompts']['x']['draws'] = four_draws(['a', 'b'])
    cb['prompts']['x']['draws'] = four_draws(['a', 'a'])
    ca['prompts']['y']['draws'] = [padded(['a', 'a']) for _ in range(4)]
    cb['prompts']['y']['draws'] = [padded(['a', 'b']) for _ in range(4)]
    result = m.draw_sensitivity(m.prepare_checkpoint(ca), m.prepare_checkpoint(cb))
    assert result['eligible_ids'] == ['x', 'y']
    assert result['eligible_draw_counts'] == {1: 1, 4: 1}
    assert result['delta'] == pytest.approx(0)
    assert [r['n_eligible'] for r in result['individual_draws']] == [2, 1, 1, 1]


def test_make_block_intersects_each_orientation_population_across_seeds_separately():
    key, method, seeds = ('level1', 'qwen05b', 'graph_coloring'), 'drgrpo', [1, 2, 3, 4, 5]
    prepared = {}
    for seed in seeds:
        a = {'both': constant_streams(), 'zero': {100: 'a', 101: 'b'}, 'one': {105: 'a', 106: 'b'}, 'vary': constant_streams()}
        b = {'both': constant_streams(), 'zero': {105: 'a', 106: 'a'}, 'one': {100: 'a', 101: 'a'}, 'vary': constant_streams() if seed < 5 else {100: 'a'}}
        cell_key = (*key, method, seed)
        prepared[(cell_key, '0')] = prepared_from_stream_keys(a)
        prepared[(cell_key, '3072')] = prepared_from_stream_keys(b)
    block = m.make_block('before_after', key, method, seeds, {}, prepared)
    fixed = block['fixed_across_seed_population']
    assert fixed['distinct_streams']['eligible_ids'] == ['both', 'one', 'zero']
    assert fixed['orientation0']['eligible_ids'] == ['both', 'zero']
    assert fixed['orientation1']['eligible_ids'] == ['both', 'one']
    for name in ('distinct_streams', 'orientation0', 'orientation1'):
        assert fixed[name]['available']
        assert fixed[name]['summary']['n'] == 5
        assert fixed[name]['summary']['ci95'] is not None
        assert all(r['eligible_ids'] == fixed[name]['eligible_ids'] for r in fixed[name]['per_seed'].values())
    assert block['summaries']['orientation_mean_descriptive']['n'] == 5
    # Losing a registered endpoint cannot quietly redefine Ecap to four seeds.
    prepared.pop(((*key, method, 5), '3072'))
    incomplete = m.make_block('before_after', key, method, seeds, {}, prepared)
    assert incomplete['summaries']['distinct_streams']['n'] == 4
    assert incomplete['summaries']['distinct_streams']['ci95'] is None
    assert not incomplete['fixed_across_seed_population']['distinct_streams']['available']



def test_checkpoint_preparation_accepts_matching_neutral_singleton_request_seeds():
    cp = checkpoint_from_stream_keys({'x': constant_streams()})
    cp['prompts']['x']['request_seeds_by_draw'] = [[seed] for seed in DEFAULT_PARENT_SEEDS]
    result = m.prepare_checkpoint(cp)
    assert len(result['x']['streams']) == 11
    assert result['x']['stream_audit']['n_saved'] == 32


@pytest.mark.parametrize('request_seeds', [
    [[100], [101], [102], [999]],
    [[100], [101], [102]],
    [[100, 101], [101], [102], [103]],
])
def test_checkpoint_preparation_rejects_mismatching_neutral_singleton_metadata(request_seeds):
    cp = checkpoint_from_stream_keys({'x': constant_streams()})
    cp['prompts']['x']['request_seeds_by_draw'] = request_seeds
    with pytest.raises(ValueError):
        m.prepare_checkpoint(cp)


def test_selected_metrics_cannot_reintroduce_duplicate_streams():
    prompt = prepared_from_stream_keys({'x': constant_streams()})['x']
    with pytest.raises(ValueError):
        m.selected_metrics(prompt, stream_ids=[100, 100])


def test_old_draw_halves_are_not_disjoint_under_parent_plus_option_mapping():
    first = {parent + option for parent in (100, 101) for option in range(8)}
    second = {parent + option for parent in (102, 103) for option in range(8)}
    assert len(first) == len(second) == 9
    assert first & second == set(range(102, 109))
    lower, upper = m.split_stream_ids(first | second, first | second)
    assert len(lower) == 5 and len(upper) == 6
    assert not set(lower) & set(upper)
