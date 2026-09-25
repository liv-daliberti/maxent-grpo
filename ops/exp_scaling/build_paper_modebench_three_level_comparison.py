#!/usr/bin/env python3
"""Compare the Qwen-0.5B factorial across all three matched difficulty levels.

The two-level module stays as it is: Levels 1 and 2 reach the manuscript
through a frozen snapshot that several other analyses also read, and widening
its registered cohort in place would move records that are bound elsewhere.
This module adds the third level on top of those same frozen inputs and keeps
every selection rule the two-level comparison already states.

One rule does more work with a third level than with two. A domain enters the
across-domain mean only when all four methods have all five registered seeds at
*every* plotted level, so a domain still filling in at one level leaves the mean
at all of them rather than letting one level's arm rest on a different task
mixture than another's.
"""
from __future__ import annotations

import statistics

from build_paper_modebench_level_comparison import (
    DOMAINS, METHODS, METRICS, SEEDS, TARGET_STEP, admitted_checkpoints,
)

LEVELS = ('level1', 'level2', 'level3')
SELECTION_RULE = (
    'Include a domain in the across-domain mean only when every registered seed has an '
    'admitted pass-8 endpoint in all four methods at all three levels; select by '
    'availability before reading effects.'
)
AGGREGATION_RULE = (
    'Mean four sampled draws within each seed, mean the five registered seeds within each '
    'complete domain, then weight complete domains equally for all twelve series.'
)
PARTIAL_DOMAIN_POLICY = (
    'Retain available terminal arm means and exact seed n separately; never substitute '
    'earlier checkpoints or include partial blocks in the across-domain mean.'
)
PMD_MATCHING_RULE = (
    'Domains where every method clears the support bar at every plotted level.'
)


def _index(evaluations, admission):
    """Admitted four-draw terminal draws, checked against the declared admission."""
    if not any(row['level'] == 'level3' for row in evaluations):
        raise RuntimeError('three-level comparison needs Level-3 terminal draws')
    admitted = admitted_checkpoints(
        [row for row in evaluations if row['level'] in ('level1', 'level2')])
    admitted.update(_level3_checkpoints(evaluations))
    if any(key[-1] != TARGET_STEP for key in admitted):
        raise RuntimeError('terminal comparison cannot include earlier checkpoints')
    declared = {}
    for cell in admission:
        key = tuple(cell[field] for field in ('level', 'domain', 'method', 'seed'))
        if key in declared:
            raise RuntimeError('duplicate terminal admission cell')
        declared[key] = cell['admitted']
    required = {(level, domain, method, seed) for level in LEVELS
                for domain in DOMAINS for method in METHODS for seed in SEEDS}
    if set(declared) != required:
        raise RuntimeError('terminal admission must enumerate all 300 registered cells')
    expected = {key for key, value in declared.items() if value}
    if {key[:-1] for key in admitted} != expected:
        raise RuntimeError('terminal draws disagree with the declared endpoint admission')
    return admitted


def _level3_checkpoints(evaluations):
    """Deduplicate Level-3 draws under the rule the two-level module applies."""
    draws: dict[tuple, dict[int, dict]] = {}
    for row in evaluations:
        if row['level'] != 'level3':
            continue
        if row.get('evaluation_kind') != 'fixed_seed_sampled_k_neutral' or row.get('sample_count') != 8:
            continue
        key = (row['level'], row['domain'], row['method'], row['seed'], row['step'])
        if row['domain'] not in DOMAINS or row['method'] not in METHODS or row['seed'] not in SEEDS:
            raise RuntimeError('unregistered Level-3 domain/method/seed')
        current = draws.setdefault(key, {})
        draw = row['draw_index']
        if draw in current:
            if current[draw]['metrics'] != row['metrics']:
                raise RuntimeError(f'conflicting duplicate Level-3 draw: {key}, draw={draw}')
            continue
        current[draw] = row
    return {
        key: {'draws': [rows[draw] for draw in range(4)],
              'means': {metric: statistics.fmean(float(rows[draw]['metrics'][field])
                                                 for draw in range(4))
                        for metric, field in METRICS.items()}}
        for key, rows in draws.items() if set(rows) == {0, 1, 2, 3}
    }


def build_terminal_comparison(evaluations: list[dict], admission: list[dict]) -> dict:
    """Terminal pass@8 and distinct@8 for all twelve series, level by level."""
    admitted = _index(evaluations, admission)
    by_domain = {}
    for domain in DOMAINS:
        series, seed_sets = {}, []
        for level in LEVELS:
            series[level] = {}
            for method in METHODS:
                values = {seed: admitted[level, domain, method, seed, TARGET_STEP]['means']
                          for seed in SEEDS
                          if (level, domain, method, seed, TARGET_STEP) in admitted}
                seed_sets.append(set(values))
                series[level][method] = {
                    'n': len(values), 'seeds': sorted(values),
                    'per_seed': {str(seed): value for seed, value in values.items()},
                    'means': {metric: statistics.fmean(value[metric] for value in values.values())
                              for metric in METRICS} if values else {},
                }
        common = sorted(set.intersection(*seed_sets))
        by_domain[domain] = {'series': series, 'matched_seeds': common, 'n': len(common),
                             'complete_block': common == list(SEEDS)}
    complete = [domain for domain in DOMAINS if by_domain[domain]['complete_block']]
    if not complete:
        raise RuntimeError('no domain block is complete at all three levels')
    means = {level: {method: {metric: statistics.fmean(
                by_domain[domain]['series'][level][method]['means'][metric]
                for domain in complete) for metric in METRICS}
             for method in METHODS} for level in LEVELS}
    return {
        'model': 'Qwen2.5-0.5B-Instruct', 'target_step': TARGET_STEP, 'training_pass': 8,
        'levels': list(LEVELS),
        'selection_rule': SELECTION_RULE,
        'aggregation_rule': AGGREGATION_RULE,
        'partial_domain_policy': PARTIAL_DOMAIN_POLICY,
        'uncertainty': 'Descriptive seed means; no confidence intervals.',
        'complete_domains': complete,
        'partial_domains': [domain for domain in DOMAINS if domain not in complete],
        'seeds_per_complete_domain': len(SEEDS),
        'admitted_terminal_cells_by_level': {
            level: sum(key[0] == level for key in admitted) for level in LEVELS},
        'expected_terminal_cells_per_level': len(DOMAINS) * len(METHODS) * len(SEEDS),
        'domain_results': by_domain, 'means': means,
    }


def level3_pmd_arms(pmd_cells: list[dict]) -> dict:
    """Arm-level PCMD under the rule the training record already applies.

    A seed enters its arm's mean when the terminal checkpoint clears the support
    bar, and an arm is reportable when at least one seed does. Requiring support
    at step 0 as well would drop blocks because the frozen model was too weak to
    measure, which says nothing about the arms being compared.
    """
    arms = {}
    for domain in DOMAINS:
        for method in METHODS:
            reportable = [cell for cell in pmd_cells
                          if cell['domain'] == domain and cell['method'] == method
                          and cell['reportable'] and cell['pmd'] is not None]
            arms[domain, method] = {
                'terminal_reportable': bool(reportable),
                'terminal_seeds': len(reportable),
                'seeds': sorted(cell['seed'] for cell in reportable),
                'pmd_after': statistics.fmean(cell['pmd'] for cell in reportable) if reportable else None,
            }
    return arms


def build_pmd_comparison(training_arms: list[dict], pmd_cells: list[dict],
                         min_defined_prompts: int, scale: str = 'qwen05b') -> dict:
    """Terminal PCMD per method and level, on domains matched across all three.

    Only domains where all four methods clear the support bar at *every* level
    can enter: otherwise a method's mean would rest on an easier subset than its
    control's, which is the accuracy coupling PCMD exists to remove.
    """
    indexed = {(arm['level'], arm['scale'], arm['domain'], arm['method']): arm
               for arm in training_arms}
    level3 = level3_pmd_arms(pmd_cells)
    matched = [domain for domain in DOMAINS
               if all(indexed.get((level, scale, domain, method), {}).get('terminal_reportable')
                      for level in ('level1', 'level2') for method in METHODS)
               and all(level3[domain, method]['terminal_reportable'] for method in METHODS)]
    if not matched:
        raise RuntimeError('no domain supports PCMD for every method at every level')

    def value(level, domain, method):
        if level == 'level3':
            return level3[domain, method]['pmd_after']
        return indexed[level, scale, domain, method]['pmd_after']

    return {
        'metric': 'pairwise correct-mode diversity (PCMD)',
        'matched_domains': matched,
        'matching_rule': PMD_MATCHING_RULE,
        'min_defined_prompts': min_defined_prompts,
        'means': {level: {method: statistics.fmean(value(level, domain, method)
                                                   for domain in matched)
                          for method in METHODS} for level in LEVELS},
        'per_domain': {level: {method: {domain: value(level, domain, method)
                                        for domain in matched}
                               for method in METHODS} for level in LEVELS},
        'level3_arms': {domain: {method: level3[domain, method] for method in METHODS}
                        for domain in DOMAINS},
    }


BASELINE_RULE = (
    'The untrained checkpoint is one policy per level, not four, so its point averages '
    'every admitted step-0 cell of that level on the same domains as the trained points '
    'beside it. PCMD uses the cells that clear the support bar at step 0; where no cell '
    'in a matched domain does, the level has no untrained breadth point rather than a '
    'point standing on too little support.'
)


def build_baseline_points(admission: list[dict], pmd_cells: list[dict],
                          complete_domains: list[str], matched_domains: list[str]) -> dict:
    """One untrained point per level: where that level's training starts.

    Accuracy and breadth are averaged on the same domain bases as the trained
    points, so the untrained mark is comparable to the marks it sits under and
    not to a different task mixture.
    """
    endpoints, breadth = {}, {}
    for row in admission:
        if row['admitted']:
            endpoints.setdefault((row['level'], row['domain']), []).append(row['endpoint'])
    for cell in pmd_cells:
        if cell['reportable'] and cell['pmd'] is not None:
            breadth.setdefault((cell['level'], cell['domain']), []).append(cell['pmd'])
    points = {}
    for level in LEVELS:
        accuracy_domains = [d for d in complete_domains if endpoints.get((level, d))]
        breadth_domains = [d for d in matched_domains if breadth.get((level, d))]
        points[level] = {
            'pass8': statistics.fmean(
                statistics.fmean(row['pass8'] for row in endpoints[level, domain])
                for domain in accuracy_domains) if len(accuracy_domains) == len(complete_domains) else None,
            'distinct8': statistics.fmean(
                statistics.fmean(row['distinct8'] for row in endpoints[level, domain])
                for domain in accuracy_domains) if len(accuracy_domains) == len(complete_domains) else None,
            'pmd': statistics.fmean(
                statistics.fmean(breadth[level, domain]) for domain in breadth_domains
            ) if len(breadth_domains) == len(matched_domains) else None,
            'accuracy_domains': accuracy_domains,
            'breadth_domains': breadth_domains,
            'cells': sum(len(endpoints.get((level, d), ())) for d in complete_domains),
            'breadth_cells': sum(len(breadth.get((level, d), ())) for d in matched_domains),
        }
    return {
        'step': 0,
        'rule': BASELINE_RULE,
        'accuracy_domains': list(complete_domains),
        'breadth_domains': list(matched_domains),
        'points': points,
    }


PAIRS = (('drgrpo', 'replay_drgrpo'), ('maxrl', 'replay_maxrl'))
GAIN_RULE = (
    'Each gain is the replay arm minus its own fresh objective at the same level, taken '
    'from the means already reported for those arms, so both sides of a bar share the '
    'level, the prompts, the seeds and the domain basis. Gains are descriptive '
    'differences of domain-equal-weighted seed means and carry no interval.'
)

def build_replay_gains(terminal: dict, pmd: dict) -> dict:
    """Replay minus its own control, per level, on both reported axes.

    The figure this feeds asks one question -- what does replay add -- so it
    plots the difference rather than four absolute positions the reader has to
    subtract by eye. Nothing new is measured: every value here is a difference
    of two means already in this record.
    """
    levels = {}
    for level in LEVELS:
        pairs = {}
        for control, replay in PAIRS:
            pairs[replay] = {
                'control': control,
                'pass8': (terminal['means'][level][replay]['pass8']
                          - terminal['means'][level][control]['pass8']),
                'distinct8': (terminal['means'][level][replay]['distinct8']
                              - terminal['means'][level][control]['distinct8']),
                'pmd': pmd['means'][level][replay] - pmd['means'][level][control],
            }
        levels[level] = pairs
    return {
        'rule': GAIN_RULE,
        'accuracy_domains': list(terminal['complete_domains']),
        'breadth_domains': list(pmd['matched_domains']),
        'levels': levels,
    }
