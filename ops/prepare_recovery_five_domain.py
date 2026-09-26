#!/usr/bin/env python3
"""Freeze a five-domain recovery protocol before any recovery call is made.

App.~\\ref{app:pantry-adaptation} measures what a changed task costs to recover
from -- extra calls, their tokens, and whether a tuned temperature or a
diversity prompt would have done as well -- in PantryPlan alone, because that
half needs generation. This freezes the same protocol for all five domains over
the E72 cohort, whose checkpoints are local and whose reference-setting draws
already supply the saved portfolio without generating it again.

Everything an outcome could influence is fixed here: which prompts are tested,
which are held back for choosing a temperature, which options are withdrawn, how
a revised task is stated, and how a recovery answer is graded. The worker may
only generate and grade.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / 'ops', ROOT / 'src'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from followup_metrics import atomic_new, file_sha, sha  # noqa: E402
from frontier_modebench_contract import profile_metadata  # noqa: E402
import portfolio_withdrawals as pw  # noqa: E402
from prepare_portfolio_withdrawals import first_line_specs  # noqa: E402

FROZEN = ROOT / 'artifacts/modebench_portfolio_withdrawals_20260917'
CONTAINER = ROOT / 'var/artifacts/mode_diversity_curves/verified_keys_by_step.jsonl.gz'
CURVES = ROOT / 'paper/results/mode_diversity_curves.json'
BASE = ROOT / 'artifacts/modebench_recovery_five_domain_20260917'
CODE = ('ops/prepare_recovery_five_domain.py', 'ops/run_recovery_five_domain.py',
        'ops/portfolio_withdrawals.py', 'src/oat_drgrpo/math_grader.py',
        'src/oat_drgrpo/mathir.py', 'src/oat_drgrpo/pantry_plan.py')
#: Two matched seeds per domain, as the PantryPlan protocol uses.
SEEDS_PER_DOMAIN = 2
SEED_ORDER = (43, 44, 45, 46, 47)
#: The fresh objective and the replay arm added to it.
ARMS = {'drgrpo': 'control', 'replay_drgrpo': 'replay'}
TEST_PROBLEMS = 32
DEV_PROBLEMS = 16
#: Withdrawals per test problem, taken in the frozen table's order.
WITHDRAWALS_PER_PROBLEM = 4
TEMPERATURE_GRID = (0.7, 1.0, 1.3)
MAX_RECOVERY_CALLS = 8
#: How a revised task is stated, and the noun the recovery interface uses. Both
#: are fixed text; no wording is chosen after an outcome is seen.
NOUN = {'graph_coloring': 'colouring', 'countdown': 'expression',
        'python_factors': 'function', 'mathir': 'action sequence',
        'pantry_plan': 'plan'}


def update_sentence(domain, option):
    """The sentence a revised task adds, from the withdrawn option alone."""
    target = option[1]
    if domain == 'graph_coloring':
        vertex, colour = target.split(':')
        return (f'Colour {colour} is no longer available for vertex {vertex}. '
                'Every other vertex, edge and fixed digit stays the same.')
    if domain == 'countdown':
        word = {'add': 'addition', 'sub': 'subtraction', 'mul': 'multiplication',
                'div': 'division'}[target]
        return (f'The {word} operation is no longer available. The numbers and the '
                'target stay the same, and every number must still be used exactly once.')
    if domain == 'python_factors':
        return (f'The value {target} is no longer accepted as a returned divisor for any '
                'listed input. Every other requirement stays the same.')
    if domain == 'mathir':
        return (f'Action {target} has been withdrawn from the menu and may no longer be '
                'used. The equation and the remaining actions stay the same.')
    if domain == 'pantry_plan':
        return (f'An ingredient outage has occurred: {target} is unavailable. Do not use '
                'it. All other quantities, nutrition targets, and rules stay the same.')
    raise ValueError('unsupported domain ' + domain)


def cohort():
    """Matched control and replay checkpoints from the reported factorial.

    The cohort is the one Appendix~M scores, read at the same terminal endpoints:
    Level-1 Qwen2.5-0.5B, Dr.GRPO against Re:Dr. Two seeds per domain are used,
    chosen as the two lowest whose matched pair still has local weights --- an
    availability rule, applied before any outcome here is read. The saved
    portfolio is that cell's own terminal draw block, so the ordinary strategy
    generates nothing.
    """
    curve = {}
    for entry in json.loads(CURVES.read_text())['curves']:
        if entry['level'] != 'level1' or entry['scale'] != 'qwen05b':
            continue
        point = max(entry['points'], key=lambda p: p['step'])
        curve[(entry['domain'], entry['method'], int(entry['seed']))] = point['step']
    cells = {}
    with gzip.open(CONTAINER, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('record_kind') != 'source':
                continue
            if record['level'] != 'level1' or record['scale'] != 'qwen05b':
                continue
            if record['method'] not in ARMS:
                continue
            key = (record['domain'], record['method'], int(record['seed']))
            step = max(draw['step'] for draw in record['draws'])
            if curve.get(key) != step:
                continue
            weights = sorted(Path(record['run_dir']).glob(
                'debug_job*/saved_models/step_*/model.safetensors'))
            if not weights:
                continue
            cells[key] = {'run_dir': record['run_dir'], 'draws': record['path'],
                          'model_path': str(weights[-1].parent), 'terminal_step': step}
    out, portfolios = {}, {}
    for domain in pw.DOMAINS:
        matched = [seed for seed in SEED_ORDER
                   if all((domain, method, seed) in cells for method in ARMS)]
        if len(matched) < SEEDS_PER_DOMAIN:
            raise ValueError('too few matched seeds with local weights for ' + domain)
        checkpoints = []
        for seed in matched[:SEEDS_PER_DOMAIN]:
            for method, label in ARMS.items():
                cell = cells[(domain, method, seed)]
                model = Path(cell['model_path'])
                checkpoints.append({
                    'label': f'{domain}/{label}_s{seed}', 'domain': domain, 'arm': label,
                    'seed': seed, 'method': method, 'model_path': str(model),
                    'run_dir': cell['run_dir'], 'terminal_step': cell['terminal_step'],
                    'files': [{'name': path.name, 'sha256': file_sha(path)}
                              for path in sorted(model.iterdir()) if path.is_file()]})
                portfolios[f'{domain}/{label}_s{seed}'] = {
                    'path': cell['draws'], 'sha256': file_sha(cell['draws']),
                    'step': cell['terminal_step'], 'draw_index': 0}
        out[domain] = checkpoints
    return out, portfolios, {'seeds_per_domain': SEEDS_PER_DOMAIN,
                             'seed_rule': 'The two lowest seeds whose matched control and replay '
                                          'weights are both retained locally.',
                             'arms': ARMS, 'scale': 'qwen05b', 'level': 1}


def _dev_row(job):
    domain, index, reference = job
    row = pw.prompt_row(domain, json.loads(reference))
    return domain, index, [o for o in row['options'] if o['feasible'] and o['binding']]


def prepare(rewrite, workers):
    if rewrite:
        (BASE / 'inputs.json').unlink(missing_ok=True)
    frozen = json.loads((FROZEN / 'inputs.json').read_text())
    payload = (FROZEN / 'table.json.gz').read_bytes()
    if hashlib.sha256(payload).hexdigest() != frozen['table_sha256']:
        raise ValueError('frozen withdrawal table changed')
    table = {(r['domain'], r['prompt_index']): r
             for r in json.loads(gzip.decompress(payload))['rows'] if r['level'] == 'level1'}
    checkpoints, saved_sources, cohort_note = cohort()
    started = time.time()

    tasks = {}
    for domain in pw.DOMAINS:
        source = frozen['prompt_identity']['level1|' + domain]['spec_source']
        references = first_line_specs(source)
        # The draw record's ``prompt`` is the dataset row's ``problem`` verbatim,
        # and its ``reference`` the row's ``answer``; both are checked by hash
        # against the frozen prompt identity above.
        problems = [entry['prompt'] for entry in sorted(
            json.loads(Path(source).open().readline())['prompts'],
            key=lambda entry: entry['prompt_index'])]
        order = sorted(range(len(references)),
                       key=lambda index: sha(['recovery-five-domain', domain, index]))
        split = {'test': order[:TEST_PROBLEMS],
                 'dev': order[TEST_PROBLEMS:TEST_PROBLEMS + DEV_PROBLEMS]}
        if set(split['test']) & set(split['dev']):
            raise ValueError('calibration and test problems overlap')
        entries = []
        for kind, indices in split.items():
            for index in indices:
                row = {'problem': problems[index], 'answer': references[index],
                       'prompt_index': index}
                options = [o for o in table[(domain, index)]['options']
                           if o['feasible'] and o['binding']]
                if kind == 'test':
                    options = options[:WITHDRAWALS_PER_PROBLEM]
                if not options:
                    continue
                entries.append({'id': f'{kind}_{index:03d}', 'split': kind, 'row': row,
                                'census': table[(domain, index)]['support'],
                                'withdrawals': [{'option': o['option'],
                                                 'update': update_sentence(domain, tuple(o['option'])),
                                                 'surviving': o['surviving']} for o in options]})
        tasks[domain] = entries

    inputs = {
        'schema': 'modebench-recovery-five-domain-inputs-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'repo_root': str(ROOT), 'base': str(BASE),
        'frozen_withdrawals': {'path': str(FROZEN / 'inputs.json'),
                               'sha256': file_sha(FROZEN / 'inputs.json'),
                               'table_sha256': frozen['table_sha256']},
        'outcome_container': {'path': str(CONTAINER), 'sha256': file_sha(CONTAINER)},
        'curve_archive': {'path': str(CURVES), 'sha256': file_sha(CURVES)},
        'cohort': cohort_note,
        'checkpoints': [c for domain in pw.DOMAINS for c in checkpoints[domain]],
        'tasks': tasks, 'saved_portfolio': saved_sources,
        'split': {'test_problems': TEST_PROBLEMS, 'dev_problems': DEV_PROBLEMS,
                  'rule': 'Disjoint deterministic split of the 128 Level-1 evaluation prompts; '
                          'the temperature is chosen on the dev problems only.'},
        'withdrawals_per_problem': WITHDRAWALS_PER_PROBLEM,
        'temperature_grid': list(TEMPERATURE_GRID), 'max_recovery_calls': MAX_RECOVERY_CALLS,
        'strategies': ['ordinary', 'temperature', 'diversity_prompt'],
        'nouns': NOUN,
        'interface': {
            'template': {domain: profile_metadata(1, domain)['template_name'] for domain in pw.DOMAINS},
            'rule': 'The prompt contract\'s Level-1 surface per domain, confirmed against the '
                    'rendered prompt each cell retains. PantryPlan is the support-mask surface: '
                    'the policy returns six binary digits and a trusted environment projects '
                    'exact quantities onto the selected rows.',
            'grading': 'A fresh generation is graded through the contract, so PantryPlan\'s mask '
                       'is projected before verification. A saved response is graded with the '
                       'plain outcome key instead, because the retained draws store that '
                       'projection rather than the policy\'s mask; projecting it twice would '
                       'reject every valid PantryPlan answer. The worker regrades every saved '
                       'response and stops if its grade disagrees with the retained one.'},
        'grading': 'A recovery answer counts when the original executable verifier accepts it '
                   'and it survives the withdrawal under the frozen predicate. Withdrawals are '
                   'exclusion only, so that is exactly validity for the revised task.',
        'counts': {domain: {'test': sum(t['split'] == 'test' for t in tasks[domain]),
                            'dev': sum(t['split'] == 'dev' for t in tasks[domain]),
                            'test_withdrawals': sum(len(t['withdrawals']) for t in tasks[domain]
                                                    if t['split'] == 'test')}
                   for domain in pw.DOMAINS},
        'code_sha256': {name: file_sha(ROOT / name) for name in CODE},
        'outcomes_read': False,
    }
    atomic_new(BASE / 'inputs.json', inputs)
    print(json.dumps({'stage': 'frozen', 'checkpoints': len(inputs['checkpoints']),
                      'counts': inputs['counts'],
                      'seconds': round(time.time() - started, 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    prepare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
