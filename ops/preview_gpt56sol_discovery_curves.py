#!/usr/bin/env python3
"""Provisional complete-model preview using unchanged frozen discovery estimands.

The user explicitly selected GPT-5.6 Sol while hosted collection was underway.
This additive view never replaces the registered full three-model report.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np

MODEL = 'gpt56sol'
REGISTERED = ('gpt56sol', 'gpt54', 'grok43')
ARMS = ('original', 'neutral')
ANALYZER_SHA = 'cf4787b809674a71429b3c2e02949ba58369f12127a1790b423aed738626865a'
DRAWS_PER_ARM = 6144
REPLICATES = 20000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def frozen_analyzer(base):
    source_manifest = json.loads((base/'analysis_source_manifest.json').read_text())
    for item in source_manifest['files'].values():
        require(digest(item['path']) == item['sha256'], 'A sealed analysis dependency changed')
    source = Path(source_manifest['files']['analyzer']['path'])
    require(digest(source) == ANALYZER_SHA, 'Preview requires the unchanged registered analyzer')
    spec = importlib.util.spec_from_file_location('_sol_preview_frozen_analysis', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_entries(registry):
    entries = {}
    for item in registry['runs']:
        key = item['model_id'], item['arm']
        require(key not in entries, 'Duplicate registered hosted cohort')
        entries[key] = item
    require(set(entries) == {(model, arm) for model in REGISTERED for arm in ARMS},
            'The preview must validate the full registered three-model inventory')
    return {arm: entries[MODEL, arm] for arm in ARMS}


def status_snapshot(registry):
    result = {}
    for item in registry['runs']:
        path = Path(item['run_dir'])/'status.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        result[item['model_id']+'/'+item['arm']] = {
            'complete': state.get('complete') is True,
            'saved_responses': state.get('completed_samples', 0),
            'expected_responses': state.get('expected_samples', DRAWS_PER_ARM),
            'grading_audit_present': (path.parent/'discovery_hosted_grading_audit.json').is_file()}
    return result


def ready(snapshot):
    return all(snapshot.get(MODEL+'/'+arm, {}).get('complete') is True
               and snapshot[MODEL+'/'+arm].get('saved_responses') == DRAWS_PER_ARM
               and snapshot[MODEL+'/'+arm].get('expected_responses') == DRAWS_PER_ARM
               and snapshot[MODEL+'/'+arm].get('grading_audit_present') is True for arm in ARMS)


def bootstrap_indices(module):
    rng = np.random.default_rng(module.SEED)
    return {(level, domain): rng.integers(0, module.PROMPTS_PER_CELL,
                                        (REPLICATES, module.PROMPTS_PER_CELL))
            for level in module.LEVELS for domain in module.DOMAINS}


def analyze_completed_model(module, design, registry):
    entries = selected_entries(registry)
    runs = [module.authenticate_hosted(design, entries[arm]) for arm in ARMS]
    module.validate_payload_pair(*runs)
    model = module.analyze_pair(design, *runs, bootstrap_indices(module))
    require(model['model_id'] == MODEL and model['family'] == 'frontier', 'Unexpected selected model')
    expected_cells = {f'level{level}/{domain}' for level in module.LEVELS for domain in module.DOMAINS}
    for grading in module.GRADINGS:
        analysis = model['analyses'][grading]
        require(set(analysis['cells']) == expected_cells, 'Preview lost a registered cell')
        for arm in ARMS:
            prompts = analysis['prompts'][arm]
            require(len(prompts) == 96 and all(p['responses'] == 64 for p in prompts),
                    'Preview requires every complete 64-draw pool')
            require(sum(p['responses'] for p in prompts) == DRAWS_PER_ARM, 'Incomplete arm')
    return model


def value(x):
    return 'undefined' if x is None else f'{x:.6f}'


def estimate(row, arm, name):
    return row['metrics'][arm][name]['estimate']


def with_interval(item):
    return value(item['estimate']) + (' ['+', '.join(value(x) for x in item['ci95'])+']' if item['ci95'] else ' [undefined]')


def render_markdown(preview, module):
    rows = module.display_rows({'models': [preview['model']], 'local_seed_groups': {}}, 'frontier', 'strict')
    label = lambda row: module.DOMAIN_LABELS[row['domain']]+' L'+str(row['level'])
    lines = ['# Provisional GPT-5.6 Sol completed-model preview', '',
             '**This is one complete selected model, not the completed registered hosted panel or the final publication report.** '
             'Both Sol wordings contain all 6,144 responses: 96 problems, 16 in each of six cells, with 64 fresh draws per problem. '
             'GPT-5.4 and Grok 4.3 remain outside this preview; their collection state at creation is recorded in preview.json. '
             'The complete local panel and official full-report target are unchanged.', '',
             'The user explicitly requested this model-specific view before hosted outcome inspection by the analysis author. '
             'No problem, cell, draw, or completed outcome was used to select the model or subset. '
             'All values below call the unchanged frozen analyzer and its identical 20,000 whole-problem paired bootstrap replicates '
             '(seed 20260911), stratified by domain and level. There is no training-seed replication for this deployment.', '',
             'Primary P(k)=1−C(64−c,k)/C(64,k), D(k)=Σj[1−C(64−n_j,k)/C(64,k)], and B(k)=D(k)−P(k) '
             'rarefy each full pool without replacement. Preassigned draw-index prefix sensitivity is retained in JSON. '
             'Strict verification is primary; the single frozen normalization is a sensitivity. '
             'Intervals are pointwise and exploratory for 16 problems per cell; zero-width intervals do not establish equivalence.', '',
             '## Strict rarefaction curves', '',
             '| Cell | k | P original | P neutral | D original | D neutral | B original | B neutral |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for k in module.GRID:
            vals = [value(estimate(row, arm, f'rarefaction/{metric}/k{k}'))
                    for metric in ('pass', 'distinct', 'breadth') for arm in ARMS]
            lines.append('| '+' | '.join([label(row), str(k), *vals])+' |')
    lines += ['', '## Eight-to-64 gains', '',
              'Each O/N pair gives original/neutral. The contrast is neutral minus original with its paired 95% interval.', '',
              '| Cell | P gain O/N | D gain O/N | B gain O/N | ΔP gain [95%] | ΔD gain [95%] | ΔB gain [95%] |',
              '|---|---|---|---|---|---|---|']
    for row in rows:
        names = ['gain8to64/rarefaction/'+metric for metric in ('pass', 'distinct', 'breadth')]
        vals = [' / '.join(value(estimate(row, arm, name)) for arm in ARMS) for name in names]
        vals += [with_interval(row['metrics'][module.CONTRAST][name]) for name in names]
        lines.append('| '+' | '.join([label(row), *vals])+' |')
    lines += ['', '## Correct-pair collision in the full pools', '',
              'Collision pools correct pairs within this deployment. All certified support counts L are lower bounds, '
              'so 1/L is an upper reference for full-support uniform collision. Eligible problems have at least two correct draws. '
              'Undefined ratios remain undefined; all failures remain in the original 64-draw pools.', '',
              '| Cell | Collision O/N | Uniform upper reference O/N | Eligible problems O/N | Correct pairs O/N |',
              '|---|---|---|---|---|']
    for row in rows:
        vals = [' / '.join(value(estimate(row, arm, name)) for arm in ARMS)
                for name in ('collision_own/observed', 'collision_own/uniform_reference')]
        vals += [' / '.join(str(row['counts'][arm]['conditional_own_eligible']['2']) for arm in ARMS),
                 ' / '.join(str(row['counts'][arm]['correct_pairs']) for arm in ARMS)]
        lines.append('| '+' | '.join([label(row), *vals])+' |')
    lines += ['', '## Jointly eligible fixed-correctness comparison', '',
              'R(m) rarefies exactly m correct draws only on problems with at least m correct draws in both wordings. '
              'The eligible set can change with m. Uniform breadth at certified L is a lower reference for full-support uniform '
              'expected breadth, not a lower bound on model breadth. Observed distinct counts may exceed L. '
              'Own-arm eligibility, unconditional correctness-matched uniform references, paired contrasts, and intervals remain in JSON.', '',
              '| Cell | m | Joint eligible problems | R original | R neutral | Uniform lower reference original | Uniform lower reference neutral |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        for budget in module.GRID:
            vals = [value(estimate(row, arm, f'conditional_joint/{metric}/m{budget}'))
                    for metric in ('distinct', 'uniform_distinct') for arm in ARMS]
            lines.append('| '+' | '.join([label(row), str(budget), str(row['counts']['original']['conditional_joint_eligible'][str(budget)]), *vals])+' |')
    normalized = module.display_rows({'models': [preview['model']], 'local_seed_groups': {}}, 'frontier', 'normalized_secondary')
    lines += ['', '## Frozen-normalization sensitivity endpoints', '',
              'The full normalized curves, counts, and intervals are retained in JSON alongside strict results.', '',
              '| Cell | P8 O/N | P64 O/N | D8 O/N | D64 O/N | B8 O/N | B64 O/N |',
              '|---|---|---|---|---|---|---|']
    for row in normalized:
        vals = [' / '.join(value(estimate(row, arm, f'rarefaction/{metric}/k{k}')) for arm in ARMS)
                for metric in ('pass', 'distinct', 'breadth') for k in (8, 64)]
        lines.append('| '+' | '.join([label(row), *vals])+' |')
    lines += ['', 'Both raw grading receipts, all strict corrections and normalization rescues, native retry accounting, '
              'request controls, frozen support references, and exact source identities are bound by preview.json. '
              'The preview manifest binds this Markdown and the complete-model JSON.']
    return '\n'.join(lines)+'\n'


def validate_destination(base, output):
    for name in ('analysis_complete', 'analysis_local_complete'):
        reserved = (base/name).resolve()
        require(output.resolve() != reserved and reserved not in output.resolve().parents,
                'A provisional preview cannot write inside an official analysis directory')
    require(not output.exists(), 'Preserve an existing preview; use a new output directory')


def create_preview(base, output, module, design, registry):
    validate_destination(base, output)
    snapshot = status_snapshot(registry)
    require(ready(snapshot), 'Both complete Sol arms and frozen grading audits are required')
    model = analyze_completed_model(module, design, registry)
    preview = {
        'schema': 'modebench-discovery-completed-model-preview-v1',
        'status': 'complete_selected_model', 'publication_status': 'provisional_not_full_panel',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_basis': 'User explicitly requested GPT-5.6 Sol before hosted outcome inspection by the analysis author; no outcome-based selection.',
        'scope': {'included_models': [MODEL], 'registered_hosted_models': list(REGISTERED),
                  'excluded_hosted_models': [m for m in REGISTERED if m != MODEL],
                  'local_panel_included': False, 'complete_responses': 2*DRAWS_PER_ARM,
                  'problems': 96, 'cells': 6, 'problems_per_cell': 16, 'draws_per_problem_per_arm': 64},
        'collection_status_at_creation': snapshot,
        'official_report_target': str((base/'analysis_complete').resolve()),
        'prior_local_report': binding(base/'analysis_local_complete/analysis.json'),
        'design': design['sources'], 'prospective_analysis_plan': binding(base/'ANALYSIS_PLAN.md'),
        'registry': binding(base/'hosted_analysis_runs.json'),
        'analysis_source_freeze': binding(base/'analysis_source_manifest.json'),
        'prospective_amendments': {'hosted_execution.json': binding(base/'hosted_execution_revisions/v2/hosted_execution.json')},
        'analyzer_source': binding(module.__file__),
        'analysis_dependencies': {'native_auditor': binding(Path(module.__file__).parent/'audit_hosted_modebench_completion.py')},
        'bootstrap': {'seed': module.SEED, 'replicates': REPLICATES,
                      'unit': 'whole paired problem', 'strata': 'domain and level',
                      'indices_identical_to_registered_full_report': True},
        'model': model}
    output.mkdir(parents=True)
    source = output/'preview_source.py';shutil.copy2(__file__, source)
    preview['preview_source'] = binding(source)
    preview['preview_location'] = str(output.resolve())
    write_json(output/'preview.json', preview)
    (output/'summary.md').write_text(render_markdown(preview, module))
    write_json(output/'artifact_manifest.json', {
        'schema': 'modebench-discovery-preview-artifacts-v1', 'status': 'complete_selected_model',
        'publication_status': 'provisional_not_full_panel',
        'outputs': {name: binding(output/name) for name in ('preview.json', 'summary.md', 'preview_source.py')},
        'analyzer_source': preview['analyzer_source'], 'analysis_dependencies': preview['analysis_dependencies']})
    return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    require(1 <= args.poll_seconds <= 60, 'Use 1–60 second polls')
    base = args.base.resolve();output = (args.output or base/'gpt56sol_complete_model_preview').resolve()
    validate_destination(base, output)
    module = frozen_analyzer(base)
    design = module.authenticate_design(base)
    registry = module.validate_hosted_registry(design, base/'hosted_analysis_runs.json')
    selected_entries(registry)
    while True:
        snapshot = status_snapshot(registry)
        state = {'status': 'ready' if ready(snapshot) else 'waiting_complete_sol_arms',
                 'at_utc': datetime.now(timezone.utc).isoformat(), 'selected_model': MODEL,
                 'collection_status': snapshot, 'output': str(output), 'api_calls': 0}
        write_json(base/'gpt56sol_preview_status.json', state)
        if ready(snapshot):
            create_preview(base, output, module, design, registry)
            state['status'] = 'complete_selected_model';write_json(base/'gpt56sol_preview_status.json', state)
            print(json.dumps(state), flush=True);return
        if not args.watch:
            print(json.dumps(state), flush=True);return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
