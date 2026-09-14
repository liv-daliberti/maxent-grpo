#!/usr/bin/env python3
"""Authenticate and plot the complete frozen-Qwen, five-level Methods grid.

Publication requires all 100 registered model/level/domain cells. A partial
preview may only be written below artifacts/ and visibly reports missing cells.
Receipt validation and all plotting are offline; this module never samples.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SCHEMA = 'modebench-base-grid-dataset-registry-v1'
RECEIPT_SCHEMA = 'modebench-base-grid-independent-v1'

SOURCE_SCHEMA = 'modebench-base-grid-figure-input-v1'
RECORD_SCHEMA = 'paper-modebench-base-grid-v1'
BASE = ROOT / 'artifacts/modebench_base_level_grid_20260911'
SOURCE = BASE / 'figure_source_runtime_v1.json'
OUTPUT = ROOT / 'paper/figures/modebench_base_level_grid'
VALIDATOR = BASE / 'collection_v3/code_snapshot/ops/evaluate_modebench_base_grid.py'
COLLECTION_PLAN = BASE / 'collection_v3/plan.json'
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry')
LEVELS = ('level1', 'level2', 'level3', 'level4', 'level5')
MODELS = ('05b', '3b', '7b', '14b')
EXPECTED = {(model, level, domain) for model in MODELS for level in LEVELS for domain in DOMAINS}
TITLES = {'countdown': 'Countdown', 'graph_coloring': 'Graph Coloring',
          'python_factors': 'Python Factors', 'mathir': 'MathIR', 'pantry': 'PantryPlan'}
COLORS = dict(zip(LEVELS, ('#4477AA', '#EE7733', '#228833', '#CCBB44', '#AA3377')))
MARKERS = {'05b': 'o', '3b': 's', '7b': '^', '14b': 'D'}
AREAS = {'05b': 16, '3b': 24, '7b': 31, '14b': 40}
MODEL_NAMES = {'05b': '0.5B', '3b': '3B', '7b': '7B', '14b': '14B'}
FIGSIZE = (7.2, 3.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def binding(path):
    return {'path': portable(path), 'sha256': file_sha(path)}


def authenticate(item, label):
    require(isinstance(item, dict) and isinstance(item.get('path'), str)
            and isinstance(item.get('sha256'), str), f'{label}: path and SHA256 binding required')
    path = Path(item['path'])
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and file_sha(path) == item['sha256'], f'{label}: missing or stale file binding')
    return path.resolve()


def key(item):
    return item.get('model_label'), item.get('level'), item.get('domain')


def reconstruct_metrics(receipt):
    """Average complete K=8 groups first; never pool the four groups into K=32."""
    prompts = receipt.get('prompt_results')
    require(isinstance(prompts, list) and len(prompts) == 128,
            'each plotted cell requires all 128 saved prompts')
    prompt_records = []
    for index, prompt in enumerate(prompts):
        require(isinstance(prompt, dict) and prompt.get('row_index') == index,
                'complete original prompt ordering required')
        draws = prompt.get('draws')
        require(isinstance(draws, list) and len(draws) == 4,
                'each prompt requires four independent eight-draw groups')
        groups = []
        for draw in draws:
            attempts = draw.get('attempts')
            require(isinstance(attempts, list) and len(attempts) == 8,
                    'each independent group requires all eight saved attempts')
            require(all(isinstance(attempt, dict) and type(attempt.get('verified')) is bool
                        and 'canonical_key' in attempt
                        and attempt['verified'] == (attempt['canonical_key'] is not None)
                        for attempt in attempts), 'verified flags and canonical keys disagree')
            canonical = {canonical_sha(attempt['canonical_key']) for attempt in attempts if attempt['verified']}
            groups.append({'pass8': float(bool(canonical)), 'distinct8': len(canonical)})
        values = {metric: statistics.mean(group[metric] for group in groups)
                  for metric in ('pass8', 'distinct8')}
        prompt_records.append({'row_index': index, 'row_sha256': prompt.get('row_sha256'),
                               'groups': groups, **values})
    metrics = {metric: statistics.mean(prompt[metric] for prompt in prompt_records)
               for metric in ('pass8', 'distinct8')}
    require(0 <= metrics['pass8'] <= 1 and 0 <= metrics['distinct8'] <= 8,
            'reconstructed coordinates outside valid bounds')
    require(all(type(receipt.get('metrics', {}).get(metric)) in (int, float)
                and math.isclose(receipt['metrics'][metric], value, rel_tol=0, abs_tol=1e-12)
                for metric, value in metrics.items()),
            'saved summary differs from independently reconstructed canonical-set metrics')
    return metrics, prompt_records


def read_registry(pin):
    path = authenticate(pin, 'dataset registry')
    registry = json.loads(path.read_text())
    require(registry.get('schema') == REGISTRY_SCHEMA, 'unsupported dataset registry')
    for field, expected in [('domains', DOMAINS), ('levels', LEVELS), ('model_labels', MODELS)]:
        values = registry.get(field)
        require(isinstance(values, list) and len(values) == len(expected) and set(values) == set(expected),
                'registry grid differs from requested Methods grid')
    datasets = registry.get('datasets')
    require(isinstance(datasets, list), 'dataset registry is missing its admitted datasets')
    dataset_keys = [(item.get('level'), item.get('domain')) for item in datasets]
    require(len(set(dataset_keys)) == len(dataset_keys)
            and set(dataset_keys) <= {(level, domain) for level in LEVELS for domain in DOMAINS},
            'duplicate or extraneous registry dataset')
    require(all(item.get('status') == 'admitted' and item.get('split') == 'eval'
                and item.get('rows') == 128 for item in datasets),
            'development or incomplete datasets cannot enter the Methods grid')
    models = registry.get('models')
    require(isinstance(models, dict) and set(models) == set(MODELS), 'all four registered model identities required')
    for model, item in models.items():
        require(item.get('label') == model and isinstance(item.get('identity'), dict)
                and item.get('identity_sha256') == canonical_sha(item['identity']),
                'registered model identity hash differs')
    return path, registry, set(dataset_keys)


def inspect_python(interpreter):
    """Record the numerical stdlib implementation as well as the Python binary."""
    script = ("import hashlib,json,statistics,sys;from pathlib import Path;"
              "p=Path(statistics.__file__).resolve();"
              "print(json.dumps({'executable':sys.executable,'version':sys.version,"
              "'implementation':sys.implementation.name,'version_info':list(sys.version_info[:3]),"
              "'statistics':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}))")
    completed = subprocess.run([str(interpreter), '-I', '-c', script],
                               capture_output=True, text=True, check=False)
    require(completed.returncode == 0, 'cannot inspect the collection Python interpreter')
    return json.loads(completed.stdout)


def plan_interpreter(plan, validator_path):
    cells = plan.get('cells')
    require(isinstance(cells, list) and bool(cells), 'collection plan has no interpreter-bound cells')
    commands = [cell.get('command') for cell in cells]
    require(all(isinstance(command, list) and len(command) >= 3
                and isinstance(command[0], str) and Path(command[0]).is_absolute()
                and Path(command[2]).resolve() == Path(validator_path).resolve() for command in commands),
            'collection plan does not bind every cell to the frozen validator')
    paths = {command[0] for command in commands}
    require(len(paths) == 1, 'collection plan mixes Python interpreters')
    return Path(next(iter(paths)))


def bind_validator_runtime(plan_path, validator_path):
    plan = json.loads(Path(plan_path).read_text())
    interpreter = plan_interpreter(plan, validator_path)
    require(interpreter.is_file(), 'collection interpreter is missing')
    return {'collection_plan': binding(plan_path),
            'interpreter': {'path': str(interpreter), 'resolved_path': str(interpreter.resolve()),
                            'sha256': file_sha(interpreter)},
            'runtime_metadata': inspect_python(interpreter)}


def authenticate_runtime(runtime, validator_path):
    require(isinstance(runtime, dict), 'bound collection validator runtime is required')
    plan_path = authenticate(runtime.get('collection_plan'), 'collection plan')
    plan = json.loads(plan_path.read_text())
    expected = plan_interpreter(plan, validator_path)
    item = runtime.get('interpreter')
    require(isinstance(item, dict) and item.get('path') == str(expected),
            'validator interpreter differs from collection plan')
    require(expected.is_file() and str(expected.resolve()) == item.get('resolved_path')
            and file_sha(expected) == item.get('sha256'),
            'collection Python binary path or hash changed')
    require(inspect_python(expected) == runtime.get('runtime_metadata'),
            'collection Python version or numerical stdlib changed')
    return expected


def make_source(registry_path, receipt_paths, validator_path=VALIDATOR, registry_extensions=(),
                collection_plan_path=COLLECTION_PLAN):
    """Create a reviewable hash index from explicitly selected receipt files."""
    receipts = []
    for path in receipt_paths:
        path = Path(path)
        item = json.loads(path.read_text())
        require(item.get('schema') == RECEIPT_SCHEMA and item.get('status') == 'complete',
                'source index accepts only completed base-grid receipts')
        receipts.append({**binding(path), 'model_label': item.get('model_label'),
                         'level': item.get('level'), 'domain': item.get('domain')})
    return {'schema': SOURCE_SCHEMA, 'registry': binding(registry_path),
            'registry_extensions': [binding(path) for path in registry_extensions],
            'validator': binding(validator_path),
            'validator_runtime': bind_validator_runtime(collection_plan_path, validator_path), 'receipts': receipts}


def validate_receipts(validator_path, entries, runtime):
    """Run frozen authentication in a fresh interpreter, independent of live imports."""
    validator_path = Path(validator_path).resolve()
    require(validator_path.name == 'evaluate_modebench_base_grid.py'
            and validator_path.is_relative_to((ROOT / 'artifacts').resolve()),
            'receipt validator must be the frozen evaluator below artifacts/')
    interpreter = authenticate_runtime(runtime, validator_path)
    if not entries:
        return []
    requests = [{'path': str(authenticate(item, 'receipt')), 'sha256': item['sha256']}
                for item in entries]
    script = r'''import hashlib,json,sys
from pathlib import Path
validator=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(validator.parent))
sys.path.insert(0,str(validator.parent.parent/'src'))
import evaluate_modebench_base_grid as evaluator
if Path(evaluator.__file__).resolve()!=validator:
    raise ValueError('Imported evaluator differs from frozen validator')
results=[]
for item in json.load(sys.stdin):
    path=Path(item['path']); content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=item['sha256']:
        raise ValueError('Receipt changed during authentication')
    receipt=json.loads(content)
    results.append(evaluator.validate_seed_receipt(receipt))
print(json.dumps(results,allow_nan=False))
'''
    completed = subprocess.run([str(interpreter), '-I', '-c', script, str(validator_path)],
                               input=json.dumps(requests), capture_output=True, text=True, check=False)
    require(completed.returncode == 0,
            'Frozen receipt authentication failed: ' + completed.stderr.strip()[-2000:])
    try:
        results = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError('Frozen validator did not return an audit list') from error
    require(isinstance(results, list) and len(results) == len(entries),
            'Frozen validator returned incomplete audit coverage')
    return results


def build_record(source=SOURCE, *, allow_partial=False):
    source = Path(source).resolve()
    manifest = json.loads(source.read_text())
    require(manifest.get('schema') == SOURCE_SCHEMA, 'unsupported base-grid figure input')
    extensions = manifest.get('registry_extensions', [])
    require(isinstance(extensions, list), 'registry_extensions must be a list of immutable registry bindings')
    pins = [manifest.get('registry'), *extensions]
    registries, datasets = {}, {}
    registry = None
    for pin in pins:
        registry_path, current, _ = read_registry(pin)
        require(registry_path not in registries, 'duplicate registry source')
        if registry is None:
            registry = current
        else:
            require(current['models'] == registry['models'], 'registry extension changes the frozen model identities')
        registries[registry_path] = deepcopy(pin)
        for entry in current['datasets']:
            cell = entry['level'], entry['domain']
            require(datasets.setdefault(cell, entry) == entry,
                    'registry extension changes a previously registered dataset identity')
    admitted = set(datasets)
    entries = manifest.get('receipts')
    require(isinstance(entries, list) and all(isinstance(item, dict) for item in entries),
            'a bound receipt list is required')
    keys = [key(item) for item in entries]
    require(len(set(keys)) == len(keys), 'duplicate model/level/domain receipt')
    require(set(keys) <= EXPECTED, 'receipt outside the requested model/level/domain grid')
    missing = sorted(EXPECTED - set(keys))
    if missing and not allow_partial:
        raise ValueError(f'Publication requires all 100 complete authenticated cells; {len(missing)} are missing.')
    if not allow_partial:
        require(len(admitted) == 25, 'publication requires admitted datasets at all five levels')
    validator_path = authenticate(manifest.get('validator'), 'frozen receipt validator')
    ordered_entries = sorted(entries, key=lambda item: key(item))
    audits = validate_receipts(validator_path, ordered_entries, manifest.get('validator_runtime'))
    points, model_identities, dataset_identities = [], {}, {}
    for entry, audit in zip(ordered_entries, audits):
        cell = key(entry)
        require(cell[1:] in admitted, 'receipt dataset has no admitted registry entry')
        path = authenticate(entry, '/'.join(cell))
        receipt = json.loads(path.read_text())
        require(key(receipt) == cell, 'receipt identity differs from source index')
        require(audit.get('prompts') == 128 and audit.get('draws_per_prompt') == 4
                and audit.get('distinct_request_blocks') == 512
                and audit.get('distinct_child_seeds') == 4096,
                'receipt lacks the full independent K=8 x 4 sampling evidence')
        identity = receipt['identity']
        dataset = identity['dataset_binding']
        receipt_registry = Path(dataset['source_manifest_path']).resolve()
        require(receipt_registry in registries
                and dataset['source_manifest_sha256'] == registries[receipt_registry]['sha256'],
                'receipt is bound to an unlisted dataset registry')
        dataset_entry = {field: value for field, value in dataset.items()
                         if field not in ('source_manifest_path', 'source_manifest_sha256')}
        require(dataset_entry == datasets[cell[1:]],
                'receipt dataset differs from the admitted immutable dataset entry')
        model = identity['model']
        pinned_model = registry['models'][cell[0]]
        require(all(model.get(field) == value for field, value in pinned_model['identity'].items())
                and model.get('path') == pinned_model.get('path')
                and model.get('revision') == pinned_model.get('revision')
                and model.get('repository') == pinned_model.get('model_repository'),
                'receipt differs from the registered upstream model')
        previous_model = model_identities.setdefault(cell[0], deepcopy(model))
        require(previous_model == model, 'model identity varies across levels or domains')
        dataset_key = cell[1] + '/' + cell[2]
        previous_dataset = dataset_identities.setdefault(dataset_key, deepcopy(dataset_entry))
        require(previous_dataset == dataset_entry, 'model scales use different evaluation datasets in a cell')
        metrics, prompts = reconstruct_metrics(receipt)
        points.append({'model_label': cell[0], 'level': cell[1], 'domain': cell[2],
                       'metrics': metrics, 'receipt': binding(path),
                       'receipt_identity_sha256': receipt['identity_sha256'],
                       'dataset_registry': deepcopy(registries[receipt_registry]),
                       'prompt_metrics_sha256': canonical_sha(prompts), 'prompt_metrics': prompts,
                       'independence_audit': audit,
                       'code_sha256': deepcopy(identity['code_sha256']),
                       'interface': deepcopy(identity['interface']),
                       'runtime': deepcopy(identity['runtime']),
                       'sampling_engine': deepcopy(identity['sampling_engine'])})
    missing_cells = [{'model_label': model, 'level': level, 'domain': domain,
                      'reason': 'awaiting_complete_measurement' if (level, domain) in admitted else 'awaiting_dataset_admission'}
                     for model, level, domain in missing]
    coverage = {domain: {'complete': sum(point['domain'] == domain for point in points), 'required': 20,
                         'missing': [item for item in missing_cells if item['domain'] == domain]}
                for domain in DOMAINS}
    return {'schema': RECORD_SCHEMA, 'status': 'complete' if not missing else 'partial_preview',
            'source': binding(source), 'registry': deepcopy(manifest['registry']),
            'registry_extensions': deepcopy(extensions),
            'validator': deepcopy(manifest['validator']),
            'validator_runtime': deepcopy(manifest['validator_runtime']),
            'renderer': binding(Path(__file__)), 'points': points, 'missing_cells': missing_cells,
            'complete_cells': len(points), 'required_cells': 100, 'coverage_by_domain': coverage,
            'models': model_identities, 'datasets': dataset_identities,
            'sampling': {'prompts_per_cell': 128, 'independent_groups_per_prompt': 4,
                         'draws_per_group': 8, 'samples_per_cell': 4096,
                         'complete_grid_samples': 409600, 'temperature': 1.0, 'top_p': 1.0,
                         'max_output_tokens': 192, 'model_role': 'frozen upstream Qwen2.5-Instruct before ModeBench training'},
            'metrics': {'pass8': 'Mean indicator of at least one verified canonical answer in each eight-draw group, averaged across four independent groups and 128 prompts.',
                        'distinct8': 'Mean number of unique verified canonical keys within each eight-draw group, averaged across four independent groups and 128 prompts.',
                        'zero_success_groups_retained': True, 'groups_pooled_into_32_draws': False},
            'display': {'domains': list(DOMAINS), 'levels': list(LEVELS), 'model_labels': list(MODELS),
                        'x': 'pass@8', 'y': 'distinct@8', 'x_limits': [0, 1], 'y_limits': [0, 8],
                        'level_colors': COLORS, 'model_markers': MARKERS, 'figure_inches': list(FIGSIZE),
                        'jitter': False, 'connections': False, 'overlap': 'unfilled translucent markers, larger markers drawn first'},
            'validation': {'all_present_receipts_authenticated': True,
                           'frozen_validator_used_in_isolated_interpreter': True,
                           'collection_interpreter_binary_version_and_stdlib_authenticated': True,
                           'all_metrics_reconstructed_from_canonical_sets': True,
                           'complete_100_cell_grid': not missing,
                           'same_dataset_across_model_scales': True,
                           'immutable_registry_extensions_preserve_existing_dataset_identities': True,
                           'only_registered_upstream_base_models': True}}


def build_figure(record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    require(record.get('schema') == RECORD_SCHEMA, 'unsupported figure record')
    points = record['points']
    require(len({key(item) for item in points}) == len(points)
            and {key(item) for item in points} <= EXPECTED, 'invalid plotted cell identities')
    complete = record.get('status') == 'complete'
    require(not complete or {key(item) for item in points} == EXPECTED,
            'complete figure requires every model at every level in every domain')
    rc = {'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7,
          'xtick.labelsize': 6, 'ytick.labelsize': 6, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 5, figsize=FIGSIZE, sharex=True, sharey=True)
        fig.subplots_adjust(left=.062, right=.988, top=.80, bottom=.37, wspace=.22)
        for axis, domain in zip(axes, DOMAINS):
            for model in reversed(MODELS):
                for level in LEVELS:
                    group = [point for point in points if point['domain'] == domain
                             and point['model_label'] == model and point['level'] == level]
                    for point in group:
                        axis.scatter(point['metrics']['pass8'], point['metrics']['distinct8'],
                                     marker=MARKERS[model], s=AREAS[model], facecolors='none',
                                     edgecolors=COLORS[level], alpha=.78, linewidths=.75,
                                     zorder=3 + list(reversed(MODELS)).index(model), clip_on=False)
            axis.set_title(TITLES[domain], fontsize=7.2, pad=5)
            axis.set_xlim(0, 1); axis.set_ylim(0, 8)
            axis.set_xticks([0, .5, 1], labels=['0', '0.5', '1'])
            axis.set_yticks([0, 2, 4, 6, 8]); axis.set_xlabel('pass@8', labelpad=3)
            axis.grid(color='#E1E6EA', linewidth=.45, zorder=0)
            for side in ('top', 'right'):
                axis.spines[side].set_visible(False)
            for side in ('bottom', 'left'):
                axis.spines[side].set_color('#ACB8C1'); axis.spines[side].set_linewidth(.6)
            axis.tick_params(length=2, width=.6)
            if not complete:
                count = record['coverage_by_domain'][domain]['complete']
                axis.text(.5, .98, f'{count}/20 complete', ha='center', va='top',
                          transform=axis.transAxes, fontsize=6.4, color='#923F35')
                if count == 0:
                    axis.text(.5, .5, 'Awaiting\nmeasurements', ha='center', va='center',
                              transform=axis.transAxes, fontsize=7, color='#77848D')
        axes[0].set_ylabel('distinct@8', labelpad=4)
        levels = [Line2D([], [], marker='o', linestyle='none', markersize=4.5,
                         markerfacecolor='none', markeredgecolor=COLORS[level], label=f'Level {i}')
                  for i, level in enumerate(LEVELS, 1)]
        models = [Line2D([], [], marker=MARKERS[model], linestyle='none', markersize=4.5,
                         markerfacecolor='none', markeredgecolor='#334455', label=MODEL_NAMES[model])
                  for model in MODELS]
        fig.legend(handles=levels, title='Benchmark level', loc='lower center',
                   bbox_to_anchor=(.5, .115), ncol=5, frameon=False, fontsize=6.6,
                   title_fontsize=6.8, columnspacing=1.3, handletextpad=.35, handlelength=1.0)
        fig.legend(handles=models, title='Frozen Qwen2.5-Instruct', loc='lower center',
                   bbox_to_anchor=(.5, .002), ncol=4, frameon=False, fontsize=6.6,
                   title_fontsize=6.8, columnspacing=1.8, handletextpad=.35, handlelength=1.0)
        if complete:
            fig.text(.062, .96, 'Frozen base-model capability across benchmark levels',
                     ha='left', va='top', fontsize=8, fontweight='bold')
        else:
            fig.text(.5, .975, f'PARTIAL PREVIEW — {len(points)}/100 complete cells; {100-len(points)} missing',
                     ha='center', va='top', fontsize=8, fontweight='bold', color='#923F35')
        fig.text(.988, .915, '128 prompts/cell · four independent groups of 8',
                 ha='right', va='top', fontsize=6.4, color='#607080')
    return fig


def render(source=SOURCE, output=OUTPUT, *, allow_partial=False):
    """Reauthenticate source evidence immediately before writing any figure."""
    output = Path(output).resolve()
    if allow_partial:
        require(output.is_relative_to((ROOT / 'artifacts').resolve()),
                'partial previews may only be written below artifacts/, never paper/')
    record = build_record(source, allow_partial=allow_partial)
    require(record['status'] == 'complete' or allow_partial, 'publication requires complete evidence')
    figure = build_figure(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ('.pdf', '.png'):
        path = output.with_suffix(suffix)
        kwargs = {'dpi': 250} if suffix == '.png' else {'metadata': {'CreationDate': None, 'ModDate': None}}
        figure.savefig(path, facecolor='white', **kwargs)
        outputs[suffix[1:]] = binding(path)
    import matplotlib.pyplot as plt
    plt.close(figure)
    record['outputs'] = outputs
    output.with_suffix('.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def caption_snippet():
    return r'''% STAGED: include only after all 100 cells pass source and receipt authentication.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/modebench_base_level_grid.pdf}
\caption{\textbf{Frozen base-model capability across benchmark levels.}
Each domain panel shows empirical \texttt{pass@8} against \texttt{distinct@8}
for every combination of Levels 1--5 (color) and frozen Qwen2.5-Instruct
0.5B, 3B, 7B, and 14B checkpoints (marker). Each point averages 128 held-out
prompts, with four independent groups of eight responses per prompt under
the common evaluation interface ($T=1$, $\mathrm{top}\mbox{-}p=1$, 192 output tokens).
Within each eight-response group, \texttt{pass@8} records whether any answer is
verified correct and \texttt{distinct@8} counts distinct verified canonical
answers; both metrics are averaged over groups and prompts, including groups
with no verified answer. Open, translucent markers retain exact coordinates
when points overlap.}
\label{fig:modebench-base-level-grid}
\end{figure*}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preview', action='store_true', help='Allow missing cells; output must remain below artifacts/')
    parser.add_argument('--registry', type=Path, help='Create the bound source index from this registry and explicit --receipt files')
    parser.add_argument('--collection-plan', type=Path, default=COLLECTION_PLAN,
                        help='Sealed plan binding the exact Python environment used to collect receipts')
    parser.add_argument('--registry-extension', type=Path, action='append', default=[],
                        help='Additional immutable registry admitting later levels; existing entries must be unchanged')
    parser.add_argument('--validator', type=Path, default=VALIDATOR, help='Frozen evaluator used for isolated receipt authentication')
    parser.add_argument('--receipt', type=Path, action='append', default=[])
    parser.add_argument('--stage-caption', type=Path, help='Write a future complete-grid LaTeX snippet below artifacts/')
    args = parser.parse_args()
    if args.registry:
        require(args.source.resolve().is_relative_to((ROOT / 'artifacts').resolve()),
                'source indexes must be staged below artifacts/')
        require(not args.source.exists(), 'refusing to replace an existing source index; use a new path')
        args.source.parent.mkdir(parents=True, exist_ok=True)
        args.source.write_text(json.dumps(make_source(args.registry, args.receipt, args.validator, args.registry_extension, args.collection_plan), indent=2) + '\n')
    elif args.receipt or args.registry_extension:
        parser.error('--receipt and --registry-extension require --registry')
    if args.stage_caption:
        require(args.stage_caption.resolve().is_relative_to((ROOT / 'artifacts').resolve()),
                'unpublished caption snippets must be staged below artifacts/')
        args.stage_caption.parent.mkdir(parents=True, exist_ok=True)
        args.stage_caption.write_text(caption_snippet())
    output = args.output or (BASE / 'partial_preview' if args.preview else OUTPUT)
    record = render(args.source, output, allow_partial=args.preview)
    print(json.dumps({'status': record['status'], 'complete_cells': record['complete_cells'],
                      'required_cells': 100, 'output': portable(output)}))


if __name__ == '__main__':
    main()
