#!/usr/bin/env python3
"""Export complete, audited hosted runs into standalone comparison evidence.

This never modifies the frozen GPT-only snapshot or manuscript entrypoints.
Pending runs are listed in the output record and omitted from reported results.
Example:
  python ops/build_frontier_paper_comparison.py \
    --run artifacts/frontier_modebench_gpt56sol_20260911 \
    --run artifacts/frontier_models_comparison_20260911/gpt54
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

try:
    from ops.paper_domain_typography import format_domain_names
except ModuleNotFoundError:
    from paper_domain_typography import format_domain_names

_DOMAIN_LANGUAGE_EXCEPTIONS = (
    "Python lambda", r"Python \texttt{lambda}", "Python modulo",
)

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {'graph_coloring': 'Graph', 'countdown': 'Countdown',
           'python_factors': 'Python', 'mathir': 'MathIR', 'pantry_plan': 'PantryPlan'}
LABELS = {'gpt-5.6-sol': 'GPT-5.6 Sol', 'gpt-5.4': 'GPT-5.4',
          'claude-opus-5': 'Claude Opus 5', 'claude-opus-4-8': 'Claude Opus 4.8',
          'grok-4.3': 'Grok 4.3', 'DeepSeek-V4-Pro': 'DeepSeek V4 Pro',
          'FW-Kimi-K3': 'Kimi K3'}
ICONS = {'gpt-5.6-sol': 'openai.png', 'gpt-5.4': 'openai.png',
         'claude-opus-5': 'claude.png', 'claude-opus-4-8': 'claude.png',
         'DeepSeek-V4-Pro': 'deepseek.png', 'FW-Kimi-K3': 'kimi.png',
         'grok-4.3': 'grok_official_docs_print.png'}
STEM = 'frontier_comparison_20260911'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relpath(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def tex(value):
    return ''.join({'\\': r'\textbackslash{}', '_': r'\_', '%': r'\%', '&': r'\&',
                    '#': r'\#', '{': r'\{', '}': r'\}', '$': r'\$', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}.get(c, c) for c in str(value))


def load_run(directory, provider_outcome_paths=None):
    directory = Path(directory).resolve()
    summary_path = directory / 'summary.json'
    audit_path = directory / 'completion_audit.json'
    if not summary_path.is_file() or not audit_path.is_file():
        return None, 'awaiting summary.json and completion_audit.json'
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if summary.get('status') != 'complete' or audit.get('status') != 'pass':
        return None, 'summary incomplete or completion audit not passed'
    for key, expected in [('expected_responses', 15360), ('received_responses', 15360),
                          ('complete_prompts', 1920), ('missing_responses', 0)]:
        if summary.get(key) != expected:
            raise ValueError(f'{directory}: inconsistent completed {key}')
    for key in ('expected_responses', 'saved_samples'):
        if audit.get(key) != 15360:
            raise ValueError(f'{directory}: audit {key} must be 15360')
    if audit.get('unique_response_choice_ids', audit.get('unique_response_ids')) != 15360:
        raise ValueError(f'{directory}: missing unique identities for 15360 sampled choices')
    config = summary['run_configuration']
    if config.get('sample_count') != 8 or config.get('training') is not False:
        raise ValueError(f'{directory}: requires eight draws and no training')
    if config.get('conversation_state') is not False or config.get('tools') != []:
        raise ValueError(f'{directory}: requires stateless requests without tools')
    expected_cells = {f'level{level}/{domain}' for level in (1, 2, 3) for domain in DOMAINS}
    if set(summary['cells']) != expected_cells:
        raise ValueError(f'{directory}: incomplete domain/level grid')
    for key, cell in summary['cells'].items():
        if cell.get('complete_prompts') != 128:
            raise ValueError(f'{directory}: incomplete prompt cohort in {key}')
        counts = cell['counts']
        for metric, count, denominator in [('pass1', 'correct_draws', 1024),
                                           ('distinct8', 'distinct_correct_modes', 128)]:
            if abs(cell['metrics'][metric]['estimate'] - counts[count] / denominator) > 1e-12:
                raise ValueError(f'{directory}: inconsistent {metric} counts in {key}')
        pair_count = counts['correct_pairs']
        actual = cell['metrics']['correct_pair_collision']['estimate']
        expected = counts['colliding_correct_pairs'] / pair_count if pair_count else None
        if actual != expected:
            raise ValueError(f'{directory}: inconsistent collision counts in {key}')
    input_hashes = summary.get('input_sha256', {})
    prompt_sha = input_hashes.get('rows.jsonl')
    for name in ('rows.jsonl', 'samples.jsonl'):
        source = directory / name
        if not input_hashes.get(name) or not source.is_file() or digest(source) != input_hashes[name]:
            raise ValueError(f'{directory}: actual {name} does not match its summary digest')
    primary_path = Path(summary.get('primary_samples_path', ''))
    if not primary_path.is_absolute():
        primary_path = directory / primary_path
    if (not primary_path.resolve().is_relative_to(directory)
            or not primary_path.is_file()
            or not summary.get('primary_samples_sha256')
            or digest(primary_path) != summary['primary_samples_sha256']):
        raise ValueError(f'{directory}: primary graded samples are absent, external, or stale')
    model = config['model']
    for audit_key in ('model', 'requested_model', 'deployment_name'):
        if audit.get(audit_key) is not None and audit[audit_key] != model:
            raise ValueError(f'{directory}: audit {audit_key} differs from requested model')
    served = audit.get('served_model_snapshots_by_logical_sample',
                       audit.get('served_model_snapshots', {}))
    returned = summary.get('models_returned', {})
    admitted_ids = {model} | set(served)
    if (not returned or sum(returned.values()) != 15360
            or not set(returned).issubset(admitted_ids)):
        raise ValueError(f'{directory}: returned model identities/counts lack requested or audited provenance')
    outcome_path = Path((provider_outcome_paths or {}).get(model, directory / 'provider_outcomes.json'))
    outcomes = None
    if outcome_path.is_file():
        native = json.loads(outcome_path.read_text())
        if (native.get('status') != 'complete' or native.get('model') != model
                or native.get('expected_responses') != 15360 or native.get('responses') != 15360
                or Path(native.get('source_run', '')).resolve() != directory):
            raise ValueError(f'{directory}: native provider-outcome audit is incomplete or mismatched')
        for name in ('samples.jsonl', 'completion_audit.json'):
            binding = native.get('sources', {}).get(name, {})
            actual = directory / name
            if binding.get('sha256') != digest(actual) or Path(binding.get('path', '')).resolve() != actual:
                raise ValueError(f'{directory}: provider outcomes bind stale {name}')
        classified = native.get('sample_outcomes', {})
        classified_path = Path(classified.get('path', ''))
        if (classified.get('records') != 15360 or not classified_path.is_file()
                or digest(classified_path) != classified.get('sha256')):
            raise ValueError(f'{directory}: per-sample provider classifications are absent or stale')
        if set(native.get('cells', {})) != expected_cells:
            raise ValueError(f'{directory}: provider-outcome grid is incomplete')
        for key, cell in native['cells'].items():
            if cell.get('responses') != 1024:
                raise ValueError(f'{directory}: provider-outcome denominator differs in {key}')
            for counter in ('refusals', 'content_filtered', 'empty_answer_text'):
                if not 0 <= cell.get(counter, -1) <= 1024:
                    raise ValueError(f'{directory}: invalid provider counter {counter} in {key}')
        outcomes = {'path': relpath(outcome_path), 'sha256': digest(outcome_path),
                    'sample_outcomes': classified, 'sources': native['sources'],
                    'cells': native['cells'],
                    'interpretation': 'Provider-declared metadata only; ordinary answer text is not a semantic refusal detector.'}
    # Check the declared normalizer before including secondary scores.
    secondary = summary.get('normalized_secondary')
    if secondary is not None and set(secondary.get('cells', {})) != expected_cells:
        raise ValueError(f'{directory}: partial secondary grid cannot enter comparison')
    model = config['model']
    collection = None
    registration_path = directory / 'interrupted_attempt_registration.json'
    if registration_path.is_file():
        adapter_path = directory / 'completion_audit_adapter.json'
        protocol_path = directory / 'provider_protocol_adapter.json'
        registration = json.loads(registration_path.read_text())
        adapter = json.loads(adapter_path.read_text())
        protocol_adapter = json.loads(protocol_path.read_text())
        accounting = audit.get('interruption_accounting', {})
        unknown = registration['orphan_attempt_count']
        if (adapter.get('registration_sha256') != digest(registration_path)
                or accounting.get('registered_interrupted_attempts_with_unknown_outcome') != unknown
                or accounting.get('registered_interrupted_attempts_with_unknown_usage') != unknown):
            raise ValueError(f'{directory}: interrupted physical attempts lack audited accounting')
        for receipt in protocol_adapter['initial_receipts'].values():
            if digest(directory / receipt['path']) != receipt['sha256']:
                raise ValueError(f'{directory}: initial nonterminal protocol receipt changed')
        final_protocol_path = directory / 'provider_protocol_attempt_audit.json'
        final_protocol = json.loads(final_protocol_path.read_text())
        if (final_protocol.get('status') != 'pass'
                or final_protocol.get('completion_audit_sha256') != digest(audit_path)
                or final_protocol.get('provider_protocol_adapter_sha256') != digest(protocol_path)
                or final_protocol.get('retained_terminal_samples') != 15360):
            raise ValueError(f'{directory}: final nonterminal protocol-attempt accounting is stale')
        collection = {'interruption_accounting': accounting,
                      'initial_nonterminal_protocol_receipts': len(protocol_adapter['initial_receipts']),
                      'nonterminal_protocol_receipts': final_protocol['nonterminal_http200_protocol_failures'],
                      'sidecars': {p.name: {'path': relpath(p), 'sha256': digest(p)}
                                   for p in (registration_path, adapter_path, protocol_path, final_protocol_path)}}
    return {'model': model, 'label': LABELS.get(model, model),
            'run_directory': relpath(directory), 'summary_sha256': digest(summary_path),
            'completion_audit_sha256': digest(audit_path),
            'served_snapshots': served, 'models_returned': returned,
            'primary_samples_path': relpath(primary_path),
            'primary_samples_sha256': summary['primary_samples_sha256'],
            'samples_sha256': input_hashes['samples.jsonl'],
            'provider_outcomes': outcomes, 'collection_attempt_accounting': collection,
            'prompt_sha256': prompt_sha, 'run_configuration': config,
            'returned_sampling': summary.get('returned_sampling', {}),
            'bootstrap': summary['bootstrap'], 'levels': summary['levels'],
            'cells': summary['cells'],
            'normalized_secondary': ({k: secondary[k] for k in
                                     ('levels', 'cells', 'normalization_source_sha256',
                                      'frozen_grader_contract_sha256') if k in secondary}
                                     if secondary is not None else None)}, None


def build_record(directories, protocol, provider_outcome_paths=None):
    included, excluded = [], []
    for directory in directories:
        run, reason = load_run(directory, provider_outcome_paths)
        if run is None:
            excluded.append({'directory': relpath(directory), 'reason': reason})
        else:
            included.append(run)
    if not included:
        raise ValueError('No complete, audited runs are available; no output was written.')
    if len({run['model'] for run in included}) != len(included):
        raise ValueError('Duplicate deployment: choose one frozen run per model.')
    if len({run['prompt_sha256'] for run in included}) != 1:
        raise ValueError('Prompt digests differ: a same-test comparison is not justified.')
    normalized = [run['normalized_secondary'] for run in included
                  if run['normalized_secondary'] is not None]
    for key in ('normalization_source_sha256', 'frozen_grader_contract_sha256'):
        if normalized and (any(not n.get(key) for n in normalized)
                           or len({n[key] for n in normalized}) != 1):
            raise ValueError(f'Secondary comparison requires identical {key}.')
    record = {'schema': 'frontier-paper-comparison-v1',
            'builder_sha256': digest(__file__),
            'protocol': {'path': relpath(protocol), 'sha256': digest(protocol)},
            'models': included, 'excluded_runs': excluded,
            'completed_model_count': len(included),
            'completed_response_count': 15360 * len(included),
            'prompt_count_per_cell': 128, 'draws_per_prompt': 8,
            'inference_only': True,
            'graph_selection': 'Selected after GPT-5.6 Sol for high strict correctness, '
                               'certified support, and no normalization effect; selected '
                               'before benchmark responses from the prospective comparison models.',
            'comparison_limits': ['Provider medium labels do not equate compute.',
                                  'No causal training, model-size, or difficulty contrast.',
                                  'Pointwise cell intervals are not paired model-difference intervals.',
                                  'Uniform references condition on each model\'s correct draws.']}

    record['python_prompt_sensitivity'] = load_python_sensitivity(
        ROOT / 'artifacts/frontier_modebench_claude_opus5_python_plain_20260911/paired_prompt_comparison.json')
    record['main_display'] = {'metrics': ['pass1', 'distinct8'],
                              'domains': list(DOMAINS), 'levels': [1, 2, 3],
                              'grading': 'frozen formatting-normalized',
                              'python_condition': 'Complete revised Opus 5 Python cohort; original cohorts elsewhere',
                              'population': 'All eight responses for every held-out prompt',
                              'figure': 'paper/figures/hosted_verified_breadth.pdf'}
    record['model_icons'] = {model: {'path': 'paper/icons/' + name,
                                    'sha256': digest(ROOT / 'paper/icons' / name)}
                             for model, name in ICONS.items() if model in {r['model'] for r in record['models']}}
    for icon in record['model_icons'].values():
        provenance = (ROOT / icon['path']).with_suffix('.json')
        if provenance.is_file():
            icon['provenance'] = {'path': relpath(provenance), 'sha256': digest(provenance)}
    return record


def metric(cell, key, percent=False, interval=False):
    item = cell['metrics'][key]
    value = item['estimate']
    if value is None:
        return '--'
    factor, digits = (100, 1) if percent else (1, 2)
    result = f'{value * factor:.{digits}f}'
    if interval and item.get('ci95') is not None:
        lo, hi = item['ci95']
        result += f' [{lo * factor:.{digits}f}, {hi * factor:.{digits}f}]'
    return result


def overview_table(record, secondary=False):
    rows = []
    for run in record['models']:
        source = run['normalized_secondary'] if secondary else run
        if source is None:
            continue
        if rows:
            rows.append(r'\midrule')
        for level in ('1', '2', '3'):
            cell = source['levels'][level]
            columns = [tex(run['label']), level, metric(cell, 'pass1', True),
                       metric(cell, 'pass8', True), metric(cell, 'distinct8'),
                       metric(cell, 'correct_pair_collision', True, True)]
            rows.append(' & '.join(columns) + r' \\')
    return '\n'.join(rows) + '\n\\bottomrule\n'


def _cell_columns(cell):
    return [metric(cell, 'pass1', True), metric(cell, 'pass8', True),
            metric(cell, 'distinct8'),
            metric(cell, 'correct_pair_collision', True, True),
            metric(cell, 'uniform_correct_pair_collision', True)]


LAYOUT_LEGEND = (r'Each original-instruction cell contains 128 prompts with eight responses each. '
                 r'M/P are \texttt{mean@8}/\texttt{pass@8} (\%); '
                 r'D averages distinct correct solution modes per prompt, including zeros. '
                 r'C is the fraction of correct pairs sharing a mode (\%), pooling pairs '
                 r'across prompts; it differs from one minus prompt-averaged \pmd{}. '
                 r'Intervals are pointwise 95\% percentiles from 2,000 whole-prompt '
                 r'bootstrap resamples within each cell. U is the uniform-correct-mode '
                 r'reference (\%), using certified support counts and the same pair weights. '
                 r'A dash in C denotes no eligible pairs; U is also undefined when total '
                 r'support is unknown (Countdown).')


def cell_tables(record, secondary=False):
    """One table for the cohort, not one per deployment.

    Every deployment's cells carry the same five columns over the same fifteen
    domain--level rows, so seven separate tables repeated the header, the
    caption legend and the row labels seven times for a difference that is only
    ever the model. They are one table with a deployment column.

    The formatting sensitivity is reported as the rows it changes. It rescues
    responses for most deployments but alters a minority of cells, and printing
    an unchanged copy of every other cell beside them is the same repetition in
    a second register; it is also the rule this paper already applies to the
    prompt-hint and discovery secondary gradings. A deployment the normalizer
    leaves entirely alone is named rather than tabulated.
    """
    rows = []
    for run in record['models']:
        source = run['normalized_secondary'] if secondary else run
        if source is None:
            continue
        for level in (1, 2, 3):
            for domain, label in DOMAINS.items():
                key = f'level{level}/{domain}'
                columns = _cell_columns(source['cells'][key])
                if secondary and columns == _cell_columns(run['cells'][key]):
                    continue
                rows.append((tex(run['label']), str(level), label, columns))
    if secondary:
        unchanged = [tex(run['label']) for run in record['models']
                     if run.get('normalized_secondary') is not None
                     and not any(r[0] == tex(run['label']) for r in rows)]
        if not rows:
            return ('Formatting-normalized and strict grading have identical displayed '
                    'entries in every cell of this cohort.\n')
        caption = (r'\caption{\textbf{Formatting normalization increases accuracy without consistently reducing collision.} '
                   + LAYOUT_LEGEND
                   + r' Typography corrections precede the same executable verifier; mathematical '
                   + r'values and program logic are not repaired. Only cells with at least one '
                   + r'changed displayed entry appear; omitted cells match at the printed precision.'
                   + (' Every displayed entry is unchanged for ' + ', '.join(unchanged) + '.'
                      if unchanged else '')
                   + r'}')
    else:
        caption = (r'\caption{\textbf{Verified hosted outputs concentrate on a few observed solution modes.} '
                   + LAYOUT_LEGEND + r' Grades use the original executable answers.}')
    # A longtable, not a table. One deployment per float fitted a page; the
    # whole cohort in one float does not, and an over-tall float does not
    # break -- it runs off the page and silently takes its last rows with it.
    # The longtable keeps the single caption and single header this merge is
    # for, and breaks across pages instead of overflowing.
    header = (r'    Deployment & Level & Domain & M & P & D & '
              r'C (95\% interval) & U \\')
    text = [r'\begingroup', r'\scriptsize', r'\setlength{\tabcolsep}{4pt}',
            r'\begin{longtable}{@{}lrlrrrrr@{}}',
            '  ' + caption + r' \\',
            r'  \toprule', header, r'  \midrule', r'\endfirsthead',
            r'  \multicolumn{8}{@{}l}{\scriptsize\itshape '
            + ('Formatting-normalized grades, continued.' if secondary
               else 'Strict executable grades, continued.') + r'}\\',
            r'  \toprule', header, r'  \midrule', r'\endhead',
            r'  \bottomrule', r'\endlastfoot']
    previous = None
    for label, level, domain, columns in rows:
        if previous is not None and label != previous:
            text.append(r'    \addlinespace[2pt]')
        text.append('    ' + ' & '.join([label if label != previous else '',
                                         level, domain, *columns]) + r' \\')
        previous = label
    text += [r'\end{longtable}', r'\endgroup', '']
    return format_domain_names('\n'.join(text), exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS)


def model_identity(run):
    name = tex(run['label'])
    if run['model'] in ICONS:
        name = (r'\raisebox{-.15ex}{\includegraphics[height=1.7ex]{icons/'
                + ICONS[run['model']] + r'}}\,' + name)
    return name


def graph_span(record, key, factor=1, digits=2):
    values = [run['cells']['level3/graph_coloring']['metrics'][key]['estimate']
              for run in record['models']]
    values = [value for value in values if value is not None]
    if not values:
        return '--'
    low, high = [f'{v * factor:.{digits}f}' for v in (min(values), max(values))]
    return low if low == high else low + '--' + high


def load_python_sensitivity(path):
    report = json.loads(path.read_text())
    if (report.get('status') != 'complete' or report.get('model') != 'claude-opus-5'
            or not all(report.get('validation', {}).values())):
        raise ValueError('Python prompt sensitivity has not passed its independent audit.')
    for condition in report['conditions'].values():
        if condition['selected_prompts'] != 384 or condition['selected_responses'] != 3072:
            raise ValueError('Python prompt sensitivity has an unexpected task population.')
        for binding in condition['source_sha256'].values():
            if digest(binding['path']) != binding['sha256']:
                raise ValueError('Python prompt sensitivity binds stale source evidence.')
    return {'path': relpath(path), 'sha256': digest(path),
            **{k: report[k] for k in ('model', 'conditions', 'levels', 'totals', 'bootstrap',
                                     'limitations', 'validation', 'analysis_source')}}


def python_sensitivity_tex(report):
    lines = [r'\subsection{Python prompt sensitivity}',
             r'\label{app:hosted-python-prompt}',
             'Opus 5 receives the same 384 Python tasks under the original instructions',
             'and a direct arithmetic-expression instruction without a system message.',
             'Each condition contains eight responses from separate requests per task across three',
             'levels. Adaptive thinking, medium effort, an 8,192-token output limit,',
             'allowed operators, public inputs, and graders are the same in both conditions.',
             'The revised instruction asks for a one-line boxed Python lambda returning',
             'any proper divisor for each listed input, without a preferred divisor.',
             'Task wording and system-message presence both change; provider state and',
             'collection time can also differ. The comparison describes these two',
             'configurations and does not isolate which change accounts for their difference.',
             'All responses contribute to accuracy and mode counts, including provider refusals.', '',
             r'\begin{table}[!htbp]', r'  \centering',
             r'  \caption{\textbf{Opus 5 rarely refuses Python tasks with direct instructions and no system message.}',
             '  Each row contains 128 tasks with eight responses each (1,024 responses).',
             '  R counts provider-declared refusals; A is per-response accuracy (\\%);',
             r'  D is mean \texttt{distinct@8}; C is correct-pair collision (\%), pooling pairs within each level.',
             '  Both conditions use the same executable grader and formatting normalizer.',
             '  Original Level-2/3 collision is undefined because no task has two correct responses.}',
             r'  \small', r'  \setlength{\tabcolsep}{4pt}',
             r'  \begin{tabular}{@{}llrrrrrrr@{}}', r'    \toprule',
             r'    & & & \multicolumn{3}{c}{Strict} & \multicolumn{3}{c}{Normalized} \\',
             r'    Condition & L & R & A & D & C & A & D & C \\', r'    \midrule']
    for condition, label in [('original', 'Original'), ('plain', 'Plain user, no system')]:
        if condition == 'plain':
            lines.append(r'    \midrule')
        for level in ('1', '2', '3'):
            cell = report['levels'][level][condition]
            values = [label if level == '1' else '', level, str(cell['counts']['native_refusals'])]
            for grade in ('strict', 'normalized'):
                for name, factor, digits in [('accuracy', 100, 1), ('distinct8', 1, 2),
                                             ('correct_pair_collision', 100, 1)]:
                    value = cell['metrics'][grade + '_' + name]
                    values.append('--' if value is None else f'{value * factor:.{digits}f}')
            lines.append('    ' + ' & '.join(values) + r' \\')
    plain = report['totals']['plain']['counts']
    collisions = '/'.join(f'{report["levels"][level]["plain"]["metrics"]["normalized_correct_pair_collision"] * 100:.1f}'
                          for level in ('1', '2', '3'))
    gains = []
    for level in ('1', '2', '3'):
        item = report['levels'][level]['plain_minus_original']['normalized_accuracy']
        gains.append(f'{item["estimate"] * 100:.1f} [{item["ci95"][0] * 100:.1f}, {item["ci95"][1] * 100:.1f}]')
    rescued = plain['normalized']['correct_responses'] - plain['strict']['correct_responses']
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '',
              'The revised condition has no API failures or token-limit truncations. Strict grading',
              f'accepts {plain["strict"]["correct_responses"]:,} responses; formatting normalization accepts {rescued:,} additional responses',
              r'by converting escaped percent signs (\verb|\%|) to Python modulo (\verb|%|).',
              'Every non-refused answer therefore passes the executable checks after normalization.',
              'Normalized collision is ' + collisions + r'\%, versus conditional uniform references',
              r'of 1.43/1.15/1.15\%. The three refusals occur on two tasks whose other',
              'identical requests produce accepted answers. These results use all eight responses',
              'per task, without selecting or replacing outputs according to their validity.', '',
              'Accuracy differences use a paired bootstrap of whole eight-response task groups',
              'across conditions (20,000 replicates). Normalized accuracy gains at Levels 1/2/3 are',
              ', '.join(gains[:-1]) + ', and ' + gains[-1] + ' percentage points',
              '(pointwise 95\\% intervals). Tasks are paired; output draws are not paired across conditions.',
              'Collision weights tasks by their numbers of correct pairs, which differ between',
              'conditions. Its raw difference therefore compares different correct-pair populations',
              'and does not estimate a prompt effect at matched accuracy or on a common set of pairs.', '']
    return format_domain_names('\n'.join(lines), exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS)


def graph_figure(record, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#8F6700', '#303030']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85), constrained_layout=False)
    fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.76, wspace=.25)
    refs = []
    graph_data = []
    for index, run in enumerate(record['models']):
        cells = [run['cells'][f'level{level}/graph_coloring'] for level in (1, 2, 3)]
        positions = np.arange(1, 4) + (index - (len(record['models']) - 1) / 2) * .022
        refs.append([100 * value if (value := cell['metrics']['uniform_correct_pair_collision']['estimate']) is not None else np.nan
                     for cell in cells])
        graph_data.append({'model': run['model'], 'cells': cells})
        for ax, key in zip(axes, ['pass1', 'correct_pair_collision']):
            values = np.array([100 * value if (value := cell['metrics'][key]['estimate']) is not None else np.nan for cell in cells])
            intervals = np.array([cell['metrics'][key]['ci95'] or [np.nan, np.nan] for cell in cells], dtype=float) * 100
            error = np.maximum(0, np.vstack((values - intervals[:, 0], intervals[:, 1] - values)))
            ax.errorbar(positions, values, yerr=error, label=run['label'], capsize=2,
                        color=colors[index % len(colors)], marker=markers[index % len(markers)],
                        markersize=4, linewidth=1.2)
    references = np.asarray(refs)
    defined = np.isfinite(references).any(axis=0)
    low = np.array([np.nanmin(references[:, i]) if defined[i] else np.nan for i in range(3)])
    high = np.array([np.nanmax(references[:, i]) if defined[i] else np.nan for i in range(3)])
    middle = np.array([np.nanmean(references[:, i]) if defined[i] else np.nan for i in range(3)])
    axes[1].fill_between([1, 2, 3], low, high, color='.65', alpha=.35)
    axes[1].plot([1, 2, 3], middle, color='.45', linestyle='--', linewidth=1,
                 label='Uniform reference range')
    for ax, title in zip(axes, ['A  Verified correctness', 'B  Same-mode correct pairs']):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xticks([1, 2, 3], ['Level 1', 'Level 2', 'Level 3'])
        ax.set_xlim(.83, 3.17)
        ax.set_ylim(0, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel('%', rotation=0, labelpad=9)
        ax.tick_params(labelsize=9)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, 1.015),
               ncol=min(4, len(handles)), frameon=False, fontsize=8)
    for suffix in ('.pdf', '.png'):
        fig.savefig(path.with_suffix(suffix), dpi=220, bbox_inches='tight')
    plt.close(fig)
    figure_record = {'schema': 'frontier-graph-comparison-v1', 'selection': record['graph_selection'],
                     'pdf_sha256': digest(path.with_suffix('.pdf')),
                     'png_sha256': digest(path.with_suffix('.png')),
                     'models': graph_data, 'metric_order': ['pass1', 'correct_pair_collision'],
                     'uncertainty': 'Saved pointwise whole-prompt bootstrap intervals.',
                     'uniform_band': 'Range across admitted models of their correct-pair-weighted references.'}
    path.with_suffix('.json').write_text(json.dumps(figure_record, indent=2) + '\n')


def export(record, output, figures, stem):
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    output.joinpath(stem + '.json').write_text(json.dumps(record, indent=2) + '\n')
    output.joinpath(stem + '_python_sensitivity.tex').write_text(format_domain_names(python_sensitivity_tex(record['python_prompt_sensitivity']), exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS))
    for secondary, suffix in [(False, 'strict'), (True, 'normalized')]:
        output.joinpath(f'{stem}_{suffix}_overview_rows.tex').write_text(format_domain_names(overview_table(record, secondary), exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS))
        output.joinpath(f'{stem}_{suffix}_cells.tex').write_text(format_domain_names(cell_tables(record, secondary), exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS))
    graph_figure(record, figures / (stem + '_graph'))
    figure_include = r'''\begin{figure}[!htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/STEM_graph.pdf}
  \caption{\textbf{Hosted Graph answers are usually correct but concentrated relative to uniform sampling.}
  Panel A shows strict per-response accuracy; panel B shows the fraction of
  correct pairs sharing a solution mode, pooling pairs across prompts. Colors
  and markers identify seven deployments, each with the same 128 prompts per
  level and eight stateless responses per prompt. Error bars are pointwise
  95\% intervals from resampling whole prompts. The gray band spans the
  deployments' uniform-correct-mode references, weighted by their observed
  correct-pair counts; the dashed line is their mean. Level labels do not
  imply equal or monotonically increasing difficulty across deployments.}
  \label{fig:hosted-graph-comparison}
\end{figure}
'''.replace('STEM', stem)
    output.joinpath(stem + '_graph_figure.tex').write_text(format_domain_names(figure_include, exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS))
    names = ', '.join(tex(run['label']) for run in record['models'])
    protocol = [r'\paragraph{Evaluation population.}',
                f'The deployments are {names}. The original-instruction cohort contains',
                f'{record["completed_response_count"]:,} responses: 15,360 per deployment, with the same 128 held-out',
                'prompts in each of five domains and three levels, and eight stateless responses',
                'per prompt. Requests use no model-side tools or conversation history, and',
                'the evaluation includes no task training or replay intervention.', '',
                r'\paragraph{Provider configurations.}']
    configurations = [run['run_configuration'] for run in record['models']]
    if all(str(config.get('reasoning_effort', '')).startswith('medium')
           and config.get('max_output_tokens') == 8192 for config in configurations):
        protocol += ['All deployments request medium reasoning effort and an 8,192-token native',
                     'output limit. Claude Opus 5 and Claude Opus 4.8 use Anthropic Messages with',
                     'adaptive thinking; the other deployments use hosted API routes.']
    else:
        for run in record['models']:
            config = run['run_configuration']
            provider = config.get('provider', config.get('api', config.get('api_type', 'hosted API')))
            protocol.append(tex(run['label']) + ': ' + tex(provider) + ', requested reasoning setting '
                            + tex(config.get('reasoning_effort', 'unspecified'))
                            + ', native output limit ' + tex(config.get('max_output_tokens', 'unspecified')) + '.')
    protocol += ['', 'The requested reasoning setting and output limit do not match compute across',
                 'providers: effort labels and token accounting have provider-specific meanings.',
                 'Temperature and nucleus sampling use provider defaults, whose exact values',
                 'are unknown when the provider does not return them.', '',
                 'Strict grading checks the original executable answer; formatting-normalized',
                 'grading applies the same typography corrections across deployments before',
                 'the executable checks. The accuracy row of the main display uses normalized',
                 'grades and the revised Opus 5 Python instruction at all three levels;',
                 'its other accuracy cells use the original instructions. The diversity row',
                 'uses strict grades and original instructions for every deployment.',
                 'The tables below report original-instruction results under both grading rules.',
                 'Prompt preferences and the Level-1 PantryPlan interface difference limit',
                 'comparisons across levels. Countdown has no certified finite-total-support',
                 'uniform reference. This inference-only comparison does not identify a training',
                 'cause, a model-size effect, or a difficulty effect. Collision intervals are',
                 'pointwise estimates for individual cells, not intervals for paired deployment differences.']
    for run in record['models']:
        collection = run.get('collection_attempt_accounting')
        if collection:
            unknown = collection['interruption_accounting']['registered_interrupted_attempts_with_unknown_outcome']
            malformed = collection['nonterminal_protocol_receipts']
            protocol += ['', r'\paragraph{Response and cost coverage: ' + tex(run['label']) + '.}',
                         'The 15,360 terminal responses include verification failures and truncations.',
                         f'Outside this cohort, {malformed} HTTP-200 responses have neither a',
                         'terminal finish reason nor a visible answer, and',
                         f'{unknown} request attempts have unknown outcomes and usage. These incomplete',
                         'attempts are not scored as model responses. Token totals cover known usage',
                         'only and can therefore understate the usage and cost of all attempted requests.']
    output.joinpath(stem + '_protocol.tex').write_text(format_domain_names('\n'.join(protocol) + '\n', exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS))
    outcome_text = [r'\subsection{Native provider outcomes and undefined concentration}',
                    r'\label{app:hosted-provider-outcomes}',
                    'Provider metadata distinguish refusal and filtering from executable correctness.',
                    'A refusal does not establish that the underlying model cannot solve the task.',
                    'Classification uses native stop reasons, refusal fields and filter metadata;',
                    'it does not detect refusals expressed only in ordinary answer text.',
                    'An empty visible answer is a failed draw, including when the response contains',
                    'reasoning output. Empty-answer and refusal counts therefore need not agree.', '',
                    'Correct-pair collision requires at least two correct draws on one prompt.',
                    'It is undefined otherwise, as is a five-domain mean containing any undefined',
                    'domain. Dashes denote these undefined values. Accuracy and distinct-mode',
                    'counts include all responses, with zero correct modes on unsuccessful prompts.', '']
    # Native completion states, one line per deployment rather than one
    # paragraph each: the sentence that followed every count was identical, so
    # it is stated once for the group and the counts become a list.
    unaudited = [tex(run['label']) for run in record['models']
                 if run.get('provider_outcomes') is None]
    states = []
    for run in record['models']:
        outcomes = run.get('provider_outcomes')
        if outcomes is None:
            continue
        stop_counts = {}
        for cell in outcomes['cells'].values():
            for reason, count in cell.get('stop_reason_counts', {}).items():
                stop_counts[reason] = stop_counts.get(reason, 0) + count
        states.append(r'\item ' + tex(run['label']) + ' --- '
                      + tex(', '.join(f'{reason}: {count:,}'
                                      for reason, count in sorted(stop_counts.items()))))
    if states:
        outcome_text += ['Native completion states over the 15,360 original-instruction responses '
                         'per deployment are:', '',
                         r'\begin{itemize}\setlength{\itemsep}{0pt}', *states,
                         r'\end{itemize}', '']
    if unaudited:
        outcome_text += ['Native refusal and filter metadata are unavailable for ' + ', '.join(unaudited)
                         + '; their refusal counts are unknown.', '']
    for run in record['models']:
        outcomes = run.get('provider_outcomes')
        if outcomes is None:
            continue
        if run['model'].startswith('claude-'):
            python_counts = [outcomes['cells'][f'level{level}/python_factors']['refusals'] for level in (1, 2, 3)]
            categories = {}
            for level in (1, 2, 3):
                for category, count in outcomes['cells'][f'level{level}/python_factors'].get('refusal_category_counts', {}).items():
                    categories[category] = categories.get(category, 0) + count
            if any(python_counts):
                outcome_text += [tex(run['label']) + ' returns provider-declared refusals for '
                                 + ', '.join(f'{count:,}' for count in python_counts)
                                 + ' of the 1,024 Python requests at Levels 1, 2 and 3, respectively.',
                                 'Across those three levels, the provider labels the refusals '
                                 + tex(', '.join(f'{key}: {value:,}' for key, value in categories.items()) or 'without categories')
                                 + '. These labels describe service behavior on factor-selection tasks;',
                                 'they do not measure concentration among correct Python outputs.', '']

    # One table for the cohort, listing only the cells that carry a counter. A
    # table per deployment printed 105 rows to report 22 non-zero ones, and
    # three deployments contributed nothing but zeros to it.
    rows, quiet = [], []
    for run in record['models']:
        outcomes = run.get('provider_outcomes')
        if outcomes is None:
            continue
        seen = False
        for level in (1, 2, 3):
            for domain, label in DOMAINS.items():
                cell = outcomes['cells'][f'level{level}/{domain}']
                counters = (cell['refusals'], cell['content_filtered'], cell['empty_answer_text'])
                if not any(counters):
                    continue
                rows.append((tex(run['label']), str(level), label, [str(c) for c in counters]))
                seen = True
        if not seen:
            quiet.append(tex(run['label']))
    total = sum(1 for run in record['models'] if run.get('provider_outcomes') is not None) * 15
    outcome_text += [r'\begin{table}[!htbp]', r'  \centering',
                     r'  \caption{\textbf{Provider-declared refusals occur only for Opus 5 in this cohort.} '
                     + r'Each original-instruction cell contains 128 prompts with eight responses '
                     + r'each (1,024 responses). R counts native refusal signals, F native content '
                     + r'filters, and E responses without visible answer text, including reasoning-only '
                     + r'outputs. Counters can overlap and are separate from executable grades. '
                     + f'Only nonzero cells appear; the other {total - len(rows)} of {total} cells have '
                     + r'$R=F=E=0$'
                     + (', including all cells for ' + ', '.join(quiet) if quiet else '')
                     + r'. Zero R does not exclude refusals expressed only in answer text.}',
                     r'  \label{tab:hosted-provider-outcomes}',
                     r'  \small', r'  \setlength{\tabcolsep}{5pt}',
                     r'  \begin{tabular}{@{}llrrrr@{}}', r'    \toprule',
                     r'    \textbf{Deployment} & \textbf{L} & \textbf{Domain} &',
                     r'      \textbf{R} & \textbf{F} & \textbf{E} \\', r'    \midrule']
    previous = None
    for label, level, domain, counters in rows:
        outcome_text.append('    ' + ' & '.join([label if label != previous else '',
                                                 level, domain, *counters]) + r' \\')
        previous = label
    outcome_text += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}', '']
    output.joinpath(stem + '_provider_outcomes.tex').write_text(format_domain_names('\n'.join(outcome_text) + '\n', exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS))
    appendix = r'''\section{Comparison across Hosted Deployments}
\label{app:hosted-comparison}

Figure~\ref{fig:hosted-verified-breadth} compares accuracy and success-conditional
solution-mode diversity across hosted deployments. Its accuracy row uses a
common formatting normalizer and the revised Opus 5 Python instructions at
each level; hollow marks show that deployment's original Python condition.
Other accuracy cells use the original instructions. The diversity row uses
strict grades and original instructions for every deployment, with the
prompt-level estimand and eligibility criteria of App.~\ref{app:hosted-mode-diversity}.
Accuracy includes every response, including refusals and verification failures.
Deployment settings differ, so the figure describes the stated configurations
and does not rank models under matched prompts and compute.

The tables and Graph figure use the original instructions for every deployment.
The Python comparison reports both instruction conditions under strict and
formatting-normalized grading. Level 3 extends the test tasks; performance
across levels need not follow a common difficulty ordering for all deployments.

\input{results/STEM_protocol.tex}
\input{results/STEM_graph_figure.tex}
\input{results/STEM_python_sensitivity.tex}
\input{results/frontier_python_retry_20260911.tex}

\input{results/frontier_temperature_20260911.tex}

\input{results/gpt56_temperature_curve_20260911_appendix.tex}
\input{results/STEM_provider_outcomes.tex}

\begin{table}[!htbp]
  \centering
  \caption{\textbf{Hosted deployments average fewer than three distinct correct modes per prompt.}
  Each row averages five domains equally under strict grading and original
  instructions, with 128 prompts per domain and eight responses per prompt.
  M/P are \texttt{mean@8}/\texttt{pass@8} (\%); D averages distinct correct
  solution modes, including zeros. C is correct-pair collision (\%), pooling
  pairs within each domain before averaging domains. Pointwise 95\% intervals
  use 2,000 whole-prompt bootstrap resamples within domains. A dash denotes
  an undefined domain collision and hence an undefined five-domain mean.
  Provider settings differ, so these are descriptive deployment comparisons.}
  \small
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}lrrrrr@{}}
    \toprule
    Deployment & Level & M & P & D & C (95\% interval) \\
    \midrule
    \input{results/STEM_strict_overview_rows.tex}
  \end{tabular}
\end{table}

\input{results/STEM_strict_cells.tex}
\input{results/STEM_normalized_cells.tex}

'''.replace('STEM', stem)
    output.joinpath(stem + '_appendix.tex').write_text(format_domain_names(appendix, exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS))
    included = '\n'.join(f'- {r["model"]}: 15,360 audited responses ({r["run_directory"]})'
                         for r in record['models'])
    excluded = '\n'.join(f'- {r["directory"]}: {r["reason"]}' for r in record['excluded_runs']) or '- None.'
    output.joinpath(stem + '_README.md').write_text(
        '# Generated hosted comparison\n\n' + included + '\n\nExcluded:\n\n' + excluded +
        '\n\nThe JSON binds source summaries, audits, prompt identity and the written protocol. '
        'The original-protocol figure and tables contain only complete admitted runs. '
        'The main diversity figure uses frozen normalization and the complete revised Opus 5 Python condition, with original evidence retained in the appendix.\n\n'
        'For paper integration, use the generated protocol and strict/normalized cell includes; '
        'the overview rows need a six-column tabular wrapper. The figure include references the '
        'PDF under figures/. Preserve the paper\'s existing detailed interface caveats and level provenance. '
        'Add the figure to the workshop figure inventory/checker if it is first activated, then '
        'synchronize the standalone workshop snapshot and rebuild both PDFs.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, default=ROOT / 'paper/results')
    parser.add_argument('--figure-directory', type=Path, default=ROOT / 'paper/figures')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'artifacts/frontier_models_comparison_20260911/PROTOCOL.md')
    parser.add_argument('--stem', default=STEM)
    parser.add_argument('--provider-outcomes', action='append', default=[], metavar='MODEL=PATH',
                        help='Use a separately stored native-outcome audit for a deployment.')
    parser.add_argument('--require-provider-outcomes', action='store_true',
                        help='Require native-outcome audits for every admitted deployment.')
    parser.add_argument('--minimum-models', type=int, default=1,
                        help='Fail without writing if fewer audited deployments are available.')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_]+', args.stem) or args.stem.startswith('frontier_hosted_'):
        parser.error('Use a safe comparison stem, distinct from the frozen frontier_hosted_ snapshot.')
    try:
        outcome_paths = {}
        for value in args.provider_outcomes:
            model, separator, path = value.partition('=')
            if not separator or not model or not path:
                raise ValueError('--provider-outcomes requires MODEL=PATH')
            outcome_paths[model] = Path(path)
        record = build_record(args.run, args.protocol, outcome_paths)
        if args.require_provider_outcomes and any(run['provider_outcomes'] is None for run in record['models']):
            raise ValueError('Native provider-outcome audits are required for every admitted model.')
        if len(record['models']) < args.minimum_models:
            raise ValueError(f'Need {args.minimum_models} complete audited models; '
                             f'only {len(record["models"])} available. No output was written.')
        export(record, args.output_directory, args.figure_directory, args.stem)
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps({'included_models': [r['model'] for r in record['models']],
                      'excluded_runs': record['excluded_runs'],
                      'responses': record['completed_response_count'],
                      'result': str(args.output_directory / (args.stem + '.json'))}, indent=2))


if __name__ == '__main__':
    main()
