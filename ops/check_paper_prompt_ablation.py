#!/usr/bin/env python3
"""Authenticate copied prompt-ablation paper artifacts and rebuild statistics.

Only complete included panels are publishable. A local-only or hosted-only
report must explicitly say partial_panels, and cannot contain figures for an
omitted panel. This checker is read-only and never invokes model generation.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
RESULT_STEM = 'modebench_prompt_ablation_20260911'
FIGURE_STEM = 'modebench_prompt_ablation'
PYTHON_DIAGNOSTIC = 'modebench_prompt_ablation_python_failure_diagnostic_20260911.json'
FAMILIES = ('frontier', 'local')
CSV_FIELDS = ('model_id', 'family', 'grading', 'cell', 'arm', 'metric', 'estimate',
              'ci95_low', 'ci95_high', 'defined_bootstrap_replicates')
CSV_KEYS = CSV_FIELDS[:6]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def nonfinite(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=nonfinite)
    json.dumps(value, allow_nan=False)  # Also rejects overflow such as 1e999.
    return value


def bound_file(record, expected_path=None):
    require(isinstance(record, dict) and isinstance(record.get('path'), str)
            and isinstance(record.get('sha256'), str) and len(record['sha256']) == 64,
            'Missing file identity binding')
    path = Path(record['path']).resolve()
    if expected_path is not None:
        require(path == Path(expected_path).resolve(), 'Bound path differs: ' + str(expected_path))
    require(path.is_file() and file_sha(path) == record['sha256'], 'Bound file changed: ' + str(path))
    return path


def exact_copy(source, copied):
    require(Path(copied).is_file(), 'Missing paper artifact: ' + str(copied))
    require(Path(source).read_bytes() == Path(copied).read_bytes(),
            'Paper copy differs from authenticated source: ' + str(copied))


def panel_scope(report):
    require(report.get('schema') == 'modebench-prompt-ablation-analysis-v1'
            and report.get('status') == 'complete', 'Source analysis is not complete')
    inventory = report.get('inventory', {})
    finalized = inventory.get('finalized_draws', inventory.get('received_draws'))
    require(inventory.get('status') == 'complete' and inventory.get('runs')
            and all(run.get('complete') is True for run in inventory['runs'])
            and inventory.get('expected_draws') == finalized
            and inventory.get('expected_draws', 0) > 0, 'Included panel has incomplete sampling')
    registries = report.get('registries', {})
    require(registries and set(registries) <= {'hosted', 'local'}, 'Unknown or missing source panel registry')
    included = [family for family, key in [('frontier', 'hosted'), ('local', 'local')] if key in registries]
    missing = [family for family in FAMILIES if family not in included]
    state = 'partial_panels' if missing else 'complete'
    scope = report.get('scope', {})
    require(report.get('experiment_status') == state
            and scope.get('included_panels') == included
            and scope.get('registered_panels') == list(FAMILIES)
            and scope.get('omitted_panels') == missing,
            'Panel scope must explicitly distinguish partial_panels from the full experiment')
    models = report.get('models', [])
    require(models and {model.get('family') for model in models} == set(included),
            'Reported model families differ from declared panel scope')
    require({run.get('family') for run in inventory['runs']} == set(included),
            'Sampling inventory differs from declared panel scope')
    return included


def publication_source(report, source_dir=None):
    source = report.get('publication_source', {})
    require(isinstance(source.get('directory'), str) and isinstance(source.get('report_path'), str),
            'Copied report does not identify its source analysis directory')
    directory = Path(source['directory']).resolve()
    if source_dir is not None:
        require(directory == Path(source_dir).resolve(), 'Requested source directory differs from copied report')
    require(Path(source['report_path']).resolve() == directory / 'analysis.json',
            'Source report path differs from its declared directory')
    return directory


def load_analyzer(report):
    source = bound_file(report['analyzer_source'])
    for name, dependency in report.get('analysis_dependencies', {}).items():
        path = bound_file(dependency)
        require(name == 'audit_hosted_modebench_completion.py', 'Unknown analysis dependency')
        dependency_spec = importlib.util.spec_from_file_location(Path(name).stem, path)
        module = importlib.util.module_from_spec(dependency_spec)
        sys.modules[dependency_spec.name] = module
        dependency_spec.loader.exec_module(module)
    # The frozen analyzer imports its sibling frozen receipt auditor. Other
    # repository helpers retain their ordinary read-only import path.
    sys.path.insert(0, str(ROOT / 'ops'))
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('_paper_prompt_ablation_analyzer_' + file_sha(source)[:16], source)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    return analyzer


def substantive_report(report):
    # Publication locations are injected after statistical computation. The
    # analyzer's bytes are pinned separately; an immutable snapshot can change
    # its own __file__ path without changing the scientific result.
    return {key: value for key, value in report.items()
            if key not in ('publication_source', 'analyzer_source', 'analysis_dependencies')}


def reconstruct(report, analyzer):
    for record in report['design'].values():
        bound_file(record)
    for record in report['registries'].values():
        bound_file(record)
    bound_file(report['prospective_analysis_plan'])
    for record in report.get('prospective_amendments', {}).values():
        bound_file(record)
    base = Path(report['design']['manifest.json']['path']).resolve().parent
    registries = report['registries']
    result = analyzer.build_report(
        base,
        hosted_registry=Path(registries['hosted']['path']) if 'hosted' in registries else None,
        local_plan=Path(registries['local']['path']) if 'local' in registries else None,
        replicates=20000,
    )
    require(substantive_report(result) == substantive_report(report),
            'Report statistics or provenance differ from authenticated reconstruction')
    return result


def csv_records(report):
    expected = {}
    for model in report['models']:
        for grading, analysis in model['analyses'].items():
            for cell, record in analysis['cells'].items():
                for arm in ('original', 'neutral', 'neutral_minus_original'):
                    for metric, value in record[arm].items():
                        key = (model['model_id'], model['family'], grading, cell, arm, metric)
                        require(key not in expected, 'Duplicate reconstructed CSV metric')
                        ci = value['ci95'] or [None, None]
                        expected[key] = (value['estimate'], ci[0], ci[1], value['defined_bootstrap_replicates'])
    return expected


def validate_csv(path, report):
    expected = csv_records(report)
    actual = {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == list(CSV_FIELDS), 'Unexpected all_cells.csv columns')
        for record in reader:
            require(set(record) == set(CSV_FIELDS), 'Malformed all_cells.csv row')
            key = tuple(record[field] for field in CSV_KEYS)
            require(key not in actual, 'Duplicate all_cells.csv metric')
            values = tuple(None if record[field] == '' else float(record[field])
                           for field in ('estimate', 'ci95_low', 'ci95_high'))
            require(all(value is None or math.isfinite(value) for value in values), 'Nonfinite CSV metric')
            actual[key] = (*values, int(record['defined_bootstrap_replicates']))
    require(actual == expected, 'all_cells.csv differs from reconstructed statistics')
    return len(actual)


def validate_figures(directory, paper, report, manifest, analyzer, included):
    require(set(manifest.get('figures', {})) == set(included), 'Figure manifest differs from report panel scope')
    verified = []
    for family in FAMILIES:
        stem = FIGURE_STEM + '_' + family
        if family not in included:
            for extension in ('pdf', 'png', 'json'):
                require(not (directory / f'{stem}.{extension}').exists()
                        and not (paper / 'figures' / f'{stem}.{extension}').exists(),
                        'Omitted panel has an unreported or fabricated figure: ' + family)
            continue
        metadata_path = bound_file(manifest['figures'][family], directory / (stem + '.json'))
        metadata = read_json(metadata_path)
        require(metadata.get('schema') == 'prompt-ablation-figure-v1' and metadata.get('family') == family
                and metadata.get('report_sha256') == file_sha(directory / 'analysis.json'),
                'Figure provenance refers to another report or family')
        plotted = [{'grading': grading, **row} for grading in ('strict', 'normalized_secondary')
                   for row in analyzer.display_rows(report, family, grading)]
        require(metadata.get('plotted_records') == plotted, 'Figure plotted values differ from reconstructed statistics')
        require(set(metadata.get('outputs', {})) == {'pdf', 'png'}, 'Missing or unexpected figure rendering')
        for extension in ('pdf', 'png'):
            source = bound_file(metadata['outputs'][extension], directory / f'{stem}.{extension}')
            exact_copy(source, paper / 'figures' / source.name)
            verified.append(source.name)
        exact_copy(metadata_path, paper / 'figures' / metadata_path.name)
        verified.append(metadata_path.name)
    return verified


def validate_python_diagnostic(paper, report):
    """Check the optional post hoc copy without changing or rerunning grading."""
    copied = paper / 'results' / PYTHON_DIAGNOSTIC
    if not copied.exists():
        return None
    root_manifest = bound_file(report['design']['manifest.json'])
    base = root_manifest.parent
    source = base / 'LOCAL_PYTHON_FAILURE_DIAGNOSTIC.json'
    exact_copy(source, copied)
    diagnostic = read_json(source)
    require(diagnostic.get('schema') == 'modebench-python-posthoc-failure-diagnostic-v1'
            and diagnostic.get('status') == 'complete' and diagnostic.get('post_hoc') is True
            and diagnostic.get('generation_calls') == 0,
            'Python diagnostic must be a complete post hoc saved-output analysis')
    require(diagnostic.get('stored_grade_disagreements') == 0
            and diagnostic.get('canonical_key_disagreements') == 0,
            'Python diagnostic disagrees with retained grades or canonical keys')
    pins = diagnostic.get('source_sha256', {})
    require(pins.get(str(root_manifest)) == report['design']['manifest.json']['sha256'],
            'Python diagnostic does not bind the report experiment')
    for name, digest in pins.items():
        bound_file({'path': name, 'sha256': digest})
    path = Path(diagnostic['draw_diagnostics_path'])
    if not path.is_absolute():
        path = ROOT / path
    path = bound_file({'path': str(path), 'sha256': diagnostic['draw_diagnostics_sha256']},
                      base / 'LOCAL_PYTHON_FAILURE_DRAW_DIAGNOSTICS.jsonl')
    draws = [json.loads(line) for line in path.read_text().splitlines()]
    identities = {(draw['checkpoint_label'], draw['arm'], draw['level'], draw['row_index'], draw['draw_index'])
                  for draw in draws}
    require(len(identities) == len(draws), 'Duplicate Python diagnostic draw')
    require({draw['arm'] for draw in draws} == {'original', 'neutral'}
            and set(diagnostic.get('by_arm', {})) == {'original', 'neutral'},
            'Python diagnostic arm inventory differs')
    require(all(draw['category'] in diagnostic['category_definition'] for draw in draws),
            'Unknown Python diagnostic category')
    for arm in ('original', 'neutral'):
        records = [draw for draw in draws if draw['arm'] == arm]
        summary = diagnostic['by_arm'][arm]
        counts = dict(Counter(draw['category'] for draw in records))
        require(summary['draws'] == len(records) and summary['category_counts'] == counts,
                'Python diagnostic category counts differ from per-draw records')
        require(summary['category_percentages'] == {key: round(100 * value / len(records), 6)
                                                  for key, value in counts.items()},
                'Python diagnostic percentages differ from per-draw records')
        candidates = [{'candidate': key, 'draws': value} for key, value in
                      Counter(draw['candidate'] for draw in records if draw['candidate']).most_common(10)]
        require(summary['most_frequent_extracted_candidates'] == candidates,
                'Python diagnostic candidate counts differ from per-draw records')
        for field, source_field in [('verified', 'verified'), ('parser_accepted', 'parser_accepted'),
                                    ('literal_constant_body_count', 'literal_constant_body'),
                                    ('input_independent_body_count', 'input_independent_body'),
                                    ('malformed_2_for_prefix_count', 'malformed_2_for_prefix')]:
            require(summary[field] == sum(draw[source_field] for draw in records),
                    'Python diagnostic ' + field + ' differs from per-draw records')
        for field, source_field in [('finish_reason_counts', 'finish_reason'),
                                    ('error_message_counts', 'error_message')]:
            require(summary[field] == dict(Counter(draw[source_field] for draw in records if source_field in draw)),
                    'Python diagnostic ' + field + ' differs from per-draw records')
        require(summary['lambda_body_ast_type_counts_among_parser_accepted'] == dict(Counter(
                    draw['lambda_body_ast_type'] for draw in records if draw['parser_accepted'])),
                'Python diagnostic AST counts differ from per-draw records')
    return {'source_path': str(source), 'source_sha256': file_sha(source),
            'draw_diagnostics_sha256': file_sha(path), 'validated_draws': len(draws),
            'post_hoc': True, 'counts_reconstructed': True}


def check(paper_dir=PAPER, source_dir=None):
    paper = Path(paper_dir).resolve()
    copied_json = paper / 'results' / (RESULT_STEM + '.json')
    copied_report = read_json(copied_json)
    directory = publication_source(copied_report, source_dir)
    report = read_json(directory / 'analysis.json')
    included = panel_scope(report)
    require(publication_source(report) == directory, 'Source publication path differs')
    manifest = read_json(directory / 'artifact_manifest.json')
    require(manifest.get('status') == 'complete', 'Source artifact manifest is incomplete')
    required = {'analysis.json', 'all_cells.csv', 'appendix.tex', 'README.md'}
    require(required <= set(manifest.get('outputs', {})), 'Unbound source analysis artifacts')
    for name, record in manifest['outputs'].items():
        require(Path(name).name == name, 'Unexpected source artifact path')
        bound_file(record, directory / name)
    bound_file(manifest['analyzer_source'])
    require(manifest['analyzer_source']['sha256'] == report['analyzer_source']['sha256'],
            'Report and artifact manifest use different analysis code')
    exact_copy(directory / 'analysis.json', copied_json)
    exact_copy(directory / 'appendix.tex', paper / 'results' / (RESULT_STEM + '.tex'))
    python_diagnostic = validate_python_diagnostic(paper, report)
    analyzer = load_analyzer(report)
    reconstruct(report, analyzer)
    expected_tex = analyzer.render_appendix(report)
    require((directory / 'appendix.tex').read_text() == expected_tex,
            'Appendix differs from reconstructed statistical report')
    if report['experiment_status'] == 'partial_panels':
        # Scope must also be visible to readers, not only machine metadata.
        lower = expected_tex.lower()
        require(any(word in lower for word in ('partial', 'incomplete')) and all(
                    any(label in lower for label in (('frontier', 'hosted') if family == 'frontier' else ('local',)))
                    for family in report['scope']['omitted_panels']),
                'Partial-panel appendix does not explicitly disclose omitted panels')
    rows = validate_csv(directory / 'all_cells.csv', report)
    figures = validate_figures(directory, paper, report, manifest, analyzer, included)
    return {'status': 'pass', 'experiment_status': report['experiment_status'],
            'included_panels': included, 'omitted_panels': report['scope']['omitted_panels'],
            'source_directory': str(directory), 'source_report_sha256': file_sha(directory / 'analysis.json'),
            'paper_directory': str(paper), 'models': len(report['models']),
            'validated_csv_metrics': rows, 'validated_figure_files': figures,
            'statistics_reconstructed': True, 'python_failure_diagnostic': python_diagnostic, 'api_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--paper-dir', type=Path, default=PAPER)
    args = parser.parse_args()
    try:
        result = check(args.paper_dir, args.source_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('[paper-prompt-ablation] FAIL: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
