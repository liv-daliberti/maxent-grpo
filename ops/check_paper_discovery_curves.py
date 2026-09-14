#!/usr/bin/env python3
"""Authenticate copied discovery curves and reconstruct their complete analysis."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from check_paper_prompt_ablation import (
    ROOT, PAPER, bound_file, exact_copy, file_sha, publication_source, read_json,
    require, substantive_report, validate_csv,
)

RESULT_STEM = 'modebench_discovery_curves_20260911'
FAMILIES = ('frontier', 'local')
REPORT_SCHEMA = 'modebench-discovery-curves-analysis-v1'
FIGURE_SCHEMA = 'modebench-discovery-figure-v1'


def figure_stems(included):
    return {key: stem for family in included for key, stem in (
        (family, 'modebench_discovery_curves_' + family),
        ('correct_budget_' + family, 'modebench_discovery_correct_budget_' + family))}


def panel_scope(report):
    require(report.get('schema') == REPORT_SCHEMA and report.get('status') == 'complete',
            'Discovery analysis is not complete')
    inventory = report.get('inventory', {})
    require(inventory.get('status') == 'complete' and inventory.get('runs')
            and all(run.get('complete') is True for run in inventory['runs'])
            and inventory.get('expected_draws', 0) > 0
            and inventory['expected_draws'] == inventory.get('finalized_draws'),
            'Discovery analysis includes incomplete sampling')
    registries = report.get('registries', {})
    require(registries and set(registries) <= {'hosted', 'local'}, 'Unknown discovery panel registry')
    included = [family for family, key in [('frontier', 'hosted'), ('local', 'local')] if key in registries]
    omitted = [family for family in FAMILIES if family not in included]
    scope = report.get('scope', {})
    require(report.get('experiment_status') == ('partial_panels' if omitted else 'complete')
            and scope.get('included_panels') == included
            and scope.get('omitted_panels') == omitted
            and scope.get('registered_panels') == list(FAMILIES),
            'Discovery scope does not distinguish partial panels from the full experiment')
    require({model.get('family') for model in report.get('models', [])} == set(included)
            and {run.get('family') for run in inventory['runs']} == set(included),
            'Discovery model or sample inventory differs from scope')
    return included


def load_analyzer(report):
    source = bound_file(report['analyzer_source'])
    for name, record in report.get('analysis_dependencies', {}).items():
        require(Path(name).name == name and name.endswith('.py'), 'Unexpected discovery code dependency')
        path = bound_file(record)
        spec = importlib.util.spec_from_file_location(Path(name).stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    sys.path.insert(0, str(ROOT / 'ops'))
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('_publication_discovery_' + file_sha(source)[:16], source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reconstruct(report, analyzer, included):
    for collection in ('design', 'registries'):
        for record in report[collection].values():
            bound_file(record)
    bound_file(report['prospective_analysis_plan'])
    for record in report.get('prospective_amendments', {}).values():
        bound_file(record)
    base = Path(report['design']['manifest.json']['path']).resolve().parent
    registries = report['registries']
    scope = 'full' if len(included) == 2 else ('hosted' if included == ['frontier'] else 'local')
    rebuilt = analyzer.build_report(
        base, local_plan=Path(registries['local']['path']) if 'local' in registries else None,
        hosted_registry=Path(registries['hosted']['path']) if 'hosted' in registries else None,
        replicates=20000, scope=scope,
    )
    require(substantive_report(rebuilt) == substantive_report(report),
            'Discovery statistics or provenance differ from authenticated reconstruction')


def validate_figures(source, paper, report, manifest, analyzer, included):
    stems = figure_stems(included)
    require(set(manifest.get('figures', {})) == set(stems),
            'Discovery figure manifest differs from included panels')
    for key, stem in figure_stems(FAMILIES).items():
        if key not in stems:
            require(not any((directory / (stem + '.' + extension)).exists()
                            for directory in (source, paper / 'figures') for extension in ('pdf', 'png', 'json')),
                    'Omitted discovery panel has a published figure: ' + key)
    verified = []
    for key, stem in stems.items():
        metadata_path = bound_file(manifest['figures'][key], source / (stem + '.json'))
        metadata = read_json(metadata_path)
        require(metadata.get('schema') == FIGURE_SCHEMA and metadata.get('family') == key
                and metadata.get('report_sha256') == file_sha(source / 'analysis.json'),
                'Discovery figure binds another report or plot')
        expected = [{'grading': grading, **row} for grading in ('strict', 'normalized_secondary')
                    for row in analyzer.display_rows(report, key, grading)]
        require(metadata.get('plotted_records') == expected,
                'Discovery figure values differ from reconstructed report')
        require(set(metadata.get('outputs', {})) == {'pdf', 'png'}, 'Missing discovery figure rendering')
        for extension in ('pdf', 'png'):
            path = bound_file(metadata['outputs'][extension], source / (stem + '.' + extension))
            exact_copy(path, paper / 'figures' / path.name)
            verified.append(path.name)
        exact_copy(metadata_path, paper / 'figures' / metadata_path.name)
        verified.append(metadata_path.name)
    return verified


def check(paper_dir=PAPER, source_dir=None):
    paper = Path(paper_dir).resolve()
    copied = paper / 'results' / (RESULT_STEM + '.json')
    source = publication_source(read_json(copied), source_dir)
    report = read_json(source / 'analysis.json')
    require(publication_source(report) == source, 'Discovery publication source differs')
    included = panel_scope(report)
    manifest = read_json(source / 'artifact_manifest.json')
    require(manifest.get('status') == 'complete', 'Discovery artifact manifest is incomplete')
    require({'analysis.json', 'all_cells.csv', 'appendix.tex'} <= set(manifest.get('outputs', {})),
            'Unbound discovery report artifacts')
    for name, record in manifest['outputs'].items():
        require(Path(name).name == name, 'Unexpected discovery source artifact path')
        bound_file(record, source / name)
    bound_file(manifest['analyzer_source'])
    require(manifest['analyzer_source']['sha256'] == report['analyzer_source']['sha256'],
            'Discovery report and manifest bind different analysis code')
    exact_copy(source / 'analysis.json', copied)
    exact_copy(source / 'appendix.tex', paper / 'results' / (RESULT_STEM + '.tex'))
    analyzer = load_analyzer(report)
    reconstruct(report, analyzer, included)
    expected = analyzer.render_appendix(report)
    require((source / 'appendix.tex').read_text() == expected,
            'Discovery appendix differs from reconstructed report')
    if report['experiment_status'] == 'partial_panels':
        require(('partial' in expected.lower() or 'incomplete' in expected.lower())
                and all(family in expected.lower() or (family == 'frontier' and 'hosted' in expected.lower())
                        for family in report['scope']['omitted_panels']),
                'Discovery appendix fails to disclose omitted panels')
    metrics = validate_csv(source / 'all_cells.csv', report)
    figures = validate_figures(source, paper, report, manifest, analyzer, included)
    return {'status': 'pass', 'experiment_status': report['experiment_status'],
            'included_panels': included, 'omitted_panels': report['scope']['omitted_panels'],
            'source_directory': str(source), 'source_report_sha256': file_sha(source / 'analysis.json'),
            'paper_directory': str(paper), 'validated_csv_metrics': metrics,
            'validated_figure_files': figures, 'statistics_reconstructed': True, 'api_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path, default=PAPER)
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    try:
        result = check(args.paper_dir, args.source_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('[paper-discovery-curves] FAIL: ' + str(error), file=sys.stderr)
        return 1
    import json
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
