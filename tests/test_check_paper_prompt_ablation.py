"""Synthetic end-to-end tamper checks; never alter experiment or paper data."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import check_paper_prompt_ablation as checker


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def binding(path):
    return {'path': str(path.resolve()), 'sha256': checker.file_sha(path)}


# This isolated fixture analyzer reconstructs actual toy eight-draw statistics
# from raw outcomes. It cannot contact a model; the real checker exercises the
# same source binding, dynamic loading, reconstruction, and copy checks.
SYNTHETIC_ANALYZER = '''from pathlib import Path
import json, hashlib

def binding(path):
    return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}

def build_report(base,hosted_registry=None,local_plan=None,replicates=20000):
    assert replicates == 20000
    report=json.loads((Path(base)/'template.json').read_text())
    outcomes=json.loads((Path(base)/'raw.json').read_text())
    metrics={}
    for arm in ('original','neutral'):
        values=outcomes[arm]
        if len(values)!=8:raise ValueError('Incomplete synthetic eight-draw group')
        correct=[v for v in values if v is not None]
        p=float(bool(correct));d=float(len(set(correct)))
        metrics[arm]={'pass1':len(correct)/8,'pass8':p,'distinct8':d,'b8':d-p}
    metrics['neutral_minus_original']={key:metrics['neutral'][key]-metrics['original'][key] for key in metrics['original']}
    cell={arm:{key:{'estimate':value,'ci95':[value,value], 'defined_bootstrap_replicates':replicates}
               for key,value in values.items()} for arm,values in metrics.items()}
    for model in report['models']:
        model['analyses']={g:{'cells':{'level2/mathir':cell}} for g in ('strict','normalized_secondary')}
    report['analyzer_source']=binding(__file__)
    return report

def render_appendix(report):
    scope=('Partial panels; omitted '+','.join(report['scope']['omitted_panels'])+'. ' if report['experiment_status']=='partial_panels' else 'Complete panels. ')
    value=report['models'][0]['analyses']['strict']['cells']['level2/mathir']['neutral_minus_original']['distinct8']['estimate']
    return scope+'Toy reconstructed distinct8 effect: '+str(value)+'\\n'

def display_rows(report,family,grading):
    return [{'label':m['model_id'],'level':2,'domain':'mathir','seed_count':1,
             'metrics':m['analyses'][grading]['cells']['level2/mathir']}
            for m in report['models'] if m['family']==family]
'''


@pytest.fixture
def fixture(tmp_path):
    base = tmp_path / 'experiment'
    source = base / 'analysis_complete'
    paper = tmp_path / 'paper'
    source.mkdir(parents=True)
    (paper / 'results').mkdir(parents=True)
    (paper / 'figures').mkdir()
    for path, value in [(base / 'manifest.json', {}), (base / 'local_plan.json', {})]:
        write_json(path, value)
    (base / 'ANALYSIS_PLAN.md').write_text('Synthetic paired eight-draw toy test.\n')
    code = source / 'analysis_code/analyzer.py'
    code.parent.mkdir()
    code.write_text(SYNTHETIC_ANALYZER)
    template = {
        'schema': 'modebench-prompt-ablation-analysis-v1', 'status': 'complete',
        'experiment_status': 'partial_panels',
        'scope': {'registered_panels': ['frontier', 'local'], 'included_panels': ['local'],
                  'omitted_panels': ['frontier'], 'completion_applies_to': 'included panels only'},
        'publication_source': {'directory': str(source.resolve()), 'report_path': str((source / 'analysis.json').resolve())},
        'design': {'manifest.json': binding(base / 'manifest.json')},
        'registries': {'local': binding(base / 'local_plan.json')},
        'prospective_analysis_plan': binding(base / 'ANALYSIS_PLAN.md'), 'prospective_amendments': {},
        'inventory': {'status': 'complete', 'expected_draws': 16, 'finalized_draws': 16, 'durable_draws': 16,
                      'runs': [{'family': 'local', 'complete': True}]},
        'models': [{'model_id': 'synthetic_local', 'family': 'local'}], 'local_seed_groups': {},
    }
    write_json(base / 'template.json', template)
    write_json(base / 'raw.json', {'original': ['a', 'a'] + [None] * 6,
                                   'neutral': ['a', 'b', 'b'] + [None] * 5})
    spec = importlib.util.spec_from_file_location('_toy_analysis', code)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    report = analyzer.build_report(base, local_plan=base / 'local_plan.json')
    write_json(source / 'analysis.json', report)
    (source / 'appendix.tex').write_text(analyzer.render_appendix(report))
    (source / 'README.md').write_text('Synthetic complete local panel; partial experiment.\n')
    with (source / 'all_cells.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(checker.CSV_FIELDS)
        for key, values in checker.csv_records(report).items():
            writer.writerow((*key, *values))
    stem = checker.FIGURE_STEM + '_local'
    for extension, content in [('pdf', b'%PDF-SYNTHETIC\n'), ('png', b'\x89PNG\r\nSYNTHETIC')]:
        (source / f'{stem}.{extension}').write_bytes(content)
    write_json(source / (stem + '.json'), {
        'schema': 'prompt-ablation-figure-v1', 'family': 'local',
        'report_sha256': checker.file_sha(source / 'analysis.json'),
        'plotted_records': [{'grading': grading, **row} for grading in ('strict', 'normalized_secondary')
                            for row in analyzer.display_rows(report, 'local', grading)],
        'outputs': {ext: binding(source / f'{stem}.{ext}') for ext in ('pdf', 'png')},
    })
    write_json(source / 'artifact_manifest.json', {
        'status': 'complete', 'analyzer_source': binding(code),
        'outputs': {name: binding(source / name) for name in ('analysis.json', 'all_cells.csv', 'appendix.tex', 'README.md')},
        'figures': {'local': binding(source / (stem + '.json'))},
    })
    for name, target in [('analysis.json', checker.RESULT_STEM + '.json'), ('appendix.tex', checker.RESULT_STEM + '.tex')]:
        shutil.copyfile(source / name, paper / 'results' / target)
    for extension in ('pdf', 'png', 'json'):
        shutil.copyfile(source / f'{stem}.{extension}', paper / 'figures' / f'{stem}.{extension}')
    return {'base': base, 'source': source, 'paper': paper, 'code': code, 'report': report}


def rebind_source(fixture, name):
    source = fixture['source']
    manifest = checker.read_json(source / 'artifact_manifest.json')
    manifest['outputs'][name] = binding(source / name)
    write_json(source / 'artifact_manifest.json', manifest)


def replace_report(fixture, report):
    write_json(fixture['source'] / 'analysis.json', report)
    write_json(fixture['paper'] / 'results' / (checker.RESULT_STEM + '.json'), report)
    rebind_source(fixture, 'analysis.json')


def test_valid_partial_panel_rebuilds_statistics_and_exact_copies(fixture):
    result = checker.check(fixture['paper'])
    assert result['status'] == 'pass'
    assert result['experiment_status'] == 'partial_panels'
    assert result['included_panels'] == ['local'] and result['omitted_panels'] == ['frontier']
    assert result['statistics_reconstructed'] is True and result['api_calls'] == 0
    assert result['validated_csv_metrics'] == 24
    assert len(result['validated_figure_files']) == 3


@pytest.mark.parametrize('relative', [
    'results/' + checker.RESULT_STEM + '.tex',
    'results/' + checker.RESULT_STEM + '.json',
    'figures/' + checker.FIGURE_STEM + '_local.pdf',
    'figures/' + checker.FIGURE_STEM + '_local.png',
    'figures/' + checker.FIGURE_STEM + '_local.json',
])
def test_paper_tampering_fails_even_when_source_is_unchanged(fixture, relative):
    path = fixture['paper'] / relative
    # Whitespace is semantically harmless in JSON but fails the required exact copy.
    path.write_bytes(path.read_bytes() + b' \n')
    with pytest.raises(ValueError, match='Paper copy differs'):
        checker.check(fixture['paper'])


def test_rebound_report_numbers_cannot_bypass_raw_reconstruction(fixture):
    report = checker.read_json(fixture['source'] / 'analysis.json')
    report['models'][0]['analyses']['strict']['cells']['level2/mathir']['neutral']['distinct8']['estimate'] = 7.0
    replace_report(fixture, report)
    with pytest.raises(ValueError, match='authenticated reconstruction'):
        checker.check(fixture['paper'])


def test_rebound_csv_numbers_cannot_bypass_reconstruction(fixture):
    path = fixture['source'] / 'all_cells.csv'
    with path.open(newline='') as handle:
        records = list(csv.reader(handle))
    records[1][6] = '0.999'
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(records)
    rebind_source(fixture, 'all_cells.csv')
    with pytest.raises(ValueError, match='all_cells.csv differs'):
        checker.check(fixture['paper'])


def test_rebound_figure_values_cannot_bypass_reconstructed_plot_data(fixture):
    name = checker.FIGURE_STEM + '_local.json'
    metadata = checker.read_json(fixture['source'] / name)
    metadata['plotted_records'][0]['metrics']['original']['pass8']['estimate'] = 0.0
    write_json(fixture['source'] / name, metadata)
    write_json(fixture['paper'] / 'figures' / name, metadata)
    manifest = checker.read_json(fixture['source'] / 'artifact_manifest.json')
    manifest['figures']['local'] = binding(fixture['source'] / name)
    write_json(fixture['source'] / 'artifact_manifest.json', manifest)
    with pytest.raises(ValueError, match='plotted values differ'):
        checker.check(fixture['paper'])


def test_local_only_report_cannot_claim_complete_experiment(fixture):
    report = checker.read_json(fixture['source'] / 'analysis.json')
    report['experiment_status'] = 'complete'
    replace_report(fixture, report)
    with pytest.raises(ValueError, match='partial_panels'):
        checker.check(fixture['paper'])


@pytest.mark.parametrize('location', ['source', 'paper'])
def test_omitted_hosted_panel_cannot_have_a_fabricated_figure(fixture, location):
    directory = fixture['source'] if location == 'source' else fixture['paper'] / 'figures'
    (directory / (checker.FIGURE_STEM + '_frontier.pdf')).write_bytes(b'fabricated')
    with pytest.raises(ValueError, match='Omitted panel'):
        checker.check(fixture['paper'])


def test_incomplete_group_blocks_paper_check(fixture):
    raw = checker.read_json(fixture['base'] / 'raw.json')
    raw['neutral'].pop()
    write_json(fixture['base'] / 'raw.json', raw)
    with pytest.raises(ValueError, match='Incomplete synthetic'):
        checker.check(fixture['paper'])


def test_changed_analyzer_source_is_rejected(fixture):
    fixture['code'].write_text(fixture['code'].read_text() + '\n# Changed code\n')
    with pytest.raises(ValueError, match='Bound file changed'):
        checker.check(fixture['paper'])


def test_explicit_source_directory_cannot_redirect_copied_report(fixture, tmp_path):
    with pytest.raises(ValueError, match='Requested source directory differs'):
        checker.check(fixture['paper'], tmp_path / 'different')


def test_durable_partial_batches_do_not_count_as_a_complete_panel(fixture):
    report = checker.read_json(fixture['source'] / 'analysis.json')
    report['inventory']['finalized_draws'] = 8
    report['inventory']['durable_draws'] = 16
    replace_report(fixture, report)
    with pytest.raises(ValueError, match='incomplete sampling'):
        checker.check(fixture['paper'])


def add_python_diagnostic(fixture):
    """An inert eight-draw-per-arm diagnostic with independently stated totals."""
    draws = []
    for arm in ('original', 'neutral'):
        for index in range(8):
            verified = arm == 'original'
            draws.append({'checkpoint_label': 'toy_python_drgrpo_s43', 'training_seed': 43,
                          'arm': arm, 'level': 2, 'row_index': 0, 'draw_index': index,
                          'category': 'verified_success' if verified else 'invalid_python_expression',
                          'candidate': 'lambda n: 2' if verified else 'lambda n: 2 for',
                          'verified': verified, 'parser_accepted': verified, 'finish_reason': 'stop',
                          'literal_constant_body': verified, 'input_independent_body': verified,
                          'malformed_2_for_prefix': not verified,
                          **({'lambda_body_ast_type': 'Constant'} if verified else {'error_message': 'invalid syntax'})})
    path = fixture['base'] / 'LOCAL_PYTHON_FAILURE_DRAW_DIAGNOSTICS.jsonl'
    path.write_text(''.join(json.dumps(draw) + '\n' for draw in draws))
    diagnostic = {'schema': 'modebench-python-posthoc-failure-diagnostic-v1', 'status': 'complete',
                  'post_hoc': True, 'generation_calls': 0, 'stored_grade_disagreements': 0,
                  'canonical_key_disagreements': 0,
                  'source_sha256': {str(p.resolve()): checker.file_sha(p)
                                    for p in [fixture['base'] / 'manifest.json', fixture['base'] / 'raw.json']},
                  'draw_diagnostics_path': str(path), 'draw_diagnostics_sha256': checker.file_sha(path),
                  'category_definition': {'verified_success': 'valid', 'invalid_python_expression': 'invalid'},
                  'by_arm': {}}
    for arm in ('original', 'neutral'):
        verified = arm == 'original'
        category = 'verified_success' if verified else 'invalid_python_expression'
        diagnostic['by_arm'][arm] = {
            'draws': 8, 'category_counts': {category: 8}, 'category_percentages': {category: 100.0},
            'most_frequent_extracted_candidates': [{'candidate': 'lambda n: 2' if verified else 'lambda n: 2 for', 'draws': 8}],
            'verified': 8 if verified else 0, 'parser_accepted': 8 if verified else 0,
            'literal_constant_body_count': 8 if verified else 0,
            'input_independent_body_count': 8 if verified else 0,
            'malformed_2_for_prefix_count': 0 if verified else 8,
            'finish_reason_counts': {'stop': 8}, 'error_message_counts': {} if verified else {'invalid syntax': 8},
            'lambda_body_ast_type_counts_among_parser_accepted': {'Constant': 8} if verified else {},
        }
    source = fixture['base'] / 'LOCAL_PYTHON_FAILURE_DIAGNOSTIC.json'
    copied = fixture['paper'] / 'results' / checker.PYTHON_DIAGNOSTIC
    write_json(source, diagnostic)
    shutil.copy2(source, copied)
    return diagnostic, source, copied, path


def test_optional_python_diagnostic_reconstructs_categories_and_candidate_counts(fixture):
    add_python_diagnostic(fixture)
    result = checker.check(fixture['paper'])['python_failure_diagnostic']
    assert result['validated_draws'] == 16 and result['counts_reconstructed'] is True
    assert result['post_hoc'] is True


@pytest.mark.parametrize('fault', ['copied', 'source_binding', 'draw_hash', 'category_counts',
                                   'percentages', 'candidate_counts', 'duplicate_draw'])
def test_python_diagnostic_tampering_fails(fixture, fault):
    diagnostic, source, copied, draws = add_python_diagnostic(fixture)
    if fault == 'copied':
        copied.write_text(copied.read_text() + ' ')
    elif fault == 'source_binding':
        raw = fixture['base'] / 'raw.json'
        raw.write_text(raw.read_text() + ' ')
    elif fault == 'draw_hash':
        draws.write_text(draws.read_text() + '\n')
    else:
        if fault == 'category_counts':
            diagnostic['by_arm']['neutral']['category_counts']['invalid_python_expression'] -= 1
        elif fault == 'percentages':
            diagnostic['by_arm']['neutral']['category_percentages']['invalid_python_expression'] = 99.0
        elif fault == 'candidate_counts':
            diagnostic['by_arm']['original']['most_frequent_extracted_candidates'][0]['draws'] -= 1
        else:
            with draws.open('a') as output:
                output.write(draws.read_text().splitlines()[0] + '\n')
            diagnostic['draw_diagnostics_sha256'] = checker.file_sha(draws)
        write_json(source, diagnostic)
        shutil.copy2(source, copied)
    with pytest.raises(ValueError, match='copy differs|Bound file changed|counts differ|percentages differ|Duplicate Python'):
        checker.check(fixture['paper'])
