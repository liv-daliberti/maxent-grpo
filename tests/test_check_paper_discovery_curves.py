"""Exercise the separate discovery publication checker on isolated 64-draw data."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'ops'))
import check_paper_discovery_curves as checker
from check_paper_prompt_ablation import CSV_FIELDS, csv_records


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def binding(path):
    return {'path': str(path.resolve()), 'sha256': checker.file_sha(path)}


ANALYZER = '''from pathlib import Path
from collections import Counter
import json,math,hashlib

def binding(path):
 return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}

def build_report(base,local_plan=None,hosted_registry=None,replicates=20000,scope='full'):
 assert replicates==20000 and scope=='local' and local_plan and hosted_registry is None
 report=json.loads((Path(base)/'template.json').read_text())
 raw=json.loads((Path(base)/'raw.json').read_text());metrics={}
 for arm in ('original','neutral'):
  outcomes=raw[arm]
  if len(outcomes)!=64:raise ValueError('Incomplete 64-draw pool')
  counts=Counter(x for x in outcomes if x is not None);correct=sum(counts.values());values={}
  for k in (8,64):
   choose=lambda n:math.comb(n,k) if n>=k else 0
   values['rarefaction/pass/k'+str(k)]=1-choose(64-correct)/math.comb(64,k)
   values['rarefaction/distinct/k'+str(k)]=sum(1-choose(n0)/math.comb(64,k) for n0 in (64-n for n in counts.values()))
  values['gain8to64/rarefaction/distinct']=values['rarefaction/distinct/k64']-values['rarefaction/distinct/k8']
  values['conditional_joint/distinct/m8']=values['rarefaction/distinct/k8']
  metrics[arm]=values
 metrics['neutral_minus_original']={k:metrics['neutral'][k]-metrics['original'][k] for k in metrics['original']}
 cell={a:{k:{'estimate':v,'ci95':[v,v],'defined_bootstrap_replicates':20000,'degenerate_ci':True} for k,v in values.items()} for a,values in metrics.items()}
 report['models'][0]['analyses']={g:{'cells':{'level2/mathir':cell}} for g in ('strict','normalized_secondary')}
 report['analyzer_source']=binding(__file__)
 return report

def render_appendix(report):
 value=report['models'][0]['analyses']['strict']['cells']['level2/mathir']['neutral_minus_original']['rarefaction/distinct/k64']['estimate']
 return 'Complete local panel; the frontier panel is omitted and the overall experiment is incomplete. Effect '+str(value)+'\\n'

def display_rows(report,figure_key,grading):
 metrics=report['models'][0]['analyses'][grading]['cells']['level2/mathir']
 if figure_key.startswith('correct_budget_'):
  metrics={a:{k:v for k,v in vs.items() if k.startswith('conditional_joint/')} for a,vs in metrics.items()}
 return [{'grading':grading,'model_id':'toy_local','cell':'level2/mathir','metrics':metrics}]
'''


@pytest.fixture
def fixture(tmp_path):
    base = tmp_path / 'experiment'
    source = base / 'analysis_complete'
    paper = tmp_path / 'paper'
    source.mkdir(parents=True)
    (paper / 'results').mkdir(parents=True)
    (paper / 'figures').mkdir()
    for name in ('manifest.json', 'local_plan.json', 'support_reference.json'):
        write_json(base / name, {})
    (base / 'ANALYSIS_PLAN.md').write_text('Frozen synthetic64 paired discovery protocol.\n')
    code = source / 'analysis_code/analyzer.py'
    code.parent.mkdir()
    code.write_text(ANALYZER)
    template = {'schema': checker.REPORT_SCHEMA, 'status': 'complete', 'experiment_status': 'partial_panels',
                'scope': {'registered_panels': ['frontier', 'local'], 'included_panels': ['local'], 'omitted_panels': ['frontier']},
                'publication_source': {'directory': str(source), 'report_path': str(source / 'analysis.json')},
                'design': {name: binding(base / name) for name in ('manifest.json', 'support_reference.json')},
                'registries': {'local': binding(base / 'local_plan.json')},
                'prospective_analysis_plan': binding(base / 'ANALYSIS_PLAN.md'), 'prospective_amendments': {},
                'inventory': {'status': 'complete', 'expected_draws': 128, 'finalized_draws': 128,
                              'runs': [{'family': 'local', 'complete': True}]},
                'models': [{'model_id': 'toy_local', 'family': 'local'}], 'local_seed_groups': {}}
    write_json(base / 'template.json', template)
    write_json(base / 'raw.json', {'original': ['a'] * 64, 'neutral': ['a'] * 32 + ['b'] * 32})
    spec = importlib.util.spec_from_file_location('_toy_discovery', code)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    report = analyzer.build_report(base, local_plan=base / 'local_plan.json', scope='local')
    write_json(source / 'analysis.json', report)
    (source / 'appendix.tex').write_text(analyzer.render_appendix(report))
    with (source / 'all_cells.csv').open('w', newline='') as output:
        writer = csv.writer(output);writer.writerow(CSV_FIELDS)
        for key, values in csv_records(report).items():
            writer.writerow((*key, *values))
    figures = {}
    for key, stem in checker.figure_stems(['local']).items():
        for suffix in ('pdf', 'png'):
            (source / (stem + '.' + suffix)).write_bytes((key + suffix).encode())
            shutil.copy2(source / (stem + '.' + suffix), paper / 'figures' / (stem + '.' + suffix))
        metadata = {'schema': checker.FIGURE_SCHEMA, 'family': key,
                    'report_sha256': checker.file_sha(source / 'analysis.json'),
                    'outputs': {suffix: binding(source / (stem + '.' + suffix)) for suffix in ('pdf', 'png')},
                    'plotted_records': [row for grading in ('strict', 'normalized_secondary')
                                        for row in analyzer.display_rows(report, key, grading)]}
        write_json(source / (stem + '.json'), metadata)
        shutil.copy2(source / (stem + '.json'), paper / 'figures' / (stem + '.json'))
        figures[key] = binding(source / (stem + '.json'))
    write_json(source / 'artifact_manifest.json', {'status': 'complete', 'analyzer_source': binding(code),
               'outputs': {name: binding(source / name) for name in ('analysis.json', 'all_cells.csv', 'appendix.tex')},
               'figures': figures})
    shutil.copy2(source / 'analysis.json', paper / 'results' / (checker.RESULT_STEM + '.json'))
    shutil.copy2(source / 'appendix.tex', paper / 'results' / (checker.RESULT_STEM + '.tex'))
    return {'base': base, 'source': source, 'paper': paper, 'report': report, 'code': code}


def rebind(fixture, name):
    path = fixture['source'] / 'artifact_manifest.json'
    manifest = checker.read_json(path)
    manifest['outputs'][name] = binding(fixture['source'] / name)
    write_json(path, manifest)


def test_complete_local_report_rebuilds_both_figures(fixture):
    result = checker.check(fixture['paper'])
    assert result['status'] == 'pass' and result['experiment_status'] == 'partial_panels'
    assert len(result['validated_figure_files']) == 6 and result['statistics_reconstructed']


@pytest.mark.parametrize('name', ['results/' + checker.RESULT_STEM + '.tex',
                                   'results/' + checker.RESULT_STEM + '.json',
                                   'figures/modebench_discovery_curves_local.pdf',
                                   'figures/modebench_discovery_correct_budget_local.json'])
def test_changed_paper_copy_is_rejected(fixture, name):
    path = fixture['paper'] / name
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='Paper copy differs'):
        checker.check(fixture['paper'])


def test_rebound_statistics_still_require_raw_reconstruction(fixture):
    report = fixture['report']
    report['models'][0]['analyses']['strict']['cells']['level2/mathir']['neutral']['rarefaction/distinct/k64']['estimate'] = 8
    write_json(fixture['source'] / 'analysis.json', report)
    shutil.copy2(fixture['source'] / 'analysis.json', fixture['paper'] / 'results' / (checker.RESULT_STEM + '.json'))
    rebind(fixture, 'analysis.json')
    with pytest.raises(ValueError, match='authenticated reconstruction'):
        checker.check(fixture['paper'])


def test_prior_eight_draws_cannot_substitute_for_fresh64(fixture):
    raw = checker.read_json(fixture['base'] / 'raw.json')
    raw['neutral'] = raw['neutral'][:8]
    write_json(fixture['base'] / 'raw.json', raw)
    with pytest.raises(ValueError, match='Incomplete 64-draw'):
        checker.check(fixture['paper'])


def test_changed_support_binding_is_rejected(fixture):
    write_json(fixture['base'] / 'support_reference.json', {'unsupported_exact_claim': 2})
    with pytest.raises(ValueError, match='Bound file changed'):
        checker.check(fixture['paper'])


def test_rebound_plotted_conditional_values_are_rejected(fixture):
    stem = 'modebench_discovery_correct_budget_local'
    path = fixture['source'] / (stem + '.json')
    metadata = checker.read_json(path)
    metadata['plotted_records'][0]['metrics']['neutral']['conditional_joint/distinct/m8']['estimate'] = 42
    write_json(path, metadata)
    shutil.copy2(path, fixture['paper'] / 'figures' / path.name)
    manifest_path = fixture['source'] / 'artifact_manifest.json'
    manifest = checker.read_json(manifest_path)
    manifest['figures']['correct_budget_local'] = binding(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='figure values differ'):
        checker.check(fixture['paper'])


@pytest.mark.parametrize('where', ['source', 'paper'])
def test_omitted_frontier_cannot_have_a_discovery_plot(fixture, where):
    directory = fixture['source'] if where == 'source' else fixture['paper'] / 'figures'
    (directory / 'modebench_discovery_correct_budget_frontier.pdf').write_bytes(b'fabricated')
    with pytest.raises(ValueError, match='Omitted discovery panel'):
        checker.check(fixture['paper'])


def test_incomplete_inventory_cannot_claim_publication(fixture):
    report = fixture['report'];report['inventory']['finalized_draws'] = 64
    write_json(fixture['source'] / 'analysis.json', report)
    with pytest.raises(ValueError, match='incomplete sampling'):
        checker.check(fixture['paper'])
