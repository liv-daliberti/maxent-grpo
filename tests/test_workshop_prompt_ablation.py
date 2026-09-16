"""Check publication scope and standalone workshop ablation assets in temp trees."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'paper/mathai2026/check_submission.py'
REPORT = 'results/modebench_prompt_ablation_20260911.json'
TEX = 'results/modebench_prompt_ablation_20260911.tex'
FAMILIES = ['frontier', 'local']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    module = load('workshop_prompt_ablation_test', SCRIPT)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    return module


def populate(module, families):
    snapshot = {'copied_files': {}}
    if not families:
        return snapshot
    missing = [family for family in FAMILIES if family not in families]
    report = {
        'schema': 'modebench-prompt-ablation-analysis-v1', 'status': 'complete',
        'experiment_status': 'partial_panels' if missing else 'complete',
        'scope': {'included_panels': families, 'registered_panels': FAMILIES,
                  'omitted_panels': missing},
        'registries': {('hosted' if family == 'frontier' else family):
                       {'path': '/unavailable/experiment/archive', 'sha256': 'unused'}
                       for family in families},
        'models': [{'family': family} for family in families],
        'inventory': {'status': 'complete', 'expected_draws': 1536 * len(families),
                      'finalized_draws': 1536 * len(families),
                      'runs': [{'family': family, 'complete': True} for family in families]},
    }
    write(module.ROOT / REPORT, report)
    write(module.ROOT / TEX, ('This local-plus-frontier experiment is incomplete.' if missing else '')
          + ''.join('\\label{fig:prompt-hints-' + family + '}' for family in families))
    names = {REPORT, TEX}
    for family in families:
        stem = f'figures/modebench_prompt_ablation_{family}'
        for suffix in ('pdf', 'png'):
            write(module.ROOT / (stem + '.' + suffix), family + suffix)
        metadata = {'schema': 'prompt-ablation-figure-v1', 'family': family,
                    'report_sha256': module.digest(module.ROOT / REPORT),
                    'outputs': {suffix: {'path': '/unavailable/source/' + suffix,
                                        'sha256': module.digest(module.ROOT / (stem + '.' + suffix))}
                                for suffix in ('pdf', 'png')}}
        write(module.ROOT / (stem + '.json'), metadata)
        names.update(stem + '.' + suffix for suffix in ('pdf', 'png', 'json'))
    snapshot['copied_files'] = {name: module.digest(module.ROOT / name) for name in names}
    return snapshot


def update_bound(module, snapshot, name, value):
    write(module.ROOT / name, value)
    snapshot['copied_files'][name] = module.digest(module.ROOT / name)


@pytest.mark.parametrize('families', [[], ['local'], ['frontier'], FAMILIES])
def test_scope_has_exact_figure_count(bundle, families):
    snapshot = populate(bundle, families)
    assert bundle.prompt_ablation_figures(snapshot) == ['fig:prompt-hints-' + x for x in families]


@pytest.mark.parametrize('fault', ['snapshot_hash', 'missing_png', 'omitted_figure', 'wrong_report_binding'])
def test_figure_tampering_is_rejected(bundle, fault):
    snapshot = populate(bundle, ['local'])
    stem = 'figures/modebench_prompt_ablation_local'
    if fault == 'snapshot_hash':
        write(bundle.ROOT / REPORT, '{}')
    elif fault == 'missing_png':
        (bundle.ROOT / (stem + '.png')).unlink()
    elif fault == 'omitted_figure':
        write(bundle.ROOT / 'figures/modebench_prompt_ablation_frontier.pdf', 'unreported panel')
    else:
        metadata = json.loads((bundle.ROOT / (stem + '.json')).read_text())
        metadata['report_sha256'] = 'incorrect'
        update_bound(bundle, snapshot, stem + '.json', metadata)
    with pytest.raises(SystemExit, match='FAIL:'):
        bundle.prompt_ablation_figures(snapshot)


@pytest.mark.parametrize('fault', ['full_claim', 'incomplete_sampling', 'missing_disclosure', 'extra_model'])
def test_semantically_invalid_bound_report_is_rejected(bundle, fault):
    snapshot = populate(bundle, ['local'])
    report = json.loads((bundle.ROOT / REPORT).read_text())
    if fault == 'full_claim':
        report['experiment_status'] = 'complete'
    elif fault == 'incomplete_sampling':
        report['inventory']['finalized_draws'] -= 1
    elif fault == 'extra_model':
        report['models'].append({'family': 'frontier'})
    else:
        update_bound(bundle, snapshot, TEX, 'All findings are complete.')
    update_bound(bundle, snapshot, REPORT, report)
    with pytest.raises(SystemExit, match='FAIL:'):
        bundle.prompt_ablation_figures(snapshot)


def compiled_fixture(module, monkeypatch, families):
    names = ['fig:story', 'fig:modebench-examples', 'fig:verified-support-story',
             'fig:cross-scale-terminal-effects', 'fig:maxrl-factorial',
             'fig:level2-admission', 'fig:hosted-verified-breadth',
             'fig:gpt56-temperature-curve', 'fig:hosted-graph-comparison']
    names += [f'fig:existing-supplement-{i}' for i in range(9)]
    names += ['fig:prompt-hints-' + x for x in families]
    aux = '\\newlabel{main:lastpage}{{}{4}}\n\\newlabel{sec:conclusion}{{}{4}}\n'
    aux += ''.join(f'\\newlabel{{{name}}}{{{{{i}}}{{{1 if i == 1 else (4 if i <= 7 else 6)}}}}}\n'
                   for i, name in enumerate(names, 1))
    write(module.ROOT / 'main.aux', aux)
    write(module.ROOT / 'main.log', 'Clean compiler output')
    footer = ('Anonymous Author(s) Submitted to The 6th Workshop on Mathematical Reasoning and AI '
              '(NeurIPS 2026). Do not distribute.')
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: footer + '\fpage2\fpage3\fpage4\fReferences\f')


@pytest.mark.parametrize('families', [[], ['local'], FAMILIES])
def test_compiled_total_is_exactly_18_19_or_20(bundle, monkeypatch, families):
    snapshot = populate(bundle, families)
    compiled_fixture(bundle, monkeypatch, families)
    assert bundle.check_output(snapshot) == 11 + len(families)
    with (bundle.ROOT / 'main.aux').open('a') as output:
        output.write('\\newlabel{fig:unexpected}{{21}{7}}\n')
    with pytest.raises(SystemExit, match='supplementary figures'):
        bundle.check_output(snapshot)


@pytest.mark.parametrize('fault', ['wrong_panel', 'main_page'])
def test_compiled_ablation_label_and_placement_follow_scope(bundle, monkeypatch, fault):
    snapshot = populate(bundle, ['local'])
    compiled_fixture(bundle, monkeypatch, ['local'])
    path = bundle.ROOT / 'main.aux'
    aux = path.read_text()
    aux = (aux.replace('fig:prompt-hints-local', 'fig:prompt-hints-frontier') if fault == 'wrong_panel'
           else aux.replace('\\newlabel{fig:prompt-hints-local}{{19}{6}}',
                            '\\newlabel{fig:prompt-hints-local}{{19}{4}}'))
    path.write_text(aux)
    with pytest.raises(SystemExit, match='ablation figure'):
        bundle.check_output(snapshot)


def test_cli_runs_in_standalone_directory_with_inaccessible_source_bindings(bundle):
    snapshot = populate(bundle, ['local'])
    write(bundle.ROOT / 'snapshot.json', snapshot)
    shutil.copy2(SCRIPT, bundle.ROOT / 'check_submission.py')
    result = subprocess.run([sys.executable, 'check_submission.py', '--check-ablation-assets'],
                            cwd=bundle.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '1 authenticated ablation figures' in result.stdout


@pytest.mark.parametrize('missing_report', [False, True])
def test_sync_retains_ablation_json_and_figure_companions(tmp_path, monkeypatch, missing_report):
    existing = load('existing_sync_tests', REPO / 'tests/test_paper_workshop_asset_sync.py')
    module = existing.sync_tree.__wrapped__(tmp_path, monkeypatch)
    existing.prepare_apply_fixture(module)
    with (module.WORKSHOP / 'appendix.tex').open('a') as output:
        output.write('\\input{' + TEX + '}')
    stem = 'figures/modebench_prompt_ablation_local'
    write(module.PAPER / TEX, '\\includegraphics{' + stem + '}')
    diagnostic = 'results/modebench_prompt_ablation_python_failure_diagnostic_20260911.json'
    if not missing_report:
        write(module.PAPER / REPORT, '{}')
        write(module.PAPER / diagnostic, '{}')
    for suffix in ('pdf', 'png', 'json'):
        write(module.PAPER / (stem + '.' + suffix), '{}')
    if missing_report:
        with pytest.raises(ValueError, match='modebench_prompt_ablation_20260911.json'):
            existing.invoke(module, monkeypatch, tmp_path / 'audit')
        return
    existing.invoke(module, monkeypatch, tmp_path / 'audit')
    snapshot = json.loads((module.WORKSHOP / 'snapshot.json').read_text())
    expected = [REPORT, TEX, diagnostic, *(stem + '.' + suffix for suffix in ('pdf', 'png', 'json'))]
    for name in expected:
        assert snapshot['copied_files'][name] == module.digest(module.PAPER / name)
        assert (module.WORKSHOP / name).read_bytes() == (module.PAPER / name).read_bytes()


@pytest.mark.parametrize('binding_kind', ['snapshot', 'overleaf'])
def test_existing_no_argument_output_check_uses_standalone_bindings(bundle, monkeypatch, binding_kind):
    snapshot = populate(bundle, ['local'])
    compiled_fixture(bundle, monkeypatch, ['local'])
    if binding_kind == 'snapshot':
        write(bundle.ROOT / 'snapshot.json', snapshot)
    else:
        write(bundle.ROOT / 'PACKAGE_MANIFEST.json', {
            'schema': 'mathai2026-overleaf-package-v1',
            'file_sha256': snapshot['copied_files'],
        })
    assert bundle.check_output() == 12
