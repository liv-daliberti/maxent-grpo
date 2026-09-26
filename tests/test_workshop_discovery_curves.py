"""Keep both discovery figures authenticated in self-contained workshop bundles."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'paper/mathai2026/check_submission.py'
REPORT = 'results/modebench_discovery_curves_20260911.json'
TEX = 'results/modebench_discovery_curves_20260911.tex'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    prior = load('prior_workshop_discovery_helpers', REPO / 'tests/test_workshop_prompt_ablation.py')
    publication = load('new_discovery_publication_helpers', REPO / 'tests/test_check_paper_discovery_curves.py')
    data = publication.fixture.__wrapped__(tmp_path / 'science')
    root = tmp_path / 'workshop'
    shutil.copytree(data['paper'], root)
    module = load('workshop_discovery_test', SCRIPT)
    monkeypatch.setattr(module, 'ROOT', root)
    snapshot = prior.populate(module, ['local'])
    snapshot['copied_files'].update({str(p.relative_to(root)): module.digest(p) for p in root.rglob('*') if p.is_file()})
    return module, snapshot, prior, data


def test_exact_two_discovery_figures(bundle):
    module, snapshot, _, _ = bundle
    assert module.discovery_figures(snapshot) == ['fig:discovery-curves-local', 'fig:discovery-correct-budget-local']


@pytest.mark.parametrize('fault', ['omitted_frontier', 'missing_budget_png', 'wrong_plot', 'full_claim'])
def test_discovery_asset_or_scope_tampering_is_rejected(bundle, fault):
    module, snapshot, _, _ = bundle
    if fault == 'omitted_frontier':
        (module.ROOT / 'figures/modebench_discovery_curves_frontier.pdf').write_bytes(b'not collected')
    elif fault == 'missing_budget_png':
        (module.ROOT / 'figures/modebench_discovery_correct_budget_local.png').unlink()
    elif fault == 'wrong_plot':
        name = 'figures/modebench_discovery_correct_budget_local.json'
        path = module.ROOT / name
        metadata = json.loads(path.read_text());metadata['family'] = 'local'
        path.write_text(json.dumps(metadata));snapshot['copied_files'][name] = module.digest(path)
    else:
        path = module.ROOT / REPORT
        report = json.loads(path.read_text());report['experiment_status'] = 'complete'
        path.write_text(json.dumps(report));snapshot['copied_files'][REPORT] = module.digest(path)
    with pytest.raises(SystemExit, match='FAIL:'):
        module.discovery_figures(snapshot)


def test_old_local_ablation_plus_new_discovery_requires_exactly21_figures(bundle, monkeypatch):
    module, snapshot, prior, _ = bundle
    prior.compiled_fixture(module, monkeypatch, ['local'])
    with (module.ROOT / 'main.aux').open('a') as output:
        output.write('\\newlabel{fig:discovery-curves-local}{{20}{8}}\n'
                     '\\newlabel{fig:discovery-correct-budget-local}{{21}{9}}\n')
    assert module.check_output(snapshot) == 14
    with (module.ROOT / 'main.aux').open('a') as output:
        output.write('\\newlabel{fig:unregistered}{{22}{9}}\n')
    with pytest.raises(SystemExit, match='supplementary figures'):
        module.check_output(snapshot)


def test_compiled_discovery_figures_must_stay_in_appendix(bundle, monkeypatch):
    module, snapshot, prior, _ = bundle
    prior.compiled_fixture(module, monkeypatch, ['local'])
    with (module.ROOT / 'main.aux').open('a') as output:
        output.write('\\newlabel{fig:discovery-curves-local}{{20}{4}}\n'
                     '\\newlabel{fig:discovery-correct-budget-local}{{21}{9}}\n')
    with pytest.raises(SystemExit, match='discovery figure must remain'):
        module.check_output(snapshot)


def test_new_assets_validate_without_the_original_experiment_directory(bundle):
    module, snapshot, _, data = bundle
    shutil.copy2(SCRIPT, module.ROOT / SCRIPT.name)
    (module.ROOT / 'snapshot.json').write_text(json.dumps(snapshot))
    shutil.rmtree(data['base'])
    result = subprocess.run([sys.executable, SCRIPT.name, '--check-discovery-assets'],
                            cwd=module.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '2 authenticated discovery figures' in result.stdout


def test_sync_retains_discovery_report_and_both_figure_companions(tmp_path, monkeypatch):
    prior = load('prior_discovery_sync_helpers', REPO / 'tests/test_paper_workshop_asset_sync.py')
    module = prior.sync_tree.__wrapped__(tmp_path, monkeypatch)
    prior.prepare_apply_fixture(module)
    with (module.WORKSHOP / 'appendix.tex').open('a') as output:
        output.write('\\input{' + TEX + '}')
    stems = ['figures/modebench_discovery_curves_local', 'figures/modebench_discovery_correct_budget_local']
    prior.write(module.PAPER / TEX, ''.join('\\includegraphics{' + stem + '}' for stem in stems))
    prior.write(module.PAPER / REPORT, '{}')
    for stem in stems:
        for suffix in ('pdf', 'png', 'json'):
            prior.write(module.PAPER / (stem + '.' + suffix), '{}')
    prior.invoke(module, monkeypatch, tmp_path / 'audit')
    snapshot = json.loads((module.WORKSHOP / 'snapshot.json').read_text())
    for name in [REPORT, TEX, *(stem + '.' + suffix for stem in stems for suffix in ('pdf', 'png', 'json'))]:
        assert snapshot['copied_files'][name] == module.digest(module.PAPER / name)
        assert (module.WORKSHOP / name).read_bytes() == (module.PAPER / name).read_bytes()
