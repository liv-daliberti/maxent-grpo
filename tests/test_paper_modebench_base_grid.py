"""Publication cannot turn incomplete or stale frozen-base evidence into points.

All fabricated fixtures stay under pytest temporary directories. Production
receipt authentication is independently exercised by the evaluator test suite.
"""
from copy import deepcopy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_base_grid', ROOT / 'ops/plot_paper_modebench_base_grid.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
AUDIT = {'prompts': 128, 'draws_per_prompt': 4, 'distinct_request_blocks': 512,
         'distinct_child_seeds': 4096, 'seed_schedule_sha256': 'fixture'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    return path


def prompts():
    def group(keys):
        return {'attempts': [{'verified': key is not None, 'canonical_key': key} for key in keys]}
    # Pooling these 32 attempts incorrectly yields 11 modes and pass@32=1.
    draws = [group(['a', 'a', 'b', None, None, None, None, None]), group([None] * 8),
             group(['c'] * 8), group(list('defghijk'))]
    return [{'row_index': index, 'row_sha256': f'{index:064x}', 'draws': draws}
            for index in range(128)]


def inputs(base, *, cells=(), admitted_levels=m.LEVELS):
    registry_path = base / 'registry.json'
    validator = base / 'artifacts/code/ops/evaluate_modebench_base_grid.py'
    validator.parent.mkdir(parents=True, exist_ok=True)
    validator.write_text('# Test-only placeholder, validator explicitly mocked.\n')
    models = {}
    for label in m.MODELS:
        identity = {'label': label, 'path': str(base / 'models' / label)}
        models[label] = {'label': label, 'path': identity['path'], 'revision': label + '-revision',
                         'model_repository': label + '-repository', 'identity': identity,
                         'identity_sha256': m.canonical_sha(identity)}
    datasets = [{'level': level, 'domain': domain, 'status': 'admitted', 'split': 'eval', 'rows': 128}
                for level in admitted_levels for domain in m.DOMAINS]
    write(registry_path, {'schema': m.REGISTRY_SCHEMA, 'domains': list(m.DOMAINS),
                         'levels': list(m.LEVELS), 'model_labels': list(m.MODELS),
                         'models': models, 'datasets': datasets})
    paths = []
    for label, level, domain in cells:
        model = models[label]
        identity = {'dataset_binding': {'source_manifest_path': str(registry_path),
                                        'source_manifest_sha256': m.file_sha(registry_path),
                                        **next(entry for entry in datasets if entry['level'] == level and entry['domain'] == domain)},
                    'model': {**model['identity'], 'revision': model['revision'],
                              'repository': model['model_repository']},
                    'code_sha256': {'fixture.py': 'frozen'}, 'interface': {'max_tokens': 192},
                    'runtime': {'dtype': 'float16'}, 'sampling_engine': {'engine': 'V0'}}
        receipt = {'schema': m.RECEIPT_SCHEMA, 'status': 'complete', 'model_label': label,
                   'level': level, 'domain': domain, 'identity': identity,
                   'identity_sha256': m.canonical_sha(identity), 'prompt_results': prompts(),
                   'metrics': {'pass8': .75, 'distinct8': 2.75}}
        paths.append(write(base / 'receipts' / f'{label}_{level}_{domain}.json', receipt))
    plan = write(base / 'collection_plan.json', {'cells': [{'command': [sys.executable, '-B', str(validator)]}]})
    source = write(base / 'figure_source.json', m.make_source(registry_path, paths, validator, collection_plan_path=plan))
    return source


@pytest.fixture(scope='module')
def full_source(tmp_path_factory):
    return inputs(tmp_path_factory.mktemp('full_base_grid'), cells=sorted(m.EXPECTED))


@pytest.fixture
def mocked_validator(monkeypatch):
    calls = []
    def validate(path, entries, runtime):
        calls.append((path, entries, runtime))
        return [deepcopy(AUDIT) for _ in entries]
    monkeypatch.setattr(m, 'validate_receipts', validate)
    return calls


def test_reconstruction_retains_k8_groups_and_zero_success_groups():
    receipt = {'prompt_results': prompts(), 'metrics': {'pass8': .75, 'distinct8': 2.75}}
    metrics, records = m.reconstruct_metrics(receipt)
    assert metrics == {'pass8': .75, 'distinct8': 2.75}
    assert records[0]['groups'] == [{'pass8': 1., 'distinct8': 2}, {'pass8': 0., 'distinct8': 0},
                                    {'pass8': 1., 'distinct8': 1}, {'pass8': 1., 'distinct8': 8}]
    assert metrics['pass8'] != 1 and metrics['distinct8'] != 11


@pytest.mark.parametrize('change', ['prompt', 'draw', 'attempt', 'canonical', 'summary'])
def test_reconstruction_rejects_incomplete_or_inconsistent_evidence(change):
    receipt = deepcopy({'prompt_results': prompts(), 'metrics': {'pass8': .75, 'distinct8': 2.75}})
    if change == 'prompt': receipt['prompt_results'].pop()
    elif change == 'draw': receipt['prompt_results'][0]['draws'].pop()
    elif change == 'attempt': receipt['prompt_results'][0]['draws'][0]['attempts'].pop()
    elif change == 'canonical': receipt['prompt_results'][0]['draws'][0]['attempts'][0]['canonical_key'] = None
    else: receipt['metrics']['pass8'] = 1.
    with pytest.raises(ValueError): m.reconstruct_metrics(receipt)


def test_full_grid_has_every_model_at_every_level_with_exact_coordinates(full_source, mocked_validator):
    record = m.build_record(full_source)
    assert record['status'] == 'complete'
    assert {m.key(point) for point in record['points']} == m.EXPECTED
    assert record['complete_cells'] == 100 and len(mocked_validator[0][1]) == 100
    assert all(point['metrics'] == {'pass8': .75, 'distinct8': 2.75} for point in record['points'])
    assert record['sampling']['complete_grid_samples'] == 409600
    assert record['metrics']['groups_pooled_into_32_draws'] is False
    fig = m.build_figure(record)
    assert len(fig.axes) == 5 and len(fig.legends) == 2
    assert len(fig.legends[0].texts) == 5 and len(fig.legends[1].texts) == 4
    for axis in fig.axes:
        assert axis.get_xlim() == (0, 1) and axis.get_ylim() == (0, 8)
        assert axis.get_xlabel() == 'pass@8'
        assert len(axis.collections) == 20 and not axis.lines
        assert all(collection.get_offsets().tolist() == [[.75, 2.75]] for collection in axis.collections)
    assert fig.axes[0].get_ylabel() == 'distinct@8'
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_publication_fails_closed_before_any_output_on_missing_cells(tmp_path):
    source = inputs(tmp_path, admitted_levels=m.LEVELS[:3])
    destination = tmp_path / 'paper/figures/grid'
    with pytest.raises(ValueError, match='100 complete authenticated cells; 100 are missing'):
        m.render(source, destination)
    assert not destination.parent.exists()


def test_partial_preview_exposes_missing_coverage_and_only_writes_artifacts(tmp_path, monkeypatch, mocked_validator):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    source = inputs(tmp_path, admitted_levels=m.LEVELS[:3])
    record = m.build_record(source, allow_partial=True)
    assert record['complete_cells'] == 0 and len(record['missing_cells']) == 100
    assert sum(cell['reason'] == 'awaiting_dataset_admission' for cell in record['missing_cells']) == 40
    assert sum(cell['reason'] == 'awaiting_complete_measurement' for cell in record['missing_cells']) == 60
    with pytest.raises(ValueError, match='below artifacts'):
        m.render(source, tmp_path / 'paper/figures/preview', allow_partial=True)
    rendered = m.render(source, tmp_path / 'artifacts/preview', allow_partial=True)
    assert rendered['status'] == 'partial_preview'
    fig = m.build_figure(rendered)
    assert any('PARTIAL PREVIEW' in text.get_text() and '100 missing' in text.get_text() for text in fig.texts)
    assert all(any(text.get_text() == '0/20 complete' for text in axis.texts) for axis in fig.axes)
    assert all(not axis.collections for axis in fig.axes)
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.parametrize('change', ['duplicate', 'unregistered', 'receipt_hash', 'registry_hash', 'validator_hash',
                                   'missing_binding', 'identity', 'summary', 'different_registry', 'model'])
def test_present_cells_need_authentic_bound_receipts(tmp_path, mocked_validator, change):
    source = inputs(tmp_path, cells=[('05b', 'level1', 'countdown')])
    manifest = json.loads(source.read_text())
    entry = manifest['receipts'][0]
    if change == 'duplicate': manifest['receipts'].append(deepcopy(entry))
    elif change == 'unregistered': entry['level'] = 'development'
    elif change == 'receipt_hash': entry['sha256'] = '0' * 64
    elif change == 'registry_hash': manifest['registry']['sha256'] = '0' * 64
    elif change == 'validator_hash': manifest['validator']['sha256'] = '0' * 64
    elif change == 'missing_binding': del entry['sha256']
    else:
        path = Path(entry['path']); receipt = json.loads(path.read_text())
        if change == 'identity': receipt['level'] = 'level2'
        elif change == 'summary': receipt['metrics']['distinct8'] = 8
        elif change == 'different_registry': receipt['identity']['dataset_binding']['source_manifest_sha256'] = '0' * 64
        else: receipt['identity']['model']['revision'] = 'wrong'
        write(path, receipt); entry['sha256'] = m.file_sha(path)
    write(source, manifest)
    with pytest.raises(ValueError): m.build_record(source, allow_partial=True)


def test_frozen_validator_failure_cannot_be_bypassed(tmp_path, monkeypatch):
    source = inputs(tmp_path, cells=[('05b', 'level1', 'countdown')])
    def reject(*_): raise ValueError('frozen receipt code or RNG evidence changed')
    monkeypatch.setattr(m, 'validate_receipts', reject)
    with pytest.raises(ValueError, match='frozen receipt code'):
        m.build_record(source, allow_partial=True)


def test_incomplete_independence_audit_cannot_be_plotted(tmp_path, monkeypatch):
    source = inputs(tmp_path, cells=[('05b', 'level1', 'countdown')])
    monkeypatch.setattr(m, 'validate_receipts', lambda *_: [{**AUDIT, 'distinct_child_seeds': 4095}])
    with pytest.raises(ValueError, match='independent K=8 x 4'):
        m.build_record(source, allow_partial=True)


def test_frozen_validator_runs_in_isolated_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    source = inputs(tmp_path, cells=[('05b', 'level1', 'countdown')])
    manifest = json.loads(source.read_text())
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stdout=json.dumps([AUDIT]), stderr='')
    monkeypatch.setattr(m.subprocess, 'run', run)
    monkeypatch.setattr(m, 'authenticate_runtime', lambda *_: Path(sys.executable))
    audits = m.validate_receipts(m.authenticate(manifest['validator'], 'test validator'), manifest['receipts'], manifest['validator_runtime'])
    assert audits == [AUDIT]
    assert calls[0][0][1:3] == ['-I', '-c']
    assert 'evaluator.validate_seed_receipt(receipt)' in calls[0][0][3]
    assert json.loads(calls[0][1]['input'])[0]['sha256'] == manifest['receipts'][0]['sha256']


def test_caption_describes_independent_groups_and_all_levels():
    caption = m.caption_snippet()
    assert 'STAGED' in caption and '100 cells' in caption
    assert 'Levels 1--5' in caption and 'four independent groups of eight' in caption
    assert '0.5B, 3B, 7B, and 14B' in caption
    assert 'including groups' in caption and 'no verified answer' in caption


def test_immutable_registry_extension_reuses_old_receipts_and_datasets(tmp_path, mocked_validator):
    cells = [('05b', 'level1', 'countdown'), ('3b', 'level1', 'countdown'), ('05b', 'level4', 'countdown')]
    source = inputs(tmp_path, cells=cells)
    manifest = json.loads(source.read_text())
    primary = Path(manifest['registry']['path'])
    full = json.loads(primary.read_text())
    extension = write(tmp_path / 'later_registry.json', full)
    earlier = deepcopy(full)
    earlier['datasets'] = [entry for entry in earlier['datasets'] if entry['level'] in m.LEVELS[:3]]
    write(primary, earlier)
    manifest['registry'] = m.binding(primary)
    manifest['registry_extensions'] = [m.binding(extension)]
    old_receipt_bytes = None
    for entry in manifest['receipts']:
        path = Path(entry['path']); receipt = json.loads(path.read_text())
        is_old = entry['model_label'] == '05b' and entry['level'] == 'level1'
        registry = primary if is_old else extension
        receipt['identity']['dataset_binding'].update(source_manifest_path=str(registry),
                                                       source_manifest_sha256=m.file_sha(registry))
        receipt['identity_sha256'] = m.canonical_sha(receipt['identity'])
        write(path, receipt); entry['sha256'] = m.file_sha(path)
        if is_old: old_receipt_bytes = path, path.read_bytes()
    write(source, manifest)
    record = m.build_record(source, allow_partial=True)
    assert record['complete_cells'] == 3
    assert old_receipt_bytes[0].read_bytes() == old_receipt_bytes[1]
    assert {point['dataset_registry']['sha256'] for point in record['points']} == {
        m.file_sha(primary), m.file_sha(extension)}
    assert len(record['datasets']) == 2  # Both scales share the same immutable L1 dataset.
    assert not any(item['reason'] == 'awaiting_dataset_admission' for item in record['missing_cells'])


@pytest.mark.parametrize('change', ['dataset', 'model', 'hash'])
def test_registry_extension_cannot_relabel_earlier_evidence(tmp_path, mocked_validator, change):
    source = inputs(tmp_path, admitted_levels=m.LEVELS[:3])
    manifest = json.loads(source.read_text())
    extension = json.loads(Path(manifest['registry']['path']).read_text())
    if change == 'dataset': extension['datasets'][0]['dataset_sha256'] = 'changed'
    elif change == 'model': extension['models']['05b']['revision'] = 'changed'
    path = write(tmp_path / 'invalid_extension.json', extension)
    manifest['registry_extensions'] = [m.binding(path)]
    if change == 'hash': manifest['registry_extensions'][0]['sha256'] = '0' * 64
    write(source, manifest)
    with pytest.raises(ValueError): m.build_record(source, allow_partial=True)


@pytest.mark.parametrize('change', ['binary_hash', 'version', 'stdlib_hash', 'plan_path', 'missing_runtime'])
def test_changed_collection_interpreter_cannot_authenticate_receipts(tmp_path, change):
    source = inputs(tmp_path)
    manifest = json.loads(source.read_text())
    runtime = deepcopy(manifest['validator_runtime'])
    if change == 'binary_hash': runtime['interpreter']['sha256'] = '0' * 64
    elif change == 'version': runtime['runtime_metadata']['version'] = 'other Python'
    elif change == 'stdlib_hash': runtime['runtime_metadata']['statistics']['sha256'] = '0' * 64
    elif change == 'plan_path': runtime['interpreter']['path'] = '/other/python'
    else: runtime = None
    with pytest.raises(ValueError):
        m.authenticate_runtime(runtime, m.authenticate(manifest['validator'], 'validator'))


def test_exact_collection_interpreter_is_accepted(tmp_path):
    source = inputs(tmp_path)
    manifest = json.loads(source.read_text())
    path = m.authenticate_runtime(manifest['validator_runtime'], m.authenticate(manifest['validator'], 'validator'))
    assert str(path) == sys.executable
    assert manifest['validator_runtime']['runtime_metadata']['version'] == sys.version
