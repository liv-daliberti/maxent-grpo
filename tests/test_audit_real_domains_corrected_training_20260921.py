import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import audit_real_domains_corrected_training_20260921 as audit


@pytest.fixture
def frozen(tmp_path):
    source = Path(audit.__file__).with_name(audit.TRAINER_FILENAME)
    assert audit.digest(source) == audit.TRAINER_SHA256
    root = tmp_path / 'run'
    (root / 'training').mkdir(parents=True)
    runner = root / 'bundle/ops' / audit.TRAINER_FILENAME
    runner.parent.mkdir(parents=True)
    runner.write_bytes(source.read_bytes())
    adapter = root / 'bundle/ops/adapter.py'
    adapter.write_text('# fixed adapter fixture\n')
    productions = {}
    for module in audit.PRODUCTION_MODULES:
        production = root / 'bundle/src' / (module.replace('.', '/') + '.py')
        production.parent.mkdir(parents=True, exist_ok=True)
        production.write_text('# fixed production fixture\n')
        productions[module] = production
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}\n')
    raw = {'model': str(model), 'adapter_module': 'adapter', 'train_microbatch_size': 1}
    (root / 'config.json').write_text(json.dumps(raw))
    defaults = next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULTS' for t in n.targets))
    files = [{'snapshot': str(p), 'sha256': audit.digest(p)} for p in (runner, adapter, *productions.values())]
    launch = {'schema': 'real-domains-frozen-job-20260921-v1', 'config_sha256': audit.digest(root / 'config.json'), 'files': files, 'request': {'entrypoint': audit.TRAINER_FILENAME, 'arm': 'maxrl'}}
    (root / 'identity.json').write_text(json.dumps(launch))
    receipt = {'arm': 'maxrl', 'schema': audit.TRAINER_SCHEMA, 'runner_sha256': audit.TRAINER_SHA256,
               'config': {**defaults, **raw}, 'objective': {'behavior_scoring': audit.BEHAVIOR_CONTRACT},
               'adapter_module_sha256': audit.digest(adapter),
               'production_source_sha256': {module: audit.digest(path) for module, path in productions.items()},
               'model_config_sha256': audit.digest(model / 'config.json')}
    helper = SimpleNamespace(read_json=lambda p: json.loads(Path(p).read_text()))
    return root, runner, launch, receipt, helper


def check(frozen):
    root, _, _, receipt, helper = frozen
    return audit.audit_corrected_frozen_sources(root / 'training', receipt, training=True, helper=helper)


def test_known_v2_source_and_seal_pass(frozen):
    assert check(frozen) == 'pass'


def test_unknown_version_rejected(frozen):
    frozen[3]['runner_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='unknown or altered'):
        check(frozen)


def test_altered_default_rejected_even_when_resealed(frozen):
    root, runner, launch, receipt, _ = frozen
    source = runner.read_text()
    assert '"learning_rate": 1e-5' in source
    runner.write_text(source.replace('"learning_rate": 1e-5', '"learning_rate": 2e-5'))
    changed = audit.digest(runner)
    receipt['runner_sha256'] = changed
    launch['files'][0]['sha256'] = changed
    (root / 'identity.json').write_text(json.dumps(launch))
    with pytest.raises(ValueError, match='unknown or altered'):
        check(frozen)


@pytest.mark.parametrize('target', ['schema', 'contract', 'microbatch', 'production', 'resolved', 'launcher', 'entrypoint', 'arm', 'partial_production'])
def test_source_and_scoring_contract_drift_rejected(frozen, target):
    root, _, launch, receipt, _ = frozen
    if target == 'schema':
        receipt['schema'] = 'historical-v1'
    elif target == 'contract':
        receipt['objective']['behavior_scoring'] = 'different'
    elif target == 'microbatch':
        raw = json.loads((root / 'config.json').read_text())
        raw['train_microbatch_size'] = receipt['config']['train_microbatch_size'] = 2
        (root / 'config.json').write_text(json.dumps(raw))
        launch['config_sha256'] = audit.digest(root / 'config.json')
        (root / 'identity.json').write_text(json.dumps(launch))
    elif target == 'production':
        (root / 'bundle/src/oat_drgrpo/learner/grpo.py').write_text('# changed\n')
    elif target == 'resolved':
        receipt['config']['learning_rate'] *= 2
    elif target == 'partial_production':
        receipt['production_source_sha256'].pop('oat_drgrpo.learner.grpo')
    elif target in ('entrypoint', 'arm'):
        launch['request'][target] = 'different'
        (root / 'identity.json').write_text(json.dumps(launch))
    else:
        (root / 'identity.json').unlink()
    with pytest.raises(ValueError):
        check(frozen)


def numerical_rows():
    return [{'completed_updates': i, **{key: 0.0 for key in audit.ZERO_METRICS}} for i in (1, 2)]


def test_exact_zero_numerical_contract_passes():
    result = audit.audit_zero_ratio_contract(numerical_rows(), 2)
    assert result['status'] == 'pass' and result['updates_checked'] == 2


@pytest.mark.parametrize('key', audit.ZERO_METRICS)
@pytest.mark.parametrize('value', [1e-12, -1e-12, float('nan'), float('inf'), None, False])
def test_nonzero_missing_or_nonfinite_diagnostics_fail(key, value):
    rows = numerical_rows()
    rows[0][key] = value
    with pytest.raises(ValueError, match='numerical contract failed'):
        audit.audit_zero_ratio_contract(rows, 2)


def test_missing_update_diagnostics_fail():
    with pytest.raises(ValueError, match='incomplete or reordered'):
        audit.audit_zero_ratio_contract(numerical_rows()[1:], 2)


def test_original_helper_is_isolated_and_pinned():
    original = __import__('summarize_real_domains_pilot_20260921')
    isolated = audit.load_pinned_helper()
    assert isolated is not original
    assert isolated.audit_pair is not original.audit_pair
    assert audit.digest(Path(audit.__file__).with_name(audit.HELPER_FILENAME)) == audit.HELPER_SHA256


def test_modified_helper_is_refused(tmp_path, monkeypatch):
    (tmp_path / audit.HELPER_FILENAME).write_text('# modified helper\n')
    monkeypatch.setattr(audit, '__file__', str(tmp_path / 'wrapper.py'))
    with pytest.raises(ValueError, match='pinned version'):
        audit.load_pinned_helper()
