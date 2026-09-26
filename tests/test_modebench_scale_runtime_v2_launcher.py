"""Runtime-v2 admission, scientific-invariance and exactly-once submission tests."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('scale_launch_runtime_v2_test_module', ROOT / 'ops/exp_scaling/launch_modebench_scale_runtime_v2.py')
ctl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ctl)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_model(path, label):
    width, layers = {'7b': (3584, 28), '14b': (5120, 48)}[label]
    write(path / 'config.json', {'architectures': ['Qwen2ForCausalLM'],
                                'hidden_size': width, 'num_hidden_layers': layers})
    write(path / 'model.safetensors.index.json', {'weight_map': {'weight': 'model-1.safetensors'}})
    write(path / 'tokenizer_config.json', {})
    write(path / 'tokenizer.json', {})
    (path / 'model-1.safetensors').write_bytes(b'fake-weights')
    return path


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, 'PYTHON', Path(sys.executable))
    data = tmp_path / 'data with spaces'
    levels = ('level4', 'level5')
    for item in ctl.discover_inputs(data, levels, 'dev'):
        write(Path(item['rows_jsonl']), {'problem': str(item), 'answer': '{}'})
    models = {label: make_model(tmp_path / ('model ' + label), label) for label in ('7b', '14b')}
    write(data / 'protocol.json', {'schema': 'modebench_scale_protocol_v1', 'files_sha256': {},
                                 'models': {label: ctl.evaluator().model_identity(path, label) for label, path in models.items()},
                                 'draw_labels': {level: {phase: ctl.draw_labels(level, phase)
                                                        for phase in ('dev', 'eval')} for level in levels}})
    for level in levels:
        for domain in ctl.DOMAINS:
            directory = data / level / 'pools' / domain
            tiers = {str(tier): {'rows': 1, 'rows_sha256': ctl.evaluator().sha([ctl.read(directory / f'difficulty_{tier}.jsonl')])}
                     for tier in range(4)}
            write(directory / 'identity.json', {'schema': 'modebench_scale_development_pools_v1',
                'status': 'verified_candidates_pending_model_calibration', 'level': level, 'domain': domain,
                'protocol_sha256': ctl.digest(data / 'protocol.json'), 'tiers': tiers})
    root = tmp_path / 'launch `literal` space'
    return data, models, root


def prepared(campaign):
    data, models, root = campaign
    plan = ctl.prepare(None, root, models, data_root=data)
    return root / 'plan.json', plan


def test_prepare_groups_tiers_and_pins_actual_protocol_paths(campaign):
    path, plan = prepared(campaign)
    data, _, _ = campaign
    assert len(plan['cells']) == 10
    assert '--array=0-9%1' in plan['submit_command']
    assert '--gres=gpu:a5000:2' in plan['submit_command']
    assert '--nodelist=node105' in plan['submit_command']
    assert '--mem=60G' in plan['submit_command']
    assert plan['rng_admission']['distinct_request_blocks'] == 40 * 4
    assert plan['rng_admission']['distinct_child_seeds'] == 40 * 4 * 8
    assert len(plan['rng_admission']['task_seed_schedule_sha256']) == 40
    for cell in plan['cells']:
        tasks = ctl.read(cell['tasks'])
        assert len(tasks) == 4
        for tier, task in enumerate(tasks):
            assert task['seeds'] == ctl.draw_labels(task['level'], 'dev')
            assert task['output'] == str(data / task['level'] / 'results/development' / task['domain'] / f'difficulty_{tier}.json')
        assert shlex.split(cell['shell_command']) == cell['command']
    assert shlex.split(plan['submit_shell_command']) == plan['submit_command']
    assert ctl.verify(path, fresh=True) == plan
    assert not (path.parent / 'submission_intent.json').exists()
    subprocess.run(['bash', '-n', str(path.parent / 'worker.slurm')], check=True)


def test_missing_shard_fails_before_campaign_claim(campaign):
    data, models, root = campaign
    (models['14b'] / 'model-1.safetensors').unlink()
    with pytest.raises(ValueError, match='shard missing'):
        ctl.prepare(None, root, models, data_root=data)
    assert not root.exists()


def test_level4_only_needs_local_7b(campaign):
    data, models, root = campaign
    plan = ctl.prepare(None, root, {'7b': models['7b']}, data_root=data, levels=('level4',))
    assert len(plan['cells']) == 5 and set(plan['models']) == {'7b'}


def test_missing_pool_and_duplicate_cells_are_rejected(campaign):
    data, _, _ = campaign
    inputs = ctl.discover_inputs(data, ('level4',), 'dev')
    with pytest.raises(ValueError, match='every registered tier'):
        ctl.validate_inputs(inputs[:-1], 'dev')
    with pytest.raises(ValueError, match='duplicate input'):
        ctl.validate_inputs(inputs + inputs[:1], 'dev')


@pytest.mark.parametrize('kind', ['input', 'model', 'plan'])
def test_changed_input_model_or_plan_cannot_submit(campaign, kind):
    path, plan = prepared(campaign)
    if kind == 'input':
        changed = Path(ctl.read(plan['cells'][0]['tasks'])[0]['rows_jsonl'])
    elif kind == 'model':
        changed = Path(plan['models']['7b']['path']) / 'model-1.safetensors'
    else:
        changed = path
    changed.write_text(changed.read_text() + ' ')
    calls = []
    with pytest.raises(ValueError):
        ctl.submit(path, runner=lambda *a, **k: calls.append(a))
    assert calls == [] and not (path.parent / 'submission_intent.json').exists()


@pytest.mark.parametrize('stdout,returncode', [('', 0), ('Submitted batch job 12', 0), ('12\n13', 0), ('12', 1)])
def test_ambiguous_submission_never_retries(campaign, stdout, returncode):
    path, plan = prepared(campaign)
    calls = []
    def runner(argv, **kwargs):
        assert ctl.read(path.parent / 'submission_intent.json')['command'] == argv
        calls.append(argv)
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr='diagnostic')
    with pytest.raises(RuntimeError, match='ambiguous'):
        ctl.submit(path, runner=runner)
    with pytest.raises(ValueError, match='already attempted'):
        ctl.submit(path, runner=runner)
    assert len(calls) == 1
    assert (path.parent / 'submission_ambiguous.json').exists()


def test_timeout_keeps_intent_without_retry(campaign):
    path, _ = prepared(campaign)
    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 60)
    with pytest.raises(subprocess.TimeoutExpired):
        ctl.submit(path, runner=runner)
    with pytest.raises(ValueError, match='already attempted'):
        ctl.submit(path, runner=runner)


def test_success_records_all_array_cell_ids_once(campaign):
    path, _ = prepared(campaign)
    runner = lambda *a, **k: SimpleNamespace(returncode=0, stdout='1234;cluster\n', stderr='')
    result = ctl.submit(path, runner=runner)
    assert result['array_job_id'] == 1234
    assert len(result['cells']) == 10
    with pytest.raises(ValueError, match='already attempted'):
        ctl.submit(path, runner=runner)


def test_confirmation_requires_frozen_recipe_gate_before_claim(campaign, monkeypatch):
    data, models, root = campaign
    for item in ctl.discover_inputs(data, ('level4', 'level5'), 'eval'):
        write(Path(item['rows_jsonl']), {'problem': str(item), 'answer': '{}'})
    def reject(*args):
        raise ValueError('passing reproducible recipe required')
    monkeypatch.setattr(ctl, 'confirmation_gate', reject)
    with pytest.raises(ValueError, match='passing reproducible recipe'):
        ctl.prepare(None, root, models, data_root=data, phase='eval')
    assert not root.exists()


def test_confirmation_rows_cannot_be_tier_candidates(campaign):
    data, _, _ = campaign
    with pytest.raises(ValueError, match='confirmation must omit tier'):
        ctl.validate_inputs(ctl.discover_inputs(data, ('level4',), 'dev'), 'eval')


def test_same_size_different_checkpoint_rejected_before_launch(campaign, tmp_path):
    data, models, root = campaign
    models['7b'] = make_model(tmp_path / 'unregistered alternative 7b', '7b')
    with pytest.raises(ValueError, match='registered protocol'):
        ctl.prepare(None, root, models, data_root=data)
    assert not root.exists()


def test_confirmation_filename_and_draw_labels_match_auditor(campaign, monkeypatch):
    data, models, root = campaign
    for item in ctl.discover_inputs(data, ('level4', 'level5'), 'eval'):
        write(Path(item['rows_jsonl']), {'problem': str(item), 'answer': '{}'})
    monkeypatch.setattr(ctl, 'confirmation_gate', lambda *args: {})
    plan = ctl.prepare(None, root, models, data_root=data, phase='eval')
    for cell in plan['cells']:
        tasks = ctl.read(cell['tasks'])
        assert len(tasks) == 1
        task = tasks[0]
        assert task['output'] == str(data / task['level'] / 'results/confirmation' / (task['domain'] + '.json'))
        assert task['seeds'] == ctl.draw_labels(task['level'], 'eval')
        assert '--confirm-eval' in cell['command']


def test_confirmation_gate_authenticates_recipe_inputs_and_frozen_rows(tmp_path, monkeypatch):
    level, domain = 'level4', 'countdown'
    base = tmp_path / 'data'
    rows = [{'problem': str(i), 'answer': '{}'} for i in range(128)]
    source = base / level / 'dataset' / domain / 'eval.jsonl'
    source.parent.mkdir(parents=True)
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write(base / 'protocol.json', {})
    evidence = base / 'receipt.json'
    write(evidence, {'status': 'complete'})
    recipe = {'schema': 'modebench_scale_development_recipe_v1', 'development_fit_pass': True,
              'input_sha256': {str(evidence): ctl.digest(evidence)}}
    recipe_path = base / level / 'recipes' / (domain + '.json')
    write(recipe_path, recipe)
    identity = {'schema': 'modebench_scale_frozen_domain_v1', 'status': 'frozen_pending_heldout_confirmation',
                'level': level, 'domain': domain, 'recipe_sha256': ctl.digest(recipe_path),
                'protocol_sha256': ctl.digest(base / 'protocol.json'),
                'splits': {'eval': {'rows': 128, 'rows_sha256': ctl.evaluator().sha(rows)}}}
    write(source.parent / 'identity.json', identity)
    monkeypatch.setitem(sys.modules, 'fit_modebench_scale', SimpleNamespace(fit_domain=lambda *a, **k: recipe))
    monkeypatch.setitem(sys.modules, 'materialize_modebench_scale', SimpleNamespace(authenticate=lambda *a: {}))
    inputs = [{'level': level, 'domain': domain, 'rows_jsonl': str(source)}]
    pins = ctl.confirmation_gate(base, inputs)
    assert pins[str(evidence)] == ctl.digest(evidence)
    evidence.write_text('{}')
    with pytest.raises(ValueError, match='development evidence changed'):
        ctl.confirmation_gate(base, inputs)
    write(evidence, {'status': 'complete'})
    source.write_text(source.read_text().replace('"problem": "0"', '"problem": "changed"'))
    with pytest.raises(ValueError, match='frozen eval rows changed'):
        ctl.confirmation_gate(base, inputs)


@pytest.mark.parametrize('field,value', [('schema', 'wrong'), ('level', 'level5'),
    ('domain', 'graph_coloring'), ('protocol_sha256', '0'*64), ('status', 'unverified')])
def test_wrong_pool_identity_rejected_before_campaign_claim(campaign, field, value):
    data, models, root = campaign
    identity_path = data / 'level4/pools/countdown/identity.json'
    identity = ctl.read(identity_path)
    identity[field] = value
    write(identity_path, identity)
    with pytest.raises(ValueError, match='development pool identity differs'):
        ctl.prepare(None, root, models, data_root=data)
    assert not root.exists()


@pytest.mark.parametrize('field,value', [('rows', 2), ('rows', True), ('rows_sha256', '0'*64)])
def test_pool_tier_count_and_hash_must_match_jsonl(campaign, field, value):
    data, models, root = campaign
    identity_path = data / 'level4/pools/countdown/identity.json'
    identity = ctl.read(identity_path)
    identity['tiers']['0'][field] = value
    write(identity_path, identity)
    with pytest.raises(ValueError, match='pool tier row count/hash'):
        ctl.prepare(None, root, models, data_root=data)
    assert not root.exists()


def test_pool_manifest_pinned_through_worker_verification(campaign):
    path, plan = prepared(campaign)
    data, _, _ = campaign
    identity_path = data / 'level4/pools/countdown/identity.json'
    assert plan['immutable_inputs_sha256'][str(identity_path)] == ctl.digest(identity_path)
    identity_path.write_text(identity_path.read_text() + ' ')
    with pytest.raises(ValueError, match='prepared input changed'):
        ctl.verify(path)
    assert not (path.parent / 'submission_intent.json').exists()


def test_repeated_prompt_across_tiers_rejected_as_rng_collision(campaign):
    data, models, root = campaign
    directory = data / 'level4/pools/countdown'
    row = ctl.read(directory / 'difficulty_0.jsonl')
    write(directory / 'difficulty_1.jsonl', row)
    identity_path = directory / 'identity.json'
    identity = ctl.read(identity_path)
    identity['tiers']['1']['rows_sha256'] = ctl.evaluator().sha([row])
    write(identity_path, identity)
    with pytest.raises(ValueError, match='RNG block collision across calibration tasks'):
        ctl.prepare(None, root, models, data_root=data)
    assert not root.exists()


@pytest.mark.parametrize('dependencies', [[0], [-1], [True], [1.0], ['42'], [42, 42]])
def test_invalid_array_dependencies_rejected_before_campaign_claim(campaign, dependencies):
    data, models, root = campaign
    with pytest.raises(ValueError, match='distinct positive integer'):
        ctl.prepare(None, root, models, data_root=data, dependency_ids=dependencies)
    assert not root.exists()


def test_afterany_array_dependencies_are_sealed_and_preserve_four_gpu_cap(campaign):
    data, models, root = campaign
    plan = ctl.prepare(None, root, models, data_root=data, dependency_ids=[200, 100])
    assert plan['dependency_ids'] == [100, 200]
    assert '--dependency=afterany:100:200' in plan['submit_command']
    assert '--array=0-9%1' in plan['submit_command']
    assert shlex.split(plan['submit_shell_command']) == plan['submit_command']
    assert ctl.verify(root / 'plan.json') == plan
    changed = ctl.read(root / 'plan.json')
    changed['dependency_ids'] = []
    write(root / 'plan.json', changed)
    with pytest.raises(ValueError, match='plan changed'):
        ctl.verify(root / 'plan.json')


@pytest.mark.parametrize('concurrency', [0, 3, 4, True, 1.0])
def test_runtime_v2_rejects_more_than_four_gpus_before_claim(campaign, concurrency):
    data, models, root = campaign
    with pytest.raises(ValueError, match='concurrency must be 1..2'):
        ctl.prepare(None, root, models, data_root=data, concurrency=concurrency)
    assert not root.exists()


def test_runtime_v2_allows_two_tp2_cells_but_defaults_one(campaign):
    data, models, root = campaign
    plan = ctl.prepare(None, root, models, data_root=data, concurrency=2)
    assert '--array=0-9%2' in plan['submit_command']
    assert plan['concurrency'] * plan['hardware']['tensor_parallel_size'] == 4
    assert ctl.verify(root / 'plan.json') == plan


def test_runtime_v2_profile_is_passed_to_evaluator_and_origin_is_pinned(campaign):
    path, plan = prepared(campaign)
    assert plan['origin_source'] == {'path': str(ctl.ORIGIN_SOURCE), 'sha256': ctl.ORIGIN_SHA256}
    assert ctl.digest(ctl.ORIGIN_SOURCE) == ctl.ORIGIN_SHA256
    assert plan['immutable_inputs_sha256'][str(ctl.ORIGIN_SOURCE)] == ctl.ORIGIN_SHA256
    assert plan['runtime_profile'] == ctl.RUNTIME_PROFILE
    for cell in plan['cells']:
        args = ctl.evaluator().parse_args(cell['command'][2:])
        assert args.tensor_parallel_size == 2
        assert args.swap_space == 4.0
        assert args.gpu_memory_utilization == .82
        assert args.enable_prefix_caching is True
        assert args.max_model_len == ctl.evaluator().frozen_interface(cell['domain'])['max_model_len']
        assert all(task['batch_size'] == 2 for task in ctl.read(cell['tasks']))
    assert ctl.verify(path) == plan


def test_runtime_v2_preserves_all_original_sources_outputs_and_rng(campaign, monkeypatch):
    spec = importlib.util.spec_from_file_location('sealed_original_scale_launcher', ctl.ORIGIN_SOURCE)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    monkeypatch.setattr(original, 'PYTHON', Path(sys.executable))
    data, models, root = campaign
    old = original.prepare(None, root.parent / 'original-plan', models, data_root=data)
    new = ctl.prepare(None, root, models, data_root=data)
    assert old['rng_admission'] == new['rng_admission']
    assert old['models'] == new['models']
    for old_cell, new_cell in zip(old['cells'], new['cells']):
        assert old_cell['id'] == new_cell['id']
        for old_task, new_task in zip(original.read(old_cell['tasks']), ctl.read(new_cell['tasks'])):
            assert {k: v for k, v in old_task.items() if k != 'batch_size'} == {
                k: v for k, v in new_task.items() if k != 'batch_size'}
            assert (old_task['batch_size'], new_task['batch_size']) == (8, 2)


def test_runtime_v2_fresh_submission_rejects_any_original_partial_outputs(campaign):
    path, plan = prepared(campaign)
    task = ctl.read(plan['cells'][0]['tasks'])[0]
    partial = Path(task['output'] + '.batches')
    partial.mkdir(parents=True)
    calls = []
    with pytest.raises(ValueError, match='preexisting calibration outputs'):
        ctl.submit(path, runner=lambda *a, **k: calls.append(a))
    assert calls == []
    assert not (path.parent / 'submission_intent.json').exists()
