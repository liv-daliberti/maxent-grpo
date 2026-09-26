"""Immutable partial/mixed-domain launch tests using synthetic data and models."""
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import launch_modebench_scale_domains as launch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def model(path, label):
    width, layers = {'7b': (3584, 28), '14b': (5120, 48)}[label]
    write(path / 'config.json', {'architectures': ['Qwen2ForCausalLM'],
                                'hidden_size': width, 'num_hidden_layers': layers})
    write(path / 'model.safetensors.index.json', {'weight_map': {'weight': 'model-1.safetensors'}})
    write(path / 'tokenizer_config.json', {})
    write(path / 'tokenizer.json', {})
    (path / 'model-1.safetensors').write_bytes(b'synthetic weights')
    return path


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'PYTHON', Path(sys.executable))
    models = {label: model(tmp_path / ('model ' + label), label) for label in ('7b', '14b')}
    protocol_models = {label: launch.evaluator().model_identity(path, label) for label, path in models.items()}
    pins = {}
    def pin(path):
        pins[str(path.resolve())] = launch.digest(path)
    def cell(level, domain, phase='dev', kind='domain_revision_v1', root=None):
        root = root or tmp_path / (kind + '_' + level + '_' + domain)
        write(root / 'protocol.json', {'schema': kind, 'models': protocol_models})
        pin(root / 'protocol.json')
        base = root / level / ('pools' if phase == 'dev' else 'dataset') / domain
        write(base / 'identity.json', {'synthetic_scientific_fixture': True})
        pin(base / 'identity.json')
        tasks = []
        if phase == 'eval':
            recipe = root / level / 'recipes' / (domain + '.json')
            write(recipe, {'synthetic_passing_recipe_fixture': True})
            pin(recipe)
        for tier in range(4) if phase == 'dev' else [None]:
            source = base / (f'difficulty_{tier}.jsonl' if phase == 'dev' else 'eval.jsonl')
            rows = [{'problem': f'{root}/{level}/{domain}/{phase}/{tier}/{index}', 'answer': '{}'} for index in range(2)]
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            pin(source)
            tasks.append({'level': level, 'domain': domain, 'split': phase, 'interface': launch.evaluator().INTERFACE,
                          'rows_jsonl': str(source), 'seeds': [7650000, 7650001, 7650002, 7650003],
                          'batch_size': 8, 'row_limit': 0, 'row_offset': 0,
                          'output': str(root / level / 'results' / phase / domain / (str(tier) + '.json'))})
        return {'id': level + '_' + domain, 'level': level, 'domain': domain, 'phase': phase,
                'source_kind': kind, 'source_root': str(root), 'tasks': tasks}
    cells = [cell('level4', domain) for domain in ('graph_coloring', 'python_factors')]
    return SimpleNamespace(models=models, cells=cells, pins=pins, cell=cell, pin=pin,
                           root=tmp_path / 'launch `literal` space', tmp=tmp_path)


def prepare(campaign, cells=None):
    plan = launch.prepare(campaign.root, campaign.cells if cells is None else cells, campaign.models,
                          dependency_ids=[900, 700], pins=campaign.pins)
    return campaign.root / 'plan.json', plan


def test_partial_two_domain_revision_preserves_explicit_labels_and_qualified_runtime(campaign):
    path, plan = prepare(campaign)
    assert len(plan['cells']) == 2 and set(plan['models']) == {'7b'}
    assert plan['concurrency'] == 1 and '--array=0-1%1' in plan['submit_command']
    assert '--gres=gpu:a5000:2' in plan['submit_command']
    assert '--dependency=afterany:700:900' in plan['submit_command']
    assert plan['runtime_profile'] == launch.runtime.RUNTIME_PROFILE
    assert plan['rng_admission']['distinct_request_blocks'] == 2 * 4 * 2 * 4
    assert plan['rng_admission']['distinct_child_seeds'] == 2 * 4 * 2 * 4 * 8
    for cell in plan['cells']:
        tasks = launch.read(cell['tasks'])
        assert len(tasks) == 4 and all(task['seeds'] == [7650000, 7650001, 7650002, 7650003] for task in tasks)
        assert shlex.split(cell['shell_command']) == cell['command']
        assert '--confirm-eval' not in cell['command']
    assert shlex.split(plan['submit_shell_command']) == plan['submit_command']
    assert launch.verify(path, fresh=True) == plan
    script = (campaign.root / 'worker.slurm').read_text()
    assert str(launch.SOURCE) in script and 'OPENBLAS_NUM_THREADS=1' in script
    subprocess.run(['bash', '-n', str(campaign.root / 'worker.slurm')], check=True)
    assert not (campaign.root / 'submission_intent.json').exists()


def test_mixed_source_confirmation_accepts_original_and_revised_domains(campaign):
    cells = [campaign.cell('level4', 'countdown', 'eval', 'campaign_v1'),
             campaign.cell('level4', 'graph_coloring', 'eval'),
             campaign.cell('level5', 'mathir', 'eval', 'campaign_v1')]
    path, plan = prepare(campaign, cells)
    assert plan['phase'] == 'eval' and len(plan['cells']) == 3
    assert set(plan['models']) == {'7b', '14b'}
    assert {cell['source_kind'] for cell in plan['cells']} == {'campaign_v1', 'domain_revision_v1'}
    assert all('--confirm-eval' in cell['command'] and len(launch.read(cell['tasks'])) == 1 for cell in plan['cells'])
    assert launch.verify(path, fresh=True) == plan


def test_cell_specific_phase_controls_confirmation_flag(campaign):
    cells = [campaign.cells[0], campaign.cell('level4', 'mathir', 'eval', 'campaign_v1')]
    _, plan = prepare(campaign, cells)
    assert plan['phase'] == 'mixed'
    assert '--confirm-eval' not in plan['cells'][0]['command']
    assert '--confirm-eval' in plan['cells'][1]['command']


@pytest.mark.parametrize('missing', ['protocol', 'certificate', 'input', 'recipe'])
def test_missing_caller_scientific_pins_fail_before_claim(campaign, missing):
    cells = [campaign.cell('level4', 'graph_coloring', 'eval')]
    root = Path(cells[0]['source_root'])
    paths = {'protocol': root / 'protocol.json', 'certificate': root / 'level4/dataset/graph_coloring/identity.json',
             'input': Path(cells[0]['tasks'][0]['rows_jsonl']), 'recipe': root / 'level4/recipes/graph_coloring.json'}
    campaign.pins.pop(str(paths[missing]))
    with pytest.raises(ValueError, match='caller must pin'):
        prepare(campaign, cells)
    assert not campaign.root.exists()


@pytest.mark.parametrize('kind', ['id', 'logical', 'task_source', 'output'])
def test_duplicate_cells_sources_or_outputs_fail_before_claim(campaign, kind):
    if kind == 'id':
        campaign.cells.append(copy.deepcopy(campaign.cells[0]))
    elif kind == 'logical':
        duplicate = copy.deepcopy(campaign.cells[0])
        duplicate['id'] += '_duplicate'
        campaign.cells.append(duplicate)
    elif kind == 'task_source':
        campaign.cells[0]['tasks'][1]['rows_jsonl'] = campaign.cells[0]['tasks'][0]['rows_jsonl']
    else:
        campaign.cells[1]['tasks'][0]['output'] = campaign.cells[0]['tasks'][0]['output']
    with pytest.raises(ValueError, match='duplicate'):
        prepare(campaign)
    assert not campaign.root.exists()


def test_collision_between_distinct_full_task_sources_fails_before_claim(campaign, monkeypatch):
    frozen = launch.evaluator().frozen
    original = frozen.schedule_record
    first = campaign.cells[0]['tasks'][0]
    rows, _ = launch.evaluator().load_rows(first)
    schedule = original(first['domain'], rows, first['seeds'])
    monkeypatch.setattr(frozen, 'schedule_record', lambda *args: schedule)
    with pytest.raises(ValueError, match='RNG block collision'):
        prepare(campaign)
    assert not campaign.root.exists()


@pytest.mark.parametrize('field,value', [('row_offset', 1), ('row_limit', 1), ('batch_size', 2)])
def test_sliced_or_unqualified_tasks_are_rejected(campaign, field, value):
    campaign.cells[0]['tasks'][0][field] = value
    with pytest.raises(ValueError, match='qualified batching and complete source rows'):
        prepare(campaign)


def test_four_tiers_must_share_the_registered_four_draw_labels(campaign):
    campaign.cells[0]['tasks'][1]['seeds'] = [8650000, 8650001, 8650002, 8650003]
    with pytest.raises(ValueError, match='common registered draw labels'):
        prepare(campaign)


@pytest.mark.parametrize('kind', ['receipt', 'batches'])
def test_existing_outputs_or_partial_batches_are_never_reused(campaign, kind):
    output = Path(campaign.cells[0]['tasks'][0]['output'])
    if kind == 'receipt':
        write(output, {'old': True})
    else:
        Path(str(output) + '.batches').mkdir(parents=True)
    with pytest.raises(ValueError, match='preexisting calibration outputs'):
        prepare(campaign)
    assert not campaign.root.exists()


@pytest.mark.parametrize('kind', ['source', 'certificate', 'model', 'task', 'worker', 'plan'])
def test_changed_pins_or_prepared_inputs_prevent_submission(campaign, kind):
    path, plan = prepare(campaign)
    cell = plan['cells'][0]
    paths = {'source': Path(launch.read(cell['tasks'])[0]['rows_jsonl']),
             'certificate': Path(cell['source_root']) / 'level4/pools/graph_coloring/identity.json',
             'model': campaign.models['7b'] / 'model-1.safetensors',
             'task': Path(cell['tasks']), 'worker': path.parent / 'worker.slurm', 'plan': path}
    changed = paths[kind]
    changed.write_text(changed.read_text() + ' ')
    calls = []
    with pytest.raises(ValueError):
        launch.submit(path, runner=lambda *a, **k: calls.append(a))
    assert calls == [] and not (path.parent / 'submission_intent.json').exists()


def test_other_same_size_model_is_rejected_by_source_protocol(campaign):
    campaign.models['7b'] = model(campaign.tmp / 'other_model', '7b')
    with pytest.raises(ValueError, match='checkpoint differs from source protocol'):
        prepare(campaign)


@pytest.mark.parametrize('result', ['success', 'ambiguous', 'timeout'])
def test_submission_intent_prevents_any_second_attempt(campaign, result):
    path, _ = prepare(campaign)
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        assert launch.read(path.parent / 'submission_intent.json')['command'] == command
        if result == 'timeout':
            raise subprocess.TimeoutExpired(command, 60)
        return SimpleNamespace(returncode=0, stdout='1200;cluster\n' if result == 'success' else 'unknown', stderr='')
    if result == 'success':
        receipt = launch.submit(path, runner=runner)
        assert receipt['array_job_id'] == 1200 and len(receipt['cells']) == 2
    else:
        with pytest.raises(subprocess.TimeoutExpired if result == 'timeout' else RuntimeError):
            launch.submit(path, runner=runner)
    with pytest.raises(ValueError, match='already attempted'):
        launch.submit(path, runner=runner)
    assert len(calls) == 1


def worker_environment(monkeypatch):
    for key, value in {'VLLM_USE_V1': '0', 'VLLM_ATTENTION_BACKEND': 'XFORMERS', 'SLURM_ARRAY_JOB_ID': '1200',
                       'SLURM_ARRAY_TASK_ID': '0', **launch.RUNTIME_PROFILE['thread_environment']}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(launch.importlib.metadata, 'version', lambda name: launch.HARDWARE['vllm_version'])


def test_worker_closes_submission_result_race_and_records_its_exact_cell(campaign, monkeypatch):
    path, plan = prepare(campaign)
    worker_environment(monkeypatch)
    launch.atomic_new(path.parent / 'submission_intent.json',
                      {'plan_sha256': launch.digest(path), 'command': plan['submit_command']})
    waits, executed = [], []
    def publish_after_start(seconds):
        waits.append(seconds)
        launch.atomic_new(path.parent / 'submission_result.json', {'status': 'submitted', 'array_job_id': 1200})
    monkeypatch.setattr(launch.time, 'sleep', publish_after_start)
    monkeypatch.setattr(launch.os, 'execv', lambda *args: executed.append(args))
    launch.worker(path, 0)
    assert waits == [1]
    assert executed == [(plan['python'], plan['cells'][0]['command'])]
    receipt = launch.read(path.parent / 'runtime/0.json')
    assert receipt['cell'] == 'level4_graph_coloring' and receipt['array_job_id'] == 1200
    assert receipt['plan_sha256'] == launch.digest(path)


def test_worker_rejects_thread_profile_drift_before_execution(campaign, monkeypatch):
    path, _ = prepare(campaign)
    worker_environment(monkeypatch)
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '32')
    with pytest.raises(ValueError, match='thread environment'):
        launch.worker(path, 0)
    assert not (path.parent / 'runtime/0.json').exists()


def test_qualified_original_launcher_sources_are_unchanged():
    assert all(launch.digest(path) == expected for path, expected in launch.ORIGIN_CHAIN.items())


def test_confirmation_arrow_inputs_require_every_saved_dataset_file(campaign):
    from datasets import Dataset, DatasetDict
    cells = [campaign.cell('level4', 'countdown', 'eval', 'campaign_v1')]
    task = cells[0]['tasks'][0]
    rows, _ = launch.evaluator().load_rows(task)
    directory = Path(task.pop('rows_jsonl')).with_suffix('')
    DatasetDict({'multi_answer': Dataset.from_list(rows)}).save_to_disk(str(directory))
    task['dataset'] = str(directory)
    with pytest.raises(ValueError, match='pin every task input file'):
        prepare(campaign, cells)
    assert not campaign.root.exists()
    for source in directory.rglob('*'):
        if source.is_file():
            campaign.pin(source)
    path, plan = prepare(campaign, cells)
    assert launch.verify(path, fresh=True) == plan
    assert plan['rng_admission']['distinct_request_blocks'] == len(rows) * 4


@pytest.mark.parametrize('count', [0, 3, 5])
def test_development_requires_all_four_tiers(campaign, count):
    task = copy.deepcopy(campaign.cells[0]['tasks'][0])
    campaign.cells[0]['tasks'] = campaign.cells[0]['tasks'][:count]
    if count == 5:
        campaign.cells[0]['tasks'].append(task)
    with pytest.raises(ValueError, match='four development tasks'):
        prepare(campaign)
    assert not campaign.root.exists()


def test_invalid_dependency_ids_fail_without_claiming_campaign(campaign):
    with pytest.raises(ValueError, match='distinct positive integer'):
        launch.prepare(campaign.root, campaign.cells, campaign.models,
                       dependency_ids=[900, 900], pins=campaign.pins)
    assert not campaign.root.exists()
