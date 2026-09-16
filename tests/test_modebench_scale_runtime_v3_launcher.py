"""Batch-eight runtime invariance and worker admission, without GPUs or Slurm."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ctl = module('scale_runtime_v3_launcher_test', ROOT / 'ops/exp_scaling/launch_modebench_scale_runtime_v3.py')
previous_tests = module('scale_runtime_v2_fixture_helpers', ROOT / 'tests/test_modebench_scale_runtime_v2_launcher.py')


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(previous_tests, 'ctl', ctl)
    return previous_tests.campaign.__wrapped__(tmp_path, monkeypatch)


def prepare(campaign, **kwargs):
    data, models, root = campaign
    return root / 'plan.json', ctl.prepare(None, root, models, data_root=data, **kwargs)


def test_batch8_profile_commands_and_actual_task_manifests_match(campaign):
    path, plan = prepare(campaign)
    assert plan['runtime_profile'] == ctl.RUNTIME_PROFILE
    assert plan['runtime_profile']['batch_size'] == 8
    assert plan['runtime_profile']['thread_environment'] == {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'}
    assert '--array=0-9%1' in plan['submit_command']
    assert '--gres=gpu:a5000:2' in plan['submit_command']
    assert '--mem=60G' in plan['submit_command']
    assert len(plan['cells']) == 10
    assert plan['rng_admission']['distinct_request_blocks'] == 160
    assert plan['rng_admission']['distinct_child_seeds'] == 1280
    for cell in plan['cells']:
        args = ctl.evaluator().parse_args(cell['command'][2:])
        assert args.tensor_parallel_size == 2 and args.swap_space == 4.0
        assert args.gpu_memory_utilization == .82 and args.enable_prefix_caching is True
        assert args.max_model_len == ctl.evaluator().frozen_interface(cell['domain'])['max_model_len']
        tasks = ctl.read(cell['tasks'])
        assert len(tasks) == 4
        assert all(task['batch_size'] == 8 and len(task['seeds']) == 4 for task in tasks)
        assert shlex.split(cell['shell_command']) == cell['command']
    script = (path.parent / 'worker.slurm').read_text()
    assert 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1' in script
    subprocess.run(['bash', '-n', str(path.parent / 'worker.slurm')], check=True)
    assert ctl.verify(path, fresh=True) == plan


def test_batch8_preserves_original_scientific_tasks_including_batch_and_rng(campaign, monkeypatch):
    original = module('scale_v3_original_v1', ctl.SCIENTIFIC_ORIGIN_SOURCE)
    monkeypatch.setattr(original, 'PYTHON', Path(sys.executable))
    data, models, root = campaign
    before = original.prepare(None, root.parent / 'v1-plan', models, data_root=data)
    _, after = prepare(campaign)
    assert before['models'] == after['models']
    assert before['rng_admission'] == after['rng_admission']
    for old_cell, new_cell in zip(before['cells'], after['cells']):
        assert old_cell['id'] == new_cell['id']
        assert original.read(old_cell['tasks']) == ctl.read(new_cell['tasks'])


def test_batch8_changes_only_batch_from_v2_tasks_and_pins_both_origins(campaign, monkeypatch):
    prior = module('scale_v3_original_v2', ctl.ORIGIN_SOURCE)
    monkeypatch.setattr(prior, 'PYTHON', Path(sys.executable))
    data, models, root = campaign
    before = prior.prepare(None, root.parent / 'v2-plan', models, data_root=data)
    path, after = prepare(campaign)
    assert before['models'] == after['models']
    assert before['rng_admission'] == after['rng_admission']
    assert after['origin_chain'] == ctl.ORIGIN_CHAIN
    for source, expected in ctl.ORIGIN_CHAIN.items():
        assert ctl.digest(source) == after['immutable_inputs_sha256'][source] == expected
    for old_cell, new_cell in zip(before['cells'], after['cells']):
        for old_task, new_task in zip(prior.read(old_cell['tasks']), ctl.read(new_cell['tasks'])):
            assert old_task == {**new_task, 'batch_size': 2}
    assert ctl.verify(path) == after


@pytest.mark.parametrize('concurrency', [0, 3, 4, True, 1.0])
def test_batch8_rejects_invalid_gpu_concurrency_before_claim(campaign, concurrency):
    with pytest.raises(ValueError, match='concurrency must be 1..2'):
        prepare(campaign, concurrency=concurrency)
    assert not campaign[2].exists()


def test_batch8_accepts_at_most_four_gpus_when_requested(campaign):
    path, plan = prepare(campaign, concurrency=2)
    assert '--array=0-9%2' in plan['submit_command']
    assert plan['concurrency'] * plan['hardware']['tensor_parallel_size'] == 4
    assert ctl.verify(path) == plan


def test_batch8_worker_rejects_unqualified_cpu_thread_environment(campaign, monkeypatch):
    path, _ = prepare(campaign)
    monkeypatch.setenv('VLLM_USE_V1', '0')
    monkeypatch.setenv('VLLM_ATTENTION_BACKEND', 'XFORMERS')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '1')
    with pytest.raises(ValueError, match='CPU thread environment'):
        ctl.worker(path, 0)
    assert not (path.parent / 'runtime/0.json').exists()


def test_batch8_worker_executes_exact_sealed_evaluator_after_array_identity(campaign, monkeypatch):
    import importlib.metadata
    path, plan = prepare(campaign)
    for key, value in {'VLLM_USE_V1': '0', 'VLLM_ATTENTION_BACKEND': 'XFORMERS',
                       'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
                       'SLURM_ARRAY_JOB_ID': '1234', 'SLURM_ARRAY_TASK_ID': '0'}.items():
        monkeypatch.setenv(key, value)
    ctl.atomic_new(path.parent / 'submission_result.json', {'array_job_id': 1234})
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.8.4')
    calls = []
    monkeypatch.setattr(ctl.os, 'execv', lambda executable, argv: calls.append((executable, argv)))
    ctl.worker(path, 0)
    assert calls == [(plan['python'], plan['cells'][0]['command'])]
    runtime = ctl.read(path.parent / 'runtime/0.json')
    assert runtime['command'] == plan['cells'][0]['command']
    assert runtime['array_job_id'] == 1234
    assert runtime['plan_sha256'] == ctl.digest(path)
