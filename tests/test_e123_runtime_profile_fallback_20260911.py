"""Admission regressions for the prospective E123 GPU Adam fallback."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def packet(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
    path = ROOT / 'ops/exp_scaling/launch_e123_level3_qwen3b_factorial_runtime_repaired_20260911.py'
    spec = importlib.util.spec_from_file_location('e123_fallback_validator_test', path)
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    monkeypatch.setattr(launch, 'expected_science_environment', lambda: {'registered': 'science'})
    counter = itertools.count()
    def put(value):
        p = tmp_path / (str(next(counter)) + '.json')
        p.write_text(json.dumps(value))
        return str(p), launch.digest(p)
    plan = 'a' * 64
    summaries = {}
    for name in launch.BENCHMARK_CANDIDATES:
        value = {'schema': 'e123_a100_candidate_result_v1', 'candidate': name,
                 'plan_sha256': plan, 'status': 'passed' if name in ('gpu_adam', 'baseline') else 'failed'}
        p, sha = put(value)
        summaries[name] = {'path': p, 'sha256': sha, 'status': value['status']}
    selected = launch.read(summaries['gpu_adam']['path'])
    def fallback(rows=None):
        p, sha = put({'schema': 'e123_a100_benchmark_suite_v1', 'status': 'qualifying_fallback',
                      'plan_sha256': plan, 'candidates': rows if rows is not None else summaries})
        return {'candidate': 'gpu_adam', 'fallback_qualification': {
            'policy': 'combined_profiles_first_v1', 'fallback_used': True,
            'suite_status_path': p, 'suite_status_sha256': sha}}
    def full_profile(name='gpu_adam'):
        candidate = selected if name == 'gpu_adam' else dict(selected, candidate=name)
        cp, ch = put(candidate)
        ep, eh = put({'status': 'passed'})
        value = fallback() if name == 'gpu_adam' else {'candidate': name}
        value.update(schema='e123_a100_selected_profile_v1', status='passed', selection_status='selected',
            profile_environment=dict(launch.SYSTEMS_CANDIDATES[name], OAT_ZERO_ACTIVATION_OFFLOADING='1'),
            resources={'node': 'node302', 'gpus': 1, 'cpus': 8, 'memory_gib': 56},
            measurements={'host_peak_bytes': 40 * 1024**3, 'gpu_peak_bytes': 60 * 1024**3},
            model=launch.expected_profile_model(), science_environment={'registered': 'science'},
            dataset={'identity_path': str(launch.IDENTITY), 'identity_sha256': launch.IDENTITY_SHA256},
            evidence={'full_shape_smoke': True, 'own_checkpoint_resume': True, 'fixed_update_equivalence': True,
                      'end_to_end_smoke': True, 'candidate_result_path': cp, 'candidate_result_sha256': ch,
                      'e2e_result_path': ep, 'e2e_result_sha256': eh},
            runtime={'files_sha256': {str(path): launch.digest(path)}})
        return value
    def admit(value):
        import hashlib
        value = copy.deepcopy(value)
        value['profile_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        p, sha = put(value)
        return launch.systems_proof(p, sha)
    return launch, put, summaries, selected, fallback, full_profile, admit


@pytest.mark.parametrize('name', ['gpu_mb4_omp4', 'gpu_mb8_omp4', 'gpu_adam'])
def test_exact_registered_maps_pass_complete_admission(packet, name):
    *_, full_profile, admit = packet
    assert admit(full_profile(name))['status'] == 'passed'


@pytest.mark.parametrize('change', [
    {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '4'}, {'OMP_NUM_THREADS': '4'},
    {'OAT_ZERO_ADAM_OFFLOAD': '1'}, {'OAT_ZERO_EVAL_BATCH_SIZE': '1'},
])
def test_fallback_cannot_change_its_registered_execution_map(packet, change):
    *_, full_profile, admit = packet
    profile = full_profile(); profile['profile_environment'].update(change)
    with pytest.raises(ValueError, match='exact registered'):
        admit(profile)


def test_fallback_requires_every_original_measurement(packet):
    launch, _, rows, selected, fallback, *_ = packet
    rows = copy.deepcopy(rows); rows.pop('baseline')
    with pytest.raises(ValueError, match='seven'):
        launch.verify_fallback_qualification(fallback(rows), selected)


def test_combined_success_without_qualification_failure_prevents_fallback(packet):
    launch, put, rows, selected, fallback, *_ = packet
    rows = copy.deepcopy(rows)
    value = dict(selected, candidate='gpu_mb4_omp4')
    p, sha = put(value); rows['gpu_mb4_omp4'] = {'path': p, 'sha256': sha, 'status': 'passed'}
    with pytest.raises(ValueError, match='without a recorded'):
        launch.verify_fallback_qualification(fallback(rows), selected)


def test_failed_e2e_is_bound_and_tampering_rejected(packet):
    launch, put, rows, selected, fallback, *_ = packet
    rows = copy.deepcopy(rows)
    p, sha = put(dict(selected, candidate='gpu_mb4_omp4'))
    ep, eh = put({'schema': 'e123_a100_e2e_result_v1', 'candidate': 'gpu_mb4_omp4',
                  'plan_sha256': selected['plan_sha256'], 'status': 'failed'})
    rows['gpu_mb4_omp4'] = {'path': p, 'sha256': sha, 'status': 'passed',
        'qualification_error': 'ValueError: end-to-end failure', 'e2e_result_path': ep, 'e2e_result_sha256': eh}
    launch.verify_fallback_qualification(fallback(rows), selected)
    rows['gpu_mb4_omp4']['e2e_result_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='hash differs'):
        launch.verify_fallback_qualification(fallback(rows), selected)


def test_same_benchmark_plan_is_required(packet):
    launch, put, rows, selected, fallback, *_ = packet
    rows = copy.deepcopy(rows)
    value = launch.read(rows['mb8']['path']); value['plan_sha256'] = 'b' * 64
    p, sha = put(value); rows['mb8'].update(path=p, sha256=sha)
    with pytest.raises(ValueError, match='identity/status'):
        launch.verify_fallback_qualification(fallback(rows), selected)


def test_original_memory_and_evaluation_gates_apply_to_fallback(packet):
    *_, full_profile, admit = packet
    for edit in ('host', 'gpu', 'evaluation'):
        profile = full_profile()
        if edit == 'host': profile['resources']['memory_gib'] = 40
        elif edit == 'gpu': profile['measurements']['gpu_peak_bytes'] = 77 * 1024**3
        else: profile['evidence']['end_to_end_smoke'] = False
        with pytest.raises(ValueError): admit(profile)


def test_failed_baseline_cannot_admit_fallback(packet):
    launch, put, rows, selected, fallback, *_ = packet
    rows = copy.deepcopy(rows)
    value = launch.read(rows['baseline']['path']); value['status'] = 'failed'
    p, sha = put(value); rows['baseline'].update(path=p, sha256=sha, status='failed')
    with pytest.raises(ValueError, match='passing same-plan baseline'):
        launch.verify_fallback_qualification(fallback(rows), selected)
