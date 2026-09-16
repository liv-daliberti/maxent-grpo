"""CPU regressions for the E123 runtime-only recovery."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def bench():
    return load('e123_repaired_test', 'ops/exp_scaling/benchmark_e123_a100_runtime_repaired_20260911.py')


@pytest.mark.parametrize('size', [0, 1, 2, 255, 256, 257, 2**24-1, 2**24, 16777220, 2**31, 3_000_000_000, 2**63-1])
def test_exact_indices_are_bounded_evenly_spaced_and_include_endpoints(bench, size):
    indices = bench.optimizer_sketch_indices(size)
    assert len(indices) == min(256, size)
    assert indices == sorted(set(indices))
    assert all(isinstance(x, int) and 0 <= x < size for x in indices)
    if size:
        assert indices[0] == 0 and indices[-1] == size - 1
    if len(indices) > 1:
        gaps = [b-a for a, b in zip(indices, indices[1:])]
        assert max(gaps) - min(gaps) <= 1


def test_float32_endpoint_reproducer_is_repaired_without_large_allocation(bench):
    size = 16777220
    rounded = struct.unpack('f', struct.pack('f', size-1))[0]
    assert int(rounded) == size
    assert bench.optimizer_sketch_indices(size)[-1] == size-1


@pytest.mark.parametrize('value', [-1, 2**63, 1.5, True])
def test_invalid_counts_fail_closed(bench, value):
    with pytest.raises(ValueError, match='int64 count'):
        bench.optimizer_sketch_indices(value)


def test_actual_optimizer_sampling_and_relative_rms(bench):
    import torch
    moment = torch.arange(513, dtype=torch.float32) + 1
    state = {'step': torch.tensor(2), 'exp_avg': moment, 'exp_avg_sq': moment * 2}
    engine = SimpleNamespace(optimizer=SimpleNamespace(optimizer=SimpleNamespace(state={0: state})))
    reference = bench.optimizer_sketch(engine)
    assert reference['states'][0]['step']['sample'] == [2.0]
    assert reference['states'][0]['exp_avg']['sample'] == moment[bench.optimizer_sketch_indices(513)].tolist()
    assert bench.compare_sketch(reference, reference)['max_relative_rms'] == 0
    candidate = copy.deepcopy(reference)
    for key in ('exp_avg', 'exp_avg_sq'):
        candidate['states'][0][key]['sample'] = [v * 1.02 for v in candidate['states'][0][key]['sample']]
    comparison = bench.compare_sketch(reference, candidate)
    assert comparison['passed'] and math.isclose(comparison['max_relative_rms'], .02, abs_tol=1e-12)
    assert not bench.compare_sketch(reference, candidate, tolerance=.01)['passed']
    candidate['states'][0]['exp_avg']['sample'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        bench.compare_sketch(reference, candidate)


def test_actual_bootstrap_finds_installed_runtime_ninja(tmp_path):
    recovery = load('e123_recovery_path_test', 'ops/exp_scaling/recover_e123_benchmark_runtime_20260911.py')
    preview = recovery.auto.launch.read(recovery.ORIGINAL / 'reviewed_preview.json')
    command = [str(recovery.auto.PYTHON), '-c', 'from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability(); print("NINJA_DISCOVERED")']
    script = tmp_path / 'preflight.sh'; script.write_text(recovery.wrapper(preview, command))
    result = subprocess.run(['bash', str(script)], env={'PATH': '/usr/bin:/bin'}, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'NINJA_DISCOVERED' in result.stdout


def test_existing_end_to_end_and_moment_gates_remain_enforced(bench, tmp_path, monkeypatch):
    prior = load('e123_prior_gate_tests', 'tests/test_e123_a100_runtime.py')
    prior.test_variants_do_not_change_effective_update_or_scientific_parameters(bench)
    prior.test_e2e_keeps_full_data_schedule_and_evaluation_while_bounding_queries(bench, tmp_path)
    prior.test_moment_gate_rejects_wrong_scale_shapes_and_nonfinite(bench)
    prior.test_selector_cannot_admit_learner_only_or_failed_evaluation(bench, tmp_path, monkeypatch)
    prior.test_selector_rejects_oom_and_insufficient_memory_headroom(bench, tmp_path, monkeypatch)


@pytest.mark.parametrize('scenario,expected,code', [
    ('combined_passes', ['gpu_mb8_omp4'], 0),
    ('combined_rejected', ['gpu_mb8_omp4', 'gpu_mb4_omp4', 'gpu_adam'], 0),
    ('combined_oom', ['gpu_adam'], 0),
    ('all_failed', [], 1),
    ('fallback_rejected', ['gpu_adam'], 1),
])
def test_suite_preserves_combined_preference_and_requires_passing_fallback(bench, tmp_path, monkeypatch, scenario, expected, code):
    plan = {'plan_sha256': 'a'*64, 'output_root': str(tmp_path)}
    plan_path = tmp_path / 'plan.json'; bench.write(plan_path, plan)
    bench.write(tmp_path / 'production_loss_contract.json', {'status': 'passed'})
    monkeypatch.setattr(bench, 'verify_pins', lambda _: None)
    for name in bench.VARIANTS:
        state = 'passed'
        if name in ('gpu_mb4_omp4', 'gpu_mb8_omp4') and scenario in ('combined_oom', 'all_failed', 'fallback_rejected'):
            state = 'failed'
        if name == 'gpu_adam' and scenario == 'all_failed': state = 'failed'
        bench.write(tmp_path / name / 'candidate_result.json', {
            'schema': 'e123_a100_candidate_result_v1', 'candidate': name, 'plan_sha256': plan['plan_sha256'],
            'status': state, 'domains': {bench.DOMAINS[0]: {'median_update_seconds': 1 if name == 'gpu_mb8_omp4' else .01 if name == 'gpu_adam' else 2}}})
    called = []
    def end_to_end(_, name):
        bench.write(tmp_path / name / 'end_to_end/e2e_result.json', {
            'schema': 'e123_a100_e2e_result_v1', 'candidate': name, 'status': 'passed', 'plan_sha256': plan['plan_sha256']})
    def selected(_, candidate, *args, fallback_attempt=None):
        name = bench.read(candidate)['candidate']; called.append(name)
        if name != 'gpu_adam' and scenario == 'combined_rejected': raise ValueError('measured GPU headroom failed')
        if name == 'gpu_adam':
            assert fallback_attempt is not None
            proof = bench.read(fallback_attempt)
            assert proof['status'] == 'qualifying_fallback' and len(proof['candidates']) == 7
            for previous in ('gpu_mb4_omp4', 'gpu_mb8_omp4'):
                entry = proof['candidates'][previous]
                assert entry['status'] == 'failed' or entry['qualification_error']
            if scenario == 'fallback_rejected': raise ValueError('fallback memory failed')
        return {'status': 'passed', 'candidate': name}
    monkeypatch.setattr(bench, 'end_to_end', end_to_end)
    monkeypatch.setattr(bench, 'selected_profile', selected)
    assert bench.suite(plan_path) == code
    assert called == expected
    assert (tmp_path / 'selected_profile.json').exists() == (code == 0)


def test_fallback_cannot_bypass_missing_baseline_or_order_evidence(bench, tmp_path, monkeypatch):
    prior = load('e123_prior_fallback_tests', 'tests/test_e123_a100_runtime.py')
    monkeypatch.setattr(bench, 'verify_pins', lambda _: None)
    plan, paths = prior.fake_evidence(bench, tmp_path)
    for path in paths[:2]:
        data = bench.read(path); data['candidate'] = 'gpu_adam'; bench.write(path, data)
    with pytest.raises(ValueError, match='both combined qualification failures'):
        bench.selected_profile(plan, *paths)
    data = bench.read(paths[2]); data['status'] = 'failed'; bench.write(paths[2], data)
    with pytest.raises(ValueError, match='stages failed'):
        bench.selected_profile(plan, *paths)


def test_only_approved_harness_functions_and_constants_changed():
    recovery = load('e123_repair_ast_test', 'ops/exp_scaling/recover_e123_benchmark_runtime_20260911.py')
    result = recovery.verify_harness_delta()
    assert set(result['changed_functions']) == {'optimizer_sketch', 'selected_profile', 'suite'}
