"""E123 systems gate: science preservation, real loss equivalence and fail closed."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def bench():
    spec = importlib.util.spec_from_file_location('e123_runtime_test', ROOT / 'ops/exp_scaling/benchmark_e123_a100_runtime.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_real_production_loss_preserves_every_factorial_arm_across_microbatches(bench):
    result = bench.cpu_update_contract({'runtime': {'source_root': str(ROOT / 'src')}})
    assert result['status'] == 'passed'
    assert {(r['arm'], r['microbatch']) for r in result['comparisons']} == {(a, m) for a in bench.ARMS for m in (1, 4, 8)}
    for row in result['comparisons']:
        assert row['optimizer_updates'] == 1
        assert row['gradient_max_abs_difference'] < 1e-7
        assert (row['applied_replay_gradient_l2'] > 0) == row['arm'].startswith('replay_')


def test_variants_do_not_change_effective_update_or_scientific_parameters(bench):
    assert list(bench.VARIANTS) == ['baseline', 'omp4', 'mb4', 'mb8', 'gpu_adam', 'gpu_mb4_omp4', 'gpu_mb8_omp4']
    for env in bench.VARIANTS.values():
        assert set(env) == {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE', 'OMP_NUM_THREADS', 'OAT_ZERO_ADAM_OFFLOAD'}
        assert 16 % int(env['OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE']) == 0
    assert bench.VARIANTS['omp4'] | {'OMP_NUM_THREADS': '1'} == bench.VARIANTS['baseline']
    assert bench.VARIANTS['gpu_adam'] | {'OAT_ZERO_ADAM_OFFLOAD': '1'} == bench.VARIANTS['baseline']


def test_e2e_keeps_full_data_schedule_and_evaluation_while_bounding_queries(bench, tmp_path):
    frozen = {'OAT_ZERO_MAX_TRAIN': '384', 'OAT_ZERO_NUM_PROMPT_EPOCH': '8', 'OAT_ZERO_MAX_STEP_ADJUSTMENT': '16.0',
        'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS': '4', 'OAT_ZERO_EVAL_MODE_COVERAGE_K': '8',
        'OAT_ZERO_TRAIN_BATCH_SIZE': '16', 'OAT_ZERO_NUM_SAMPLES': '16', 'OAT_ZERO_ROLLOUT_BATCH_SIZE': '1',
        'OAT_ZERO_PROMPT_MAX_LENGTH': '1024', 'OAT_ZERO_GENERATE_MAX_LENGTH': '192', 'OAT_ZERO_MAX_MODEL_LEN': '2048'}
    plan = {'environments': {'graph_coloring': frozen}}
    fresh = bench.e2e_environment(plan, 'gpu_mb4_omp4', 'graph_coloring', tmp_path / 'fresh')
    resumed = bench.e2e_environment(plan, 'gpu_mb4_omp4', 'graph_coloring', tmp_path / 'resume', resume=tmp_path / 'checkpoint' / 'step_00002')
    for key, value in frozen.items(): assert fresh[key] == resumed[key] == value
    assert fresh['OAT_ZERO_MAX_QUERIES'] == '16' and resumed['OAT_ZERO_MAX_QUERIES'] == '32'
    assert 'OAT_ZERO_RESUME_DIR' not in fresh
    assert resumed['OAT_ZERO_RESUME_TAG'] == 'step_00002'
    assert fresh['OAT_ZERO_WATCHDOG_REQUEUE'] == '0'
    assert fresh['OAT_ZERO_PRUNE_RESUME_ON_SUCCESS'] == '0'


def sketch(value):
    return {'states': [{'exp_avg': {'numel': 3, 'sample': [value, value, value]}, 'exp_avg_sq': {'numel': 3, 'sample': [value, value, value]}}]}


def test_moment_gate_rejects_wrong_scale_shapes_and_nonfinite(bench):
    assert bench.compare_sketch(sketch(1), sketch(1))['passed']
    assert not bench.compare_sketch(sketch(1), sketch(1.1))['passed']
    wrong = sketch(1); wrong['states'][0]['exp_avg']['numel'] = 4
    with pytest.raises(ValueError, match='shape'): bench.compare_sketch(sketch(1), wrong)
    with pytest.raises(ValueError, match='nonfinite'): bench.compare_sketch(sketch(1), sketch(float('nan')))
    with pytest.raises(ValueError, match='empty'): bench.compare_sketch({'states': []}, {'states': []})


def fake_evidence(bench, tmp_path, *, host=40, gpu=60):
    plan = {'plan_sha256': 'x', 'runtime': {}, 'dataset': {}, 'profile_gate': {'relative_moment_sketch_rms_tolerance': .03},
            'environments': {bench.DOMAINS[0]: {'OAT_ZERO_PRETRAIN': '/pinned/model', 'OAT_ZERO_ACTIVATION_OFFLOADING': '1'}}}
    mem = {'available': True, 'peak_current_bytes': host * bench.GIB, 'peak_nonreclaimable_dirty_bytes': host * bench.GIB, 'events_delta': {'oom': 0, 'oom_kill': 0}}
    c = {'status': 'passed', 'candidate': 'gpu_mb4_omp4', 'plan_sha256': 'x', 'full_shape_smoke': True, 'own_checkpoint_resume': True,
         'fixed_update_sketch': sketch(1), 'host_memory': mem, 'gpu_peak_reserved_bytes': gpu * bench.GIB}
    e = {'status': 'passed', 'candidate': 'gpu_mb4_omp4', 'plan_sha256': 'x', 'domains': {d: {} for d in bench.DOMAINS}, 'host_memory': mem, 'gpu_peak_bytes': gpu * bench.GIB}
    b = dict(c, candidate='baseline')
    paths = [tmp_path / n for n in ('candidate.json', 'e2e.json', 'baseline.json', 'contract.json')]
    for path, payload in zip(paths, (c, e, b, {'status': 'passed'})): bench.write(path, payload)
    return plan, paths


def test_selector_cannot_admit_learner_only_or_failed_evaluation(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'verify_pins', lambda _: None)
    plan, paths = fake_evidence(bench, tmp_path)
    e = bench.read(paths[1]); e['status'] = 'failed'; bench.write(paths[1], e)
    with pytest.raises(ValueError, match='stages failed'): bench.selected_profile(plan, *paths)
    e['status'] = 'passed'; e['domains'].pop('pantry_plan'); bench.write(paths[1], e)
    with pytest.raises(ValueError, match='five-domain'): bench.selected_profile(plan, *paths)


def test_selector_rejects_oom_and_insufficient_memory_headroom(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'verify_pins', lambda _: None)
    plan, paths = fake_evidence(bench, tmp_path, gpu=77)
    with pytest.raises(ValueError, match='GPU memory'): bench.selected_profile(plan, *paths)
    plan, paths = fake_evidence(bench, tmp_path, host=120)
    with pytest.raises(ValueError, match='host footprint'): bench.selected_profile(plan, *paths)
    plan, paths = fake_evidence(bench, tmp_path)
    e = bench.read(paths[1]); e['host_memory']['events_delta']['oom'] = 1; bench.write(paths[1], e)
    with pytest.raises(ValueError, match='OOM'): bench.selected_profile(plan, *paths)


def test_pins_reject_changed_plan_and_source(bench, tmp_path):
    source = tmp_path / 'source.py'; source.write_text('pass\n')
    data = tmp_path / 'identity.json'; data.write_text('{}\n')
    p = {'runtime': {'files_sha256': {str(source): bench.digest(source)}}, 'dataset': {'identity_path': str(data), 'identity_sha256': bench.digest(data)}}
    p['plan_sha256'] = bench.identity(p)
    bench.verify_pins(p)
    source.write_text('raise RuntimeError\n')
    with pytest.raises(ValueError, match='runtime changed'): bench.verify_pins(p)
    p['plan_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='plan digest'): bench.verify_pins(p)
