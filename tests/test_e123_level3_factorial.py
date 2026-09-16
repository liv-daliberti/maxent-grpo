"""E123 scientific invariants, measured-profile gate and once-only scheduler audit."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shlex
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'ops/exp_scaling/launch_e123_level3_qwen3b_factorial.py'


@pytest.fixture(scope='module')
def launcher():
    spec = importlib.util.spec_from_file_location('e123_launcher_test', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profile():
    return {'profile_environment': {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '4',
        'OAT_ZERO_ADAM_OFFLOAD': '0', 'OAT_ZERO_ACTIVATION_OFFLOADING': '1',
        'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
        'resources': {'cpus': 8, 'memory_gib': 56, 'node': 'node302', 'gpus': 1}}


@pytest.fixture
def cells(launcher, profile):
    return launcher.planned_cells('3b', '/reviewed/e123/snapshot', {'path': '/frozen/qwen3b'}, profile)


def test_complete_qwen3b_factorial_preserves_scale_aware_science(launcher, cells):
    assert len(cells) == 100
    assert {(c['domain'], c['arm'], c['seed']) for c in cells} == {
        (d, a, s) for d in launcher.DOMAINS for a in launcher.ARMS for s in range(70, 75)}
    for cell in cells:
        env = cell['environment']
        expected = {'OAT_ZERO_LEARNING_RATE': '1e-07', 'OAT_ZERO_LR_SCHEDULER': 'cosine_with_min_lr',
            'OAT_ZERO_LR_WARMUP_RATIO': '0.1', 'OAT_ZERO_MAX_STEP_ADJUSTMENT': '16.0',
            'OAT_ZERO_ADAM_BETA_1': '0.9', 'OAT_ZERO_ADAM_BETA_2': '0.999',
            'OAT_ZERO_MAX_NORM': '1.0', 'OAT_ZERO_BETA': '0.0', 'OAT_ZERO_NUM_SAMPLES': '16',
            'OAT_ZERO_NUM_PPO_EPOCHS': '1', 'OAT_ZERO_NUM_PROMPT_EPOCH': '8', 'OAT_ZERO_MAX_TRAIN': '384',
            'OAT_ZERO_ROLLOUT_BATCH_SIZE': '1', 'OAT_ZERO_TRAIN_BATCH_SIZE': '16',
            'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '4', 'OAT_ZERO_EVAL_BATCH_SIZE': '32',
            'OAT_ZERO_EVAL_MODE_COVERAGE_K': '8', 'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS': '4',
            'OAT_ZERO_EVAL_PROMPT_INTERVAL': '96', 'OAT_ZERO_ALLOW_SPARSE_EVAL': '0',
            'OAT_ZERO_PROMPT_MAX_LENGTH': '1024', 'OAT_ZERO_GENERATE_MAX_LENGTH': '192',
            'OAT_ZERO_EVAL_GENERATE_MAX_LENGTH': '192', 'OAT_ZERO_MAX_MODEL_LEN': '2048',
            'OAT_ZERO_TEMPERATURE': '1.0', 'OAT_ZERO_TOP_P': '1.0',
            'OAT_ZERO_ADAM_OFFLOAD': '0', 'OMP_NUM_THREADS': '4'}
        assert {k: env[k] for k in expected} == expected
        assert cell['target_steps'] == 3072
        assert env['OAT_ZERO_RESUME_STEPS'] == ('96' if cell['domain'] == 'pantry_plan' else '192')
        assert env['OAT_ZERO_SAVE_STEPS'] == env['OAT_ZERO_RESUME_STEPS']
        assert '--hold' in cell['command'] and '--mem=56G' in cell['command']
        assert '--nodelist=node302' in cell['command']


def test_only_factorial_objectives_differ_within_matched_cells(launcher, cells):
    changing = {'OAT_ZERO_VARIANT', 'OAT_ZERO_MAXRL_TASK_OBJECTIVE',
                'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY', 'SAVE_PATH', 'RUN_STAMP'}
    for domain in launcher.DOMAINS:
        paired = [c for c in cells if c['domain'] == domain and c['seed'] == 70]
        assert len({tuple((k, v) for k, v in c['environment'].items() if k not in changing) for c in paired}) == 1
        for cell in paired:
            env, arm = cell['environment'], cell['arm']
            assert env['OAT_ZERO_MAXRL_TASK_OBJECTIVE'] == ('1' if 'maxrl' in arm else '0')
            assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY'] == ('0' if arm.startswith('replay_') else '1')
            assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE'] == 'verified_likelihood_per_rollout'
            assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA'] == '0.1'


def test_raw_level3_frozen_split_paths_and_fresh_e123_namespace(launcher, cells):
    assert launcher.MODEL_REVISIONS['3b'] == 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert launcher.IDENTITY_SHA256 == '890d7697af7789e0ae53c803586ec7f722685b7eb2239643175a1933fa45650d'
    for cell in cells:
        env = cell['environment']
        assert env['OAT_ZERO_DATA_ROOT'] == str(launcher.DATA_ROOT / cell['dataset_domain'])
        assert env['OAT_ZERO_REQUIRE_EXISTING_DATA'] == '1'
        assert env['OAT_ZERO_PROMPT_DATA'] == str(launcher.DATA_ROOT / cell['dataset_domain'] / 'train')
        assert env['OAT_ZERO_EVAL_DATA'] == str(launcher.DATA_ROOT / cell['dataset_domain'] / 'eval')
        assert env['OAT_ZERO_CANONICAL_ACTION_TASK'] == 'none'
        assert env['OAT_ZERO_CANONICAL_GRAPH_ACTIONS'] == '0'
        assert cell['run_stamp'].startswith('e123_level3_')
        assert cell['run_dir'] == str(ROOT / 'var/data/e123_level3_factorial' / cell['domain'] / cell['arm'] / f's{cell["seed"]}')


def seal_profile(launcher, tmp_path, profile):
    path = tmp_path / 'selected.json'
    payload = deepcopy(profile)
    payload.pop('profile_sha256', None)
    payload['profile_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path.write_text(json.dumps(payload))
    return path, launcher.digest(path)


@pytest.fixture
def proved_profile(launcher, profile, tmp_path):
    candidate = tmp_path / 'candidate.json'
    candidate.write_text(json.dumps({'schema': 'e123_a100_candidate_result_v1', 'status': 'passed'}))
    e2e = tmp_path / 'e2e.json'
    e2e.write_text(json.dumps({'status': 'passed'}))
    runtime = tmp_path / 'measured_runtime.py'
    runtime.write_text('measured immutable source\n')
    profile.update(schema='e123_a100_selected_profile_v1', status='passed', selection_status='selected',
        model=launcher.expected_profile_model(), science_environment=launcher.expected_science_environment(),
        dataset={'identity_path': str(launcher.IDENTITY), 'identity_sha256': launcher.IDENTITY_SHA256},
        runtime={'files_sha256': {str(runtime): launcher.digest(runtime)}},
        measurements={'host_peak_bytes': 40 * 1024**3, 'gpu_peak_bytes': 60 * 1024**3},
        evidence={'candidate_result_path': str(candidate), 'candidate_result_sha256': launcher.digest(candidate),
            'e2e_result_path': str(e2e), 'e2e_result_sha256': launcher.digest(e2e),
            'full_shape_smoke': True, 'own_checkpoint_resume': True,
            'fixed_update_equivalence': True, 'end_to_end_smoke': True})
    return profile


def test_missing_benchmark_previews_conservatively_but_never_admits(launcher):
    assert launcher.systems_proof(required=False)['status'] == 'awaiting_benchmark'
    assert launcher.resources('3b', 'graph_coloring')['memory_gib'] == 128
    with pytest.raises(ValueError, match='awaits a measured passing'):
        launcher.systems_proof()


def test_selected_benchmark_has_explicit_file_and_payload_pins(launcher, proved_profile, tmp_path):
    path, sha = seal_profile(launcher, tmp_path, proved_profile)
    proof = launcher.systems_proof(path, sha)
    assert proof['status'] == 'passed'
    with pytest.raises(ValueError, match='explicit systems profile'):
        launcher.systems_proof(path)
    with pytest.raises(ValueError, match='file hash'):
        launcher.systems_proof(path, '0' * 64)
    source = Path(next(iter(proved_profile['runtime']['files_sha256'])))
    source.write_text('drift')
    with pytest.raises(ValueError):
        launcher.systems_proof(path, sha)


@pytest.mark.parametrize('mutation', ['not_selected', 'missing_resume', 'no_e2e', 'science_override', 'no_host_headroom',
    'gpu_overfull', 'other_dataset', 'no_equivalence', 'wrong_identity', 'cpu_adam', 'wrong_threads', 'other_model', 'other_science'])
def test_unmeasured_or_science_changing_profiles_fail_closed(launcher, proved_profile, tmp_path, mutation):
    profile = deepcopy(proved_profile)
    if mutation == 'not_selected': profile['selection_status'] = 'candidate'
    elif mutation == 'missing_resume': profile['evidence']['own_checkpoint_resume'] = False
    elif mutation == 'no_e2e': profile['evidence']['end_to_end_smoke'] = False
    elif mutation == 'science_override': profile['profile_environment']['OAT_ZERO_LEARNING_RATE'] = '1e-4'
    elif mutation == 'no_host_headroom': profile['measurements']['host_peak_bytes'] = 53 * 1024**3
    elif mutation == 'gpu_overfull': profile['measurements']['gpu_peak_bytes'] = 79 * 1024**3
    elif mutation == 'other_dataset': profile['dataset']['identity_sha256'] = '0' * 64
    elif mutation == 'no_equivalence': profile['evidence']['fixed_update_equivalence'] = False
    elif mutation == 'cpu_adam': profile['profile_environment']['OAT_ZERO_ADAM_OFFLOAD'] = '1'
    elif mutation == 'wrong_threads': profile['profile_environment']['OMP_NUM_THREADS'] = '1'
    elif mutation == 'other_model': profile['model']['revision'] = '0' * 40
    elif mutation == 'other_science': profile['science_environment']['OAT_ZERO_ADAM_BETA_2'] = '0.95'
    path, sha = seal_profile(launcher, tmp_path, profile)
    if mutation == 'wrong_identity':
        payload = json.loads(path.read_text()); payload['profile_sha256'] = '0' * 64
        path.write_text(json.dumps(payload)); sha = launcher.digest(path)
    with pytest.raises(ValueError):
        launcher.systems_proof(path, sha)


def scheduler_record(cell):
    r = cell['resources']
    values = {'JobId': '12345', 'JobName': cell['run_stamp'], 'JobState': 'PENDING', 'Reason': 'JobHeldUser',
        'Account': r['account'], 'Partition': r['partition'], 'QOS': r['expected_qos'], 'NumCPUs': str(r['cpus']),
        'NumNodes': '1-1', 'NumTasks': '1', 'Requeue': '1', 'Nice': str(r['nice']),
        'UserId': f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})',
        'WorkDir': str(ROOT), 'Command': cell['command'][-1], 'RunTime': '00:00:00', 'Restarts': '0',
        'MinMemoryNode': str(r['memory_gib']) + 'G', 'TimeLimit': r['walltime'],
        'ExcNodeList': r['exclude'], 'ReqNodeList': 'node302', 'ReqTRES': 'cpu=8,mem=56G,gres/gpu=1'}
    return ' '.join(k + '=' + v for k, v in values.items()) + ' ' + ','.join(k + '=' + v for k, v in cell['environment'].items())


def test_slurm2511_display_audit_preserves_raw_and_rejects_resource_drift(launcher, cells):
    cell = cells[0]; record = scheduler_record(cell)
    assert launcher.audit_held_record(record, 12345, cell) == record
    assert launcher.audit_held_record(record.replace('NumNodes=1-1', 'NumNodes=1'), 12345, cell)
    for before, after in [('NumNodes=1-1', 'NumNodes=1-2'), ('NumNodes=1-1', 'NumNodes=1 NumNodes=2'),
        ('MinMemoryNode=56G', 'MinMemoryNode=32G'), ('ReqNodeList=node302', 'ReqNodeList=node205'),
        ('OAT_ZERO_SEED=70', 'OAT_ZERO_SEED=700'), ('OMP_NUM_THREADS=4', 'OMP_NUM_THREADS=1'),
        ('JobState=PENDING', 'JobState=RUNNING')]:
        with pytest.raises(ValueError):
            launcher.audit_held_record(record.replace(before, after), 12345, cell)


def test_ambiguous_submission_preserves_intent_and_forbids_retry(launcher, cells, tmp_path):
    calls = []
    def ambiguous(*args, **kwargs):
        calls.append(args); raise subprocess.TimeoutExpired(args[0], 120)
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.submit_one_held(cells[0], 0, directory=tmp_path, runner=ambiguous)
    assert json.loads((tmp_path / 'submission_000_result.json').read_text())['status'] == 'ambiguous_exception'
    with pytest.raises(ValueError, match='fresh output'):
        launcher.submit_one_held(cells[0], 0, directory=tmp_path, runner=ambiguous)
    assert len(calls) == 1


@pytest.mark.parametrize('microbatch', ['4', '8'])
def test_effective_shell_command_preserves_science_and_applies_profile(launcher, profile, tmp_path, microbatch):
    operations = tmp_path / 'ops'; operations.mkdir()
    for name in ('run_experiment.sh', 'train.sh', 'repo_env.sh', 'resolve_eval_cadence.py', 'validate_deepspeed_checkpoint.py'):
        source = launcher.HEALTHY_OPS / name
        target = operations / name
        target.write_bytes(launcher.runtime_bytes('ops/' + name, source.read_bytes())); target.chmod(0o755)
    template = next(t for t in launcher.templates() if t['domain'] == 'pantry_plan' and t['seed'] == 70)
    profile['profile_environment']['OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE'] = microbatch
    model = launcher.model_identity('3b')
    env = launcher.clean_submit_environment()
    env.update(launcher.environment(template, 'replay_maxrl', '3b', tmp_path, model, profile))
    env.update(OAT_ZERO_SOURCE_ROOT=str(ROOT / 'src'), OAT_ZERO_OPS_SNAPSHOT_ROOT=str(operations),
        OAT_ZERO_TRAIN_SCRIPT=str(operations / 'train.sh'), OAT_ZERO_DRY_RUN='1',
        SAVE_PATH=str(tmp_path / 'never_train'), MAXENT_GRPO_VAR_ROOT=str(tmp_path / 'var'))
    caches = ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'PIP_CACHE_DIR', 'TMPDIR', 'HF_HOME',
        'HUGGINGFACE_HUB_CACHE', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE',
        'HF_ASSETS_CACHE', 'TORCH_HOME', 'WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR',
        'WANDB_DATA_DIR', 'PYTHONPYCACHEPREFIX', 'TORCH_EXTENSIONS_DIR')
    env.update({key: str(tmp_path / 'cache' / key.lower()) for key in caches})
    result = subprocess.run(['bash', str(operations / 'run_experiment.sh')], env=env,
        capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith('[train] command:'))
    command = shlex.split(line.removeprefix('[train] command:'))
    def value(flag): return command[command.index(flag) + 1]
    for flag, expected in {'--pretrain': model['path'], '--learning_rate': '1e-07',
        '--lr_scheduler': 'cosine_with_min_lr', '--lr_warmup_ratio': '0.1', '--max_step_adjustment': '16.0',
        '--adam_beta_2': '0.999', '--eval_batch_size': '32', '--eval_steps': '96',
        '--train_batch_size_per_device': microbatch, '--train_batch_size': '16', '--num_samples': '16'}.items():
        assert value(flag) == expected
    assert '--adam_offload' not in command and '--activation_offloading' in command
    assert '--canonical-action-task' not in command
    assert not (tmp_path / 'never_train').exists()


def test_profile_must_bind_exact_runtime_before_plan_is_admissible(launcher, profile):
    snapshot = {'root': '/exact/runtime', 'sha256': 'a' * 64, 'inventory': {
        'src/oat_drgrpo/learner/grpo.py': {'sha256': 'b' * 64},
        'ops/train.sh': {'sha256': 'c' * 64}, 'ops/status.py': {'sha256': 'd' * 64}}}
    profile['runtime'] = {'snapshot_root': snapshot['root'], 'identity_sha256': snapshot['sha256'],
        'files_sha256': {'/exact/runtime/src/oat_drgrpo/learner/grpo.py': 'b' * 64,
                         '/exact/runtime/ops/train.sh': 'c' * 64}}
    launcher.verify_systems_runtime(profile, snapshot)
    for mutation in ('other_root', 'other_identity', 'different_learner', 'missing_wrapper'):
        changed = deepcopy(profile)
        if mutation == 'other_root': changed['runtime']['snapshot_root'] = '/other/runtime'
        elif mutation == 'other_identity': changed['runtime']['identity_sha256'] = '0' * 64
        elif mutation == 'different_learner': changed['runtime']['files_sha256']['/exact/runtime/src/oat_drgrpo/learner/grpo.py'] = '0' * 64
        else: del changed['runtime']['files_sha256']['/exact/runtime/ops/train.sh']
        with pytest.raises(ValueError): launcher.verify_systems_runtime(changed, snapshot)


def test_existing_e123_scheduler_job_prevents_duplicate_claim(launcher, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0,
        stdout='123|e123_level3_graph_drgrpo_s70|PENDING\n', stderr=''))
    with pytest.raises(ValueError, match='existing E123 campaign job'):
        launcher.assert_no_existing_campaign_jobs()


def test_already_published_identical_runtime_is_verified_without_rewriting(launcher, tmp_path, monkeypatch):
    snapshot = {'root': str(tmp_path)}
    calls = []
    monkeypatch.setattr(launcher, 'verify_snapshot', lambda value: calls.append(value))
    launcher.publish_snapshot(snapshot)
    assert calls == [snapshot]


def test_science_projection_is_identical_across_all_registered_cells(launcher, cells):
    assert {tuple(launcher.science_environment(c['environment']).items()) for c in cells} == {
        tuple(launcher.expected_science_environment().items())}
