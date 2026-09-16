"""Science, admission and once-only scheduler boundaries for E122."""
from __future__ import annotations
import importlib.util
import json
import os
import pwd
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'ops/exp_scaling/launch_e122_level3_factorial.py'

@pytest.fixture(scope='module')
def launcher():
    spec = importlib.util.spec_from_file_location('e122_test_launcher', PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

@pytest.fixture(autouse=True)
def clear_process_admission_cache(launcher, monkeypatch):
    monkeypatch.setattr(launcher, '_ADMISSION_CACHE', None)

@pytest.fixture(scope='module')
def cells(launcher):
    return launcher.planned_cells('3b', '/reviewed/new/snapshot', {'path': '/frozen/qwen3b'})

def test_complete_new_factorial_preserves_effective_e119_science(launcher, cells):
    assert len(cells) == 100
    assert {(c['domain'], c['arm'], c['seed']) for c in cells} == {
        (d, a, s) for d in launcher.DOMAINS for a in launcher.ARMS for s in range(43, 48)}
    for cell in cells:
        env = cell['environment']
        expected = {'OAT_ZERO_LEARNING_RATE': '2e-07', 'OAT_ZERO_LR_SCHEDULER': 'constant',
            'OAT_ZERO_LR_WARMUP_RATIO': '0.0', 'OAT_ZERO_ADAM_BETA_1': '0.9', 'OAT_ZERO_ADAM_BETA_2': '0.95',
            'OAT_ZERO_MAX_NORM': '1.0', 'OAT_ZERO_BETA': '0.0', 'OAT_ZERO_NUM_SAMPLES': '16',
            'OAT_ZERO_NUM_PPO_EPOCHS': '1', 'OAT_ZERO_NUM_PROMPT_EPOCH': '8', 'OAT_ZERO_MAX_TRAIN': '384',
            'OAT_ZERO_ROLLOUT_BATCH_SIZE': '1', 'OAT_ZERO_TRAIN_BATCH_SIZE': '16',
            'OAT_ZERO_EVAL_BATCH_SIZE': '64', 'OAT_ZERO_EVAL_PROMPT_INTERVAL': '96',
            'OAT_ZERO_ALLOW_SPARSE_EVAL': '0', 'OAT_ZERO_PROMPT_MAX_LENGTH': '1024',
            'OAT_ZERO_GENERATE_MAX_LENGTH': '192', 'OAT_ZERO_EVAL_GENERATE_MAX_LENGTH': '192',
            'OAT_ZERO_MAX_MODEL_LEN': '2048', 'OAT_ZERO_TEMPERATURE': '1.0', 'OAT_ZERO_TOP_P': '1.0'}
        assert {k: env[k] for k in expected} == expected
        assert env['OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE'] == ('4' if cell['domain'] == 'pantry_plan' else '1')
        assert env['OAT_ZERO_RESUME_STEPS'] == ('96' if cell['domain'] == 'pantry_plan' else '192')
        assert env['OAT_ZERO_SAVE_STEPS'] == env['OAT_ZERO_RESUME_STEPS']
        assert cell['target_steps'] == 3072

def test_only_factorial_objectives_differ_within_paired_cells(launcher, cells):
    changing = {'OAT_ZERO_VARIANT', 'OAT_ZERO_MAXRL_TASK_OBJECTIVE',
                'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY', 'SAVE_PATH', 'RUN_STAMP'}
    for domain in launcher.DOMAINS:
        pair = [c for c in cells if c['domain'] == domain and c['seed'] == 43]
        assert len({tuple((k, v) for k, v in c['environment'].items() if k not in changing) for c in pair}) == 1
        for cell in pair:
            env, arm = cell['environment'], cell['arm']
            assert env['OAT_ZERO_MAXRL_TASK_OBJECTIVE'] == ('1' if 'maxrl' in arm else '0')
            assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY'] == ('0' if arm.startswith('replay_') else '1')
            assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE'] == 'verified_likelihood_per_rollout'
            assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA'] == '0.1'
            for key in ('OAT_ZERO_MAXENT_ALPHA', 'OAT_ZERO_SEMANTIC_SHANNON_COEF',
                        'OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA', 'OAT_ZERO_POLICY_ENTROPY_COEF'):
                assert float(env[key]) == 0

def test_dataset_paths_and_pantry_support_mask_repair(launcher, cells):
    for cell in cells:
        env = cell['environment']; directory = launcher.DOMAIN_DIR[cell['domain']]
        assert env['OAT_ZERO_PROMPT_DATA'] == str(launcher.DATA_ROOT / directory / 'train')
        assert env['OAT_ZERO_EVAL_DATA'] == str(launcher.DATA_ROOT / directory / 'eval')
        assert '/dev' not in env['OAT_ZERO_PROMPT_DATA'] + env['OAT_ZERO_EVAL_DATA']
        assert env['OAT_ZERO_CANONICAL_ACTION_TASK'] == 'none'
        assert env['OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT'] == '3'
        assert all(env[k] == '0' for k in ('OAT_ZERO_CANONICAL_GRAPH_ACTIONS',
            'OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING', 'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING'))
        assert cell['run_dir'] == str(launcher.ROOT / 'var/data/e122_level3_factorial' /
            cell['domain'] / cell['arm'] / f's{cell["seed"]}')
        assert '--hold' in cell['command']

def test_model_scaling_changes_only_model_and_offload_exports(launcher, cells):
    small = launcher.planned_cells('05b', '/reviewed/new/snapshot', {'path': '/frozen/qwen05b'})
    changed = {'OAT_ZERO_MODEL', 'OAT_ZERO_PRETRAIN', 'OAT_ZERO_ADAM_OFFLOAD', 'OAT_ZERO_ACTIVATION_OFFLOADING'}
    for large_cell, small_cell in zip(cells, small):
        large, little = large_cell['environment'], small_cell['environment']
        assert {k: v for k, v in large.items() if k not in changed} == {k: v for k, v in little.items() if k not in changed}
        assert large['OAT_ZERO_ADAM_OFFLOAD'] == large['OAT_ZERO_ACTIVATION_OFFLOADING'] == '1'
        assert large['OAT_ZERO_MAX_RESUME_NUM'] == large['OAT_ZERO_MAX_EXPORT_NUM'] == '1'
        assert large['OAT_ZERO_EXPORT_STEPS'] == '0'
        assert large['OAT_ZERO_PRUNE_RESUME_ON_SUCCESS'] == '1'
    for choice in ('05b', '3b'):
        profile = launcher.storage_profile(choice); observed = profile['evidence']
        assert profile['peak_bytes'] > 2 * (observed['model_state_bytes'] + observed['optimizer_state_bytes'])
        assert profile['terminal_bytes'] > observed['terminal_weights_bytes']
        assert profile['measurement_verified'] is True

def test_runtime_amendments_change_only_new_bytes(launcher):
    for relative in ('ops/run_experiment.sh', 'ops/train.sh', 'ops/slurm/train_node302.slurm'):
        source = launcher.HEALTHY_OPS / relative.removeprefix('ops/')
        before = source.read_bytes(); after = launcher.runtime_bytes(relative, before)
        assert source.read_bytes() == before and after != before
        if relative != 'ops/slurm/train_node302.slurm':
            assert b'e119_level2_' not in after and b'e122_level3_' in after
        else:
            assert b'${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out' in after
    assert launcher.runtime_bytes('src/oat_drgrpo/learner/grpo.py', b'untouched') == b'untouched'

def test_missing_confirmation_preview_pending_but_admission_fails(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'REPORT', tmp_path / 'missing.json')
    assert launcher.admission_proof(required=False)['status'] == 'awaiting_admission'
    with pytest.raises(ValueError, match='awaits canonical'):
        launcher.admission_proof()

def test_dataset_identity_drift_fails_even_preview(launcher, tmp_path, monkeypatch):
    changed = tmp_path / 'identity.json'; changed.write_text('{}')
    monkeypatch.setattr(launcher, 'IDENTITY', changed)
    with pytest.raises(ValueError, match='identity drift'):
        launcher.admission_proof(required=False)

def test_completed_failed_or_partial_confirmation_cannot_admit(launcher, tmp_path, monkeypatch):
    path = tmp_path / 'report.json'; path.write_text('{}'); monkeypatch.setattr(launcher, 'REPORT', path)
    evidence = {'status': 'outside_fixed_reference_tolerance', 'confirmation_match_verified': False, 'report': {}}
    monkeypatch.setitem(sys.modules, 'audit_modebench_level3_v3', SimpleNamespace(validate_confirmation_report=lambda **kw: evidence))
    with pytest.raises(ValueError, match='all five authenticated'):
        launcher.admission_proof()
    evidence.update(status='matched_fixed_reference', confirmation_match_verified=True, report={'all_five_domains_complete': False})
    with pytest.raises(ValueError, match='all five authenticated'):
        launcher.admission_proof()

def test_success_requires_authenticator_and_exact_dataset(launcher, tmp_path, monkeypatch):
    path = tmp_path / 'report.json'; path.write_text('{}'); monkeypatch.setattr(launcher, 'REPORT', path)
    report = {'all_five_domains_complete': True, 'errors': {}, 'missing_domains': [],
        'domains': {d: {'observed_approximate_match': True, 'within_tolerance': {'pass1': True, 'pass8': True}}
                    for d in launcher.DOMAIN_DIR.values()},
        'dataset': {'path': str(launcher.IDENTITY), 'sha256': launcher.IDENTITY_SHA256,
                    'dataset_root': str(launcher.DATA_ROOT), 'split_sizes': launcher.common.SPLITS},
        'information_boundary': {'reference_semantics': 'fixed_measured_level1_benchmark',
            'adaptive_confirmation_round': 2, 'historical_level1_confirmation_used_as_fixed_reference': True,
            'statistical_equivalence_claimed': False}}
    evidence = {'path': str(path), 'sha256': launcher.digest(path), 'status': 'matched_fixed_reference',
        'confirmation_match_verified': True, 'report': report,
        'files_sha256': {str(path): launcher.digest(path)}, 'directory_files': {}}
    calls = []
    def authenticate(**kwargs):
        calls.append(kwargs); return evidence
    monkeypatch.setitem(sys.modules, 'audit_modebench_level3_v3', SimpleNamespace(validate_confirmation_report=authenticate))
    assert launcher.admission_proof()['sha256'] == launcher.digest(path) and calls == [{'path': path}]
    monkeypatch.setattr(launcher, '_ADMISSION_CACHE', None)
    report['dataset']['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='dataset or adaptive'):
        launcher.admission_proof()

def test_ambiguous_submission_keeps_intent_result_and_forbids_retry(launcher, cells, tmp_path):
    calls = []
    def ambiguous(*args, **kwargs):
        calls.append(args); raise subprocess.TimeoutExpired(args[0], 120)
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.submit_one_held(cells[0], 0, directory=tmp_path, runner=ambiguous)
    assert json.loads((tmp_path / 'submission_000_result.json').read_text())['status'] == 'ambiguous_exception'
    assert (tmp_path / 'submission_000_intent.json').is_file()
    with pytest.raises(ValueError, match='fresh output'):
        launcher.submit_one_held(cells[0], 0, directory=tmp_path, runner=ambiguous)
    assert len(calls) == 1

def test_malformed_scheduler_success_preserved_without_retry(launcher, cells, tmp_path):
    def malformed(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout='accepted without unique ID', stderr='')
    with pytest.raises(ValueError, match='ambiguous or failed'):
        launcher.submit_one_held(cells[0], 0, directory=tmp_path, runner=malformed)
    assert json.loads((tmp_path / 'submission_000_result.json').read_text())['returncode'] == 0

def scheduler_record(cell):
    r = cell['resources']
    values = {'JobId': '12345', 'JobName': cell['run_stamp'], 'JobState': 'PENDING', 'Reason': 'JobHeldUser',
        'Account': r['account'], 'Partition': r['partition'], 'QOS': r['expected_qos'], 'NumCPUs': str(r['cpus']), 'NumNodes': '1', 'NumTasks': '1',
        'Requeue': '1', 'Nice': '0', 'UserId': f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})',
        'WorkDir': str(ROOT), 'Command': cell['command'][-1], 'RunTime': '00:00:00', 'Restarts': '0', 'MinMemoryNode': str(r['memory_gib']) + 'G', 'TimeLimit': r['walltime'],
        'ExcNodeList': r['exclude'] or '(null)', 'ReqNodeList': 'node[205-207,302]', 'ReqTRES': 'cpu=16,mem=128G,gres/gpu=1'}
    return ' '.join(k + '=' + v for k, v in values.items()) + ' ' + ','.join(k + '=' + v for k, v in cell['environment'].items())

def test_held_audit_detects_resources_objective_and_seed_prefix_drift(launcher, cells):
    cell = cells[0]; record = scheduler_record(cell)
    assert launcher.audit_held_record(record, 12345, cell) == record
    for before, after in [('MinMemoryNode=128G', 'MinMemoryNode=64G'), ('NumNodes=1', 'NumNodes=2'),
                          ('NumTasks=1', 'NumTasks=2'), ('Restarts=0', 'Restarts=1'), ('QOS=medium', 'QOS=none'),
                          ('Command=/reviewed/new/snapshot', 'Command=/wrong'), ('JobState=PENDING', 'JobState=RUNNING'),
                          ('OAT_ZERO_SEED=43', 'OAT_ZERO_SEED=430'),
                          ('OAT_ZERO_MAXRL_TASK_OBJECTIVE=0', 'OAT_ZERO_MAXRL_TASK_OBJECTIVE=1')]:
        with pytest.raises(ValueError):
            launcher.audit_held_record(record.replace(before, after), 12345, cell)

def test_submit_environment_removes_unregistered_overrides(launcher, monkeypatch):
    monkeypatch.setenv('OAT_ZERO_SEMANTIC_SHANNON_COEF', '999')
    monkeypatch.setenv('SBATCH_DEPENDENCY', 'afterok:123'); monkeypatch.setenv('SAVE_PATH', '/wrong')
    env = launcher.clean_submit_environment()
    assert 'OAT_ZERO_SEMANTIC_SHANNON_COEF' not in env and 'SBATCH_DEPENDENCY' not in env and 'SAVE_PATH' not in env

def test_reviewed_draft_hash_and_explicit_model_required(launcher, tmp_path):
    draft = tmp_path / 'draft.json'; draft.write_text('{}')
    with pytest.raises(ValueError, match='reviewed E122 draft hash'):
        launcher.prepare(draft, '0' * 64, '3b')
    with pytest.raises(SystemExit):
        launcher.main(['--dry-run'])
    assert "['scontrol', 'release'" not in PATH.read_text()


def test_zero_scheduler_job_id_is_rejected(launcher, cells, tmp_path):
    def zero(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout='0\n', stderr='')
    with pytest.raises(ValueError, match='ambiguous or failed'):
        launcher.submit_one_held(cells[0], 0, directory=tmp_path, runner=zero)

@pytest.mark.parametrize('choice', ['05b', '3b'])
def test_effective_shell_command_binds_selected_model_and_e119_science(launcher, tmp_path, choice):
    import shlex
    operations = tmp_path / 'ops'; operations.mkdir()
    for name in ('run_experiment.sh', 'train.sh', 'repo_env.sh', 'resolve_eval_cadence.py', 'validate_deepspeed_checkpoint.py'):
        source = launcher.HEALTHY_OPS / name
        target = operations / name
        target.write_bytes(launcher.runtime_bytes('ops/' + name, source.read_bytes()))
        target.chmod(0o755)
    template = next(t for t in launcher.templates() if t['domain'] == 'pantry_plan' and t['seed'] == 43)
    model = launcher.model_identity(choice)
    env = launcher.clean_submit_environment()
    env.update(launcher.environment(template, 'replay_maxrl', choice, tmp_path, model))
    env.update(OAT_ZERO_SOURCE_ROOT=str(ROOT / 'src'), OAT_ZERO_OPS_SNAPSHOT_ROOT=str(operations),
        OAT_ZERO_TRAIN_SCRIPT=str(operations / 'train.sh'), OAT_ZERO_DRY_RUN='1',
        SAVE_PATH=str(tmp_path / 'never_train'), MAXENT_GRPO_VAR_ROOT=str(tmp_path / 'var'))
    caches = ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'PIP_CACHE_DIR', 'TMPDIR', 'HF_HOME',
        'HUGGINGFACE_HUB_CACHE', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE',
        'HF_ASSETS_CACHE', 'TORCH_HOME', 'WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR',
        'WANDB_DATA_DIR', 'PYTHONPYCACHEPREFIX', 'TORCH_EXTENSIONS_DIR')
    env.update({key: str(tmp_path / 'cache' / key.lower()) for key in caches})
    result = subprocess.run(['bash', str(operations / 'run_experiment.sh')], env=env,
        capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith('[train] command:'))
    command = shlex.split(line.removeprefix('[train] command:'))
    def value(flag):
        return command[command.index(flag) + 1]
    assert env['OAT_ZERO_MODEL'] == ('qwen2.5-3b-instruct' if choice == '3b' else 'qwen2.5-0.5b-instruct')
    assert value('--pretrain') == model['path']
    assert value('--learning_rate') == '2e-07'
    assert value('--lr_scheduler') == 'constant' and value('--lr_warmup_ratio') == '0.0'
    assert value('--adam_beta_1') == '0.9' and value('--adam_beta_2') == '0.95'
    assert value('--eval_batch_size') == '64' and value('--eval_steps') == '96'
    assert value('--prompt_max_length') == '1024' and value('--generate_max_length') == '192'
    assert value('--resume-steps') == '96'
    assert ('--adam_offload' in command) is (choice == '3b')
    assert ('--activation_offloading' in command) is (choice == '3b')
    assert '--canonical-action-task' not in command
    assert not (tmp_path / 'never_train').exists()


def test_reviewed_borrowing_route_preserves_priority_resources_and_requeue(launcher, cells):
    for cell in cells:
        resource = cell['resources']
        assert resource['partition'] == 'lowprio' and resource['account'] == 'allcs'
        assert resource['nodes'] == launcher.SAFE_NODES
        assert resource['exclude'] == launcher.PVL == ''
        # Every pool member carries at least 48 GiB of bf16-capable GPU; the
        # registered route never admits a 24 GiB card.
        assert set(resource['nodes'].split(',')) == {
            'node101', 'node103', 'node104', 'node205', 'node206', 'node207',
            'node208', 'node302', 'node403', 'node805'}
        assert resource['walltime'] == '1-12:00:00'
        assert resource['cpus'] == 16 and resource['memory_gib'] == 128
        assert '--nice=0' in cell['command'] and '--requeue' in cell['command']
        assert resource['expected_qos'] == 'medium'
        assert not any(part.startswith('--qos') for part in cell['command'])
        assert cell['environment']['OAT_ZERO_WATCHDOG_REQUEUE'] == '1'
        assert launcher.audit_held_record(scheduler_record(cell), 12345, cell)
    altered = scheduler_record(cells[0]).replace('Partition=lowprio', 'Partition=cs')
    with pytest.raises(ValueError):
        launcher.audit_held_record(altered, 12345, cells[0])


def test_registered_campaign_model_blocks_3b_mutation_but_allows_preview(launcher, tmp_path):
    assert launcher.CAMPAIGN_MODEL_CHOICE == '05b'
    assert len(launcher.planned_cells('3b', '/preview', {'path': '/frozen/3b'})) == 100
    draft = tmp_path / 'draft.json'; draft.write_text('{}')
    with pytest.raises(ValueError, match='3B preview only'):
        launcher.prepare(draft, launcher.digest(draft), '3b')
    with pytest.raises(ValueError, match='3B preview only'):
        launcher.verify_plan(path=draft, expected_sha256=launcher.digest(draft), model_choice='3b')

@pytest.fixture
def small_admission_certificate(launcher, tmp_path, monkeypatch):
    report_path = tmp_path / 'canonical_report.json'; report_path.write_text('{}')
    source_root = tmp_path / 'scientific_sources'; source_root.mkdir()
    source = source_root / 'registered.py'; source.write_text('frozen scientific input\n')
    monkeypatch.setattr(launcher, 'REPORT', report_path)
    report = {'all_five_domains_complete': True, 'errors': {}, 'missing_domains': [],
        'domains': {d: {'observed_approximate_match': True, 'within_tolerance': {'pass1': True, 'pass8': True}}
                    for d in launcher.DOMAIN_DIR.values()},
        'dataset': {'path': str(launcher.IDENTITY), 'sha256': launcher.IDENTITY_SHA256,
                    'dataset_root': str(launcher.DATA_ROOT), 'split_sizes': launcher.common.SPLITS},
        'information_boundary': {'reference_semantics': 'fixed_measured_level1_benchmark',
            'adaptive_confirmation_round': 2, 'historical_level1_confirmation_used_as_fixed_reference': True,
            'statistical_equivalence_claimed': False}}
    evidence = {'path': str(report_path), 'sha256': launcher.digest(report_path),
        'status': 'matched_fixed_reference', 'confirmation_match_verified': True, 'report': report,
        'files_sha256': {str(p): launcher.digest(p) for p in (report_path, source)},
        'directory_files': {str(source_root): [str(source)]}}
    calls = []
    def authenticate(**kwargs):
        calls.append(kwargs); return evidence
    monkeypatch.setitem(sys.modules, 'audit_modebench_level3_v3', SimpleNamespace(validate_confirmation_report=authenticate))
    return SimpleNamespace(evidence=evidence, calls=calls, report=report_path, source=source, tree=source_root)


def test_admission_cache_full_validation_once_and_fresh_pins_every_reuse(launcher, small_admission_certificate, monkeypatch):
    certificate = small_admission_certificate
    checked = []
    original = launcher.common.verify_pins
    def verify(files, trees=None):
        checked.append((dict(files), dict(trees or {}))); return original(files, trees)
    monkeypatch.setattr(launcher.common, 'verify_pins', verify)
    first = launcher.admission_proof()
    second = launcher.admission_proof()
    third = launcher.admission_proof()
    assert first == second == third
    assert certificate.calls == [{'path': certificate.report}]
    assert len(checked) == 2
    assert all(files == certificate.evidence['files_sha256'] and trees == certificate.evidence['directory_files']
               for files, trees in checked)


@pytest.mark.parametrize('drift', ['report', 'source', 'added_file', 'removed_file'])
def test_admission_cache_rejects_byte_and_inventory_drift_without_revalidation(launcher, small_admission_certificate, drift):
    certificate = small_admission_certificate
    launcher.admission_proof()
    if drift == 'report':
        certificate.report.write_text('{"changed":true}')
    elif drift == 'source':
        certificate.source.write_text('altered scientific input\n')
    elif drift == 'added_file':
        (certificate.tree / 'unregistered.py').write_text('unregistered')
    else:
        certificate.source.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.admission_proof()
    assert certificate.calls == [{'path': certificate.report}]


def test_returned_proof_or_validator_object_cannot_mutate_process_cache(launcher, small_admission_certificate):
    certificate = small_admission_certificate
    first = launcher.admission_proof()
    first['files_sha256'].clear(); first['directory_files'].clear()
    certificate.evidence['report']['dataset']['sha256'] = '0' * 64
    again = launcher.admission_proof()
    assert len(again['files_sha256']) == 2 and len(again['directory_files']) == 1
    assert len(certificate.calls) == 1


def test_cache_reuse_still_checks_explicit_admission_provenance(launcher, small_admission_certificate):
    launcher.admission_proof()
    launcher._ADMISSION_CACHE['report']['information_boundary']['reference_semantics'] = 'relabeled_development'
    with pytest.raises(ValueError, match='dataset or adaptive'):
        launcher.admission_proof()


def test_failed_full_validation_does_not_seed_cache(launcher, small_admission_certificate):
    certificate = small_admission_certificate
    certificate.evidence['report']['domains']['graph_coloring']['observed_approximate_match'] = False
    with pytest.raises(ValueError, match='all five authenticated'):
        launcher.admission_proof()
    assert launcher._ADMISSION_CACHE is None
    certificate.evidence['report']['domains']['graph_coloring']['observed_approximate_match'] = True
    launcher.admission_proof()
    assert len(certificate.calls) == 2


def test_canonical_report_must_be_in_cached_file_inventory(launcher, small_admission_certificate):
    certificate = small_admission_certificate
    del certificate.evidence['files_sha256'][str(certificate.report)]
    with pytest.raises(ValueError, match='canonical report must remain bound'):
        launcher.admission_proof()
    assert launcher._ADMISSION_CACHE is None
