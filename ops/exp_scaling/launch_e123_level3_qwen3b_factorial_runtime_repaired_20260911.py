#!/usr/bin/env python3
"""Prepare E123 and submit its 100 cells held; never release training jobs.

The default preview is read-only. Publishing a runtime requires a reviewed draft
hash. Submission requires the published plan hash and a fresh, exclusive claim;
ambiguous scheduler responses retain their intent and can never be retried here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import pwd
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    sys.path.insert(0, str(ROOT / directory))
import launch_e119_level2_qwen05b_factorial as e119
import launch_e80r1_qwen3b_aligned_verified_replay as e80
import modebench_level3_v3_common as common

SOURCE = Path(__file__).resolve()
TEST = ROOT / 'tests/test_e123_level3_factorial.py'
PROTOCOL = ROOT / 'paper/preregistration/e123_level3_qwen3b_factorial_20260909.md'
LEDGER = ROOT / 'var/artifacts/e123_level3_factorial_jobs.json'
PLAN = ROOT / 'var/artifacts/e123_level3_factorial_plan.json'
HERE = ROOT / 'var/artifacts/e123_level3_factorial'
CLAIM = HERE / 'submission_claim.json'
DATA_ROOT = ROOT / 'var/data/modebench_level3_matched_v3'
IDENTITY = DATA_ROOT / 'identity.json'
IDENTITY_SHA256 = '890d7697af7789e0ae53c803586ec7f722685b7eb2239643175a1933fa45650d'
REPORT = ROOT / 'var/artifacts/modebench_level3_v3/confirmation/confirmation_report.json'
TEMPLATES = ROOT / 'var/artifacts/e72_frontier_source_runs.json'
HEALTHY_OPS = ROOT / 'var/artifacts/source_snapshots/e76_tuned_scale_50d36295558a8958/ops'
DOMAINS, ARMS, SEEDS = e119.DOMAINS, e119.ARMS, e80.SEEDS
DOMAIN_TAGS, DOMAIN_DIR = e119.DOMAIN_TAGS, e119.DOMAIN_DIR
PASSES, TRAIN_ROWS, EVAL_ROWS, TARGET_STEPS = 8, 384, 128, 3072
EVALUATION_INTERVAL = 96
CHECKPOINT_INTERVALS = {domain: 96 if domain == 'pantry_plan' else 192 for domain in DOMAINS}
MODEL_REVISIONS = {'05b': '7ae557604adf67be50417f59c2c2f167def9a775',
                   '3b': 'aa8e72537993ba99e69dfaafa59ed015b17504d1'}
MODEL_NAMES = {'05b': 'Qwen2.5-0.5B-Instruct', '3b': 'Qwen2.5-3B-Instruct'}
CAMPAIGN_MODEL_CHOICE = '3b'
PVL = 'node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]'
SAFE_NODES = 'node302'
PLAN_SCHEMA = 'e123_level3_factorial_plan_v1'
LEDGER_SCHEMA = 'e123_level3_factorial_jobs_v1'
require, read, digest, sha = common.require, common.read, common.digest, common.sha
atomic_new = common.atomic_new
# Valid only inside this process, after full successful canonical authentication.
_ADMISSION_CACHE = None


def now():
    return datetime.now(timezone.utc).isoformat()


def run_stamp(domain, arm, seed):
    return f'e123_level3_{DOMAIN_TAGS[domain]}_{arm}_s{seed}'


def run_dir(domain, arm, seed):
    return ROOT / 'var/data/e123_level3_factorial' / domain / arm / f's{seed}'


def templates():
    """Load only the frozen schedule metadata, without requiring old admissions."""
    runs = [run for run in read(TEMPLATES)['runs'] if run['arm'] == 'xgrpo'
            and run['domain'] in DOMAINS and int(run['seed']) == 43]
    require(len(runs) == 5 and {run['domain'] for run in runs} == set(DOMAINS),
            'exact five frozen E72 domain schedule templates required')
    return [e119.base.reseed(run, seed) for run in sorted(runs, key=lambda row: row['domain']) for seed in SEEDS]



def model_identity(choice):
    require(choice in MODEL_NAMES, 'explicit model choice 05b or 3b required')
    from evaluate_modebench_level3 import model_identity as identify
    path = ROOT / 'var/cache/huggingface/transformers' / ('models--Qwen--' + MODEL_NAMES[choice]) / 'snapshots' / MODEL_REVISIONS[choice]
    return identify(path, choice)


def admission_proof(*, required=True):
    """Authenticate once per process; rehash every frozen input on each reuse."""
    global _ADMISSION_CACHE
    require(digest(IDENTITY) == IDENTITY_SHA256, 'E123 Level-3 dataset identity drift')
    identity = read(IDENTITY)
    require(identity['split_sizes'] == {'train': 384, 'dev': 128, 'eval': 128}
            and set(identity['domains']) == set(DOMAIN_DIR.values()), 'E123 exact all-five dataset required')
    if _ADMISSION_CACHE is None:
        if not REPORT.is_file():
            require(not required, 'E123 awaits canonical completed V3 confirmation')
            return {'status': 'awaiting_admission', 'path': str(REPORT), 'sha256': None}
        from audit_modebench_level3_v3 import validate_confirmation_report
        evidence = validate_confirmation_report(path=REPORT)
    else:
        # Never reload a modified proof, refit a decision, or skip inventory checks.
        require(_ADMISSION_CACHE['path'] == str(REPORT), 'cached canonical report path changed')
        common.verify_pins(_ADMISSION_CACHE['files_sha256'], _ADMISSION_CACHE['directory_files'])
        evidence = _ADMISSION_CACHE
    report = evidence['report']
    require(evidence['status'] == 'matched_fixed_reference' and evidence['confirmation_match_verified'] is True
            and report['all_five_domains_complete'] is True and report['errors'] == {}
            and report['missing_domains'] == [] and set(report['domains']) == set(DOMAIN_DIR.values())
            and all(cell['observed_approximate_match'] is True
                    and all(cell['within_tolerance'].values()) for cell in report['domains'].values()),
            'E123 requires all five authenticated fixed-reference comparisons to pass')
    require(report['dataset']['dataset_root'] == str(DATA_ROOT)
            and report['dataset']['path'] == str(IDENTITY)
            and report['dataset']['sha256'] == IDENTITY_SHA256
            and report['dataset']['split_sizes'] == common.SPLITS
            and report['information_boundary']['reference_semantics'] == 'fixed_measured_level1_benchmark'
            and report['information_boundary']['adaptive_confirmation_round'] == 2
            and report['information_boundary']['historical_level1_confirmation_used_as_fixed_reference'] is True
            and report['information_boundary']['statistical_equivalence_claimed'] is False,
            'E123 admission dataset or adaptive fixed-reference provenance differs')
    require(evidence['path'] == str(REPORT)
            and evidence['files_sha256'].get(str(REPORT)) == evidence['sha256'] == digest(REPORT),
            'canonical report must remain bound in the authenticated file inventory')
    if _ADMISSION_CACHE is None:
        _ADMISSION_CACHE = deepcopy(evidence)
    # Plans and callers cannot mutate the trusted process-local proof.
    return deepcopy({key: evidence[key] for key in ('path', 'sha256', 'status', 'files_sha256', 'directory_files')})


SYSTEMS_ENV_KEYS = {
    'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE', 'OAT_ZERO_ADAM_OFFLOAD',
    'OAT_ZERO_ACTIVATION_OFFLOADING', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'OAT_ZERO_EVAL_BATCH_SIZE', 'OAT_ZERO_VLLM_GPU_RATIO',
}

SCIENCE_ENV_KEYS = (
    'OAT_ZERO_LEARNING_RATE', 'OAT_ZERO_LR_SCHEDULER', 'OAT_ZERO_LR_WARMUP_RATIO',
    'OAT_ZERO_MAX_STEP_ADJUSTMENT', 'OAT_ZERO_ADAM_BETA_1', 'OAT_ZERO_ADAM_BETA_2',
    'OAT_ZERO_L2', 'OAT_ZERO_BETA', 'OAT_ZERO_MAX_NORM', 'OAT_ZERO_NUM_PPO_EPOCHS',
    'OAT_ZERO_NUM_SAMPLES', 'OAT_ZERO_TRAIN_BATCH_SIZE', 'OAT_ZERO_ROLLOUT_BATCH_SIZE',
    'OAT_ZERO_NUM_PROMPT_EPOCH', 'OAT_ZERO_MAX_PROMPT_EPOCHS', 'OAT_ZERO_MAX_TRAIN',
    'OAT_ZERO_PROMPT_MAX_LENGTH', 'OAT_ZERO_GENERATE_MAX_LENGTH',
    'OAT_ZERO_EVAL_GENERATE_MAX_LENGTH', 'OAT_ZERO_MAX_MODEL_LEN',
    'OAT_ZERO_TEMPERATURE', 'OAT_ZERO_TOP_P', 'OAT_ZERO_EVAL_PROMPT_INTERVAL',
    'OAT_ZERO_EVAL_MODE_COVERAGE_K', 'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS',
    'OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE', 'OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P',
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE', 'OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA',
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY', 'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP',
    'OAT_ZERO_CRITIC_TYPE', 'OAT_ZERO_CANONICAL_ACTION_TASK', 'OAT_ZERO_CANONICAL_GRAPH_ACTIONS',
)


def science_environment(env):
    """Common registered science, excluding physical resources and factorial factors."""
    require(all(key in env for key in SCIENCE_ENV_KEYS), 'complete E123 scientific environment required')
    return {key: env[key] for key in SCIENCE_ENV_KEYS}


def expected_profile_model():
    path = ROOT / 'var/cache/huggingface/transformers' / ('models--Qwen--' + MODEL_NAMES['3b']) / 'snapshots' / MODEL_REVISIONS['3b']
    return {'choice': '3b', 'revision': MODEL_REVISIONS['3b'], 'path': str(path)}


def expected_science_environment():
    model = expected_profile_model()
    return science_environment(environment(templates()[0], ARMS[0], '3b', '/not-published', model))


SYSTEMS_CANDIDATE_KEYS = ('OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE', 'OMP_NUM_THREADS', 'OAT_ZERO_ADAM_OFFLOAD')
SYSTEMS_CANDIDATES = {
    name: dict(zip(SYSTEMS_CANDIDATE_KEYS, values))
    for name, values in {
        'gpu_mb4_omp4': ('4', '4', '0'),
        'gpu_mb8_omp4': ('8', '4', '0'),
        'gpu_adam': ('1', '1', '0'),
    }.items()
}
BENCHMARK_CANDIDATES = frozenset(('baseline', 'omp4', 'mb4', 'mb8', 'gpu_adam', 'gpu_mb4_omp4', 'gpu_mb8_omp4'))
COMBINED_CANDIDATES = ('gpu_mb4_omp4', 'gpu_mb8_omp4')


def fallback_evidence(path, expected):
    """Authenticate bounded operational evidence without changing any file."""
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.stat().st_size <= 32 * 1024**2,
            'fallback evidence requires a bounded absolute regular file')
    require(re.fullmatch(r'[0-9a-f]{64}', expected or '') is not None
            and digest(path) == expected, 'fallback evidence hash differs')
    value = read(path)
    require(isinstance(value, dict) and digest(path) == expected, 'fallback evidence changed while reading')
    return value


def verify_fallback_qualification(profile, selected_candidate):
    """MB1 may follow only authenticated rejection of both combined profiles."""
    proof = profile.get('fallback_qualification')
    if profile['candidate'] != 'gpu_adam':
        require(proof is None or (isinstance(proof, dict)
                and proof.get('policy') == 'combined_profiles_first_v1'
                and proof.get('fallback_used') is False), 'combined profile mislabeled as fallback')
        return
    require(isinstance(proof, dict) and proof.get('policy') == 'combined_profiles_first_v1'
            and proof.get('fallback_used') is True, 'MB1 requires explicit combined-first fallback evidence')
    suite = fallback_evidence(proof.get('suite_status_path', ''), proof.get('suite_status_sha256'))
    plan_sha = selected_candidate.get('plan_sha256')
    require(re.fullmatch(r'[0-9a-f]{64}', plan_sha or '') is not None
            and suite.get('schema') == 'e123_a100_benchmark_suite_v1'
            and suite.get('status') == 'qualifying_fallback'
            and suite.get('plan_sha256') == plan_sha,
            'fallback suite must be the same benchmark before fallback qualification')
    summaries = suite.get('candidates')
    require(isinstance(summaries, dict) and set(summaries) == BENCHMARK_CANDIDATES,
            'fallback requires all seven registered candidate measurements')
    for name, summary in summaries.items():
        require(isinstance(summary, dict), 'malformed candidate summary')
        receipt = fallback_evidence(summary.get('path', ''), summary.get('sha256'))
        require(receipt.get('schema') == 'e123_a100_candidate_result_v1'
                and receipt.get('candidate') == name and receipt.get('plan_sha256') == plan_sha
                and receipt.get('status') in ('passed', 'failed')
                and summary.get('status') == receipt['status'], 'fallback candidate receipt identity/status differs')
        if name == 'baseline':
            require(receipt['status'] == 'passed', 'fallback requires a passing same-plan baseline')
        if name == 'gpu_adam':
            require(receipt == selected_candidate, 'fallback selected measurement differs from seven-profile suite')
        if name not in COMBINED_CANDIDATES or receipt['status'] == 'failed':
            continue
        error = summary.get('qualification_error')
        require(isinstance(error, str) and bool(error.strip()),
                'a combined profile passed without a recorded qualification rejection')
        e2e_path, e2e_sha = summary.get('e2e_result_path'), summary.get('e2e_result_sha256')
        require(bool(e2e_path) == bool(e2e_sha), 'partial combined end-to-end evidence reference')
        if e2e_path:
            e2e = fallback_evidence(e2e_path, e2e_sha)
            require(e2e.get('schema') == 'e123_a100_e2e_result_v1'
                    and e2e.get('candidate') == name and e2e.get('plan_sha256') == plan_sha
                    and e2e.get('status') in ('passed', 'failed'), 'combined end-to-end evidence identity differs')


def systems_proof(path=None, expected_sha256=None, *, required=True):
    """Only measured, selected systems profiles may reach a held submission."""
    if path is None:
        require(not required, 'E123 awaits a measured passing A100 systems profile')
        return {'status': 'awaiting_benchmark', 'path': None, 'sha256': None}
    path = Path(path).resolve()
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha256 or ''), 'explicit systems profile file SHA256 required')
    require(digest(path) == expected_sha256, 'E123 systems profile file hash differs')
    profile = read(path)
    require(profile.get('schema') == 'e123_a100_selected_profile_v1'
            and profile.get('status') == 'passed' and profile.get('selection_status') == 'selected',
            'E123 requires a passing selected A100 profile')
    identity = {key: value for key, value in profile.items() if key != 'profile_sha256'}
    canonical = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(profile.get('profile_sha256') == canonical, 'E123 systems profile identity differs')
    env = profile.get('profile_environment', {})
    require(set(env).issubset(SYSTEMS_ENV_KEYS) and all(isinstance(value, str) for value in env.values()),
            'systems profile may change only registered physical runtime settings')
    selected = SYSTEMS_CANDIDATES.get(profile.get('candidate'))
    require(selected is not None and set(env) == set(SYSTEMS_CANDIDATE_KEYS) | {'OAT_ZERO_ACTIVATION_OFFLOADING'}
            and {key: env.get(key) for key in SYSTEMS_CANDIDATE_KEYS} == selected
            and env.get('OAT_ZERO_ACTIVATION_OFFLOADING') in ('0', '1'),
            'selected E123 profile must match an exact registered GPU Adam candidate')
    resource = profile.get('resources', {})
    require(resource.get('node') == SAFE_NODES and resource.get('gpus') == 1
            and isinstance(resource.get('cpus'), int) and 4 <= resource['cpus'] <= 8
            and isinstance(resource.get('memory_gib'), int) and 1 <= resource['memory_gib'] <= 128,
            'invalid measured node302 single-A100 resource profile')
    measured = profile.get('measurements', {})
    host, gpu = measured.get('host_peak_bytes', 0), measured.get('gpu_peak_bytes', 0)
    require(isinstance(host, (int, float)) and isinstance(gpu, (int, float)) and host > 0 and gpu > 0,
            'observed host and GPU memory peaks are required')
    require(resource['memory_gib'] * 1024**3 >= max(host * 1.2, host + 8 * 1024**3),
            'host-memory request lacks measured 20 percent / 8 GiB headroom')
    require(gpu <= 76 * 1024**3, 'A100 GPU peak lacks at least 4 GiB nominal headroom')
    require(profile.get('model') == expected_profile_model(), 'benchmark must use the exact pretrained Qwen3B model')
    require(profile.get('science_environment') == expected_science_environment(),
            'benchmark must bind the registered Qwen3B scientific environment')
    dataset = profile.get('dataset', {})
    require(dataset.get('identity_path') == str(IDENTITY) and dataset.get('identity_sha256') == IDENTITY_SHA256,
            'benchmark must bind the exact frozen Level-3 dataset')
    evidence = profile.get('evidence', {})
    require(all(evidence.get(key) is True for key in (
        'full_shape_smoke', 'own_checkpoint_resume', 'fixed_update_equivalence', 'end_to_end_smoke')),
        'full-shape training, own-checkpoint restore, update equivalence and end-to-end smoke must pass')
    candidate = Path(evidence.get('candidate_result_path', ''))
    require(candidate.is_file() and digest(candidate) == evidence.get('candidate_result_sha256'),
            'passing candidate result evidence hash differs')
    require(read(candidate).get('schema') == 'e123_a100_candidate_result_v1'
            and read(candidate).get('status') == 'passed', 'selected candidate result has not passed')
    require(read(candidate).get('candidate') == profile.get('candidate'), 'selected candidate identity differs')
    verify_fallback_qualification(profile, read(candidate))
    end_to_end = Path(evidence.get('e2e_result_path', ''))
    require(end_to_end.is_file() and digest(end_to_end) == evidence.get('e2e_result_sha256'),
            'end-to-end result evidence hash differs')
    require(read(end_to_end).get('status') == 'passed', 'whole-job actor/evaluation/checkpoint smoke has not passed')
    runtime = profile.get('runtime', {})
    require(isinstance(runtime.get('files_sha256'), dict) and runtime['files_sha256'],
            'benchmark runtime must pin the measured implementation')
    common.verify_pins(runtime['files_sha256'])
    require(digest(path) == expected_sha256, 'systems profile changed while verifying')
    return {'status': 'passed', 'path': str(path), 'sha256': expected_sha256, 'profile': profile}



def verify_systems_runtime(profile, snapshot):
    """Tie measured training code to the exact campaign runtime, including its wrapper."""
    runtime = profile['runtime']
    require(runtime.get('snapshot_root') == snapshot['root']
            and runtime.get('identity_sha256') == snapshot['sha256'],
            'measured runtime identity must equal the E123 campaign snapshot')
    for relative, item in snapshot['inventory'].items():
        if relative.startswith('src/oat_drgrpo/') or relative == 'ops/train.sh':
            path = str(Path(snapshot['root']) / relative)
            require(runtime['files_sha256'].get(path) == item['sha256'],
                    'measured scientific runtime differs from campaign: ' + relative)

def prospective_ledger(plan):
    keys = ('domain', 'dataset_domain', 'arm', 'seed', 'run_stamp', 'run_dir')
    return {'schema': LEDGER_SCHEMA, 'status': 'prospective', 'model_choice': '3b',
        'model_choice_pending': False, 'model': plan['model'], 'model_revision': plan['model_revision'],
        'domains': list(DOMAINS), 'arms': list(ARMS), 'seeds': list(SEEDS),
        'target_steps': TARGET_STEPS, 'protocol': str(PROTOCOL), 'protocol_sha256': digest(PROTOCOL),
        'planned_runs': [{key: cell[key] for key in keys} for cell in plan['cells']],
        'runs': [], 'released': False, 'outcomes_inspected_before_release': False}


def runtime_bytes(relative, original):
    """Apply only operational E123 guards to bytes destined for a NEW snapshot."""
    if relative not in ('ops/run_experiment.sh', 'ops/train.sh', 'ops/slurm/train_node302.slurm'):
        return original
    text = original.decode()
    if relative in ('ops/run_experiment.sh', 'ops/train.sh'):
        require('e119_level2_' in text, 'healthy E119 runtime guard missing')
        text = text.replace('e119_level2_', 'e123_level3_')
        text = text.replace('E119_', 'E123_').replace('e119_', 'e123_').replace('E119 ', 'E123 ')
        text = text.replace('_s(43|44|45|46|47)', '_s(70|71|72|73|74)')
    else:
        old = 'export OAT_ZERO_WATCHDOG_LOG_PATH="$ROOT_DIR/var/artifacts/logs/xdr_train-${SLURM_JOB_ID}.out"'
        require(text.count(old) == 1, 'expected immutable Slurm log-path site missing')
        text = text.replace(old, 'export OAT_ZERO_WATCHDOG_LOG_PATH="$ROOT_DIR/var/artifacts/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"')
    return text.encode()


def source_files():
    for directory, prefix in ((ROOT / 'src/oat_drgrpo', 'src/oat_drgrpo'), (HEALTHY_OPS, 'ops')):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                yield prefix + '/' + str(path.relative_to(directory)), path


def snapshot_description():
    inventory = {}
    for relative, path in source_files():
        original = path.read_bytes()
        inventory[relative] = {'source': str(path), 'source_sha256': hashlib.sha256(original).hexdigest(),
            'sha256': hashlib.sha256(runtime_bytes(relative, original)).hexdigest(),
            'mode': stat.S_IMODE(path.stat().st_mode)}
    identity = sha(inventory)
    return {'schema': 'e123_runtime_snapshot_v1', 'sha256': identity,
            'root': str(ROOT / 'var/artifacts/source_snapshots' / ('e123_level3_' + identity[:20])),
            'inventory': inventory, 'source_policy': 'current_src_and_healthy_e119_ops_with_operational_e123_guards'}


def environment(template, arm, choice, snapshot, model, profile=None):
    domain, seed = template['domain'], int(template['seed'])
    env, _ = e119.environment(ROOT, template, arm, Path(snapshot))
    checkpoint = CHECKPOINT_INTERVALS[domain]
    env.update({'SAVE_PATH': str(run_dir(domain, arm, seed)), 'RUN_STAMP': run_stamp(domain, arm, seed),
        'OAT_ZERO_PRETRAIN': model['path'],
        'OAT_ZERO_MODEL': 'qwen2.5-3b-instruct' if choice == '3b' else 'qwen2.5-0.5b-instruct',
        'OAT_ZERO_DATA_ROOT': str(DATA_ROOT / DOMAIN_DIR[domain]), 'OAT_ZERO_REQUIRE_EXISTING_DATA': '1',
        'OAT_ZERO_PROMPT_DATA': str(DATA_ROOT / DOMAIN_DIR[domain] / 'train'),
        'OAT_ZERO_EVAL_DATA': str(DATA_ROOT / DOMAIN_DIR[domain] / 'eval'),
        'OAT_ZERO_EVAL_PROMPT_INTERVAL': str(EVALUATION_INTERVAL), 'OAT_ZERO_ALLOW_SPARSE_EVAL': '0',
        'OAT_ZERO_SAVE_STEPS': str(checkpoint), 'OAT_ZERO_SAVE_FROM': str(checkpoint),
        'OAT_ZERO_RESUME_STEPS': str(checkpoint), 'OAT_ZERO_RESUME_FROM': str(checkpoint),
        'OAT_ZERO_EXPORT_STEPS': '0', 'OAT_ZERO_MAX_EXPORT_NUM': '1', 'OAT_ZERO_MAX_RESUME_NUM': '1',
        'OAT_ZERO_CANONICAL_ACTION_TASK': 'none', 'OAT_ZERO_CANONICAL_GRAPH_ACTIONS': '0',
        'OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT': '3', 'OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING': '0',
        'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING': '0',
        'OAT_ZERO_LR_SCHEDULER': 'constant', 'OAT_ZERO_LR_WARMUP_RATIO': '0.0',
        'OAT_ZERO_MAX_STEP_ADJUSTMENT': '1.0', 'OAT_ZERO_ADAM_BETA_1': '0.9',
        'OAT_ZERO_ADAM_BETA_2': '0.95', 'OAT_ZERO_L2': '0.0',
        'OAT_ZERO_EVAL_BATCH_SIZE': '64', 'OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P': '1.0',
        'OAT_ZERO_N_GPU': '1', 'OAT_ZERO_NUM_GPUS_PER_ACTOR': '1',
        'OAT_ZERO_ADAM_OFFLOAD': '1' if choice == '3b' else '0',
        'OAT_ZERO_ACTIVATION_OFFLOADING': '1' if choice == '3b' else '0',
        'OAT_ZERO_WATCHDOG_STALE_SECONDS': '7200', 'OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS': '3600',
        'OAT_ZERO_WATCHDOG_MAX_RESTARTS': '12', 'OAT_ZERO_WATCHDOG_LOG_PROGRESS': '1',
        'OAT_ZERO_WATCHDOG_ARTIFACT_PROGRESS': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'OAT_ZERO_XDR_TAU': 'inf', 'OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS': '0',
        'OAT_ZERO_XDR_TAU_CONTROL_TARGET_RATIO': '0.0', 'OAT_ZERO_XDR_SAC_DUAL_TARGET_RATIO': '0.0',
        'OAT_ZERO_XDR_MODE_ADAPTIVE': '0', 'OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION': '0'})
    env.update(e80.optimizer_env())
    env.update(e80.memory_env())
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    if profile is not None:
        env.update(profile['profile_environment'])
    return dict(sorted(env.items()))


def resources(choice, domain, profile=None):
    selected = profile['resources'] if profile is not None else {'cpus': 16, 'memory_gib': 128}
    return {'cpus': selected['cpus'], 'memory_gib': selected['memory_gib'],
            'partition': 'mltheory', 'account': 'mltheory', 'expected_qos': 'none', 'nodes': SAFE_NODES, 'exclude': PVL,
            'gpus': 1, 'walltime': '3-00:00:00', 'nice': 100}



def storage_profile(choice):
    """Recorded file-size observations, with conservative allocation headroom."""
    gib = 1024 ** 3
    evidence = {
        '05b': {'model_state_bytes': 988211320, 'optimizer_state_bytes': 5928420892,
            'terminal_weights_bytes': 988097824,
            'checkpoint_run': 'var/data/xdr_qwen25_0p5b_instruct_grpo_compute_matched_e119_level2_pantry_drgrpo_s43/debug_job31037827/checkpoints/step_00096',
            'terminal_run': 'var/data/xdr_qwen25_0p5b_instruct_grpo_compute_matched_e119_level2_graph_drgrpo_s43/debug_job31014418/saved_models/step_03073'},
        '3b': {'model_state_bytes': 6172096696, 'optimizer_state_bytes': 37031303168,
            'terminal_weights_bytes': 6171927000,
            'checkpoint_run': 'var/data/xdr_qwen25_3b_instruct_maxrl_compute_matched_e118q3_mathir_maxrl_s70/debug_job31137691/checkpoints/step_01344',
            'terminal_run': 'var/data/xdr_qwen25_3b_instruct_maxrl_compute_matched_e118q3_graph_maxrl_s70/debug_job31100509/saved_models/step_03073'}}
    return {'model_choice': choice, 'peak_bytes': (84 if choice == '3b' else 16) * gib,
            'terminal_bytes': int((6.25 if choice == '3b' else 1.25) * gib),
            'measurement_verified': True, 'measurement_date': '2026-09-09',
            'evidence': evidence[choice],
            'evidence_semantics': 'recorded_read_only_file_size_observations; prior files may retire naturally'}


def command_for(cell, choice):
    env = cell['environment']; resource = cell['resources']
    require(all(',' not in key and ',' not in value and '\n' not in value for key, value in env.items()),
            'Slurm exports must not contain delimiters')
    return ['sbatch', '--parsable', '--hold', '--job-name=' + cell['run_stamp'],
            '--export=ALL,' + ','.join(key + '=' + value for key, value in env.items()),
            '--partition=' + resource['partition'], '--account=' + resource['account'],
            '--nodelist=' + resource['nodes'], '--exclude=' + resource['exclude'], '--gres=gpu:1',
            '--cpus-per-task=' + str(resource['cpus']), '--mem=' + str(resource['memory_gib']) + 'G',
            '--time=' + resource['walltime'], '--nice=' + str(resource['nice']), '--requeue', '--chdir=' + str(ROOT),
            str(Path(env['OAT_ZERO_OPS_SNAPSHOT_ROOT']) / 'slurm/train_node302.slurm')]


def implementation_pins():
    files = [SOURCE, TEST, PROTOCOL, IDENTITY, TEMPLATES,
             ROOT / 'ops/exp_scaling/control_e123_level3_release.py',
             ROOT / 'tests/test_e123_level3_release_controller.py',
             ROOT / 'ops/evaluate_modebench_level3.py']
    files += [ROOT / 'ops/exp_scaling' / name for name in
        ('launch_e119_level2_qwen05b_factorial.py', 'launch_e80r1_qwen3b_aligned_verified_replay.py', 'launch_e72_b3a_replay_ablation.py',
         'launch_e78_verified_replay_only_05b.py', 'launch_e76_tuned_scale.py', 'modebench_level3_v3_common.py')]
    return {str(path): digest(path) for path in files}


def planned_cells(choice, snapshot, model, profile=None):
    cells = []
    for template in templates():
        domain, seed = template['domain'], int(template['seed'])
        for arm in ARMS:
            cell = {'domain': domain, 'dataset_domain': DOMAIN_DIR[domain], 'arm': arm, 'seed': seed,
                'run_stamp': run_stamp(domain, arm, seed), 'run_dir': str(run_dir(domain, arm, seed)),
                'target_steps': TARGET_STEPS, 'environment': environment(template, arm, choice, snapshot, model, profile),
                'resources': resources(choice, domain, profile)}
            cell['command'] = command_for(cell, choice); cells.append(cell)
    require(len(cells) == 100 and len({cell['run_dir'] for cell in cells}) == 100, 'exact unique 100-cell factorial required')
    return cells


def build_plan(choice='3b', *, require_admission=False, profile_path=None, profile_sha256=None):
    require(choice == CAMPAIGN_MODEL_CHOICE, 'E123 requires Qwen2.5-3B')
    proof = admission_proof(required=require_admission)
    systems = systems_proof(profile_path, profile_sha256, required=require_admission)
    profile = systems.get('profile')
    model = model_identity(choice); snapshot = snapshot_description()
    if profile is not None:
        verify_systems_runtime(profile, snapshot)
    return {'schema': PLAN_SCHEMA, 'model_choice': choice, 'model': MODEL_NAMES[choice],
        'registered_campaign_model_choice': CAMPAIGN_MODEL_CHOICE,
        'matches_registered_campaign_model': choice == CAMPAIGN_MODEL_CHOICE,
        'model_revision': MODEL_REVISIONS[choice], 'model_identity': model,
        'domains': list(DOMAINS), 'arms': list(ARMS), 'seeds': list(SEEDS),
        'train_rows': TRAIN_ROWS, 'eval_rows': EVAL_ROWS, 'passes': PASSES, 'target_steps': TARGET_STEPS,
        'evaluation_interval_steps': EVALUATION_INTERVAL, 'checkpoint_interval_steps_by_domain': CHECKPOINT_INTERVALS,
        'protocol': str(PROTOCOL), 'identity': str(IDENTITY), 'identity_sha256': IDENTITY_SHA256,
        'dataset_identity_sha256': IDENTITY_SHA256, 'storage_profile': storage_profile(choice),
        'admission_proof': proof, 'files_sha256': implementation_pins(),
        'snapshot_root': snapshot['root'], 'snapshot': snapshot,
        'systems_proof': systems,
        'cells': planned_cells(choice, snapshot['root'], model, profile),
        'submission_policy': 'all_100_held_with_once_only_intents_and_exact_audits',
        'release_policy': 'separate_storage_aware_controller_requires_explicit_arming',
        'outcomes_inspected_before_release': False}


def verify_snapshot(description):
    snapshot = Path(description['root'])
    expected = set(description['inventory']) | {'SNAPSHOT_IDENTITY.json'}
    actual = {str(path.relative_to(snapshot)) for path in snapshot.rglob('*') if path.is_file()}
    require(actual == expected, 'E123 snapshot file inventory drift')
    require(read(snapshot / 'SNAPSHOT_IDENTITY.json') == description, 'E123 snapshot identity differs')
    for relative, item in description['inventory'].items():
        require(digest(snapshot / relative) == item['sha256'], 'E123 frozen snapshot bytes changed: ' + relative)


def publish_snapshot(description):
    target = Path(description['root'])
    if target.exists():
        verify_snapshot(description)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + target.name + '.', dir=target.parent))
    # Preserve partial staging evidence on failure; never edit a predecessor.
    for relative, item in description['inventory'].items():
        source = Path(item['source']); original = source.read_bytes()
        require(hashlib.sha256(original).hexdigest() == item['source_sha256'], 'snapshot source changed: ' + str(source))
        value = runtime_bytes(relative, original)
        require(hashlib.sha256(value).hexdigest() == item['sha256'], 'snapshot transform differs: ' + relative)
        destination = temporary / relative; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value); destination.chmod(item['mode'])
    atomic_new(temporary / 'SNAPSHOT_IDENTITY.json', description)
    require(not target.exists(), 'E123 runtime snapshot appeared concurrently')
    os.rename(temporary, target)
    verify_snapshot(description)


def verify_plan(path=PLAN, *, expected_sha256, model_choice, require_admission=True):
    require(model_choice == CAMPAIGN_MODEL_CHOICE, 'E123 campaign is registered for Qwen-3B')
    path = Path(path)
    require(digest(path) == expected_sha256, 'explicit E123 plan hash differs')
    plan = read(path)
    require(plan['schema'] == PLAN_SCHEMA and plan['model_choice'] == model_choice
            and plan['registered_campaign_model_choice'] == CAMPAIGN_MODEL_CHOICE
            and plan['matches_registered_campaign_model'] is True,
            'E123 plan schema or chosen model differs')
    require(plan['identity_sha256'] == plan['dataset_identity_sha256'] == IDENTITY_SHA256
            and plan['storage_profile'] == storage_profile(model_choice), 'E123 dataset or storage profile differs')
    common.verify_pins(plan['files_sha256'])
    proof = admission_proof(required=require_admission)
    require(proof == plan['admission_proof'], 'E123 admission proof changed')
    require(model_identity(model_choice) == plan['model_identity'], 'E123 frozen model identity changed')
    selected = plan['systems_proof']
    require(systems_proof(selected['path'], selected['sha256']) == selected, 'E123 measured systems profile drift')
    verify_systems_runtime(selected['profile'], plan['snapshot'])
    verify_snapshot(plan['snapshot'])
    require(plan['snapshot_root'] == plan['snapshot']['root']
            and plan['cells'] == planned_cells(model_choice, plan['snapshot_root'], plan['model_identity'], selected['profile']),
            'E123 frozen cell commands or objective environment changed')
    require(digest(path) == expected_sha256, 'E123 plan changed while verifying')
    return plan


def verify_initial_ledger(ledger, cells):
    require(ledger['schema'] == LEDGER_SCHEMA and ledger['runs'] == [] and ledger['released'] is False,
            'E123 initial prospective ledger required')
    keys = ('domain', 'arm', 'seed', 'run_stamp', 'run_dir')
    expected = {tuple(cell[key] for key in keys) for cell in cells}
    actual = {tuple(cell[key] for key in keys) for cell in ledger['planned_runs']}
    require(len(ledger['planned_runs']) == 100 and actual == expected, 'E123 prospective matrix differs')


def prepare(draft_path, draft_sha256, choice):
    require(digest(draft_path) == draft_sha256, 'reviewed E123 draft hash required')
    require(choice == CAMPAIGN_MODEL_CHOICE, 'E123 campaign is registered for Qwen-3B')
    draft = read(draft_path)
    selected = draft['systems_proof']
    require(draft == build_plan(choice, require_admission=True, profile_path=selected['path'],
                               profile_sha256=selected['sha256']), 'reviewed E123 draft inputs changed')
    require(not PLAN.exists() and not CLAIM.exists(), 'E123 was already prepared or claimed')
    if LEDGER.exists():
        verify_initial_ledger(read(LEDGER), draft['cells'])
    require(not any(Path(cell['run_dir']).exists() for cell in draft['cells']), 'fresh E123 run directories required')
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not LEDGER.exists():
            atomic_new(LEDGER, prospective_ledger(draft))
        atomic_new(HERE / 'preparation_intent.json', {'schema': 'e123_preparation_intent_v1',
            'created_at': now(), 'draft': str(Path(draft_path).resolve()), 'draft_sha256': draft_sha256,
            'prospective_ledger_sha256': digest(LEDGER), 'snapshot_root': draft['snapshot_root']})
        publish_snapshot(draft['snapshot'])
        common.verify_pins(draft['files_sha256'])
        require(admission_proof() == draft['admission_proof'], 'admission changed during preparation')
        atomic_new(PLAN, draft)
        verify_plan(expected_sha256=digest(PLAN), model_choice=choice)
    return {'plan': str(PLAN), 'plan_sha256': digest(PLAN), 'snapshot_root': draft['snapshot_root']}


def field(record, key):
    matches = list(re.finditer(r'(?:^|\s)' + re.escape(key) + r'=([^\s]+)', record))
    require(len(matches) == 1, 'scheduler field missing or duplicated: ' + key)
    return matches[0].group(1)


def audit_held_record(record, job_id, cell):
    resource = cell['resources']
    require(field(record, 'JobId') == str(job_id) and field(record, 'JobName') == cell['run_stamp']
            and field(record, 'JobState') == 'PENDING' and field(record, 'Reason') == 'JobHeldUser'
            and field(record, 'Account') == resource['account'] and field(record, 'Partition') == resource['partition']
            and field(record, 'QOS') == resource['expected_qos']
            and field(record, 'NumCPUs') == str(resource['cpus'])
            and field(record, 'NumNodes') in ('1', '1-1') and field(record, 'NumTasks') == '1'
            and field(record, 'Requeue') == '1' and field(record, 'Nice') == str(resource['nice'])
            and field(record, 'UserId') == f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})'
            and field(record, 'WorkDir') == str(ROOT) and field(record, 'Command') == cell['command'][-1]
            and field(record, 'RunTime') == '00:00:00' and field(record, 'Restarts') == '0'
            and field(record, 'MinMemoryNode') in (str(resource['memory_gib']) + 'G', str(resource['memory_gib'] * 1024))
            and field(record, 'TimeLimit') == resource['walltime']
            and field(record, 'ExcNodeList') == resource['exclude']
            and field(record, 'ReqNodeList') == resource['nodes']
            and re.search(r'(?:^|,)gres/gpu=1(?:,|$)', field(record, 'ReqTRES')),
            'E123 held scheduler resources or identity differ')
    for key, value in cell['environment'].items():
        require(re.search(r'(?:^|[ ,])' + re.escape(key + '=' + value) + r'(?=,|\s|$)', record),
                'E123 held environment differs: ' + key)
    return record


def audit_held(job_id, cell):
    result = subprocess.run(['scontrol', 'show', 'job', '-dd', '-o', str(job_id)],
                            capture_output=True, text=True, check=False, timeout=60)
    require(result.returncode == 0, 'cannot audit E123 held job: ' + result.stderr)
    return audit_held_record(result.stdout, job_id, cell)


def clean_submit_environment():
    # --export=ALL must never inherit a caller's unregistered treatment knobs.
    return {key: value for key, value in os.environ.items()
            if not key.startswith(('OAT_ZERO_', 'SBATCH_', 'SLURM_'))
            and key not in ('SAVE_PATH', 'RUN_STAMP', 'ROOT_DIR', 'PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}


def submit_one_held(cell, index, *, directory=HERE, runner=subprocess.run):
    directory = Path(directory)
    intent = directory / f'submission_{index:03d}_intent.json'
    result_path = directory / f'submission_{index:03d}_result.json'
    atomic_new(intent, {'created_at': now(), 'cell': cell, 'command_sha256': sha(cell['command'])})
    try:
        result = runner(cell['command'], capture_output=True, text=True, check=False, timeout=120,
                        env=clean_submit_environment())
    except BaseException as error:
        atomic_new(result_path, {'created_at': now(), 'status': 'ambiguous_exception',
                                'error_type': type(error).__name__, 'error': str(error)})
        raise
    atomic_new(result_path, {'created_at': now(), 'returncode': result.returncode,
                            'stdout': result.stdout, 'stderr': result.stderr})
    require(result.returncode == 0 and re.fullmatch(r'[1-9]\d*(?:;[^\s]+)?\s*', result.stdout),
            'E123 submission ambiguous or failed; retain intents and reconcile manually; never retry')
    return int(result.stdout.strip().split(';', 1)[0])



def assert_no_existing_campaign_jobs():
    result = subprocess.run(['squeue', '--noheader', '--user', str(os.getuid()), '--format=%i|%j|%T'],
                            capture_output=True, text=True, check=False, timeout=60)
    require(result.returncode == 0, 'cannot enumerate queue before E123 submission')
    for line in result.stdout.splitlines():
        fields = line.strip().split('|')
        require(len(fields) == 3, 'malformed queue record before E123 submission')
        require(not fields[1].startswith('e123_level3_'), 'existing E123 campaign job; refuse duplicate submission')

def submit_held(plan_sha256, choice):
    plan = verify_plan(expected_sha256=plan_sha256, model_choice=choice)
    ledger = read(LEDGER); verify_initial_ledger(ledger, plan['cells'])
    require(not any(Path(cell['run_dir']).exists() for cell in plan['cells']), 'fresh E123 cells required')
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_existing_campaign_jobs()
        atomic_new(CLAIM, {'schema': 'e123_once_only_submission_claim_v1', 'created_at': now(),
            'plan_sha256': plan_sha256, 'initial_ledger_sha256': digest(LEDGER), 'jobs': 100})
        initial_sha = digest(LEDGER); records = []
        for index, cell in enumerate(plan['cells']):
            job_id = submit_one_held(cell, index)
            record = {key: cell[key] for key in ('domain', 'dataset_domain', 'arm', 'seed', 'run_stamp', 'run_dir', 'target_steps')}
            record.update(job_id=job_id, held_scheduler_record=audit_held(job_id, cell))
            atomic_new(HERE / f'held_audit_{index:03d}.json', record); records.append(record)
            print(json.dumps({'event': 'held_audited', 'count': len(records), 'job_id': job_id}), flush=True)
        require(len({row['job_id'] for row in records}) == 100, 'duplicate E123 job ID')
        verify_plan(expected_sha256=plan_sha256, model_choice=choice)
        for record, cell in zip(records, plan['cells']):
            audit_held(record['job_id'], cell)
        require(digest(LEDGER) == initial_sha, 'E123 prospective ledger changed during submission')
        atomic_new(HERE / 'prospective_ledger_before_submission.json', ledger)
        ledger.update(runs=records, status='held_audited', released=False,
            model_choice=choice, model_choice_pending=False, model=plan['model'], model_revision=plan['model_revision'],
            model_family='qwen05b' if choice == '05b' else 'qwen3b',
            admission_proof=plan['admission_proof'], plan_path=str(PLAN), plan_sha256=plan_sha256,
            snapshot_root=plan['snapshot_root'], held_audited_at=now())
        e119.e78.atomic_json(LEDGER, ledger)
        atomic_new(HERE / 'held_submission_complete.json', {'schema': 'e123_held_submission_complete_v1',
            'created_at': now(), 'plan_sha256': plan_sha256, 'ledger_sha256': digest(LEDGER),
            'job_ids': [record['job_id'] for record in records], 'released': False})
    return {'held': 100, 'released': 0, 'ledger': str(LEDGER), 'ledger_sha256': digest(LEDGER)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--submit-held', action='store_true')
    action.add_argument('--verify-plan', action='store_true')
    parser.add_argument('--model-choice', default='3b', choices=('3b',))
    parser.add_argument('--systems-profile', type=Path)
    parser.add_argument('--systems-profile-sha256')
    parser.add_argument('--output', type=Path, help='exclusively write a review draft; omit to print JSON')
    parser.add_argument('--draft', type=Path)
    parser.add_argument('--draft-sha256')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args(argv)
    if args.prepare:
        require(args.draft is not None and args.draft_sha256, 'prepare requires reviewed draft path and SHA-256')
        result = prepare(args.draft, args.draft_sha256, args.model_choice)
    elif args.submit_held:
        require(args.plan_sha256, 'submit-held requires explicit plan SHA-256')
        result = submit_held(args.plan_sha256, args.model_choice)
    elif args.verify_plan:
        require(args.plan_sha256, 'verify-plan requires explicit plan SHA-256')
        plan = verify_plan(expected_sha256=args.plan_sha256, model_choice=args.model_choice)
        result = {'verified': True, 'cells': len(plan['cells']), 'plan_sha256': args.plan_sha256}
    else:
        result = build_plan(args.model_choice, profile_path=args.systems_profile, profile_sha256=args.systems_profile_sha256)
        if args.output:
            atomic_new(args.output, result)
            result = {'draft': str(args.output.resolve()), 'draft_sha256': digest(args.output),
                      'cells': 100, 'admission_status': result['admission_proof']['status'],
                      'systems_status': result['systems_proof']['status'], 'snapshot_published': False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
