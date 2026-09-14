#!/usr/bin/env python3
"""Prepare E122 and submit its 100 cells held; never release training jobs.

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
import modebench_level3_v3_common as common

SOURCE = Path(__file__).resolve()
TEST = ROOT / 'tests/test_e122_level3_factorial.py'
PROTOCOL = ROOT / 'paper/preregistration/e122_level3_factorial_20260909.md'
LEDGER = ROOT / 'var/artifacts/e122_level3_factorial_jobs.json'
PLAN = ROOT / 'var/artifacts/e122_level3_factorial_plan.json'
HERE = ROOT / 'var/artifacts/e122_level3_factorial'
CLAIM = HERE / 'submission_claim.json'
DATA_ROOT = ROOT / 'var/data/modebench_level3_matched_v3'
IDENTITY = DATA_ROOT / 'identity.json'
IDENTITY_SHA256 = '890d7697af7789e0ae53c803586ec7f722685b7eb2239643175a1933fa45650d'
REPORT = ROOT / 'var/artifacts/modebench_level3_v3/confirmation/confirmation_report.json'
TEMPLATES = ROOT / 'var/artifacts/e72_frontier_source_runs.json'
HEALTHY_OPS = ROOT / 'var/artifacts/source_snapshots/e76_tuned_scale_50d36295558a8958/ops'
DOMAINS, ARMS, SEEDS = e119.DOMAINS, e119.ARMS, e119.SEEDS
DOMAIN_TAGS, DOMAIN_DIR = e119.DOMAIN_TAGS, e119.DOMAIN_DIR
PASSES, TRAIN_ROWS, EVAL_ROWS, TARGET_STEPS = 8, 384, 128, 3072
EVALUATION_INTERVAL = 96
CHECKPOINT_INTERVALS = {domain: 96 if domain == 'pantry_plan' else 192 for domain in DOMAINS}
MODEL_REVISIONS = {'05b': '7ae557604adf67be50417f59c2c2f167def9a775',
                   '3b': 'aa8e72537993ba99e69dfaafa59ed015b17504d1'}
MODEL_NAMES = {'05b': 'Qwen2.5-0.5B-Instruct', '3b': 'Qwen2.5-3B-Instruct'}
CAMPAIGN_MODEL_CHOICE = '05b'
# Amendment 2026-09-13 (48-GiB pool widening).  The original route carried an
# ownership-based exclusion and a four-node pool, which left the campaign
# memory-bound rather than GPU-bound: 14 idle GPUs sat behind nodes whose free
# host RAM was under the smallest 64 GiB cell.  The pool is now defined
# positively by the registered hardware criterion -- at least 48 GiB of
# bf16-capable GPU, never a 24 GiB card -- so --nodelist is authoritative and no
# ownership exclusion is requested.  Retained for provenance:
PVL_OWNERSHIP_LEGACY = 'node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]'
PVL = ''
SAFE_NODES = ('node101,node103,node104,node205,node206,node207,node208,node302,'
              'node403,node805')
# Slurm renders an accepted pool either as the submitted comma list or as a
# hostlist; both denote the same route.
REQ_NODE_FORMS = ('node[205-207,302]', 'node[205,206,207,302]',
                  'node[101,103-104,205-208,302,403,805]', SAFE_NODES)
PLAN_SCHEMA = 'e122_level3_factorial_plan_v1'
LEDGER_SCHEMA = 'e122_level3_factorial_jobs_v1'
require, read, digest, sha = common.require, common.read, common.digest, common.sha
atomic_new = common.atomic_new
# Valid only inside this process, after full successful canonical authentication.
_ADMISSION_CACHE = None


def now():
    return datetime.now(timezone.utc).isoformat()


def run_stamp(domain, arm, seed):
    return f'e122_level3_{DOMAIN_TAGS[domain]}_{arm}_s{seed}'


def run_dir(domain, arm, seed):
    return ROOT / 'var/data/e122_level3_factorial' / domain / arm / f's{seed}'


def templates():
    """Load only the frozen schedule metadata, without requiring old admissions."""
    runs = [run for run in read(TEMPLATES)['runs'] if run['arm'] == 'xgrpo'
            and run['domain'] in DOMAINS and int(run['seed']) in SEEDS]
    require(len(runs) == 25 and {(run['domain'], int(run['seed'])) for run in runs}
            == {(domain, seed) for domain in DOMAINS for seed in SEEDS}, 'exact E119 schedule templates required')
    return sorted(runs, key=lambda run: (run['domain'], int(run['seed'])))


def model_identity(choice):
    require(choice in MODEL_NAMES, 'explicit model choice 05b or 3b required')
    from evaluate_modebench_level3 import model_identity as identify
    path = ROOT / 'var/cache/huggingface/transformers' / ('models--Qwen--' + MODEL_NAMES[choice]) / 'snapshots' / MODEL_REVISIONS[choice]
    return identify(path, choice)


def admission_proof(*, required=True):
    """Authenticate once per process; rehash every frozen input on each reuse."""
    global _ADMISSION_CACHE
    require(digest(IDENTITY) == IDENTITY_SHA256, 'E122 Level-3 dataset identity drift')
    identity = read(IDENTITY)
    require(identity['split_sizes'] == {'train': 384, 'dev': 128, 'eval': 128}
            and set(identity['domains']) == set(DOMAIN_DIR.values()), 'E122 exact all-five dataset required')
    if _ADMISSION_CACHE is None:
        if not REPORT.is_file():
            require(not required, 'E122 awaits canonical completed V3 confirmation')
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
            'E122 requires all five authenticated fixed-reference comparisons to pass')
    require(report['dataset']['dataset_root'] == str(DATA_ROOT)
            and report['dataset']['path'] == str(IDENTITY)
            and report['dataset']['sha256'] == IDENTITY_SHA256
            and report['dataset']['split_sizes'] == common.SPLITS
            and report['information_boundary']['reference_semantics'] == 'fixed_measured_level1_benchmark'
            and report['information_boundary']['adaptive_confirmation_round'] == 2
            and report['information_boundary']['historical_level1_confirmation_used_as_fixed_reference'] is True
            and report['information_boundary']['statistical_equivalence_claimed'] is False,
            'E122 admission dataset or adaptive fixed-reference provenance differs')
    require(evidence['path'] == str(REPORT)
            and evidence['files_sha256'].get(str(REPORT)) == evidence['sha256'] == digest(REPORT),
            'canonical report must remain bound in the authenticated file inventory')
    if _ADMISSION_CACHE is None:
        _ADMISSION_CACHE = deepcopy(evidence)
    # Plans and callers cannot mutate the trusted process-local proof.
    return deepcopy({key: evidence[key] for key in ('path', 'sha256', 'status', 'files_sha256', 'directory_files')})


def runtime_bytes(relative, original):
    """Apply only operational E122 guards to bytes destined for a NEW snapshot."""
    if relative not in ('ops/run_experiment.sh', 'ops/train.sh', 'ops/slurm/train_node302.slurm'):
        return original
    text = original.decode()
    if relative in ('ops/run_experiment.sh', 'ops/train.sh'):
        require('e119_level2_' in text, 'healthy E119 runtime guard missing')
        text = text.replace('e119_level2_', 'e122_level3_')
        text = text.replace('E119_', 'E122_').replace('e119_', 'e122_').replace('E119 ', 'E122 ')
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
    return {'schema': 'e122_runtime_snapshot_v1', 'sha256': identity,
            'root': str(ROOT / 'var/artifacts/source_snapshots' / ('e122_level3_' + identity[:20])),
            'inventory': inventory, 'source_policy': 'current_src_and_healthy_e119_ops_with_operational_e122_guards'}


def environment(template, arm, choice, snapshot, model):
    domain, seed = template['domain'], int(template['seed'])
    env, _ = e119.environment(ROOT, template, arm, Path(snapshot))
    checkpoint = CHECKPOINT_INTERVALS[domain]
    env.update({'SAVE_PATH': str(run_dir(domain, arm, seed)), 'RUN_STAMP': run_stamp(domain, arm, seed),
        'OAT_ZERO_PRETRAIN': model['path'],
        'OAT_ZERO_MODEL': 'qwen2.5-3b-instruct' if choice == '3b' else 'qwen2.5-0.5b-instruct',
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
    return dict(sorted(env.items()))


def resources(choice, domain):
    return {'cpus': 16 if choice == '3b' else 8,
            'memory_gib': 128 if choice == '3b' or domain == 'countdown' else 96 if domain == 'pantry_plan' else 64,
            'partition': 'lowprio', 'account': 'allcs', 'expected_qos': 'medium', 'nodes': SAFE_NODES, 'exclude': PVL,
            'gpus': 1, 'walltime': '1-12:00:00'}


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
    env = cell['environment']; resource = resources(choice, cell['domain'])
    require(all(',' not in key and ',' not in value and '\n' not in value for key, value in env.items()),
            'Slurm exports must not contain delimiters')
    return ['sbatch', '--parsable', '--hold', '--job-name=' + cell['run_stamp'],
            '--export=ALL,' + ','.join(key + '=' + value for key, value in env.items()),
            '--partition=' + resource['partition'], '--account=' + resource['account'],
            '--nodelist=' + resource['nodes'], '--gres=gpu:1',
            '--cpus-per-task=' + str(resource['cpus']), '--mem=' + str(resource['memory_gib']) + 'G',
            '--time=' + resource['walltime'], '--nice=0', '--requeue', '--chdir=' + str(ROOT),
            str(Path(env['OAT_ZERO_OPS_SNAPSHOT_ROOT']) / 'slurm/train_node302.slurm')]


def implementation_pins():
    files = [SOURCE, TEST, PROTOCOL, IDENTITY, TEMPLATES,
             ROOT / 'ops/exp_scaling/control_e122_level3_release.py',
             ROOT / 'tests/test_e122_level3_release_controller.py',
             ROOT / 'ops/evaluate_modebench_level3.py']
    files += [ROOT / 'ops/exp_scaling' / name for name in
        ('launch_e119_level2_qwen05b_factorial.py', 'launch_e72_b3a_replay_ablation.py',
         'launch_e78_verified_replay_only_05b.py', 'launch_e76_tuned_scale.py', 'modebench_level3_v3_common.py')]
    return {str(path): digest(path) for path in files}


def planned_cells(choice, snapshot, model):
    cells = []
    for template in templates():
        domain, seed = template['domain'], int(template['seed'])
        for arm in ARMS:
            cell = {'domain': domain, 'dataset_domain': DOMAIN_DIR[domain], 'arm': arm, 'seed': seed,
                'run_stamp': run_stamp(domain, arm, seed), 'run_dir': str(run_dir(domain, arm, seed)),
                'target_steps': TARGET_STEPS, 'environment': environment(template, arm, choice, snapshot, model),
                'resources': resources(choice, domain)}
            cell['command'] = command_for(cell, choice); cells.append(cell)
    require(len(cells) == 100 and len({cell['run_dir'] for cell in cells}) == 100, 'exact unique 100-cell factorial required')
    return cells


def build_plan(choice, *, require_admission=False):
    proof = admission_proof(required=require_admission)
    model = model_identity(choice); snapshot = snapshot_description()
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
        'cells': planned_cells(choice, snapshot['root'], model),
        'submission_policy': 'all_100_held_with_once_only_intents_and_exact_audits',
        'release_policy': 'separate_storage_aware_controller_requires_explicit_arming',
        'outcomes_inspected_before_release': False}


def verify_snapshot(description):
    snapshot = Path(description['root'])
    expected = set(description['inventory']) | {'SNAPSHOT_IDENTITY.json'}
    actual = {str(path.relative_to(snapshot)) for path in snapshot.rglob('*') if path.is_file()}
    require(actual == expected, 'E122 snapshot file inventory drift')
    require(read(snapshot / 'SNAPSHOT_IDENTITY.json') == description, 'E122 snapshot identity differs')
    for relative, item in description['inventory'].items():
        require(digest(snapshot / relative) == item['sha256'], 'E122 frozen snapshot bytes changed: ' + relative)


def publish_snapshot(description):
    target = Path(description['root'])
    require(not target.exists(), 'fresh E122 runtime snapshot required')
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
    require(not target.exists(), 'E122 runtime snapshot appeared concurrently')
    os.rename(temporary, target)
    verify_snapshot(description)


def verify_plan(path=PLAN, *, expected_sha256, model_choice, require_admission=True):
    require(model_choice == CAMPAIGN_MODEL_CHOICE, 'E122 campaign is registered for Qwen-0.5B; 3B preview only')
    path = Path(path)
    require(digest(path) == expected_sha256, 'explicit E122 plan hash differs')
    plan = read(path)
    require(plan['schema'] == PLAN_SCHEMA and plan['model_choice'] == model_choice
            and plan['registered_campaign_model_choice'] == CAMPAIGN_MODEL_CHOICE
            and plan['matches_registered_campaign_model'] is True,
            'E122 plan schema or chosen model differs')
    require(plan['identity_sha256'] == plan['dataset_identity_sha256'] == IDENTITY_SHA256
            and plan['storage_profile'] == storage_profile(model_choice), 'E122 dataset or storage profile differs')
    common.verify_pins(plan['files_sha256'])
    proof = admission_proof(required=require_admission)
    require(proof == plan['admission_proof'], 'E122 admission proof changed')
    require(model_identity(model_choice) == plan['model_identity'], 'E122 frozen model identity changed')
    verify_snapshot(plan['snapshot'])
    require(plan['snapshot_root'] == plan['snapshot']['root']
            and plan['cells'] == planned_cells(model_choice, plan['snapshot_root'], plan['model_identity']),
            'E122 frozen cell commands or objective environment changed')
    require(digest(path) == expected_sha256, 'E122 plan changed while verifying')
    return plan


def verify_initial_ledger(ledger, cells):
    require(ledger['schema'] == LEDGER_SCHEMA and ledger['runs'] == [] and ledger['released'] is False,
            'E122 initial prospective ledger required')
    keys = ('domain', 'arm', 'seed', 'run_stamp', 'run_dir')
    expected = {tuple(cell[key] for key in keys) for cell in cells}
    actual = {tuple(cell[key] for key in keys) for cell in ledger['planned_runs']}
    require(len(ledger['planned_runs']) == 100 and actual == expected, 'E122 prospective matrix differs')


def prepare(draft_path, draft_sha256, choice):
    require(digest(draft_path) == draft_sha256, 'reviewed E122 draft hash required')
    require(choice == CAMPAIGN_MODEL_CHOICE, 'E122 campaign is registered for Qwen-0.5B; 3B preview only')
    draft = read(draft_path)
    require(draft == build_plan(choice, require_admission=True), 'reviewed E122 draft inputs changed')
    require(not PLAN.exists() and not CLAIM.exists(), 'E122 was already prepared or claimed')
    verify_initial_ledger(read(LEDGER), draft['cells'])
    require(not any(Path(cell['run_dir']).exists() for cell in draft['cells']), 'fresh E122 run directories required')
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_new(HERE / 'preparation_intent.json', {'schema': 'e122_preparation_intent_v1',
            'created_at': now(), 'draft': str(Path(draft_path).resolve()), 'draft_sha256': draft_sha256,
            'prospective_ledger_sha256': digest(LEDGER), 'snapshot_root': draft['snapshot_root']})
        publish_snapshot(draft['snapshot'])
        common.verify_pins(draft['files_sha256'])
        require(admission_proof() == draft['admission_proof'], 'admission changed during preparation')
        atomic_new(PLAN, draft)
        verify_plan(expected_sha256=digest(PLAN), model_choice=choice)
    return {'plan': str(PLAN), 'plan_sha256': digest(PLAN), 'snapshot_root': draft['snapshot_root']}


def field(record, key):
    match = re.search(r'(?:^|\s)' + re.escape(key) + r'=([^\s]+)', record)
    require(match is not None, 'scheduler field missing: ' + key)
    return match.group(1)


def audit_held_record(record, job_id, cell):
    resource = cell['resources']
    require(field(record, 'JobId') == str(job_id) and field(record, 'JobName') == cell['run_stamp']
            and field(record, 'JobState') == 'PENDING' and field(record, 'Reason') == 'JobHeldUser'
            and field(record, 'Account') == resource['account'] and field(record, 'Partition') == resource['partition']
            and field(record, 'QOS') == resource['expected_qos']
            and field(record, 'NumCPUs') == str(resource['cpus'])
            and field(record, 'NumNodes') == '1' and field(record, 'NumTasks') == '1'
            and field(record, 'Requeue') == '1' and field(record, 'Nice') == '0'
            and field(record, 'UserId') == f'{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})'
            and field(record, 'WorkDir') == str(ROOT) and field(record, 'Command') == cell['command'][-1]
            and field(record, 'RunTime') == '00:00:00' and field(record, 'Restarts') == '0'
            and field(record, 'MinMemoryNode') in (str(resource['memory_gib']) + 'G', str(resource['memory_gib'] * 1024))
            and field(record, 'TimeLimit') == resource['walltime']
            and field(record, 'ExcNodeList') == (resource['exclude'] or '(null)')
            and field(record, 'ReqNodeList') in REQ_NODE_FORMS
            and re.search(r'(?:^|,)gres/gpu=1(?:,|$)', field(record, 'ReqTRES')),
            'E122 held scheduler resources or identity differ')
    for key, value in cell['environment'].items():
        require(re.search(r'(?:^|[ ,])' + re.escape(key + '=' + value) + r'(?=,|\s|$)', record),
                'E122 held environment differs: ' + key)
    return record


def audit_held(job_id, cell):
    result = subprocess.run(['scontrol', 'show', 'job', '-dd', '-o', str(job_id)],
                            capture_output=True, text=True, check=False, timeout=60)
    require(result.returncode == 0, 'cannot audit E122 held job: ' + result.stderr)
    return audit_held_record(result.stdout, job_id, cell)


def clean_submit_environment():
    # --export=ALL must never inherit a caller's unregistered treatment knobs.
    return {key: value for key, value in os.environ.items()
            if not key.startswith(('OAT_ZERO_', 'SBATCH_', 'SLURM_'))
            and key not in ('SAVE_PATH', 'RUN_STAMP', 'ROOT_DIR', 'PYTHONPATH')}


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
            'E122 submission ambiguous or failed; retain intents and reconcile manually; never retry')
    return int(result.stdout.strip().split(';', 1)[0])


def submit_held(plan_sha256, choice):
    plan = verify_plan(expected_sha256=plan_sha256, model_choice=choice)
    ledger = read(LEDGER); verify_initial_ledger(ledger, plan['cells'])
    require(not any(Path(cell['run_dir']).exists() for cell in plan['cells']), 'fresh E122 cells required')
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / '.submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_new(CLAIM, {'schema': 'e122_once_only_submission_claim_v1', 'created_at': now(),
            'plan_sha256': plan_sha256, 'initial_ledger_sha256': digest(LEDGER), 'jobs': 100})
        initial_sha = digest(LEDGER); records = []
        for index, cell in enumerate(plan['cells']):
            job_id = submit_one_held(cell, index)
            record = {key: cell[key] for key in ('domain', 'dataset_domain', 'arm', 'seed', 'run_stamp', 'run_dir', 'target_steps')}
            record.update(job_id=job_id, held_scheduler_record=audit_held(job_id, cell))
            atomic_new(HERE / f'held_audit_{index:03d}.json', record); records.append(record)
            print(json.dumps({'event': 'held_audited', 'count': len(records), 'job_id': job_id}), flush=True)
        require(len({row['job_id'] for row in records}) == 100, 'duplicate E122 job ID')
        verify_plan(expected_sha256=plan_sha256, model_choice=choice)
        for record, cell in zip(records, plan['cells']):
            audit_held(record['job_id'], cell)
        require(digest(LEDGER) == initial_sha, 'E122 prospective ledger changed during submission')
        atomic_new(HERE / 'prospective_ledger_before_submission.json', ledger)
        ledger.update(runs=records, status='held_audited', released=False,
            model_choice=choice, model_choice_pending=False, model=plan['model'], model_revision=plan['model_revision'],
            model_family='qwen05b' if choice == '05b' else 'qwen3b',
            admission_proof=plan['admission_proof'], plan_path=str(PLAN), plan_sha256=plan_sha256,
            snapshot_root=plan['snapshot_root'], held_audited_at=now())
        e119.e78.atomic_json(LEDGER, ledger)
        atomic_new(HERE / 'held_submission_complete.json', {'schema': 'e122_held_submission_complete_v1',
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
    parser.add_argument('--model-choice', required=True, choices=tuple(MODEL_NAMES))
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
        result = build_plan(args.model_choice)
        if args.output:
            atomic_new(args.output, result)
            result = {'draft': str(args.output.resolve()), 'draft_sha256': digest(args.output),
                      'cells': 100, 'admission_status': result['admission_proof']['status'], 'snapshot_published': False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
