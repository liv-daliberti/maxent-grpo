"""Read-only, fixed scientific recipes for E124's 30 Qwen2.5-7B cells.

This module neither snapshots code nor submits jobs. Admission authenticates
existing benchmark decisions; it never fits datasets to 7B outcomes. The caller
must freeze the source, validate the model snapshot, qualify physical resources,
and persist these proofs before releasing any treatment job.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    if str(ROOT / directory) not in sys.path:
        sys.path.insert(0, str(ROOT / directory))
import launch_e80r1_qwen3b_aligned_verified_replay as e80
import launch_e119_level2_qwen05b_factorial as e119

SCHEMA = 'e124_qwen7b_three_level_recipe_v1'
MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'
MODEL_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
MODEL_ALIAS = 'qwen2.5-7b-instruct'
MODEL_TAG = 'qwen25_7b_instruct'
LEVELS = (1, 2, 3)
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
ARMS = ('maxrl', 'replay_maxrl')
SEED = 70
SEEDS = (SEED,)
DOMAIN_TAGS = dict(e119.DOMAIN_TAGS)
DOMAIN_DIR = dict(e119.DOMAIN_DIR)
PASSES, TRAIN_ROWS, EVAL_ROWS, TARGET_STEPS = 8, 384, 128, 3072
EVALUATION_INTERVAL, EVAL_BATCH_SIZE, CHECKPOINT_INTERVAL = 192, 32, 96
OUTPUT_ROOT = ROOT / 'var/data/e124_qwen7b_three_level'
L1_LEDGER = ROOT / 'var/artifacts/e118q3_maxrl_verified_replay_extension_jobs.json'
DATA_ROOTS = {2: ROOT / 'var/data/modebench_harder_v2_matched_r5',
              3: ROOT / 'var/data/modebench_level3_matched_v3'}
L1_DIRECTORIES = dict(zip(DOMAINS, ('graph_coloring_modebench_v2', 'exact_countdown_easy3_probe',
    'python_factor_modebench_v1', 'mathir_action_menu_v1', 'pantry_plan_modebench_v2')))
# SHA256 of sorted relative filename -> file SHA256 for native train + eval trees.
# Includes the native unique_answer subsets where present; training uses only
# train and evaluation uses only multi_answer, exactly as E118.
L1_INVENTORY_SHA256 = dict(zip(DOMAINS, (
    '270a3d231376d24e4021ea42ea658690b7d02305e3b28b83b4b44fd28b1bfe30',
    '8db43adfb926c850b4bb2f35fe8e63c7ab8587df9a15d0bdadc770b10746968c',
    '85ec4bac8a48e71f021a26da3df5c6567c643d001131d80c2c5ce4411d40ed28',
    'bda6c0066eabee28c2087f24d1b067b25240899381d4657e4d338fe4c1d5433c',
    '67d5d412c1c240d17a10e9d37bf66e8c8baf0b66e78c3c6552f5acc15e7712b2')))
# Literal pin, not an inference from the current mutable campaign ledger.
L2_IDENTITY_SHA256 = '2af80f4d31a44482574b314cc37ef84bde48c53ecc2ff78dadf97571f0d73fb2'
L2_REPORTS = {
    ROOT / e119.BASELINE_REPORT: '0070600d687715fadff0cdf920e8959aeb6bcb398c566047bcdc285a05a611cb',
    ROOT / e119.REPEAT_REPORT: '1a73becd7a959d12ab375e3cc41f2ff2b14886a5071dc1f3dd7291963f8585f9'}
L3_IDENTITY_SHA256 = '890d7697af7789e0ae53c803586ec7f722685b7eb2239643175a1933fa45650d'
L3_REPORT_SHA256 = 'b685cfa6b20d6f9b714239546a7946947af95af3d4e83ed8ab1b7ad96c453195'
EVAL_SEEDS = dict(zip(DOMAINS, (610100, 610200, 610300, 610400, 76299)))
PHYSICAL_ENV_KEYS = frozenset({
    'OAT_ZERO_N_GPU', 'OAT_ZERO_NUM_GPUS_PER_ACTOR', 'OAT_ZERO_ADAM_OFFLOAD',
    'OAT_ZERO_ACTIVATION_OFFLOADING', 'OAT_ZERO_ZERO_STAGE', 'OAT_ZERO_VLLM_GPU_RATIO',
    'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS'})
LOCATION_ENV_KEYS = frozenset({'SAVE_PATH', 'RUN_STAMP', 'OAT_ZERO_REPO_ROOT',
    'OAT_ZERO_SOURCE_ROOT', 'OAT_ZERO_OPS_SNAPSHOT_ROOT', 'OAT_ZERO_PRETRAIN'})
PAIR_DIFFERENCE_KEYS = frozenset({'SAVE_PATH', 'RUN_STAMP', 'OAT_ZERO_VARIANT',
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY'})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def data_directory(level, domain):
    require(level in LEVELS and domain in DOMAINS, 'unknown E124 level or domain')
    return (ROOT / 'var/data' / L1_DIRECTORIES[domain] if level == 1
            else DATA_ROOTS[level] / DOMAIN_DIR[domain])


def interface(level, domain):
    """Native L1 interface; ordinary generation on the frozen L2/r5 and L3/v3."""
    data = data_directory(level, domain)
    pantry = level == 1 and domain == 'pantry_plan'
    prompt, response, model_len = ((640, 8, 704) if pantry else
        (256, 64, 384) if level == 1 and domain == 'mathir' else
        (256, 192, 512) if level == 1 else (1024, 192, 2048))
    return {'OAT_ZERO_DATA_ROOT': str(data), 'OAT_ZERO_REQUIRE_EXISTING_DATA': '1',
        'OAT_ZERO_PROMPT_DATA': str(data / 'train'), 'OAT_ZERO_EVAL_DATA': str(data / 'eval'),
        'OAT_ZERO_PROMPT_TEMPLATE': ('qwen_pantry_support_mask' if pantry else 'qwen_boxed')
            if level == 1 else e119.PROMPTS[domain],
        'OAT_ZERO_MODEBENCH_DOMAIN': domain,
        'OAT_ZERO_MODEBENCH_SYNTAX_PROFILE': 'none' if level == 1 else e119.SYNTAX[domain],
        'OAT_ZERO_TEST_SPLIT': 'multi_answer', 'OAT_ZERO_INPUT_KEY': 'problem',
        'OAT_ZERO_OUTPUT_KEY': 'answer', 'OAT_ZERO_EVAL_INPUT_KEY': 'problem',
        'OAT_ZERO_EVAL_OUTPUT_KEY': 'answer', 'OAT_ZERO_VERIFIER_VERSION': 'fast',
        'OAT_ZERO_PROMPT_MAX_LENGTH': str(prompt), 'OAT_ZERO_GENERATE_MAX_LENGTH': str(response),
        'OAT_ZERO_EVAL_GENERATE_MAX_LENGTH': str(response), 'OAT_ZERO_MAX_MODEL_LEN': str(model_len),
        'OAT_ZERO_CANONICAL_ACTION_TASK': 'pantry_support_mask' if pantry else 'none',
        'OAT_ZERO_CANONICAL_GRAPH_ACTIONS': '0',
        'OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT': '6' if pantry else '3',
        'OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING': '1' if pantry else '0',
        'OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING': '1' if pantry else '0',
        'OAT_ZERO_EVAL_MODE_COVERAGE_SEED': str(EVAL_SEEDS[domain])}


def physical_profile(gpu_class='a100'):
    require(gpu_class in {'a100', 'a6000'}, 'only A100 80GiB / A6000 48GiB profiles are defined')
    env = e80.memory_env()
    env.pop('OAT_ZERO_EVAL_BATCH_SIZE')  # Evaluation is scientific, not a physical tuning knob.
    env.update({'OAT_ZERO_VLLM_GPU_RATIO': '0.25' if gpu_class == 'a100' else '0.40',
        'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '1', 'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'})
    require(set(env) == PHYSICAL_ENV_KEYS, 'physical profile unexpectedly changes scientific settings')
    return {'name': 'e124_conservative_' + gpu_class, 'qualified': False,
        'qualification_required': True, 'profile_environment': env,
        'resources': {'gpus': 1, 'gpu_class': gpu_class, 'gpu_memory_gib': 80 if gpu_class == 'a100' else 48,
                      'cpus': 8, 'memory_gib': 256}}


def common_environment():
    env = {
        'OAT_ZERO_MODEL': MODEL_ALIAS, 'OAT_ZERO_SEED': str(SEED),
        'OAT_ZERO_MAX_TRAIN': str(TRAIN_ROWS), 'OAT_ZERO_NUM_PROMPT_EPOCH': str(PASSES),
        'OAT_ZERO_MAX_PROMPT_EPOCHS': str(PASSES), 'OAT_ZERO_MAX_QUERIES': '100000000',
        'OAT_ZERO_TRAIN_BATCH_SIZE': '16', 'OAT_ZERO_ROLLOUT_BATCH_SIZE': '1',
        'OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE': '1', 'OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE': '16',
        'OAT_ZERO_SYNC_PARAMS_EVERY': '1', 'OAT_ZERO_COLLOCATE': '1',
        'OAT_ZERO_EVAL_TEMPERATURE': '0.0', 'OAT_ZERO_EVAL_MODE_COVERAGE_K': '8',
        'OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE': '1.0',
        'OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P': '1.0', 'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS': '4',
        'OAT_ZERO_SAVE_STEPS': str(CHECKPOINT_INTERVAL), 'OAT_ZERO_SAVE_FROM': str(CHECKPOINT_INTERVAL),
        'OAT_ZERO_RESUME_STEPS': str(CHECKPOINT_INTERVAL), 'OAT_ZERO_RESUME_FROM': str(CHECKPOINT_INTERVAL),
        'OAT_ZERO_EXPORT_STEPS': '0', 'OAT_ZERO_MAX_EXPORT_NUM': '1',
        'OAT_ZERO_MAX_SAVE_NUM': '1', 'OAT_ZERO_MAX_RESUME_NUM': '1',
        'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS': '1', 'OAT_ZERO_AUTO_RESUME': '1',
        'OAT_ZERO_WATCHDOG_REQUEUE': '1', 'OAT_ZERO_WATCHDOG_STALE_SECONDS': '7200',
        'OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS': '3600', 'OAT_ZERO_WATCHDOG_MAX_RESTARTS': '12',
        'OAT_ZERO_WATCHDOG_LOG_PROGRESS': '1', 'OAT_ZERO_WATCHDOG_ARTIFACT_PROGRESS': '1',
        'OAT_ZERO_USE_WB': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'VLLM_USE_V1': '0', 'PYTHONDONTWRITEBYTECODE': '1',
        'OAT_ZERO_ONLINE_CANONICAL_KEY_MODE': 'modebench_outcome',
        'OAT_ZERO_VERIFIED_DISCOVERY_TRACKING': '1',
        'OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING': 'uniform',
        'OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP': '0',
        'OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_NORMALIZED': '0',
        'OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE': '0',
        'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY': '0',
        'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS': '0',
        'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS': '0',
        'OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS': '0',
        'OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY': '0',
        'OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP': '5.0',
        'OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT': '1.0',
        'OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT': '1.0',
        'OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP': '5.0',
        'OAT_ZERO_XDR_TAU': 'inf', 'OAT_ZERO_XDR_TASK_ADVANTAGE_WEIGHTS': '0',
        'OAT_ZERO_XDR_TAU_CONTROL_TARGET_RATIO': '0.0', 'OAT_ZERO_XDR_SAC_DUAL_TARGET_RATIO': '0.0',
        'OAT_ZERO_XDR_MODE_ADAPTIVE': '0', 'OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION': '0'}
    env.update(e80.optimizer_env())
    # Apply after every inherited overlay. The wrapper otherwise caps 192 to 96.
    env.update({'OAT_ZERO_EVAL_PROMPT_INTERVAL': str(EVALUATION_INTERVAL),
        'OAT_ZERO_EVAL_BATCH_SIZE': str(EVAL_BATCH_SIZE), 'OAT_ZERO_ALLOW_SPARSE_EVAL': '1'})
    return env


def scientific_constants():
    return {'schema': SCHEMA, 'model': MODEL_NAME, 'model_revision': MODEL_REVISION,
        'levels': list(LEVELS), 'domains': list(DOMAINS), 'arms': list(ARMS), 'seeds': list(SEEDS),
        'cells': 30, 'train_rows': TRAIN_ROWS, 'eval_rows': EVAL_ROWS, 'passes': PASSES,
        'target_steps': TARGET_STEPS, 'total_target_steps': 30 * TARGET_STEPS,
        'common_environment': common_environment(),
        'interfaces': {str(level): {domain: interface(level, domain) for domain in DOMAINS} for level in LEVELS},
        'objectives': {arm: e119.objective(arm) for arm in ARMS},
        'source_default_requirements': {'adam_epsilon': 1e-8, 'minimum_learning_rate': 1e-8,
            'precision': 'bf16', 'gradient_checkpointing': True,
            'checkpoint_state': ['model', 'optimizer', 'scheduler', 'prompt_cursor', 'replay_bank']},
        'dataset_pins': {'level1': L1_INVENTORY_SHA256, 'level2_identity': L2_IDENTITY_SHA256,
            'level2_reports': {str(k): v for k, v in L2_REPORTS.items()},
            'level3_identity': L3_IDENTITY_SHA256, 'level3_confirmation': L3_REPORT_SHA256},
        'information_boundary': {'treatment_uses_dev': False, 'rematch_for_7b': False,
            'level3_adaptive_confirmation_round': 2, 'statistical_equivalence_claimed': False}}


def scientific_environment(env):
    return {key: value for key, value in sorted(env.items())
            if key not in PHYSICAL_ENV_KEYS | LOCATION_ENV_KEYS}


def build_cells(snapshot_root, model_path, gpu_class='a100', output_root=None):
    """Build deterministic cells only; paths need not exist during review.

    ``output_root`` replaces the campaign directory, not its level/domain/arm
    suffix. Model download/source validation/admission and profile qualification
    are explicit caller gates; merely constructing cells never admits a run.
    """
    snapshot, model = Path(snapshot_root).resolve(), Path(model_path).resolve()
    output = Path(output_root).resolve() if output_root is not None else OUTPUT_ROOT
    profile, constants = physical_profile(gpu_class), scientific_constants()
    constant_seal = sha(constants)
    cells = []
    for level in LEVELS:
        for domain in DOMAINS:
            for arm in ARMS:
                stamp = f'e124_l{level}_{DOMAIN_TAGS[domain]}_{arm}_s{SEED}'
                target = output / f'level{level}' / domain / arm / f's{SEED}'
                env = common_environment()
                env.update(interface(level, domain))
                env.update(e119.objective(arm))
                env.update(profile['profile_environment'])
                env.update({'OAT_ZERO_REPO_ROOT': str(ROOT), 'OAT_ZERO_PRETRAIN': str(model),
                    'OAT_ZERO_SOURCE_ROOT': str(snapshot / 'src'),
                    'OAT_ZERO_OPS_SNAPSHOT_ROOT': str(snapshot / 'ops'),
                    'SAVE_PATH': str(target), 'RUN_STAMP': stamp,
                    'OAT_ZERO_EVAL_PROMPT_INTERVAL': str(EVALUATION_INTERVAL),
                    'OAT_ZERO_EVAL_BATCH_SIZE': str(EVAL_BATCH_SIZE), 'OAT_ZERO_ALLOW_SPARSE_EVAL': '1'})
                require(all(isinstance(v, str) and ',' not in v and '\n' not in v for v in env.values()),
                        'environment contains an unsafe Slurm export delimiter')
                science = {'level': level, 'domain': domain, 'arm': arm, 'seed': SEED,
                    'model': MODEL_NAME, 'model_revision': MODEL_REVISION,
                    'constants_sha256': constant_seal, 'environment': scientific_environment(env)}
                cells.append({'level': level, 'domain': domain, 'dataset_domain': DOMAIN_DIR[domain],
                    'arm': arm, 'seed': SEED, 'run_stamp': stamp, 'run_dir': str(target),
                    'target_steps': TARGET_STEPS, 'environment': dict(sorted(env.items())),
                    'resources': deepcopy(profile['resources']), 'runtime_profile': deepcopy(profile),
                    'scientific_identity': science, 'scientific_sha256': sha(science),
                    'constants_sha256': constant_seal,
                    'proof': {'dataset_admission_required': True, 'model_validation_required': True,
                        'source_freeze_required': True, 'physical_qualification_required': True,
                        'treatment_outcomes_used_for_selection': False}})
    require(len(cells) == len({c['run_dir'] for c in cells}) == len({c['run_stamp'] for c in cells}) == 30,
            'E124 must have 30 distinct run directories and identities')
    return cells


def _tree(path):
    path = Path(path)
    require(path.is_dir(), f'missing dataset directory: {path}')
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob('*')) if p.is_file()}


def _rows(path, subset):
    """Read Arrow in saved shard order; never invoke a dataset generator/cache."""
    import pyarrow.ipc as ipc
    path = Path(path)
    metadata = read(path / 'dataset_dict.json')
    require(subset in metadata['splits'], f'missing {subset} dataset: {path}')
    state = read(path / subset / 'state.json')
    shards = [item['filename'] for item in state['_data_files']]
    require(shards and len(shards) == len(set(shards)), 'missing or repeated dataset shards')
    rows = []
    for name in shards:
        require(Path(name).name == name and name.endswith('.arrow'), 'unsafe dataset shard path')
        with (path / subset / name).open('rb') as stream:
            rows.extend(ipc.open_stream(stream).read_all().to_pylist())
    return rows


def _row_hash(rows):
    return hashlib.sha256('\n'.join(json.dumps(row, sort_keys=True, separators=(',', ':'))
                                    for row in rows).encode()).hexdigest()


def _identities(domain, rows):
    if domain == 'pantry_plan':
        return {('pantry', row['instance_fingerprint']) for row in rows}
    from materialize_e117_evaluation_reserves import _identities as canonical_identities
    return canonical_identities(domain, rows)


def _validate_splits(level, domain, records=None):
    splits = ('train', 'eval') if level == 1 else ('train', 'dev', 'eval')
    data = data_directory(level, domain)
    evidence, identities = {}, {}
    for split in splits:
        path, expected = data / split, TRAIN_ROWS if split == 'train' else EVAL_ROWS
        rows = _rows(path, 'train' if split == 'train' else 'multi_answer')
        ids = _identities(domain, rows)
        require(len(rows) == len(ids) == expected, f'{level}/{domain}/{split}: row count or unique identities differ')
        row_hash = _row_hash(rows)
        if records is not None:
            record = records[split]
            require(record['rows'] == expected and record['rows_sha256'] == row_hash,
                    f'{level}/{domain}/{split}: published row hash drift')
            require(record['checks'] and all(v is True for v in record['checks'].values()),
                    f'{level}/{domain}/{split}: canonical structural checks failed')
            histogram = {str(k): v for k, v in Counter(int(r['answer_mode_count']) for r in rows).items()}
            expected_histogram = record.get('answer_mode_count_histogram', record.get('support_histogram'))
            require(histogram == expected_histogram, f'{level}/{domain}/{split}: support histogram drift')
        tree = _tree(path)
        evidence[split] = {'path': str(path), 'rows': expected, 'rows_sha256': row_hash,
                          'files_sha256': tree, 'inventory_sha256': sha(tree)}
        identities[split] = ids
    for first in splits:
        for second in splits:
            if first < second:
                require(not identities[first] & identities[second], f'{level}/{domain}: dataset splits overlap')
    return evidence


def native_level1_proof():
    """Compare the fixed native interfaces to archived E118 Qwen3B exports."""
    rows = read(L1_LEDGER)['runs']
    proof = {}
    absent_in_old_exports = {'OAT_ZERO_DATA_ROOT', 'OAT_ZERO_REQUIRE_EXISTING_DATA',
                            'OAT_ZERO_MODEBENCH_DOMAIN', 'OAT_ZERO_MODEBENCH_SYNTAX_PROFILE'}
    for domain in DOMAINS:
        candidates = [row for row in rows if row['domain'] == domain and row['arm'] == 'maxrl' and int(row['seed']) == SEED]
        require(len(candidates) == 1, f'one archived E118 seed70 template required for {domain}')
        tokens = shlex.split(candidates[0]['held_scheduler_record'].split('SubmitLine=', 1)[1])
        exports = [token[len('--export='):] for token in tokens if token.startswith('--export=')]
        require(len(exports) == 1, 'one archived export argument required')
        env = dict(item.split('=', 1) for item in exports[0].split(',') if '=' in item)
        expected = {key: value for key, value in interface(1, domain).items() if key not in absent_in_old_exports}
        require({key: env.get(key) for key in expected} == expected, f'E118 native {domain} interface drift')
        proof[domain] = {'interface': expected, 'interface_sha256': sha(expected)}
    return {'ledger_path': str(L1_LEDGER), 'semantic_templates': proof, 'sha256': sha(proof)}


def dataset_admission():
    """Authenticate canonical L1/L2/L3 benchmarks and return sealable inventories.

    L2 immutable row hashes bind its historical generation/disjointness checks;
    those checks are not recomputed against subsequently created candidate pools.
    L3 uses the complete existing canonical fixed-reference confirmation audit.
    This is benchmark provenance, not a new 7B base-model viability claim.
    """
    evidence = {'schema': 'e124_dataset_admission_v1', 'native_level1': native_level1_proof(), 'levels': {}}
    l1 = {}
    for domain in DOMAINS:
        data = data_directory(1, domain)
        inventory = {f'{split}/{name}': value for split in ('train', 'eval') for name, value in _tree(data / split).items()}
        require(sha(inventory) == L1_INVENTORY_SHA256[domain], f'Level 1 {domain} dataset inventory drift')
        l1[domain] = {'inventory_sha256': sha(inventory), 'files_sha256': inventory,
                      'splits': _validate_splits(1, domain)}
    evidence['levels']['1'] = {'status': 'native_e118_dataset_verified', 'domains': l1}
    identity_path = DATA_ROOTS[2] / 'identity.json'
    require(digest(identity_path) == L2_IDENTITY_SHA256, 'Level 2 identity drift')
    identity = read(identity_path)
    require(identity['split_sizes'] == {'train': 384, 'dev': 128, 'eval': 128}
            and set(identity['domains']) == set(DOMAIN_DIR.values()), 'Level 2 exact all-five dataset required')
    reports = {}
    for path, expected_hash in L2_REPORTS.items():
        require(digest(path) == expected_hash, f'Level 2 admission report drift: {path}')
        report = read(path)
        require(report.get('status') == 'pass'
                and report.get('decision') == 'admit_all_domains_for_treatment_training'
                and report.get('domain_decisions') == {d: 'admit' for d in DOMAIN_DIR.values()},
                f'Level 2 requires both all-five admission reports: {path}')
        reports[str(path)] = expected_hash
    evidence['levels']['2'] = {'status': 'frozen_r5_admission_verified', 'identity_path': str(identity_path),
        'identity_sha256': L2_IDENTITY_SHA256, 'reports_sha256': reports,
        'domains': {domain: _validate_splits(2, domain, identity['domains'][DOMAIN_DIR[domain]]) for domain in DOMAINS}}
    import launch_e122_level3_factorial as e122
    require(digest(DATA_ROOTS[3] / 'identity.json') == L3_IDENTITY_SHA256, 'Level 3 identity drift')
    canonical = e122.admission_proof(required=True)
    require(canonical['status'] == 'matched_fixed_reference' and canonical['sha256'] == L3_REPORT_SHA256,
            'Level 3 requires the pinned completed v3 fixed-reference confirmation')
    identity = read(DATA_ROOTS[3] / 'identity.json')
    evidence['levels']['3'] = {'status': 'matched_fixed_reference', 'canonical_confirmation': canonical,
        'identity_sha256': L3_IDENTITY_SHA256,
        'domains': {domain: _validate_splits(3, domain, identity['domains'][DOMAIN_DIR[domain]]) for domain in DOMAINS}}
    evidence['information_boundary'] = scientific_constants()['information_boundary'] | {
        'new_7b_viability_or_matching_claimed': False, 'dev_used_by_treatment': False}
    # Flat absolute pins allow frozen supervisors to recheck bytes without
    # importing mutable launchers or rerunning expensive admission analysis.
    files = dict(canonical['files_sha256'])
    directories = deepcopy(canonical['directory_files'])
    files[str(identity_path)] = L2_IDENTITY_SHA256
    files.update(reports)
    for level in LEVELS:
        for domain, domain_evidence in evidence['levels'][str(level)]['domains'].items():
            splits = domain_evidence['splits'] if level == 1 else domain_evidence
            for split_evidence in splits.values():
                directory = split_evidence['path']
                inventory = split_evidence['files_sha256']
                directories[directory] = sorted(str((Path(directory) / name).resolve()) for name in inventory)
                for relative, file_hash in inventory.items():
                    absolute = str(Path(directory) / relative)
                    require(absolute not in files or files[absolute] == file_hash,
                            'canonical proof and actual dataset file pins disagree')
                    files[absolute] = file_hash
    evidence['files_sha256'] = dict(sorted(files.items()))
    evidence['directory_files'] = dict(sorted(directories.items()))
    evidence['sha256'] = sha(evidence)
    return evidence


def dataset_inventory():
    """Alias retaining the same complete admission gates; no inventory-only bypass."""
    return dataset_admission()
