#!/usr/bin/env python3
"""Isolated E124 single-A6000 qualification; never submits or releases Slurm jobs.

The sole candidate is CPUAdam + activation offload + microbatch1 + OMP4.
Synthetic stress is systems evidence only; production dataset outcomes never
select a recipe. Completed stages are reusable; partial/failed stages require
explicit reconciliation and are never silently retried.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickletools
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path(os.environ.get('OAT_ZERO_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
SOURCE = Path(__file__).resolve()
GIB = 1024 ** 3
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
ARMS = ('maxrl', 'replay_maxrl')
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
PROFILE = {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '1', 'OAT_ZERO_ADAM_OFFLOAD': '1',
           'OAT_ZERO_ACTIVATION_OFFLOADING': '1', 'OMP_NUM_THREADS': '4',
           'OAT_ZERO_VLLM_GPU_RATIO': '0.40'}
PROFILE_ID = 'a6000_cpu_adam_mb1_omp4'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def cgroup_sample():
    """Read this process's whole job ancestor, including actor shared mappings."""
    try:
        line = next(l for l in Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::'))
        path = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
        job = next((p for p in [path, *path.parents] if re.match(r'job[_-]\d+', p.name)), path)
        result = {'path': str(job)}
        for name in ('memory.current', 'memory.peak', 'memory.high', 'memory.max'):
            s = (job / name).read_text().strip()
            result[name] = int(s) if s != 'max' else None
        for name in ('memory.stat', 'memory.events'):
            result[name] = {k: int(v) for k, v in (l.split() for l in (job / name).read_text().splitlines())}
        result['noncache_bytes'] = sum(result['memory.stat'].get(k, 0) for k in ('anon', 'shmem', 'kernel'))
        result['nonreclaimable_dirty_bytes'] = result['noncache_bytes'] + sum(result['memory.stat'].get(k, 0) for k in ('file_dirty', 'file_writeback'))
        return result
    except (OSError, ValueError, StopIteration):
        return {'unavailable': True}


class MemorySampler:
    def __init__(self):
        self.samples = []
        self.done = threading.Event()

    def __enter__(self):
        self.samples.append(cgroup_sample())
        def loop():
            while not self.done.wait(1):
                self.samples.append(cgroup_sample())
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        self.samples.append(cgroup_sample())

    def report(self):
        samples = [x for x in self.samples if not x.get('unavailable')]
        if not samples:
            return {'available': False}
        events = {k: samples[-1]['memory.events'].get(k, 0) - samples[0]['memory.events'].get(k, 0) for k in ('high', 'oom', 'oom_kill')}
        return {'available': True, 'scope': 'whole job cgroup; includes cache; no RSS double counting', 'path': samples[0]['path'],
                'peak_current_bytes': max(x['memory.current'] for x in samples), 'peak_noncache_bytes': max(x['noncache_bytes'] for x in samples),
                'peak_nonreclaimable_dirty_bytes': max(x['nonreclaimable_dirty_bytes'] for x in samples), 'cgroup_peak_bytes': max(x['memory.peak'] for x in samples), 'events_delta': events, 'sample_count': len(samples)}


def production_update(learner, input_ids, prompt_tokens, arm):
    """Use the actual production loss/replay method, not a surrogate loss."""
    import torch
    from oat_drgrpo.maxrl import binary_maxrl_advantages
    from oat_drgrpo.online_canonical_bank import VerifiedCanonicalReplayGroup
    a = learner.args
    a.online_canonical_replay = True
    a.maxrl_task_objective = 'maxrl' in arm
    a.online_canonical_replay_compute_only = not arm.startswith('replay_')
    attention = torch.ones_like(input_ids)
    masks = torch.zeros((16, input_ids.shape[1] - 1), device=input_ids.device)
    masks[:, prompt_tokens - 1:] = 1
    rewards = torch.tensor([1.] * 8 + [0.] * 8, device=input_ids.device)
    advantages = binary_maxrl_advantages(rewards[None])[0] if a.maxrl_task_objective else rewards - rewards.mean()
    old = torch.zeros_like(masks)
    with torch.no_grad():
        for start in range(0, 16, a.train_batch_size_per_device):
            stop = start + a.train_batch_size_per_device
            logits = learner.model(input_ids[start:stop], attention_mask=attention[start:stop])['logits']
            old[start:stop], _ = learner._policy_logps_and_optional_entropy(logits, input_ids[start:stop], masks[start:stop], need_entropy=False)
            del logits
    group = VerifiedCanonicalReplayGroup(prompt_token_ids=tuple(input_ids[0, :prompt_tokens].tolist()),
        response_token_ids=tuple(tuple(r[prompt_tokens:].tolist()) for r in input_ids), outcome_keys=tuple(f'synthetic_stress_{i}' for i in range(16)), fresh_observation_counts=(1,) * 16)
    info = learner._baseline_update_with_precomputed_advantages(input_ids=input_ids, att_mask=attention,
        prompt_id_lens=[prompt_tokens] * 16, loss_masks=torch.ones(16, device=input_ids.device), response_masks=masks,
        logps=old, ref_logps=None, advantages=advantages[:, None], final_rewards=rewards[:, None],
        policy_vocab_upper_bound=learner._resolve_scoring_vocab_upper_bound(learner.model), canonical_replay_groups=[group])
    required = ('canonical_replay_compute_only', 'canonical_replay_weighted_loss',
                'canonical_replay_raw_weighted_loss', 'canonical_replay_applied_score_gradient_l2')
    require(all(key in info and bool(torch.isfinite(info[key]).all()) for key in required),
            'missing or nonfinite production replay diagnostics')
    require(float(info['canonical_replay_compute_only']) == float(a.online_canonical_replay_compute_only),
            'production replay compute-only setting differs')
    if a.online_canonical_replay_compute_only:
        require(float(info['canonical_replay_weighted_loss']) == 0.
                and float(info['canonical_replay_applied_score_gradient_l2']) == 0.,
                'compute-only replay applied a nonzero derivative')
    else:
        require(float(info['canonical_replay_applied_score_gradient_l2']) > 0.,
                'eligible live replay must apply a finite nonzero derivative')
    return info


def full_training_state(engine, scheduler):
    """Hash all model/master/moment bytes in bounded chunks for restoration."""
    import torch
    def freeze(value):
        if isinstance(value, torch.Tensor):
            flat = value.detach().reshape(-1)
            h = hashlib.sha256()
            width = max(1, (8 * 1024**2) // value.element_size())
            for start in range(0, flat.numel(), width):
                h.update(flat[start:start + width].contiguous().cpu().view(torch.uint8).numpy().tobytes())
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': h.hexdigest()}
        if isinstance(value, dict):
            return {str(key): freeze(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [freeze(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f'unsupported training state value: {type(value).__name__}')
    base = engine.optimizer.optimizer
    return {'model': freeze(engine.module.state_dict()),
            'optimizer': freeze(base.state_dict()),
            'fp32_master_parameters': freeze([group['params'] for group in base.param_groups]),
            'scheduler': freeze(scheduler.state_dict()), 'global_steps': int(engine.global_steps)}


def optimizer_sketch(engine):
    """Sample FP32 optimizer moments, avoiding rounded-away BF16 weight deltas."""
    import torch
    base = engine.optimizer.optimizer
    report = {'optimizer_class': type(base).__module__ + '.' + type(base).__qualname__, 'states': []}
    for index, state in enumerate(base.state.values()):
        item = {'index': index}
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                flat = value.detach().flatten()
                ix = torch.linspace(0, max(0, flat.numel() - 1), min(256, flat.numel()), device=flat.device).long()
                item[key] = {'numel': flat.numel(), 'sample': flat[ix].float().cpu().tolist()}
            elif value is not None:
                item[key] = float(value)
        report['states'].append(item)
    return report


def compare_sketch(reference, candidate, tolerance=0.03):
    errors = []
    require(len(reference['states']) == len(candidate['states']) > 0, 'optimizer state grouping differs or empty')
    for a, b in zip(reference['states'], candidate['states']):
        for key in ('exp_avg', 'exp_avg_sq'):
            require(a[key]['numel'] == b[key]['numel'], 'optimizer state shape differs')
            av, bv = a[key]['sample'], b[key]['sample']
            require(len(av) == len(bv) and av, 'missing optimizer moment sketch')
            require(all(math.isfinite(v) for v in av + bv), 'nonfinite optimizer moment')
            denom = max(sum(v * v for v in av), 1e-30)
            errors.append(math.sqrt(sum((x-y)**2 for x, y in zip(av, bv)) / denom))
    maximum = max(errors)
    return {'passed': maximum <= tolerance, 'max_relative_rms': maximum, 'tolerance': tolerance,
            'scope': 'deterministic FP32 moment sketches; mathematical accumulation equivalence separately tested on production loss', 'full_tensor_bitwise_equivalence_claimed': False}


def allocated_gpu_uuid(torch):
    device = torch.cuda.get_device_properties(0)
    value = str(getattr(device, 'uuid', ''))
    if value.startswith('GPU-'):
        return value
    # Older torch builds omit the UUID property. Match this process's CUDA
    # context to NVML's process table rather than interpreting remapped indices.
    probe = torch.empty(1, device='cuda'); torch.cuda.synchronize()
    record = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    found = {row.split(',')[1].strip() for row in record.stdout.splitlines() if len(row.split(',')) == 2 and row.split(',')[0].strip() == str(os.getpid())}
    del probe
    require(len(found) == 1, 'cannot resolve the allocated GPU UUID from process context')
    return found.pop()


def cell_key(cell):
    return f"l{int(cell['level'])}_{cell['domain']}_{cell['arm']}"


def extract_cells(manifest):
    cells = manifest.get('cells', manifest.get('planned_cells', manifest.get('runs', [])))
    require(len(cells) == 30, 'exactly30 cells required')
    wanted = {(level, domain, arm) for level in (1, 2, 3) for domain in DOMAINS for arm in ARMS}
    require({(int(c['level']), c['domain'], c['arm']) for c in cells} == wanted, 'missing/duplicate level-domain-arm cell')
    require(len({int(c['seed']) for c in cells}) == 1, 'one common seed required')
    normalized = []
    for cell in cells:
        c = dict(cell); c['level'] = int(c['level'])
        env = c.get('environment', c.get('env'))
        require(isinstance(env, dict), 'resolved environment missing')
        c['environment'] = {k: str(v) for k, v in env.items()}
        e = c['environment']
        expected = {'OAT_ZERO_NUM_SAMPLES': '16', 'OAT_ZERO_TRAIN_BATCH_SIZE': '16',
                    'OAT_ZERO_ROLLOUT_BATCH_SIZE': '1', 'OAT_ZERO_NUM_PPO_EPOCHS': '1',
                    'OAT_ZERO_NUM_PROMPT_EPOCH': '8', 'OAT_ZERO_MAX_TRAIN': '384',
                    'OAT_ZERO_N_GPU': '1', 'OAT_ZERO_NUM_GPUS_PER_ACTOR': '1',
                    'OAT_ZERO_MAXRL_TASK_OBJECTIVE': '1',
                    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE': 'verified_likelihood_per_rollout',
                    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY': '0' if c['arm'] == 'replay_maxrl' else '1'}
        for key, value in expected.items():
            require(e.get(key) == value, f'{cell_key(c)}: scientific contract differs at{key}')
        require(Path(e['OAT_ZERO_PRETRAIN']).name == REVISION, 'wrong pinned7B revision')
        require('Qwen2.5-7B-Instruct' in e['OAT_ZERO_PRETRAIN'], 'wrong model family/size')
        require(e.get('OAT_ZERO_SEED') == str(c['seed']), 'seed export differs')
        require(bool(e.get('RUN_STAMP')), 'production runtime stamp missing')
        require(int(e.get('OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS', '0')) == 4, 'four evaluation draws required')
        require(int(e.get('OAT_ZERO_EVAL_MODE_COVERAGE_K', '0')) == 8, 'K8 evaluation required')
        for key, value in PROFILE.items():
            require(e.get(key) == value, f'sole runtime profile differs at{key}')
        normalized.append(c)
    return sorted(normalized, key=cell_key)


def file_inventory(root):
    root = Path(root)
    require(root.is_dir(), f'missing immutable input directory:{root}')
    files = sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    require(files, f'empty input directory:{root}')
    return {str(p.absolute()): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files}


def prepare(manifest_path, output):
    manifest_path = Path(manifest_path).resolve(); output = Path(output).resolve()
    require(not (output / 'plan.json').exists(), 'refuse to overwrite prepared benchmark')
    m = read(manifest_path); cells = extract_cells(m)
    roots = {c['environment']['OAT_ZERO_REPO_ROOT'] for c in cells}
    require(len(roots) == 1, 'one true workspace root required')
    root = Path(next(iter(roots))).resolve()
    snapshots = {str(Path(c['environment']['OAT_ZERO_SOURCE_ROOT']).parent) for c in cells}
    require(len(snapshots) == 1, 'one immutable campaign runtime required')
    snapshot = Path(next(iter(snapshots)))
    require(snapshot == Path(m['snapshot']['root']), 'snapshot identity mismatch')
    require(re.fullmatch(r'[0-9a-f]{64}', m['snapshot']['sha256']), 'snapshot digest missing')
    ops = {c['environment']['OAT_ZERO_OPS_SNAPSHOT_ROOT'] for c in cells}
    require(ops == {str(snapshot / 'ops')}, 'source and ops snapshots differ')
    source = snapshot / 'src'; pins = file_inventory(source)
    for name in ('train.sh', 'repo_env.sh'):
        p = snapshot / 'ops' / name; pins[str(p)] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
    control = snapshot / 'control' / SOURCE.name
    require(control.is_file() and digest(control) == digest(SOURCE), 'frozen benchmark control differs')
    pins[str(control)] = {'bytes': control.stat().st_size, 'sha256': digest(control)}
    model = Path(cells[0]['environment']['OAT_ZERO_PRETRAIN'])
    require(all(c['environment']['OAT_ZERO_PRETRAIN'] == str(model) for c in cells), 'model path differs')
    model_files = file_inventory(model)
    require((model / 'config.json').is_file() and (model / 'tokenizer.json').is_file(), 'incomplete pinned model')
    index = read(model / 'model.safetensors.index.json')
    require(all((model / shard).is_file() for shard in set(index['weight_map'].values())), 'model shards missing')
    data = {}
    for c in cells:
        for key in ('OAT_ZERO_PROMPT_DATA', 'OAT_ZERO_EVAL_DATA'):
            path = c['environment'][key]
            if path not in data: data[path] = file_inventory(path)
    replay = [c for c in cells if c['arm'] == 'replay_maxrl']
    worst = max((c for c in cells if c['arm'] == 'maxrl'), key=lambda c: (int(c['environment']['OAT_ZERO_PROMPT_MAX_LENGTH']) + int(c['environment']['OAT_ZERO_GENERATE_MAX_LENGTH']), c['level'], c['domain']))
    coverage = [cell_key(c) for c in replay] + [cell_key(worst)]
    plan = {'schema': 'e124_qwen7b_benchmark_plan_v1', 'workspace_root': str(root), 'output_root': str(output),
            'manifest_path': str(manifest_path), 'manifest_sha256': digest(manifest_path), 'benchmark_sha256': digest(SOURCE), 'cells': cells,
            'runtime': {'snapshot_root': str(snapshot), 'source_root': str(source), 'ops_root': str(snapshot / 'ops'), 'identity_sha256': m['snapshot']['sha256'], 'files': pins},
            'model': {'revision': REVISION, 'path': str(model), 'files': model_files,
                      'weight_bytes': sum((model / shard).stat().st_size for shard in set(index['weight_map'].values()))}, 'datasets': data,
            'profile_id': PROFILE_ID, 'profile_environment': PROFILE, 'resources': {'gpu_class': 'A6000', 'gpus': 1, 'host_memory_gib': 256, 'cpus': 8},
            'coverage': coverage, 'worst_cell': cell_key(worst), 'scientific_horizon_steps': 3072,
            'synthetic_stress_is_scientific_outcome': False, 'outcomes_used_for_selection': False,
            'gates': {'host_headroom_fraction': .20, 'host_headroom_bytes': 8 * GIB, 'gpu_headroom_bytes': 2 * GIB},
            'partial_stage_policy': 'stop for review; completed stages only may be reused; no blind retry'}
    plan['plan_sha256'] = identity(plan); write(output / 'plan.json', plan)
    return plan


def verify_pins(plan, *, full=True):
    require(identity({k: v for k, v in plan.items() if k != 'plan_sha256'}) == plan['plan_sha256'], 'plan digest changed')
    require(plan['profile_environment'] == PROFILE and plan['coverage'], 'candidate or coverage changed')
    require(digest(SOURCE) == plan['benchmark_sha256'], 'executing benchmark differs from frozen control')
    require(digest(plan['manifest_path']) == plan['manifest_sha256'], 'manifest changed')
    groups = [plan['runtime']['files']]
    if full: groups += [plan['model']['files'], *plan['datasets'].values()]
    for files in groups:
        for path, frozen in files.items():
            require(Path(path).is_file() and Path(path).stat().st_size == frozen['bytes'] and digest(path) == frozen['sha256'], f'frozen input changed:{path}')


def runtime_environment(plan):
    root = Path(plan['workspace_root'])
    return {'OAT_ZERO_REPO_ROOT': str(root), 'MAXENT_GRPO_ROOT': str(root), 'ROOT_DIR': str(root),
            'MAXENT_GRPO_VAR_ROOT': str(root / 'var'), 'PYTHONDONTWRITEBYTECODE': '1',
            'CUDA_HOME': str(root / 'var/cuda124_toolkit'), 'CUDA_PATH': str(root / 'var/cuda124_toolkit'),
            'PYTHONPATH': plan['runtime']['source_root'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'TOKENIZERS_PARALLELISM': 'false', 'TRANSFORMERS_NO_TF': '1', 'USE_TF': '0', 'USE_FLAX': '0', 'VLLM_USE_V1': '0'}


def clean_environment(plan, resolved):
    env = {k: v for k, v in os.environ.items() if not k.startswith('OAT_ZERO_') and k not in ('SAVE_PATH', 'RUN_STAMP', 'PYTORCH_CUDA_ALLOC_CONF')}
    env.update(resolved); env.update(runtime_environment(plan))
    require(Path(env['CUDA_HOME'], 'bin/nvcc').is_file(), 'workspace CUDA toolkit missing')
    return env


def torch_imports(plan):
    import ctypes
    os.environ.update(runtime_environment(plan)); os.environ.update(PROFILE)
    sys.path.insert(0, plan['runtime']['source_root'])
    root = Path(plan['workspace_root'])
    for base in (Path(sys.base_prefix) / 'lib', root / 'var/seed_paper_eval/paper310/lib'):
        library = base / f'libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0'
        if library.is_file(): ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL); break
    import torch
    from oat_drgrpo import fused_adam_shim
    fused_adam_shim.install()
    return torch


def require_gpu(torch):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one allocated GPU required')
    p = torch.cuda.get_device_properties(0)
    require('A6000' in p.name and 45 * GIB <= p.total_memory <= 49 * GIB, 'only the fixed48GiB A6000 candidate is qualified')
    return p


def checkpoint_metadata(path, step):
    path = Path(path); files = {}; counters = {}; keys = set()
    for p in sorted(path.glob('*.pt')):
        with zipfile.ZipFile(p) as z:
            raw = z.read(next(n for n in z.namelist() if n.endswith('data.pkl'))); tokens = list(pickletools.genops(raw))
            if p.name == 'mp_rank_00_model_states.pt':
                for i, (_, arg, _) in enumerate(tokens):
                    if isinstance(arg, str): keys.add(arg)
                    if arg in ('global_steps', 'global_step', 'prompt_batches_consumed_total'):
                        vals = [v for op, v, _ in tokens[i + 1:i + 4] if op.name.startswith('BININT')]
                        require(vals, 'missing scalar counter'); counters[arg] = vals[0]
        files[p.name] = {'bytes': p.stat().st_size, 'sha256': digest(p), 'metadata_sha256': hashlib.sha256(raw).hexdigest()}
    require('mp_rank_00_model_states.pt' in files and any('optim_states' in n for n in files), 'full optimizer checkpoint missing')
    require(counters == dict.fromkeys(('global_steps', 'global_step', 'prompt_batches_consumed_total'), step), 'saved counters differ')
    require('online_canonical_bank_state' in keys, 'saved bank state missing')
    return {'path': str(path), 'step': step, 'counters': counters, 'bank_state_present': True, 'files': files, 'bytes': sum(f['bytes'] for f in files.values())}


def e2e_environment(plan, cell, output, resume=None):
    env = dict(cell['environment'])
    # Preserve RUN_STAMP: the frozen shell uses it to select Pantry runtime dtype.
    # SAVE_PATH alone isolates every smoke artifact from scientific runs.
    env.update({'SAVE_PATH': str(output),
                'OAT_ZERO_FIXED_EXP_SUFFIX': 'e124benchmark', 'OAT_ZERO_MAX_QUERIES': '32' if resume else '16',
                'OAT_ZERO_SAVE_CKPT': '1', 'OAT_ZERO_SAVE_STEPS': '2', 'OAT_ZERO_SAVE_FROM': '2',
                'OAT_ZERO_RESUME_STEPS': '2', 'OAT_ZERO_RESUME_FROM': '2', 'OAT_ZERO_MAX_RESUME_NUM': '1',
                'OAT_ZERO_EXPORT_STEPS': '-1', 'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS': '0', 'OAT_ZERO_AUTO_RESUME': '0',
                'OAT_ZERO_WATCHDOG_STALE_SECONDS': '0', 'OAT_ZERO_WATCHDOG_REQUEUE': '0', 'OAT_ZERO_USE_WB': '0'})
    env.pop('OAT_ZERO_RESUME_DIR', None); env.pop('OAT_ZERO_RESUME_TAG', None)
    if resume: env.update(OAT_ZERO_RESUME_DIR=str(Path(resume).parent), OAT_ZERO_RESUME_TAG=Path(resume).name)
    return env


def validate_evaluation(draws, env, rows):
    endpoints = {}
    for r in draws:
        if r.get('evaluation_kind') != 'fixed_seed_sampled_k_neutral': continue
        require(r.get('sample_count') == int(env['OAT_ZERO_EVAL_MODE_COVERAGE_K']), 'evaluationK changed')
        require(float(r.get('temperature', -1)) == float(env['OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE']), 'evaluation temperature changed')
        require(float(r.get('top_p', -1)) == float(env['OAT_ZERO_TOP_P']), 'evaluationtop_p changed')
        index = int(r['draw_index']); require(r['seed'] == int(env['OAT_ZERO_EVAL_MODE_COVERAGE_SEED']) + index, 'drawseed changed')
        prompts = r.get('prompts', [])
        require(len(prompts) == rows and {q.get('prompt_index') for q in prompts} == set(range(rows)), 'incomplete evaluation prompts')
        require(all(len(q.get('responses', [])) == int(env['OAT_ZERO_EVAL_MODE_COVERAGE_K']) for q in prompts), 'incomplete evaluation responses')
        for q in prompts:
            for response in q['responses']:
                text = response if isinstance(response, str) else response.get('response', response.get('text', ''))
                require(isinstance(text, str) and '\x00' not in text, 'invalid decoded output')
        require(index not in endpoints.setdefault(r['step'], set()), 'duplicate evaluation draw')
        endpoints[r['step']].add(index)
    require(endpoints and all(v == set(range(int(env['OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS']))) for v in endpoints.values()), 'missing complete draw set')
    return {str(k): sorted(v) for k, v in endpoints.items()}


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def safe_retire_checkpoint(path, output):
    path, output = Path(path), Path(output).resolve()
    require(path.resolve().is_relative_to(output) and not path.is_symlink() and path.name.startswith('step_'), 'unsafe checkpoint cleanup')
    shutil.rmtree(path)


def worker(plan):
    candidate, warmup, steps, domains = PROFILE_ID, 2, 2, list(DOMAINS)
    plan = dict(plan)
    worst = next(c for c in plan["cells"] if cell_key(c) == plan["worst_cell"])
    plan["environments"] = {d: dict(worst["environment"]) for d in DOMAINS}
    plan["dataset"] = {"source": "synthetic systems stress; never scientific outcomes"}
    require(candidate == PROFILE_ID, 'wrong candidate')
    require(domains and len(set(domains)) == len(domains) and all(d in DOMAINS for d in domains), 'unknown or duplicate domain')
    env = dict(plan['environments'][DOMAINS[0]])
    env.update(PROFILE)
    os.environ.update({k: str(v) for k, v in env.items() if k in PROFILE})
    torch = torch_imports(plan)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'benchmark needs exactly one allocated visible GPU')
    props = torch.cuda.get_device_properties(0)
    require_gpu(torch)
    torch.set_num_threads(int(env['OMP_NUM_THREADS']))
    from types import SimpleNamespace
    import functools
    from oat.model import LLM
    from oat.utils.ops import masked_sum
    from oat.utils.deepspeed import get_strategy
    from oat_drgrpo.args import ZeroMathArgs
    from oat_drgrpo.learner.grpo import ZeroMathGrpoMixin
    from oat_drgrpo.learner.base import ZeroMathLearnerBaseMixin
    from transformers import AutoTokenizer, get_scheduler
    class Learner(ZeroMathGrpoMixin, ZeroMathLearnerBaseMixin):
        def _should_skip_baseline_grad_norm_logging(self):
            return True
    a = ZeroMathArgs()
    a.local_rank, a.seed, a.rnd_seed, a.zero_stage, a.bf16 = 0, int(env['OAT_ZERO_SEED']), False, 2, True
    a.train_batch_size, a.num_samples, a.num_ppo_epochs = 16, 16, 1
    a.train_batch_size_per_device = int(env['OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE'])
    a.adam_offload = env['OAT_ZERO_ADAM_OFFLOAD'] == '1'
    a.activation_offloading = True
    a.critic_type, a.beta, a.max_norm, a.temperature = 'drgrpo', 0., 1., 1.
    a.generate_max_length = int(env['OAT_ZERO_GENERATE_MAX_LENGTH'])
    a.online_canonical_replay = True
    a.online_canonical_replay_objective = 'verified_likelihood_per_rollout'
    a.online_canonical_replay_alpha = float(env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA'])
    a.online_canonical_replay_bank_normalized = False
    a.policy_entropy_coef = a.maxent_alpha = 0.
    os.environ.setdefault('RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1'); os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    if 'MASTER_PORT' not in os.environ:
        import socket
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); os.environ['MASTER_PORT'] = str(sock.getsockname()[1])
    result = {'schema': 'e124_qwen7b_stress_result_v1', 'status': 'running', 'scope': 'learner_only_stress', 'candidate': candidate,
              'plan_sha256': plan['plan_sha256'], 'profile_environment': PROFILE, 'runtime': plan['runtime'], 'dataset': plan['dataset'],
              'device': {'name': props.name, 'total_memory_bytes': props.total_memory}, 'domains': {}, 'full_shape_smoke': False,
              'own_checkpoint_resume': False, 'end_to_end_smoke': False}
    output = Path(plan['output_root']) / candidate
    require(not (output / 'stress_result.json').exists(), 'refuse to overwrite candidate evidence')
    output.mkdir(parents=True, exist_ok=True)
    with MemorySampler() as sampler:
        try:
            a, strategy = get_strategy(a); strategy.setup_distributed()
            model = LLM(env['OAT_ZERO_PRETRAIN'], use_flash_attention_2=False, bf16=True)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout): module.p = 0.
            optimizer = strategy.create_optimizer(model, lr=float(env['OAT_ZERO_LEARNING_RATE']), betas=(float(env['OAT_ZERO_ADAM_BETA_1']), float(env['OAT_ZERO_ADAM_BETA_2'])), eps=1e-8, weight_decay=0.)
            scheduler_name = env['OAT_ZERO_LR_SCHEDULER']
            scheduler_kw = {'min_lr': float(env['OAT_ZERO_LEARNING_RATE']) * .1} if scheduler_name == 'cosine_with_min_lr' else {}
            scheduler = get_scheduler(scheduler_name, optimizer, num_warmup_steps=math.ceil(3072 * float(env['OAT_ZERO_LR_WARMUP_RATIO'])), num_training_steps=3072, scheduler_specific_kwargs=scheduler_kw)
            model, optimizer, scheduler = strategy.prepare((model, optimizer, scheduler))
            learner = Learner(); learner.args = a; learner.model = model; learner.strategy = strategy
            learner.optimizer, learner.scheduler = optimizer, scheduler
            learner.tokenizer = AutoTokenizer.from_pretrained(env['OAT_ZERO_PRETRAIN'], local_files_only=True)
            learner.masked_aggregator = functools.partial(masked_sum, constant_normalizer=a.generate_max_length)
            learner._baseline_grad_norm_logging_disabled_warned = False
            learner._invalid_scoring_token_ids_warned_contexts = set(); learner._invalid_logit_columns_warned_contexts = set()
            learner._canonical_action_token_ids = None; learner._canonical_action_space = None
            engine = model.model
            torch.cuda.reset_peak_memory_stats()
            for domain_index, domain in enumerate(domains):
                de = plan['environments'][domain]
                prompt = int(de['OAT_ZERO_PROMPT_MAX_LENGTH']); response = int(de['OAT_ZERO_GENERATE_MAX_LENGTH'])
                generator = torch.Generator(device='cpu').manual_seed(7100 + DOMAINS.index(domain))
                ids = torch.randint(100, 1000, (16, prompt + response), generator=generator).cuda()
                # The model sees a shared prompt and 16 distinct synthetic responses.
                ids[:, :prompt] = ids[0, :prompt]
                times = []; checks = []
                domain_warmup, domain_steps = (warmup, steps) if domain_index == 0 else (0, 1)
                for i in range(domain_warmup + domain_steps):
                    before = engine.global_steps
                    torch.cuda.synchronize(); started = time.monotonic()
                    info = production_update(learner, ids, prompt, 'replay_maxrl')
                    torch.cuda.synchronize(); elapsed = time.monotonic() - started
                    require(engine.global_steps == before + 1, 'physical microbatch changed effective optimizer count')
                    norm = float(engine.get_global_grad_norm())
                    require(math.isfinite(norm) and norm > 0, 'full-model Re:MaxRL gradient is zero or nonfinite')
                    require(all(bool(torch.isfinite(v).all()) for v in info.values() if isinstance(v, torch.Tensor)), 'nonfinite production update')
                    if domain_index == i == 0:
                        result['fixed_update_sketch'] = optimizer_sketch(engine)
                    if i >= domain_warmup: times.append(elapsed)
                    checks.append({'optimizer_step': int(engine.global_steps), 'replay_rows': 16, 'fresh_rows': 16, 'gradient_norm': norm})
                    write(output / 'progress.json', {'domain': domain, 'completed_updates': i + 1, 'elapsed_seconds': elapsed, 'candidate': candidate})
                result['domains'][domain] = {'prompt_tokens': prompt, 'response_tokens': response, 'replay_rows': 16, 'fresh_rows': 16,
                    'warmup_updates': domain_warmup, 'measured_updates': domain_steps, 'timing_role': 'matched_throughput' if domain_index == 0 else 'full_shape_stress', 'median_update_seconds': statistics.median(times), 'update_seconds': times, 'effective_update_checks': checks}
                del ids
            # Exercise compute-only replay through the same full model once.
            prompt = int(env['OAT_ZERO_PROMPT_MAX_LENGTH']); response = int(env['OAT_ZERO_GENERATE_MAX_LENGTH'])
            ids = torch.randint(100, 1000, (16, prompt + response), generator=torch.Generator().manual_seed(42)).cuda(); ids[:, :prompt] = ids[0, :prompt]
            control_before = engine.global_steps
            info = production_update(learner, ids, prompt, 'maxrl')
            require(engine.global_steps == control_before + 1, 'MaxRL effective optimizer count changed')
            control_norm = float(engine.get_global_grad_norm())
            require(math.isfinite(control_norm) and control_norm > 0, 'full-model MaxRL gradient is zero or nonfinite')
            result['maxrl_gradient_norm'] = control_norm
            require(engine.global_steps > 0, 'MaxRL optimizer not initialized')
            result['full_shape_arms'] = list(ARMS)
            result['control_invariants'] = {k: float(info[k]) for k in
                ('canonical_replay_compute_only', 'canonical_replay_weighted_loss',
                 'canonical_replay_raw_weighted_loss', 'canonical_replay_applied_score_gradient_l2')}
            import random
            import numpy as np
            client = {'benchmark_plan_sha256': plan['plan_sha256'], 'candidate': candidate,
                      'steps': int(engine.global_steps), 'synthetic_replay_state': {'rows': 16, 'seed': 42},
                      'rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                      'numpy_rng_state': np.random.get_state(), 'python_rng_state': random.getstate()}
            before = full_training_state(engine, scheduler)
            ckpt = output / 'checkpoint'
            engine.save_checkpoint(str(ckpt), tag='own_resume', client_state=client)
            checkpoint_files = {str(path.relative_to(ckpt)): {'sha256': digest(path), 'bytes': path.stat().st_size}
                                for path in sorted(ckpt.rglob('*')) if path.is_file()}
            require(checkpoint_files, 'own optimizer checkpoint wrote no files')
            # An intervening real update makes a no-op loader fail restoration.
            production_update(learner, ids, prompt, 'replay_maxrl')
            perturbed_sketch = optimizer_sketch(engine)
            perturbed_scheduler = scheduler.state_dict()
            require(engine.global_steps == client['steps'] + 1, 'checkpoint perturbation did not update')
            loaded, state = engine.load_checkpoint(str(ckpt), tag='own_resume', load_optimizer_states=True, load_lr_scheduler_states=True)
            require(loaded and state['candidate'] == candidate and state['steps'] == client['steps']
                    and state['benchmark_plan_sha256'] == plan['plan_sha256']
                    and state['synthetic_replay_state'] == client['synthetic_replay_state'],
                    'own optimizer checkpoint restore failed')
            restored = full_training_state(engine, scheduler)
            require(identity(before) == identity(restored), 'model/master/optimizer/scheduler state differs after restore')
            require(torch.equal(state['rng_state'], client['rng_state']), 'checkpoint CPU RNG mismatch')
            require(len(state['cuda_rng_state']) == len(client['cuda_rng_state'])
                    and all(torch.equal(x, y) for x, y in zip(state['cuda_rng_state'], client['cuda_rng_state'])),
                    'checkpoint CUDA RNG mismatch')
            require(np.array_equal(state['numpy_rng_state'][1], client['numpy_rng_state'][1])
                    and state['numpy_rng_state'][0] == client['numpy_rng_state'][0]
                    and state['numpy_rng_state'][2:] == client['numpy_rng_state'][2:]
                    and state['python_rng_state'] == client['python_rng_state'], 'checkpoint NumPy/Python RNG mismatch')
            torch.set_rng_state(state['rng_state']); torch.cuda.set_rng_state_all(state['cuda_rng_state'])
            np.random.set_state(state['numpy_rng_state']); random.setstate(state['python_rng_state'])
            before_steps = engine.global_steps
            production_update(learner, ids, prompt, 'replay_maxrl')
            require(engine.global_steps == before_steps + 1, 'resumed optimizer did not update once')
            continuation = compare_sketch(perturbed_sketch, optimizer_sketch(engine), tolerance=1e-5)
            require(continuation['passed'] and identity(scheduler.state_dict()) == identity(perturbed_scheduler),
                    'resumed fixed update differs from uninterrupted continuation')
            result['own_checkpoint_resume'] = True
            result['checkpoint_path'] = str(ckpt)
            result['checkpoint_bytes'] = sum(v['bytes'] for v in checkpoint_files.values())
            result['checkpoint_files'] = checkpoint_files
            result['checkpoint_restore_evidence'] = {'full_training_state_sha256': identity(before),
                'intervening_update_steps': 1, 'restored_global_steps': before_steps,
                'uninterrupted_vs_resumed_moments': continuation,
                'scope': 'all model/master/optimizer bytes and scheduler; synthetic replay descriptor, actual bank/cursor/request state belongs to end-to-end gate'}
            # Only this isolated successfully verified temporary checkpoint is retired.
            import shutil
            require(ckpt.resolve().parent == output.resolve() and ckpt.name == 'checkpoint', 'unexpected checkpoint cleanup path')
            shutil.rmtree(ckpt)
            result['checkpoint_retired_after_verified_restore'] = True
            result['full_shape_smoke'] = set(result['domains']) == set(DOMAINS)
            result['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            result['status'] = 'passed'
        except Exception as exc:
            result['status'] = 'failed'; result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            result['host_memory'] = sampler.report()
            write(output / 'stress_result.json', result)
    return result


def run_process(command, env, log, timeout):
    with Path(log).open('x') as stream:
        p = subprocess.Popen(command, env=env, cwd=env['OAT_ZERO_REPO_ROOT'], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except BaseException:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()
            raise


class GPUSampler:
    def __init__(self, torch):
        self.uuid = allocated_gpu_uuid(torch); self.samples = []; self.done = threading.Event()
    def __enter__(self):
        def loop():
            while not self.done.is_set():
                try:
                    p = subprocess.run(['nvidia-smi', '-i', self.uuid, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
                    self.samples.extend(int(x.strip()) * 1024**2 for x in p.stdout.splitlines() if x.strip().isdigit())
                except (OSError, subprocess.SubprocessError): pass
                self.done.wait(1)
        self.thread = threading.Thread(target=loop, daemon=True); self.thread.start(); return self
    def __exit__(self, *args):
        self.done.set(); self.thread.join(timeout=8)
    def report(self):
        return {'uuid': self.uuid, 'sample_count': len(self.samples), 'peak_bytes': max(self.samples, default=0), 'scope': 'allocated GPU only;1Hz NVML samples'}


def read_jsonl(target, name):
    return [json.loads(line) for p in Path(target).glob('debug*/' + name) for line in p.read_text().splitlines() if line.strip()]


def e2e_cell(plan, cell):
    key = cell_key(cell); root = Path(plan['output_root']); output = root / 'end_to_end' / key
    result_path = output / 'result.json'
    if result_path.exists():
        result = read(result_path)
        require(result.get('status') == 'passed' and result.get('plan_sha256') == plan['plan_sha256'], 'old failed or mismatched cell cannot rerun')
        for phase in result['phases']:
            require(digest(phase['log_path']) == phase['log_sha256'], 'completed cell log evidence changed')
        return result
    require(not output.exists(), 'partial cell requires review before retry')
    output.mkdir(parents=True)
    result = {'schema': 'e124_qwen7b_e2e_cell_v1', 'status': 'running', 'cell': key, 'plan_sha256': plan['plan_sha256'], 'phases': [], 'outcomes_used_for_selection': False}
    write(output / 'intent.json', result)
    torch = torch_imports(plan); props = require_gpu(torch)
    checkpoint = None; peak_storage = 0
    with MemorySampler() as memory, GPUSampler(torch) as gpu:
        try:
            for phase in ('fresh', 'resume'):
                target = output / phase
                resolved = e2e_environment(plan, cell, target, resume=checkpoint)
                env = clean_environment(plan, resolved); log = output / (phase + '.log')
                started = time.monotonic()
                code = run_process(['bash', str(Path(plan['runtime']['ops_root']) / 'train.sh')], env, log, 12 * 3600)
                require(code == 0, f'{key}/{phase} process failed:{code}')
                metrics = read_jsonl(target, 'train_metrics.jsonl')
                require(metrics, f'{key}/{phase} missing actual optimizer metrics')
                last = max(float(r.get('misc/global_step', r.get('trainer/global_step', 0))) for r in metrics)
                expected = 3 if phase == 'resume' else 2
                require(last == expected, 'wrong smoke optimizer boundary')
                draws = read_jsonl(target, 'eval_mode_coverage_draws.jsonl')
                rows = int(cell.get('eval_rows', 128))
                endpoints = validate_evaluation(draws, resolved, rows)
                logs = log.read_text(errors='replace')
                require('max_steps=3072' in logs, 'smoke changed LR scheduler horizon')
                require('sync params to actors done step=' in logs, 'actor synchronization not observed')
                require('Finished eval benchmark' in logs and 'Finished sampled mode-coverage eval' in logs, 'complete greedy and sampled evaluation missing')
                require(not re.search(r'Traceback|OutOfMemoryError|CUDA out of memory|buffer.*(?:mismatch|missing)', logs, re.I), 'fatal runtime evidence')
                if phase == 'resume':
                    require('resume actor weight sync done checkpoint_step=2' in logs, 'resume actor synchronization missing')
                    require('Restored prompt traversal state: checkpoint_step=2 completed_batches=2' in logs, 'prompt cursor was not restored')
                require((target / 'TRAINING_COMPLETE.json').is_file(), 'smoke completion receipt missing')
                peak_storage = max(peak_storage, directory_bytes(output))
                row = {'phase': phase, 'last_step': last, 'evaluation_endpoints': endpoints, 'seconds': time.monotonic() - started,
                       'log_path': str(log), 'log_sha256': digest(log), 'environment_sha256': identity(resolved), 'resolved_environment': resolved,
                       'completion': read(target / 'TRAINING_COMPLETE.json')}
                if phase == 'fresh':
                    choices = list(target.glob('debug*/checkpoints/step_00002'))
                    require(len(choices) == 1, 'exact ownCP2 missing')
                    checkpoint = choices[0]; row['checkpoint'] = checkpoint_metadata(checkpoint, 2)
                result['phases'].append(row); write(output / 'progress.json', result)
            require(checkpoint is not None, 'missing resumed checkpoint')
            # Re-read full files after the resumed child exits; preserve corrupt evidence.
            require(checkpoint_metadata(checkpoint, 2) == result['phases'][0]['checkpoint'], 'checkpoint changed during resume')
            for p in output.glob('*/debug*/checkpoints/step_*'):
                safe_retire_checkpoint(p, output)
            result.update(status='passed', own_checkpoint_resume=True, checkpoints_retired_after_verification=True,
                          checkpoint_bytes=result['phases'][0]['checkpoint']['bytes'], peak_artifact_bytes=peak_storage)
        except BaseException as exc:
            result.update(status='failed', error=f'{type(exc).__name__}:{exc}')
            raise
        finally:
            result['host_memory'] = memory.report(); result['gpu'] = gpu.report(); result['device_total_bytes'] = props.total_memory
            write(result_path, result)
    return result


def qualify(plan, stress, cells):
    require(stress.get('status') == 'passed' and stress.get('plan_sha256') == plan['plan_sha256'], 'stress failed or wrong plan')
    require(stress.get('own_checkpoint_resume') is True and set(stress.get('full_shape_arms', [])) == set(ARMS), 'both full-shape arms and full optimizer restore required')
    require(set(cells) == set(plan['coverage']), 'real path coverage incomplete')
    require(all(c.get('status') == 'passed' and c.get('plan_sha256') == plan['plan_sha256'] and c.get('own_checkpoint_resume') is True for c in cells.values()), 'real path failed or wrong plan')
    memories = [stress['host_memory']] + [c['host_memory'] for c in cells.values()]
    require(all(m.get('available') and m['sample_count'] > 0 for m in memories), 'whole-job memory measurement missing')
    require(all(all(m['events_delta'].get(k, 0) == 0 for k in ('high', 'oom', 'oom_kill')) for m in memories), 'memory pressure during qualification')
    host = max(m['peak_nonreclaimable_dirty_bytes'] for m in memories)
    host_required = max(host * 1.2, host + 8 * GIB)
    require(host_required <= 256 * GIB, 'fixed256GiB request lacks host headroom')
    require(all(c['gpu']['sample_count'] > 0 and c['gpu']['peak_bytes'] > 0 for c in cells.values()), 'real GPU memory evidence missing')
    total = min([stress['device']['total_memory_bytes']] + [c['device_total_bytes'] for c in cells.values()])
    peak = max([stress['gpu_peak_reserved_bytes']] + [c['gpu']['peak_bytes'] for c in cells.values()])
    require(peak > 0 and peak + 2 * GIB <= total, 'fixed48GiB GPU lacks2GiB headroom')
    checkpoint = max([stress['checkpoint_bytes']] + [c['checkpoint_bytes'] for c in cells.values()])
    require(checkpoint > 0, 'checkpoint storage was not measured')
    terminal_bytes = int(plan['model']['weight_bytes'])
    require(terminal_bytes > 0, 'terminal model size missing')
    storage_reserve = 2 * checkpoint + terminal_bytes + 4 * GIB
    require(storage_reserve <= 220 * GIB, 'measured two-checkpoint plus terminal/log storage exceeds220GiB reserve')
    result = {'schema': 'e124_qwen7b_qualified_profile_v1', 'status': 'passed', 'plan_sha256': plan['plan_sha256'],
              'profile_id': PROFILE_ID, 'profile_environment': PROFILE, 'resources': plan['resources'],
              'runtime': plan['runtime'], 'model': plan['model'], 'dataset_pins': plan['datasets'],
              'coverage': plan['coverage'], 'manifest_path': plan['manifest_path'], 'manifest_sha256': plan['manifest_sha256'],
              'hardware': {'gpu_class': 'A6000', 'gpus': 1, 'total_memory_bytes': total}, 'outcomes_used_for_selection': False,
              'measurements': {'host_nonreclaimable_dirty_peak_bytes': host, 'host_with_headroom_bytes': math.ceil(host_required),
                               'gpu_peak_bytes': peak, 'gpu_total_bytes': total, 'checkpoint_bytes': checkpoint,
                               'rolling_two_checkpoint_bytes': 2 * checkpoint, 'terminal_model_bytes': terminal_bytes,
                               'log_headroom_bytes': 4 * GIB, 'required_storage_reserve_bytes': storage_reserve,
                               'fixed_storage_reserve_bytes': 220 * GIB, 'concurrency_tested': 1},
              'evidence': {'stress_sha256': identity(stress), 'cells_sha256': {k: identity(v) for k, v in cells.items()}}}
    output = Path(plan['output_root'])
    paths = [output / PROFILE_ID / 'stress_result.json'] + [output / 'end_to_end' / key / 'result.json' for key in plan['coverage']]
    result['evidence']['files'] = {str(p): digest(p) for p in paths}
    result['profile_sha256'] = identity(result)
    return result


def run_suite(plan_path):
    plan_path = Path(plan_path).resolve(); plan = read(plan_path); verify_pins(plan)
    root = Path(plan['output_root']); root.mkdir(parents=True, exist_ok=True)
    with (root / 'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'schema': 'e124_qwen7b_suite_v1', 'status': 'running', 'plan_sha256': plan['plan_sha256'], 'completed_cells': []}
        try:
            stress_path = root / PROFILE_ID / 'stress_result.json'
            if not stress_path.exists():
                directory = stress_path.parent
                require(not directory.exists(), 'partial stress requires review')
                directory.mkdir(parents=True)
                write(directory / 'intent.json', {'plan_sha256': plan['plan_sha256'], 'status': 'starting'})
                worst = next(c for c in plan['cells'] if cell_key(c) == plan['worst_cell'])
                env = clean_environment(plan, worst['environment'])
                python = env.get('OAT_ZERO_PYTHON', str(Path(plan['workspace_root']) / 'var/seed_paper_eval/paper310/bin/python'))
                code = run_process([python, str(SOURCE), 'worker', '--plan', str(plan_path)], env, directory / 'worker.log', 12 * 3600)
                require(code == 0 and stress_path.exists(), f'stress process failed:{code}')
            stress = read(stress_path)
            require(stress.get('status') == 'passed' and stress.get('plan_sha256') == plan['plan_sha256'], 'failed/mismatched stress cannot rerun')
            cells = {}; lookup = {cell_key(c): c for c in plan['cells']}
            for key in plan['coverage']:
                verify_pins(plan, full=False)
                cells[key] = e2e_cell(plan, lookup[key])
                status['completed_cells'].append(key); write(root / 'suite_status.json', status)
            verify_pins(plan)
            profile = qualify(plan, stress, cells); write(root / 'qualified_profile.json', profile)
            status.update(status='passed', qualified_profile=str(root / 'qualified_profile.json'), qualified_profile_sha256=digest(root / 'qualified_profile.json'))
            return 0
        except BaseException as exc:
            status.update(status='failed', error=f'{type(exc).__name__}:{exc}', review_required=True)
            return 1
        finally:
            write(root / 'suite_status.json', status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    for name in ('run-suite', 'worker'):
        p = sub.add_parser(name); p.add_argument('--plan', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        plan = prepare(args.manifest, args.output); print(json.dumps({'plan_sha256': plan['plan_sha256']})); return 0
    if args.command == 'run-suite': return run_suite(args.plan)
    plan = read(args.plan); verify_pins(plan); worker(plan); return 0


if __name__ == '__main__':
    raise SystemExit(main())
