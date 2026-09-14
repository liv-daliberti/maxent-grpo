#!/usr/bin/env python3
"""Isolated E123 runtime measurements; this program never submits Slurm jobs.

``prepare --manifest PLAN --output DIR`` accepts the E123 preview's cells.
``run --plan DIR/plan.json --candidate gpu_mb4_omp4`` runs on an already
allocated GPU. It exercises the production GRPO/replay update, then saves and
restores its OWN fresh optimizer checkpoint. Synthetic full-capacity sequences
are deliberately labelled stress inputs, never validator-positive discoveries.

Learner-only measurements cannot admit a production resource profile. The
``end-to-end`` command runs the actual actor/learner shell entry under the same
allocation and captures full-job cgroup pressure. ``select`` requires both
stages and a matched fixed-update baseline before issuing a profile.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
GIB = 1024 ** 3
DOMAINS = ('graph_coloring', 'countdown', 'python_factors', 'mathir', 'pantry_plan')
ARMS = ('drgrpo', 'replay_drgrpo', 'maxrl', 'replay_maxrl')
VARIANTS = {
    'baseline': {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '1', 'OMP_NUM_THREADS': '1', 'OAT_ZERO_ADAM_OFFLOAD': '1'},
    'omp4': {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '1', 'OMP_NUM_THREADS': '4', 'OAT_ZERO_ADAM_OFFLOAD': '1'},
    'mb4': {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '4', 'OMP_NUM_THREADS': '1', 'OAT_ZERO_ADAM_OFFLOAD': '1'},
    'mb8': {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '8', 'OMP_NUM_THREADS': '1', 'OAT_ZERO_ADAM_OFFLOAD': '1'},
    'gpu_adam': {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '1', 'OMP_NUM_THREADS': '1', 'OAT_ZERO_ADAM_OFFLOAD': '0'},
    'gpu_mb4_omp4': {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '4', 'OMP_NUM_THREADS': '4', 'OAT_ZERO_ADAM_OFFLOAD': '0'},
    'gpu_mb8_omp4': {'OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE': '8', 'OMP_NUM_THREADS': '4', 'OAT_ZERO_ADAM_OFFLOAD': '0'},
}
RUNTIME_KEYS = frozenset(next(iter(VARIANTS.values())))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def verify_pins(plan):
    require(identity({k: v for k, v in plan.items() if k != 'plan_sha256'}) == plan['plan_sha256'], 'benchmark plan digest mismatch')
    for path, expected in plan['runtime']['files_sha256'].items():
        require(digest(path) == expected, f'measured runtime changed: {path}')
    require(digest(plan['dataset']['identity_path']) == plan['dataset']['identity_sha256'], 'frozen dataset identity changed')


def extract_cells(manifest):
    cells = manifest.get('cells', manifest.get('planned_cells', manifest.get('runs', [])))
    require(isinstance(cells, list) and cells, 'manifest requires cells with resolved environment')
    selected = {}
    for cell in cells:
        if cell.get('domain') not in DOMAINS or cell.get('arm') not in ARMS:
            continue
        env = cell.get('environment', cell.get('env'))
        require(isinstance(env, dict), 'each benchmark cell needs its resolved environment')
        if cell['domain'] not in selected or (int(cell.get('seed', 10**6)), cell['arm']) < (int(selected[cell['domain']].get('seed', 10**6)), selected[cell['domain']]['arm']):
            selected[cell['domain']] = cell
    require(set(selected) == set(DOMAINS), 'all five frozen Level 3 domains are required')
    return selected


def prepare(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    require(not (output / 'plan.json').exists(), 'refuse to overwrite a benchmark plan')
    manifest = read(manifest_path)
    cells = extract_cells(manifest)
    envs = {d: dict(cells[d]['environment']) for d in DOMAINS}
    for domain, env in envs.items():
        for key, expected in {'OAT_ZERO_NUM_SAMPLES': '16', 'OAT_ZERO_TRAIN_BATCH_SIZE': '16', 'OAT_ZERO_ROLLOUT_BATCH_SIZE': '1', 'OAT_ZERO_NUM_PPO_EPOCHS': '1', 'OAT_ZERO_PROMPT_MAX_LENGTH': '1024', 'OAT_ZERO_GENERATE_MAX_LENGTH': '192', 'OAT_ZERO_MAX_MODEL_LEN': '2048'}.items():
            require(env.get(key) == expected, f'{domain}: E123 scientific/shape contract differs at {key}')
        require(env.get('OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE') == 'verified_likelihood_per_rollout', 'verified replay objective required')
        require(int(env.get('OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY', 16)) == 16, 'expected replay capacity 16')
        require('3b' in env.get('OAT_ZERO_MODEL', '').lower(), 'Qwen 3B model required')
    source_roots = {e['OAT_ZERO_SOURCE_ROOT'] for e in envs.values()}
    ops_roots = {e['OAT_ZERO_OPS_SNAPSHOT_ROOT'] for e in envs.values()}
    require(len(source_roots) == len(ops_roots) == 1, 'one immutable runtime is required')
    source_root, ops_root = Path(next(iter(source_roots))), Path(next(iter(ops_roots)))
    pins = {str(SOURCE): digest(SOURCE)}
    launcher = ROOT / 'ops/exp_scaling/launch_e123_level3_qwen3b_factorial.py'
    pins[str(launcher)] = digest(launcher)
    for p in sorted((source_root / 'oat_drgrpo').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            pins[str(p.resolve())] = digest(p)
    for name in ('train.sh', 'repo_env.sh'):
        pins[str((ops_root / name).resolve())] = digest(ops_root / name)
    dataset_root = Path(envs[DOMAINS[0]]['OAT_ZERO_PROMPT_DATA']).parents[1]
    dataset_identity = dataset_root / 'identity.json'
    require(dataset_identity.is_file(), 'frozen Level 3 dataset identity missing')
    snapshot = manifest.get('snapshot', {})
    require(snapshot.get('root') == str(source_root.parent) and re.fullmatch(r'[0-9a-f]{64}', snapshot.get('sha256', '')), 'manifest must bind the exact campaign snapshot identity')
    plan = {'schema': 'e123_a100_benchmark_plan_v1', 'output_root': str(output), 'manifest_path': str(manifest_path), 'manifest_sha256': digest(manifest_path),
            'runtime': {'snapshot_root': str(source_root.parent), 'source_root': str(source_root), 'ops_root': str(ops_root), 'files_sha256': pins, 'identity_sha256': snapshot['sha256']},
            'dataset': {'identity_path': str(dataset_identity), 'identity_sha256': digest(dataset_identity)},
            'environments': envs, 'variants': VARIANTS, 'stress': {'fresh_rows': 16, 'replay_rows': 16, 'prompt_tokens': 1024, 'response_tokens': 192, 'semantics': 'synthetic tensor capacity test, never scientific outcomes'},
            'defaults': {'warmup': 8, 'steps': 32, 'additional_domain_stress_updates': 1}, 'scientific_horizon_steps': 3072, 'new_runs_start_pretrained': True,
            'profile_gate': {'host_headroom_fraction': 0.20, 'host_headroom_bytes': 8 * GIB, 'gpu_headroom_bytes': 4 * GIB, 'relative_moment_sketch_rms_tolerance': 0.03}}
    plan['plan_sha256'] = identity(plan)
    write(output / 'plan.json', plan)
    return plan


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


def torch_imports(plan):
    # The shell launcher supplies LD_LIBRARY_PATH. Direct benchmark commands
    # also need libpython visible to Courier's linked C++ extension.
    import ctypes
    for base in (Path(sys.base_prefix) / 'lib', ROOT / 'var/seed_paper_eval/paper310/lib'):
        library = base / f'libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0'
        if library.is_file():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            break
    os.environ.update({'TRANSFORMERS_NO_TF': '1', 'USE_TF': '0', 'USE_FLAX': '0', 'TOKENIZERS_PARALLELISM': 'false'})
    sys.path.insert(0, plan['runtime']['source_root'])
    import torch
    from oat_drgrpo import fused_adam_shim
    fused_adam_shim.install()
    return torch


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


def optimizer_sketch_indices(numel):
    """At most 256 exact, evenly spaced indices, including both endpoints.

    Python integer arithmetic prevents FP32 rounding above 2**24 and avoids
    intermediate int64 multiplication overflow for very large state sizes.
    No optimizer values or updates are changed by this diagnostic sampling.
    """
    require(isinstance(numel, int) and not isinstance(numel, bool) and 0 <= numel <= 2**63 - 1,
            'optimizer sketch size must be a nonnegative int64 count')
    count = min(256, numel)
    if count < 2:
        return [0] if count else []
    return [index * (numel - 1) // (count - 1) for index in range(count)]


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
                ix = torch.tensor(optimizer_sketch_indices(flat.numel()), dtype=torch.long, device=flat.device)
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


def worker(plan, candidate, warmup, steps, domains):
    verify_pins(plan)
    require(candidate in VARIANTS and warmup >= 8 and steps >= 32, 'benchmark requires eight warmup and 32 measured updates')
    require(domains and len(set(domains)) == len(domains) and all(d in DOMAINS for d in domains), 'unknown or duplicate domain')
    env = dict(plan['environments'][DOMAINS[0]])
    env.update(VARIANTS[candidate])
    os.environ.update({k: str(v) for k, v in env.items() if k in RUNTIME_KEYS})
    torch = torch_imports(plan)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'benchmark needs exactly one allocated visible GPU')
    props = torch.cuda.get_device_properties(0)
    require('A100' in props.name and props.total_memory >= 75 * GIB, 'E123 runtime gate requires A100 80 GB')
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
    a.local_rank, a.seed, a.rnd_seed, a.zero_stage, a.bf16 = 0, 70, False, 2, True
    a.train_batch_size, a.num_samples, a.num_ppo_epochs = 16, 16, 1
    a.train_batch_size_per_device = int(env['OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE'])
    a.adam_offload = env['OAT_ZERO_ADAM_OFFLOAD'] == '1'
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
    result = {'schema': 'e123_a100_candidate_result_v1', 'status': 'running', 'scope': 'learner_only_stress', 'candidate': candidate,
              'plan_sha256': plan['plan_sha256'], 'profile_environment': VARIANTS[candidate], 'runtime': plan['runtime'], 'dataset': plan['dataset'],
              'device': {'name': props.name, 'total_memory_bytes': props.total_memory}, 'domains': {}, 'full_shape_smoke': False,
              'own_checkpoint_resume': False, 'end_to_end_smoke': False}
    output = Path(plan['output_root']) / candidate
    require(not (output / 'candidate_result.json').exists(), 'refuse to overwrite candidate evidence')
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
                    require(all(bool(torch.isfinite(v).all()) for v in info.values() if isinstance(v, torch.Tensor)), 'nonfinite production update')
                    if domain_index == i == 0:
                        result['fixed_update_sketch'] = optimizer_sketch(engine)
                    if i >= domain_warmup: times.append(elapsed)
                    checks.append({'optimizer_step': int(engine.global_steps), 'replay_rows': 16, 'fresh_rows': 16})
                    write(output / 'progress.json', {'domain': domain, 'completed_updates': i + 1, 'elapsed_seconds': elapsed, 'candidate': candidate})
                result['domains'][domain] = {'prompt_tokens': prompt, 'response_tokens': response, 'replay_rows': 16, 'fresh_rows': 16,
                    'warmup_updates': domain_warmup, 'measured_updates': domain_steps, 'timing_role': 'matched_throughput' if domain_index == 0 else 'full_shape_stress', 'median_update_seconds': statistics.median(times), 'update_seconds': times, 'effective_update_checks': checks}
                del ids
            # Exercise compute-only replay through the same full model once.
            ids = torch.randint(100, 1000, (16, 40), generator=torch.Generator().manual_seed(42)).cuda(); ids[:, :32] = ids[0, :32]
            info = production_update(learner, ids, 32, 'drgrpo')
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
            production_update(learner, ids, 32, 'replay_drgrpo')
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
            production_update(learner, ids, 32, 'replay_drgrpo')
            require(engine.global_steps == before_steps + 1, 'resumed optimizer did not update once')
            continuation = compare_sketch(perturbed_sketch, optimizer_sketch(engine), tolerance=1e-5)
            require(continuation['passed'] and identity(scheduler.state_dict()) == identity(perturbed_scheduler),
                    'resumed fixed update differs from uninterrupted continuation')
            result['own_checkpoint_resume'] = True
            result['checkpoint_path'] = str(ckpt)
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
            write(output / 'candidate_result.json', result)
    return result


def run_candidate(plan_path, candidate, warmup, steps, domains):
    plan = read(plan_path); verify_pins(plan)
    env = {k: v for k, v in os.environ.items() if not k.startswith('OAT_ZERO_') and k not in ('SAVE_PATH', 'RUN_STAMP')}
    env.update(plan['environments'][DOMAINS[0]]); env.update(VARIANTS[candidate])
    python = plan['environments'][DOMAINS[0]].get('OAT_ZERO_PYTHON', str(ROOT / 'var/seed_paper_eval/paper310/bin/python'))
    command = [python, str(SOURCE), 'worker', '--plan', str(Path(plan_path).resolve()), '--candidate', candidate,
               '--warmup', str(warmup), '--steps', str(steps), '--domains', ','.join(domains)]
    directory = Path(plan['output_root']) / candidate; directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.log').open('x') as log:
        return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode



def cpu_update_contract(plan):
    """Compare complete production-loss gradients across four arms and batches."""
    torch = torch_imports(plan)
    import functools
    from types import SimpleNamespace
    from oat.utils.ops import masked_sum
    from oat_drgrpo.args import ZeroMathArgs
    from oat_drgrpo.learner.grpo import ZeroMathGrpoMixin
    from oat_drgrpo.learner.base import ZeroMathLearnerBaseMixin
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([.2, -.1, .3], dtype=torch.float64))
        def forward(self, input_ids, attention_mask):
            return {'logits': self.weight[None, None].expand(input_ids.shape[0], input_ids.shape[1], 3)}
    class Strategy:
        def __init__(self, micro): self.grad_acc_step = 16 // micro; self.calls = 0; self.updates = 0
        def backward(self, loss, model, optimizer): (loss / self.grad_acc_step).backward()
        def optimizer_step(self, optimizer, model, scheduler):
            self.calls += 1
            if self.calls % self.grad_acc_step == 0:
                self.gradient = model.weight.grad.detach().clone(); optimizer.step(); self.updates += 1
        def get_gradient_norm(self, model): return float(model.weight.grad.norm())
        def is_rank_0(self): return False
    class Learner(ZeroMathGrpoMixin, ZeroMathLearnerBaseMixin):
        def _resolve_scoring_vocab_upper_bound(self, model): return 3
        def _should_skip_baseline_grad_norm_logging(self): return True
    comparisons = []
    for arm in ARMS:
        reference = None
        for micro in (1, 4, 8):
            learner = Learner(); a = ZeroMathArgs()
            a.train_batch_size = a.num_samples = 16; a.train_batch_size_per_device = micro
            a.generate_max_length = 1; a.num_ppo_epochs = 1; a.critic_type = 'drgrpo'; a.beta = a.maxent_alpha = a.policy_entropy_coef = 0.
            a.online_canonical_replay = True; a.online_canonical_replay_objective = 'verified_likelihood_per_rollout'; a.online_canonical_replay_alpha = .1; a.temperature = 1.
            learner.args = a; learner.model = Policy(); learner.strategy = Strategy(micro)
            learner.optimizer = torch.optim.SGD(learner.model.parameters(), lr=.1); learner.scheduler = None
            learner.masked_aggregator = functools.partial(masked_sum, constant_normalizer=1)
            learner.tokenizer = SimpleNamespace(pad_token_id=2, eos_token_id=2)
            learner._canonical_action_token_ids = None; learner._canonical_action_space = None
            learner._invalid_scoring_token_ids_warned_contexts = set(); learner._invalid_logit_columns_warned_contexts = set()
            learner._baseline_grad_norm_logging_disabled_warned = False
            ids = torch.tensor([[2, i % 3] for i in range(16)])
            info = production_update(learner, ids, 1, arm)
            require(learner.strategy.updates == 1, 'microbatch changed logical optimizer count')
            gradient = learner.strategy.gradient
            if reference is None: reference = gradient
            difference = float((gradient - reference).abs().max())
            require(difference < 1e-7, 'production replay/fresh gradient changed with microbatch')
            applied = float(info['canonical_replay_applied_score_gradient_l2'])
            require((applied > 0) == arm.startswith('replay_'), 'factorial replay derivative invariant failed')
            comparisons.append({'arm': arm, 'microbatch': micro, 'gradient_max_abs_difference': difference, 'optimizer_updates': 1, 'applied_replay_gradient_l2': applied})
    return {'schema': 'e123_a100_production_loss_contract_v1', 'status': 'passed', 'comparisons': comparisons,
            'scope': 'actual production GRPO/replay loss on complete tiny-model gradients; full-model moment comparison is separate'}


def e2e_environment(plan, candidate, domain, output, *, resume=None):
    env = dict(plan['environments'][domain]); env.update(VARIANTS[candidate])
    updates = 3 if resume else 2
    env.update({'SAVE_PATH': str(output), 'RUN_STAMP': f'e123bench_{candidate}_{domain}',
        'OAT_ZERO_REPO_ROOT': str(ROOT), 'ROOT_DIR': str(ROOT), 'OAT_ZERO_FIXED_EXP_SUFFIX': 'e123benchmark', 'OAT_ZERO_MAX_QUERIES': str((updates - 1) * 16),
        'OAT_ZERO_EVAL_STEPS': '96', 'OAT_ZERO_SAVE_CKPT': '1', 'OAT_ZERO_VARIANT': 'maxrl_verified_replay', 'OAT_ZERO_SAVE_STEPS': '2', 'OAT_ZERO_SAVE_FROM': '2',
        'OAT_ZERO_RESUME_STEPS': '2', 'OAT_ZERO_RESUME_FROM': '2', 'OAT_ZERO_MAX_RESUME_NUM': '1',
        'OAT_ZERO_EXPORT_STEPS': '-1', 'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS': '0',
        'OAT_ZERO_AUTO_RESUME': '0', 'OAT_ZERO_WATCHDOG_STALE_SECONDS': '0', 'OAT_ZERO_WATCHDOG_REQUEUE': '0',
        'OAT_ZERO_USE_WB': '0', 'OAT_ZERO_MAXRL_TASK_OBJECTIVE': '1', 'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY': '0',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'VLLM_USE_V1': '0'})
    env.pop('OAT_ZERO_RESUME_DIR', None); env.pop('OAT_ZERO_RESUME_TAG', None)
    if resume:
        env['OAT_ZERO_RESUME_DIR'] = str(resume.parent); env['OAT_ZERO_RESUME_TAG'] = resume.name
    return env



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


def end_to_end(plan, candidate, timeout=10800):
    """Five real actor/learner domains, original full evaluation, save/resume."""
    import shutil
    verify_pins(plan)
    output = Path(plan['output_root']) / candidate / 'end_to_end'
    require(not (output / 'e2e_result.json').exists(), 'refuse to overwrite end-to-end evidence')
    result = {'schema': 'e123_a100_e2e_result_v1', 'status': 'running', 'candidate': candidate, 'plan_sha256': plan['plan_sha256'],
        'runtime': plan['runtime'], 'dataset': plan['dataset'], 'domains': {}, 'short_run_only': True, 'scientific_horizon_steps': 3072,
        'outcomes_used_for_selection': False, 'gpu_peak_bytes': 0}
    output.mkdir(parents=True, exist_ok=True)
    torch = torch_imports(plan)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'end-to-end gate needs one allocated GPU')
    gpu_uuid = allocated_gpu_uuid(torch)
    require(gpu_uuid.startswith('GPU-'), 'allocated GPU UUID unavailable; refuse to sample peer GPUs')
    result['gpu_uuid'] = gpu_uuid
    gpu_peaks = []; done = threading.Event()
    def gpu_loop():
        while not done.is_set():
            try:
                p = subprocess.run(['nvidia-smi', '-i', gpu_uuid, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                values = [int(x.strip()) * 1024 ** 2 for x in p.stdout.splitlines() if x.strip().isdigit()]
                if values: gpu_peaks.append(max(values))
            except (OSError, subprocess.TimeoutExpired): pass
            done.wait(1)
    thread = threading.Thread(target=gpu_loop, daemon=True); thread.start()
    with MemorySampler() as memory:
        try:
            for domain in DOMAINS:
                phases = []; checkpoint = None
                for phase in ('fresh', 'resume'):
                    target = output / domain / phase
                    require(not target.exists(), 'benchmark end-to-end output must be fresh')
                    env = {k: v for k, v in os.environ.items() if not k.startswith('OAT_ZERO_') and k not in ('SAVE_PATH', 'RUN_STAMP')}
                    resolved = e2e_environment(plan, candidate, domain, target, resume=checkpoint)
                    env.update(resolved)
                    env['PYTHONPATH'] = plan['runtime']['source_root'] + os.pathsep + env.get('PYTHONPATH', '')
                    log = output / domain / (phase + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
                    started = time.monotonic()
                    with log.open('x') as stream:
                        process = subprocess.Popen(['bash', str(Path(plan['runtime']['ops_root']) / 'train.sh')], env=env,
                            cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        try: code = process.wait(timeout=timeout)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM)
                            try: process.wait(timeout=30)
                            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
                            raise RuntimeError(f'end-to-end {domain}/{phase} timed out')
                    require(code == 0, f'end-to-end {domain}/{phase} process failed: {code}')
                    metrics = []
                    for path in target.glob('debug*/train_metrics.jsonl'):
                        metrics.extend(json.loads(line) for line in path.read_text().splitlines())
                    require(metrics, f'no real training metrics: {domain}/{phase}')
                    last = max(float(r.get('misc/global_step', r.get('trainer/global_step', 0))) for r in metrics)
                    require(last == (3 if phase == 'resume' else 2), 'end-to-end optimizer did not reach requested boundary')
                    draws = []
                    for path in target.glob('debug*/eval_mode_coverage_draws.jsonl'):
                        draws.extend(json.loads(line) for line in path.read_text().splitlines())
                    sampled = [r for r in draws if r.get('evaluation_kind') == 'fixed_seed_sampled_k_neutral']
                    endpoints = {}
                    for row in sampled:
                        require(row.get('sample_count') == 8 and float(row.get('temperature', -1)) == 1.0 and float(row.get('top_p', -1)) == 1.0, 'evaluation changed k/temperature/top-p')
                        require(row.get('seed') == int(resolved['OAT_ZERO_EVAL_MODE_COVERAGE_SEED']) + int(row['draw_index']), 'evaluation draw seed changed')
                        prompts = row.get('prompts', [])
                        require(len(prompts) == 128 and {q.get('prompt_index') for q in prompts} == set(range(128)), 'evaluation needs 128 unique prompts per draw')
                        require(all(len(q.get('responses', [])) == 8 for q in prompts), 'evaluation response count incomplete')
                        endpoints.setdefault(row['step'], set()).add(row['draw_index'])
                    require(endpoints and all(indices == set(range(4)) for indices in endpoints.values()), 'four complete fixed evaluation draws required')
                    logs = log.read_text(errors='replace')
                    require('max_steps=3072' in logs, 'benchmark shortened the registered learning-rate horizon')
                    if phase == 'resume': require('resume actor weight sync done checkpoint_step=2' in logs, 'resumed actor not synchronized before generation')
                    require((target / 'TRAINING_COMPLETE.json').is_file(), 'end-to-end completion receipt missing')
                    phases.append({'phase': phase, 'last_optimizer_step': last, 'evaluation_draw_records': len(draws),
                        'elapsed_seconds': time.monotonic()-started, 'log_path': str(log), 'environment_sha256': identity(resolved),
                        'completion_receipt': str(target / 'TRAINING_COMPLETE.json')})
                    if phase == 'fresh':
                        choices = list(target.glob('debug*/checkpoints/step_00002'))
                        require(len(choices) == 1, 'exact own step-2 checkpoint missing')
                        checkpoint = choices[0]
                        phases[-1]['checkpoint_files'] = {str(p.relative_to(checkpoint)): p.stat().st_size for p in checkpoint.rglob('*') if p.is_file()}
                require(checkpoint is not None and checkpoint.is_relative_to(output), 'unsafe benchmark checkpoint cleanup path')
                shutil.rmtree(checkpoint)
                result['domains'][domain] = {'status': 'passed', 'phases': phases, 'own_checkpoint_removed_after_verified_resume': True}
                write(output / 'progress.json', result)
            result['status'] = 'passed'
        except Exception as exc:
            result['status'] = 'failed'; result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            done.set(); thread.join(timeout=10)
            result['host_memory'] = memory.report()
            result['gpu_peak_bytes'] = max(gpu_peaks, default=0)
            result['gpu_sampling_scope'] = '1 Hz device-memory samples of allocated GPU UUID only'
            write(output / 'e2e_result.json', result)
    return result


def selected_profile(plan, candidate_result, e2e_result, baseline_result, contract_path, *, fallback_attempt=None):
    verify_pins(plan)
    candidate_path, e2e_path, baseline_path = map(Path, (candidate_result, e2e_result, baseline_result))
    c, e, b = (read(p) for p in (candidate_path, e2e_path, baseline_path))
    require(all(x.get('status') == 'passed' and x.get('plan_sha256') == plan['plan_sha256'] for x in (c, e, b)), 'benchmark stages failed or belong to another plan')
    require(c['candidate'] in ('gpu_mb4_omp4', 'gpu_mb8_omp4', 'gpu_adam') and e['candidate'] == c['candidate'], 'only matching registered GPU candidates can be selected')
    if c['candidate'] == 'gpu_adam':
        require(fallback_attempt is not None, 'GPU Adam fallback requires both combined qualification failures')
        attempts = read(fallback_attempt)
        require(attempts.get('status') == 'qualifying_fallback' and attempts.get('plan_sha256') == plan['plan_sha256']
                and set(attempts.get('candidates', {})) == set(VARIANTS), 'fallback must preserve all seven measurements')
        for name in ('gpu_mb4_omp4', 'gpu_mb8_omp4'):
            entry = attempts['candidates'][name]
            require(digest(entry['path']) == entry['sha256'], 'combined candidate receipt changed')
            prior = read(entry['path'])
            require(prior.get('candidate') == name and prior.get('plan_sha256') == plan['plan_sha256'], 'combined receipt identity differs')
            require(prior.get('status') == 'failed' or
                    (prior.get('status') == 'passed' and bool(entry.get('qualification_error'))),
                    'combined candidate was not rejected before fallback')
            if entry.get('e2e_result_path'):
                require(digest(entry['e2e_result_path']) == entry.get('e2e_result_sha256'), 'combined end-to-end receipt changed')
    require(c.get('full_shape_smoke') is True and c.get('own_checkpoint_resume') is True and set(e['domains']) == set(DOMAINS), 'full five-domain shapes and real resume/evaluation required')
    require(read(contract_path).get('status') == 'passed', 'production loss contract did not pass')
    comparison = compare_sketch(b['fixed_update_sketch'], c['fixed_update_sketch'], plan['profile_gate']['relative_moment_sketch_rms_tolerance'])
    require(comparison['passed'], 'fixed-update moment comparison failed')
    memories = [c['host_memory'], e['host_memory']]
    require(all(m.get('available') for m in memories), 'whole-job host memory evidence missing')
    require(all(m['events_delta'].get('oom', 0) == m['events_delta'].get('oom_kill', 0) == 0 for m in memories), 'OOM observed during qualification')
    host = max(m['peak_nonreclaimable_dirty_bytes'] for m in memories)
    host_with_clean_cache = max(m['peak_current_bytes'] for m in memories)
    gpu = max(c['gpu_peak_reserved_bytes'], e['gpu_peak_bytes'])
    require(gpu > 0 and gpu <= 76 * GIB, 'GPU memory lacks measured headroom')
    requested = max(56, math.ceil(max(host * 1.2, host + 8 * GIB) / GIB))
    require(requested <= 128, 'host footprint exceeds supported per-job allocation')
    env = dict(VARIANTS[c['candidate']]); env['OAT_ZERO_ACTIVATION_OFFLOADING'] = plan['environments'][DOMAINS[0]].get('OAT_ZERO_ACTIVATION_OFFLOADING', '1')
    profile = {'schema': 'e123_a100_selected_profile_v1', 'status': 'passed', 'selection_status': 'selected', 'candidate': c['candidate'],
        'profile_environment': env, 'resources': {'cpus': 8, 'memory_gib': requested, 'node': 'node302', 'gpus': 1},
        'runtime': plan['runtime'], 'dataset': plan['dataset'], 'measurements': {'host_peak_bytes': host, 'gpu_peak_bytes': gpu,
            'host_semantics': 'peak anon+shmem+kernel+file_dirty+file_writeback; excludes clean reclaimable cache; dirty/writeback may conservatively overlap',
            'host_peak_including_clean_cache_bytes': host_with_clean_cache, 'reduced_memory_allocation_tested': False, 'eight_job_concurrency_tested': False,
            'eight_jobs_memory_feasible': requested * 8 <= 503, 'node_memory_gib': 503},
        'evidence': {'full_shape_smoke': True, 'own_checkpoint_resume': True, 'fixed_update_equivalence': True, 'end_to_end_smoke': True,
            'candidate_result_path': str(candidate_path.resolve()), 'candidate_result_sha256': digest(candidate_path),
            'e2e_result_path': str(e2e_path.resolve()), 'e2e_result_sha256': digest(e2e_path), 'baseline_result_path': str(baseline_path.resolve()),
            'baseline_result_sha256': digest(baseline_path), 'production_contract_path': str(Path(contract_path).resolve()),
            'production_contract_sha256': digest(contract_path), 'fixed_update_comparison': comparison},
        'outcomes_used_for_selection': False, 'eight_jobs_is_target_not_requirement': True}
    # Bind model/science to the same launcher projection used at admission.
    sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
    import launch_e123_level3_qwen3b_factorial as launch
    if hasattr(launch, 'science_environment'):
        profile['science_environment'] = launch.science_environment(plan['environments'][DOMAINS[0]])
    profile['model'] = {'choice': '3b', 'revision': 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 'path': plan['environments'][DOMAINS[0]]['OAT_ZERO_PRETRAIN']}
    if c['candidate'] == 'gpu_adam':
        profile['fallback_qualification'] = {'policy': 'combined_profiles_first_v1', 'fallback_used': True,
            'suite_status_path': str(Path(fallback_attempt).resolve()), 'suite_status_sha256': digest(fallback_attempt)}
    profile['profile_sha256'] = identity(profile)
    return profile


def suite(plan_path, warmup=8, steps=32):
    """Restartable at completed stage boundaries; failed stages never pass."""
    plan = read(plan_path); verify_pins(plan)
    output = Path(plan['output_root'])
    contract_path = output / 'production_loss_contract.json'
    if not contract_path.exists(): write(contract_path, cpu_update_contract(plan))
    require(read(contract_path).get('status') == 'passed', 'production loss contract failed')
    summary = {'schema': 'e123_a100_benchmark_suite_v1', 'status': 'running', 'plan_sha256': plan['plan_sha256'], 'candidates': {}, 'warmup': warmup, 'steps': steps}
    for candidate in VARIANTS:
        result_path = output / candidate / 'candidate_result.json'
        if not result_path.exists():
            try: code = run_candidate(plan_path, candidate, warmup, steps, list(DOMAINS))
            except Exception as exc: code = -1; summary['candidates'][candidate] = {'status': 'failed', 'error': str(exc)}
            if not result_path.exists(): write(result_path, {'schema': 'e123_a100_candidate_result_v1', 'status': 'failed', 'candidate': candidate, 'exit_code': code, 'plan_sha256': plan['plan_sha256']})
        result = read(result_path)
        require(result.get('plan_sha256') == plan['plan_sha256'], 'candidate result from another plan')
        summary['candidates'][candidate] = {'status': result['status'], 'path': str(result_path), 'sha256': digest(result_path)}
        write(output / 'suite_status.json', summary)
    baseline = output / 'baseline/candidate_result.json'
    require(read(baseline).get('status') == 'passed', 'baseline failed; cannot qualify a changed optimizer')
    candidates = []
    for name in ('gpu_mb4_omp4', 'gpu_mb8_omp4'):
        path = output / name / 'candidate_result.json'; result = read(path)
        if result.get('status') == 'passed': candidates.append((result['domains'][DOMAINS[0]]['median_update_seconds'], name))
    fallback = read(output / 'gpu_adam/candidate_result.json')
    if fallback.get('status') == 'passed':
        candidates.append((math.inf, 'gpu_adam'))  # Both combined candidates retain first preference.
    for _, name in sorted(candidates):
        path = output / name / 'candidate_result.json'; e2e = output / name / 'end_to_end/e2e_result.json'
        try:
            fallback_attempt = None
            if name == 'gpu_adam':
                fallback_attempt = output / 'fallback_attempt.json'
                attempts = json.loads(json.dumps(summary))
                attempts['status'] = 'qualifying_fallback'
                for combined in ('gpu_mb4_omp4', 'gpu_mb8_omp4'):
                    prior_e2e = output / combined / 'end_to_end/e2e_result.json'
                    if prior_e2e.exists():
                        attempts['candidates'][combined].update(e2e_result_path=str(prior_e2e), e2e_result_sha256=digest(prior_e2e))
                if not fallback_attempt.exists():
                    write(fallback_attempt, attempts)
            if not e2e.exists(): end_to_end(plan, name)
            profile = selected_profile(plan, path, e2e, baseline, contract_path, fallback_attempt=fallback_attempt)
            write(output / 'selected_profile.json', profile)
            summary.update(status='passed', selected_profile=str(output / 'selected_profile.json'), selected_profile_sha256=digest(output / 'selected_profile.json'))
            write(output / 'suite_status.json', summary)
            return 0
        except Exception as exc:
            summary['candidates'][name]['qualification_error'] = f'{type(exc).__name__}: {exc}'
            write(output / 'suite_status.json', summary)
    summary['status'] = 'failed'; write(output / 'suite_status.json', summary)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    for name in ('run', 'worker'):
        p = sub.add_parser(name); p.add_argument('--plan', required=True); p.add_argument('--candidate', choices=VARIANTS, required=True)
        p.add_argument('--warmup', type=int, default=8); p.add_argument('--steps', type=int, default=32); p.add_argument('--domains', default=','.join(DOMAINS))
    p = sub.add_parser('suite'); p.add_argument('--plan', required=True); p.add_argument('--warmup', type=int, default=8); p.add_argument('--steps', type=int, default=32)
    p = sub.add_parser('end-to-end'); p.add_argument('--plan', required=True); p.add_argument('--candidate', choices=VARIANTS, required=True)
    p = sub.add_parser('contract'); p.add_argument('--plan', required=True)
    p = sub.add_parser('select'); p.add_argument('--plan', required=True); p.add_argument('--candidate', choices=VARIANTS, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.manifest, args.output); print(json.dumps({'plan': str(Path(args.output) / 'plan.json'), 'plan_sha256': result['plan_sha256']})); return 0
    if args.command == 'suite': return suite(args.plan, args.warmup, args.steps)
    if args.command == 'end-to-end': end_to_end(read(args.plan), args.candidate); return 0
    if args.command == 'contract':
        plan = read(args.plan); write(Path(plan['output_root']) / 'production_loss_contract.json', cpu_update_contract(plan)); return 0
    if args.command == 'select':
        plan = read(args.plan); output = Path(plan['output_root'])
        profile = selected_profile(plan, output / args.candidate / 'candidate_result.json', output / args.candidate / 'end_to_end/e2e_result.json', output / 'baseline/candidate_result.json', output / 'production_loss_contract.json')
        write(output / 'selected_profile.json', profile); return 0
    domains = args.domains.split(',')
    if args.command == 'run': return run_candidate(args.plan, args.candidate, args.warmup, args.steps, domains)
    worker(read(args.plan), args.candidate, args.warmup, args.steps, domains)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
