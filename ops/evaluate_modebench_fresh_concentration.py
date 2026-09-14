#!/usr/bin/env python3
"""Collect a new, plan-bound 64-draw panel without training or old-draw reuse.

Eight n=8 requests use the audited vLLM 0.8.4 V0 child-seed contract. Every
(task, prompt, block) owns a disjoint hash-aligned block of eight seeds. A
retry reuses only authenticated, atomically committed batches; an interrupted
uncommitted batch may be executed again, but each output slot is committed once.
Scientific settings come from each task's hashed normalized evaluation evidence,
not the unrelated 192-token/native-chat calibration defaults.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable
import uuid

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'ops', ROOT / 'src'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from evaluate_modebench_level3 import atomic_new, file_sha, sha
from modebench_independent_seeds import POLICY, SAMPLES, seed_schedule

SCHEMA = 'modebench-fresh-concentration-v1'
ENGINE_CONTRACT = {'vllm_version': '0.8.4', 'engine': 'V0',
                   'parallel_sample_seed_policy': 'request_seed_plus_sample_index'}
RUNTIME_ENV_KEYS = ('VLLM_USE_V1', 'VLLM_ATTENTION_BACKEND', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')
PROMPT_ENCODING = 'rendered_text_vllm_v0'
DOMAINS = {'graph_coloring', 'pantry_plan'}
SAMPLING_KEYS = {'n', 'temperature', 'top_p', 'max_tokens', 'min_tokens',
                 'ignore_eos', 'stop', 'stop_token_ids',
                 'include_stop_str_in_output', 'allowed_token_ids'}
ENGINE_KEYS = {'dtype', 'max_model_len', 'tensor_parallel_size',
               'gpu_memory_utilization', 'swap_space', 'enable_prefix_caching',
               'enforce_eager'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in resolve(path).read_text().splitlines() if line.strip()]


def code_identity() -> dict[str, str]:
    paths = [Path(__file__), ROOT / 'ops/modebench_independent_seeds.py',
             ROOT / 'ops/evaluate_modebench_level3.py',
             ROOT / 'ops/evaluate_modebench_level2_viability.py']
    # Bind the local verifier, canonical decoder, templates, and their helpers.
    paths += sorted((ROOT / 'src/oat_drgrpo').glob('*.py'))
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def default_templates() -> dict[str, Callable]:
    from oat_drgrpo.templates import TEMPLATE_FACTORY
    return TEMPLATE_FACTORY


def source_contract(task: dict) -> dict:
    """Fields normalized from recorded evaluation settings, with provenance elsewhere."""
    return {key: task[key] for key in ('prompt_template', 'syntax_profile',
            'response_decoder', 'prompt_encoding', 'sampling')} | {'dtype': task['engine']['dtype']}


def validate_file_manifest(task: dict, *, hash_files: bool) -> None:
    model = resolve(task['model_path'])
    entries = task['files']
    require(isinstance(entries, list) and entries, 'checkpoint file manifest is required')
    names = [item['name'] for item in entries]
    require(len(names) == len(set(names)), 'duplicate checkpoint file')
    require('config.json' in names, 'checkpoint config.json must be hashed')
    require(any(name.endswith(('.safetensors', '.bin')) for name in names),
            'checkpoint weight files must be hashed')
    for entry in entries:
        name = entry['name']
        require(isinstance(name, str) and not Path(name).is_absolute()
                and '..' not in Path(name).parts, 'invalid checkpoint file name')
        require(type(entry['bytes']) is int and entry['bytes'] >= 0
                and re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None,
                'invalid checkpoint file identity')
        if hash_files:
            path = model / name
            require(path.is_file() and path.stat().st_size == entry['bytes']
                    and file_sha(path) == entry['sha256'], f'checkpoint missing or changed: {path}')
    if hash_files:
        # All recognized loader inputs must be bound, including optional generation
        # and tokenizer files. Unexpected weights/configs cannot silently take effect.
        consumed = {path.name for path in model.iterdir() if path.is_file() and
                    (path.suffix in {'.safetensors', '.bin'} or path.name in {
                        'config.json', 'generation_config.json', 'model.safetensors.index.json',
                        'pytorch_model.bin.index.json', 'tokenizer.json', 'tokenizer.model',
                        'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                        'vocab.json', 'vocab.txt', 'merges.txt', 'spiece.model'})}
        require(consumed <= set(names), f'unhashed model/tokenizer loader files: {sorted(consumed-set(names))}')


def task_schedule(plan: dict, task: dict, rows: list[dict]) -> list[list[int]]:
    namespace = json.dumps([plan['seed_namespace'], task['task_id'], task['domain']],
                           separators=(',', ':'))
    return seed_schedule(namespace, [row['problem'] for row in rows], plan['draw_labels'])


def validate_plan(plan: dict, *, templates: dict | None = None) -> dict[str, list[dict]]:
    """CPU preflight; hashes small inputs/code, never opens large model weights."""
    require(plan.get('schema') == SCHEMA, 'wrong collection schema')
    require(isinstance(plan.get('campaign_id'), str) and bool(plan['campaign_id']), 'campaign_id required')
    require(isinstance(plan.get('seed_namespace'), str) and bool(plan['seed_namespace']), 'fresh seed_namespace required')
    require(plan.get('draws_per_prompt') == 64 and plan.get('prompts_per_task') == 128,
            'this panel requires 64 new draws for each of 128 prompts')
    labels = plan.get('draw_labels')
    require(isinstance(labels, list) and len(labels) == 8
            and all(type(label) is int and label >= 0 for label in labels)
            and len(set(labels)) == 8, 'eight distinct nonnegative draw labels required')
    require(type(plan.get('batch_size')) is int and 1 <= plan['batch_size'] <= 128,
            'batch_size must be between 1 and 128 prompt requests')
    require(isinstance(plan.get('output_root'), str) and bool(plan['output_root']), 'output_root required')
    expected_code = code_identity()
    require(plan.get('code_sha256') == expected_code, 'collector/verifier/helper code hashes changed or incomplete')
    input_hashes = plan.get('input_sha256')
    require(isinstance(input_hashes, dict) and input_hashes, 'input file hashes required')
    for name, digest in input_hashes.items():
        require(file_sha(resolve(name)) == digest, f'input file changed: {name}')
    if templates is None:
        templates = default_templates()
    tasks = plan.get('tasks')
    require(isinstance(tasks, list) and tasks, 'nonempty tasks required')
    seen_ids, seen_seeds, all_rows = set(), set(), {}
    for task in tasks:
        task_id = task['task_id']
        require(isinstance(task_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', task_id)
                and task_id not in seen_ids, 'invalid or duplicate task_id')
        seen_ids.add(task_id)
        require(task['domain'] in DOMAINS and task['level'] == 1, 'only Level-1 Graph/Pantry tasks admitted')
        require(task['checkpoint_stage'] in ('initial', 'terminal'), 'checkpoint_stage must be initial or terminal')
        if task['checkpoint_stage'] == 'initial':
            require(task['method'] == 'initial' and task['training_seed'] is None
                    and type(task.get('eval_replica_id')) is int,
                    'initial tasks require method=initial, training_seed=null, and integer eval_replica_id')
        else:
            require(task['method'] in ('drgrpo', 'replay_drgrpo', 'maxrl', 'replay_maxrl')
                    and type(task['training_seed']) is int and task.get('eval_replica_id') is None,
                    'terminal tasks require a registered method, integer training_seed, and no eval_replica_id')
        require(isinstance(task['model_scale'], str), 'model_scale required')
        require(task['prompt_encoding'] == PROMPT_ENCODING, 'sampled evaluation must use recorded rendered-text encoding')
        require(task['prompt_template'] in templates, 'unknown recorded prompt template')
        require(task['syntax_profile'] in ('none', 'domain_legal_v1'), 'unsupported syntax profile')
        require(task['response_decoder'] in ('identity', 'pantry_support_mask'), 'unsupported response decoder')
        require(set(task['sampling']) == SAMPLING_KEYS, 'sampling settings must be complete and explicit')
        sampling = task['sampling']
        require(sampling['n'] == SAMPLES and type(sampling['n']) is int, 'n=8 child-seed contract required')
        require(0 < sampling['temperature'] <= 2 and 0 < sampling['top_p'] <= 1, 'invalid sampling temperature/top_p')
        require(type(sampling['max_tokens']) is int and sampling['max_tokens'] > 0
                and type(sampling['min_tokens']) is int and 0 <= sampling['min_tokens'] <= sampling['max_tokens'],
                'invalid token budget')
        require(type(sampling['ignore_eos']) is bool and type(sampling['include_stop_str_in_output']) is bool,
                'EOS and stop behavior must be explicit booleans')
        allowed = sampling['allowed_token_ids']
        require(allowed is None or (isinstance(allowed, list) and allowed
                and all(type(token) is int and token >= 0 for token in allowed)
                and len(allowed) == len(set(allowed))), 'invalid allowed token support')
        if task['response_decoder'] == 'pantry_support_mask':
            require(task['domain'] == 'pantry_plan' and allowed is not None
                    and sampling['max_tokens'] == sampling['min_tokens'] == 6
                    and sampling['ignore_eos'] and sampling['stop'] is None
                    and sampling['stop_token_ids'] is None, 'Pantry mask needs its recorded fixed action support')
        require(set(task['engine']) == ENGINE_KEYS, 'engine settings must be explicit')
        require(task['engine']['dtype'] in ('float16', 'bfloat16', 'float32'), 'explicit engine dtype required')
        require(type(task['engine']['max_model_len']) is int
                and task['engine']['max_model_len'] > sampling['max_tokens'], 'invalid context budget')
        validate_file_manifest(task, hash_files=False)
        source = task['source_eval_config']
        require(source['path'] in input_hashes and input_hashes[source['path']] == source['sha256'],
                'source evaluation evidence must be included in input hashes')
        evidence = json.loads(resolve(source['path']).read_text())
        require(all(evidence.get(key) == value for key, value in source_contract(task).items()),
                'task settings differ from normalized source evaluation evidence')
        require(task['prompts_path'] in input_hashes, 'prompt records must be file-hashed')
        rows = read_jsonl(task['prompts_path'])
        require(len(rows) == 128, 'exactly 128 fixed held-out prompts required')
        require(len({row['prompt_id'] for row in rows}) == 128
                and len({row['row_index'] for row in rows}) == 128, 'duplicate prompt identity or row index')
        for row in rows:
            require(isinstance(row['prompt_id'], str) and bool(row['prompt_id'])
                    and type(row['row_index']) is int and row['row_index'] >= 0, 'invalid prompt identity')
            require(isinstance(row['problem'], str) and bool(row['problem'])
                    and isinstance(row['answer'], str), 'raw problem and verifier reference strings required')
            require(row['rendered_prompt'] == templates[task['prompt_template']](row['problem']),
                    'rendered prompt differs from recorded training template')
        schedule = task_schedule(plan, task, rows)
        blocks = {base for seeds in schedule for base in seeds}
        require(not seen_seeds & blocks, 'RNG collision across checkpoint tasks')
        seen_seeds.update(blocks)
        all_rows[task_id] = rows
    return all_rows


def make_params(task: dict, row: dict, request_seed: int):
    import vllm
    from oat_drgrpo.modebench_guided import guided_sampling_params
    params = vllm.SamplingParams(**task['sampling'], seed=request_seed)
    guided = guided_sampling_params(params, task['syntax_profile'], task['domain'], [row['answer']])
    return guided[0] if isinstance(guided, list) else guided


def default_decoder(task: dict, text: str, reference: str) -> str:
    text = text.strip()  # Same boundary as generate_for_mode_coverage.
    if task['response_decoder'] == 'identity':
        return text
    from oat_drgrpo.canonical_actions import decode_canonical_action_response
    return decode_canonical_action_response(task['response_decoder'], text, reference)


def request_identity(task: dict, row: dict, block: int, label: int, seed: int) -> dict:
    return {'task_id': task['task_id'], 'prompt_id': row['prompt_id'], 'row_index': row['row_index'],
            'row_sha256': sha(row), 'problem_sha256': sha(row['problem']), 'reference_sha256': sha(row['answer']),
            'rendered_prompt_sha256': sha(row['rendered_prompt']), 'draw_block': block,
            'draw_label': label, 'request_seed': seed, 'child_seeds': [seed+i for i in range(SAMPLES)]}


def grade_request(task: dict, row: dict, generated: Any, expected: dict,
                  grader: Callable, decoder: Callable = default_decoder) -> dict:
    require(generated.prompt == row['rendered_prompt'], 'engine returned a different prompt/order')
    require(len(generated.outputs) == SAMPLES, 'engine must return exactly eight child outputs')
    children = {sample.index: sample for sample in generated.outputs}
    require(len(children) == SAMPLES and set(children) == set(range(SAMPLES)), 'duplicate/missing child output index')
    attempts = []
    allowed = task['sampling']['allowed_token_ids']
    for index in range(SAMPLES):
        sample = children[index]
        tokens = list(sample.token_ids)
        require(all(type(token) is int and token >= 0 for token in tokens), 'invalid output token ids')
        require(len(tokens) <= task['sampling']['max_tokens'], 'output exceeds registered token budget')
        if allowed is not None:
            require(set(tokens) <= set(allowed), 'output escaped recorded action support')
        if task['response_decoder'] == 'pantry_support_mask':
            require(len(tokens) == task['sampling']['max_tokens'] and sample.finish_reason == 'length', 'Pantry mask violates fixed horizon')
        raw_text = sample.text
        require(isinstance(raw_text, str), 'raw output must be text')
        verifier_text = decoder(task, raw_text, row['answer'])
        key = grader(verifier_text, row['answer'])
        attempts.append({'child_index': index, 'draw_index': expected['draw_block']*8+index,
                         'child_sampling_seed': expected['request_seed']+index,
                         'text': raw_text, 'verifier_text': verifier_text, 'verified': key is not None,
                         'canonical_key': key, 'token_ids': tokens, 'token_count': len(tokens),
                         'finish_reason': sample.finish_reason,
                         'stop_reason': getattr(sample, 'stop_reason', None)})
    return {**expected, 'prompt_token_ids': list(generated.prompt_token_ids), 'attempts': attempts}


def validate_batch(batch: dict, identity_sha: str, expected: list[dict], task: dict,
                   rows: list[dict], grader: Callable, decoder: Callable = default_decoder) -> None:
    require(batch.get('identity_sha256') == identity_sha and batch.get('requests_sha256') == sha(batch.get('requests')),
            'batch identity or content hash changed')
    requests = batch.get('requests')
    require(isinstance(requests, list) and len(requests) == len(expected), 'batch request count differs')
    for record, fields, row in zip(requests, expected, rows):
        require(all(record.get(key) == value for key, value in fields.items()), 'batch prompt/block/RNG identity differs')
        require(isinstance(record.get('prompt_token_ids'), list)
                and all(type(token) is int and token >= 0 for token in record['prompt_token_ids']), 'missing prompt tokens')
        attempts = record.get('attempts')
        require(isinstance(attempts, list) and len(attempts) == 8, 'missing child output slots')
        for index, attempt in enumerate(attempts):
            require(attempt.get('child_index') == index and attempt.get('draw_index') == fields['draw_block']*8+index
                    and attempt.get('child_sampling_seed') == fields['request_seed']+index, 'child slot or RNG changed')
            tokens = attempt.get('token_ids')
            require(isinstance(tokens, list) and all(type(token) is int and token >= 0 for token in tokens)
                    and len(tokens) == attempt.get('token_count')
                    and len(tokens) <= task['sampling']['max_tokens'], 'saved output token count differs')
            allowed = task['sampling']['allowed_token_ids']
            require(allowed is None or set(tokens) <= set(allowed), 'saved output escaped action support')
            if task['response_decoder'] == 'pantry_support_mask':
                require(len(tokens) == task['sampling']['max_tokens'] and attempt.get('finish_reason') == 'length', 'saved mask violates horizon')
            text = decoder(task, attempt['text'], row['answer'])
            key = grader(text, row['answer'])
            require(attempt.get('verifier_text') == text and attempt.get('canonical_key') == key
                    and type(attempt.get('verified')) is bool and attempt['verified'] == (key is not None),
                    'saved decoded response or verifier key differs')


def runtime_fingerprint(runtime: dict) -> str:
    """Bind numerical runtime settings while allowing a new physical allocation."""
    stable = dict(runtime)
    if 'environment' in stable:
        stable['environment'] = {key: value for key, value in stable['environment'].items()
                                 if key != 'CUDA_VISIBLE_DEVICES'}
    return sha(stable)


def runtime_info() -> dict:
    require(os.environ.get('VLLM_USE_V1') == '0', 'child seed contract requires VLLM_USE_V1=0')
    require(importlib.metadata.version('vllm') == '0.8.4', 'child seed contract requires vLLM 0.8.4')
    import torch
    versions = {}
    for package in ('vllm', 'torch', 'transformers', 'tokenizers', 'numpy', 'sympy',
                    'math-verify', 'latex2sympy2-extended', 'pylatexenc'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'engine_contract': ENGINE_CONTRACT, 'versions': versions,
            'environment': {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS},
            'python': sys.version, 'cuda': torch.version.cuda,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'gpu_capabilities': [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]}


def default_engine(task: dict):
    import vllm
    return vllm.LLM(model=str(resolve(task['model_path'])), **task['engine'])


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def collect_task(plan_path: Path, task_index: int, *, templates: dict | None = None,
                 engine_factory: Callable = default_engine, runtime_factory: Callable = runtime_info,
                 params_factory: Callable = make_params, grader: Callable | None = None,
                 decoder: Callable = default_decoder) -> dict:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    rows_by_task = validate_plan(plan, templates=templates)
    require(type(task_index) is int and 0 <= task_index < len(plan['tasks']), 'task index out of range')
    task = plan['tasks'][task_index]
    rows = rows_by_task[task['task_id']]
    schedule = task_schedule(plan, task, rows)
    identity = {'schema': SCHEMA, 'plan_sha256': file_sha(plan_path), 'task': task,
                'seed_policy': POLICY, 'engine_contract': ENGINE_CONTRACT,
                'seed_namespace': plan['seed_namespace'], 'draw_labels': plan['draw_labels'],
                'request_seeds': schedule, 'rows_sha256': sha(rows), 'batch_size': plan['batch_size']}
    identity_sha = sha(identity)
    output = resolve(plan['output_root']) / task['task_id']
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path = output/'run.json'
        run = {'identity_sha256': identity_sha, 'identity': identity}
        if run_path.exists():
            require(json.loads(run_path.read_text()) == run, 'resumed run identity differs')
        else:
            atomic_new(run_path, run)
        validate_file_manifest(task, hash_files=True)
        if grader is None:
            from oat_drgrpo.math_grader import validated_modebench_outcome_key
            grader = validated_modebench_outcome_key
        llm, runtime_sha, runtime = None, None, None
        final_path = output/'result.json'
        final_saved = json.loads(final_path.read_text()) if final_path.exists() else None
        runtime_path = output/'runtime.json'
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text())
            require(runtime.get('identity_sha256') == identity_sha
                    and runtime.get('runtime_sha256') == runtime_fingerprint(runtime.get('runtime')), 'runtime receipt changed')
            runtime_sha = runtime['runtime_sha256']
        batches, flattened = [], []
        for block, label in enumerate(plan['draw_labels']):
            for start in range(0, len(rows), plan['batch_size']):
                selected = rows[start:start+plan['batch_size']]
                expected = [request_identity(task, row, block, label, schedule[index][block])
                            for index, row in enumerate(selected, start)]
                path = output/f'batch_b{block:02d}_{start:04d}.json'
                if path.exists():
                    batch = json.loads(path.read_text())
                else:
                    require(final_saved is None, 'completed panel has a missing committed batch')
                    if llm is None:
                        current_runtime = runtime_factory()
                        require(current_runtime.get('engine_contract') == ENGINE_CONTRACT, 'runtime engine contract mismatch')
                        current_sha = runtime_fingerprint(current_runtime)
                        if runtime is None:
                            runtime = {'identity_sha256': identity_sha, 'runtime_sha256': current_sha,
                                       'runtime': current_runtime}
                            atomic_new(runtime_path, runtime)
                            runtime_sha = current_sha
                        else:
                            require(current_sha == runtime_sha, 'runtime versions/hardware changed during resume')
                        stamp = datetime.now(timezone.utc).isoformat()
                        attempts_dir = output/'attempts'
                        attempts_dir.mkdir(exist_ok=True)
                        atomic_new(attempts_dir/f'{uuid.uuid4().hex}.json', {
                            'identity_sha256': identity_sha, 'runtime_sha256': runtime_sha,
                            'started_at': stamp, 'hostname': os.uname().nodename,
                            'environment': current_runtime.get('environment', {}),
                            'pid': os.getpid(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID')})
                        llm = engine_factory(task)
                        tokenizer = llm.get_tokenizer()
                        if task['response_decoder'] == 'pantry_support_mask':
                            support = [tokenizer.encode(bit, add_special_tokens=False) for bit in ('0', '1')]
                            require(all(len(ids) == 1 for ids in support)
                                    and [ids[0] for ids in support] == task['sampling']['allowed_token_ids'],
                                    'recorded Pantry token ids differ from loaded tokenizer')
                        for row in rows:
                            # Direct-text inference tokenization matches sampled eval.
                            tokens = tokenizer.encode(row['rendered_prompt'])
                            require(len(tokens)+task['sampling']['max_tokens'] <= task['engine']['max_model_len'],
                                    'fixed prompt plus response exceeds recorded context budget')
                    params = [params_factory(task, row, fields['request_seed'])
                              for row, fields in zip(selected, expected)]
                    require(all(getattr(param, 'n', None) == 8 and getattr(param, 'seed', None) == fields['request_seed']
                                for param, fields in zip(params, expected)), 'factory changed n or scheduled seed')
                    generated = llm.generate([row['rendered_prompt'] for row in selected], params, use_tqdm=False)
                    require(len(generated) == len(selected), 'engine request count differs')
                    requests = [grade_request(task, row, result, fields, grader, decoder)
                                for row, result, fields in zip(selected, generated, expected)]
                    batch = {'identity_sha256': identity_sha, 'runtime_sha256': runtime_sha,
                             'requests': requests, 'requests_sha256': sha(requests)}
                    validate_batch(batch, identity_sha, expected, task, selected, grader, decoder)
                    atomic_new(path, batch)
                require(runtime_sha is not None and batch.get('runtime_sha256') == runtime_sha,
                        'batch is not bound to the recorded runtime')
                validate_batch(batch, identity_sha, expected, task, selected, grader, decoder)
                batches.append({'name': path.name, 'sha256': file_sha(path), 'requests': len(selected)})
                for row, record in zip(selected, batch['requests']):
                    meta = {key: value for key, value in record.items() if key != 'attempts'}
                    for attempt in record['attempts']:
                        flattened.append({'schema': SCHEMA, 'domain': task['domain'], 'level': task['level'],
                                          'model_scale': task['model_scale'], 'method': task['method'],
                                          'training_seed': task['training_seed'], 'checkpoint_stage': task['checkpoint_stage'],
                                          'eval_replica_id': task.get('eval_replica_id'),
                                          **meta, **attempt})
                print(json.dumps({'event': 'batch_complete', 'task_id': task['task_id'],
                                  'draw_block': block, 'prompts_done': start+len(selected)}), flush=True)
        slots = {(row['prompt_id'], row['draw_index']) for row in flattened}
        require(len(flattened) == len(slots) == 128*64, 'final draw slots are missing or duplicated')
        require(len({row['child_sampling_seed'] for row in flattened}) == 128*64, 'final child streams overlap')
        responses_path = output/'responses.jsonl'
        if responses_path.exists():
            require(read_jsonl(responses_path) == flattened, 'existing responses differ from committed batches')
        else:
            require(final_saved is None, 'completed panel is missing its responses file')
            atomic_jsonl(responses_path, flattened)
        result = {'schema': SCHEMA, 'status': 'complete', 'identity_sha256': identity_sha,
                  'runtime_sha256': runtime_sha, 'task_id': task['task_id'], 'prompts': 128,
                  'draws_per_prompt': 64, 'draws': len(flattened), 'batches': batches,
                  'responses_path': str(responses_path), 'responses_sha256': file_sha(responses_path),
                  'interpretation': 'fresh disjoint request/child RNG streams; initial replicas are evaluation replicas; no old draws pooled'}
        if final_saved is None:
            atomic_new(final_path, result)
        else:
            require(final_saved == result, 'completed result receipt differs from validated batches')
        print(json.dumps({'event': 'complete', 'task_id': task['task_id'], 'draws': len(flattened)}), flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--validate-plan', '--validate-only', dest='validate_plan', action='store_true')
    parser.add_argument('--verify-checkpoint', action='store_true', help='also hash selected task weights during preflight')
    args = parser.parse_args()
    if args.validate_plan:
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        if args.verify_checkpoint:
            if args.task_index is None:
                parser.error('--verify-checkpoint requires --task-index')
            require(0 <= args.task_index < len(plan['tasks']), 'task index out of range')
            validate_file_manifest(plan['tasks'][args.task_index], hash_files=True)
        print(json.dumps({'status': 'pass', 'tasks': len(plan['tasks']),
                          'draws': len(plan['tasks'])*128*64, 'large_weights_hashed': args.verify_checkpoint}))
    else:
        if args.task_index is None:
            parser.error('--task-index required for collection')
        collect_task(args.plan, args.task_index)


if __name__ == '__main__':
    main()
