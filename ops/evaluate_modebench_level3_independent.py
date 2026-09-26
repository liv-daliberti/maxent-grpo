#!/usr/bin/env python3
"""Independent prompt/draw RNG streams for the frozen Level 3 interface.

The registered --seeds are draw labels. Effective vLLM request seeds are
aligned eight-child blocks derived from domain, raw problem text, and label.
This version uses a separate receipt/interface namespace and cannot resume v1
receipts. Prompts, syntax, model precision, and sampling budgets are unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'ops', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import evaluate_modebench_level3 as original
from evaluate_modebench_level3 import (
    DOMAINS, atomic_new, file_sha, grade_samples, load_rows, model_identity,
    prompt_messages, sampling_params, sha, summarize,
)
from modebench_independent_seeds import POLICY, SAMPLES, seed_schedule

SCHEMA = 'modebench-level3-frozen-calibration-independent-v2'
INTERFACE = 'level2_qwen_r5_independent_v2'
ENGINE_CONTRACT = {'vllm_version': '0.8.4', 'engine': 'V0',
                   'parallel_sample_seed_policy': 'request_seed_plus_sample_index'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def frozen_interface(domain: str, profile: str = INTERFACE) -> dict[str, Any]:
    require(profile == INTERFACE, 'independent evaluator requires the independent v2 interface')
    return {**original.frozen_interface(domain, 'level2_qwen_r5'),
            'name': INTERFACE, 'seed_policy': POLICY}


def code_identity() -> dict[str, str]:
    return {**original.code_identity(),
            'ops/evaluate_modebench_level3_independent.py': file_sha(Path(__file__)),
            'ops/modebench_independent_seeds.py': file_sha(ROOT / 'ops/modebench_independent_seeds.py')}


def validate_runtime_contract() -> str:
    require(os.environ.get('VLLM_USE_V1') == '0', 'independent seed contract requires VLLM_USE_V1=0')
    version = importlib.metadata.version('vllm')
    require(version == ENGINE_CONTRACT['vllm_version'], 'independent seed contract requires vLLM 0.8.4')
    return version


def validate_task(task: dict[str, Any], confirm_eval: bool) -> None:
    frozen_interface(task['domain'], task.get('interface', INTERFACE))
    original.validate_task({**task, 'interface': 'level2_qwen_r5'}, confirm_eval)
    require(all(type(seed) is int for seed in task['seeds']), 'draw labels must be integers, not booleans')


def schedule_record(domain: str, rows: list[dict], labels: list[int]) -> dict[str, Any]:
    requests = seed_schedule(domain, [row['problem'] for row in rows], labels)
    return {'policy': POLICY, 'sample_count': SAMPLES, 'draw_labels': list(labels),
            'problem_sha256': [sha(row['problem']) for row in rows],
            'request_seeds': requests,
            'child_seeds': [[[base + sample for sample in range(SAMPLES)] for base in prompt]
                            for prompt in requests]}


def draw_seed_metadata(schedule: dict, row_index: int, draw_index: int) -> dict:
    return {'seed': schedule['draw_labels'][draw_index],
            'request_seed': schedule['request_seeds'][row_index][draw_index],
            'child_seeds': schedule['child_seeds'][row_index][draw_index]}


def validate_draw_seed_metadata(draw: dict, schedule: dict, row_index: int, draw_index: int) -> None:
    expected = draw_seed_metadata(schedule, row_index, draw_index)
    require(isinstance(draw, dict) and all(draw.get(key) == value for key, value in expected.items()),
            f'draw RNG metadata differs from authenticated schedule at row {row_index}, draw {draw_index}')
    require(type(draw['seed']) is int and type(draw['request_seed']) is int
            and isinstance(draw['child_seeds'], list) and all(type(seed) is int for seed in draw['child_seeds']),
            'draw RNG metadata must contain exact integers')


def validate_seed_receipt(receipt: dict[str, Any], rows: list[dict] | None = None) -> dict[str, Any]:
    """Authenticate v2 RNG evidence; general metric validation remains separate.

    Supplied rows must be the selected source rows in receipt order. Without
    rows, the receipt's saved source is reopened and authenticated. No model is
    loaded. Registered labels remain under ``seeds`` for campaign bookkeeping;
    only the effective request/child schedule identifies the actual RNG streams.
    """
    require(receipt.get('schema') == SCHEMA and receipt.get('status') == 'complete',
            'independent v2 complete receipt required')
    identity = receipt.get('identity', {})
    require(isinstance(identity, dict) and identity.get('schema') == SCHEMA,
            'independent v2 identity required')
    require(receipt.get('identity_sha256') == sha(identity), 'receipt identity hash mismatch')
    domain = identity.get('domain')
    require(domain in DOMAINS and receipt.get('domain') == domain, 'receipt domain mismatch')
    interface = frozen_interface(domain)
    require(identity.get('interface') == interface and identity.get('interface_sha256') == sha(interface),
            'receipt independent interface/policy mismatch')
    require(identity.get('code_sha256') == code_identity(), 'receipt evaluator/helper code hashes missing or changed')
    require(identity.get('sampling_engine') == ENGINE_CONTRACT
            and identity.get('model', {}).get('vllm_version') == ENGINE_CONTRACT['vllm_version'],
            'receipt vLLM V0 child-seed contract mismatch')
    labels = identity.get('seeds')
    require(isinstance(labels, list) and labels and all(type(seed) is int and seed >= 0 for seed in labels)
            and len(labels) == len(set(labels)), 'invalid registered draw labels')
    require(receipt.get('sampling') == {**interface, 'seeds': labels}, 'receipt sampling metadata mismatch')
    source = identity.get('source', {})
    require(isinstance(source, dict), 'missing source identity')
    if rows is None:
        require(source.get('kind') in ('jsonl', 'saved_dataset') and source.get('path'), 'unsupported source identity')
        task = {'domain': domain, 'rows_jsonl' if source['kind'] == 'jsonl' else 'dataset': source['path'],
                'row_offset': source.get('row_offset', 0), 'row_limit': source.get('row_limit', 0)}
        rows, reopened_source = load_rows(task)
        require(reopened_source == source, 'saved source identity changed')
    require(isinstance(rows, list) and len(rows) == source.get('selected_rows')
            and sha(rows) == source.get('rows_sha256'), 'selected source rows differ from receipt')
    expected = schedule_record(domain, rows, labels)
    require(identity.get('seed_schedule') == expected
            and sha(identity.get('seed_schedule')) == sha(expected)
            and identity.get('seed_schedule_sha256') == sha(expected), 'receipt RNG schedule mismatch')
    results = receipt.get('prompt_results')
    require(isinstance(results, list) and len(results) == len(rows), 'missing prompt RNG evidence')
    for row_index, (source_row, result) in enumerate(zip(rows, results)):
        require(result.get('problem_sha256') == sha(source_row['problem']), 'prompt hash differs from seed source')
        draws = result.get('draws')
        require(isinstance(draws, list) and len(draws) == len(labels), 'missing draw RNG evidence')
        for draw_index, draw in enumerate(draws):
            validate_draw_seed_metadata(draw, expected, row_index, draw_index)
    return {'policy': POLICY, 'prompts': len(rows), 'draws_per_prompt': len(labels),
            'distinct_request_blocks': len(rows) * len(labels),
            'distinct_child_seeds': len(rows) * len(labels) * SAMPLES,
            'seed_schedule_sha256': sha(expected)}


def evaluate_task(llm: Any, tokenizer: Any, task: dict[str, Any], *,
                  model: dict[str, Any], code: dict[str, str], confirm_eval: bool = False,
                  resume: bool = False, grader: Callable | None = None,
                  params_factory: Callable = sampling_params) -> dict[str, Any]:
    validate_task(task, confirm_eval)
    require(code == code_identity(), 'evaluation requires complete current evaluator/helper code hashes')
    output_path = Path(task['output'])
    if output_path.exists() and not resume:
        raise FileExistsError(f'fresh final receipt required: {output_path}')
    rows, source = load_rows(task)
    domain = task['domain']
    interface = frozen_interface(domain, task.get('interface', INTERFACE))
    schedule = schedule_record(domain, rows, task['seeds'])
    prompts = [tokenizer.apply_chat_template(prompt_messages(domain, row['problem'], interface['prompt_profile']),
                                             tokenize=False, add_generation_prompt=True) for row in rows]
    model_context = int(getattr(llm, '_modebench_max_model_len', interface['max_model_len']))
    require(model_context >= interface['max_model_len'], 'loaded model context is too short for the frozen interface')
    for index, prompt in enumerate(prompts):
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        require(token_count + interface['max_tokens'] <= interface['max_model_len'],
                f'row {index} exceeds frozen context budget')
    identity = {'schema': SCHEMA, 'domain': domain, 'level': task['level'], 'split': task.get('split', 'dev'),
                'model': model, 'interface': interface, 'interface_sha256': sha(interface),
                'source': source, 'seeds': task['seeds'], 'batch_size': task.get('batch_size', 8),
                'code_sha256': code, 'rendered_prompts_sha256': sha(prompts),
                'seed_schedule': schedule, 'seed_schedule_sha256': sha(schedule),
                'sampling_engine': dict(ENGINE_CONTRACT)}
    run_sha = sha(identity)
    if output_path.exists():
        completed = json.loads(output_path.read_text())
        require(completed.get('identity_sha256') == run_sha and completed.get('identity') == identity
                and completed.get('status') == 'complete', 'completed receipt identity mismatch; use a fresh output path')
        validate_seed_receipt(completed, rows)
        print(json.dumps({'event': 'task_already_complete', 'output': str(output_path)}), flush=True)
        return completed
    batch_dir = Path(str(output_path) + '.batches')
    manifest_path = batch_dir / 'run.json'
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f'partial run exists; use --resume: {manifest_path}')
        prior = json.loads(manifest_path.read_text())
        require(prior.get('identity_sha256') == run_sha and prior.get('identity') == identity,
                'resume identity mismatch; use a fresh output path')
    else:
        atomic_new(manifest_path, {'identity_sha256': run_sha, 'identity': identity})
    if grader is None:
        from oat_drgrpo.math_grader import validated_modebench_outcome_key
        grader = validated_modebench_outcome_key
    draws_by_row: list[list[dict[str, Any]]] = [[] for _ in rows]
    batch_size = task.get('batch_size', 8)
    for draw_index, label in enumerate(task['seeds']):
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_path = batch_dir / f'seed-{label}__rows-{start:06d}-{end:06d}.json'
            if batch_path.exists():
                batch = json.loads(batch_path.read_text())
                require(batch.get('identity_sha256') == run_sha and batch.get('seed') == label
                        and batch.get('start') == start and batch.get('end') == end
                        and len(batch.get('draws', [])) == end - start
                        and batch.get('draws_sha256') == sha(batch['draws']), f'invalid resumed batch: {batch_path}')
                for row_index, draw in enumerate(batch['draws'], start):
                    validate_draw_seed_metadata(draw, schedule, row_index, draw_index)
            else:
                batch_params = []
                for row_index in range(start, end):
                    base = schedule['request_seeds'][row_index][draw_index]
                    # Each row gets a fresh namespace. A mutating factory cannot
                    # corrupt the next row's registered label or derived seed.
                    params = params_factory(SimpleNamespace(**interface, seed=base), domain, rows[row_index])
                    require(getattr(params, 'seed', None) == base and getattr(params, 'n', None) == SAMPLES,
                            'sampling parameters must retain the scheduled seed and n=8')
                    batch_params.append(params)
                generated = llm.generate(prompts[start:end], batch_params, use_tqdm=False)
                if len(generated) != end - start:
                    raise RuntimeError('vLLM output count mismatch')
                draws = []
                for local, result in enumerate(generated):
                    row_index = start + local
                    if getattr(result, 'prompt', prompts[row_index]) != prompts[row_index]:
                        raise RuntimeError('vLLM returned a mismatched prompt')
                    draws.append({**draw_seed_metadata(schedule, row_index, draw_index),
                                  **grade_samples(rows[row_index], result, grader)})
                batch = {'identity_sha256': run_sha, 'seed': label, 'start': start, 'end': end,
                         'draws': draws, 'draws_sha256': sha(draws)}
                atomic_new(batch_path, batch)
            for row_index, draw in enumerate(batch['draws'], start):
                draws_by_row[row_index].append(draw)
            print(json.dumps({'event': 'batch_complete', 'domain': domain, 'level': task['level'],
                              'seed': label, 'rows_done': end, 'rows_total': len(rows),
                              'batch_pass8': statistics.mean(d['pass8'] for d in batch['draws'])}), flush=True)
    prompt_results = []
    for index, (row, draws) in enumerate(zip(rows, draws_by_row)):
        spec = json.loads(row['answer']) if isinstance(row['answer'], str) else row['answer']
        result = {'row_index': source['row_offset'] + index, 'row_sha256': sha(row),
                  'problem_sha256': sha(row['problem']), 'spec_sha256': sha(spec),
                  'row_metadata': {key: value for key, value in row.items() if key not in ('problem', 'answer')},
                  'draws': draws}
        result.update({metric: statistics.mean(draw[metric] for draw in draws)
                       for metric in ('pass1', 'pass8', 'distinct8')})
        prompt_results.append(result)
    payload = {'schema': SCHEMA, 'generated_at': datetime.now(timezone.utc).isoformat(),
               'status': 'complete', 'identity_sha256': run_sha, 'identity': identity,
               'domain': domain, 'level': task['level'], 'split': task.get('split', 'dev'),
               'model_label': model['label'], 'sampling': {**interface, 'seeds': task['seeds']},
               'metrics': summarize(prompt_results), 'prompt_results': prompt_results,
               'answer_mode_histogram': dict(Counter(str(row.get('answer_mode_count')) for row in rows)),
               'metric_definitions': {'pass1': 'mean verified fraction among each n=8 draw',
                                      'pass8': 'mean indicator of at least one verified sample in each n=8 draw',
                                      'distinct8': 'mean count of unique verified canonical keys in each n=8 draw'},
               'information_boundary': {'evaluation_prompts_loaded': task.get('split', 'dev') == 'eval',
                                        'confirmation_explicitly_authorized': confirm_eval,
                                        'treatment_training_started': False}}
    validate_seed_receipt(payload, rows)
    atomic_new(output_path, payload)
    print(json.dumps({'event': 'task_complete', 'domain': domain, 'level': task['level'],
                      'output': str(output_path), 'metrics': payload['metrics']}), flush=True)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--model-label', required=True, choices=('3b', '05b'))
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--level', default='level3', choices=('level1', 'level2', 'level3'))
    parser.add_argument('--split', default='dev', choices=('dev', 'eval'))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--rows-jsonl', type=Path)
    source.add_argument('--dataset', type=Path)
    parser.add_argument('--tasks-json', type=Path)
    parser.add_argument('--interface', default=INTERFACE, choices=(INTERFACE,))
    parser.add_argument('--output', type=Path)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed', type=int)
    seeds.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--row-limit', type=int, default=0)
    parser.add_argument('--row-offset', type=int, default=0)
    parser.add_argument('--confirm-eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    defaults = {'interface': args.interface, 'level': args.level, 'split': args.split, 'batch_size': args.batch_size,
                'row_limit': args.row_limit, 'row_offset': args.row_offset,
                'seeds': args.seeds or ([args.seed] if args.seed is not None else [])}
    if args.tasks_json:
        raw = json.loads(args.tasks_json.read_text())
        require(isinstance(raw, list) and raw, '--tasks-json must contain a nonempty list')
        tasks = [{**defaults, **task} for task in raw]
        for task in tasks:
            if 'seed' in task:
                task['seeds'] = [task.pop('seed')]
    else:
        require(args.domain and args.output, '--domain and --output are required without --tasks-json')
        tasks = [{**defaults, 'domain': args.domain, 'output': str(args.output),
                  'rows_jsonl': str(args.rows_jsonl) if args.rows_jsonl else None,
                  'dataset': str(args.dataset) if args.dataset else None}]
    require(len({str(Path(task['output']).resolve()) for task in tasks}) == len(tasks), 'tasks must have distinct output paths')
    # Detect cross-task stream collisions before constructing the model. A
    # process can evaluate several domains or separately published pool files.
    seen_blocks = set()
    for task in tasks:
        validate_task(task, args.confirm_eval)
        if Path(task['output']).exists() and not args.resume:
            raise FileExistsError(f"fresh final receipt required: {task['output']}")
        rows, _ = load_rows(task)
        schedule = schedule_record(task['domain'], rows, task['seeds'])
        blocks = {base for row in schedule['request_seeds'] for base in row}
        require(not seen_blocks & blocks, 'duplicate or colliding RNG blocks across task manifest')
        seen_blocks |= blocks
    runtime_version = validate_runtime_contract()
    identity = model_identity(args.model, args.model_label)
    code = code_identity()
    import vllm
    identity['vllm_version'] = runtime_version
    max_context = max(frozen_interface(task['domain'], task['interface'])['max_model_len'] for task in tasks)
    llm = vllm.LLM(model=str(args.model.resolve()), dtype='float16', max_model_len=max_context,
                   gpu_memory_utilization=.82, swap_space=16.0, enable_prefix_caching=True)
    llm._modebench_max_model_len = max_context
    tokenizer = llm.get_tokenizer()
    for task in tasks:
        evaluate_task(llm, tokenizer, task, model=identity, code=code,
                      confirm_eval=args.confirm_eval, resume=args.resume)


if __name__ == '__main__':
    main()
