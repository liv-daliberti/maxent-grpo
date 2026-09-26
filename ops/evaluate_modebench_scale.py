#!/usr/bin/env python3
"""Evaluate Qwen scale candidates without changing the frozen Level 3 evaluator.

This receipt namespace records scale levels and the actual engine settings. The
Level 3 prompt, grammar, grader, sampling budgets, and independent RNG helpers
are reused unchanged. Four registered draw labels are required. ``eval`` is the
held-out test split and requires --confirm-eval; ``dev`` supports development
pool slices. This evaluator makes no difficulty admission or training decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ops') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ops'))
import evaluate_modebench_level3_independent as frozen

DOMAINS = frozen.DOMAINS
SCHEMA = 'modebench-scale-calibration-independent-v1'
INTERFACE = 'modebench_qwen_scale_independent_v1'
LEVELS = ('level1', 'level3', 'level4', 'level5')
MODEL_LABELS = ('05b', '7b', '14b')
DRAW_COUNT = 4
ENGINE_CONTRACT = dict(frozen.ENGINE_CONTRACT)
sha = frozen.sha
load_rows = frozen.load_rows
model_identity = frozen.model_identity
require = frozen.require


def frozen_interface(domain: str, profile: str = INTERFACE) -> dict[str, Any]:
    require(profile == INTERFACE, 'scale evaluator requires the scale interface namespace')
    return {**frozen.frozen_interface(domain), 'name': INTERFACE}


def code_identity() -> dict[str, str]:
    return {**frozen.code_identity(),
            'ops/evaluate_modebench_scale.py': frozen.file_sha(Path(__file__))}


def runtime_settings(*, max_model_len: int, tensor_parallel_size: int = 1,
                     gpu_memory_utilization: float = .82, swap_space: float = 16.0,
                     enable_prefix_caching: bool = True) -> dict[str, Any]:
    settings = {'dtype': 'float16', 'max_model_len': max_model_len,
                'tensor_parallel_size': tensor_parallel_size,
                'gpu_memory_utilization': gpu_memory_utilization,
                'swap_space': swap_space, 'enable_prefix_caching': enable_prefix_caching}
    validate_runtime_settings(settings)
    return settings


def validate_runtime_settings(runtime: dict, domain: str | None = None) -> None:
    require(isinstance(runtime, dict) and set(runtime) == {
        'dtype', 'max_model_len', 'tensor_parallel_size', 'gpu_memory_utilization',
        'swap_space', 'enable_prefix_caching'}, 'complete scale runtime settings required')
    require(runtime['dtype'] == 'float16', 'frozen sampling requires float16')
    require(type(runtime['tensor_parallel_size']) is int and runtime['tensor_parallel_size'] >= 1,
            'tensor_parallel_size must be a positive integer')
    minimum_context = frozen_interface(domain)['max_model_len'] if domain else 1
    require(type(runtime['max_model_len']) is int and runtime['max_model_len'] >= minimum_context,
            'loaded model context is too short for the frozen interface')
    for name in ('gpu_memory_utilization', 'swap_space'):
        require(type(runtime[name]) in (int, float) and math.isfinite(runtime[name]),
                f'{name} must be a finite number')
    require(0 < runtime['gpu_memory_utilization'] <= 1, 'gpu_memory_utilization must be in (0, 1]')
    require(runtime['swap_space'] >= 0, 'swap_space must be nonnegative')
    require(type(runtime['enable_prefix_caching']) is bool, 'enable_prefix_caching must be boolean')


def validate_labels(labels: Any) -> None:
    require(isinstance(labels, list) and len(labels) == DRAW_COUNT
            and all(type(label) is int and label >= 0 for label in labels)
            and len(set(labels)) == DRAW_COUNT,
            'exactly four distinct nonnegative integer draw labels are required')


def validate_task(task: dict[str, Any], confirm_eval: bool) -> None:
    frozen_interface(task['domain'], task.get('interface', INTERFACE))
    require(task.get('level') in LEVELS, f'level must be one of {LEVELS}')
    require(bool(task.get('rows_jsonl')) != bool(task.get('dataset')),
            'choose exactly one explicit rows_jsonl or dataset source')
    for name, default, minimum in (('row_offset', 0, 0), ('row_limit', 0, 0), ('batch_size', 8, 1)):
        require(type(task.get(name, default)) is int and task.get(name, default) >= minimum,
                f'{name} must be an integer >= {minimum}')
    require(isinstance(task.get('output'), str) and bool(task['output']), 'output path is required')
    validate_labels(task.get('seeds'))
    # This checks the unchanged source, split, and held-out slicing rules. The
    # translated task is validation-only: it is never evaluated or published.
    frozen.validate_task({**task, 'level': 'level3', 'interface': frozen.INTERFACE}, confirm_eval)


def _validate_draw_metrics(draw: dict) -> None:
    attempts = draw.get('attempts')
    require(isinstance(attempts, list) and len(attempts) == frozen.SAMPLES,
            'each scale draw requires exactly eight saved attempts')
    for attempt in attempts:
        require(isinstance(attempt, dict) and isinstance(attempt.get('text'), str)
                and type(attempt.get('verified')) is bool
                and 'canonical_key' in attempt
                and attempt['verified'] == (attempt['canonical_key'] is not None)
                and type(attempt.get('token_count')) is int and attempt['token_count'] >= 0,
                'invalid saved graded attempt')
    verified = sum(attempt['verified'] for attempt in attempts)
    expected = {'verified_count': verified, 'pass1': verified / frozen.SAMPLES,
                'pass8': float(verified > 0),
                'distinct8': len({sha(attempt['canonical_key']) for attempt in attempts if attempt['verified']})}
    require(all(draw.get(key) == value for key, value in expected.items()),
            'draw metrics differ from saved verified attempts')


def validate_seed_receipt(receipt: dict[str, Any], rows: list[dict] | None = None) -> dict[str, Any]:
    """Authenticate scale identity, runtime, source rows, and independent RNG.

    This intentionally rejects the frozen L3 schemas rather than relabeling
    them. Sampling and grading semantics come from the frozen helper modules.
    """
    require(receipt.get('schema') == SCHEMA and receipt.get('status') == 'complete',
            'complete scale receipt required')
    identity = receipt.get('identity', {})
    require(isinstance(identity, dict) and identity.get('schema') == SCHEMA,
            'scale identity namespace required')
    require(receipt.get('identity_sha256') == sha(identity), 'receipt identity hash mismatch')
    domain = identity.get('domain')
    require(domain in DOMAINS and receipt.get('domain') == domain, 'receipt domain mismatch')
    require(identity.get('level') in LEVELS and receipt.get('level') == identity['level'],
            'receipt scale level mismatch')
    require(identity.get('split') in ('dev', 'eval') and receipt.get('split') == identity['split'],
            'receipt split mismatch')
    model = identity.get('model', {})
    require(isinstance(model, dict) and model.get('label') in MODEL_LABELS
            and receipt.get('model_label') == model['label'], 'receipt scale model label mismatch')
    require(identity.get('sampling_engine') == ENGINE_CONTRACT
            and model.get('vllm_version') == ENGINE_CONTRACT['vllm_version'],
            'receipt vLLM V0 child-seed contract mismatch')
    validate_runtime_settings(identity.get('runtime'), domain)
    interface = frozen_interface(domain)
    require(identity.get('interface') == interface and identity.get('interface_sha256') == sha(interface),
            'receipt scale interface mismatch')
    require(identity.get('code_sha256') == code_identity(), 'receipt evaluator/helper code hashes missing or changed')
    labels = identity.get('seeds')
    validate_labels(labels)
    require(receipt.get('sampling') == {**interface, 'seeds': labels}, 'receipt sampling metadata mismatch')
    source = identity.get('source', {})
    require(isinstance(source, dict), 'missing source identity')
    boundary = receipt.get('information_boundary', {})
    require(isinstance(boundary, dict)
            and boundary.get('evaluation_prompts_loaded') is (identity['split'] == 'eval')
            and boundary.get('treatment_training_started') is False
            and (identity['split'] != 'eval' or (
                boundary.get('confirmation_explicitly_authorized') is True
                and source.get('row_offset') == 0 and source.get('row_limit') == 0)),
            'receipt held-out evaluation boundary mismatch')
    if rows is None:
        require(source.get('kind') in ('jsonl', 'saved_dataset') and source.get('path'),
                'unsupported source identity')
        rows, reopened_source = load_rows({
            'domain': domain,
            'rows_jsonl' if source['kind'] == 'jsonl' else 'dataset': source['path'],
            'row_offset': source.get('row_offset', 0), 'row_limit': source.get('row_limit', 0)})
        require(reopened_source == source, 'saved source identity changed')
    require(isinstance(rows, list) and len(rows) == source.get('selected_rows')
            and sha(rows) == source.get('rows_sha256'), 'selected source rows differ from receipt')
    schedule = frozen.schedule_record(domain, rows, labels)
    require(sha(identity.get('seed_schedule')) == sha(schedule)
            and identity.get('seed_schedule_sha256') == sha(schedule), 'receipt RNG schedule mismatch')
    results = receipt.get('prompt_results')
    require(isinstance(results, list) and len(results) == len(rows), 'missing prompt RNG evidence')
    for row_index, (row, result) in enumerate(zip(rows, results)):
        spec = json.loads(row['answer']) if isinstance(row['answer'], str) else row['answer']
        require(isinstance(result, dict) and result.get('problem_sha256') == sha(row['problem'])
                and result.get('row_sha256') == sha(row) and result.get('spec_sha256') == sha(spec)
                and result.get('row_index') == source['row_offset'] + row_index
                and result.get('row_metadata') == {key: value for key, value in row.items()
                                                  if key not in ('problem', 'answer')},
                'prompt or executable spec differs from source')
        draws = result.get('draws')
        require(isinstance(draws, list) and len(draws) == DRAW_COUNT, 'missing draw RNG evidence')
        for draw_index, draw in enumerate(draws):
            frozen.validate_draw_seed_metadata(draw, schedule, row_index, draw_index)
            _validate_draw_metrics(draw)
        require(all(result.get(metric) == statistics.mean(draw[metric] for draw in draws)
                    for metric in ('pass1', 'pass8', 'distinct8')),
                'prompt metrics differ from saved draws')
    require(receipt.get('metrics') == frozen.summarize(results), 'summary metrics differ from saved prompts')
    require(receipt.get('answer_mode_histogram') == dict(Counter(str(row.get('answer_mode_count')) for row in rows)),
            'answer mode histogram differs from source')
    return {'policy': frozen.POLICY, 'prompts': len(rows), 'draws_per_prompt': DRAW_COUNT,
            'distinct_request_blocks': len(rows) * DRAW_COUNT,
            'distinct_child_seeds': len(rows) * DRAW_COUNT * frozen.SAMPLES,
            'seed_schedule_sha256': sha(schedule)}


def _batch(llm: Any, rows: list[dict], prompts: list[str], interface: dict,
           schedule: dict, run_sha: str, batch_dir: Path, start: int, end: int,
           draw_index: int, domain: str, grader: Callable, params_factory: Callable) -> list[dict]:
    label = schedule['draw_labels'][draw_index]
    path = batch_dir / f'seed-{label}__rows-{start:06d}-{end:06d}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        require(saved.get('identity_sha256') == run_sha and saved.get('seed') == label
                and saved.get('start') == start and saved.get('end') == end
                and len(saved.get('draws', [])) == end - start
                and saved.get('draws_sha256') == sha(saved['draws']), f'invalid resumed batch: {path}')
        draws = saved['draws']
    else:
        params = []
        for row_index in range(start, end):
            seed = schedule['request_seeds'][row_index][draw_index]
            value = params_factory(SimpleNamespace(**interface, seed=seed), domain, rows[row_index])
            require(getattr(value, 'seed', None) == seed and getattr(value, 'n', None) == frozen.SAMPLES,
                    'sampling parameters must retain the scheduled seed and n=8')
            params.append(value)
        generated = llm.generate(prompts[start:end], params, use_tqdm=False)
        require(len(generated) == end - start, 'vLLM output count mismatch')
        draws = []
        for row_index, output in enumerate(generated, start):
            require(getattr(output, 'prompt', prompts[row_index]) == prompts[row_index],
                    'vLLM returned a mismatched prompt')
            draws.append({**frozen.draw_seed_metadata(schedule, row_index, draw_index),
                          **frozen.grade_samples(rows[row_index], output, grader)})
        frozen.atomic_new(path, {'identity_sha256': run_sha, 'seed': label, 'start': start, 'end': end,
                                 'draws': draws, 'draws_sha256': sha(draws)})
    for row_index, draw in enumerate(draws, start):
        frozen.validate_draw_seed_metadata(draw, schedule, row_index, draw_index)
        _validate_draw_metrics(draw)
    return draws


def evaluate_task(llm: Any, tokenizer: Any, task: dict[str, Any], *,
                  model: dict[str, Any], code: dict[str, str], runtime: dict[str, Any],
                  confirm_eval: bool = False, resume: bool = False,
                  grader: Callable | None = None,
                  params_factory: Callable = frozen.sampling_params) -> dict[str, Any]:
    """Evaluate using frozen helpers, with scale provenance from the first batch."""
    validate_task(task, confirm_eval)
    require(code == code_identity(), 'evaluation requires current evaluator/helper code hashes')
    require(model.get('label') in MODEL_LABELS and model.get('vllm_version') == ENGINE_CONTRACT['vllm_version'],
            'registered scale model label and vLLM 0.8.4 provenance required')
    domain = task['domain']
    validate_runtime_settings(runtime, domain)
    require(getattr(llm, '_modebench_runtime', None) == runtime,
            'loaded model runtime differs from declared scale runtime')
    path = Path(task['output'])
    if path.exists() and not resume:
        raise FileExistsError(f'fresh final receipt required: {path}')
    rows, source = load_rows(task)
    interface = frozen_interface(domain)
    schedule = frozen.schedule_record(domain, rows, task['seeds'])
    prompts = [tokenizer.apply_chat_template(
        frozen.prompt_messages(domain, row['problem'], interface['prompt_profile']),
        tokenize=False, add_generation_prompt=True) for row in rows]
    for index, prompt in enumerate(prompts):
        require(len(tokenizer.encode(prompt, add_special_tokens=False)) + interface['max_tokens']
                <= interface['max_model_len'], f'row {index} exceeds frozen context budget')
    identity = {'schema': SCHEMA, 'domain': domain, 'level': task['level'], 'split': task.get('split', 'dev'),
                'model': model, 'runtime': runtime, 'interface': interface, 'interface_sha256': sha(interface),
                'source': source, 'seeds': task['seeds'], 'batch_size': task.get('batch_size', 8),
                'code_sha256': code, 'rendered_prompts_sha256': sha(prompts),
                'seed_schedule': schedule, 'seed_schedule_sha256': sha(schedule),
                'sampling_engine': dict(ENGINE_CONTRACT)}
    run_sha = sha(identity)
    if path.exists():
        receipt = json.loads(path.read_text())
        require(receipt.get('identity_sha256') == run_sha and receipt.get('identity') == identity,
                'completed receipt identity mismatch; use a fresh output path')
        validate_seed_receipt(receipt, rows)
        return receipt
    batch_dir = Path(str(path) + '.batches')
    manifest = batch_dir / 'run.json'
    if manifest.exists():
        if not resume:
            raise FileExistsError(f'partial run exists; use --resume: {manifest}')
        require(json.loads(manifest.read_text()) == {'identity_sha256': run_sha, 'identity': identity},
                'resume identity mismatch; use a fresh output path')
    else:
        frozen.atomic_new(manifest, {'identity_sha256': run_sha, 'identity': identity})
    if grader is None:
        from oat_drgrpo.math_grader import validated_modebench_outcome_key
        grader = validated_modebench_outcome_key
    draws_by_row: list[list[dict]] = [[] for _ in rows]
    for draw_index, label in enumerate(task['seeds']):
        for start in range(0, len(rows), identity['batch_size']):
            end = min(start + identity['batch_size'], len(rows))
            draws = _batch(llm, rows, prompts, interface, schedule, run_sha, batch_dir,
                           start, end, draw_index, domain, grader, params_factory)
            for row_index, draw in enumerate(draws, start):
                draws_by_row[row_index].append(draw)
            print(json.dumps({'event': 'batch_complete', 'domain': domain, 'level': task['level'],
                              'seed': label, 'rows_done': end, 'rows_total': len(rows)}), flush=True)
    results = []
    for index, (row, draws) in enumerate(zip(rows, draws_by_row)):
        spec = json.loads(row['answer']) if isinstance(row['answer'], str) else row['answer']
        results.append({'row_index': source['row_offset'] + index, 'row_sha256': sha(row),
                        'problem_sha256': sha(row['problem']), 'spec_sha256': sha(spec),
                        'row_metadata': {key: value for key, value in row.items() if key not in ('problem', 'answer')},
                        'draws': draws, **{metric: statistics.mean(draw[metric] for draw in draws)
                                          for metric in ('pass1', 'pass8', 'distinct8')}})
    receipt = {'schema': SCHEMA, 'generated_at': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
               'identity_sha256': run_sha, 'identity': identity, 'domain': domain, 'level': task['level'],
               'split': identity['split'], 'model_label': model['label'],
               'sampling': {**interface, 'seeds': task['seeds']},
               'metrics': frozen.summarize(results), 'prompt_results': results,
               'answer_mode_histogram': dict(Counter(str(row.get('answer_mode_count')) for row in rows)),
               'metric_definitions': {'pass1': 'mean verified fraction among each n=8 draw',
                                      'pass8': 'mean indicator of at least one verified sample in each n=8 draw',
                                      'distinct8': 'mean count of unique verified canonical keys in each n=8 draw'},
               'information_boundary': {'evaluation_prompts_loaded': identity['split'] == 'eval',
                                        'confirmation_explicitly_authorized': confirm_eval,
                                        'treatment_training_started': False}}
    validate_seed_receipt(receipt, rows)
    frozen.atomic_new(path, receipt)
    print(json.dumps({'event': 'task_complete', 'output': str(path), 'metrics': receipt['metrics']}), flush=True)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--model-label', required=True, choices=MODEL_LABELS)
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--level', default='level4', choices=LEVELS)
    parser.add_argument('--split', default='dev', choices=('dev', 'eval'))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--rows-jsonl', type=Path)
    source.add_argument('--dataset', type=Path)
    source.add_argument('--tasks-json', type=Path)
    parser.add_argument('--interface', default=INTERFACE, choices=(INTERFACE,))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seeds', type=int, nargs=DRAW_COUNT)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--row-limit', type=int, default=0)
    parser.add_argument('--row-offset', type=int, default=0)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.82)
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--swap-space', type=float, default=16.0)
    parser.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--confirm-eval', action='store_true')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    defaults = {'interface': args.interface, 'level': args.level, 'split': args.split,
                'batch_size': args.batch_size, 'row_limit': args.row_limit, 'row_offset': args.row_offset,
                'seeds': args.seeds}
    if args.tasks_json:
        raw = json.loads(args.tasks_json.read_text())
        require(isinstance(raw, list) and raw and all(isinstance(task, dict) for task in raw),
                '--tasks-json must contain a nonempty list of task objects')
        tasks = [{**defaults, **task} for task in raw]
    else:
        require(args.domain and args.output, '--domain and --output are required without --tasks-json')
        tasks = [{**defaults, 'domain': args.domain, 'output': str(args.output),
                  'rows_jsonl': str(args.rows_jsonl) if args.rows_jsonl else None,
                  'dataset': str(args.dataset) if args.dataset else None}]
    seen_outputs, seen_blocks = set(), set()
    for task in tasks:
        validate_task(task, args.confirm_eval)
        output = Path(task['output']).resolve()
        require(output not in seen_outputs, 'tasks must have distinct output paths')
        seen_outputs.add(output)
        if output.exists() and not args.resume:
            raise FileExistsError(f'fresh final receipt required: {output}')
        rows, _ = load_rows(task)
        schedule = frozen.schedule_record(task['domain'], rows, task['seeds'])
        blocks = {seed for row in schedule['request_seeds'] for seed in row}
        require(not seen_blocks & blocks, 'duplicate or colliding RNG blocks across task manifest')
        seen_blocks |= blocks
    required_context = max(frozen_interface(task['domain'])['max_model_len'] for task in tasks)
    runtime = runtime_settings(max_model_len=args.max_model_len if args.max_model_len is not None else required_context,
                               tensor_parallel_size=args.tensor_parallel_size,
                               gpu_memory_utilization=args.gpu_memory_utilization,
                               swap_space=args.swap_space, enable_prefix_caching=args.enable_prefix_caching)
    for task in tasks:
        validate_runtime_settings(runtime, task['domain'])
    version = frozen.validate_runtime_contract()
    model = {**model_identity(args.model, args.model_label), 'vllm_version': version}
    code = code_identity()
    import vllm
    llm = vllm.LLM(model=str(args.model.resolve()), **runtime)
    llm._modebench_runtime = dict(runtime)
    tokenizer = llm.get_tokenizer()
    for task in tasks:
        evaluate_task(llm, tokenizer, task, model=model, code=code, runtime=runtime,
                      confirm_eval=args.confirm_eval, resume=args.resume)


if __name__ == '__main__':
    main()
