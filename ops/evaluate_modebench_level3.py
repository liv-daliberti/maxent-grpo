#!/usr/bin/env python3
"""Frozen-interface ModeBench difficulty calibration, with resumable batch receipts.

Examples:
  python ops/evaluate_modebench_level3.py --model PATH --model-label 3b \
    --domain graph_coloring --level level3 --rows-jsonl pool.jsonl \
    --seed 5317100 --output scores.json

A --tasks-json file is a list of objects with domain, level, split, output and
rows_jsonl or dataset fields. It reuses one model across all tasks. Confirmation
splits require --confirm-eval. Completed receipts are immutable; --resume reuses
validated completed batches only. No training or admission decision happens here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'ops', ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from evaluate_modebench_level2_viability import prompt_messages, sampling_params

DOMAINS = ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')
SCHEMA = 'modebench-level3-frozen-calibration-v1'


def sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_new(path: Path, payload: Any) -> None:
    """Publish a complete JSON file atomically, without replacing any receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def frozen_interface(domain: str, profile: str = 'original_level1') -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f'unknown domain: {domain}')
    if profile not in ('original_level1', 'level2_qwen_r5'):
        raise ValueError(f'unknown frozen interface: {profile}')
    original = profile == 'original_level1'
    return {
        'name': profile,
        'prompt_profile': 'boxed_direct_v1' if original or domain == 'graph_coloring' else 'hybrid_solver_v4',
        'syntax_profile': 'none' if original else {'graph_coloring': 'none', 'countdown': 'countdown_legal_v3'}.get(domain, 'domain_legal_v1'),
        'sample_count': 8, 'temperature': 1.0, 'top_p': 1.0,
        'max_tokens': 192, 'max_model_len': 1024 if original or domain == 'graph_coloring' else 2048,
        'dtype': 'float16', 'prompt_template': 'model_native_chat_template',
        'seed_policy': 'same_explicit_seed_for_each_prompt_independent_n8_draw',
    }


def model_identity(model: Path, label: str) -> dict[str, Any]:
    model = model.resolve()
    if not (model / 'config.json').is_file():
        raise ValueError(f'local model config missing: {model}')
    small_names = ('config.json', 'generation_config.json', 'tokenizer.json',
                   'tokenizer_config.json', 'special_tokens_map.json',
                   'model.safetensors.index.json', 'pytorch_model.bin.index.json')
    weights = sorted(set(model.glob('*.safetensors')) | set(model.glob('pytorch_model*.bin')))
    if not weights:
        raise ValueError(f'local model weights missing: {model}')
    return {
        'label': label, 'path': str(model),
        'configuration_sha256': {name: file_sha(model / name) for name in small_names if (model / name).is_file()},
        'weight_file_manifest': [{'name': p.name, 'bytes': p.stat().st_size,
                                  'mtime_ns': p.stat().st_mtime_ns, 'resolved_path': str(p.resolve())} for p in weights],
        'weight_identity_method': 'resolved_snapshot_paths_and_size_mtime_manifest; configuration_files_sha256',
    }


def code_identity() -> dict[str, str]:
    paths = [Path(__file__), ROOT / 'ops/evaluate_modebench_level2_viability.py']
    paths += sorted((ROOT / 'src/oat_drgrpo').glob('*.py'))
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def validate_task(task: dict[str, Any], confirm_eval: bool) -> None:
    frozen_interface(task['domain'], task.get('interface', 'original_level1'))
    if task['domain'] not in DOMAINS:
        raise ValueError(f"unknown domain: {task['domain']}")
    if task['level'] not in ('level1', 'level2', 'level3'):
        raise ValueError('level must be level1, level2, or level3')
    if task.get('split', 'dev') not in ('dev', 'eval'):
        raise ValueError('split must be dev or eval')
    if task.get('split', 'dev') == 'eval' and not confirm_eval:
        raise ValueError('confirmation prompts require --confirm-eval')
    if task.get('split', 'dev') == 'eval' and (task.get('row_limit', 0) or task.get('row_offset', 0)):
        raise ValueError('confirmation must score the full split, without row slicing')
    if task.get('row_limit', 0) < 0 or task.get('row_offset', 0) < 0:
        raise ValueError('row_limit and row_offset must be nonnegative')
    if task.get('rows_jsonl') and task.get('dataset'):
        raise ValueError('choose rows_jsonl or dataset')
    if not (task.get('rows_jsonl') or task.get('dataset')) and task['level'] != 'level1':
        raise ValueError('level2/level3 require rows_jsonl or dataset')
    seeds = task['seeds']
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('seeds must be distinct nonnegative integers')
    if task.get('batch_size', 8) < 1:
        raise ValueError('batch_size must be positive')


def load_rows(task: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if task.get('rows_jsonl'):
        path = Path(task['rows_jsonl']).resolve()
        with path.open() as handle:
            all_rows = [json.loads(line) for line in handle if line.strip()]
        source = {'kind': 'jsonl', 'path': str(path), 'file_sha256': file_sha(path)}
    else:
        from datasets import load_from_disk
        dataset_path = task.get('dataset')
        if not dataset_path:
            sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
            from materialize_modebench_harder_v2 import LEVEL1
            dataset_path = LEVEL1[task['domain']][task.get('split', 'dev')]
        path = Path(dataset_path).resolve()
        dataset = load_from_disk(str(path))
        if hasattr(dataset, 'keys'):
            dataset = dataset['multi_answer']
        all_rows = [dict(row) for row in dataset]
        source = {'kind': 'saved_dataset', 'path': str(path)}
    offset = task.get('row_offset', 0)
    limit = task.get('row_limit', 0)
    rows = all_rows[offset:offset + limit if limit else None]
    if not rows:
        raise ValueError('no input rows')
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('problem'), str) or 'answer' not in row:
            raise ValueError(f'invalid ModeBench row {offset + index}')
        spec = json.loads(row['answer']) if isinstance(row['answer'], str) else row['answer']
        if not isinstance(spec, dict):
            raise ValueError(f'invalid executable spec at row {offset + index}')
    source.update(total_rows=len(all_rows), selected_rows=len(rows), row_offset=offset,
                  row_limit=limit, all_rows_sha256=sha(all_rows), rows_sha256=sha(rows))
    return rows, source


def grade_samples(row: dict, output: Any, grader: Callable) -> dict[str, Any]:
    if len(output.outputs) != 8:
        raise RuntimeError(f'expected 8 samples, received {len(output.outputs)}')
    attempts = []
    for sample in output.outputs:
        text = str(sample.text)
        key = grader(text, row['answer'])
        attempts.append({'text': text, 'verified': key is not None, 'canonical_key': key,
                         'token_count': len(sample.token_ids),
                         'finish_reason': getattr(sample, 'finish_reason', None)})
    verified = sum(attempt['verified'] for attempt in attempts)
    distinct = len({sha(a['canonical_key']) for a in attempts if a['verified']})
    return {'pass1': verified / 8, 'pass8': float(verified > 0), 'distinct8': distinct,
            'verified_count': verified, 'attempts': attempts}


def summarize(prompt_results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {'rows': len(prompt_results)}
    for metric in ('pass1', 'pass8', 'distinct8'):
        values = [row[metric] for row in prompt_results]
        summary[metric] = statistics.mean(values)
        summary[metric + '_standard_error_across_prompts'] = (
            statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None)
    return summary


def evaluate_task(llm: Any, tokenizer: Any, task: dict[str, Any], *,
                  model: dict[str, Any], code: dict[str, str], confirm_eval: bool = False,
                  resume: bool = False, grader: Callable | None = None,
                  params_factory: Callable = sampling_params) -> dict[str, Any]:
    """Evaluate one task using an already-loaded LLM; useful for pool runners."""
    validate_task(task, confirm_eval)
    output_path = Path(task['output'])
    if output_path.exists() and not resume:
        raise FileExistsError(f'fresh final receipt required: {output_path}')
    rows, source = load_rows(task)
    domain = task['domain']
    interface = frozen_interface(domain, task.get('interface', 'original_level1'))
    prompts = [tokenizer.apply_chat_template(prompt_messages(domain, row['problem'], interface['prompt_profile']),
                                             tokenize=False, add_generation_prompt=True) for row in rows]
    model_context = int(getattr(llm, '_modebench_max_model_len', interface['max_model_len']))
    if model_context < interface['max_model_len']:
        raise ValueError('loaded model context is too short for the frozen interface')
    for index, prompt in enumerate(prompts):
        token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        if token_count + interface['max_tokens'] > interface['max_model_len']:
            raise ValueError(f'row {index} exceeds frozen context budget')
    identity = {'schema': SCHEMA, 'domain': domain, 'level': task['level'], 'split': task.get('split', 'dev'),
                'model': model, 'interface': interface, 'interface_sha256': sha(interface),
                'source': source, 'seeds': task['seeds'], 'batch_size': task.get('batch_size', 8),
                'code_sha256': code, 'rendered_prompts_sha256': sha(prompts)}
    run_sha = sha(identity)
    if output_path.exists():
        completed = json.loads(output_path.read_text())
        if completed.get('identity_sha256') != run_sha or completed.get('identity') != identity or completed.get('status') != 'complete':
            raise ValueError('completed receipt identity mismatch; use a fresh output path')
        print(json.dumps({'event': 'task_already_complete', 'output': str(output_path)}), flush=True)
        return completed
    batch_dir = Path(str(output_path) + '.batches')
    manifest_path = batch_dir / 'run.json'
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f'partial run exists; use --resume: {manifest_path}')
        prior = json.loads(manifest_path.read_text())
        if prior.get('identity_sha256') != run_sha or prior.get('identity') != identity:
            raise ValueError('resume identity mismatch; use a fresh output path')
    else:
        atomic_new(manifest_path, {'identity_sha256': run_sha, 'identity': identity})
    if grader is None:
        from oat_drgrpo.math_grader import validated_modebench_outcome_key
        grader = validated_modebench_outcome_key
    draws_by_row: list[list[dict[str, Any]]] = [[] for _ in rows]
    batch_size = task.get('batch_size', 8)
    for seed in task['seeds']:
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_path = batch_dir / f'seed-{seed}__rows-{start:06d}-{end:06d}.json'
            if batch_path.exists():
                batch = json.loads(batch_path.read_text())
                if (batch.get('identity_sha256') != run_sha or batch.get('seed') != seed
                        or batch.get('start') != start or batch.get('end') != end
                        or len(batch.get('draws', [])) != end - start
                        or batch.get('draws_sha256') != sha(batch['draws'])):
                    raise ValueError(f'invalid resumed batch: {batch_path}')
            else:
                params = SimpleNamespace(**interface, seed=seed)
                batch_params = [params_factory(params, domain, row) for row in rows[start:end]]
                generated = llm.generate(prompts[start:end], batch_params, use_tqdm=False)
                if len(generated) != end - start:
                    raise RuntimeError('vLLM output count mismatch')
                draws = []
                for local, result in enumerate(generated):
                    if getattr(result, 'prompt', prompts[start + local]) != prompts[start + local]:
                        raise RuntimeError('vLLM returned a mismatched prompt')
                    draws.append(grade_samples(rows[start + local], result, grader))
                batch = {'identity_sha256': run_sha, 'seed': seed, 'start': start, 'end': end,
                         'draws': draws, 'draws_sha256': sha(draws)}
                atomic_new(batch_path, batch)
            for index, draw in enumerate(batch['draws'], start=start):
                draws_by_row[index].append({'seed': seed, **draw})
            print(json.dumps({'event': 'batch_complete', 'domain': domain, 'level': task['level'],
                              'seed': seed, 'rows_done': end, 'rows_total': len(rows),
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
    parser.add_argument('--interface', default='original_level1', choices=('original_level1', 'level2_qwen_r5'))
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
        if not isinstance(raw, list) or not raw:
            raise ValueError('--tasks-json must contain a nonempty list')
        tasks = [{**defaults, **task} for task in raw]
        for task in tasks:
            if 'seed' in task:
                task['seeds'] = [task.pop('seed')]
    else:
        if not args.domain or not args.output:
            raise ValueError('--domain and --output are required without --tasks-json')
        tasks = [{**defaults, 'domain': args.domain, 'output': str(args.output),
                  'rows_jsonl': str(args.rows_jsonl) if args.rows_jsonl else None,
                  'dataset': str(args.dataset) if args.dataset else None}]
    if len({str(Path(task['output']).resolve()) for task in tasks}) != len(tasks):
        raise ValueError('tasks must have distinct output paths')
    for task in tasks:
        validate_task(task, args.confirm_eval)
        if Path(task['output']).exists() and not args.resume:
            raise FileExistsError(f"fresh final receipt required: {task['output']}")
    identity = model_identity(args.model, args.model_label)
    code = code_identity()
    import vllm
    identity['vllm_version'] = importlib.metadata.version('vllm')
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
