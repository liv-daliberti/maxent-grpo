#!/usr/bin/env python3
"""Collect 64 fresh draws as eight n8 blocks for frozen paired prompt curves."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'ops', ROOT / 'src'):
    sys.path.insert(0, str(p))
from evaluate_modebench_level3 import atomic_new, file_sha, sha, grade_samples
from evaluate_modebench_level2_viability import sampling_params
from frontier_modebench_contract import make_messages

SCHEMA = 'modebench-discovery-curves-local-v1'
DOMAINS = ('python_factors', 'mathir', 'pantry')
SETTINGS = {'sample_count': 8, 'temperature': 1.0, 'top_p': 1.0,
            'max_tokens': 192, 'max_model_len': 2048, 'dtype': 'float16',
            'syntax_profile': 'domain_legal_v1', 'batch_size': 8,
            'gpu_memory_utilization': 0.65, 'enable_prefix_caching': True,
            'enforce_eager': True, 'seed_base': 911640000, 'seed_stride': 128,
            'total_sample_count': 64, 'draw_blocks': 8,
            'seed_policy': 'disjoint_n64_problem_blocks_in_fresh_namespace_shared_across_arms_and_checkpoints',
            'template': 'model_native_chat_template', 'vllm_use_v1': '0',
            'arm_order': 'alternate_pair_order_by_selected_position_plus_draw_block; both arms in one batch'}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def domain_name(value):
    return 'pantry' if value == 'pantry_plan' else value


def level_number(value):
    return int(str(value).replace('level', '').replace('L', ''))


def row_key(row):
    return level_number(row['level']), domain_name(row['domain']), row['row_index']


def selected_keys(selection):
    return {row_key(row) for row in selection['candidates'] if row['rank_in_cell'] <= 16}


def validate_inputs(rows, prompts, prior_rows, prior_prompts, selection):
    from evaluate_modebench_prompt_ablation_local_v2 import validate_inputs as validate_prior
    prior_lookup, prior_pairs = validate_prior(prior_rows, prior_prompts)
    lookup = {row_key(row): row for row in rows}
    expected_keys = selected_keys(selection)
    if len(rows) != 96 or set(lookup) != expected_keys:
        raise ValueError('expected exact preselected 16-of-32 rows in every cell')
    if Counter((k[0], k[1]) for k in lookup) != Counter({(level, domain): 16 for level in (2, 3) for domain in DOMAINS}):
        raise ValueError('expected six cells of 16 rows')
    grouped = defaultdict(dict)
    for key, row in lookup.items():
        if row != prior_lookup[key]:
            raise ValueError('selected row differs from immutable parent')
    for prompt in prompts:
        key, arm = row_key(prompt), prompt['arm']
        if key not in lookup or arm not in ('original', 'neutral') or arm in grouped[key]:
            raise ValueError('unexpected or duplicate selected prompt')
        if prompt != prior_pairs[key][arm]:
            raise ValueError('selected prompt differs from immutable parent')
        grouped[key][arm] = prompt
    if len(prompts) != 192 or any(set(grouped[k]) != {'original', 'neutral'} for k in lookup):
        raise ValueError('expected exactly both arms for all selected rows')
    return lookup, grouped


def problem_seed(key, draw_block=0):
    level, domain, index = key
    if level not in (2, 3) or domain not in DOMAINS or not 0 <= index < 128 or not 0 <= draw_block < 8:
        raise ValueError('invalid problem coordinate or draw block')
    return SETTINGS['seed_base'] + SETTINGS['seed_stride'] * ((level - 2) * 10000 + DOMAINS.index(domain) * 1000 + index) + 8 * draw_block


def ordered_items(selected, draw_block):
    return [(key, arm) for position, key in enumerate(selected)
            for arm in (('original', 'neutral') if (position + draw_block) % 2 == 0 else ('neutral', 'original'))]


def validate_plan(plan):
    if plan['schema'] != SCHEMA or plan['settings'] != SETTINGS:
        raise ValueError('local panel or sampling settings differ')
    for kind in ('input_sha256', 'code_sha256'):
        for path, digest in plan[kind].items():
            if file_sha(Path(path)) != digest:
                raise ValueError(f'{kind} changed: {path}')
    if len(plan['checkpoints']) != 25 or sum(c['expected_draws'] for c in plan['checkpoints']) != 110592:
        raise ValueError('checkpoint panel differs from expected full panel')
    prior = json.loads(Path(plan['prior_plan_path']).read_text())
    for checkpoint, old in zip(plan['checkpoints'], prior['checkpoints']):
        expected = dict(old, expected_draws=old['expected_draws'] * 4)
        if checkpoint != expected:
            raise ValueError('checkpoint identity or sequence changed')
    return validate_inputs(read_jsonl(plan['rows_path']), read_jsonl(plan['prompts_path']),
                           read_jsonl(prior['rows_path']), read_jsonl(prior['prompts_path']),
                           json.loads(Path(plan['prior_selection_path']).read_text()))


def verify_checkpoint(checkpoint):
    model = Path(checkpoint['model_path'])
    for item in checkpoint['files']:
        p = model / item['name']
        if not p.is_file() or p.stat().st_size != item['bytes'] or file_sha(p) != item['sha256']:
            raise ValueError(f'checkpoint file missing or changed: {p}')
    return model


def completion_records(checkpoint, row, prompt, draw, draw_block):
    if len(draw['attempts']) != 8:
        raise ValueError('generation receipt must contain exactly eight attempts')
    request_seed = problem_seed(row_key(row), draw_block)
    return [{'schema': SCHEMA, 'checkpoint_label': checkpoint['label'],
             'training_method': checkpoint['training_method'],
             'training_seed': checkpoint['training_seed'],
             'trained_on_level': checkpoint['trained_on_level'],
             'domain': domain_name(row['domain']), 'level': level_number(row['level']),
             'pair_id': prompt['pair_id'], 'arm': prompt['arm'],
             'row_index': row['row_index'], 'row_sha256': sha(row),
             'messages_sha256': prompt['messages_sha256'],
             'draw_index': 8 * draw_block + i, 'draw_block': draw_block, 'block_draw_index': i,
             'sampling_seed': request_seed, 'child_sampling_seed': request_seed + i, **attempt}
            for i, attempt in enumerate(draw['attempts'])]


def validate_batch(batch, run_sha, draw_block, start, items, pairs, rows):
    if (batch.get('identity_sha256') != run_sha or batch.get('draw_block') != draw_block or
        batch.get('start') != start or batch.get('records_sha256') != sha(batch['records']) or
        len(batch['records']) != len(items)):
        raise ValueError('completed batch differs from immutable panel')
    for (key, arm), record in zip(items, batch['records']):
        if (record['pair_id'], record['arm'], record['draw_block'], record['sampling_seed'],
            record['level'], record['domain'], record['row_index']) != (
                pairs[key][arm]['pair_id'], arm, draw_block, problem_seed(key, draw_block), *key):
            raise ValueError('resumed pair/block/seed identity differs')
        if len(record['attempts']) != 8 or any(a['token_count'] > 192 for a in record['attempts']):
            raise ValueError('saved attempt count or token limit differs')


def atomic_jsonl(path, records):
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def evaluate(plan_path, task_index):
    plan = json.loads(plan_path.read_text())
    rows, pairs = validate_plan(plan)
    checkpoint = plan['checkpoints'][task_index]
    output = Path(plan['output_root']) / checkpoint['label']
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'schema': SCHEMA, 'plan_sha256': file_sha(plan_path),
                    'checkpoint': checkpoint, 'settings': SETTINGS}
        run_sha = sha(identity)
        final = output / 'result.json'
        if final.exists():
            saved = json.loads(final.read_text())
            if saved.get('identity_sha256') != run_sha or saved.get('status') != 'complete':
                raise ValueError('existing final receipt differs')
            print(json.dumps({'event': 'already_complete', 'checkpoint': checkpoint['label']}), flush=True)
            return
        model = verify_checkpoint(checkpoint)
        selected = sorted(k for k in rows if checkpoint['domain'] is None or k[1] == checkpoint['domain'])
        expected = len(selected) * 128
        if expected != checkpoint['expected_draws']:
            raise ValueError('checkpoint expected draw count differs')
        import vllm
        from oat_drgrpo.math_grader import validated_modebench_outcome_key
        llm = vllm.LLM(model=str(model), dtype=SETTINGS['dtype'], max_model_len=2048,
                       gpu_memory_utilization=SETTINGS['gpu_memory_utilization'],
                       swap_space=4, enable_prefix_caching=True, enforce_eager=True)
        tokenizer = llm.get_tokenizer()
        rendered = {(key, arm): tokenizer.apply_chat_template(pairs[key][arm]['messages'],
                     tokenize=False, add_generation_prompt=True)
                    for key in selected for arm in ('original', 'neutral')}
        for text in rendered.values():
            if len(tokenizer.encode(text, add_special_tokens=False)) + 192 > 2048:
                raise ValueError('rendered prompt exceeds frozen context budget')
        runtime_path = output / 'runtime.json'
        if not runtime_path.exists():
            atomic_new(runtime_path, {'identity_sha256': run_sha,
                       'generated_at': datetime.now(timezone.utc).isoformat(),
                       'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                       'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
                       'hostname': os.uname().nodename, 'model_path': str(model),
                       'versions': {x: importlib.metadata.version(x) for x in ('torch', 'vllm', 'transformers')},
                       'rendered_prompts_sha256': sha({f'{k}:{a}': v for (k,a),v in rendered.items()})})
        all_records, response_records = [], []
        # Every block uses the original n8 request shape; only seed/order change.
        for draw_block in range(8):
            ordered = ordered_items(selected, draw_block)
            for start in range(0, len(ordered), SETTINGS['batch_size']):
                items = ordered[start:start + SETTINGS['batch_size']]
                batch_path = output / f'batch_b{draw_block:02d}_{start:04d}.json'
                if batch_path.exists():
                    batch = json.loads(batch_path.read_text())
                else:
                    params = [sampling_params(SimpleNamespace(**SETTINGS, seed=problem_seed(key, draw_block)), key[1], rows[key])
                              for key, arm in items]
                    results = llm.generate([rendered[item] for item in items], params, use_tqdm=False)
                    if len(results) != len(items):
                        raise ValueError('generation count differs')
                    records = []
                    for (key, arm), generated in zip(items, results):
                        if generated.prompt != rendered[(key, arm)]:
                            raise ValueError('generation prompt order differs')
                        draw = grade_samples(rows[key], generated, validated_modebench_outcome_key)
                        records.append({'pair_id': pairs[key][arm]['pair_id'], 'arm': arm,
                                        'draw_block': draw_block, 'sampling_seed': problem_seed(key, draw_block),
                                        'level': key[0], 'domain': key[1], 'row_index': key[2], **draw})
                    batch = {'identity_sha256': run_sha, 'draw_block': draw_block, 'start': start,
                             'records': records, 'records_sha256': sha(records)}
                    validate_batch(batch, run_sha, draw_block, start, items, pairs, rows)
                    atomic_new(batch_path, batch)
                validate_batch(batch, run_sha, draw_block, start, items, pairs, rows)
                for (key, arm), record in zip(items, batch['records']):
                    all_records.append(record)
                    response_records.extend(completion_records(checkpoint, rows[key], pairs[key][arm], record, draw_block))
                print(json.dumps({'event': 'batch_complete', 'checkpoint': checkpoint['label'],
                                  'draw_block': draw_block, 'requests_done_in_block': start + len(items),
                                  'requests_per_block': len(ordered)}), flush=True)
        if len(response_records) != expected:
            raise ValueError('final draw count differs')
        responses = output / 'responses.jsonl'
        if not responses.exists():
            atomic_jsonl(responses, response_records)
        elif read_jsonl(responses) != response_records:
            raise ValueError('existing responses differ')
        atomic_new(final, {'schema': SCHEMA, 'status': 'complete', 'identity_sha256': run_sha,
                   'identity': identity, 'checkpoint_label': checkpoint['label'], 'draws': expected,
                   'prompt_results_unit': 'n8 generation request; eight blocks per prompt; full64 estimands computed offline',
                   'responses_path': str(responses),
                   'responses_sha256': file_sha(responses), 'prompt_results': all_records,
                   'interpretation': '64 fresh draws per prompt; L3 transfer for L2-trained checkpoints; no old draws pooled.'})
        print(json.dumps({'event': 'complete', 'checkpoint': checkpoint['label'], 'draws': expected}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        for checkpoint in plan['checkpoints']:
            verify_checkpoint(checkpoint)
        print(json.dumps({'status': 'pass', 'checkpoints': len(plan['checkpoints']), 'draws': sum(c['expected_draws'] for c in plan['checkpoints'])}))
    else:
        if args.task_index is None:
            parser.error('--task-index required')
        evaluate(args.plan.resolve(), args.task_index)

if __name__ == '__main__':
    main()
