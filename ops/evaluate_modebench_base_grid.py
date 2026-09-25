#!/usr/bin/env python3
"""Benchmark frozen upstream Qwen models across registered ModeBench levels.

This separate namespace uses the unchanged native-chat guided interface, four
independent groups of eight samples, temperature/top-p 1, and 192 output tokens.
Every task authenticates all 128 evaluation rows against an admitted dataset
registry before sampling. This measures frozen base-model capability; it does
not calibrate difficulty, admit a candidate construction, or train a treatment.
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
SCHEMA = 'modebench-base-grid-independent-v1'
REGISTRY_SCHEMA = 'modebench-base-grid-dataset-registry-v1'
PURPOSE = 'frozen_base_model_benchmarking'
EVAL_ROWS = 128
# The frozen grid also scores the 384-row training split. These checkpoints are
# off-the-shelf and trained on none of it, so those rows are simply more unseen
# prompts, and scoring them raises how many prompts return the two verified
# responses PCMD needs. A trained checkpoint did see them, so a receipt carries
# its split and evaluation_prompts_loaded is false for train: the two can be
# told apart and must never be pooled.
TRAIN_ROWS = 384
SPLIT_ROWS = {'eval': EVAL_ROWS, 'train': TRAIN_ROWS}
INTERFACE = 'modebench_qwen_base_grid_independent_v1'
LEVELS = ('level1', 'level2', 'level3', 'level4', 'level5')
MODEL_LABELS = ('smol135', 'smol360', 'olmo1b', 'qwen15b', 'smol17b', '05b',
                'falcon1b', 'falcon3b', '3b', 'falcon7b', 'falcon10b', 'olmo7b',
                '7b', 'olmo13b', '14b', 'qwen32b', 'qwen72b')
DRAW_COUNT = 4
ENGINE_CONTRACT = dict(frozen.ENGINE_CONTRACT)
sha = frozen.sha
load_rows = frozen.load_rows
require = frozen.require


def frozen_interface(domain: str, profile: str = INTERFACE) -> dict[str, Any]:
    require(profile == INTERFACE, 'base-grid evaluator requires the base-grid interface namespace')
    return {**frozen.frozen_interface(domain), 'name': INTERFACE}


def code_identity() -> dict[str, str]:
    return {**frozen.code_identity(),
            'ops/evaluate_modebench_base_grid.py': frozen.file_sha(Path(__file__))}


# These are the existing upstream snapshots, before any ModeBench treatment.
# Architecture checks prevent assigning another local scale the same label.
# (repository, revision, architecture dimensions, model_type, architectures).
# The four Qwen2.5 entries keep the revisions and dimensions the original
# 60-cell grid was collected against; their receipts authenticate against the
# frozen code snapshot under artifacts/, not this file.
MODEL_SPECS = {
    '05b': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775',
            (896, 24, 14, 2, 4864, 151936), 'qwen2', ['Qwen2ForCausalLM']),
    '3b': ('Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
           (2048, 36, 16, 2, 11008, 151936), 'qwen2', ['Qwen2ForCausalLM']),
    '7b': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28',
           (3584, 28, 28, 4, 18944, 152064), 'qwen2', ['Qwen2ForCausalLM']),
    '14b': ('Qwen/Qwen2.5-14B-Instruct', 'cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8',
            (5120, 48, 40, 8, 13824, 152064), 'qwen2', ['Qwen2ForCausalLM']),
    'qwen15b': ('Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                (1536, 28, 12, 2, 8960, 151936), 'qwen2', ['Qwen2ForCausalLM']),
    'qwen32b': ('Qwen/Qwen2.5-32B-Instruct', '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd',
                (5120, 64, 40, 8, 27648, 152064), 'qwen2', ['Qwen2ForCausalLM']),
    'qwen72b': ('Qwen/Qwen2.5-72B-Instruct', '495f39366efef23836d0cfae4fbe635880d2be31',
                (8192, 80, 64, 8, 29568, 152064), 'qwen2', ['Qwen2ForCausalLM']),
    'smol135': ('HuggingFaceTB/SmolLM2-135M-Instruct', '12fd25f77366fa6b3b4b768ec3050bf629380bac',
                (576, 30, 9, 3, 1536, 49152), 'llama', ['LlamaForCausalLM']),
    'smol360': ('HuggingFaceTB/SmolLM2-360M-Instruct', 'a10cc1512eabd3dde888204e902eca88bddb4951',
                (960, 32, 15, 5, 2560, 49152), 'llama', ['LlamaForCausalLM']),
    'smol17b': ('HuggingFaceTB/SmolLM2-1.7B-Instruct', '31b70e2e869a7173562077fd711b654946d38674',
                (2048, 24, 32, 32, 8192, 49152), 'llama', ['LlamaForCausalLM']),
    'falcon1b': ('tiiuae/Falcon3-1B-Instruct', '28ba2251970a01dd1edc7ba7dad2eb71216ccfdf',
                 (2048, 18, 8, 4, 8192, 131072), 'llama', ['LlamaForCausalLM']),
    'falcon3b': ('tiiuae/Falcon3-3B-Instruct', '411bb94318f94f7a5735b77109f456b1e74b42a1',
                 (3072, 22, 12, 4, 9216, 131072), 'llama', ['LlamaForCausalLM']),
    'falcon7b': ('tiiuae/Falcon3-7B-Instruct', '1e57a0ecd176c7c139f289c60a74e57f887c3dfb',
                 (3072, 28, 12, 4, 23040, 131072), 'llama', ['LlamaForCausalLM']),
    'falcon10b': ('tiiuae/Falcon3-10B-Instruct', '8799bc6aec0152757221dc6b272d824642db6202',
                  (3072, 40, 12, 4, 23040, 131072), 'llama', ['LlamaForCausalLM']),
    'olmo1b': ('allenai/OLMo-2-0425-1B-Instruct', '48d788eca847d4d7548f375ad03d3c9312f6139e',
               (2048, 16, 16, 16, 8192, 100352), 'olmo2', ['Olmo2ForCausalLM']),
    'olmo7b': ('allenai/OLMo-2-1124-7B-Instruct', '470b1fba1ae01581f270116362ee4aa1b97f4c84',
               (4096, 32, 32, 32, 11008, 100352), 'olmo2', ['Olmo2ForCausalLM']),
    'olmo13b': ('allenai/OLMo-2-1124-13B-Instruct', '3a5c85baefbb1896a54d56fe2e76c0395627ddf4',
                (5120, 40, 40, 40, 13824, 100352), 'olmo2', ['Olmo2ForCausalLM']),
}
# Parameter counts for ordering the scale axis; not used for authentication.
MODEL_PARAMS = {'smol135': .135, 'smol360': .36, 'olmo1b': 1.5, 'qwen15b': 1.5,
                'smol17b': 1.7, '05b': .5, 'falcon1b': 1.7, 'falcon3b': 3.2,
                '3b': 3.1, 'falcon7b': 7.5, 'falcon10b': 10.3, 'olmo7b': 7.3,
                '7b': 7.6, 'olmo13b': 13.7, '14b': 14.8, 'qwen32b': 32.8,
                'qwen72b': 72.7}
MODEL_FAMILY = {'05b': 'Qwen2.5', '3b': 'Qwen2.5', '7b': 'Qwen2.5', '14b': 'Qwen2.5',
                'qwen15b': 'Qwen2.5', 'qwen32b': 'Qwen2.5', 'qwen72b': 'Qwen2.5',
                'smol135': 'SmolLM2', 'smol360': 'SmolLM2', 'smol17b': 'SmolLM2',
                'falcon1b': 'Falcon3', 'falcon3b': 'Falcon3', 'falcon7b': 'Falcon3',
                'falcon10b': 'Falcon3',
                'olmo1b': 'OLMo-2', 'olmo7b': 'OLMo-2', 'olmo13b': 'OLMo-2'}
ARCHITECTURE_FIELDS = ('hidden_size', 'num_hidden_layers', 'num_attention_heads',
                       'num_key_value_heads', 'intermediate_size', 'vocab_size')


def model_identity(model: Path, label: str) -> dict[str, Any]:
    require(label in MODEL_SPECS, 'unregistered base-model scale')
    model = model.resolve()
    repository, revision, dimensions, model_type, architectures = MODEL_SPECS[label]
    require(model.name == revision and model.parent.name == 'snapshots'
            and model.parent.parent.name == 'models--' + repository.replace('/', '--'),
            'base models must use the pinned upstream Hugging Face snapshot')
    require(not any(model.glob('*adapter*')), 'tuned adapter checkpoints cannot be base models')
    metadata = frozen.model_identity(model, label)
    config = json.loads((model / 'config.json').read_text())
    architecture = {key: config.get(key) for key in ARCHITECTURE_FIELDS}
    require(config.get('model_type') == model_type
            and config.get('architectures') == architectures
            and tuple(architecture.values()) == dimensions,
            'model architecture does not match its registered scale')
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        require((model / name).is_file(), 'complete native tokenizer required: ' + name)
    tokenizer = json.loads((model / 'tokenizer_config.json').read_text())
    require(bool(tokenizer.get('chat_template')), 'native chat template is missing')
    index_path = model / 'model.safetensors.index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        names = set(index.get('weight_map', {}).values())
        require(names and all(Path(name).name == name and (model / name).is_file() for name in names),
                'model snapshot has missing or invalid weight shards')
        require(names == {item['name'] for item in metadata['weight_file_manifest']},
                'weight files differ from the upstream shard index')
    metadata.update(source_kind='upstream_huggingface_snapshot', repository=repository,
                    revision=revision, model_role='frozen_base_model_before_modebench_training',
                    architecture=architecture)
    return metadata


def validate_model_identity(model: dict[str, Any]) -> None:
    require(isinstance(model, dict) and model.get('label') in MODEL_LABELS
            and model.get('vllm_version') == ENGINE_CONTRACT['vllm_version']
            and isinstance(model.get('path'), str), 'complete frozen upstream model identity required')
    expected = model_identity(Path(model['path']), model['label'])
    require(model == {**expected, 'vllm_version': ENGINE_CONTRACT['vllm_version']},
            'frozen upstream model identity changed')


def dataset_inventory(path: Path) -> dict[str, str]:
    require(path.is_dir(), 'registered original dataset directory is missing')
    inventory = {item.relative_to(path).as_posix(): frozen.file_sha(item)
                 for item in sorted(path.rglob('*')) if item.is_file()}
    require(bool(inventory), 'registered original dataset directory is empty')
    return inventory


def validate_dataset_binding(binding: Any, *, domain: str, level: str,
                             rows_jsonl: str | None) -> list[dict]:
    """Authenticate a complete exported eval set and its original admission.

    The separately frozen registry records which construction is a benchmark
    level. A file hash by itself cannot establish that semantic assignment.
    Candidate or unavailable entries cannot be relabeled as admitted tasks.
    """
    require(isinstance(binding, dict), 'dataset_binding is required')
    require(binding.get('level') == level and binding.get('domain') == domain
            and binding.get('split') in SPLIT_ROWS and binding.get('status') == 'admitted',
            'dataset binding level/domain must identify an admitted split')
    expected = SPLIT_ROWS[binding['split']]
    require(type(binding.get('rows')) is int and binding['rows'] == expected,
            f'registered benchmark dataset must contain exactly {expected} rows')
    for name in ('source_manifest_path', 'dataset_path', 'rows_jsonl'):
        require(isinstance(binding.get(name), str) and Path(binding[name]).is_absolute(),
                'absolute dataset binding path required: ' + name)
    manifest_path = Path(binding['source_manifest_path'])
    require(manifest_path.is_file() and frozen.file_sha(manifest_path) == binding.get('source_manifest_sha256'),
            'dataset registry source hash changed')
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema') == REGISTRY_SCHEMA, 'unsupported dataset registry schema')
    entries = manifest.get('datasets')
    require(isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries),
            'dataset registry must contain dataset objects')
    selected = [entry for entry in entries if entry.get('level') == level and entry.get('domain') == domain]
    expected = {key: value for key, value in binding.items()
                if key not in ('source_manifest_path', 'source_manifest_sha256')}
    require(selected == [expected], 'dataset binding differs from its registered level/domain entry')
    original = Path(binding['dataset_path'])
    require(sha(dataset_inventory(original)) == binding.get('dataset_sha256'),
            'registered original dataset file inventory changed')
    evidence = binding.get('admission_evidence')
    require(isinstance(evidence, list) and bool(evidence), 'dataset admission provenance is required')
    for item in evidence:
        require(isinstance(item, dict) and isinstance(item.get('path'), str)
                and Path(item['path']).is_absolute() and bool(item.get('role')),
                'invalid dataset admission evidence')
        path = Path(item['path'])
        require(path.is_file() and frozen.file_sha(path) == item.get('sha256'),
                'dataset admission evidence changed')
    path = Path(binding['rows_jsonl'])
    require(isinstance(rows_jsonl, str) and Path(rows_jsonl).resolve() == path.resolve(),
            'task row file differs from registered dataset binding')
    require(path.is_file() and frozen.file_sha(path) == binding.get('rows_jsonl_sha256'),
            'registered frozen row file hash changed')
    rows, source = load_rows({'rows_jsonl': str(path), 'row_offset': 0, 'row_limit': 0})
    require(len(rows) == SPLIT_ROWS[binding['split']]
            and source['rows_sha256'] == binding.get('rows_sha256'),
            'registered frozen rows differ from the 128-row dataset identity')
    require(len({sha(row['problem']) for row in rows}) == SPLIT_ROWS[binding['split']],
            'registered evaluation prompts must be distinct')
    return rows


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
        'swap_space', 'enable_prefix_caching'}, 'complete base-grid runtime settings required')
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
    require(isinstance(task, dict), 'task must be an object')
    domain = task.get('domain')
    frozen_interface(domain, task.get('interface', INTERFACE))
    require(task.get('level') in LEVELS, f'level must be one of {LEVELS}')
    require(task.get('split') in SPLIT_ROWS, 'base-grid benchmarking requires a registered split')
    require(confirm_eval is True, 'held-out benchmarking requires --confirm-eval')
    require(isinstance(task.get('rows_jsonl'), str) and bool(task['rows_jsonl'])
            and not task.get('dataset'), 'registered frozen rows_jsonl is required')
    require(type(task.get('row_offset', 0)) is int and task.get('row_offset', 0) == 0
            and type(task.get('row_limit', 0)) is int and task.get('row_limit', 0) == 0,
            'benchmarking must score the full split without row slicing')
    require(type(task.get('batch_size', 8)) is int and task.get('batch_size', 8) >= 1,
            'batch_size must be a positive integer')
    require(isinstance(task.get('output'), str) and bool(task['output']), 'output path is required')
    validate_labels(task.get('seeds'))
    validate_dataset_binding(task.get('dataset_binding'), domain=domain, level=task['level'],
                             rows_jsonl=task['rows_jsonl'])


def _validate_draw_metrics(draw: dict) -> None:
    attempts = draw.get('attempts')
    require(isinstance(attempts, list) and len(attempts) == frozen.SAMPLES,
            'each benchmark draw requires exactly eight saved attempts')
    for attempt in attempts:
        require(isinstance(attempt, dict) and isinstance(attempt.get('text'), str)
                and type(attempt.get('verified')) is bool
                and 'canonical_key' in attempt
                and attempt['verified'] == (attempt['canonical_key'] is not None)
                and type(attempt.get('token_count')) is int and 0 <= attempt['token_count'] <= 192,
                'invalid saved graded attempt')
    verified = sum(attempt['verified'] for attempt in attempts)
    expected = {'verified_count': verified, 'pass1': verified / frozen.SAMPLES,
                'pass8': float(verified > 0),
                'distinct8': len({sha(attempt['canonical_key']) for attempt in attempts if attempt['verified']})}
    require(all(type(draw.get(key)) in (int, float) and draw[key] == value
                for key, value in expected.items()),
            'draw metrics differ from saved verified attempts')


def validate_seed_receipt(receipt: dict[str, Any], rows: list[dict] | None = None) -> dict[str, Any]:
    """Authenticate base-grid identity, runtime, source rows, and independent RNG.

    This intentionally rejects the frozen L3 schemas rather than relabeling
    them. Sampling and grading semantics come from the frozen helper modules.
    """
    require(receipt.get('schema') == SCHEMA and receipt.get('status') == 'complete',
            'complete base-grid receipt required')
    identity = receipt.get('identity', {})
    require(isinstance(identity, dict) and identity.get('schema') == SCHEMA,
            'base-grid identity namespace required')
    require(receipt.get('identity_sha256') == sha(identity), 'receipt identity hash mismatch')
    domain = identity.get('domain')
    require(domain in DOMAINS and receipt.get('domain') == domain, 'receipt domain mismatch')
    require(identity.get('level') in LEVELS and receipt.get('level') == identity['level'],
            'receipt benchmark level mismatch')
    require(identity.get('purpose') == PURPOSE, 'receipt benchmark purpose mismatch')
    require(identity.get('split') in SPLIT_ROWS and receipt.get('split') == identity['split'],
            'receipt split mismatch')
    model = identity.get('model', {})
    require(isinstance(model, dict) and model.get('label') in MODEL_LABELS
            and receipt.get('model_label') == model['label'], 'receipt base-grid model label mismatch')
    validate_model_identity(model)
    require(identity.get('sampling_engine') == ENGINE_CONTRACT
            and model.get('vllm_version') == ENGINE_CONTRACT['vllm_version'],
            'receipt vLLM V0 child-seed contract mismatch')
    validate_runtime_settings(identity.get('runtime'), domain)
    interface = frozen_interface(domain)
    require(identity.get('interface') == interface and identity.get('interface_sha256') == sha(interface),
            'receipt base-grid interface mismatch')
    require(identity.get('code_sha256') == code_identity(), 'receipt evaluator/helper code hashes missing or changed')
    labels = identity.get('seeds')
    validate_labels(labels)
    require(receipt.get('sampling') == {**interface, 'seeds': labels}, 'receipt sampling metadata mismatch')
    source = identity.get('source', {})
    require(isinstance(source, dict), 'missing source identity')
    bound_rows = validate_dataset_binding(identity.get('dataset_binding'), domain=domain,
                                         level=identity['level'], rows_jsonl=source.get('path'))
    boundary = receipt.get('information_boundary', {})
    require(isinstance(boundary, dict)
            and boundary.get('evaluation_prompts_loaded') is (identity['split'] == 'eval')
            and boundary.get('treatment_training_started') is False
            and boundary.get('confirmation_explicitly_authorized') is True
            # Full coverage is required of either split: a cell is the whole
            # split or it is not a cell. Only the held-out marker differs.
            and source.get('row_offset') == 0 and source.get('row_limit') == 0,
            'receipt held-out evaluation boundary mismatch')
    require(source.get('kind') == 'jsonl', 'registered JSONL source identity required')
    reopened_rows, reopened_source = load_rows({
        'rows_jsonl': source['path'], 'row_offset': 0, 'row_limit': 0})
    require(reopened_source == source, 'saved source identity changed')
    if rows is None:
        rows = reopened_rows
    require(rows == bound_rows, 'receipt source differs from its registered dataset binding')
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
    """Evaluate using frozen helpers, with benchmark provenance from the first batch."""
    validate_task(task, confirm_eval)
    require(code == code_identity(), 'evaluation requires current evaluator/helper code hashes')
    validate_model_identity(model)
    require(getattr(llm, '_modebench_model_identity', None) == model,
            'loaded model differs from declared frozen upstream model')
    domain = task['domain']
    validate_runtime_settings(runtime, domain)
    require(getattr(llm, '_modebench_runtime', None) == runtime,
            'loaded model runtime differs from declared base-grid runtime')
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
    identity = {'schema': SCHEMA, 'purpose': PURPOSE, 'domain': domain, 'level': task['level'], 'split': task['split'],
                'dataset_binding': task['dataset_binding'],
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
    parser.add_argument('--level', default='level1', choices=LEVELS)
    parser.add_argument('--split', default='eval', choices=tuple(SPLIT_ROWS))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--rows-jsonl', type=Path)
    parser.add_argument('--dataset-binding', type=Path,
                        help='JSON object binding the single task to its registered frozen dataset')
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
                  'dataset_binding': json.loads(args.dataset_binding.read_text()) if args.dataset_binding else None}]
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
    llm._modebench_model_identity = dict(model)
    tokenizer = llm.get_tokenizer()
    for task in tasks:
        evaluate_task(llm, tokenizer, task, model=model, code=code, runtime=runtime,
                      confirm_eval=args.confirm_eval, resume=args.resume)


if __name__ == '__main__':
    main()
