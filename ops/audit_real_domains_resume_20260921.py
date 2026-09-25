#!/usr/bin/env python3
"""Compare a sealed resumed trajectory against its uninterrupted counterpart."""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from prepare_real_domains_resume_20260921 import canonical_hash, digest, verify_prepared


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def checkpoint_seal(path):
    seal = json.loads((path / 'complete.json').read_text())
    expected = {'bank.json': seal['bank_sha256'], 'training.pt': seal['training_state_sha256'],
                **{'adapter/' + name: sha for name, sha in seal['adapter_files'].items()}}
    for name, sha in expected.items():
        if digest(path / name) != sha:
            raise ValueError(f'checkpoint artifact drift: {path / name}')
    return seal


def compare_tree(left, right):
    result = {'equal': True, 'tensor_count': 0, 'tensor_elements': 0,
              'different_tensors': 0, 'maximum_absolute_tensor_difference': 0.0,
              'non_tensor_differences': [], 'different_tensor_paths': []}
    def compare(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            result['tensor_count'] += 1
            result['tensor_elements'] += a.numel()
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                result['equal'] = False
                result['different_tensors'] += 1
                result['different_tensor_paths'].append(path)
                if a.shape == b.shape and a.numel():
                    value = (a.double() - b.double()).abs().max().item()
                    result['maximum_absolute_tensor_difference'] = max(result['maximum_absolute_tensor_difference'], value)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
                result['equal'] = False
                result['non_tensor_differences'].append(path)
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                result['equal'] = False
                result['non_tensor_differences'].append(path + ':keys')
            for key in a.keys() & b.keys():
                compare(a[key], b[key], f'{path}/{key}')
        elif isinstance(a, (tuple, list)) and isinstance(b, type(a)):
            if len(a) != len(b):
                result['equal'] = False
                result['non_tensor_differences'].append(path + ':length')
            for i, (aa, bb) in enumerate(zip(a, b)):
                compare(aa, bb, f'{path}/{i}')
        elif type(a) is not type(b) or a != b:
            result['equal'] = False
            result['non_tensor_differences'].append(path)
    compare(left, right, '')
    return result


def run(resume_root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    provenance = verify_prepared(resume_root / 'identity.json')
    original = Path(provenance['training_identity_path']).parent
    resumed = resume_root / 'training'
    original_result = json.loads((original / 'result.json').read_text())
    resumed_result = json.loads((resumed / 'result.json').read_text())
    target, start = provenance['original_target_updates'], provenance['completed_updates_before_resume']
    for result in (original_result, resumed_result):
        if result['status'] != 'complete' or result['completed_updates'] != target or result['config_sha256'] != provenance['resolved_config_sha256']:
            raise ValueError('both trajectories must complete the same sealed target configuration')
    a, b = original / f'checkpoint-{target}', resumed / f'checkpoint-{target}'
    original_seal, resumed_seal = checkpoint_seal(a), checkpoint_seal(b)
    for seal in (original_seal, resumed_seal):
        if seal['config_sha256'] != provenance['resolved_config_sha256'] or seal['arm'] != provenance['arm'] or seal['completed_updates'] != target:
            raise ValueError('endpoint checkpoint metadata mismatch')
    saved_adapter = load_file(str(Path(provenance['checkpoint']) / 'adapter/adapter_model.safetensors'))
    named_hashes = {name.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight'): hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for name, tensor in saved_adapter.items()}
    saved_initial_hash = canonical_hash(named_hashes)
    resumed_identity = json.loads((resumed / 'identity.json').read_text())
    initial_weights_restored = saved_initial_hash == resumed_identity['initial_trainable_parameters_sha256']
    if not initial_weights_restored:
        raise ValueError('resumed initial adapter does not match sealed checkpoint weights')
    old_raw = [r for r in rows(original / 'candidates.jsonl') if r['phase'] == 'train' and start <= r['step'] < target]
    new_raw = [r for r in rows(resumed / 'candidates.jsonl') if r['phase'] == 'train']
    config = json.loads((original / 'identity.json').read_text())['config']
    expected = {(step, index) for step in range(start, target) for index in range(config['group_size'])}
    def index(samples):
        identities = [(r['step'], r['sample_index']) for r in samples]
        if len(identities) != len(set(identities)) or set(identities) != expected:
            raise ValueError('continuation fresh-rollout denominator mismatch')
        return dict(zip(identities, samples))
    old_by_id, new_by_id = index(old_raw), index(new_raw)
    raw_differences = [list(key) for key in sorted(expected) if old_by_id[key] != new_by_id[key]]
    bank_equal = json.loads((a / 'bank.json').read_text()) == json.loads((b / 'bank.json').read_text())
    adapter_comparison = compare_tree(load_file(str(a / 'adapter/adapter_model.safetensors')), load_file(str(b / 'adapter/adapter_model.safetensors')))
    original_state = torch.load(a / 'training.pt', map_location='cpu', weights_only=False)
    resumed_state = torch.load(b / 'training.pt', map_location='cpu', weights_only=False)
    state_comparison = compare_tree(original_state, resumed_state)
    old_eval = {r['request_id']: r for r in rows(original / 'candidates.jsonl') if r['phase'] == 'final'}
    new_eval = {r['request_id']: r for r in rows(resumed / 'candidates.jsonl') if r['phase'] == 'final'}
    final_evaluation_equal = old_eval == new_eval
    exact = not raw_differences and bank_equal and adapter_comparison['equal'] and state_comparison['equal'] and final_evaluation_equal
    result = {'schema': 'real-domains-resume-equivalence-20260921-v1',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'status': 'exact_match' if exact else 'completed_with_differences',
              'resume_provenance': provenance,
              'initial_adapter_exactly_restored': initial_weights_restored,
              'initial_adapter_named_tensor_sha256': saved_initial_hash,
              'fresh_continuation_samples': len(expected), 'fresh_candidate_records_equal': not raw_differences,
              'different_fresh_requests': raw_differences, 'final_bank_equal': bank_equal,
              'adapter_tensors': adapter_comparison, 'optimizer_rng_and_training_state': state_comparison,
              'original_strategy_updates': original_state['strategy_updates'],
              'resumed_strategy_updates': resumed_state['strategy_updates'],
              'original_strategy_micro_calls': original_state['strategy_micro_calls'],
              'resumed_strategy_micro_calls': resumed_state['strategy_micro_calls'],
              'final_evaluation_samples': len(old_eval), 'final_evaluation_raw_records_equal': final_evaluation_equal,
              'original_final_seal_sha256': digest(a / 'complete.json'), 'resumed_final_seal_sha256': digest(b / 'complete.json'),
              'original_result_sha256': digest(original / 'result.json'), 'resumed_result_sha256': digest(resumed / 'result.json'),
              'audit_source_sha256': digest(Path(__file__))}
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.resume_root, args.output), indent=2, sort_keys=True))
