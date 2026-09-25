#!/usr/bin/env python3
"""Read-only checkpoint likelihood audit on fixed, observed development tokens.

This is a mechanistic diagnostic, not a new task evaluation. It scores exactly
the same bank exemplars under every checkpoint, including the untouched base.
Both full-vocabulary and tokenizer-masked log probabilities are retained so
vLLM prompt_logprobs can be compared without conflating policy normalization.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def validate(config):
    ids = [r['row_id'] for r in config['rows']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('nonempty unique diagnostic row IDs required')
    for row in config['rows']:
        if not row['prompt_token_ids'] or not row['response_token_ids']:
            raise ValueError('empty token sequence')
        if row.get('split') not in {'train', 'dev', 'validation'}:
            raise ValueError('only development/training data may be diagnosed')
    names = [c['name'] for c in config['checkpoints']]
    if len(names) != len(set(names)) or names.count('base') != 1:
        raise ValueError('unique checkpoint names and one base required')
    for checkpoint in config['checkpoints']:
        if not checkpoint.get('adapter'):
            if checkpoint['name'] != 'base':
                raise ValueError('only base can lack an adapter')
            continue
        adapter = Path(checkpoint['adapter'])
        seal = json.loads((adapter.parent / 'complete.json').read_text())
        for relative, expected in seal['adapter_files'].items():
            if digest(adapter / relative) != expected:
                raise ValueError('checkpoint adapter seal mismatch')
        if digest(adapter.parent / 'bank.json') != seal['bank_sha256']:
            raise ValueError('checkpoint bank seal mismatch')


def run(config_path, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    started = time.monotonic()
    config = json.loads(config_path.read_text())
    validate(config)
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('one allocated GPU required')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config['model'], local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': 0})
    model.config.use_cache = False
    upper = min(len(tokenizer), model.config.vocab_size)
    adapters = [c for c in config['checkpoints'] if c.get('adapter')]
    model = PeftModel.from_pretrained(model, adapters[0]['adapter'], adapter_name=adapters[0]['name'])
    for checkpoint in adapters[1:]:
        model.load_adapter(checkpoint['adapter'], adapter_name=checkpoint['name'])
    model.eval()
    results = []

    def score(checkpoint, precision):
        base = checkpoint['name'] == 'base'
        if not base:
            model.set_adapter(checkpoint['name'])
        context = model.disable_adapter() if base else contextlib.nullcontext()
        with context, torch.inference_mode():
            for row in config['rows']:
                p, r = row['prompt_token_ids'], row['response_token_ids']
                ids = torch.tensor([p + r], device='cuda')
                logits = model(ids, attention_mask=torch.ones_like(ids)).logits[0, len(p)-1:-1]
                raw, masked, inaccessible_mass = [], [], []
                # Chunk float32 normalization to keep 7B scoring comfortably
                # below the 48GB allocation even for long code responses.
                for start in range(0, len(r), 32):
                    block = logits[start:start+32].float()
                    target = ids[0, len(p)+start:len(p)+start+len(block)]
                    selected = block.gather(-1, target[:, None]).squeeze(-1)
                    raw_z = torch.logsumexp(block, dim=-1)
                    masked_z = torch.logsumexp(block[:, :upper], dim=-1)
                    raw.extend((selected - raw_z).cpu().tolist())
                    masked.extend((selected - masked_z).cpu().tolist())
                    inaccessible_mass.extend((-torch.expm1(masked_z-raw_z)).cpu().tolist())
                results.append({
                    'checkpoint': checkpoint['name'], 'adapter_precision': precision,
                    'row_id': row['row_id'], 'task_id': row['task_id'],
                    'token_logprobs': masked, 'raw_token_logprobs': raw,
                    'sum_logprob': sum(masked), 'mean_logprob': sum(masked)/len(masked),
                    'raw_sum_logprob': sum(raw), 'raw_mean_logprob': sum(raw)/len(raw),
                    'max_inaccessible_probability': max(inaccessible_mass),
                })
                del logits, ids
        print(f"scored {checkpoint['name']} {precision}: {len(config['rows'])} rows", flush=True)

    native_dtypes = {}
    for checkpoint in config['checkpoints']:
        name = checkpoint['name']
        native_dtypes[name] = sorted({str(p.dtype) for n, p in model.named_parameters() if f'.{name}.' in n})
        score(checkpoint, 'native')
    if config.get('score_bf16_adapters', True):
        # vLLM's endpoint default stores LoRA matrices in the BF16 model dtype.
        # A second HF pass quantifies how much of cross-engine error this explains.
        for n, p in model.named_parameters():
            if 'lora_' in n:
                p.data = p.data.to(torch.bfloat16)
        for checkpoint in adapters:
            score(checkpoint, 'bfloat16')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'schema': 'real-domains-likelihood-audit-20260921-v1', 'status': 'complete',
        'config_sha256': digest(config_path), 'runner_sha256': digest(__file__),
        'model': config['model'], 'native_adapter_dtypes': native_dtypes,
        'tokenizer_vocab_upper': upper, 'model_vocab_size': model.config.vocab_size,
        'scores': results, 'seconds': time.monotonic()-started,
    }, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
