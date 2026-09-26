#!/usr/bin/env python3
"""Fixed 21-question, 64-draw native-HF 3B capability screen; no training."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import train_real_domains_pilot_20260921_v2 as trainer
import evaluate_real_domains_native_hf_20260922 as native
import evaluate_real_domains_20260921 as metrics

SCHEMA = 'code3b-native-capability-screen-20260922-v1'
TRAIN_IDS = ['1454_A','1569_A','361_A','1408_A','988_A','1323_A','1380_A','1038_B']
DEV_IDS = ['1513_A','1352_B','1016_D','1360_G','244_A','1102_B','1095_C','1352_G','482_A','1339_B','1371_D','1051_B','545_B']
REVISION = '488639f1ff808d1d3d0ba301aef8c11461451ec5'


def validate_config(config):
    if config['schema'] != SCHEMA or config['task_ids'] != TRAIN_IDS + DEV_IDS:
        raise ValueError('fixed screen schema/cohort differs')
    if config['samples_per_task'] != 64 or config['eval_seed'] != 119411:
        raise ValueError('fixed screen sample count/seed differs')
    training = trainer.resolve_config(config['training_config'])
    expected = dict(model_revision=REVISION, seed=88411, generation_batch_size=4,
                    max_new_tokens=1024, max_context_tokens=8192, attention_implementation='sdpa',
                    train_ids=TRAIN_IDS, eval_ids=DEV_IDS)
    if any(training[k] != v for k,v in expected.items()):
        raise ValueError('native sampling/model contract differs')
    if training['adapter_config']['problem_ids'] != config['task_ids']:
        raise ValueError('adapter must use exactly the fixed 21 questions')
    if training['adapter_config'].get('allow_heldout') is not False:
        raise ValueError('reserved test access forbidden')
    if config['max_seconds'] != 6900:
        raise ValueError('fixed runtime cap differs')
    return training


def verify_bindings(config):
    manifest_path = Path(config['bindings_path'])
    if trainer.digest(manifest_path) != config['bindings_sha256']:
        raise ValueError('binding manifest drift')
    manifest = json.loads(manifest_path.read_text())
    for row in manifest['files']:
        path = Path(row['path'])
        if path.stat().st_size != row['size_bytes'] or trainer.digest(path) != row['sha256']:
            raise ValueError(f'frozen input drift: {path}')
    return {'manifest_sha256': config['bindings_sha256'], 'files_verified': len(manifest['files'])}


def make_model(training):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    trainer.seed_all(training['seed'])
    model = AutoModelForCausalLM.from_pretrained(training['model'], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':0})
    model = get_peft_model(model, LoraConfig(r=training['lora_rank'], lora_alpha=training['lora_alpha'],
            target_modules=training['lora_target_modules'], lora_dropout=0.0, bias='none', task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for child in model.modules():
        if isinstance(child, torch.nn.Dropout):
            child.p = 0.0
    # Fresh LoRA-B must be zero so the screened initial policy equals the base policy.
    if any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if 'lora_B' in n):
        raise ValueError('fresh LoRA-B is nonzero')
    base_dtypes = sorted({str(p.dtype) for p in model.parameters() if not p.requires_grad})
    adapter_dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
    if base_dtypes != ['torch.bfloat16'] or adapter_dtypes != ['torch.float32']:
        raise ValueError('model precision differs from native training contract')
    model.eval()
    return model, {'initial_trainable_parameters_sha256':native.trainable_hash(model),
            'base_dtypes':base_dtypes,'adapter_dtypes':adapter_dtypes,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'initial_lora_b_all_zero':True}


def timeout_handler(signum, frame):
    raise TimeoutError('fixed screen wall time exceeded')


def write_report(output, result):
    lines = ['# Qwen2.5-Coder-3B capability screen', '',
        'Same 8 training and 13 existing development questions; 64 draws each. '
        'This is an untrained capability screen, not a Re:Max versus MaxRL result.', '',
        '| Cohort | Task | Accepted / 64 | Distinct valid behaviors | pass@8 | ED@32 |',
        '|---|---|---:|---:|---:|---:|']
    for row in result['task_results']:
        lines.append(f"| {row['cohort']} | {row['task_id']} | {row['accepted']} | {row['distinct_valid_modes']} | {row['pass_at_k']['8']:.3f} | {row['expected_distinct_valid_modes_at_k']['32']:.3f} |")
    lines += ['', 'All questions and draws retained. PCMD is reported in result.json only where at least 30 draws were accepted. '
              'Behavior keys represent outputs on the fixed verification suite, not distinct algorithms. '
              'Candidate Python module randomness and hash seed are controlled; accepted programs must pass the independent stability recheck.',
              '', 'No optimizer steps, replay bank, reserved-test access, or automatic training launch. '
              'Before paired training, inspect task-level learning signal and validate actual training memory and numerical correctness.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def run(config_path, output, preflight=False):
    started = time.monotonic()
    config = json.loads(config_path.read_text())
    training = validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    identity = {'schema':SCHEMA, 'config':config, 'config_sha256':trainer.digest(config_path),
                'host':socket.gethostname(), 'job_id':os.environ.get('SLURM_JOB_ID'),
                'account':os.environ.get('SLURM_JOB_ACCOUNT'), 'partition':os.environ.get('SLURM_JOB_PARTITION')}
    rows = []
    try:
        if not preflight:
            if socket.gethostname().split('.')[0] != 'node105' or identity['account'] != 'mltheory':
                raise RuntimeError('screen must run on physical node105 under mltheory account')
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(config['max_seconds'])
        identity['bindings'] = verify_bindings(config)
        module = importlib.import_module(training['adapter_module'])
        task_list = module.load_tasks(training['adapter_config'])
        tasks = {task.task_id:task for task in task_list}
        if len(task_list) != 21 or set(tasks) != set(config['task_ids']):
            raise ValueError('task loader differs from fixed cohort')
        identity['dataset'] = module.dataset_identity(training['adapter_config'])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(training['model'], local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        prompts = {task_id:tokenizer.encode(tasks[task_id].prompt, add_special_tokens=False) for task_id in config['task_ids']}
        if any(not tokens or len(tokens)+1024>8192 for tokens in prompts.values()):
            raise ValueError('prompt truncation forbidden')
        identity['tasks'] = [{'task_id':task_id,'prompt_sha256':hashlib.sha256(tasks[task_id].prompt.encode()).hexdigest(),
                              'prompt_token_ids':prompts[task_id]} for task_id in config['task_ids']]
        if preflight:
            result = {'status':'cpu_preflight_pass_not_gpu_test','identity':identity,'seconds':time.monotonic()-started}
            trainer.write_json(output/'result.json', result)
            return result
        import torch, transformers, peft
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('one allocated BF16-capable GPU required')
        model, model_receipt = make_model(training)
        identity.update(model=model_receipt,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                        transformers_version=transformers.__version__,peft_version=peft.__version__)
        eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
        eos_ids = sorted(eos if isinstance(eos,(tuple,list)) else [eos])
        sampler = {'eos_token_ids':eos_ids}
        enrich_config = {'global_task_order':config['task_ids'],'cohort_ids':{'train':TRAIN_IDS,'development':DEV_IDS},
                         'arm':'base','checkpoint_step':0,'eval_seed':config['eval_seed']}
        identity['sampler'] = {'temperature':1.0,'top_p':1.0,'top_k':0,'batch_size':4,'max_new_tokens':1024,
                              'seed_rule':native.SEED_RULE,'eos_token_ids':eos_ids,'prompt_truncation':False}
        trainer.write_json(output/'identity.json',identity)
        with (output/'responses.jsonl').open('x') as raw, (output/'attempts.jsonl').open('x') as attempts:
            writer = native.ReceiptWriter(raw,enrich_config,sampler,started+config['max_seconds'])
            for index,task_id in enumerate(config['task_ids']):
                samples,timing = trainer.generate_samples(model,tokenizer,tasks[task_id],prompts[task_id],64,
                        training,'code3b_screen',0,writer,seed_base=config['eval_seed']+index*10000)
                enriched = [{**native.enrich_row(row,enrich_config,sampler),**row['verdict']} for row in samples]
                trainer.append_rows(attempts,enriched)
                summary = metrics.mode_metrics([row['verdict'] for row in samples])
                rows.append({'task_id':task_id,'cohort':'train' if task_id in TRAIN_IDS else 'development',
                             'family':tasks[task_id].family,**summary,'timing':timing})
                trainer.write_json(output/'progress.json',{'status':'running','completed_samples':len(rows)*64,
                                   'task_results':rows,'seconds':time.monotonic()-started})
                print(f"{task_id}: {summary['accepted']}/64 accepted, {summary['distinct_valid_modes']} modes",flush=True)
        if native.trainable_hash(model)!=model_receipt['initial_trainable_parameters_sha256']:
            raise RuntimeError('capability screen changed adapter weights')
        result = {'schema':SCHEMA,'status':'complete','identity':identity,'task_results':rows,
                  'summary':native.summarize_rows(rows),
                  'cohort_summaries':{c:native.summarize_rows([r for r in rows if r['cohort']==c]) for c in ('train','development')},
                  'optimizer_steps':0,'bank_mutations':0,'seconds':time.monotonic()-started,
                  'artifacts':{name:trainer.digest(output/name) for name in ('responses.jsonl','attempts.jsonl')},
                  'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
                  'peak_reserved_gpu_bytes':torch.cuda.max_memory_reserved()}
        if result['summary']['samples']!=1344 or result['summary']['hard_violation_count']!=0:
            raise RuntimeError('incomplete or invalid screen')
        write_report(output,result)
        trainer.write_json(output/'result.json',result)
        trainer.write_json(output/'progress.json',{'status':'complete','completed_samples':1344})
        signal.alarm(0)
        return result
    except BaseException as exc:
        trainer.write_json(output/'failure.json',{'status':'failed','identity':identity,'task_results':rows,
                           'error':f'{type(exc).__name__}: {exc}','seconds':time.monotonic()-started})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    result=run(a.config,a.output_dir,a.preflight)
    print(json.dumps({'status':result['status'],'output_dir':str(a.output_dir)}),flush=True)
