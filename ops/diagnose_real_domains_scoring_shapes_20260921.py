#!/usr/bin/env python3
"""One-GPU, no-parameter-update scoring-shape diagnostic for the real-domain pilot.

Config: model, model_revision, seed, rows_path (verified JSONL), source_step,
max_new_tokens; optional lora_path and original_config_path. Exactly 16 rows from
one mixed-reward prompt are selected. No new text is generated and no optimizer
parameter update is applied (SGD lr=0). Current and trimmed behavior scores feed
the unchanged production loss to isolate numerical scoring differences.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import train_real_domains_pilot_20260921 as pilot


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):pilot.write_json(path,value)
def parameter_hash(model):
    return pilot.identity({name:hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for name,p in model.named_parameters() if p.requires_grad})


class CapturingStrategy(pilot.TorchAccumulationStrategy):
    def __init__(self,accumulation,max_norm):
        super().__init__(accumulation,max_norm);self.gradient=None
    def optimizer_step(self,optimizer,model,scheduler):
        if (self.micro_calls+1)%self.grad_acc_step==0:
            self.gradient=torch.cat([p.grad.detach().float().cpu().reshape(-1) for p in model.parameters() if p.requires_grad])
        super().optimizer_step(optimizer,model,scheduler)


def width_for(attention):
    occupied=attention.sum(0)>0
    return int(occupied.nonzero()[-1])+1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    assert not args.output.exists();args.output.mkdir(parents=True)
    output=args.output
    def progress(stage):write(output/'progress.json',{'stage':stage})
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported()
    assert Path(config['model']).name==config['model_revision'] and len(config['model_revision'])==40
    assert sha(config['rows_path'])==config['rows_sha256'], 'verified row source changed'
    rows=[json.loads(line) for line in Path(config['rows_path']).read_text().splitlines()]
    rows=[r for r in rows if r['phase']=='train' and r['step']==config['source_step']]
    rows=sorted(rows,key=lambda r:r['sample_index'])
    assert len(rows)==16 and [r['sample_index'] for r in rows]==list(range(16))
    prompt=rows[0]['prompt_token_ids'];assert all(r['prompt_token_ids']==prompt for r in rows)
    assert 0<sum(r['verdict']['accepted'] for r in rows)<16
    assert all(not r['verdict']['hard_violations'] for r in rows)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,PeftModel,get_peft_model
    pilot.seed_all(config['seed']);progress('model_loading');started=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0})
    if config.get('lora_path'):
        model=PeftModel.from_pretrained(model,config['lora_path'],is_trainable=True)
    else:
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,target_modules=pilot.DEFAULTS['lora_target_modules'],lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    model.config.use_cache=False;model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for child in model.modules():
        if isinstance(child,torch.nn.Dropout):child.p=0.
    learner_config={**pilot.DEFAULTS,'seed':config['seed'],'group_size':16,'train_microbatch_size':1,'max_new_tokens':config['max_new_tokens']}
    learner=pilot.build_learner(model,tokenizer,learner_config,'maxrl',torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=0.))
    learner.strategy=CapturingStrategy(16,1.)
    ids,attention,masks=pilot.tensor_batch(prompt,[r['token_ids'] for r in rows],tokenizer.pad_token_id,'cuda')
    upper=learner._resolve_scoring_vocab_upper_bound(model)
    initial_hash=parameter_hash(model)
    identity={'schema':'real-domains-scoring-shapes-20260921-v1','config':config,'config_sha256':sha(args.config),'rows_sha256':sha(config['rows_path']),
        'model_config_sha256':sha(Path(config['model'])/'config.json'),'runner_sha256':sha(__file__),'pilot_runner_sha256':sha(pilot.__file__),
        'initial_trainable_parameters_sha256':initial_hash,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'fresh_successes':sum(r['verdict']['accepted'] for r in rows),'group_width':ids.shape[1],
        'response_lengths':[len(r['token_ids']) for r in rows],'generation_performed':False,'optimizer':'SGD lr=0','arm':'fresh MaxRL only; no replay'}
    write(output/'identity.json',identity)
    behavior={}
    for mode in ('full_eval','trimmed_eval','trimmed_train'):
        progress('behavior_'+mode);model.train(mode.endswith('train'));scores=torch.zeros_like(masks)
        with torch.no_grad():
            for i in range(16):
                width=ids.shape[1] if mode=='full_eval' else width_for(attention[i:i+1])
                logits=model(ids[i:i+1,:width],attention_mask=attention[i:i+1,:width])['logits']
                logits=learner._mask_invalid_scoring_logit_columns(logits,valid_vocab_size=upper,context='shape_diagnostic_behavior')
                selected,_=learner._policy_logps_and_optional_entropy(logits,ids[i:i+1,:width],masks[i:i+1,:width-1],need_entropy=False)
                scores[i,:width-1]=selected[0]
                del logits,selected
        behavior[mode]=scores.detach()
        write(output/'progress.json',{'stage':'behavior_complete','mode':mode})
    rewards=torch.tensor([float(r['verdict']['accepted']) for r in rows],device='cuda')
    advantages=pilot.binary_maxrl_advantages(rewards[None])[0]
    summaries={};gradients={};live_by_mode={}
    score_method=learner._policy_logps_and_optional_entropy
    for mode in behavior:
        progress('production_'+mode);model.train();learner.optimizer.zero_grad(set_to_none=True);np.random.seed(39123)
        recorded=[]
        def capture(logits,labels,response_masks,*,need_entropy):
            selected,entropy=score_method(logits,labels,response_masks,need_entropy=need_entropy)
            if need_entropy:
                recorded.append({'tokens':labels[0].detach().cpu().tolist(),'mask':response_masks[0].detach().cpu(),'logps':selected[0].detach().cpu()})
            return selected,entropy
        learner._policy_logps_and_optional_entropy=capture
        info=learner._baseline_update_with_precomputed_advantages(input_ids=ids,att_mask=attention,prompt_id_lens=[len(prompt)]*16,
            loss_masks=torch.ones(16,device='cuda'),response_masks=masks,logps=behavior[mode],ref_logps=None,
            advantages=advantages[:,None],final_rewards=rewards[:,None],policy_vocab_upper_bound=upper,canonical_replay_groups=[])
        learner._policy_logps_and_optional_entropy=score_method
        assert len(recorded)==16 and parameter_hash(model)==initial_hash
        # Reconstruct exact randomized row order instead of matching duplicate responses.
        order=np.random.RandomState(39123).permutation(16);live=torch.zeros_like(masks,device='cpu')
        for i,record in zip(order,recorded):
            width=width_for(attention[i:i+1]);assert record['tokens']==ids[i,:width].cpu().tolist()
            live[i,:width-1]=record['logps']
        delta=(live-behavior[mode].cpu())[masks.cpu().bool()]
        relative=(delta.exp()-1).abs();gradients[mode]=learner.strategy.gradient.clone();live_by_mode[mode]=live
        summaries[mode]={'logp_delta_min':float(delta.min()),'logp_delta_max':float(delta.max()),'logp_delta_mean_absolute':float(delta.abs().mean()),
            'logp_nonzero_tokens':int(torch.count_nonzero(delta)),'response_tokens':len(delta),'ratio_outside_20pct_tokens':int((relative>.2).sum()),
            'production_pg_clipfrac':float(info['pg_clipfrac']),'gradient_norm_before_clipping':learner.strategy.last_grad_norm,
            'trainable_parameter_hash_unchanged':True}
        write(output/'progress.json',{'stage':'production_complete','mode':mode,**summaries[mode]})
    reference=gradients['trimmed_eval'];full=gradients['full_eval'];denom=float(reference.norm())
    comparisons={'full_vs_trimmed_gradient_relative_l2':float((full-reference).norm())/max(denom,1e-30),
        'full_vs_trimmed_gradient_cosine':float(torch.nn.functional.cosine_similarity(full,reference,dim=0)),
        'trimmed_eval_vs_trimmed_train_gradient_relative_l2':float((gradients['trimmed_train']-reference).norm())/max(denom,1e-30),
        'live_score_max_repeat_difference':max(float((v-live_by_mode['full_eval']).abs().max()) for v in live_by_mode.values())}
    assert parameter_hash(model)==initial_hash
    output_result={**identity,'status':'pass','comparisons':comparisons,'variants':summaries,'elapsed_seconds':time.monotonic()-started,
        'shape_correction_eliminates_logp_drift':summaries['trimmed_eval']['logp_nonzero_tokens']==0,
        'all_trainable_parameter_hashes_unchanged':True,'optimizer_steps_at_zero_learning_rate':3}
    torch.save({'behavior_logps':{k:v.cpu() for k,v in behavior.items()},'live_logps':live_by_mode,'response_masks':masks.cpu()},output/'token_scores.pt')
    write(output/'result.json',output_result);print(json.dumps(output_result,indent=2))


if __name__=='__main__':main()
