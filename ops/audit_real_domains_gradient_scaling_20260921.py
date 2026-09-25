#!/usr/bin/env python3
"""CPU-only literal-gradient audit of the frozen real-domain learner.

No model/data/GPU jobs are loaded. A tiny autoregressive table policy exercises
production loss, accumulation, clipping and optimizer calls against an
independently written full-batch expression from the manuscript.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import train_real_domains_pilot_20260921 as pilot
from oat_drgrpo.online_canonical_bank import VerifiedCanonicalReplayGroup


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator=torch.Generator().manual_seed(919)
        self.weight=torch.nn.Parameter(torch.randn(7,7,generator=generator,dtype=torch.float64)*.4)
        self.config=SimpleNamespace(vocab_size=7)
    def forward(self,input_ids,attention_mask):
        return {'logits':self.weight[input_ids]}


class CaptureStrategy(pilot.TorchAccumulationStrategy):
    def __init__(self,accumulation,max_norm):
        super().__init__(accumulation,max_norm)
        self.preclip=[]
        self.backward_calls=0
    def backward(self,loss,model,optimizer):
        self.backward_calls+=1
        super().backward(loss,model,optimizer)
    def optimizer_step(self,optimizer,model,scheduler):
        if (self.micro_calls+1)%self.grad_acc_step==0:
            self.preclip.append(torch.cat([p.grad.detach().reshape(-1).clone() for p in model.parameters()]))
        super().optimizer_step(optimizer,model,scheduler)


def logps(model,prompt,response):
    # Intentionally does not call the production tokenizer/mask/loss helpers.
    sequence=torch.tensor([list(prompt)+list(response)])
    logits=model(sequence,torch.ones_like(sequence))['logits'][0,:-1].float()
    values=torch.log_softmax(logits,dim=-1)
    return values.gather(1,sequence[0,1:,None]).squeeze(1)[len(prompt)-1:]


def samples(successes,group_size=16):
    rows=[]
    for i in range(group_size):
        # Unequal lengths, including EOS=6, exercise token/sequence reductions.
        tokens=[(i+2*j)%6 for j in range(1+i%5)]+[6]
        rows.append({'request_id':str(i),'token_ids':tokens,'text':str(tokens),
                     'verdict':{'accepted':i<successes,'canonical_key':str(i) if i<successes else None,
                                'hard_violations':[],'receipt':{}}})
    return rows


def replay_group(modes):
    if not modes:return []
    return [VerifiedCanonicalReplayGroup((3,2),tuple(f'mode-{i}' for i in range(modes)),
        tuple(tuple([(2*i+j)%6 for j in range(i%7)]+[6]) for i in range(modes)),
        tuple(10000 if i==0 else 1 for i in range(modes)))]


def literal_loss(model,rows,groups,arm,tmax,offset=0.):
    count=len(rows);successes=sum(r['verdict']['accepted'] for r in rows)
    fresh=[]
    for i,row in enumerate(rows):
        lp=logps(model,(0,1),row['token_ids'])
        # Match the mathematical binary coefficients represented in float32.
        advantage=torch.tensor(float(count*int(row['verdict']['accepted'])/successes-1) if successes else 0.,dtype=torch.float32)
        old=lp.detach()+offset*((i%3)-1)
        ratio=torch.exp(lp-old)
        fresh.append(torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2)).sum()/tmax)
    loss=torch.stack(fresh).mean()
    if arm=='remax' and groups:
        bank_loss=torch.stack([-torch.stack([logps(model,g.prompt_token_ids,r).mean() for r in g.response_token_ids]).mean() for g in groups]).mean()
        loss=loss+.1*(count-1)/(count*count)*bank_loss
    return loss


def run_case(*,micro=1,modes=3,successes=3,tmax=1024,arm='remax',max_norm=1000.,optimizer='sgd',steps=1,offset=0.):
    model=TinyPolicy();reference=deepcopy(model)
    config={**pilot.DEFAULTS,'group_size':16,'train_microbatch_size':micro,'max_new_tokens':tmax,'max_grad_norm':max_norm}
    make=lambda m: torch.optim.SGD(m.parameters(),lr=.03) if optimizer=='sgd' else torch.optim.AdamW(m.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    actual_optimizer=make(model);expected_optimizer=make(reference)
    learner=pilot.build_learner(model,SimpleNamespace(pad_token_id=6,eos_token_id=6,vocab_size=7),config,arm,actual_optimizer)
    learner.strategy=CaptureStrategy(16//micro,max_norm)
    rows=samples(successes);groups=replay_group(modes)
    records=[]
    for step in range(steps):
        expected_optimizer.zero_grad(set_to_none=True)
        loss=literal_loss(reference,rows,groups,arm,tmax,offset)
        loss.backward()
        expected_gradient=reference.weight.grad.detach().clone().reshape(-1)
        expected_norm=torch.nn.utils.clip_grad_norm_(reference.parameters(),max_norm,error_if_nonfinite=True)
        expected_optimizer.step()
        np.random.seed(739+step)
        if offset==0:
            info=pilot.policy_update(learner,(0,1),rows,groups)
        else:
            ids,attention,masks=pilot.tensor_batch((0,1),[r['token_ids'] for r in rows],6,'cpu')
            old=torch.zeros_like(masks)
            with torch.no_grad():
                for i,row in enumerate(rows):
                    old[i,1:1+len(row['token_ids'])]=logps(model,(0,1),row['token_ids'])+offset*((i%3)-1)
            advantages=torch.tensor([16*float(r['verdict']['accepted'])/successes-1 if successes else 0 for r in rows])
            info=learner._baseline_update_with_precomputed_advantages(input_ids=ids,att_mask=attention,
                prompt_id_lens=[2]*16,loss_masks=torch.ones(16),response_masks=masks,logps=old,
                ref_logps=None,advantages=advantages[:,None],final_rewards=torch.tensor([float(r['verdict']['accepted']) for r in rows])[:,None],
                policy_vocab_upper_bound=7,canonical_replay_groups=groups)
        actual_gradient=learner.strategy.preclip[-1]
        torch.testing.assert_close(actual_gradient,expected_gradient,atol=2e-8,rtol=3e-5)
        torch.testing.assert_close(model.weight,reference.weight,atol=2e-9,rtol=2e-7)
        assert learner.strategy.updates==step+1
        records.append({'step':step,'preclip_gradient_max_absolute_error':float((actual_gradient-expected_gradient).abs().max()),
                        'preclip_gradient_norm':float(expected_gradient.norm()),'gradient_clipping_active':float(expected_norm)>max_norm,
                        'parameter_max_absolute_error':float((model.weight-reference.weight).abs().max()),
                        'production_pg_clipfrac':float(info.get('pg_clipfrac',0.))})
    if optimizer=='adamw':
        left=actual_optimizer.state[model.weight];right=expected_optimizer.state[reference.weight]
        for name in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(left[name],right[name],atol=2e-9,rtol=3e-5)
    assert learner.strategy.micro_calls==steps*16//micro
    return {'config':{'micro':micro,'modes':modes,'successes':successes,'tmax':tmax,'arm':arm,'max_norm':max_norm,'optimizer':optimizer,'steps':steps,'old_logp_offset':offset},
            'steps':records,'optimizer_updates':learner.strategy.updates,'backward_calls':learner.strategy.backward_calls,
            'final_gradient':learner.strategy.preclip[-1].tolist()}


def cases():
    for micro in (1,2,4,8,16):
        for arm in ('maxrl','remax'):
            yield {'micro':micro,'arm':arm,'modes':16}
    for successes in (0,16):
        for modes in (0,1,3):
            for arm in ('maxrl','remax'):
                yield {'successes':successes,'modes':modes,'arm':arm}
    for tmax in (16,1024):
        for arm in ('maxrl','remax'):
            yield {'tmax':tmax,'arm':arm,'modes':3,'max_norm':.003,'optimizer':'adamw','steps':3}
    for micro in (1,4,16):
        yield {'micro':micro,'offset':.5,'modes':3}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();assert not args.output.exists()
    results=[run_case(**case) for case in cases()]
    loss_scaling={}
    for tmax in (16,1024):
        baseline=run_case(tmax=tmax,arm='maxrl');treatment=run_case(tmax=tmax,arm='remax')
        fresh=torch.tensor(baseline['final_gradient']);replay=torch.tensor(treatment['final_gradient'])-fresh
        loss_scaling[str(tmax)]={'fresh_gradient_norm':float(fresh.norm()),'replay_only_gradient_norm':float(replay.norm()),'replay_to_fresh_norm_ratio':float(replay.norm()/fresh.norm())}
    for result in results:result.pop('final_gradient')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    output={'schema':'real-domains-literal-gradient-audit-20260921-v1','status':'pass','cpu_only':True,'cases':results,'case_count':len(results),
            'effective_replay_coefficient':.1*15/256,'length_scaling_diagnostic':loss_scaling,
            'sources':{p:sha(p) for p in ['ops/train_real_domains_pilot_20260921.py','src/oat_drgrpo/learner/grpo.py','src/oat_drgrpo/canonical_replay.py',__file__]}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'pass','case_count':len(results),'output':str(args.output),'max_gradient_error':max(s['preclip_gradient_max_absolute_error'] for r in results for s in r['steps']),'length_scaling':loss_scaling},indent=2))


if __name__=='__main__':main()
