#!/usr/bin/env python3
"""Independently recompute saved scoring-probe gradient metrics on CPU.

Uses blockwise NumPy float64 dot products with math.fsum over blocks rather
than the probe's Torch reduction path. Does not execute generated programs or
load a language model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dot(left,right,difference=False):
    values=[]
    for start in range(0,len(left),65536):
        a=left[start:start+65536].astype(np.float64)
        b=right[start:start+65536].astype(np.float64)
        if difference:a=a-b;b=a
        values.append(float(np.dot(a,b)))
    return math.fsum(values)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();run=a.run.resolve();directory=run/'diagnostic'
    result=json.loads((directory/'result.json').read_text());launch=json.loads((run/'identity.json').read_text())
    assert result['status']=='pass' and result['comparisons']['reduction_dtype']=='float64'
    assert sha(run/'config.json')==launch['config_sha256']==result['config_sha256']
    records={Path(row['snapshot']).resolve():row['sha256'] for row in launch['files']}
    for relative,expected in [('ops/diagnose_real_domains_scoring_shapes_20260921_v2.py',result['runner_sha256']),('ops/train_real_domains_pilot_20260921.py',result['pilot_runner_sha256'])]:
        path=run/'bundle'/relative;assert records[path]==sha(path)==expected
    assert sha(result['config']['rows_path'])==result['rows_sha256']==result['config']['rows_sha256']
    assert result['all_trainable_parameter_hashes_unchanged'] and all(v['trainable_parameter_hash_unchanged'] for v in result['variants'].values())
    assert sha(directory/'gradient_vectors.pt')==result['comparisons']['gradient_vectors_sha256']
    vectors=torch.load(directory/'gradient_vectors.pt',map_location='cpu',weights_only=True)
    assert set(vectors)=={'full_eval','trimmed_eval','trimmed_train'}
    arrays={}
    for name,value in vectors.items():
        assert value.device.type=='cpu' and value.dtype==torch.float32 and value.ndim==1 and len(value)==result['trainable_parameters']
        array=value.numpy();assert np.isfinite(array).all()
        assert hashlib.sha256(array.tobytes()).hexdigest()==result['comparisons']['gradient_vector_sha256'][name]
        arrays[name]=array
    norms={name:math.sqrt(dot(v,v)) for name,v in arrays.items()}
    full,trim,repeat=(arrays[k] for k in ('full_eval','trimmed_eval','trimmed_train'))
    values={'full_vs_trimmed_gradient_cosine':dot(full,trim)/(norms['full_eval']*norms['trimmed_eval']),
            'full_vs_trimmed_gradient_relative_l2':math.sqrt(dot(full,trim,True))/norms['trimmed_eval'],
            'trimmed_eval_vs_trimmed_train_gradient_relative_l2':math.sqrt(dot(repeat,trim,True))/norms['trimmed_eval']}
    assert -1<=values['full_vs_trimmed_gradient_cosine']<=1
    for key,value in values.items():assert math.isclose(value,result['comparisons'][key],rel_tol=1e-11,abs_tol=1e-13),(key,value,result['comparisons'][key])
    for key,value in norms.items():assert math.isclose(value,result['comparisons']['gradient_norms_float64'][key],rel_tol=1e-11,abs_tol=1e-13)
    tokens=torch.load(directory/'token_scores.pt',map_location='cpu',weights_only=True);mask=tokens['response_masks'].bool()
    token_audit={}
    for name,behavior in tokens['behavior_logps'].items():
        delta=(tokens['live_logps'][name]-behavior)[mask]
        expected=result['variants'][name]
        assert len(delta)==expected['response_tokens'] and int(torch.count_nonzero(delta))==expected['logp_nonzero_tokens']
        assert float(delta.min())==expected['logp_delta_min'] and float(delta.max())==expected['logp_delta_max']
        assert float(delta.abs().mean())==expected['logp_delta_mean_absolute']
        token_audit[name]={'nonzero_tokens':int(torch.count_nonzero(delta)),'response_tokens':len(delta),'max_absolute_logp_difference':float(delta.abs().max())}
    assert torch.equal(tokens['live_logps']['full_eval'],tokens['live_logps']['trimmed_eval'])
    assert torch.equal(tokens['live_logps']['full_eval'],tokens['live_logps']['trimmed_train'])
    assert token_audit['trimmed_eval']['nonzero_tokens']==token_audit['trimmed_train']['nonzero_tokens']==0
    output={'schema':'real-domains-independent-scoring-vector-audit-20260921-v1','status':'pass','method':'Blockwise NumPy float64 dot products and math.fsum; independent of probe Torch reductions.',
        'inputs_sha256':{str(directory/f):sha(directory/f) for f in ('result.json','gradient_vectors.pt','token_scores.pt')},
        'source_sha256':sha(__file__),'gradient_coordinates':len(full),'gradient_norms':norms,'gradient_comparisons':values,'token_logp_check':token_audit,
        'live_logps_identical_between_all_backward_runs':True,'unchanged_trainable_parameter_hash_receipt_verified':True,
        'limitations':['One fixed mixed-reward coding group at one saved checkpoint.','Residual gradient variation is observed despite identical live token scores; its hardware/kernel cause is not identified.','Difference between full and trimmed gradients includes that residual variation; it is not an isolated causal estimate.','No inference about Re:Max endpoint underperformance follows from this diagnostic.','V1 gradient comparisons remain invalid.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
