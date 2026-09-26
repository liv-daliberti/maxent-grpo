#!/usr/bin/env python3
"""Separately confirm corrected saved grades once in a fresh warmed verifier."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_curves_20260911'
LOCAL = BASE / 'local'
OUT = LOCAL / 'measurement_integrity'
WORKER = r'''
import json,sys,time
sys.dont_write_bytecode=True
sys.path.insert(0,sys.argv[1])
from oat_drgrpo.math_grader import validated_modebench_outcome_key
from oat_drgrpo.python_modebench_process import _SHARED_VERIFIER
_SHARED_VERIFIER._start()
time.sleep(1.25)
warm_spec={'verifier':'python_factor_function','python_version':'factor-v1','cases':[6,8]}
warm_key=validated_modebench_outcome_key(r'\boxed{lambda n: 2}',warm_spec)
assert warm_key == 'python_factor:2,2', ('synthetic_warmup_failed',warm_key)
print(json.dumps({'synthetic_warmup_canonical_key':warm_key}),flush=True)
for line in sys.stdin:
    item=json.loads(line)
    key=validated_modebench_outcome_key(item['text'],item['answer'])
    print(json.dumps({'verified':key is not None,'canonical_key':key}),flush=True)
'''

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def objsha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def domain(value):
    return 'pantry' if value=='pantry_plan' else value

def key(value,draw_field):
    return value['arm'],value['level'],domain(value['domain']),value['row_index'],value[draw_field]

def main():
    OUT.mkdir(exist_ok=True)
    plan=json.loads((LOCAL/'plan.json').read_text())
    for path,digest in {**plan['input_sha256'],**plan['code_sha256']}.items():
        assert sha(path)==digest
    rows={(r['level'],domain(r['domain']),r['row_index']):r for r in lines(BASE/'rows.jsonl')}
    total=0
    for checkpoint in plan['checkpoints']:
        directory=LOCAL/'results'/checkpoint['label']
        audit_path=directory/'discovery_grading_audit.json'
        output=OUT/(checkpoint['label']+'.json')
        if not audit_path.exists():
            continue
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='complete'
        if output.exists():
            saved=json.loads(output.read_text());assert saved['grading_audit_sha256']==sha(audit_path)
            total+=saved['corrections_confirmed'];continue
        raw_path=directory/'responses.jsonl';sidecar=directory/'discovery_grades.jsonl'
        assert sha(raw_path)==audit['responses_sha256'] and sha(sidecar)==audit['cache_sha256']
        raw={key(r,'draw_index'):r for r in lines(raw_path)}
        changes=[]
        for entry in lines(sidecar):
            k=key(entry,'sample_index');sample=raw[k]
            normalized_raw={**sample,'domain':entry['domain'],'sample_index':sample['draw_index'],
                            'stop_reason':sample.get('finish_reason','unknown'),'graded_text':sample['text']}
            assert objsha(normalized_raw)==entry['raw_sample_sha256']
            strict=entry['strict']
            if (sample['verified'],sample['canonical_key']) != (strict['verified'],strict['canonical_key']):
                assert strict['graded_text']==sample['text']
                changes.append((k,sample,strict))
        assert len(changes)==audit['strict_changed_records']
        confirmations=[];warmup=None
        if changes:
            payload=[{'text':sample['text'],'answer':rows[k[1:4]]['answer']} for k,sample,_ in changes]
            result=subprocess.run([sys.executable,'-I','-B','-c',WORKER,str(LOCAL/'code/src')],
                input=''.join(json.dumps(p)+'\n' for p in payload),text=True,capture_output=True,timeout=120,check=True)
            returned=[json.loads(line) for line in result.stdout.splitlines()]
            assert len(returned)==len(changes)+1
            warmup=returned[0]
            for (k,sample,strict),confirmed in zip(changes,returned[1:]):
                assert confirmed=={'verified':strict['verified'],'canonical_key':strict['canonical_key']}
                confirmations.append({'identity':{field:sample[field] for field in ('checkpoint_label','arm','level','domain','row_index','pair_id','draw_index','draw_block','block_draw_index','sampling_seed','child_sampling_seed')},
                    'raw_sample_sha256':objsha(sample),'text':sample['text'],'row':rows[k[1:4]],
                    'raw_grade':{'verified':sample['verified'],'canonical_key':sample['canonical_key']},
                    'offline_strict_grade':{'verified':strict['verified'],'canonical_key':strict['canonical_key']},
                    'fresh_warmed_confirmation':confirmed,'target_verifier_calls':1,
                    'first_collected_request_and_draw':sample['draw_index']==0 and sample['arm']=='original' and k[1:4]==min(rkey for rkey in rows if checkpoint['domain'] is None or rkey[1]==checkpoint['domain'])})
        record={'status':'confirmed','created_at':datetime.now(timezone.utc).isoformat(),'checkpoint':checkpoint['label'],
            'plan_sha256':sha(LOCAL/'plan.json'),'grading_audit_sha256':sha(audit_path),
            'raw_responses_sha256':sha(raw_path),'grading_sidecar_sha256':sha(sidecar),
            'confirmation_source_sha256':sha(__file__),'frozen_math_grader_sha256':sha(LOCAL/'code/src/oat_drgrpo/math_grader.py'),
            'corrections_confirmed':len(changes),'confirmations':confirmations,'synthetic_warmup':warmup,
            'generation_calls':0,'modified_original_grades':False,'modified_verifier_or_timeouts':False,
            'method':'Each corrected target is checked once in a separate fresh Python process after synthetic Python verifier warmup, using the exact frozen validated_modebench_outcome_key. Original worker timeouts are unchanged.',
            'cause_attribution':None,'cause_limit':'Saved generation receipts do not retain the original verifier error reason. Request position is descriptive and does not establish a cause.'}
        with output.open('x') as handle:json.dump(record,handle,indent=2,sort_keys=True);handle.write('\n')
        total+=len(changes)
        print(json.dumps({'event':'cohort_corrections_confirmed','checkpoint':checkpoint['label'],'corrections':len(changes)}),flush=True)
    print(json.dumps({'event':'available_confirmation_status','cohorts':len(list(OUT.glob('qwen*.json'))),'corrections_confirmed':total}),flush=True)

if __name__=='__main__':
    main()
