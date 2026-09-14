#!/usr/bin/env python3
"""Prepare 64 fresh native hosted draws per selected prompt, without model calls."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/modebench_discovery_curves_20260911'
PARENT = ROOT / 'artifacts/modebench_prompt_ablation_20260911'
CONDITION = 'sampling_budget_ablation_v1'
MODELS = ('gpt56sol','gpt54','grok43')
ARMS = ('original','neutral')

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sha_object(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def identity(row):
    return row['level'],row['domain'],row['row_index']

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def write_rows(path,rows):
    Path(path).write_text(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in rows))

def authenticate(directory, manifest):
    for name,digest in manifest['artifact_sha256'].items():
        if sha_file(directory/name)!=digest:raise ValueError('Changed input: '+name)
    for name,digest in manifest.get('code_sha256',{}).items():
        if sha_file(directory/'code'/name)!=digest:raise ValueError('Changed frozen code: '+name)

def expand(source_items, rows, prompts, arm):
    first={identity(x):x for x in source_items if x['sample_index']==0}
    prompt_lookup={identity(x):x for x in prompts if x['arm']==arm}
    items=[]
    for row in rows:
        ref=first[identity(row)]
        field='input' if 'input' in ref['request'] else 'messages'
        assert ref['row_sha256']==sha_object(row)
        assert ref['request_sha256']==sha_object(ref['request'])
        assert ref['request'][field]==prompt_lookup[identity(row)]['messages']
        for draw in range(64):
            item=dict(ref)
            sid=f"L{row['level']}_{row['domain']}_{row['row_index']:03d}_{draw}"
            item.update(sample_id=sid,sample_index=draw,group_id=sid,choice_index=0,
                        experiment_condition=CONDITION,prompt_arm=arm,
                        reference_sample_id=ref['sample_id'],fresh_response_cohort=True)
            items.append(item)
    items.sort(key=lambda x:hashlib.sha256(('discovery64-order-v1:'+x['sample_id']).encode()).hexdigest())
    assert len(items)==6144 and len({x['sample_id'] for x in items})==6144
    assert set(Counter(identity(x) for x in items).values())=={64}
    return items

def prepare(base=BASE,parent=PARENT):
    base,parent=Path(base).resolve(),Path(parent).resolve()
    manifest=json.loads((base/'manifest.json').read_text());authenticate(base,manifest)
    rows=read_rows(base/'rows.jsonl');prompts=read_rows(base/'prompts.jsonl')
    assert len(rows)==96 and len(prompts)==192
    registry=[]
    for slug in MODELS:
        for arm in ARMS:
            source=parent/'hosted'/slug/arm
            original=json.loads((source/'manifest.json').read_text());authenticate(source,original)
            out=base/'hosted'/slug/arm
            if (out/'manifest.json').exists():
                current=json.loads((out/'manifest.json').read_text());authenticate(out,current)
                assert current['experiment_condition']==CONDITION
                assert current['ablation_manifest_sha256']==sha_file(base/'manifest.json')
                assert current['request_count']==6144 and current['sample_count']==64
            else:
                if out.exists() and any(out.iterdir()):raise ValueError('Refusing unfinished nonempty output: '+str(out))
                out.mkdir(parents=True,exist_ok=True)
                items=expand(read_rows(source/'requests.jsonl'),rows,prompts,arm)
                groups=[{'group_id':x['group_id'],'request':x['request'],'request_sha256':x['request_sha256'],
                         'sample_ids':[x['sample_id']],'sample_count':1,'protocol':original['protocol']} for x in items]
                shutil.copytree(source/'code',out/'code')
                for name in ('datasets.json','model_profile.json'):shutil.copyfile(source/name,out/name)
                write_rows(out/'rows.jsonl',rows);write_rows(out/'requests.jsonl',items);write_rows(out/'http_requests.jsonl',groups)
                current={**original,'prepared_at_utc':datetime.now(timezone.utc).isoformat(),
                         'experiment_condition':CONDITION,'sample_count':64,'prompt_count':96,
                         'request_count':6144,'http_request_count':6144,'ablation_manifest_sha256':sha_file(base/'manifest.json'),
                         'parent_prompt_ablation_run':str(source),'parent_prompt_ablation_manifest_sha256':sha_file(source/'manifest.json'),
                         'preparer_sha256':sha_file(Path(__file__)),
                         'notes':['64 fresh responses per selected prompt; all model failures count.',
                                  'No historical samples enter this ablation. Native per-model generation settings are unchanged.',
                                  'Original and neutral collectors follow the same outcome-independent sample order concurrently.',
                                  'Transport retries and native usage remain separately recorded. No provider RNG independence claim.']}
                current['artifact_sha256']={name:sha_file(out/name) for name in ('datasets.json','model_profile.json','rows.jsonl','requests.jsonl','http_requests.jsonl')}
                write(out/'manifest.json',current);authenticate(out,current)
            registry.append({'model_id':slug,'model':current['model'],'family':'frontier','arm':arm,
                             'run_dir':str(out),'manifest_sha256':sha_file(out/'manifest.json')})
    result={'experiment_condition':CONDITION,'ablation_manifest_sha256':sha_file(base/'manifest.json'),
            'sample_count':64,'planned_response_count':36864,'runs':registry}
    target=base/'hosted_analysis_runs.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:write(target,result)
    print(json.dumps({'prepared_runs':6,'responses_planned':36864,'model_calls':0,'registry':str(target)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--parent',type=Path,default=PARENT);args=parser.parse_args();prepare(args.base,args.parent)
