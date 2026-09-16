import hashlib,itertools,json,math,re
from fractions import Fraction as F
from pathlib import Path
ROOT=Path('/n/fs/similarity/maxent-grpo')
AUDIT=ROOT/'paper/audits/submission_repair_20260906/results/e120_falcon_graph_audit.json'
a=json.loads(AUDIT.read_text())
seeds=list(range(55,60)); metrics=('breadth8','pass8','mean8','distinct8')
assert a['registered_seeds']==seeds
assert set(a['per_seed'])==set(map(str,seeds))
output={'input_sha256':hashlib.sha256(AUDIT.read_bytes()).hexdigest(),'summaries':{},'endpoint_checks':[]}
def frac(x):return F(str(x))
def quantile(xs,p):
    ix=(len(xs)-1)*p; lo=ix.numerator//ix.denominator; rem=ix-lo
    return xs[lo]*(1-rem)+xs[min(lo+1,len(xs)-1)]*rem
for metric in metrics:
    effects=[]
    for seed in seeds:
        r=a['per_seed'][str(seed)]
        for arm in ('uniform','frequency'):
            assert frac(r[arm]['breadth8'])==frac(r[arm]['distinct8'])-frac(r[arm]['pass8'])
        effect=frac(r['uniform'][metric])-frac(r['frequency'][metric])
        assert effect==frac(r['uniform_minus_frequency'][metric])
        assert effect==frac(a['contrasts'][metric]['per_seed'][str(seed)])
        effects.append(effect)
    draws=sorted(sum(x,F(0))/5 for x in itertools.product(effects,repeat=5))
    mean=sum(effects,F(0))/5
    interval=[quantile(draws,F(1,40)),quantile(draws,F(39,40))]
    assert float(mean)==a['contrasts'][metric]['mean']
    assert [float(x) for x in interval]==a['contrasts'][metric]['paired_bootstrap_percentile_95']
    output['summaries'][metric]={'mean':float(mean),'ci95':[float(x) for x in interval],'ordered_resamples':len(draws),'all_seed_effects_positive':all(x>0 for x in effects),'all_seed_effects_negative':all(x<0 for x in effects),'per_seed':[float(x) for x in effects]}
ledger_path=ROOT/a['ledger']; ledger=json.loads(ledger_path.read_text())
assert hashlib.sha256(ledger_path.read_bytes()).hexdigest()==a['ledger_sha256']
assert ledger['source_gate']==a['source_gate'] and ledger['source_gate']['pytest']=='passed'
assert ledger['outcomes_inspected_before_release'] is False
for key in ('protocol','amendment'):
    p=Path(ledger[key]); assert hashlib.sha256(p.read_bytes()).hexdigest()==ledger[key+'_sha256']
output['source_gate']='preregistered source gate and protocol/amendment hashes verified'
raw_by_arm={}; source_count=0; prompt_count=0
metric_keys={'pass8':'any_correct_at_k','mean8':'mean_at_k','distinct8':'distinct_correct_modes_at_k'}
for endpoint in a['endpoint_integrity_audit']:
    run=Path(endpoint['run_dir']); seed=int(re.search(r'_s(\d+)$',run.name)[1]); arm='frequency' if 'fresh_frequency' in run.name else 'uniform'
    assert endpoint['status']=='admitted' and endpoint['step']==3072
    assert not endpoint['conflicting_retry_selected']
    source_set={Path(s['path']) for s in endpoint['sources']}
    observed_set=set(run.glob('debug*/eval_mode_coverage_draws.jsonl'))
    assert source_set==observed_set,(run,source_set,observed_set)
    terminal=[]
    for source in endpoint['sources']:
        path=Path(source['path']); data=path.read_bytes(); source_count+=1
        assert hashlib.sha256(data).hexdigest()==source['sha256'],path
        for lineno,line in enumerate(data.splitlines(),1):
            r=json.loads(line)
            if r.get('step')!=3072:continue
            if r.get('evaluation_kind')=='deterministic_greedy_trace_neutral':
                assert r['sample_count']==1 and r['temperature']==0 and r['draw_index'] is None
                continue
            assert r['sample_count']==8 and r['evaluation_kind']=='fixed_seed_sampled_k_neutral'
            assert r['temperature']==1.0 and r['top_p']==1.0 and len(r['prompts'])==128
            assert r['seed']==610100+r['draw_index']
            sums={k:F(0) for k in metric_keys}
            for prompt in r['prompts']:
                rewards=prompt['rewards']; keys=prompt['answer_keys']
                assert len(rewards)==len(keys)==8
                assert all(x in (0,1) for x in rewards)
                d=len({key for key,reward in zip(keys,rewards) if reward>0 and key is not None})
                vals={'pass8':F(int(any(rewards))),'mean8':F(sum(rewards))/8,'distinct8':F(d)}
                for k,v in vals.items():
                    assert v==frac(prompt['metrics'][metric_keys[k]])
                    sums[k]+=v
                prompt_count+=1
            for k,v in sums.items():assert v/128==frac(r['metrics'][metric_keys[k]])
            terminal.append((str(path),lineno,r))
    assert sorted(r['draw_index'] for _,_,r in terminal)==list(range(4))
    origins={(x['path'],x['line'],x['draw_index']) for x in endpoint['unique_draw_records']}
    assert origins=={(p,line,r['draw_index']) for p,line,r in terminal}
    raw={k:sum(frac(r['metrics'][name]) for _,_,r in terminal)/4 for k,name in metric_keys.items()}
    raw['breadth8']=raw['distinct8']-raw['pass8']
    assert all(raw[k]==frac(a['per_seed'][str(seed)][arm][k]) for k in raw)
    receipt=endpoint['completion_receipt']; receipt_path=Path(receipt['path'])
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest()==receipt['sha256']
    completion=json.loads(receipt_path.read_text()); assert completion['terminal_step']>=3072
    raw_by_arm[(seed,arm)]={r['draw_index']:[(p['prompt_index'],p['prompt'],p['reference']) for p in r['prompts']] for _,_,r in terminal}
    output['endpoint_checks'].append({'seed':seed,'arm':arm,'step':3072,'draws':4,'raw_metrics_match':True,'source_files':len(source_set)})
for seed in seeds:assert raw_by_arm[(seed,'uniform')]==raw_by_arm[(seed,'frequency')]
assert all(r['status']=='pass' and r['violations']==[] for r in a['telemetry'])
output['counts']={'source_files':source_count,'terminal_draws':len(output['endpoint_checks'])*4,'prompts_recomputed':prompt_count,'responses_recomputed':prompt_count*8}
output['scope']='Independent source hashes, complete terminal draws across every visible attempt, prompt/reward/key aggregates, matched prompt identities, endpoint subtraction, and exhaustive bootstrap checked. Full telemetry vectors are not independently replayed; visible audit reports five passing runs and zero violations.'
out=Path('/tmp/e120_falcon_graph_independent_review.json');out.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
