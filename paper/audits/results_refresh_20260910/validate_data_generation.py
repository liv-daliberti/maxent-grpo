"""Independent arithmetic/provenance checks for the dated September 10 assets."""
from pathlib import Path
import datetime, hashlib, itertools, json, math, statistics, sys
ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
STAMP = '20260910'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b): assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12), (a,b)
def key(r): return (r.get('scale',r.get('model_key','qwen05b')),r['domain'],r.get('arm','frequency'),int(r['seed']))
def interval(values,bootstrap=False):
    if bootstrap:
        vals=sorted(statistics.fmean(x) for x in itertools.product(values,repeat=5))
        def quant(q):
            pos=q*(len(vals)-1); lo=math.floor(pos); hi=math.ceil(pos)
            return vals[lo]+(pos-lo)*(vals[hi]-vals[lo])
        return [quant(.025),quant(.975)]
    half=2.7764451051977987*statistics.stdev(values)/math.sqrt(5)
    mean=statistics.fmean(values)
    return [mean-half,mean+half]
before=read(AUDIT/'data_generation_before.json')
for name,value in before['preserved_inputs'].items(): assert sha(ROOT/name)==value, name
new=read(ROOT/f'paper/results/latest_results_{STAMP}.json')
old=read(ROOT/'paper/results/latest_results_20260909.json')
new_audit=read(AUDIT/'latest_endpoints.json')
old_audit=read(ROOT/'paper/audits/results_refresh_20260909/latest_endpoints.json')
assert new['source_audit']['sha256']==sha(AUDIT/'latest_endpoints.json')
result={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_inputs_unchanged':len(before['preserved_inputs']),'comparison':'September 9 to September 10; distinct from the standard effects table baseline of September 6','campaigns':{},'checked_contrasts':0,'checked_metric_summaries':0}
for campaign,c in new['campaigns'].items():
    a=new_audit['campaigns'][campaign]; prev=old_audit['campaigns'][campaign]
    assert not a['ledger_changed_during_collection'], campaign
    assert a['ledger_sha256']==sha(ROOT/a['ledger_snapshot']['path'])
    rows=a['rows']+a.get('comparator_rows',[])
    mapped={key(r):r for r in rows}; assert len(mapped)==len(rows)
    for r in rows:
        if r['endpoint_status']=='admitted':
            integrity=r['integrity_audit']; assert integrity['step']==3072 and integrity['observed_draws']==[0,1,2,3] and integrity['conflicting_retry_selected'] is False
            close(r['endpoint']['breadth8'],r['endpoint']['distinct8']-r['endpoint']['pass8'])
    prev_mapped={key(r):r for r in prev['rows']+prev.get('comparator_rows',[])}
    added=[];lost=[];changed=[]
    for k,r in mapped.items():
        p=prev_mapped[k]
        if r['endpoint_status']=='admitted' and p['endpoint_status']!='admitted': added.append(list(k))
        if p['endpoint_status']=='admitted' and r['endpoint_status']!='admitted': lost.append(list(k))
        if p['endpoint_status']==r['endpoint_status']=='admitted' and p['endpoint']!=r['endpoint']: changed.append(list(k))
    assert not lost and not changed, (campaign,lost,changed)
    newly_complete=[];block_deltas=[]
    old_blocks={(b['model_key'],b['domain']):b for b in old['campaigns'][campaign]['blocks']}
    for b in c['blocks']:
        old_b=old_blocks[b['model_key'],b['domain']]
        if b['complete_five_seed_block'] and not old_b['complete_five_seed_block']: newly_complete.append([b['model_key'],b['domain']])
        if b['terminal_seeds_by_arm']!=old_b['terminal_seeds_by_arm']:
            block_deltas.append({'model_key':b['model_key'],'domain':b['domain'],'before':old_b['terminal_seeds_by_arm'],'after':b['terminal_seeds_by_arm'],'paired_seeds_before':old_b['paired_seeds'],'paired_seeds_after':b['paired_seeds'],'contrasts':b['contrasts'],'mechanism_validated_seeds':b.get('mechanism_validated_seeds')})
        for name,contrast in b['contrasts'].items():
            left,right=name.removeprefix('four_arm_').split('_minus_')
            expected=sorted(set(b['terminal_seeds_by_arm'][left])&set(b['terminal_seeds_by_arm'][right]))
            if name.startswith('four_arm_'): expected=b['paired_seeds']
            assert contrast['paired_seeds']==expected
            for metric,s in contrast['summaries'].items():
                vals={str(seed):mapped[b['model_key'],b['domain'],left,seed]['endpoint'][metric]-mapped[b['model_key'],b['domain'],right,seed]['endpoint'][metric] for seed in expected}
                assert s['n']==len(expected) and set(s['per_seed'])==set(vals)
                for seed,v in vals.items(): close(s['per_seed'][seed],v)
                if vals:
                    close(s['mean'],statistics.fmean(vals.values()))
                    for arm,field in [(left,'left_mean'),(right,'right_mean')]: close(s[field],statistics.fmean(mapped[b['model_key'],b['domain'],arm,seed]['endpoint'][metric] for seed in expected))
                ikey='paired_bootstrap_percentile_95' if campaign=='e120' else 'student_t_95'
                if len(vals)==5:
                    for actual,target in zip(s[ikey],interval(list(vals.values()),campaign=='e120')):close(actual,target)
                else:assert 'student_t_95' not in s and 'paired_bootstrap_percentile_95' not in s
                result['checked_metric_summaries']+=1
            result['checked_contrasts']+=1
    result['campaigns'][campaign]={'before':{k:old['campaigns'][campaign][k] for k in ['completion_receipts','admitted_terminal_endpoints','complete_five_seed_blocks']},'after':{k:c[k] for k in ['completion_receipts','admitted_terminal_endpoints','complete_five_seed_blocks']},'added_endpoints_including_comparators':added,'lost_endpoints':lost,'changed_retained_endpoint_values':changed,'new_complete_blocks':newly_complete,'changed_blocks':block_deltas,'ledger_changed_during_collection':False}
factorial=read(ROOT/f'paper/results/level2_factorial_contrasts_{STAMP}.json')
assert factorial['source_audit']['sha256']==sha(AUDIT/'latest_endpoints.json')
mapped={key(r):r for r in new_audit['campaigns']['e119']['rows']}
for b in factorial['blocks']:
    expected=sorted(set.intersection(*(set(v) for v in b['terminal_seeds_by_arm'].values())))
    assert b['paired_seeds']==expected
    for name,c in b['contrasts'].items():
        for metric,s in c['summaries'].items():
            vals={str(seed):sum(coef*mapped['qwen05b',b['domain'],arm,seed]['endpoint'][metric] for arm,coef in c['coefficients'].items()) for seed in expected}
            assert s['n']==len(expected)
            for seed,v in vals.items():close(s['per_seed'][seed],v)
            if vals:close(s['mean'],statistics.fmean(vals.values()))
            if len(vals)==5:
                for actual,target in zip(s['student_t_95'],interval(list(vals.values()))):close(actual,target)
            else: assert 'student_t_95' not in s
            result['checked_metric_summaries']+=1
        result['checked_contrasts']+=1
result['outputs']={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'paper/results').glob(f'*{STAMP}*'))+list((ROOT/'paper/figures').glob(f'current_campaign_results_{STAMP}*'))}
(AUDIT/'data_generation_validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['campaigns','outputs']},indent=2))
for campaign,c in result['campaigns'].items():print(campaign,json.dumps({k:v for k,v in c.items() if k!='changed_blocks'}))
