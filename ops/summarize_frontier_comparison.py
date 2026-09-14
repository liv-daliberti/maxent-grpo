#!/usr/bin/env python3
"""Compare completed hosted cohorts and independently reconstruct occupancy rates."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

DOMAINS = ('countdown','graph_coloring','mathir','pantry_plan','python_factors')
METRICS = ('accuracy','distinct8','collision','uniform_collision','excess_uniform')
SOURCE_METRICS = ('pass1','distinct8','correct_pair_collision','uniform_correct_pair_collision','correct_pair_collision_excess_uniform')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def values(a):
    with np.errstate(divide='ignore',invalid='ignore'):
        accuracy=a[...,1]/(8*a[...,0]); distinct=a[...,2]/a[...,0]
        collision=a[...,4]/a[...,3]; uniform=a[...,5]/a[...,6]
        excess=a[...,7]/a[...,6]-uniform
    return np.stack((accuracy,distinct,collision,uniform,excess),axis=-1)
def read_jsonl(path): return [json.loads(line) for line in path.open() if line.strip()]
def sample_key(s): return (s['level'],s['domain'],s['row_index'],s['sample_index'])
def compact_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def independent_prompts(run,summary,grading):
    primary_path=Path(summary['primary_samples_path'])
    if digest(primary_path)!=summary['primary_samples_sha256']:
        raise ValueError('Primary grading evidence changed since summary')
    original={sample_key(s):s for s in read_jsonl(run/'samples.jsonl')}
    samples=read_jsonl(primary_path)
    if len(samples)!=15360 or len(original)!=15360:raise ValueError('Incomplete source samples')
    if grading=='normalized_secondary':
        if digest(run/'normalized_samples.jsonl')!=summary[grading]['cache_sha256']:
            raise ValueError('Normalized cache changed since summary')
        normalizer=summary[grading]['normalization_source_sha256']
        cache={r['strict_receipt_sha256']:r['normalization'] for r in read_jsonl(run/'normalized_samples.jsonl')
               if r['normalization_source_sha256']==normalizer}
        derived=[]
        for sample in samples:
            receipt={**original[sample_key(sample)],**{k:sample[k] for k in ('verified','canonical_key','graded_text')}}
            grade=cache[compact_sha(receipt)]
            if sample['verified'] and (not grade['verified'] or sample['canonical_key']!=grade['canonical_key']):
                raise ValueError('Normalization changed an original strict success')
            derived.append({**sample,'verified':grade['verified'],'canonical_key':grade['canonical_key']})
        samples=derived
    grouped=defaultdict(list)
    for sample in samples:grouped[sample_key(sample)[:3]].append(sample)
    prompts=[]
    for row in read_jsonl(run/'rows.jsonl'):
        ss=grouped[row['level'],row['domain'],row['row_index']]
        if len(ss)!=8 or {s['sample_index'] for s in ss}!=set(range(8)):
            raise ValueError('Incomplete/duplicated eight-draw prompt')
        modes=Counter(json.dumps(s['canonical_key'],sort_keys=True) for s in ss if s['verified'])
        correct=sum(modes.values());pairs=correct*(correct-1)//2
        m=row['metadata'].get('answer_mode_count') if row['domain']!='countdown' else None
        if row['metadata'].get('support_is_open'):m=None
        prompts.append({'level':row['level'],'domain':row['domain'],'row_index':row['row_index'],
          'correct_draws':correct,'correct_pairs':pairs,'distinct8':len(modes),
          'colliding_correct_pairs':sum(n*(n-1)//2 for n in modes.values()),
          'certified_support_count':m,'uniform_expected_colliding_correct_pairs':pairs/m if m else None})
    return prompts
def matrix(prompts):
    return np.asarray([[1,p['correct_draws'],p['distinct8'],p['correct_pairs'],p['colliding_correct_pairs'],
        p['uniform_expected_colliding_correct_pairs'] or 0,
        p['correct_pairs'] if p['certified_support_count'] else 0,
        p['colliding_correct_pairs'] if p['certified_support_count'] else 0] for p in prompts],float)
def describe(point,bootstrap):
    result={}
    for j,name in enumerate(METRICS):
        valid=bootstrap[:,j][np.isfinite(bootstrap[:,j])]
        result[name]={'estimate':float(point[j]) if np.isfinite(point[j]) else None,
            'ci95':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,
            'defined_replicates':len(valid)}
    return result

def fmt(m,name='collision',percent=True):
    r=m[name];v=r['estimate'];scale=100 if percent else 1
    if v is None:return '—'
    return f"{scale*v:.1f}"+(f" [{scale*r['ci95'][0]:.1f}, {scale*r['ci95'][1]:.1f}]" if r['ci95'] else '')

def validate_interruption_provenance(run, audit, inventory):
    adapter_path = run / 'completion_audit_adapter.json'
    if not adapter_path.exists():
        if audit.get('interruption_accounting'):
            raise ValueError('Missing registered completion audit adapter')
        return
    if digest(adapter_path) != audit.get('completion_audit_adapter_sha256'):
        raise ValueError('Completion audit adapter changed since audit')
    adapter = json.loads(adapter_path.read_text())
    if adapter.get('schema') != 'hosted-completion-audit-adapter-v1':
        raise ValueError('Unsupported completion audit adapter')
    registration_path = run / adapter['registration']
    registration_digest = digest(registration_path)
    if registration_digest != adapter['registration_sha256'] or registration_digest != audit.get('interruption_registration_sha256'):
        raise ValueError('Interrupted-attempt registration changed since audit')
    for relative, expected in adapter['source_sha256'].items():
        path = (run / relative).resolve()
        if not path.is_relative_to(run.resolve()) or digest(path) != expected or inventory.get(relative) != expected:
            raise ValueError('Interruption audit source differs from authenticated inventory')
    for relative in ('events.jsonl', 'completion_audit_adapter.json', adapter['registration']):
        if digest(run / relative) != inventory.get(relative):
            raise ValueError('Interruption evidence changed since audit: ' + relative)
    helper_relative = 'interruption_analysis_code/ops/audit_hosted_interrupted_attempts.py'
    if helper_relative not in adapter['source_sha256']:
        raise ValueError('Interruption accounting helper is not source-bound')
    spec = importlib.util.spec_from_file_location('_independent_interruption_accounting', run / helper_relative)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    raw = {}
    for relative, expected in inventory.items():
        if not relative.startswith('raw_responses/'):
            continue
        data = (run / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Raw attempt receipt changed since interruption audit')
        record = json.loads(data)
        key = (record.get('group_id') or record.get('sample_id'), record['attempt'])
        if key in raw:
            raise ValueError('Duplicate raw attempt identity')
        raw[key] = record
    result = helper.validate_attempt_accounting(read_jsonl(run / 'events.jsonl'), raw, json.loads(registration_path.read_text()))
    if result != audit.get('interruption_accounting'):
        raise ValueError('Independent interrupted-attempt accounting differs from audit')

def validate_evidence(run,s,model):
    audit=json.loads((run/'completion_audit.json').read_text())
    if audit.get('status')!='pass':raise ValueError('Completion audit not passing: '+str(run))
    if audit.get('expected_responses')!=15360 or audit.get('saved_samples')!=15360 or audit.get('unique_response_choice_ids',audit.get('unique_response_ids'))!=15360:
        raise ValueError('Completion audit counts do not identify a full cohort')
    if audit.get('model',model)!=model or s['models_returned'][model]!=15360 or s['run_configuration']['model']!=model:
        raise ValueError('Completion audit or summary has a different deployment identity')
    inventory=json.loads((run/'evidence_file_sha256.json').read_text())
    if compact_sha(inventory)!=audit['evidence_inventory_sha256']:
        raise ValueError('Completion evidence inventory changed since audit')
    for name in ('manifest.json','rows.jsonl','datasets.json','requests.jsonl','samples.jsonl'):
        if name not in inventory or digest(run/name)!=inventory[name]:
            raise ValueError('Selected cohort differs from audited evidence: '+name)
    validate_interruption_provenance(run, audit, inventory)
    secondary=s.get('normalized_secondary')
    if secondary is not None and digest(run/'normalized_samples.jsonl')!=secondary['cache_sha256']:
        raise ValueError('Normalized cache changed since summary')
    return audit

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,action='append',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--replicates',type=int,default=20000)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20260913)
    # Same bootstrap prompt indices across models permit paired model contrasts.
    indices={(l,d):rng.integers(0,128,(args.replicates,128)) for l in (1,2,3) for d in DOMAINS}
    output={'schema':'hosted-frontier-comparison-v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'analysis_source_sha256':digest(Path(__file__)),'bootstrap':{'replicates':args.replicates,'seed':20260913,
        'unit':'Whole prompts within domain/level; shared prompt indices across models; independent across levels.',
        'interval':'Pointwise 95% percentile; no multiple-comparison adjustment.'},'models':{},'validation':{}}
    matrices={};bootstraps={};all_identity=None;row_hash=None
    for run in args.run:
        s=json.loads((run/'summary.json').read_text())
        if s['status']!='complete' or s['received_responses']!=15360 or s['complete_prompts']!=1920:
            raise ValueError('Incomplete run: '+str(run))
        current=digest(run/'rows.jsonl')
        if current!=s['input_sha256']['rows.jsonl'] or digest(run/'samples.jsonl')!=s['input_sha256']['samples.jsonl']:
            raise ValueError('Source data changed since summary')
        if row_hash is not None and current!=row_hash:raise ValueError('Different evaluation rows')
        row_hash=current
        model=next(iter(s['models_returned']))
        if len(s['models_returned'])!=1 or model in output['models']:raise ValueError('Model identity ambiguous/duplicated')
        audit=validate_evidence(run,s,model)
        rec={'directory':str(run.resolve()),'source_sha256':{n:digest(run/n) for n in ('summary.json','rows.jsonl','completion_audit.json')},'response_status_counts':s['response_status_counts'],'analyses':{}}
        if audit.get('interruption_accounting'):
            rec['interruption_accounting'] = audit['interruption_accounting']
            rec['source_sha256']['completion_audit_adapter.json'] = digest(run/'completion_audit_adapter.json')
            rec['source_sha256']['interrupted_attempt_registration.json'] = digest(run/'interrupted_attempt_registration.json')
        stops=Counter();refusal_cells=Counter()
        for sample in read_jsonl(run/'samples.jsonl'):
            stops[str(sample.get('stop_reason','not_reported'))]+=1
            if sample.get('stop_reason')=='refusal':
                refusal_cells[f"level{sample['level']}/{sample['domain']}"]+=1
        rec['explicit_stop_reason_counts']=dict(stops)
        rec['native_stop_reason_refusals_by_cell']=dict(refusal_cells)
        output['models'][model]=rec
        for grading,source in [('strict',s),('normalized_secondary',s.get('normalized_secondary'))]:
            if source is None:continue
            grouped=defaultdict(list)
            for p in independent_prompts(run,s,grading):grouped[p['level'],p['domain']].append(p)
            cells={};points={};boots={}
            for cell in indices:
                ps=sorted(grouped[cell],key=lambda p:p['row_index'])
                if [p['row_index'] for p in ps]!=list(range(128)):raise ValueError('Prompt identities differ')
                a=matrix(ps);point=values(a.sum(axis=0));boot=np.empty((args.replicates,len(METRICS)))
                for start in range(0,args.replicates,256):
                    boot[start:start+256]=values(a[indices[cell][start:start+256]].sum(axis=1))
                key=f'level{cell[0]}/{cell[1]}'
                for j,source_name in enumerate(SOURCE_METRICS):
                    expected=source['cells'][key]['metrics'][source_name]['estimate']
                    if not ((expected is None and not np.isfinite(point[j])) or (expected is not None and abs(expected-point[j])<1e-12)):
                        raise ValueError(f'Independent metric mismatch: {model}/{grading}/{key}/{source_name}')
                cells[key]=describe(point,boot);points[cell]=point;boots[cell]=boot
                matrices[model,grading,cell]=a;bootstraps[model,grading,cell]=boot
            groups={}
            for label,domains in [('five_domain_macro',DOMAINS),('four_closed_support_domains',DOMAINS[1:])]:
                lp={l:np.mean([points[l,d] for d in domains],axis=0) for l in (1,2,3)}
                lb={l:np.mean([boots[l,d] for d in domains],axis=0) for l in (1,2,3)}
                groups[label]={'levels':{str(l):describe(lp[l],lb[l]) for l in (1,2,3)},
                    'level_contrasts':{f'L{b}-L{a}':describe(lp[b]-lp[a],lb[b]-lb[a]) for a,b in ((1,2),(1,3),(2,3))}}
            rec['analyses'][grading]={'cells':cells,'groups':groups,'domain_level_contrasts':{d:{f'L{b}-L{a}':describe(points[b,d]-points[a,d],boots[b,d]-boots[a,d]) for a,b in ((1,2),(1,3),(2,3))} for d in DOMAINS}}
    reference=next(iter(output['models']))
    output['paired_model_contrasts']={'reference_model':reference,'contrasts':{}}
    for model,rec in output['models'].items():
        if model==reference:continue
        for grading in rec['analyses']:
            if grading not in output['models'][reference]['analyses']:continue
            cells={};points={};boots={}
            for cell in indices:
                a=matrices[model,grading,cell];ref=matrices[reference,grading,cell]
                point=values(a.sum(axis=0))-values(ref.sum(axis=0))
                boot=bootstraps[model,grading,cell]-bootstraps[reference,grading,cell]
                cells[f'level{cell[0]}/{cell[1]}']=describe(point,boot);points[cell]=point;boots[cell]=boot
            output['paired_model_contrasts']['contrasts'].setdefault(model,{})[grading]={
                'cells':cells,'five_domain_macro_by_level':{str(l):describe(
                    np.mean([points[l,d] for d in DOMAINS],axis=0),
                    np.mean([boots[l,d] for d in DOMAINS],axis=0)) for l in (1,2,3)}}
    output['validation']={'identical_rows_sha256':row_hash,'independent_receipt_grouping_matches_all_source_cells':True}
    output['limitations']=[
        'Static inference concentration does not identify training-induced collapse, a causal scale effect, or zero-probability unseen modes.',
        'Provider-specific decoding and reasoning controls are not matched compute budgets. Report deployment IDs exactly.',
        'Benchmark levels use different prompt populations and do not guarantee increasing difficulty for a given model.',
        'Several prompts prefer particular solution forms, and Pantry output interfaces differ across levels.',
        'Pair collision conditions on correct draws but emphasizes prompts with more correct pairs.',
        'The frozen normalization diagnostic was designed after the first 15 GPT-5.6 Sol responses and before new-model evaluations.',
        'Countdown verifier support exceeds its enumerated expression library; its uniform reference is undefined.',
        'Models share test prompts; displayed intervals are descriptive and do not resample API draws separately.']
    (args.output/'comparison.json').write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# Hosted ModeBench comparison','',f"{len(output['models'])} completed model cohorts; 15,360 saved responses each. No training. All model cohorts use identical evaluation rows and eight stateless requests per prompt.",'',
        'Collision is the fraction of pairs of correct answers to the same prompt that share a canonical mode. Domain rates pool correct pairs; macros weight five domains equally. Intervals resample whole prompts.','']
    for grading in ('strict','normalized_secondary'):
        lines += [f'## {grading.replace("_"," ")}', '', '| Model | Level | Accuracy % | Distinct correct modes / 8 | Collision % (95% CI) |','|---|---:|---:|---:|---:|']
        for model,rec in output['models'].items():
            if grading not in rec['analyses']:continue
            for l,m in rec['analyses'][grading]['groups']['five_domain_macro']['levels'].items():
                lines.append(f"| {model} | {l} | {100*m['accuracy']['estimate']:.1f} | {m['distinct8']['estimate']:.3f} | {fmt(m)} |")
        lines += ['', '| Model | L2 − L1 collision, pp | L3 − L1, pp | L3 − L2, pp |','|---|---:|---:|---:|']
        for model,rec in output['models'].items():
            if grading not in rec['analyses']:continue
            c=rec['analyses'][grading]['groups']['five_domain_macro']['level_contrasts']
            lines.append(f"| {model} | {fmt(c['L2-L1'])} | {fmt(c['L3-L1'])} | {fmt(c['L3-L2'])} |")
        lines += ['']
    refusal_models=[(model,r) for model,r in output['models'].items() if r['native_stop_reason_refusals_by_cell']]
    if refusal_models:
        lines+=['## Explicit native refusals','',
          'Refusals remain sampled outputs in accuracy. A domain with no prompt containing at least two correct draws has undefined correct-pair collision; the five-domain mean is then undefined rather than averaging only the remaining domains.',
          '', '| Model | Cell | Native refusal responses / 1,024 |','|---|---|---:|']
        for model,r in refusal_models:
            for cell,n in sorted(r['native_stop_reason_refusals_by_cell'].items()):
                lines.append(f'| {model} | {cell} | {n:,} |')
        lines+=['','These counts use explicit native refusal stop reasons. Detailed provider outcome audits retain categories and other filter markers. Refusal rates are deployment behavior, not evidence of low mathematical ability or concentration among valid Python solutions.','']
    lines += ['## Level-3 Graph, strict grading','','This diagnostic was selected after the GPT-5.6 Sol run and before collecting the additional models. It uses the same system prompt at all three levels and has certified finite mode counts.','',
        '| Model | Accuracy % | Distinct modes / 8 | Collision % (95% CI) | Uniform reference % |','|---|---:|---:|---:|---:|']
    for model,r in output['models'].items():
        m=r['analyses']['strict']['cells']['level3/graph_coloring']
        uniform=m['uniform_collision']['estimate']
        uniform_text='—' if uniform is None else f'{100*uniform:.1f}'
        lines.append(f"| {model} | {100*m['accuracy']['estimate']:.1f} | {m['distinct8']['estimate']:.3f} | {fmt(m)} | {uniform_text} |")
    if output['paired_model_contrasts']['contrasts']:
        lines+=['',f'Level-3 Graph differences from {reference}, using paired resampling of the same test prompts:',
          '', '| Model | Collision difference, percentage points (95% CI) |','|---|---:|']
        for model,rec in output['paired_model_contrasts']['contrasts'].items():
            lines.append(f"| {model} | {fmt(rec['strict']['cells']['level3/graph_coloring'])} |")
        lines+=['','These compare the specified deployed configurations; they do not isolate architecture, training, or compute.']
    for model, record in output['models'].items():
        if record.get('interruption_accounting'):
            count = record['interruption_accounting']['registered_interrupted_attempts_with_unknown_outcome']
            lines += ['', f'{model}: {count} historical request starts lost their response receipts when collection was interrupted. Their outcomes and usage remain unknown and possibly billed. They are explicitly registered and timestamp-audited separately from the 15,360 retained terminal samples; no original events or raw receipts were changed.']
    lines+=['','## Interpretation limits','']+[f'- {x}' for x in output['limitations']]
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'models':list(output['models']),'independent_rate_validation':True,'output':str(args.output)}))
if __name__=='__main__':main()
