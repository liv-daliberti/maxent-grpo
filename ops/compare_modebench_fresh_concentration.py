#!/usr/bin/env python3
"""Retrospective old-versus-new sensitivity; never pool response collections.

Consumes the immutable historical reference and a COMPLETE authenticated fresh
panel report. Scientific contrasts remain the separately registered primary
analyses. Common-population comparisons here are explicitly descriptive.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import analyze_modebench_fresh_concentration as fresh

require=fresh.require


def read_bound(binding):
    path=Path(binding['path']);path=path if path.is_absolute() else ROOT/path
    require(fresh.file_binding(path)['sha256']==binding['sha256'],f'Bound source changed: {path}')
    return path


def read_published(path):
    path=Path(path).resolve();manifest=json.loads((path.parent/'manifest.json').read_text())
    require(path.name in manifest['files'],'Requested report absent from publication manifest')
    for relative,digest in manifest['files'].items():
        require(fresh.file_binding(path.parent/relative)['sha256']==digest,'Immutable publication content changed')
    return json.loads(path.read_text())


def historical_endpoints(reference):
    """Reconstruct the exact original promptwise estimator from its bound cache."""
    source=json.loads(read_bound(reference['sources']['retrospective_result']).read_text())
    helper_path=ROOT/'ops/exp_scaling/analyze_paper_conditional_concentration.py'
    require(fresh.file_binding(helper_path)['sha256']==source['analysis_code_sha256'],'Historical analysis helper changed')
    spec=importlib.util.spec_from_file_location('matched_historical_collision',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cache=read_bound({'path':reference['old_cache_binding']['path'],'sha256':reference['old_cache_binding']['sha256']})
    read_bound({'path':reference['old_cache_binding']['binding_path'],'sha256':reference['old_cache_binding']['binding_sha256']})
    needed={(c['model_scale'],c['domain'],method,s) for c in reference['contrasts'] for s in c['registered_seeds']
            for method in ((c['right_method'],) if c['left_method']=='initial' else (c['left_method'],c['right_method']))}
    endpoints={};seen=set()
    with gzip.open(cache,'rt') as handle:
        for line in handle:
            cell=json.loads(line)
            if cell.get('level')!='level1':continue
            key=cell['scale'],cell['domain'],cell['method'],cell['seed']
            if key not in needed:continue
            require(key not in seen,'Duplicate historical cell in bound cache');seen.add(key)
            for step in ('0','3072'):
                cp=helper.prepare_checkpoint(cell['checkpoints'].get(step))
                endpoints[key,step]=cp
    require(seen==needed,'Missing registered historical cells')
    for c in reference['contrasts']:
        for r in c['per_seed']:
            left,right=old_pair(c,r['paired_seed'],endpoints)
            saved=r['historical_primary']
            if saved is None:
                require(left is None or right is None,'Reference hides available historical pair')
                continue
            require(left is not None and right is not None,'Historical reference pair cannot be reconstructed')
            rebuilt=helper.paired_streams(left,right)
            require(rebuilt['eligible_ids']==saved['eligible_ids'],'Reconstructed historical eligibility differs')
            for metric,value in saved['delta'].items():
                actual=rebuilt['delta'][metric]
                require(actual==value if value is None else abs(actual-value)<1e-12,'Reconstructed historical effect differs')
    return helper,endpoints,{'cache':fresh.file_binding(cache),'historical_helper':fresh.file_binding(helper_path),
                            'reconstructed_cells':len(seen),'checked_reference_contrasts':len(reference['contrasts'])}


def old_pair(contrast,seed,endpoints):
    scale,domain=contrast['model_scale'],contrast['domain']
    if contrast['left_method']=='initial':
        cell=scale,domain,contrast['right_method'],seed
        return endpoints[cell,'0'],endpoints[cell,'3072']
    return (endpoints[(scale,domain,contrast['left_method'],seed),'3072'],
            endpoints[(scale,domain,contrast['right_method'],seed),'3072'])


def direction(value):
    return 'undefined' if value is None else 'positive' if value>0 else 'negative' if value<0 else 'zero'


def common_values(old_left,old_right,new_prompts,ids,helper):
    """Different response pools, identical prompt IDs; no cross-study pairs."""
    rows=[]
    for pid in sorted(ids):
        a,b=helper.selected_metrics(old_left[pid]),helper.selected_metrics(old_right[pid])
        n=new_prompts[pid]
        require(a['collision'] is not None and b['collision'] is not None and n['jointly_eligible'],
                'Common population contains an undefined old or new conditional')
        old_delta=b['collision']-a['collision'];new_delta=n['delta']['collision']
        rows.append({'prompt_id':pid,'old_left_collision':a['collision'],'old_right_collision':b['collision'],
                     'new_left_collision':n['left']['collision'],'new_right_collision':n['right']['collision'],
                     'old_delta_collision':old_delta,'new_delta_collision':new_delta,'new_minus_old_delta':new_delta-old_delta,
                     'old_mean8_delta':b['mean8']-a['mean8'],
                     'new_mean_correct_delta':n['delta']['mean_correct'],
                     'old_left_correct':a['selected_correct'],'old_right_correct':b['selected_correct'],
                     'new_left_correct':n['left']['correct_draws'],'new_right_correct':n['right']['correct_draws'],
                     'old_left_budget':a['selected_total'],'old_right_budget':b['selected_total'],
                     'new_left_budget':n['left']['draws'],'new_right_budget':n['right']['draws']})
    mean=lambda name:statistics.mean(r[name] for r in rows) if rows else None
    return {'n_prompts':len(rows),'prompt_ids':sorted(ids),'old_delta':mean('old_delta_collision'),
            'new_delta':mean('new_delta_collision'),'new_minus_old_delta':mean('new_minus_old_delta'),
            'old_mean8_delta':mean('old_mean8_delta'),'new_mean_correct_delta':mean('new_mean_correct_delta'),
            'per_prompt':rows,'ci95':None,'interpretation':'descriptive cross-study sensitivity on identical prompt identities; separate response pools and reference conditions'}


def compare_seed(old_record,new_record,old_left,old_right,helper,*,expected_prompt_ids):
    require(old_record['paired_seed']==new_record['training_seed'],'Historical/new seed pairing differs')
    new_prompts={p['prompt_id']:p for p in new_record['prompts']}
    require(len(new_prompts)==len(new_record['prompts']) and set(new_prompts)==set(expected_prompt_ids),
            'New comparison has missing/duplicate/extra fixed prompts')
    new_ids={p for p,n in new_prompts.items() if n['jointly_eligible']}
    require(len(new_ids)==new_record['jointly_eligible_prompts'],'New eligibility count differs')
    saved=old_record['historical_primary']
    if saved is None:
        return {'paired_seed':old_record['paired_seed'],'status':'historical_source_unavailable','old_joint_prompt_ids':None,
                'new_joint_prompt_ids':sorted(new_ids),'overlap':None,'common_population':None,
                'historical_source_issues':old_record['source_issues']}
    require(old_left is not None and old_right is not None and set(old_left)==set(old_right)==set(expected_prompt_ids),
            'Old/new full task populations differ')
    old_ids=set(saved['eligible_ids']);common=old_ids&new_ids
    return {'paired_seed':old_record['paired_seed'],'status':'common_population_defined' if common else 'no_common_eligible_prompts',
            'old_joint_prompt_ids':sorted(old_ids),'new_joint_prompt_ids':sorted(new_ids),
            'overlap':{'old':len(old_ids),'new':len(new_ids),'both':len(common),'old_only':len(old_ids-new_ids),
                       'new_only':len(new_ids-old_ids),'neither':len(set(expected_prompt_ids)-(old_ids|new_ids)),
                       'jaccard':len(common)/len(old_ids|new_ids) if old_ids|new_ids else None},
            'common_population':common_values(old_left,old_right,new_prompts,common,helper)}


def summarize_common(records):
    defined=[r for r in records if r.get('common_population') and r['common_population']['n_prompts']>0]
    return {'registered_seeds':[r['paired_seed'] for r in records],
            'defined_seeds':[r['paired_seed'] for r in defined],
            'undefined_seeds':[r['paired_seed'] for r in records if r not in defined],
            'n_defined':len(defined),'n_registered':len(records),
            'equal_seed_means':{m:statistics.mean(r['common_population'][m] for r in defined) if defined else None
                                for m in ('old_delta','new_delta','new_minus_old_delta','old_mean8_delta','new_mean_correct_delta')},
            'ci95':None,'scope':'descriptive retrospective comparison sensitivity; no test based on interval overlap'}


def identity(c):
    return c['model_scale'],c['domain'],c['level'],c['wording'],c['contrast']


def build_report(reference_path,new_path):
    reference=read_published(reference_path);new=read_published(new_path)
    require(new['status']=='complete' and new['scope']['panel']=='registered_Level1_fresh'
            and new['completeness_audit']['status']=='complete'
            and new['completeness_audit']['expected_tasks']==new['completeness_audit']['authenticated_tasks']==150,
            'Requires the complete authenticated 150-task fresh panel')
    require(new['completeness_audit']['authenticated_response_slots']==1228800,'Fresh response inventory incomplete')
    oldmap={identity(c):c for c in reference['contrasts']};newmap={identity(c):c for c in new['contrasts']}
    require(len(reference['contrasts'])==len(new['contrasts'])==len(oldmap)==len(newmap)==36
            and set(oldmap)==set(newmap),'Exact36old/new contrast mapping differs')
    helper,endpoints,cache_audit=historical_endpoints(reference)
    comparisons=[]
    for key,old in sorted(oldmap.items()):
        current=newmap[key]
        new_seeds={r['training_seed']:r for r in current['seeds']}
        require(len(new_seeds)==len(current['seeds'])==5 and set(new_seeds)==set(old['registered_seeds']),
                'Old/new registered training seed inventories differ')
        records=[]
        expected=set(reference['prompt_identity_to_row_index'][old['domain']])
        for r in old['per_seed']:
            left,right=old_pair(old,r['paired_seed'],endpoints)
            records.append(compare_seed(r,new_seeds[r['paired_seed']],left,right,helper,expected_prompt_ids=expected))
        fixed=None
        if all(r['common_population'] is not None for r in records):
            ids=set.intersection(*[set(r['common_population']['prompt_ids']) for r in records])
            fixed_rows=[]
            for r in records:
                left,right=old_pair(old,r['paired_seed'],endpoints)
                prompts={p['prompt_id']:p for p in new_seeds[r['paired_seed']]['prompts']}
                fixed_rows.append({'paired_seed':r['paired_seed'],'common_population':common_values(left,right,prompts,ids,helper)})
            fixed={'prompt_ids':sorted(ids),'n_prompts':len(ids),'summary':summarize_common(fixed_rows),'per_seed':fixed_rows}
        a=old['primary']['mean'];b=current['summary']['equal_prompt_delta_equal_seed']
        da,db=direction(a),direction(b)
        category='undefined' if 'undefined' in (da,db) else 'same_'+da if da==db else 'opposite' if {da,db}=={'positive','negative'} else 'zero_vs_nonzero'
        comparisons.append({'model_scale':old['model_scale'],'domain':old['domain'],'level':1,'wording':'original',
                            'contrast':old['contrast'],'old_primary':old['primary'],'new_primary':current['summary'],
                            'old_direction':da,'new_direction':db,'descriptive_direction_comparison':category,
                            'difference_of_separate_primary_estimates_descriptive':b-a if a is not None and b is not None else None,
                            'per_seed':records,'common_population_summary':summarize_common(records),
                            'fixed_across_all_registered_seeds_common_population':fixed})
    counts={name:sum(c['descriptive_direction_comparison']==name for c in comparisons)
            for name in ('same_positive','same_negative','same_zero','opposite','zero_vs_nonzero','undefined')}
    return {'schema':'modebench-old-new-concentration-comparison-v1','status':'complete',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'analysis_role':'retrospective comparison sensitivity; not the primary preregistered extension',
            'generation_calls':0,'verifier_calls':0,'old_artifacts_modified':False,'old_new_responses_pooled':False,
            'sources':{'reference':fresh.file_binding(reference_path),'fresh_report':fresh.file_binding(new_path),
                       'historical_reconstruction':cache_audit,'builder':fresh.file_binding(__file__)},
            'comparison_count':len(comparisons),'direction_counts_descriptive':counts,'comparisons':comparisons,
            'limits':['No interval-overlap test or pooling of old/new response draws is performed.',
                      'Separate primary estimates can concern different eligible populations and different initial-output reference designs.',
                      'Common-prompt sensitivities recompute both old and new estimators on precisely the same task identities; no old full-population mean substitutes for a subset.',
                      'Common populations remain outcome-selected and can differ across seeds. Their equal-seed means are descriptive; no additional confidence intervals are attached.',
                      'Old eleven nominal-stream representatives and new 64-draw pools have different sampling/runtime provenance. Current receipts do not establish exact historical runtime identity.',
                      'Old mean8 uses intact original groups; new mean_correct uses fresh64. Collision changes do not establish correctness-controlled or causal diversity effects.']}


def publish(report,output):
    output=Path(output).resolve();require(not output.exists(),'Comparison output already exists; refusing overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.old-new-comparison-',dir=output.parent) as temp:
        stage=Path(temp);fresh.json_write(stage/'report.json',report);rows=[]
        for c in report['comparisons']:
            rows.append({**{k:c[k] for k in ('model_scale','domain','contrast','old_direction','new_direction','descriptive_direction_comparison')},
                         'old_primary':c['old_primary']['mean'],'new_primary':c['new_primary']['equal_prompt_delta_equal_seed'],
                         'old_defined_seeds':c['old_primary']['n'],'new_defined_seeds':c['new_primary']['n_defined'],
                         'common_defined_seeds':c['common_population_summary']['n_defined'],
                         **{'common_'+k:v for k,v in c['common_population_summary']['equal_seed_means'].items()}})
        fresh._csv(stage/'comparisons.csv',rows)
        seed_rows=[];prompt_rows=[]
        for c in report['comparisons']:
            meta={k:c[k] for k in ('model_scale','domain','contrast')}
            for r in c['per_seed']:
                seed_rows.append({**meta,'paired_seed':r['paired_seed'],'status':r['status'],**(r['overlap'] or {}),
                                  **{k:v for k,v in (r['common_population'] or {}).items() if k in ('old_delta','new_delta','new_minus_old_delta','n_prompts')}})
                for p in (r['common_population'] or {}).get('per_prompt',[]):
                    prompt_rows.append({**meta,'paired_seed':r['paired_seed'],**p})
        fresh._csv(stage/'seed_overlaps.csv',seed_rows);fresh._csv(stage/'common_prompt_contrasts.csv',prompt_rows)
        lines=['# Retrospective old-versus-new concentration comparison','',report['analysis_role']+'.',
               '', 'All 36 contrasts and all registered seeds are retained. Direction counts are descriptive; they are not a replication test.',
               '',json.dumps(report['direction_counts_descriptive'],sort_keys=True),'',
               'The JSON retains separate primary intervals, every eligible-set overlap, recomputed promptwise estimates on each old/new intersection, and a fixed-all-seeds intersection sensitivity. CSV files expose block, seed and common-prompt values.','']
        lines += ['- '+s for s in report['limits']]
        (stage/'README.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,stage/Path(__file__).name)
        fresh.json_write(stage/'manifest.json',{'schema':report['schema']+'-manifest',
                         'files':{str(p.relative_to(stage)):fresh.file_binding(p)['sha256'] for p in sorted(stage.iterdir()) if p.is_file()}})
        require(not output.exists(),'Comparison output appeared during build');stage.rename(output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'artifacts/modebench_fresh_concentration_20260912/comparison_reference/reference.json')
    parser.add_argument('--fresh-report',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify-historical-only',action='store_true')
    args=parser.parse_args()
    if args.verify_historical_only:
        _,_,audit=historical_endpoints(read_published(args.reference));print(json.dumps(audit,indent=2));return
    require(args.fresh_report is not None and args.output is not None,'--fresh-report and --output required')
    require(not args.output.exists(),'Immutable comparison output already exists')
    print(publish(build_report(args.reference,args.fresh_report),args.output))


if __name__=='__main__':main()
