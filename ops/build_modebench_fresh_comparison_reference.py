#!/usr/bin/env python3
"""Freeze all matching retrospective contrasts without reading new outcomes."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'artifacts/modebench_fresh_concentration_20260912'
SOURCE=ROOT/'paper/results/conditional_concentration_20260912.json'
EXPECTED_SHA='22cb288ebcca55bbd23562bb4e170655fb2f19122106d794ee16c4e2ac0811a7'
METHODS=('drgrpo','replay_drgrpo','maxrl','replay_maxrl')


def binding(path):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def require(condition,message):
    if not condition:raise ValueError(message)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def task_id(row):
    reference=json.loads(row['answer']) if isinstance(row['answer'],str) else row['answer']
    return hashlib.sha256(json.dumps({'prompt':row['problem'],'reference':reference},sort_keys=True,
                                    separators=(',',':'),allow_nan=False).encode()).hexdigest()


def direction(x):
    return 'undefined' if x is None else 'positive' if x>0 else 'negative' if x<0 else 'zero'


def build():
    require(binding(SOURCE)['sha256']==EXPECTED_SHA,'Authoritative retrospective source changed')
    source=json.loads(SOURCE.read_text())
    manifest_path=SOURCE.with_suffix('')/'build_manifest.json'
    require(json.loads(manifest_path.read_text())['source']['sha256']==EXPECTED_SHA,'Rendered source binding differs')
    inventory_path=CAMPAIGN/'checkpoint_inventory.json';inventory=json.loads(inventory_path.read_text())
    audit_path=CAMPAIGN/'prompt_population_audit.json'
    require(json.loads(audit_path.read_text())['all_cells_match_exact_domain_representative'],'Historical prompt populations differ')
    prompt_ids={};prompt_bindings={}
    for domain in ('graph_coloring','pantry_plan'):
        path=CAMPAIGN/'prompts'/(domain+'_raw_source.jsonl')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows)==len({r['prompt_id'] for r in rows})==128,'Wrong fixed prompt inventory')
        require(all(task_id(r)==r['prompt_id'] for r in rows),'Canonical task identity differs')
        prompt_ids[domain]={r['prompt_id']:r['row_index'] for r in rows};prompt_bindings[domain]=binding(path)
    available={(r['model_scale'],r['domain'],r['method'],r['training_seed']):r for r in inventory['records']}
    blocks=[]
    for position,b in enumerate(source['blocks']):
        if b['level']!='level1' or b['domain'] not in prompt_ids:continue
        if b['kind']=='before_after' and b['method'] in METHODS:left,right='initial',b['method']
        elif b['kind']=='replay_effect' and b['method'] in ('drgrpo','maxrl'):left,right=b['method'],'replay_'+b['method']
        else:continue
        summary=b['summaries']['distinct_streams'];seeds=[]
        require(summary['registered_n']==5,'Unexpected registered seed count')
        for seed in b['admitted_seeds']:
            record=b['per_seed'].get(str(seed),{}).get('distinct_streams')
            if record:
                require(record['n_total']==128 and len(record['eligible_ids'])==record['n_eligible'],'Population fields inconsistent')
                require(set(record['eligible_ids'])<=set(prompt_ids[b['domain']]),'Old eligible task absent from new cohort')
                require(summary['values'].get(str(seed))==record['delta']['collision'],'Seed effect differs from summary')
            methods=(right,) if left=='initial' else (left,right)
            seeds.append({'paired_seed':seed,'status':'defined' if record and record['delta']['collision'] is not None else 'undefined_or_source_unavailable',
                          'historical_primary':record,'source_issues':[x for x in b['issues'] if x.get('seed')==seed],
                          'terminal_cell_ids':{m:available[b['scale'],b['domain'],m,seed]['cell_id'] for m in methods}})
        blocks.append({'model_scale':b['scale'],'domain':b['domain'],'level':1,'wording':'original',
                       'contrast':right+'_minus_'+left,'left_method':left,'right_method':right,
                       'source_block_pointer':f'/blocks/{position}','source_kind':b['kind'],'source_method':b['method'],
                       'primary':summary,'primary_direction':direction(summary['mean']),
                       'sensitivities':{k:b['summaries'][k] for k in ('orientation0','orientation1','first_k8','intact_k8')},
                       'fixed_across_seed_population':{k:v for k,v in b['fixed_across_seed_population']['distinct_streams'].items() if k!='per_seed'},
                       'registered_seeds':b['admitted_seeds'],'per_seed':seeds})
    pairs={('initial',m) for m in METHODS}|{('drgrpo','replay_drgrpo'),('maxrl','replay_maxrl')}
    require(len(blocks)==36 and {(b['model_scale'],b['domain'],b['left_method'],b['right_method']) for b in blocks}==
            {(s,d,a,b) for s in ('qwen05b','falcon1b','qwen3b') for d in prompt_ids for a,b in pairs},'Exact36contrast mapping incomplete')
    terminals=[]
    for r in inventory['records']:
        require(r['logged_evaluation_step']==3072 and r['terminal_export_step']==3073,'Unexpected endpoint step mapping')
        terminals.append({k:r[k] for k in ('cell_id','model_scale','domain','method','training_seed','logged_evaluation_step',
                                         'terminal_export_step','model_path','completion_receipt','source_eval_config')})
    return {'schema':'modebench-fresh-retrospective-comparison-reference-v1','status':'complete',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'new_collection_outcomes_read':False,
            'generation_calls':0,'verifier_calls':0,'old_artifacts_modified':False,
            'sources':{'retrospective_result':binding(SOURCE),'rendered_build_manifest':binding(manifest_path),
                       'checkpoint_inventory':binding(inventory_path),'prompt_population_audit':binding(audit_path),
                       'prompt_populations':prompt_bindings,'builder':binding(__file__)},
            'old_sampling_audit':source['stream_source_audit'],
            'old_cache_binding':{k:source['cache'][k] for k in ('path','sha256','binding_path','binding_sha256')},
            'matching':{'block_key':['model_scale','domain','level','wording','contrast'],
                        'seed_key':'historical paired seed = new terminal training_seed = new initial integer eval_replica_id',
                        'prompt_key':'SHA256(canonical JSON {prompt:raw problem, reference:parsed reference}); preserved in all256raw prompt records',
                        'new_initial_training_seed':None,
                        'initial_reference_change':'Historical before/after uses each method/run own recorded initial evaluation; new methods share an independently sampled initial replica per paired index, with identical base weights.',
                        'weighting':'Both primary analyses equal-weight promptwise collision on joint R>=2, then equal-weight defined seed contrasts.',
                        'draw_budget_change':'Historical32saved positions yield11earliest nominal-stream representatives; new64distinct child streams per prompt, disjoint across tasks.',
                        'uncertainty':'Nominal paired Student-t for five defined seed effects only; partial blocks descriptive.',
                        'units':'Collision/correctness are probabilities; multiply by100for percentage points. Distinct/extra modes are counts.'},
            'metric_mapping':{'mean8':'new mean_correct uses all fresh64; old mean8 uses four intact K8groups.',
                              'pass8':'new pass8_rarefied is without-replacement rarefaction of fresh64; old pass8 averages four original K8groups.',
                              'distinct8':'new distinct8_rarefied matches budget concept; fresh finite-pool rarefaction versus old four groups.',
                              'extra8':'new extra8_rarefied equals distinct8_rarefied minus pass8_rarefied.'},
            'checkpoint_step_semantics':inventory['checkpoint_step_semantics'],'terminal_references':terminals,
            'contrasts':blocks,'prompt_identity_to_row_index':prompt_ids}


GUIDE='''# Comparing the new Level-1 panel with the retrospective results

This reference was extracted without reading new collection outcomes. It retains all 36 matching contrasts at full precision, every historical seed's eligible prompt IDs and endpoint metrics, disjoint-orientation sensitivities, and source hashes. `contrasts.csv` is the compact index; `reference.json` includes populations and provenance.

Join model scale, domain, and right-minus-left method contrast. Match the historical seed to new terminal `training_seed` and initial `eval_replica_id`; initial `training_seed` is null. Prompt IDs preserve the historical raw-problem-plus-parsed-reference hash, and all 128 prompts per domain match. Dr.GRPO and MaxRL replay pairs have distinct method identities. GRPO is outside this new panel.

The primary estimator and weighting match: promptwise correct-key collision on each comparison's joint R>=2 prompts, then equal weights for defined seed effects. Sampling changes from eleven earliest representatives of overlapping historical nominal streams to 64 fresh, disjoint child streams. The larger pool can change both estimates and eligibility. Historical initial outputs came from each method/run's own saved evaluation; new methods share one initial sampling replica within each paired index. Five replicas use the same initial weights.

Compare every new contrast with its old estimate, per-seed values, eligible counts and intervals. Retain all directions and undefined cases. Explain population changes with overlap counts and a common-population sensitivity. When subsetting an old eligible set, reconstruct old prompt values from its bound cache; do not reuse its full-set mean for a smaller subset. Crossing zero does not establish no effect; comparing whether two intervals cross zero does not test whether the studies differ.

Old collision uses eleven nominal stream representatives, while old mean@8, pass@8, distinct@8 and extra@8 retain four original eight-output groups. New correctness uses fresh64; its eight-output breadth values are exact finite-pool rarefactions. These share scientific targets/budgets but use different samples. Read correctness on selected prompts separately from full128-prompt outcomes. Never pool old/new responses, average their intervals, or algebraically subtract contrasts defined on different selected populations.

The historical final evaluation is logged at step3072; the archived export directory is step_03073. The preserved E118 Qwen3B loop evaluates the final update, increments its bookkeeping counter, and exports without another policy update; its duplicate-terminal check binds unchanged global_step. This authenticates that naming offset in the preserved implementation. Other historical source roots are unavailable locally. The reference retains every checkpoint's completion and export provenance without asserting that all historical runs used the same implementation or runtime. Frozen current code and runtime receipts authenticate the new evaluation; they do not restore missing historical dependencies.

No new outcome is assumed here. The new evidence may strengthen, qualify or disagree with historical observations. Decompose disagreements into population, sampling/reference and implementation differences before revising the scientific claim. Concentration reduction need not hold correctness fixed, recover unseen modes, or identify a causal diversity benefit.
'''


def main():
    output=CAMPAIGN/'comparison_reference';require(not output.exists(),'Immutable reference already exists')
    report=build()
    with tempfile.TemporaryDirectory(prefix='.comparison-reference-',dir=CAMPAIGN) as temp:
        stage=Path(temp);write_json(stage/'reference.json',report);rows=[]
        for b in report['contrasts']:
            s=b['primary'];ci=s['ci95']
            rows.append({'model_scale':b['model_scale'],'domain':b['domain'],'contrast':b['contrast'],
                         'source_pointer':b['source_block_pointer'],'mean':s['mean'],'ci95_lower':ci[0] if ci else None,
                         'ci95_upper':ci[1] if ci else None,'defined_seeds':s['n'],'registered_seeds':s['registered_n'],
                         'direction':b['primary_direction'],'coverage_range':json.dumps(s.get('coverage_range')),
                         'eligible_counts':json.dumps(s.get('eligible_counts',{}),sort_keys=True),
                         'orientation0_mean':b['sensitivities']['orientation0']['mean'],
                         'orientation1_mean':b['sensitivities']['orientation1']['mean'],
                         'same_population_correctness_delta':s['original_metrics_on_eligible']['mean8']['mean']})
        with (stage/'contrasts.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (stage/'README.md').write_text(GUIDE);shutil.copy2(__file__,stage/Path(__file__).name)
        write_json(stage/'manifest.json',{'schema':report['schema']+'-manifest',
                   'files':{str(p.relative_to(stage)):binding(p)['sha256'] for p in sorted(stage.iterdir()) if p.is_file()}})
        require(not output.exists(),'Reference appeared during build');stage.rename(output)
    print(output)


if __name__=='__main__':main()
