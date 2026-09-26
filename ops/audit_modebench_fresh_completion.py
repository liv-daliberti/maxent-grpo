#!/usr/bin/env python3
"""Fail-closed, offline completion audit for the registered fresh campaign.

No generation, verifier calls, scheduler operations, or model-payload hashing.
Default output is JSON on stdout; --output publishes only a successful audit.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import analyze_modebench_fresh_concentration as analysis
import analyze_modebench_fresh_panel as panel
import compare_modebench_fresh_concentration as comparison
import evaluate_modebench_fresh_concentration as collector
import launch_modebench_fresh_concentration as launcher
from restore_modebench_fresh_concentration import atomic_new, digest

BASE=launcher.BASE
PLAN_SHA256='63b18cbd4917914766ec6626a6077aac126a23862d5803fb725ab6d2b8c7c69f'
SEEDS={'qwen05b':tuple(range(43,48)),'falcon1b':tuple(range(55,60)),'qwen3b':tuple(range(70,75))}
DOMAINS=('graph_coloring','pantry_plan')
METHODS=panel.METHODS
require=analysis.require


class Incomplete(ValueError):
    def __init__(self,missing):
        super().__init__('Required completion artifacts are missing')
        self.missing=missing


def binding(path):
    return {'path':str(Path(path).resolve()),'sha256':digest(path)}


def verify_binding(reference,expected_path=None):
    path=Path(reference['path'])
    path=path if path.is_absolute() else ROOT/path
    if expected_path is not None:
        require(path.resolve()==Path(expected_path).resolve(),'Bound artifact path differs')
    require(path.is_file() and digest(path)==reference['sha256'],f'Bound artifact changed: {path}')
    return path


def read_publication(directory,required):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    files=manifest['files']
    require(set(required)<=set(files),'Required files absent from publication manifest')
    for name,expected in files.items():
        rel=Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe publication manifest path')
        require((directory/rel).is_file() and digest(directory/rel)==expected,'Published artifact hash differs')
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual==set(files)|{'manifest.json'},'Unlisted or missing publication file')
    if 'source_report_sha256' in manifest:
        require(manifest['source_report_sha256']==digest(directory/'report.json'),'Manifest report binding differs')
    require(manifest.get('status','complete')=='complete','Publication is incomplete')
    return manifest


def stable(value):
    if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k!='created_at_utc'}
    if isinstance(value,list):return [stable(v) for v in value]
    return value


def reconstructed_equal(left,right):
    """Allow arithmetic rounding only; identities, types and counts stay exact."""
    if type(left) is not type(right):return False
    if isinstance(left,float):return math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-14)
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(reconstructed_equal(left[k],right[k]) for k in left)
    if isinstance(left,list):
        return len(left)==len(right) and all(reconstructed_equal(x,y) for x,y in zip(left,right))
    return left==right


def validate_scope(plan):
    require(len(plan['code_sha256'])==112,'Exactly 112 frozen code bindings required')
    require(plan['prompts_per_task']==128 and plan['draws_per_prompt']==64,'Prompt/draw budget differs')
    panel.validate_panel_inventory(plan)
    expected={(scale,domain,method,seed) for scale,seeds in SEEDS.items() for domain in DOMAINS
              for method in ('initial',*METHODS) for seed in seeds}
    actual=[]
    for task in plan['tasks']:
        initial=task['checkpoint_stage']=='initial'
        require(task['level']==1,'Only the registered Level-1 panel is admitted')
        require((initial and task['method']=='initial' and task['training_seed'] is None)
                or (not initial and task['method'] in METHODS and task.get('eval_replica_id') is None),
                'Initial Monte Carlo replica or terminal training-seed metadata differs')
        actual.append((task['model_scale'],task['domain'],task['method'],panel.seed_index(task)))
    require(len(actual)==len(set(actual))==150 and set(actual)==expected,'Exact registered task/seed inventory differs')
    require(sum(t['checkpoint_stage']=='terminal' for t in plan['tasks'])==120,'Exactly 120 terminal tasks required')
    initial_sets=[]
    for scale in SEEDS:
        hashes={collector.sha(t['files']) for t in plan['tasks'] if t['model_scale']==scale and t['checkpoint_stage']=='initial'}
        require(len(hashes)==1,'Initial replicas and domains must share the same initial weight set per scale')
        initial_sets.extend(hashes)
    require(len(set(initial_sets))==3,'Exactly three distinct initial weight sets required')


def preflight(base,plan):
    required=['fresh_panel/report.json','fresh_panel/manifest.json','collection_and_analysis_complete.json',
              'audits/final_runtime_hardware.json','reanalysis/report.json','reanalysis/manifest.json',
              'reanalysis_reconciliation/report.json','reanalysis_reconciliation/manifest.json',
              'comparison_reference/reference.json','comparison_reference/manifest.json',
              'fresh_vs_retrospective/report.json','fresh_vs_retrospective/manifest.json',
              'fresh_figures/manifest.json','INTERPRETATION.md']
    missing=[str(base/name) for name in required if not (base/name).is_file()]
    missing += [str(Path(plan['output_root'])/t['task_id']/'result.json') for t in plan['tasks']
                if not (Path(plan['output_root'])/t['task_id']/'result.json').is_file()]
    if missing:raise Incomplete(missing)


def contrast_identity(c,*,fresh):
    fields=('model_scale','domain','level','wording','contrast') if fresh else ('grading','wording','level','domain','contrast')
    return tuple(c[k] for k in fields)


def validate_contrasts(contrasts,*,fresh):
    if fresh:
        expected={(scale,domain,1,'original',right+'_minus_'+left)
                  for scale in SEEDS for domain in DOMAINS for left,right in panel.PAIRS}
    else:
        expected={(grade,wording,level,domain,right+'_minus_'+left)
                  for grade in ('strict','normalized_secondary') for wording in ('original','neutral')
                  for level in (2,3) for domain in analysis.EXPECTED_SEEDS for left,right in analysis.CONTRASTS}
    actual=[contrast_identity(c,fresh=fresh) for c in contrasts]
    require(len(actual)==len(set(actual))==len(expected) and set(actual)==expected,'Contrast inventory is incomplete or duplicated')
    for c in contrasts:
        seeds=SEEDS[c['model_scale']] if fresh else analysis.EXPECTED_SEEDS[c['domain']]
        records=c['seeds']
        require(len(records)==len(seeds) and {r['training_seed'] for r in records}==set(seeds),'Registered comparison seeds differ')
        prompts=128 if fresh else 16
        require(all(r['expected_prompts']==prompts and len(r['prompts'])==prompts
                    and len({p['prompt_id'] for p in r['prompts']})==prompts for r in records),
                'Full fixed prompt cohort or missingness records were dropped')
        if c['left_method']=='initial':
            require(c['summary']['independent_initial_checkpoints']==1,'Initial reference incorrectly counts independent checkpoints')
            if fresh:
                require(c['summary']['initial_sampling_replicas']==5 and c['summary']['initial_weights_shared'] is True,
                        'Fresh initial Monte Carlo replica interpretation differs')


def validate_restoration_metadata(plan):
    receipts=[]
    directory=ROOT/'var/cache/modebench_fresh_concentration_20260912/receipts'
    for task in plan['tasks']:
        if task['checkpoint_stage']!='terminal':continue
        path=directory/(task['task_id']+'.json')
        require(path.is_file(),'Missing terminal checkpoint restoration receipt')
        receipt=json.loads(path.read_text())
        require(receipt['model_path']==task['model_path'] and receipt['files']==task['files'],
                'Restoration metadata differs from frozen checkpoint manifest')
        require(receipt.get('original_run_modified') is False,'Restoration must preserve original runs')
        for field in ('archive_manifest','completion_receipt'):
            if receipt.get(field):verify_binding(receipt[field])
        receipts.append(binding(path))
    return {'receipts':receipts,'terminal_receipts':len(receipts),'model_payload_bytes_rehashed_by_this_audit':0,
            'validation_basis':'The frozen collector calls validate_file_manifest(task, hash_files=True) before engine initialization. Every authenticated task initialization therefore follows full loader-input size/SHA validation. This audit checks restoration and task manifest bindings without rereading model payloads.',
            'collector_source':binding(collector.__file__),
            'limitation':'This is provenance verification of the frozen collector execution, not an independent attestation of GPU memory or a new check of current weight-file bytes.'}


def audit_campaign(base=BASE):
    base=Path(base).resolve();pp=base/'plan.json'
    require(digest(pp)==PLAN_SHA256,'Original frozen plan SHA differs')
    plan=json.loads(pp.read_text());validate_scope(plan);preflight(base,plan)
    registration=json.loads((base/'analysis_registration.json').read_text())
    verify_binding(registration,base/'ANALYSIS_PLAN.md')
    require(registration['collection_started'] is False,'Analysis registration did not precede collection')
    publications={}
    for directory,required in [('fresh_panel',['report.json','contrasts.csv','seed_contrasts.csv','prompt_contrasts.csv']),
                               ('reanalysis',['report.json','contrasts.csv','seed_contrasts.csv','prompt_contrasts.csv']),
                               ('reanalysis_reconciliation',['report.json','reconciliation.csv']),
                               ('comparison_reference',['reference.json']),
                               ('fresh_vs_retrospective',['report.json','comparisons.csv','seed_overlaps.csv','common_prompt_contrasts.csv']),
                               ('fresh_figures',['training_concentration.pdf','training_concentration.png','replay_concentration.pdf','replay_concentration.png'])]:
        read_publication(base/directory,required);publications[directory]=binding(base/directory/'manifest.json')
    submissions=[json.loads(p.read_text()) for p in (base/'slurm/submissions').glob('*.json')]
    require(submissions and all(r['plan_sha256']==PLAN_SHA256 for r in submissions),'Submission plan binding differs')
    resolved={r['intent_id'] for r in submissions}
    require(not [p for p in (base/'slurm/intents').glob('*.json') if p.stem not in resolved],'Unresolved scheduler submission intent')
    amendment=launcher.load_execution_amendment(base,plan,submissions)
    require(amendment is not None,'Falcon hardware amendment is required')
    effective=[i for r in submissions for i in launcher.effective_indices(r,amendment)]
    require(len(effective)==len(set(effective))==150 and set(effective)==set(range(150)),
            'Exactly one nonsuperseded submission per registered task required')
    for receipt in submissions:
        require(0<receipt['max_concurrent_owned_gpus']<=8,'Submission exceeds campaign GPU limit')
        require(receipt['worker_sha256']==digest(base/'slurm/collect.sh'),'Submitted worker source changed')
    hardware={'plan_sha256':PLAN_SHA256,
              'tasks':[launcher.assert_runtime_hardware(t,amendment,plan['output_root']) for t in plan['tasks']],
              'execution_amendment':amendment['reference']}
    require(json.loads((base/'audits/final_runtime_hardware.json').read_text())==hardware,'Final hardware audit differs')
    restoration=validate_restoration_metadata(plan)
    # Full adapter authenticates all 1,228,800 saved slots, global schedules,
    # cached grade/code bindings, batches, flat files, and checkpoint identities.
    rebuilt=panel.build_report(pp)
    published=json.loads((base/'fresh_panel/report.json').read_text())
    require(reconstructed_equal(stable(rebuilt),stable(published)),'Published fresh report differs from authenticated reconstruction')
    validate_contrasts(published['contrasts'],fresh=True)
    old=json.loads((base/'reanalysis/report.json').read_text())
    require(old.get('status')=='complete' and old.get('generation_calls')==0 and old.get('verifier_calls')==0
            and old.get('old_artifacts_modified') is False,'Old reanalysis completion contract differs')
    validate_contrasts(old['contrasts'],fresh=False)
    # Reuse the serial-graded old-data adapter; it does not invoke a verifier.
    with redirect_stdout(io.StringIO()):old_rebuilt=analysis.build_report(Path(old['input_campaign']))
    for key in ('contrasts','scope','correction_audit','checkpoint_sources','design_sources','plan_source','estimand'):
        require(reconstructed_equal(old[key],old_rebuilt[key]),f'Old fresh-output reanalysis reconstruction differs: {key}')
    reconciliation=json.loads((base/'reanalysis_reconciliation/report.json').read_text())
    verify_binding(reconciliation['source_report'],base/'reanalysis/report.json')
    require(reconciliation['status']=='complete' and reconstructed_equal(reconciliation['reconciliations'],
            [analysis.population_reconciliation(c) for c in old['contrasts']]),
            'Old population/weighting reconciliation differs or is incomplete')
    compared=json.loads((base/'fresh_vs_retrospective/report.json').read_text())
    comparison_rebuilt=comparison.build_report(base/'comparison_reference/reference.json',base/'fresh_panel/report.json')
    require(reconstructed_equal(stable(compared),stable(comparison_rebuilt)),'Published old/new comparison differs from reconstruction')
    require(compared['comparison_count']==len(compared['comparisons'])==36
            and {contrast_identity(c,fresh=True) for c in compared['comparisons']}==
                {contrast_identity(c,fresh=True) for c in published['contrasts']},'Exact 36 old/new comparisons required')
    figures=json.loads((base/'fresh_figures/manifest.json').read_text())
    verify_binding({'path':figures['report'],'sha256':figures['report_sha256']},base/'fresh_panel/report.json')
    require(figures['contrasts_plotted']==36,'Figures must include all 36 registered contrasts')
    marker=json.loads((base/'collection_and_analysis_complete.json').read_text())
    require(marker['plan_sha256']==PLAN_SHA256 and marker['tasks']==150 and marker['response_slots']==1228800,
            'Final collection/analysis marker inventory differs')
    verify_binding({'path':marker['report'],'sha256':marker['report_sha256']},base/'fresh_panel/report.json')
    verify_binding(marker['runtime_hardware_audit'],base/'audits/final_runtime_hardware.json')
    require((base/'INTERPRETATION.md').stat().st_size>0,'Interpretation file is empty')
    return {'schema':'modebench-fresh-completion-audit-v1','status':'complete',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'plan':binding(pp),
            'analysis_plan':binding(base/'ANALYSIS_PLAN.md'),'auditor':binding(__file__),
            'scope':{'tasks':150,'terminal_tasks':120,'initial_monte_carlo_tasks':30,'initial_weight_sets':3,
                     'model_scales':list(SEEDS),'domains':list(DOMAINS),'trained_methods':list(METHODS),
                     'seeds_by_scale':SEEDS,'prompts_per_task':128,'draws_per_prompt':64,'response_slots':1228800,
                     'frozen_code_bindings':112,'fresh_contrasts':36,'old_strict_contrasts':36,
                     'old_normalized_sensitivity_contrasts':36,'old_reconciliations':72,'old_new_comparisons':36,'figure_contrasts':36},
            'publications':publications,'runtime_hardware_audit':binding(base/'audits/final_runtime_hardware.json'),
            'execution_amendment':amendment['reference'],'superseded_attempts':amendment['superseded_attempts'],
            'scheduler_scope':'Exactly one effective submission per task after explicit supersession. Failed/cancelled historical attempts remain preserved; this does not claim all historical jobs completed.',
            'restoration':restoration,'collection_marker':binding(base/'collection_and_analysis_complete.json'),
            'interpretation':{'source':binding(base/'INTERPRETATION.md'),'validation':'nonempty file presence and hash only',
                              'scientific_semantics_validated':False,'human_or_root_scientific_review_required':True},
            'statistical_reconstruction_tolerance':{'relative':1e-12,'absolute':1e-14,'scope':'Recomputed statistical floats only; hashes, identities, counts, and authenticated response/batch equality remain exact.'},
            'generation_calls':0,'verifier_calls':0,'job_submissions':0,
            'grading_scope':'Cached collector grades and old serial correction receipts authenticated; no independent regrading is performed by this audit.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:report=audit_campaign(args.base)
    except (ValueError,KeyError,TypeError,OSError) as error:
        report={'schema':'modebench-fresh-completion-audit-v1','status':'incomplete' if isinstance(error,Incomplete) else 'invalid',
                'error':str(error),'generation_calls':0,'verifier_calls':0,'job_submissions':0}
        if isinstance(error,Incomplete):report['missing']=error.missing
        print(json.dumps(report,sort_keys=True,indent=2));return 2
    if args.output:atomic_new(args.output,report)
    print(json.dumps(report,sort_keys=True,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
