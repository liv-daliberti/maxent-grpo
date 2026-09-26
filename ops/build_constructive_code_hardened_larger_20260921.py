#!/usr/bin/env python3
"""Build the larger strict source slate without repeating identical initial audits.

Reuses only byte-bound 24-program replay ledgers with exactly the same executable
verification contract. All other source-admitted tasks get fresh full replays.
The initial dataset and its loader remain immutable.
"""
from build_constructive_code_hardened_20260921 import *
import build_constructive_code_hardened_20260921 as h


def verification_contract(folder, record):
    inputs = [json.loads(line)['stdin'] for line in gzip.open(folder/record['suite_file'],'rt')]
    if base.digest(folder/record['suite_file']) != record['suite']['compressed_jsonl_sha256']:
        raise ValueError('suite file drift before replay reuse')
    if base.digest(folder/'checker.cpp') != record['checker_sha256'] or base.digest(folder/'py3_replays.jsonl') != record['references']['jsonl_sha256']:
        raise ValueError('checker/reference byte drift before replay reuse')
    return {
        'problem_id':record['source_problem_id'],
        'suite_id':record['suite_id'], 'suite_sha256':record['suite']['suite_sha256'],
        'inputs_in_order_sha256':base.canonical_hash(inputs),
        'checker_sha256':record['checker_sha256'],
        'reference_ledger_sha256':record['references']['jsonl_sha256'],
        'limits':record['limits'],
        'statement_sha256':record['statement_sha256'],
        'witness_family':record['witness_family'],
        'canonicalizer_sources':h.relocated_source_identity(h.source_code_identity()),
        'verifier_support_sources':h.verifier_support_identity(),
    }


def reuse_initial(args):
    quality = h.validate_quality({'slate_root':str(args.initial_root)})
    source = json.loads((args.initial_root/'manifest.json').read_text())
    target = json.loads((args.output/'manifest.json').read_text())
    by_id = {r['source_problem_id']:r for r in source['tasks']}
    quality_by_id = {r['task_id']:r for r in quality['tasks']}
    reused = []
    for summary in target['tasks']:
        problem = summary['source_problem_id']
        if problem not in quality_by_id:
            continue
        origin = args.initial_root/by_id[problem]['relative_path']
        folder = args.output/summary['relative_path']
        old = json.loads((origin/'task.json').read_text()); new = json.loads((folder/'task.json').read_text())
        row = quality_by_id[problem]
        if old['task_record_sha256'] != row['task_record_sha256'] or base.digest(origin/'admission_audit.json') != row['admission_audit_sha256'] or base.digest(origin/'audit_replays.jsonl') != row['reference_replays_sha256']:
            raise ValueError('initial source audit reuse binding drift')
        contract = verification_contract(origin,old)
        if contract != verification_contract(folder,new):
            raise ValueError('verification contract differs; replay reuse forbidden')
        h.compare_relocated_sources(json.loads((origin/'admission_audit.json').read_text())['canonicalizer_source_files'],h.source_code_identity())
        audit = json.loads((origin/'admission_audit.json').read_text())
        if audit['status'] != row['status']:
            raise ValueError('initial task status and quality receipt disagree')
        shutil.copy2(origin/'audit_replays.jsonl',folder/'audit_replays.jsonl')
        audit['audit_method'] = 'reused_exact_hardened_verification_contract_24_reference_replays'
        audit['replay_reuse_origin'] = {
            'root':str(args.initial_root.resolve()),
            'manifest_sha256':base.digest(args.initial_root/'manifest.json'),
            'hardening_quality_sha256':base.digest(args.initial_root/'hardening_quality.json'),
            'task_record_sha256':old['task_record_sha256'],
            'admission_audit_sha256':base.digest(origin/'admission_audit.json'),
            'reference_replays_sha256':base.digest(origin/'audit_replays.jsonl'),
            'verification_contract':contract,
            'verification_contract_sha256':base.canonical_hash(contract),
            'metadata_changes_only':['split','source_manifest','original_task_record'],
            'original_status':audit['status'],
        }
        base.write_json(folder/'admission_audit.json',audit)
        new.update({'admission_status':'admitted' if audit['status']=='pass' else 'quarantined','admission_audit_sha256':base.digest(folder/'admission_audit.json')})
        new['task_record_sha256']=base.canonical_hash({k:v for k,v in new.items() if k!='task_record_sha256'})
        base.write_json(folder/'task.json',new)
        reused.append(problem)
    base.write_json(args.output/'replay_reuse_manifest.json',{'schema':'constructive-code-exact-reference-replay-reuse-v1','initial_manifest_sha256':base.digest(args.initial_root/'manifest.json'),'initial_quality_sha256':base.digest(args.initial_root/'hardening_quality.json'),'reused_problem_ids':reused,'policy_model_outputs_used':False})
    return reused


def audit_remaining(args,reused):
    manifest=json.loads((args.output/'manifest.json').read_text())
    pending=[p for p in manifest['hardening_source_problem_ids'] if p not in reused]
    def worker(problem):
        cmd=[sys.executable,str(Path(h.__file__).resolve()),'--phase','audit-one','--output',str(args.output.resolve()),'--task',problem,'--image',str(args.image.resolve()),'--runtime-root',str(args.runtime_root.resolve()),'--work-root',str(args.work_root.resolve()),'--reference-workers',str(args.reference_workers)]
        completed=subprocess.run(cmd,capture_output=True,text=True)
        print(completed.stdout,flush=True,end='')
        if completed.returncode:
            raise RuntimeError(f'{problem}:audit worker failed:{completed.stderr[-3000:]}')
    with ThreadPoolExecutor(max_workers=args.task_workers) as pool:
        for future in as_completed([pool.submit(worker,p) for p in pending]):
            future.result()
    return pending


def finalize(args):
    manifest=json.loads((args.output/"manifest.json").read_text());pending=manifest["hardening_source_problem_ids"]
    quality=[]
    for summary in manifest['tasks']:
        if summary['source_problem_id'] not in pending:continue
        folder=args.output/summary['relative_path'];row=json.loads((folder/'task.json').read_text());audit=json.loads((folder/'admission_audit.json').read_text());origin_hardening=row['hardening']
        summary.update({'status':row['admission_status'],'task_record_sha256':row['task_record_sha256'],'admission_audit_sha256':row['admission_audit_sha256']})
        quality.append({'task_id':row['source_problem_id'],'status':audit['status'],'audit_method':audit['audit_method'],'task_record_sha256':row['task_record_sha256'],'admission_audit_sha256':row['admission_audit_sha256'],'reference_replays_sha256':audit.get('replays_sha256'),'reference_ledger_sha256':row['references']['jsonl_sha256'],'positive_replays':audit.get('positive_replays',0),'negative_replays':audit.get('negative_replays',0),'positive_accepted':audit.get('positive_accepted',0),'negative_rejected':audit.get('negative_rejected',0),'tpr':audit.get('tpr'),'tnr':audit.get('tnr'),'checker_sha256':row['checker_sha256'],'statement_sha256':row['statement_sha256'],'original_checker_sha256':origin_hardening['checker_sha256'],'original_statement_sha256':origin_hardening['statement_sha256'],'original_task_record_sha256':origin_hardening['original_task_record_sha256'],'original_suite_id':origin_hardening['original_suite_id'],'original_suite_sha256':origin_hardening['original_suite']['suite_sha256'],'original_suite_file_sha256':origin_hardening['original_suite']['compressed_jsonl_sha256'],'hardened_suite_id':row['suite_id'],'hardened_suite_sha256':row['suite']['suite_sha256'],'hardened_suite_file_sha256':row['suite']['compressed_jsonl_sha256'],'original_input_count':origin_hardening['original_input_count'],'hardened_input_count':row['suite']['test_count'],'added_input_sha256s':[r['input_sha256'] for r in origin_hardening['added_inputs']],'original_inputs_preserved_in_order':audit.get('original_inputs_preserved_in_order',False),'original_checker_unchanged':audit.get('original_checker_unchanged',False),'original_prompt_unchanged':audit.get('original_prompt_unchanged',False),'violations':audit['violations']})
    admitted=[r['task_id'] for r in quality if r['status']=='pass']
    gate={'schema':QUALITY_SCHEMA,'status':'pass' if admitted else 'fail','hardening_schema':SCHEMA,'policy':POLICY,'fixed_probe_manifest_sha256':FIXTURE_SHA256,'source_manifest_sha256':manifest['source_manifest_sha256'],'source_candidates':len(quality),'admitted':len(admitted),'quarantined':len(quality)-len(admitted),'admitted_problem_ids':admitted,'tasks':quality,'canonicalizer_source_files':source_code_identity(),'builder_sha256':base.digest(Path(h.__file__)),'loader_guard_sha256':base.digest(Path(h.__file__)),'verifier_support_files':verifier_support_identity(),'pinned_testlib_sha256':require_pinned_testlib()}
    gate['canonicalizer_source_files']=h.relocated_source_identity(gate['canonicalizer_source_files'])
    gate['coordinator_source_sha256']=base.digest(Path(__file__))
    gate['replay_reuse_manifest_sha256']=base.digest(args.output/'replay_reuse_manifest.json')
    base.write_json(args.output/'hardening_quality.json',gate)
    manifest.update({'status':'audited','admitted_problem_ids':admitted,'tasks_sha256':base.canonical_hash(manifest['tasks']),'hardening_quality_sha256':base.digest(args.output/'hardening_quality.json')})
    for field in ('default_training_problem_ids','validation_problem_ids','admitted_test_problem_ids'):
        if field in manifest:manifest[field]=[p for p in manifest[field] if p in admitted]
    if 'pool_counts' in manifest:
        manifest['pool_counts']={split:sum(r['status']=='admitted' and r.get('split')==split for r in manifest['tasks']) for split in ('train','validation','test')}
        manifest['larger_ready_source_gate']=manifest['pool_counts']['train']>=32 and manifest['pool_counts']['validation']==2 and manifest['pool_counts']['test']>=16
    base.write_json(args.output/'manifest.json',manifest)
    base.write_json(args.output/'admission_summary.json',{'schema_version':SCHEMA,'policy':POLICY,'candidates':len(quality),'admitted':len(admitted),'tasks':quality,'hardening_quality_sha256':manifest['hardening_quality_sha256']})
    print(json.dumps({'status':gate['status'],'admitted':len(admitted),'source_candidates':len(quality),'manifest_sha256':base.digest(args.output/'manifest.json'),'hardening_quality_sha256':manifest['hardening_quality_sha256']},sort_keys=True),flush=True)



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--initial-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixture',type=Path,default=DEFAULT_FIXTURE)
    p.add_argument('--resume-materialized',action='store_true')
    p.add_argument('--image',type=Path,default=base.ROOT/'var/images/python-3.10-slim-c1e4e6c01eb4.sqsh')
    p.add_argument('--runtime-root',type=Path,default=Path('/tmp/constructive_wider_20260921/runtime'))
    p.add_argument('--work-root',type=Path,default=Path('/tmp/constructive_hardened_larger_20260921'))
    p.add_argument('--task-workers',type=int,default=4)
    p.add_argument('--reference-workers',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.task_workers<=4 or not 1<=args.reference_workers<=8:
        p.error('bounded audit worker counts required')
    if args.resume_materialized:
        existing=json.loads((args.output/'manifest.json').read_text())
        if existing['status']!='pending_hardened_audit' or existing['source_manifest_sha256']!=base.digest(args.source_root/'manifest.json') or existing['fixed_probe_manifest_sha256']!=h.FIXTURE_SHA256:
            raise ValueError('resume requires the exact pending source materialization')
    else:
        h.materialize(args)
    reused=reuse_initial(args)
    print(json.dumps({'reused_initial_reference_panels':len(reused),'reused_problem_ids':reused}),flush=True)
    fresh=audit_remaining(args,reused)
    finalize(args)
    h.validate_quality({'slate_root':str(args.output)})
    print(json.dumps({'fresh_reference_panels':len(fresh),'final_quality_validated':True}),flush=True)


if __name__=='__main__':
    main()
