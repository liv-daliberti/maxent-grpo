#!/usr/bin/env python3
"""Versioned source-only verifier hardening with strict 12/12 reference gates.

Appends only the five already frozen source-audit counterexamples. Original
prompts, checkers, and suite rows are preserved. No policy output is consulted.
"""
from __future__ import annotations
import argparse
import importlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping
import build_constructive_code_wider_20260921 as base

SCHEMA = "constructive-code-hardened-20260921-v1"
QUALITY_SCHEMA = "constructive-code-hardening-quality-20260921-v1"
FIXTURE_SHA256 = "3e7c34e2d8b2e3bb6a5d4b329525bffaeced7e8fb2791f2d09f257bca65513cb"
DEFAULT_FIXTURE = base.ROOT / "var/artifacts/codecontests_out_of_reward_stress_20260921.json"
SUITE_ID = "codecontests_source_hardened_20260921_v1"
POLICY = {"required_tpr": 1.0, "required_tnr": 1.0, "positive_replays": 12, "negative_replays": 12, "append_only": True, "checker_modified": False, "prompt_modified": False, "policy_outputs_used_for_test_selection": False, "source_derived_supplemental_inputs_declared": True, "remaining_source_false_acceptance": "quarantine task; no model-conditioned test selection"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def source_code_identity() -> dict:
    from oat_drgrpo import constructive_code_wider_adapters_20260921 as wider
    from oat_drgrpo import constructive_code_reserved_adapters_20260921 as reserved
    from oat_drgrpo import constructive_code_holdout_extension_20260921 as extension
    from oat_drgrpo import constructive_code_adapters as old
    from oat_drgrpo import constructive_code_sandbox as sandbox
    from oat_drgrpo import constructive_code as core
    import replay_constructive_code_v2 as replay
    return {str(Path(module.__file__).resolve()): base.digest(Path(module.__file__)) for module in (wider, reserved, extension, old, sandbox, core, replay)}



def verifier_support_identity() -> dict:
    # Imported module paths follow the frozen execution snapshot, not ROOT.
    names = (
        "build_constructive_code_hardened_20260921",
        "build_constructive_code_wider_20260921",
        "evaluate_constructive_code_pilot_20260921",
        "evaluate_constructive_code_v3_coder_viability",
        "evaluate_constructive_code_v6_coder_viability",
        "replay_constructive_code_review_slate",
        "audit_constructive_code_sources",
        "audit_constructive_code_checker_equivalence",
        "audit_constructive_code_v2",
        "materialize_constructive_code_v2",
        "materialize_constructive_code_review_slate",
        "materialize_constructive_code_plus_suites",
    )
    result = {
        "ops/" + name + ".py": base.digest(Path(importlib.import_module(name).__file__))
        for name in names
    }
    import replay_constructive_code_v2 as replay
    result["ops/constructive_code_sandbox.c"] = base.digest(replay.SANDBOX_SOURCE)
    result["third_party/testlib/testlib.h"] = base.digest(replay.TESTLIB_ROOT / "testlib.h")
    return result


def relocated_source_identity(values: Mapping[str,str]) -> dict:
    result = {}
    for path, digest in values.items():
        parts = Path(path).parts
        starts = [i for i, part in enumerate(parts) if part in ("src", "ops", "third_party")]
        if not starts:
            raise ValueError("verifier source lacks project-relative module identity")
        name = "/".join(parts[starts[-1]:])
        if name in result:
            raise ValueError("duplicate project-relative verifier source identity")
        result[name] = digest
    return result


def compare_relocated_sources(expected: Mapping[str,str], current: Mapping[str,str]) -> None:
    if relocated_source_identity(expected) != relocated_source_identity(current):
        raise ValueError("active verifier source bytes differ from hardened admission audit")


def require_pinned_testlib() -> str:
    import replay_constructive_code_v2 as replay
    actual=base.digest(replay.TESTLIB_ROOT/'testlib.h')
    if actual!=replay.TESTLIB_SHA256:
        raise ValueError('pinned testlib source drift')
    return actual


def materialize(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    if base.digest(args.fixture) != FIXTURE_SHA256:
        raise ValueError("fixed source-derived probe fixture SHA drift")
    fixture = json.loads(args.fixture.read_text()); probes = {r["task_id"]:r for r in fixture["cases"]}
    if set(probes) != {"1016_D", "1360_G", "1408_A", "1323_A", "1038_B"} or fixture["model_sampling_used_in_probe_selection"]:
        raise ValueError("fixture is not the exact pre-policy source-audit probe set")
    original = json.loads((args.source_root / "manifest.json").read_text())
    if original["status"] != "audited" or original["tasks_sha256"] != base.canonical_hash(original["tasks"]):
        raise ValueError("source slate is not intact and audited")
    source_manifest_sha = base.digest(args.source_root / "manifest.json")
    shutil.copytree(args.source_root, args.output)
    shutil.copy2(args.source_root / "manifest.json", args.output / "pre_hardening_manifest.json")
    shutil.copy2(args.fixture, args.output / "fixed_source_probe_fixture.json")
    manifest = dict(original); originals = []
    for summary in manifest["tasks"]:
        folder = args.output / summary["relative_path"]
        record = json.loads((folder / "task.json").read_text())
        if record["task_record_sha256"] != summary["task_record_sha256"] or record["task_record_sha256"] != base.canonical_hash({k:v for k,v in record.items() if k!="task_record_sha256"}):
            raise ValueError("source task record drift")
        if record["admission_status"] != "admitted":
            continue
        problem = record["source_problem_id"]
        if base.digest(folder / "checker.cpp") != record["checker_sha256"] or base.digest(folder / "py3_replays.jsonl") != record["references"]["jsonl_sha256"]:
            raise ValueError("source checker/reference ledger drift")
        if base.raw_sha256(record["statement"]) != record["statement_sha256"]:
            raise ValueError("source prompt binding drift")
        source_file = record.get("suite_file", "inputs.jsonl.gz")
        if base.digest(folder / source_file) != record["suite"]["compressed_jsonl_sha256"]:
            raise ValueError("source verifier input bytes drift")
        old_rows = [json.loads(line) for line in gzip.open(folder / source_file, "rt")]
        source_inputs = [row["stdin"] for row in old_rows]
        additions = []
        if problem in probes:
            probe = probes[problem]
            if probe["checker_sha256"] != record["checker_sha256"] or probe["frozen_reward_suite_sha256"] != record["suite"]["suite_sha256"] or base.raw_sha256(probe["stdin"]) != probe["input_sha256"]:
                raise ValueError("fixed probe does not match original checker/suite")
            if probe["stdin"] in source_inputs:
                raise ValueError("source-audit counterexample unexpectedly already in suite")
            additions.append({"stdin":probe["stdin"],"input_sha256":probe["input_sha256"],"provenance":probe["probe_provenance"]})
        shutil.copy2(folder / "admission_audit.json", folder / "pre_hardening_admission_audit.json")
        shutil.copy2(folder / "audit_replays.jsonl", folder / "pre_hardening_audit_replays.jsonl")
        hardening = {"schema":SCHEMA,"original_task_record_sha256":record["task_record_sha256"],"source_manifest_sha256":source_manifest_sha,"original_suite_id":record["suite_id"],"original_suite":record["suite"],"original_suite_file":source_file,"original_admission_sha256":base.digest(folder/'pre_hardening_admission_audit.json'),"original_replays_sha256":base.digest(folder/'pre_hardening_audit_replays.jsonl'),"checker_sha256":record["checker_sha256"],"statement_sha256":record["statement_sha256"],"reference_ledger_sha256":record["references"]["jsonl_sha256"],"fixed_probe_manifest_sha256":FIXTURE_SHA256,"added_inputs":additions,"original_input_count":len(source_inputs),"original_input_bytes_preserved_in_order":True,"checker_modified":False,"prompt_modified":False}
        record.update({"hardening":hardening,"suite_id":SUITE_ID,"suite":base._write_suite(folder/'hardened_inputs.jsonl.gz',source_inputs+[r['stdin'] for r in additions]),"suite_file":"hardened_inputs.jsonl.gz","admission_status":"pending_hardened_audit"})
        record["task_record_sha256"] = base.canonical_hash({k:v for k,v in record.items() if k!="task_record_sha256"})
        base.write_json(folder/'task.json',record)
        summary.update({"status":"pending_hardened_audit","task_record_sha256":record["task_record_sha256"]})
        originals.append(problem)
    manifest.update({"status":"pending_hardened_audit","hardening_schema":SCHEMA,"source_manifest_sha256":source_manifest_sha,"fixed_probe_manifest_sha256":FIXTURE_SHA256,"hardening_source_problem_ids":originals,"admitted_problem_ids":[],"tasks_sha256":base.canonical_hash(manifest['tasks'])})
    manifest.pop('hardening_quality_sha256',None)
    base.write_json(args.output/'manifest.json',manifest)


def audit_one(args) -> dict:
    import evaluate_constructive_code_v3_coder_viability as prior
    from oat_drgrpo.constructive_code_wider_adapters_20260921 import input_cases
    replay = base.configure_replay()
    require_pinned_testlib()
    manifest = json.loads((args.output/'manifest.json').read_text())
    summary = next(r for r in manifest['tasks'] if r['source_problem_id']==args.task)
    folder=args.output/summary['relative_path']; record=json.loads((folder/'task.json').read_text()); hardening=record['hardening']
    task_work=args.work_root/args.task.lower(); runtime_args=SimpleNamespace(image=args.image,runtime_root=args.runtime_root,launcher=task_work/'launcher',scratch_root=task_work/'scratch')
    result={"source_problem_id":args.task,"status":"fail","violations":[],"audit_method":"full_independent_12_positive_12_negative_replay","hardening_policy":POLICY,"hardening_schema":SCHEMA}
    try:
        runtime=base._runtime(replay,runtime_args)
        task,build=base._task_from_record(replay,folder,record,task_work/'build')
        original=[json.loads(line)['stdin'] for line in gzip.open(folder/hardening['original_suite_file'],'rt')]
        actual=[test.stdin.decode() for test in task.tests]
        expected=original+[r['stdin'] for r in hardening['added_inputs']]
        if actual!=expected or len(original)!=hardening['original_input_count']:
            raise ValueError("hardening must preserve every original input byte in order and append exact fixed probes")
        for text in actual:
            input_cases(args.task,text)
        validator_probes=[]
        if hardening['added_inputs']:
            validator=task_work/'build'/'validator';validator_build=replay._compile_checker(folder/'validator.cpp',validator,record['validator_sha256'])
            for row in hardening['added_inputs']:
                completed=subprocess.run([str(validator)],input=row['stdin'],capture_output=True,text=True,timeout=3)
                if completed.returncode!=0:
                    raise ValueError("fixed valid probe rejected by original input validator")
                validator_probes.append({'input_sha256':row['input_sha256'],'validator_returncode':completed.returncode,'validator_build':validator_build})
        canonical=base._canonicalizer_audit(replay,task,runtime_args.scratch_root)
        references=read_rows(folder/'py3_replays.jsonl'); counts=Counter(row['known_label'] for row in references)
        if counts != Counter({'correct':12,'incorrect':12}):
            raise ValueError("hardening requires exactly12references of each label")
        receipts=[]
        with ThreadPoolExecutor(max_workers=args.reference_workers) as pool:
            futures={pool.submit(replay._replay_submission,task=task,submission=replay.Submission(row['code'],row['known_label'],row['submission_sha256']),launcher=runtime_args.launcher,runtime_root=args.runtime_root,scratch_root=runtime_args.scratch_root):row for row in references}
            for future in as_completed(futures):
                row=futures[future]
                try:
                    receipt=future.result(); violations=prior._hard_replay_violations(receipt)
                    if row['known_label']=='incorrect':
                        violations=[v for v in violations if v!='candidate execution-bound violation']
                    receipt['audit_violations']=violations
                except Exception as error:
                    receipt={'source_problem_id':args.task,'known_label':row['known_label'],'submission_sha256':row['submission_sha256'],'released_checker_accepted':False,'wrapper_accepted':False,'behavior_key':None,'audit_violations':[f'{type(error).__name__}:{error}']}
                receipts.append(receipt)
        receipts.sort(key=lambda r:(r['known_label'],r['submission_sha256']));base.write_jsonl(folder/'audit_replays.jsonl',receipts)
        positive=[r for r in receipts if r['known_label']=='correct'];negative=[r for r in receipts if r['known_label']=='incorrect']
        positive_pass=sum(r['released_checker_accepted'] and r['wrapper_accepted'] for r in positive);negative_rejected=sum(not r['released_checker_accepted'] and not r['wrapper_accepted'] for r in negative)
        violations=[f"{r['submission_sha256']}:{v}" for r in receipts for v in r['audit_violations']]
        if positive_pass!=12:violations.append('source-positive gate requires12/12acceptance')
        if negative_rejected!=12:violations.append('source-negative gate requires12/12rejection')
        result.update({'status':'pass' if not violations else 'fail','violations':violations,'positive_replays':12,'negative_replays':12,'positive_accepted':positive_pass,'negative_rejected':negative_rejected,'tpr':positive_pass/12,'tnr':negative_rejected/12,'replays_sha256':base.digest(folder/'audit_replays.jsonl'),'checker_build':build,'runtime':runtime,'canonicalizer_audit':canonical,'added_input_validator_checks':validator_probes,'original_inputs_preserved_in_order':True,'original_checker_unchanged':base.digest(folder/'checker.cpp')==hardening['checker_sha256'],'original_prompt_unchanged':base.raw_sha256(record['statement'])==hardening['statement_sha256'],'canonicalizer_source_files':source_code_identity()})
    except Exception as error:
        result['violations'].append(f'{type(error).__name__}:{error}')
    base.write_json(folder/'admission_audit.json',result)
    record.update({'admission_status':'admitted' if result['status']=='pass' else 'quarantined','admission_audit_sha256':base.digest(folder/'admission_audit.json')})
    record['task_record_sha256']=base.canonical_hash({k:v for k,v in record.items() if k!='task_record_sha256'});base.write_json(folder/'task.json',record)
    print(json.dumps({'task_id':args.task,'status':result['status'],'positive_accepted':result.get('positive_accepted'),'negative_rejected':result.get('negative_rejected'),'violations':result['violations'][:3]},sort_keys=True),flush=True)
    return result


def audit_all(args) -> None:
    manifest=json.loads((args.output/'manifest.json').read_text());pending=manifest['hardening_source_problem_ids']
    order=sorted(pending,key=lambda p:(p not in {'1016_D','1360_G','1408_A','1323_A','1038_B'},pending.index(p)))
    args.work_root.mkdir(parents=True,exist_ok=True)
    def worker(problem):
        cmd=[sys.executable,str(Path(__file__).resolve()),'--phase','audit-one','--output',str(args.output.resolve()),'--task',problem,'--image',str(args.image.resolve()),'--runtime-root',str(args.runtime_root.resolve()),'--work-root',str(args.work_root.resolve()),'--reference-workers',str(args.reference_workers)]
        completed=subprocess.run(cmd,capture_output=True,text=True)
        print(completed.stdout,flush=True,end='')
        if completed.returncode:
            raise RuntimeError(f'{problem}:workerfailed:{completed.stderr[-3000:]}')
    with ThreadPoolExecutor(max_workers=args.task_workers) as pool:
        futures=[pool.submit(worker,p) for p in order]
        for future in as_completed(futures):future.result()
    quality=[]
    for summary in manifest['tasks']:
        if summary['source_problem_id'] not in pending:continue
        folder=args.output/summary['relative_path'];row=json.loads((folder/'task.json').read_text());audit=json.loads((folder/'admission_audit.json').read_text());h=row['hardening']
        summary.update({'status':row['admission_status'],'task_record_sha256':row['task_record_sha256'],'admission_audit_sha256':row['admission_audit_sha256']})
        quality.append({'task_id':row['source_problem_id'],'status':audit['status'],'audit_method':audit['audit_method'],'task_record_sha256':row['task_record_sha256'],'admission_audit_sha256':row['admission_audit_sha256'],'reference_replays_sha256':audit.get('replays_sha256'),'reference_ledger_sha256':row['references']['jsonl_sha256'],'positive_replays':audit.get('positive_replays',0),'negative_replays':audit.get('negative_replays',0),'positive_accepted':audit.get('positive_accepted',0),'negative_rejected':audit.get('negative_rejected',0),'tpr':audit.get('tpr'),'tnr':audit.get('tnr'),'checker_sha256':row['checker_sha256'],'statement_sha256':row['statement_sha256'],'original_checker_sha256':h['checker_sha256'],'original_statement_sha256':h['statement_sha256'],'original_task_record_sha256':h['original_task_record_sha256'],'original_suite_id':h['original_suite_id'],'original_suite_sha256':h['original_suite']['suite_sha256'],'original_suite_file_sha256':h['original_suite']['compressed_jsonl_sha256'],'hardened_suite_id':row['suite_id'],'hardened_suite_sha256':row['suite']['suite_sha256'],'hardened_suite_file_sha256':row['suite']['compressed_jsonl_sha256'],'original_input_count':h['original_input_count'],'hardened_input_count':row['suite']['test_count'],'added_input_sha256s':[r['input_sha256'] for r in h['added_inputs']],'original_inputs_preserved_in_order':audit.get('original_inputs_preserved_in_order',False),'original_checker_unchanged':audit.get('original_checker_unchanged',False),'original_prompt_unchanged':audit.get('original_prompt_unchanged',False),'violations':audit['violations']})
    admitted=[r['task_id'] for r in quality if r['status']=='pass']
    gate={'schema':QUALITY_SCHEMA,'status':'pass' if admitted else 'fail','hardening_schema':SCHEMA,'policy':POLICY,'fixed_probe_manifest_sha256':FIXTURE_SHA256,'source_manifest_sha256':manifest['source_manifest_sha256'],'source_candidates':len(quality),'admitted':len(admitted),'quarantined':len(quality)-len(admitted),'admitted_problem_ids':admitted,'tasks':quality,'canonicalizer_source_files':source_code_identity(),'builder_sha256':base.digest(Path(__file__)),'loader_guard_sha256':base.digest(Path(__file__)),'verifier_support_files':verifier_support_identity(),'pinned_testlib_sha256':require_pinned_testlib()}
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


def validate_quality(config: Mapping[str,Any]) -> dict:
    root=Path(config['slate_root']);manifest=json.loads((root/'manifest.json').read_text());quality=json.loads((root/'hardening_quality.json').read_text())
    if manifest.get('hardening_schema')!=SCHEMA or manifest.get('hardening_quality_sha256')!=base.digest(root/'hardening_quality.json') or quality.get('schema')!=QUALITY_SCHEMA or quality.get('status')!='pass' or quality.get('policy')!=POLICY or quality.get('fixed_probe_manifest_sha256')!=FIXTURE_SHA256 or base.digest(root/'fixed_source_probe_fixture.json')!=FIXTURE_SHA256:
        raise ValueError('hardened verifier quality/fixture binding failed')
    if quality.get('loader_guard_sha256')!=base.digest(Path(__file__)) or quality.get('pinned_testlib_sha256')!=require_pinned_testlib():
        raise ValueError('hardened loader or testlib source binding drift')
    compare_relocated_sources(quality['canonicalizer_source_files'],source_code_identity())
    compare_relocated_sources(quality['verifier_support_files'],verifier_support_identity())
    by_id={r['task_id']:r for r in quality['tasks']}
    wanted=config.get('problem_ids',manifest.get('default_training_problem_ids',manifest['admitted_problem_ids']))
    for problem in wanted:
        row=by_id.get(problem)
        if not row or row['status']!='pass' or (row['positive_replays'],row['negative_replays'],row['positive_accepted'],row['negative_rejected'])!=(12,12,12,12) or row['tpr']!=1.0 or row['tnr']!=1.0 or row['checker_sha256']!=row['original_checker_sha256'] or row['statement_sha256']!=row['original_statement_sha256'] or not all(row[k] for k in ('original_inputs_preserved_in_order','original_checker_unchanged','original_prompt_unchanged')):
            raise ValueError('task lacks strict hardened source-quality gate')
        summary=next(r for r in manifest['tasks'] if r['source_problem_id']==problem)
        if row['task_record_sha256']!=summary['task_record_sha256'] or row['admission_audit_sha256']!=summary['admission_audit_sha256']:
            raise ValueError('hardened task/quality receipt binding drift')
        folder=root/summary['relative_path']
        if base.digest(folder/'audit_replays.jsonl')!=row['reference_replays_sha256']:
            raise ValueError('hardened reference replay ledger drift')
        references=read_rows(folder/'py3_replays.jsonl');replays=read_rows(folder/'audit_replays.jsonl')
        reference_labels={(r['submission_sha256'],r['known_label']) for r in references}
        replay_labels={(r['submission_sha256'],r['known_label']) for r in replays}
        if len(references)!=24 or len(replays)!=24 or len(reference_labels)!=24 or reference_labels!=replay_labels:
            raise ValueError('hardened references/replays are not an exact24program bijection')
        for receipt in replays:
            expected=receipt['known_label']=='correct'
            if receipt['released_checker_accepted'] is not expected or receipt['wrapper_accepted'] is not expected or receipt.get('audit_violations') or receipt['suite_id']!=row['hardened_suite_id'] or receipt['suite_sha256']!=row['hardened_suite_sha256'] or receipt['checker_sha256']!=row['checker_sha256'] or receipt['source_problem_id']!=problem:
                raise ValueError('hardened reference verdict or source binding mismatch')
            if expected and receipt['execution']['executed_tests']!=row['hardened_input_count']:
                raise ValueError('positive source replay omitted hardened suite inputs')
    return quality


def load_tasks(adapter_config: Mapping[str,Any]):
    validate_quality(adapter_config)
    return base.load_tasks(adapter_config)


def dataset_identity(adapter_config: Mapping[str,Any]):
    quality=validate_quality(adapter_config);identity=base.dataset_identity(adapter_config);root=Path(adapter_config['slate_root'])
    return {**identity,'hardening_schema':SCHEMA,'hardening_quality_sha256':base.digest(root/'hardening_quality.json'),'fixed_probe_manifest_sha256':FIXTURE_SHA256,'strict_source_tpr':1.0,'strict_source_tnr':1.0,'hardening_source_manifest_sha256':quality['source_manifest_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('materialize','audit','audit-one','build'),required=True);p.add_argument('--source-root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',type=Path,default=DEFAULT_FIXTURE);p.add_argument('--task');p.add_argument('--image',type=Path,default=base.ROOT/'var/images/python-3.10-slim-c1e4e6c01eb4.sqsh');p.add_argument('--runtime-root',type=Path,default=Path('/tmp/constructive_wider_20260921/runtime'));p.add_argument('--work-root',type=Path,default=Path('/tmp/constructive_hardened_initial_20260921'));p.add_argument('--task-workers',type=int,default=4);p.add_argument('--reference-workers',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.task_workers<=4 or not 1<=args.reference_workers<=8:p.error('bounded task/reference worker counts required')
    if args.phase in ('materialize','build'):
        if args.source_root is None:p.error('--source-root required')
        materialize(args)
    if args.phase in ('audit','build'):audit_all(args)
    elif args.phase=='audit-one':audit_one(args)


if __name__=='__main__':main()
