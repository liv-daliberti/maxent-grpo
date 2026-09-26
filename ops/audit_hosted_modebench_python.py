"""Audit hosted-model Python grades serially; preserve every original receipt.

Never calls a model or changes raw receipts, prompts, verifiers, or timeouts.
All Python positives and negatives are regraded once per unique row/text pair.
Every discrepant pair receives two further serial confirmations before repair.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from audit_hosted_modebench_completion import load_inventory, validate_native_records


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def publish(path,data):
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as h:h.write(data);h.flush();os.fsync(h.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--expected-samples',type=int,default=15360,help='Explicit frozen condition size; full original cohorts default to 15360')
    args=ap.parse_args();root=args.output_root.resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit_dir=root/'primary_python_audits'/stamp;audit_dir.mkdir(parents=True,exist_ok=False)
    script_copy=audit_dir/'audit_primary_python.py';script_copy.write_bytes(Path(__file__).read_bytes())
    inventory=load_inventory(root,expected_samples=args.expected_samples)
    manifest=inventory['manifest']
    helper_source=Path(__file__).with_name('audit_hosted_modebench_completion.py')
    helper_copy=audit_dir/helper_source.name;helper_copy.write_bytes(helper_source.read_bytes())
    for rel,value in manifest['artifact_sha256'].items():assert digest(root/rel)==value,rel
    for rel,value in manifest['code_sha256'].items():assert digest(root/'code'/rel)==value,rel
    sys.path[:0]=[str(root/'code/ops'),str(root/'code/src')]
    from frontier_modebench_contract import grade_response
    from oat_drgrpo.python_modebench_process import _SHARED_VERIFIER
    rows={(r['level'],r['domain'],r['row_index']):r for r in map(json.loads,(root/'rows.jsonl').read_text().splitlines())}
    records=[];receipt_hashes={}
    paths=sorted((root/'sample_receipts').glob('*.json'))
    def load_receipt(path):
        data=path.read_bytes();record=json.loads(data)
        if path.name!=record['sample_id']+'.json':
            raise ValueError('Atomic sample filename differs from its identity')
        return record,{'path':str(path.relative_to(root)),'sha256':hashlib.sha256(data).hexdigest()}
    with ThreadPoolExecutor(max_workers=16) as pool:
        loaded=list(pool.map(load_receipt,paths))
    for record,receipt in loaded:
        records.append(record);receipt_hashes[record['sample_id']]=receipt
    assert len({r['sample_id'] for r in records})==len(records)
    if len(records)!=manifest['request_count'] or {r['sample_id'] for r in records}!=set(inventory['expected']):
        raise ValueError('Final strict Python audit requires every expected response')
    validate_native_records(inventory,records,io_workers=16)
    groups={}
    for record in records:
        if record['domain']!='python_factors':continue
        row=rows[record['level'],record['domain'],record['row_index']]
        assert sha(row)==record['row_sha256']
        key=sha([record['row_sha256'],record['text']])
        groups.setdefault(key,[]).append(record)
    candidate='lambda n: 2'
    warm_spec={'verifier':'python_factor_function','python_version':'factor-v1','cases':[6,8]}
    warmups=[]
    for index in range(3):
        _SHARED_VERIFIER._start();time.sleep(1.25)
        result=_SHARED_VERIFIER.validate(candidate,warm_spec)
        warmups.append(result is not None and result.outputs==(2,2))
        if warmups[-1]:break
    else:raise RuntimeError('Synthetic Python fixture failed bounded frozen-worker warmup')
    evidence=[];grades={};discrepancies=[];unresolved=[]
    grade_fields=('verified','canonical_key','graded_text')
    for index,(key,group) in enumerate(sorted(groups.items())):
        representative=group[0];row=rows[representative['level'],representative['domain'],representative['row_index']]
        begin=time.monotonic()
        result=grade_response(representative['level'],representative['domain'],row,representative['text'])
        changed=[r for r in group if any(r[field]!=result[field] for field in grade_fields)]
        detail={'group_id':key,'row_sha256':representative['row_sha256'],'level':representative['level'],'row_index':representative['row_index'],'text':representative['text'],'sample_ids':[r['sample_id'] for r in group],'original_grades':dict(Counter(json.dumps({field:r[field] for field in grade_fields},sort_keys=True) for r in group)),'serial_result':result,'elapsed_seconds':time.monotonic()-begin,'worker_alive_after':_SHARED_VERIFIER._process is not None and _SHARED_VERIFIER._process.poll() is None,'confirmations':[]}
        if changed:
            for repeat in range(2):
                detail['confirmations'].append(grade_response(representative['level'],representative['domain'],row,representative['text']))
            detail['repeat_confirmed']=all(g==result for g in detail['confirmations'])
            detail['changed_sample_ids']=[r['sample_id'] for r in changed]
            if not detail['repeat_confirmed']:unresolved.append(detail)
            discrepancies.append(detail)
        grades[key]=result;evidence.append(detail)
        if (index+1)%250==0:print(json.dumps({'unique_python_groups_done':index+1,'total':len(groups),'discrepant_groups':len(discrepancies)}),flush=True)
    frozen_sources={}
    for name,module in sorted(sys.modules.items()):
        if name=='frontier_modebench_contract' or name.startswith('oat_drgrpo.'):
            filename=getattr(module,'__file__',None)
            if filename is None:continue
            path=Path(filename).resolve();assert path.is_relative_to(root/'code')
            rel=str(path.relative_to(root/'code'));actual=digest(path);assert actual==manifest['code_sha256'][rel]
            frozen_sources[name]={'path':str(path),'sha256':actual}
    full=len(records)==manifest['request_count']
    report={'schema':'frontier-modebench-primary-python-serial-audit-v1','status':'unresolved' if unresolved else 'complete_for_snapshot','generated_at_utc':datetime.now(timezone.utc).isoformat(),'full_sampling_complete':full,'snapshot_samples':len(records),'expected_samples':manifest['request_count'],'python_samples':sum(len(v) for v in groups.values()),'unique_python_row_text_groups':len(groups),'discrepant_groups':len(discrepancies),'corrected_samples':sum(len(d['changed_sample_ids']) for d in discrepancies),'unresolved_groups':len(unresolved),'primary_manifest_sha256':digest(root/'manifest.json'),'source_receipt_manifest':receipt_hashes,'source_receipt_manifest_sha256':sha(receipt_hashes),'frozen_grader_modules':frozen_sources,'audit_source':str(script_copy.relative_to(root)),'audit_source_sha256':digest(script_copy),'protocol':{'api_calls':0,'independent_receipt_io_workers':16,'scientific_python_grading_workers':1,'strict_text_and_raw_receipts_changed':False,'unchanged_frozen_executable_grader':True,'parent_timeout_seconds':_SHARED_VERIFIER.timeout_seconds,'child_candidate_timeout_seconds':0.25,'all_python_positives_and_negatives_regraded':True,'deduplication_key':'exact row_sha256 plus original response text','independent_repeat_confirmations_per_discrepancy':2,'warmup':'original frozen worker started, allowed1.25s to initialize, then bounded synthetic known-valid lambda check','operational_correction_not_formatting_normalization':True,'interpretation':'Repeated differences identify original runtime grading discrepancies; cold-start timeout is a plausible cause, not directly observed in original receipts.'},'warmup_synthetic_fixture':{'candidate':candidate,'spec':warm_spec},'audit_helper_source':str(helper_copy.relative_to(root)),'audit_helper_sha256':digest(helper_copy),'warmup_results':warmups,'discrepancies':discrepancies,'group_evidence':evidence}
    if unresolved:
        (audit_dir/'audit.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        raise RuntimeError('Unstable repeated grades; no derived primary published')
    corrected_false_to_true=corrected_true_to_false=changed_key=0
    derived=[]
    audit_reference=str((audit_dir/'audit.json').relative_to(root))
    for original in records:
        record=dict(original);receipt=receipt_hashes[record['sample_id']]
        record.update(raw_strict_verified=original['verified'],raw_strict_canonical_key=original['canonical_key'],raw_strict_graded_text=original['graded_text'],raw_strict_receipt=receipt['path'],raw_strict_receipt_sha256=receipt['sha256'],primary_regrade_corrected=False,primary_regrade_audit=audit_reference)
        if original['domain']=='python_factors':
            result=grades[sha([original['row_sha256'],original['text']])]
            changed=any(original[field]!=result[field] for field in grade_fields)
            record.update(result);record['primary_regrade_corrected']=changed
            if changed:
                corrected_false_to_true+=int(not original['verified'] and result['verified'])
                corrected_true_to_false+=int(original['verified'] and not result['verified'])
                changed_key+=int(original['verified'] and result['verified'] and original['canonical_key']!=result['canonical_key'])
        derived.append(record)
    sidecar=b''.join((json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode() for r in derived)
    sidecar_path=audit_dir/'audited_primary_samples.jsonl';sidecar_path.write_bytes(sidecar)
    report['correction_counts']={'false_to_true':corrected_false_to_true,'true_to_false':corrected_true_to_false,'verified_key_changes':changed_key}
    report['derived_primary']={'path':str(sidecar_path.relative_to(root)),'sha256':digest(sidecar_path),'records':len(derived)}
    audit_bytes=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    (audit_dir/'audit.json').write_bytes(audit_bytes)
    publish(root/'audited_primary_samples.jsonl',sidecar)
    publish(root/'primary_python_regrade_audit.json',audit_bytes)
    print(json.dumps({'status':report['status'],'snapshot_samples':len(records),'full_sampling_complete':full,'python_samples':report['python_samples'],'unique_groups':len(groups),'correction_counts':report['correction_counts'],'audit':str(audit_dir/'audit.json'),'audit_sha256':digest(audit_dir/'audit.json'),'derived_primary_sha256':digest(sidecar_path)},sort_keys=True),flush=True)


if __name__=='__main__':main()
