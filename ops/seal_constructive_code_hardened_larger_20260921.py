#!/usr/bin/env python3
"""Seal exact replay reuse and the strict larger split after source audits finish.

Does not alter tasks, reference verdicts, checkers, prompts, or verifier inputs.
The independent receipt and this post-audit validation are explicitly separate
from guards executed by the larger build coordinator.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import build_constructive_code_hardened_20260921 as h

REUSE_AUDIT_SHA256='3f63d3c2c04b13ae7a20ffc71b69a8c525f11b4d23476006fb15b524cee79527'
FIELDS=('source_problem_id','problem_key','task_adapter','witness_family','limits','checker_sha256','statement','statement_sha256','references','suite_id','suite')


def verify_reuse(root, initial, receipt):
    b=h.base
    if receipt['status']!='pass' or receipt['initial_manifest_sha256']!=b.digest(initial/'manifest.json') or receipt['replay_reuse_manifest_sha256']!=b.digest(root/'replay_reuse_manifest.json'):
        raise ValueError('independent reuse audit root binding failed')
    old_m=json.loads((initial/'manifest.json').read_text());new_m=json.loads((root/'manifest.json').read_text())
    old_by={r['source_problem_id']:r for r in old_m['tasks']};new_by={r['source_problem_id']:r for r in new_m['tasks']}
    expected=set(json.loads((root/'replay_reuse_manifest.json').read_text())['reused_problem_ids'])
    if {r['task_id'] for r in receipt['tasks']}!=expected or len(receipt['tasks'])!=len(expected):
        raise ValueError('independent reuse audit task bijection failed')
    for audit in receipt['tasks']:
        problem=audit['task_id'];of=initial/old_by[problem]['relative_path'];nf=root/new_by[problem]['relative_path']
        old=json.loads((of/'task.json').read_text());new=json.loads((nf/'task.json').read_text())
        for record,prefix in ((old,'old'),(new,'new')):
            digest=b.canonical_hash({k:v for k,v in record.items() if k!='task_record_sha256'})
            if digest!=record['task_record_sha256'] or digest!=audit[prefix+'_task_record_sha256']:
                raise ValueError('reused task canonical record hash drift')
        if any(old[field]!=new[field] for field in FIELDS):
            raise ValueError('reused adapter/prompt/source/runtime contract drift')
        if b.digest(of/'audit_replays.jsonl')!=audit['reference_replays_sha256'] or b.digest(nf/'audit_replays.jsonl')!=audit['reference_replays_sha256'] or b.digest(nf/'admission_audit.json')!=audit['admission_audit_sha256']:
            raise ValueError('reused source replay or admission bytes drift')
    return len(expected)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--initial-root',type=Path,required=True);p.add_argument('--reuse-audit',type=Path,required=True);args=p.parse_args()
    b=h.base;r=args.root;m=json.loads((r/'manifest.json').read_text())
    if m['status']!='audited' or b.digest(args.reuse_audit)!=REUSE_AUDIT_SHA256:
        raise ValueError('completed source audit and pinned independent reuse receipt required')
    q=h.validate_quality({'slate_root':str(r),'problem_ids':m['admitted_problem_ids']})
    reused=verify_reuse(r,args.initial_root,json.loads(args.reuse_audit.read_text()))
    admitted=[row for row in m['tasks'] if row['status']=='admitted'];quarantined=[row for row in m['tasks'] if row['status']!='admitted']
    counts={split:sum(row['split']==split for row in admitted) for split in ('train','validation','test')}
    if counts!={'train':29,'validation':2,'test':13}:
        raise ValueError('source outcomes differ from accepted strict larger scope; explicit review required')
    for field,split in (('default_training_problem_ids','train'),('validation_problem_ids','validation'),('admitted_test_problem_ids','test')):
        if m[field]!=[row['source_problem_id'] for row in admitted if row['split']==split]:
            raise ValueError('admitted split membership differs from original order')
    shutil.copy2(args.reuse_audit,r/'independent_replay_reuse_audit.json')
    q['post_replay_pool_sealing']={'sealer_source_sha256':b.digest(Path(__file__)),'independent_reuse_audit_sha256':REUSE_AUDIT_SHA256,'reused_initial_source_panels':reused,'fresh_source_panels':q['source_candidates']-reused,'validation_stage':'independent post-reuse binding validation; no change to execution or verdicts','task_adapter_problem_key_and_canonical_record_hashes_revalidated':True}
    b.write_json(r/'hardening_quality.json',q);m['hardening_quality_sha256']=b.digest(r/'hardening_quality.json')
    m['pool_counts']=counts;m['family_counts']={split:dict(Counter(row['family'] for row in admitted if row['split']==split)) for split in counts}
    m['strict_larger_ready_source_gate']={'status':'pass','required_tpr':1.0,'required_tnr':1.0,'accepted_pool_counts':counts,'original_32_2_16_quota_met':False,'heldout_policy_outputs_used':False}
    m['source_candidate_counts']={'prospective':len(m['tasks']),'pre_hardening_admitted':q['source_candidates'],'strict_admitted':len(admitted),'total_quarantined':len(quarantined),'new_strict_quarantines':q['quarantined']}
    m['independent_replay_reuse_audit_sha256']=REUSE_AUDIT_SHA256
    old_split=r/'split_manifest.json';preserved=r/'pre_hardening_split_manifest.json'
    if not preserved.exists():
        shutil.copy2(old_split,preserved)
    split={'schema':'constructive-code-hardened-larger-splits-20260921-v1','hardening_quality_sha256':m['hardening_quality_sha256'],'original_split_manifest_sha256':b.digest(preserved),'train_ids':m['default_training_problem_ids'],'validation_ids':m['validation_problem_ids'],'heldout_ids':m['admitted_test_problem_ids'],'protected_unused_original_heldout_ids':m['protected_unused_original_heldout_ids'],'selection_used_policy_outcomes':False,'excluded_candidates':quarantined}
    b.write_json(old_split,split);m['split_manifest_sha256']=b.digest(old_split);b.write_json(r/'manifest.json',m)
    summary=json.loads((r/'admission_summary.json').read_text());summary['hardening_quality_sha256']=m['hardening_quality_sha256'];b.write_json(r/'admission_summary.json',summary)
    h.validate_quality({'slate_root':str(r),'problem_ids':m['admitted_problem_ids']})
    print(json.dumps({'counts':counts,'manifest_sha256':b.digest(r/'manifest.json'),'quality_sha256':m['hardening_quality_sha256'],'train':m['default_training_problem_ids'],'validation':m['validation_problem_ids'],'heldout':m['admitted_test_problem_ids'],'quarantined':[row['source_problem_id'] for row in quarantined]},indent=2))


if __name__=='__main__':
    main()
