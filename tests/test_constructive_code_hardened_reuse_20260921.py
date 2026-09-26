import copy
import json
import pytest
import seal_constructive_code_hardened_larger_20260921 as seal


def panel(tmp_path, changed_adapter=False):
    b=seal.h.base;old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    record={key:key for key in seal.FIELDS};record['source_problem_id']='task';record['split']='development'
    newer=copy.deepcopy(record);newer['split']='train'
    if changed_adapter:newer['task_adapter']='different-dispatch'
    tasks=[]
    for root,row in ((old,record),(new,newer)):
        folder=root/'task';folder.mkdir();row['task_record_sha256']=b.canonical_hash(row)
        b.write_json(folder/'task.json',row);b.write_json(root/'manifest.json',{'tasks':[{'source_problem_id':'task','relative_path':'task'}]})
        (folder/'audit_replays.jsonl').write_text('a fixed source audit ledger\n')
        b.write_json(folder/'admission_audit.json',{'status':'pass'})
    b.write_json(new/'replay_reuse_manifest.json',{'reused_problem_ids':['task']})
    audit={'task_id':'task','old_task_record_sha256':record['task_record_sha256'],'new_task_record_sha256':newer['task_record_sha256'],'reference_replays_sha256':b.digest(old/'task/audit_replays.jsonl'),'admission_audit_sha256':b.digest(new/'task/admission_audit.json')}
    receipt={'status':'pass','initial_manifest_sha256':b.digest(old/'manifest.json'),'replay_reuse_manifest_sha256':b.digest(new/'replay_reuse_manifest.json'),'tasks':[audit]}
    return old,new,receipt


def test_split_metadata_change_preserves_source_replay_contract(tmp_path):
    old,new,receipt=panel(tmp_path)
    assert seal.verify_reuse(new,old,receipt)==1


def test_new_adapter_dispatch_cannot_reuse_same_source_audit(tmp_path):
    old,new,receipt=panel(tmp_path,changed_adapter=True)
    with pytest.raises(ValueError,match='contract drift'):
        seal.verify_reuse(new,old,receipt)


def test_changed_reference_receipt_bytes_prevent_reuse(tmp_path):
    old,new,receipt=panel(tmp_path)
    (new/'task/audit_replays.jsonl').write_text('a different verdict\n')
    with pytest.raises(ValueError,match='admission bytes drift'):
        seal.verify_reuse(new,old,receipt)
