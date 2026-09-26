#!/usr/bin/env python3
"""Prepare an immutable combined plan; never upload, delete, or launch workers."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
from archive_expanded_registry import ADDITIONAL_METHODS, ADDITIONAL_STUDIES
from archive_public_docs import METHODS, MODELS, STUDIES
import archive_completed_models as base

LABELS = {**{k: v['name'] for k,v in STUDIES.items()}, **{k: v['name'] for k,v in ADDITIONAL_STUDIES.items()}}
METHOD_KEYS = set(METHODS) | set(ADDITIONAL_METHODS)
WEIGHT = re.compile(r'(?:model(?:-\d{5}-of-\d{5})?\.safetensors|pytorch_model(?:-\d{5}-of-\d{5})?\.bin)')


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''):h.update(b)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_record(record, *, data_root=None):
    """Read only small receipts and file metadata; payload hashing stays in transfer."""
    data_root = Path(data_root or ROOT / 'var/data').resolve()
    require(record.get('audited_candidate') is True and record.get('endpoint_status') == 'admitted', 'Supplement is not independently admitted')
    require(record.get('source_experiment') in LABELS and record.get('source_arm') in METHOD_KEYS, 'Unknown supplemental scientific identity')
    require(record.get('model_key') in MODELS and type(record.get('seed')) is int, 'Unknown supplemental model/seed')
    require(record.get('terminal_step') == 3073, 'Supplemental export is outside the approved terminal-step convention')
    root, export = Path(record['run_dir']), Path(record['terminal_export'])
    require(root.is_absolute() and root.resolve(strict=True) == root and data_root in root.parents, 'Supplemental run is outside approved var/data')
    require(export.resolve(strict=True) == export and root in export.parents and not export.is_symlink(), 'Supplemental export differs or escapes run')
    receipt_path = Path(record['receipt_path'])
    require(receipt_path == root / 'TRAINING_COMPLETE.json' and not receipt_path.is_symlink(), 'Original completion receipt location differs')
    require(digest(receipt_path) == record['receipt_sha256'], 'Original completion receipt hash changed')
    receipt = read(receipt_path)
    require(receipt.get('schema') == 'oat_zero_training_complete_v1', 'Supplemental receipt schema differs')
    attempt = Path(receipt['terminal_attempt'])
    require(attempt.parent == root and attempt.name.startswith('debug_') and export.parent == attempt / 'saved_models', 'Supplemental attempt/export hierarchy differs')
    require(receipt['terminal_export'] == str(export) and receipt['terminal_step'] == record['terminal_step'], 'Supplemental terminal binding differs')
    require(record.get('all_related_job_ids') and not record.get('live_references'), 'Supplemental job identities/consumer evidence missing')
    names = record.get('large_weight_files', [])
    require(len(names) == len(set(names)) > 0 and all(WEIGHT.fullmatch(n) for n in names), 'Supplemental weight allowlist differs')
    files = {}
    for path in export.iterdir():
        info=path.lstat()
        require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), 'Supplemental export contains a nonregular file')
        files[path.name] = info.st_size
    require({n for n in files if WEIGHT.fullmatch(n)} == set(names), 'Supplemental weights are absent or differ')
    require(all(files[n] > 0 for n in names), 'Supplemental weight payload is empty')
    require(sum(files.values()) == record['terminal_bytes'], 'Supplemental export byte inventory changed')
    expected = {f['relative_path']:f['bytes'] for f in record['terminal_files']}
    require(files == expected and len(expected) == len(record['terminal_files']), 'Supplemental terminal file inventory changed')
    return {'files':len(files), 'bytes':sum(files.values()), 'weight_bytes':sum(files[n] for n in names)}


def compose_plan(original, supplemental, *, original_sha, inventory_sha, expected_initial=418,
                 expected_supplemental=286, logical_count=759, record_only_by_source=None):
    require(original.get('schema') == 'completed-model-hf-archive-plan-v1' and original.get('private') is False, 'Original plan contract differs')
    require(len(original['models']) == original['expected_model_count'] == expected_initial, 'Original selection count differs')
    records = supplemental['records']
    require(len(records) == expected_supplemental, 'Supplemental selection count differs')
    require(supplemental.get('ledger_pins'), 'Supplemental ledger pins are missing')
    require(supplemental.get('model_count', len(records)) == len(records), 'Supplemental model count differs')
    require(supplemental.get('terminal_bytes', sum(r['terminal_bytes'] for r in records)) == sum(r['terminal_bytes'] for r in records), 'Supplemental total byte count differs')
    pins = dict(original['ledger_pins'])
    for path, value in supplemental['ledger_pins'].items():
        require(re.fullmatch(r'[0-9a-f]{64}', value), 'Invalid supplemental ledger pin')
        require(path not in pins or pins[path] == value, 'Conflicting original/supplemental ledger pin: ' + path)
        pins[path] = value
    old_models = original['models']
    prefixes = {m['repo_prefix'] for m in old_models}
    ids = {m['archive_id'] for m in old_models}
    exports = {m['terminal_export'] for m in old_models}
    require(len(prefixes) == len(ids) == len(exports) == expected_initial, 'Original model identities collide')
    additions=[]
    for value in records:
        require(value.get('audited_candidate') is True and value.get('endpoint_status') == 'admitted', 'Unadmitted supplemental record')
        record=dict(value)
        require(record.get('source_experiment') in LABELS and record.get('source_arm') in METHOD_KEYS, 'Unregistered supplemental source or arm')
        name=MODELS[record['model_key']];exp=LABELS[record['source_experiment']]
        prefix=f"experiments/{exp}/{name}/{record['domain']}/{record['source_arm']}/seed-{record['seed']}/step-{record['terminal_step']:05d}"
        archive_id=hashlib.sha256(prefix.encode()).hexdigest()[:24]
        if 'repo_prefix' in record:require(record['repo_prefix'] == prefix, 'Supplemental preassigned prefix differs')
        if 'archive_id' in record:require(record['archive_id'] == archive_id, 'Supplemental preassigned archive id differs')
        require(prefix not in prefixes and archive_id not in ids and record['terminal_export'] not in exports, 'Scientific or physical export collision')
        prefixes.add(prefix);ids.add(archive_id);exports.add(record['terminal_export'])
        record.update(repo_prefix=prefix,archive_id=archive_id)
        additions.append(record)
    additions.sort(key=lambda r:(-r['terminal_bytes'],r['repo_prefix']))
    models=old_models+additions
    total=sum(r['terminal_bytes'] for r in models)
    require(total < 9_000_000_000_000, 'Expanded plan exceeds approved public allocation ceiling')
    record_only=dict(record_only_by_source or {})
    if 'missing_model_records' in supplemental:
        missing=supplemental['missing_model_records']
        require(supplemental.get('missing_model_count') == len(missing) and dict(Counter(r['source_experiment'] for r in missing)) == record_only, 'Record-only source census differs')
        require(all(r.get('endpoint_admitted') is True and r.get('weight_availability') == 'missing_terminal_weights' and not r.get('audited_candidate') for r in missing), 'Record-only evidence does not identify admitted results with absent weights')
    require(all(k in LABELS and type(v) is int and v>=0 for k,v in record_only.items()), 'Invalid record-only coverage')
    require(logical_count == len(models)+sum(record_only.values()), 'Logical records and deployable exports do not reconcile')
    result={**original,'models':models,'ledger_pins':pins,'expected_model_count':len(models),'expected_total_export_bytes':total,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'extension':{'schema':'paper-model-archive-plan-extension-v1','original_plan_sha256':original_sha,
                         'supplemental_inventory_sha256':inventory_sha,'original_models_preserved':len(old_models),
                         'original_model_records_sha256':object_sha(old_models),'appended_models':len(additions),'supplemental_audit_source_pins':supplemental.get('audit_source_pins', {}),'supplemental_admission_proof':supplemental.get('admission_proof')},
            'paper_coverage':{'logical_model_records':logical_count,'deployable_model_exports':len(models),
                              'record_only_count':sum(record_only.values()),'record_only_by_source':record_only,
                              'record_only_status':'Scientific records retained; model weights unavailable. Excluded from downloadable-model catalog.'}}
    require(result['models'][:expected_initial] == original['models'], 'Original model records were altered')
    require(result['output_dir'] == original['output_dir'], 'Original state directory was altered')
    return result


def prepare(original_path, inventory_path, output, *, inventory_sha256, expected_supplemental=286, logical_count=759, record_only_by_source=None):
    original_path,inventory_path,output=map(lambda p:Path(p).resolve(),(original_path,inventory_path,output))
    require(digest(original_path) == original_path.with_suffix('.sha256').read_text().strip(), 'Original plan SHA differs')
    require(digest(inventory_path) == inventory_sha256, 'Supplemental inventory differs from reviewed SHA')
    original, supplemental=read(original_path),read(inventory_path)
    require(output.parent == Path(original['output_dir']).resolve() == original_path.parent, 'Expanded plan must share original state directory')
    require(output.name != original_path.name and not output.exists() and not output.with_suffix('.sha256').exists(), 'Expanded plan destination already exists or replaces original')
    require(supplemental.get('schema') == 'hf_all_paper_supplemental_models_inventory_v1', 'Supplemental inventory schema differs')
    require(supplemental.get('audit_source_pins') and supplemental.get('admission_proof'), 'Supplemental admission source pins missing')
    for path,value in supplemental['audit_source_pins'].items():require(digest(path)==value,'Endpoint audit source changed before preparation: '+path)
    proof=supplemental['admission_proof'];require(digest(proof['path'])==proof['sha256'],'Supplemental admission proof changed')
    candidate=compose_plan(original,supplemental,original_sha=digest(original_path),inventory_sha=digest(inventory_path),
                           expected_supplemental=expected_supplemental,logical_count=logical_count,record_only_by_source=record_only_by_source)
    for path,value in candidate['ledger_pins'].items():require(digest(path)==value,'Ledger changed before expanded-plan preparation: '+path)
    queue=base.queue_snapshot()
    checks=[]
    for record in candidate['models'][len(original['models']):]:
        check=validate_record(record);base.assert_inactive(record,queue)
        checks.append({'archive_id':record['archive_id'],'repo_prefix':record['repo_prefix'],**check})
    require(digest(original_path)==candidate['extension']['original_plan_sha256'] and digest(inventory_path)==candidate['extension']['supplemental_inventory_sha256'],'Preparation inputs changed')
    for path,value in candidate['ledger_pins'].items():require(digest(path)==value,'Ledger changed during expanded-plan preparation: '+path)
    with output.open('x') as stream:json.dump(candidate,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    sha=digest(output)
    with output.with_suffix('.sha256').open('x') as stream:stream.write(sha+'\n');stream.flush();os.fsync(stream.fileno())
    report={'status':'prepared_for_review','plan_sha256':sha,'original_models_unchanged':len(original['models']),
            'supplemental_models':len(checks),'output_dir_unchanged':True,'models_state_paths_unchanged':True,
            'source_files':{str(p):digest(p) for p in (Path(__file__),ROOT/'ops/archive_expanded_registry.py')},
            'paper_coverage':candidate['paper_coverage'],'supplemental_file_checks':checks,'scheduler_mutations':False,
            'weight_payload_bytes_read':0,'worker_launched':False}
    with output.with_name(output.stem+'.preparation.json').open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-plan',type=Path,required=True);parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inventory-sha256',required=True);parser.add_argument('--expected-supplemental',type=int,default=286)
    parser.add_argument('--logical-count',type=int,default=759);parser.add_argument('--record-only-e95',type=int,default=55)
    args=parser.parse_args()
    r=prepare(args.original_plan,args.inventory,args.output,inventory_sha256=args.inventory_sha256,expected_supplemental=args.expected_supplemental,
              logical_count=args.logical_count,record_only_by_source={'e95':args.record_only_e95})
    print(json.dumps({k:r[k] for k in ('status','plan_sha256','original_models_unchanged','supplemental_models','worker_launched')},indent=2))

if __name__=='__main__':main()
