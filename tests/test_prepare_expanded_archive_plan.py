"""Offline preservation and admission checks for a combined archive plan."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ops'))
spec=importlib.util.spec_from_file_location('expanded_plan_under_test',ROOT/'ops/prepare_expanded_archive_plan.py')
plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)

@pytest.fixture
def inputs():
    old=json.loads((ROOT/'var/artifacts/hf_model_archive_20260911/all/plan.json').read_text())
    new=json.loads((ROOT/'var/artifacts/hf_paper_full_inventory_20260911/supplemental_inventory_final.json').read_text())
    return old,new

def combine(old,new):
    return plan.compose_plan(old,new,original_sha='a'*64,inventory_sha='b'*64,record_only_by_source={'e95':55})

def test_original418_and_all_state_paths_unchanged(inputs):
    old,new=inputs;before=copy.deepcopy(old);combined=combine(old,new)
    assert old==before
    assert combined['models'][:418]==old['models']
    assert combined['output_dir']==old['output_dir']
    assert len(combined['models'])==704
    assert len({r['archive_id'] for r in combined['models']})==704
    assert len({r['repo_prefix'] for r in combined['models']})==704
    assert combined['paper_coverage']['logical_model_records']==759
    assert combined['paper_coverage']['record_only_count']==55
    assert not any(r['source_experiment']=='e95' for r in combined['models'])
    assert combined['expected_total_export_bytes']==old['expected_total_export_bytes']+new['terminal_bytes']
    assert len(combined['extension']['supplemental_audit_source_pins'])==292

def test_e85_two_repair_arms_do_not_collide(inputs):
    value=combine(*inputs)
    rows=[r for r in value['models'] if r['source_experiment']=='e85']
    assert len(rows)==10
    assert {r['source_arm'] for r in rows}=={'semantic','semantic_only'}
    for seed in range(43,48):assert len({r['repo_prefix'] for r in rows if r['seed']==seed})==2

@pytest.mark.parametrize('change',['pin','prefix','unadmitted','original_export','missing_census','missing_audit'])
def test_bad_extensions_fail_before_plan_creation(inputs,change):
    old,new=inputs
    if change=='pin':new['ledger_pins'][next(iter(old['ledger_pins']))]='f'*64
    elif change=='prefix':
        new['records'][1]=copy.deepcopy(new['records'][0])
        new['terminal_bytes']=sum(r['terminal_bytes'] for r in new['records'])
    elif change=='unadmitted':new['records'][0]['audited_candidate']=False
    elif change=='original_export':new['records'][0]['terminal_export']=old['models'][0]['terminal_export']
    elif change=='missing_census':new['missing_model_count']=54
    elif change=='missing_audit':new['missing_model_records'][0]['endpoint_admitted']=False
    with pytest.raises(ValueError):combine(old,new)

@pytest.fixture
def local_record(tmp_path):
    root=tmp_path/'var/data/run';export=root/'debug_example/saved_models/step_03073';export.mkdir(parents=True)
    (export/'model.safetensors').write_bytes(b'weight fixture only')
    (export/'config.json').write_text('{}')
    receipt=root/'TRAINING_COMPLETE.json';receipt.write_text(json.dumps({'schema':'oat_zero_training_complete_v1','terminal_attempt':str(export.parent.parent),'terminal_export':str(export),'terminal_step':3073}))
    files=[{'relative_path':p.name,'bytes':p.stat().st_size} for p in export.iterdir()]
    return {'audited_candidate':True,'endpoint_status':'admitted','source_experiment':'e97','source_arm':'ucpo','model_key':'qwen05b','seed':43,'terminal_step':3073,'run_dir':str(root),'terminal_export':str(export),'receipt_path':str(receipt),'receipt_sha256':plan.digest(receipt),'all_related_job_ids':['123'],'live_references':[],'large_weight_files':['model.safetensors'],'terminal_bytes':sum(f['bytes'] for f in files),'terminal_files':files},root.parent

def test_small_metadata_validation_preserves_payloads(local_record):
    record,data_root=local_record;path=Path(record['terminal_export'])/'model.safetensors';before=path.read_bytes()
    report=plan.validate_record(record,data_root=data_root)
    assert report['weight_bytes']==len(before) and path.read_bytes()==before

@pytest.mark.parametrize('change',['weight_missing','weight_changed','receipt_changed','symlink'])
def test_changed_or_missing_supplemental_sources_rejected(local_record,change):
    record,data_root=local_record;weight=Path(record['terminal_export'])/'model.safetensors'
    if change=='weight_missing':weight.unlink()
    elif change=='weight_changed':weight.write_bytes(b'changed')
    elif change=='receipt_changed':Path(record['receipt_path']).write_text('{}')
    else:weight.unlink();weight.symlink_to('config.json')
    with pytest.raises(ValueError):plan.validate_record(record,data_root=data_root)

def test_registry_covers_all_deployable_and_record_only_sources(inputs):
    from archive_expanded_registry import registry_for
    from archive_public_docs import render_readmes
    value=combine(*inputs);studies,methods=registry_for(value)
    rows=[]
    for r in value['models']:
        rows.append({'experiment':plan.LABELS[r['source_experiment']],'model':plan.MODELS[r['model_key']],'domain':r['domain'],'method':r['source_arm'],'seed':r['seed'],'step':r['terminal_step'],'repo_prefix':r['repo_prefix'],'commit_sha':'a'*40})
    catalog={'repo_id':value['repo_id'],'expected_model_count':704,'verified_model_count':704,'models':rows}
    pages=render_readmes(value,catalog,additional_studies=studies,additional_methods=methods)
    assert 'experiments/E95/README.md' in pages
    assert '/tree/' not in pages['experiments/E95/README.md']
    assert 'semantic_only' in pages['experiments/E85/README.md']
