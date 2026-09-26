"""Regression checks for stale evidence that can preserve aggregate statistics."""
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).resolve().parents[1]/'ops/summarize_frontier_comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def cohort(tmp_path):
    names=('manifest.json','rows.jsonl','datasets.json','requests.jsonl','samples.jsonl')
    for name in names:(tmp_path/name).write_text('{}\n')
    inventory={name:m.digest(tmp_path/name) for name in names}
    (tmp_path/'evidence_file_sha256.json').write_text(json.dumps(inventory))
    audit={'status':'pass','expected_responses':15360,'saved_samples':15360,'unique_response_ids':15360,
           'model':'model','evidence_inventory_sha256':m.compact_sha(inventory)}
    (tmp_path/'completion_audit.json').write_text(json.dumps(audit))
    (tmp_path/'normalized_samples.jsonl').write_text('{}\n')
    summary={'models_returned':{'model':15360},'run_configuration':{'model':'model'},
             'normalized_secondary':{'cache_sha256':m.digest(tmp_path/'normalized_samples.jsonl')}}
    return tmp_path,summary,audit

def test_valid_bound_evidence(cohort):
    root,summary,audit=cohort
    assert m.validate_evidence(root,summary,'model')==audit

def test_cache_append_with_unchanged_parsed_records_is_rejected(cohort):
    root,summary,_=cohort
    with (root/'normalized_samples.jsonl').open('a') as handle:handle.write('\n')
    with pytest.raises(ValueError,match='Normalized cache changed'):
        m.validate_evidence(root,summary,'model')

def test_changed_selected_rows_are_rejected(cohort):
    root,summary,_=cohort
    (root/'rows.jsonl').write_text('{"changed":true}\n')
    with pytest.raises(ValueError,match='Selected cohort differs'):
        m.validate_evidence(root,summary,'model')

def test_replaced_inventory_requires_new_matching_audit(cohort):
    root,summary,_=cohort
    inventory=json.loads((root/'evidence_file_sha256.json').read_text());inventory['extra']='not-original'
    (root/'evidence_file_sha256.json').write_text(json.dumps(inventory))
    with pytest.raises(ValueError,match='inventory changed'):
        m.validate_evidence(root,summary,'model')

@pytest.mark.parametrize('field,value',[('saved_samples',15359),('unique_response_ids',15359),('model','different')])
def test_stale_cohort_identity_rejected(cohort,field,value):
    root,summary,audit=cohort;audit[field]=value
    (root/'completion_audit.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError):m.validate_evidence(root,summary,'model')
