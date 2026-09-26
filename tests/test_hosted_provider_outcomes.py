"""Explicit native policy metadata is separate from empty or malformed answers."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ops'))
import audit_hosted_provider_outcomes as outcomes
import audit_hosted_modebench_completion as completion
import evaluate_claude_modebench as claude
from test_hosted_modebench_completion import fixture, write_jsonl


def test_claude_empty_content_and_thinking_only_refusals_remain_distinct():
    body={'content':[],'stop_reason':'refusal','stop_details':{'type':'refusal','category':'cyber'}}
    empty=outcomes.classify_native(body,'anthropic_messages')
    assert empty['refusal'] and empty['answer_text_empty'] and empty['content_empty']
    assert not empty['thinking_only'] and empty['category_labels']==['cyber']
    body['content']=[{'type':'thinking','thinking':'provider exposed block'}]
    thinking=outcomes.classify_native(body,'anthropic_messages')
    assert thinking['refusal'] and thinking['thinking_only'] and not thinking['content_empty']
    totals=outcomes.aggregate([empty,thinking])
    assert totals['refusals']==totals['refusals_with_empty_answer_text']==2
    assert totals['refusals_with_empty_content']==totals['refusals_thinking_only']==1


def test_empty_answer_alone_does_not_imply_refusal():
    result=outcomes.classify_native({'content':[],'stop_reason':'end_turn'},'anthropic_messages')
    assert result['answer_text_empty'] and not result['refusal'] and not result['content_filtered']
    assert outcomes.aggregate([result])['nonrefusal_empty_answer_text']==1


def test_responses_explicit_refusal_block_and_content_filter_reason():
    refusal={'status':'completed','output':[{'type':'message','content':[{'type':'refusal','refusal':'blocked'}]}]}
    assert outcomes.classify_native(refusal,'responses')['refusal']
    filtered={'status':'incomplete','incomplete_details':{'reason':'content_filter'},'output':[]}
    result=outcomes.classify_native(filtered,'responses')
    assert result['content_filtered'] and not result['refusal']


def test_chat_native_filter_metadata_counts_only_true_flags():
    body={'choices':[{'index':0,'finish_reason':'stop','message':{'content':'answer'},
                      'content_filter_results':{'violence':{'filtered':False}}}]}
    assert not outcomes.classify_native(body,'chat_completions')['content_filtered']
    body['choices'][0]['content_filter_results']['violence']['filtered']=True
    result=outcomes.classify_native(body,'chat_completions')
    assert result['content_filtered'] and result['filtered_category_labels']==['violence']
    assert not result['refusal'] and not result['answer_text_empty']


def test_chat_refusal_field_is_explicit_but_ordinary_refusal_text_is_not_inferred():
    body={'choices':[{'index':0,'finish_reason':'stop','message':{'content':'I cannot help with that.'}}]}
    assert not outcomes.classify_native(body,'chat_completions')['refusal']
    body['choices'][0]['message']={'content':None,'refusal':'Provider refusal field'}
    assert outcomes.classify_native(body,'chat_completions')['refusal']


def test_full_audit_binds_refusal_counts_to_completed_native_evidence(tmp_path):
    items,samples,_=fixture(tmp_path)
    row=json.loads((tmp_path/'rows.jsonl').read_text())
    raw_path=tmp_path/samples[0]['raw_receipt']
    raw=json.loads(raw_path.read_text())
    raw['response'].update(content=[],stop_reason='refusal',stop_details={'type':'refusal','category':'cyber'})
    completion.atomic(raw_path,raw)
    samples[0]=claude.grade_receipt(items[0],raw,row,lambda *args:{'verified':False,'canonical_key':None,'graded_text':''})
    completion.atomic(tmp_path/'sample_receipts'/(samples[0]['sample_id']+'.json'),samples[0])
    write_jsonl(tmp_path/'samples.jsonl',samples)
    completion.audit(tmp_path,2)
    output=tmp_path/'separate_outcomes'
    result=outcomes.audit(tmp_path,output,expected_samples=2)
    assert result['status']=='complete' and result['totals']['refusals']==1
    assert result['non_python_refusals']==1
    assert result['cells']['level1/countdown']['refusal_category_counts']=={'cyber':1}
    assert result['sources']['samples.jsonl']['sha256']==completion.file_sha(tmp_path/'samples.jsonl')
    records=outcomes.read_jsonl(output/'provider_outcome_samples.jsonl')
    assert records[0]['native_stop_details']=={'type':'refusal','category':'cyber'}
    assert records[0]['raw_receipt_file_sha256']==hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert result['sample_outcomes']['sha256']==completion.file_sha(output/'provider_outcome_samples.jsonl')
    assert not (tmp_path/'provider_outcomes.json').exists()


def test_raw_receipt_drift_after_completion_blocks_classification(tmp_path):
    _,samples,_=fixture(tmp_path)
    completion.audit(tmp_path,2)
    raw_path=tmp_path/samples[0]['raw_receipt']
    raw=json.loads(raw_path.read_text());raw['response']['stop_reason']='refusal'
    completion.atomic(raw_path,raw)
    with pytest.raises(ValueError,match='Raw native receipt changed'):
        outcomes.audit(tmp_path,expected_samples=2)


def test_legacy_sample_without_optional_canonical_digest_uses_authenticated_file_hash(tmp_path):
    _,samples,_=fixture(tmp_path)
    for sample in samples:
        sample.pop('raw_receipt_sha256')
        completion.atomic(tmp_path/'sample_receipts'/(sample['sample_id']+'.json'),sample)
    write_jsonl(tmp_path/'samples.jsonl',samples)
    completion.audit(tmp_path,2)
    result=outcomes.audit(tmp_path,expected_samples=2)
    assert result['totals']['responses']==2
    records=outcomes.read_jsonl(tmp_path/'provider_outcome_samples.jsonl')
    for record in records:
        raw=json.loads((tmp_path/record['raw_receipt']).read_text())
        assert record['raw_receipt_sha256']==completion.sha(raw)
        assert record['raw_receipt_digest_present_in_source_sample'] is False
        assert record['raw_receipt_file_sha256']==completion.file_sha(tmp_path/record['raw_receipt'])
