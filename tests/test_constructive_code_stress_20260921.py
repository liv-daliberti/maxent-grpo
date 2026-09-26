from __future__ import annotations
import json
import pytest
import evaluate_constructive_code_stress_20260921 as stress
import build_constructive_code_wider_20260921 as b


def fixture(tmp_path, response_change=None, attempt_change=None):
    response = {"task_id":"1016_D","sample_index":0,"request_seed":1,"text":"print(1)","text_sha256":b.raw_sha256("print(1)"),"prompt_sha256":"a"*64}
    attempt = {**{k:v for k,v in response.items() if k!="text"},"accepted":True,"hard_violations":[]}
    response.update(response_change or {});attempt.update(attempt_change or {})
    output={"artifacts":{}}
    for name,row in (("responses",response),("attempts",attempt)):
        path=tmp_path/(name+'.jsonl');path.write_text(json.dumps(row)+'\n')
        output["artifacts"][name]={"path":str(path),"sha256":b.digest(path)}
    return output


def test_source_attempt_binding_preserved(tmp_path):
    assert stress.bound_rows(fixture(tmp_path))["attempts"][("1016_D",0)]["accepted"]


def test_even_rehashed_sidecar_cannot_reassign_request(tmp_path):
    with pytest.raises(ValueError,match="identity mismatch"):
        stress.bound_rows(fixture(tmp_path,attempt_change={"request_seed":2}))


def test_raw_program_must_match_recorded_digest(tmp_path):
    with pytest.raises(ValueError,match="source-text SHA drift"):
        stress.bound_rows(fixture(tmp_path,response_change={"text":"print(2)"}))


def test_posthoc_sidecar_mutation_is_detected(tmp_path):
    evaluation=fixture(tmp_path)
    (tmp_path/'attempts.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match="sidecar identity drift"):
        stress.bound_rows(evaluation)
