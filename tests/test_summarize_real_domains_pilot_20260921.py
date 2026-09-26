"""Tamper, denominator, replay-reconstruction and paired-audit regression tests."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest

spec = spec_from_file_location("real_domains_auditor", Path(__file__).parents[1]/"ops/summarize_real_domains_pilot_20260921.py")
audit = module_from_spec(spec)
spec.loader.exec_module(audit)


class Codec:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(t for t in tokens if t != 0).decode()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+"\n")


def rows(path, values):
    path.write_text("".join(json.dumps(v, sort_keys=True)+"\n" for v in values))


def verdict(letter):
    return {"accepted":letter in "AB", "canonical_key":{"A":"reuters_topic:copper","B":"reuters_topic:nickel"}.get(letter), "hard_violations":[], "receipt":{"response_sha256":audit.sha_bytes(letter.encode())}}


def source(tmp_path):
    path=tmp_path/"data.jsonl"
    records=[{"id":task,"options":[{"display":"A","text":"copper","id":"reuters_topic:copper","label":1},{"display":"B","text":"nickel","id":"reuters_topic:nickel","label":1},{"display":"C","text":"wheat","id":"reuters_topic:wheat","label":0}],"gold_topic_ids":["reuters_topic:copper","reuters_topic:nickel"]} for task in ("train","dev")]
    rows(path,records)
    return {"records_path":str(path),"records_sha256":audit.sha_file(path)}


def reported(verdicts):
    m=audit.metrics(verdicts)
    return {"task_id":"dev","family":"qa","samples":m["samples"],"accepted":m["accepted"],"mode_counts":m["mode_counts"],"pcmd":m["pcmd"],"pcmd_eligible":m["pcmd_eligible"], **{f"distinct_valid_at_{k}":v for k,v in m["expected_distinct_valid_modes_at_k"].items() if k != "1"}}


def training(tmp_path, arm, data):
    path=tmp_path/arm; path.mkdir()
    config={"adapter_module":"oat_drgrpo.noncoding_multi_answer_sata","adapter_config":data,"updates":2,"group_size":2,"generation_batch_size":1,"max_new_tokens":4,"replay_capacity":16,"train_ids":["train"],"eval_ids":["dev"],"eval_samples":32,"initial_evaluation":True,"seed":7,"eval_seed":8,"model":"same-model","learning_rate":1e-5}
    tasks=[{"task_id":t,"family":"qa","split":"train" if t=="train" else "dev","prompt_sha256":audit.sha_bytes(("prompt "+t).encode()),"prompt_tokens":len("prompt "+t)} for t in ("train","dev")]
    identity={"arm":arm,"config":config,"config_sha256":audit.object_sha(config),"resume":None,"objective":{"initial_bank":"empty","compute_only":arm=="maxrl","extra_reward_or_entropy_terms":False},"tasks":tasks,"initial_trainable_parameters_sha256":"same-parameters","production_source_sha256":{"maxrl":"same-source"},"runner_sha256":"runner","adapter_module_sha256":"adapter","model_config_sha256":"model-config"}
    write(path/"identity.json",identity)
    candidates=[]; verified=[]; phase_results={}
    for phase,step,task,count,base in [("initial",0,"dev",32,8),("train",0,"train",2,1000007),("train",1,"train",2,1010007),("final",2,"dev",32,8)]:
        vs=[]
        for sample in range(count):
            letter="A" if phase=="final" and arm=="maxrl" else "AB"[sample%2]
            raw={"request_id":f"{phase}:{step}:{task}:{sample}","task_id":task,"phase":phase,"step":step,"sample_index":sample,"batch_seed":base+sample,"batch_offset":0,"token_ids":[ord(letter),0],"text":letter,"prompt_sha256":audit.sha_bytes(("prompt "+task).encode()),"prompt_token_ids":list(("prompt "+task).encode()),"finish_reason":"eos"}
            v=verdict(letter); candidates.append(raw);verified.append({**raw,"verdict":v});vs.append(v)
        if phase != "train":
            phase_results[phase]=[reported(vs)]
            write(path/f"{phase}.json",{"completed_updates":step,"tasks":phase_results[phase]})
    rows(path/"candidates.jsonl",candidates);rows(path/"verified.jsonl",verified)
    metrics=[{"completed_updates":step+1,"task_id":"train","optimizer_updates":step+1,"fresh_successes":2,"fresh_tokens":4,"mixed_group":0,"new_modes":2 if step==0 else 0,"bank_tasks":1,"bank_modes":2,"replay_task_id":"train","replay_modes":2,"canonical_replay_applied_score_gradient_l2":1.0 if arm=="remax" else 0.0} for step in range(2)]
    rows(path/"metrics.jsonl",metrics)
    checkpoint=path/"checkpoint-2";checkpoint.mkdir();(checkpoint/"adapter").mkdir()
    (checkpoint/"adapter/adapter.safetensors").write_bytes(b"weights")
    (checkpoint/"training.pt").write_bytes(b"sealed optimizer bytes never deserialized")
    modes={}
    for sample in verified:
        if sample["phase"]=="train":
            key=sample["verdict"]["canonical_key"]
            if key not in modes:
                modes[key]={"token_ids":sample["token_ids"],"fresh_count":1,"request_id":sample["request_id"],"text_sha256":audit.sha_bytes(sample["text"].encode())}
            else:modes[key]["fresh_count"]+=1
    write(checkpoint/"bank.json",{"capacity":16,"entries":{"train":{"prompt":list(b"prompt train"),"modes":modes}},"cursor":2})
    write(checkpoint/"complete.json",{"arm":arm,"config_sha256":identity["config_sha256"],"completed_updates":2,"bank_sha256":audit.sha_file(checkpoint/"bank.json"),"training_state_sha256":audit.sha_file(checkpoint/"training.pt"),"adapter_files":{"adapter.safetensors":audit.sha_file(checkpoint/"adapter/adapter.safetensors")}})
    write(path/"result.json",{"status":"complete","arm":arm,"config_sha256":identity["config_sha256"],"completed_updates":2,"evaluation":phase_results})
    return path


def capability(tmp_path,data):
    path=tmp_path/"evaluation.json"
    config={"adapter_module":"oat_drgrpo.noncoding_multi_answer_sata","adapter_config":data,"task_ids":["dev"],"samples_per_task":32,"seed":17}
    raw=[];verified=[]
    for i in range(32):
        letter="AB"[i%2] if i<16 else "C"
        r={"task_id":"dev","sample_index":i,"request_seed":17+i,"prompt_sha256":audit.sha_bytes(b"prompt dev"),"text":letter,"text_sha256":audit.sha_bytes(letter.encode()),"token_ids":[ord(letter),0],"token_count":2,"finish_reason":"stop"}
        raw.append(r);verified.append({**{k:v for k,v in r.items() if k not in ("text","token_ids")},**verdict(letter)})
    rp=tmp_path/"responses.jsonl";ap=tmp_path/"attempts.jsonl";rows(rp,raw);rows(ap,verified)
    dataset={"source":"frozen-fixture"}
    write(path,{"status":"complete","config":config,"dataset_identity":dataset,"dataset_identity_sha256":audit.object_sha(dataset),"task_prompts":[{"task_id":"dev","prompt":"prompt dev","prompt_sha256":audit.sha_bytes(b"prompt dev")}],"artifacts":{"responses":{"path":str(rp),"sha256":audit.sha_file(rp)},"attempts":{"path":str(ap),"sha256":audit.sha_file(ap)}},"task_results":[{"task_id":"dev",**audit.metrics(verified)}]})
    return path


def test_capability_raw_bijection_native_regrade_and_denominator(tmp_path):
    data=source(tmp_path);path=capability(tmp_path,data)
    result=audit.audit_evaluation(path,Codec())
    m=result["task_metrics"]["dev"]
    assert m["accuracy"]==.5 and m["annotated_topic_coverage"]==1
    assert not m["pcmd_eligible"] and m["pcmd"] is None
    assert m["expected_distinct_valid_modes_at_k"]["32"]==2
    raw=audit.read_rows(tmp_path/"responses.jsonl");raw[0]["text"]="C";rows(tmp_path/"responses.jsonl",raw)
    with pytest.raises(audit.AuditError,match="sidecar hash"):
        audit.audit_evaluation(path,Codec())


def test_valid_pair_reconstructs_bank_and_reports_finite_mode_deltas(tmp_path):
    data=source(tmp_path);a=training(tmp_path,"maxrl",data);b=training(tmp_path,"remax",data)
    result=audit.audit_pair(a,b,(Codec(),Codec()))
    assert result["status"]=="pass"
    assert result["arms"][0]["replay_gradient_positive_updates"]==0
    assert result["arms"][1]["replay_gradient_positive_updates"]==2
    assert result["paired_final_remax_minus_maxrl"]["dev"]["expected_distinct_at_k"]["32"]==1
    assert result["paired_final_remax_minus_maxrl"]["dev"]["accuracy"]==0


@pytest.mark.parametrize("tamper,pattern",[("duplicate","duplicate request"),("tokens","original raw"),("bank","first raw"),("checkpoint","adapter content"),("gradient","wrong arm"),("updates","before requested"),("initial","initial LoRA")])
def test_material_tampering_cannot_pass(tmp_path,tamper,pattern):
    data=source(tmp_path);a=training(tmp_path,"maxrl",data);b=training(tmp_path,"remax",data)
    if tamper=="duplicate":
        r=audit.read_rows(a/"candidates.jsonl");r.append(r[0]);rows(a/"candidates.jsonl",r)
    elif tamper=="tokens":
        r=audit.read_rows(a/"verified.jsonl");r[0]["token_ids"]=[67,0];rows(a/"verified.jsonl",r)
    elif tamper=="bank":
        p=a/"checkpoint-2/bank.json";r=audit.read_json(p);r["cursor"]=100;write(p,r)
        s=audit.read_json(a/"checkpoint-2/complete.json");s["bank_sha256"]=audit.sha_file(p);write(a/"checkpoint-2/complete.json",s)
    elif tamper=="checkpoint":
        (a/"checkpoint-2/adapter/adapter.safetensors").write_bytes(b"tampered")
    elif tamper=="gradient":
        r=audit.read_rows(a/"metrics.jsonl");r[0]["canonical_replay_applied_score_gradient_l2"]=1;rows(a/"metrics.jsonl",r)
    elif tamper=="updates":
        r=audit.read_json(a/"result.json");r["completed_updates"]=1;write(a/"result.json",r)
    elif tamper=="initial":
        r=audit.read_json(b/"identity.json");r["initial_trainable_parameters_sha256"]="different";write(b/"identity.json",r)
    with pytest.raises(audit.AuditError,match=pattern):audit.audit_pair(a,b,(Codec(),Codec()))


def test_recomputed_gold_disagrees_even_if_receipts_resealed(tmp_path):
    data=source(tmp_path);p=capability(tmp_path,data)
    r=audit.read_rows(tmp_path/"attempts.jsonl");r[0]["accepted"]=False;r[0]["canonical_key"]=None;rows(tmp_path/"attempts.jsonl",r)
    receipt=audit.read_json(p);receipt["artifacts"]["attempts"]["sha256"]=audit.sha_file(tmp_path/"attempts.jsonl");write(p,receipt)
    with pytest.raises(audit.AuditError,match="independently regraded"):
        audit.audit_evaluation(p,Codec())


def test_raw_text_and_tokens_are_bound_even_with_resealed_files(tmp_path):
    data=source(tmp_path);p=capability(tmp_path,data)
    r=audit.read_rows(tmp_path/"responses.jsonl");r[0]["token_ids"]=[67,0];rows(tmp_path/"responses.jsonl",r)
    receipt=audit.read_json(p);receipt["artifacts"]["responses"]["sha256"]=audit.sha_file(tmp_path/"responses.jsonl");write(p,receipt)
    with pytest.raises(audit.AuditError,match="decode"):
        audit.audit_evaluation(p,Codec())


def test_pcmd_threshold_and_unknown_both_domain_gate():
    assert audit.metrics([verdict("A")]*29)["pcmd"] is None
    assert audit.metrics([verdict("A")]*30)["pcmd"]==0
    r=audit.report({}, {})
    assert r["readiness_gate"]=="unknown"
    assert "not treatment efficacy" in " ".join(r["interpretation"])


def test_accounting_uses_allocations_not_job_steps(tmp_path):
    p=tmp_path/"sacct.psv"
    p.write_text("JobIDRaw|ElapsedRaw|AllocTRES|State|ExitCode|\n123|1800|cpu=8,gres/gpu=2|COMPLETED|0:0|\n123.batch|1800|cpu=8|COMPLETED|0:0|\n")
    result=audit.accounting([p])
    assert result["allocated_gpu_hours"]==1 and len(result["jobs"])==1
    assert not result["online_scheduler_queried"]


def test_only_proven_frozen_paths_may_differ(tmp_path):
    job=tmp_path/"job";directory=job/"training";directory.mkdir(parents=True)
    write(job/"identity.json",{"schema":"real-domains-frozen-job-20260921-v1","request":{"freeze_data_roots":["/original/data"]}})
    config={"data":str((job/"bundle/data/0/file.json").resolve()),"temperature":1,"unproven":"/other/path"}
    assert audit.normalized_config(config,directory)=={"data":"/original/data/file.json","temperature":1,"unproven":"/other/path"}


def test_code_requires_stable_full_suite_receipts():
    text="print(1)";h=audit.sha_bytes(text.encode())
    inner={"emitted_text_sha256":h,"executed_source_sha256":h,"fence_stripped":False,"accepted":True,"canonical_key":"behavior:x","terminal_worker_record":True,"hard_violations":[],"released_checker_accepted":True,"wrapper_accepted":True,"execution":{"suite_tests":2,"executed_tests":2}}
    v={"accepted":True,"canonical_key":"behavior:x","hard_violations":[],"receipt":{**inner,"stability_recheck_required":True,"stability_recheck":deepcopy(inner)}}
    audit.inspect_verdict(text,v,"code")
    v["receipt"]["stability_recheck"]["execution"]["executed_tests"]=1
    with pytest.raises(audit.AuditError,match="full suite"):
        audit.inspect_verdict(text,v,"code")


def endpoint(tmp_path, label, data, training_path=None):
    import shutil
    directory=tmp_path/("endpoint_"+label);directory.mkdir()
    path=capability(directory,data);r=audit.read_json(path)
    r['config']['task_ids']=['dev','train']
    r['task_prompts'][0].update(split='dev',family='qa')
    r['task_prompts'].append({'task_id':'train','prompt':'prompt train','prompt_sha256':audit.sha_bytes(b'prompt train'),'split':'train','family':'qa'})
    for key in ('responses','attempts'):
        p=Path(r['artifacts'][key]['path']);rr=audit.read_rows(p)
        additions=[]
        for old in rr:
            n=deepcopy(old);n.update(task_id='train',request_seed=n['request_seed']+10000,prompt_sha256=audit.sha_bytes(b'prompt train'));additions.append(n)
        rows(p,rr+additions);r['artifacts'][key]['sha256']=audit.sha_file(p)
    m=deepcopy(r['task_results'][0]);m['task_id']='train';r['task_results'].append(m)
    if training_path:
        checkpoint=directory/'checkpoint';shutil.copytree(training_path/'checkpoint-2',checkpoint)
        seal=audit.read_json(checkpoint/'complete.json')
        r['config'].update(lora_path=str(checkpoint/'adapter'),lora_arm=label,lora_completed_updates=2,lora_checkpoint_config_sha256=seal['config_sha256'],max_lora_rank=16)
        r['lora_checkpoint']={'seal':seal,'seal_sha256':audit.sha_file(checkpoint/'complete.json')}
        r['lora_files']=seal['adapter_files']
    write(path,r)
    return path


def endpoint_fixture(tmp_path):
    data=source(tmp_path)
    rr=audit.read_rows(data['records_path'])
    for r in rr:r['split']=r['id']
    rows(Path(data['records_path']),rr);data['records_sha256']=audit.sha_file(data['records_path'])
    a=training(tmp_path,'maxrl',data);b=training(tmp_path,'remax',data)
    pair=audit.audit_pair(a,b,(Codec(),Codec()))
    paths=[endpoint(tmp_path,'base',data),endpoint(tmp_path,'maxrl',data,a),endpoint(tmp_path,'remax',data,b)]
    return data,pair,paths


def test_endpoint_strata_keep_all_trained_prompts_and_native_bank_support(tmp_path):
    data,pair,paths=endpoint_fixture(tmp_path)
    result=audit.audit_endpoints(paths,pair,[Codec()]*3)
    assert result['status']=='pass'
    assert set(result['strata'])=={'dev','trained_train'}
    assert result['strata']['trained_train']['task_ids']==['train']
    assert result['trained_prompt_table']['train']['remax']['discovered_bank_modes']==['reuters_topic:copper','reuters_topic:nickel']
    assert result['trained_prompt_table']['train']['remax']['bank_topic_observed_fraction']==1
    assert result['training_regime']=='mechanistic ceiling/zero-fresh-gradient smoke'


@pytest.mark.parametrize('change,pattern',[('sampling','configuration mismatch'),('seal','audited final checkpoint'),('split','split differs')])
def test_endpoint_pair_cannot_mix_sampling_checkpoint_or_split(tmp_path,change,pattern):
    data,pair,paths=endpoint_fixture(tmp_path);r=audit.read_json(paths[2])
    if change=='sampling':r['config']['max_tokens']=99
    elif change=='seal':r['lora_checkpoint']['seal_sha256']='changed'
    else:r['task_prompts'][0]['split']='train'
    write(paths[2],r)
    with pytest.raises(audit.AuditError,match=pattern):audit.audit_endpoints(paths,pair,[Codec()]*3)


def provenance_fixture(tmp_path):
    data=source(tmp_path);rr=audit.read_rows(data['records_path']);source_receipts=[]
    for i,r in enumerate(rr):
        r.update(split=r['id'],paragraph='Original news paragraph '+str(i),source_row_index=i)
        source_receipts.append({'source_index':i,'paragraph_sha256':audit.sha_bytes(r['paragraph'].encode()),'sata_gold':['copper','nickel'],'sata_distractors':['wheat'],'match_count':1,'matches':[{'reuters_id':str(i),'original_topics':['copper','nickel']}]})
    rows(Path(data['records_path']),rr);data['records_sha256']=audit.sha_file(data['records_path'])
    manifest=tmp_path/'manifest.json';write(manifest,{'source':{'raw_sha256':'original-release'}});data['manifest_path']=str(manifest)
    archive=tmp_path/'reuters_uci.zip';archive.write_bytes(b'original archive fixture')
    path=tmp_path/'provenance.json';write(path,{'schema':'sata-reuters-independent-source-comparison-v3','sata_release_sha256':'original-release','archive_sha256':audit.sha_file(archive),'rows':source_receipts,'sata_news_rows':2,'unmatched':[],'original_gold_used_as_distractor':[],'uniquely_matched':2,'multiple_matches':[]})
    return path,{'adapter_module':'oat_drgrpo.noncoding_multi_answer_sata','adapter_config':data}


def test_original_source_provenance_binds_labels_and_disallows_story_leakage(tmp_path):
    path,config=provenance_fixture(tmp_path)
    assert audit.audit_qa_provenance(path,config)['frozen_rows_checked']==2
    r=audit.read_json(path);r['rows'][1]['matches'][0]['reuters_id']='0';write(path,r)
    with pytest.raises(audit.AuditError,match='leaks across splits'):audit.audit_qa_provenance(path,config)


def test_original_source_provenance_rejects_unannotated_gold_conflict(tmp_path):
    path,config=provenance_fixture(tmp_path)
    r=audit.read_json(path);r['rows'][0]['matches'][0]['original_topics'].append('wheat');write(path,r)
    with pytest.raises(audit.AuditError,match='positive topic used as distractor'):audit.audit_qa_provenance(path,config)


def test_frozen_source_and_literal_default_resolution_are_bound(tmp_path):
    root=tmp_path/'job';directory=root/'training';directory.mkdir(parents=True)
    runner=root/'bundle/ops/train_real_domains_pilot_20260921.py';runner.parent.mkdir(parents=True)
    runner.write_text('DEFAULTS = {"learning_rate": 1e-5}\n')
    adapter=root/'bundle/ops/fixture_adapter.py';adapter.write_text('ADAPTER = 1\n')
    model=tmp_path/'model';model.mkdir();write(model/'config.json',{'model':'fixture'})
    config={'model':str(model),'adapter_module':'fixture_adapter'};write(root/'config.json',config)
    launch={'schema':'real-domains-frozen-job-20260921-v1','config_sha256':audit.sha_file(root/'config.json'),'files':[{'snapshot':str(p),'sha256':audit.sha_file(p)} for p in (runner,adapter)]};write(root/'identity.json',launch)
    receipt={'config':{**config,'learning_rate':1e-5},'runner_sha256':audit.sha_file(runner),'adapter_module_sha256':audit.sha_file(adapter),'model_config_sha256':audit.sha_file(model/'config.json')}
    assert audit.audit_frozen_sources(directory,receipt,True)=='pass'
    bad=deepcopy(receipt);bad['config']['learning_rate']=1e-3
    with pytest.raises(audit.AuditError,match='sealed defaults'):audit.audit_frozen_sources(directory,bad,True)
    adapter.write_text('ADAPTER = 2\n')
    with pytest.raises(audit.AuditError,match='implementation hash'):audit.audit_frozen_sources(directory,receipt,True)


def test_registered_external_baseline_can_omit_internal_initial_eval(tmp_path):
    data=source(tmp_path);paths=[training(tmp_path,arm,data) for arm in ('maxrl','remax')]
    for p in paths:
        identity=audit.read_json(p/'identity.json');identity['config']['initial_evaluation']=False;identity['config_sha256']=audit.object_sha(identity['config']);write(p/'identity.json',identity)
        for name in ('candidates','verified'):
            rows(p/(name+'.jsonl'),[r for r in audit.read_rows(p/(name+'.jsonl')) if r['phase']!='initial'])
        result=audit.read_json(p/'result.json');result['config_sha256']=identity['config_sha256'];del result['evaluation']['initial'];write(p/'result.json',result)
        seal=audit.read_json(p/'checkpoint-2/complete.json');seal['config_sha256']=identity['config_sha256'];write(p/'checkpoint-2/complete.json',seal)
        (p/'initial.json').unlink()
    audited=audit.audit_pair(*paths,(Codec(),Codec()))
    assert audited['status']=='pass'
    assert set(audited['arms'][0]['evaluation'])=={'final'}
    assert audited['arms'][0]['initial_trainable_parameters_sha256']==audited['arms'][1]['initial_trainable_parameters_sha256']


def test_historical_false_acceptance_cannot_pass_readiness():
    old=audit.HISTORICAL_CODE_REVOCATION
    with pytest.raises(audit.AuditError,match='verifier revoked'):
        audit.require_unrevoked_code_manifest(old['manifest_sha256'])
    audit.require_unrevoked_code_manifest('different-hardened-manifest')
    summary=audit.report({'code':[{'status':'fail','reason':'historical verifier revoked'}]}, {})
    assert summary['readiness_gate']=='fail'
    assert summary['historical_code_quality_revocation']['stress_rejected_original_acceptances']==6
    assert 'six of fourteen' in audit.markdown(summary)


def reference_panel(tmp_path):
    import gzip
    directory=tmp_path/'reference_panel';directory.mkdir()
    (directory/'checker.cpp').write_text('// released checker fixture\n')
    (directory/'validator.cpp').write_text('// released input validator fixture\n')
    stdin='3\n';case={'stdin':stdin,'test_index':0,'input_bytes':2,'input_sha256':audit.sha_bytes(stdin.encode())}
    with gzip.open(directory/'admitted_inputs.jsonl.gz','wt',encoding='ascii') as handle:handle.write(json.dumps(case)+'\n')
    suite={'test_count':1,'compressed_jsonl_sha256':audit.sha_file(directory/'admitted_inputs.jsonl.gz'),'suite_sha256':audit.object_sha([{k:v for k,v in case.items() if k!='stdin'}])}
    record={'source_problem_id':'fixture','checker_sha256':audit.sha_file(directory/'checker.cpp'),'validator_sha256':audit.sha_file(directory/'validator.cpp'),'suite_file':'admitted_inputs.jsonl.gz','suite':suite,'suite_id':'hardened','task_adapter':'fixed-adapter'}
    refs=[];replays=[]
    for i in range(24):
        code=f'print({i})\n';key=audit.sha_bytes(code.encode());label='correct' if i<12 else 'incorrect'
        refs.append({'code':code,'submission_sha256':key,'known_label':label})
        replays.append({'submission_sha256':key,'known_label':label,'audit_violations':[],'released_checker_accepted':i<12,'wrapper_accepted':i<12,'source_problem_id':'fixture','checker_sha256':record['checker_sha256'],'suite_id':'hardened','suite_sha256':suite['suite_sha256'],'task_adapter':'fixed-adapter','execution':{'executed_tests':1,'suite_tests':1}})
    rows(directory/'py3_replays.jsonl',refs);rows(directory/'audit_replays.jsonl',replays)
    record['references']={'jsonl_sha256':audit.sha_file(directory/'py3_replays.jsonl')}
    admitted={'tpr':1.0,'tnr':1.0,'positive_replays':12,'negative_replays':12,'replays_sha256':audit.sha_file(directory/'audit_replays.jsonl')}
    return directory,record,admitted


def test_hardened_reference_rejections_recomputed_from_bound_panel(tmp_path):
    directory,record,admitted=reference_panel(tmp_path)
    assert audit.audit_hardened_reference_ledger(directory,record,admitted)['negative_rejected']==12
    replays=audit.read_rows(directory/'audit_replays.jsonl');replays[-1]['released_checker_accepted']=True;rows(directory/'audit_replays.jsonl',replays)
    admitted['replays_sha256']=audit.sha_file(directory/'audit_replays.jsonl')
    with pytest.raises(audit.AuditError,match='human-negative program accepted'):
        audit.audit_hardened_reference_ledger(directory,record,admitted)


def test_hardened_reference_reuse_requires_identical_effective_suite(tmp_path):
    directory,record,admitted=reference_panel(tmp_path)
    replays=audit.read_rows(directory/'audit_replays.jsonl');replays[0]['suite_sha256']='different';rows(directory/'audit_replays.jsonl',replays)
    admitted['replays_sha256']=audit.sha_file(directory/'audit_replays.jsonl')
    with pytest.raises(audit.AuditError,match='different checker/suite/task'):
        audit.audit_hardened_reference_ledger(directory,record,admitted)


def quality_header(tmp_path,monkeypatch):
    fixture=tmp_path/'fixed_source_probe_fixture.json';write(fixture,{'model_sampling_used_in_probe_selection':False,'cases':[{'task_id':'fixture'}]})
    probe_hash=audit.sha_file(fixture);monkeypatch.setattr(audit,'FIXED_SOURCE_PROBE_SHA256',probe_hash)
    write(tmp_path/'pre_hardening_manifest.json',{'source':'original manifest fixture'});source_hash=audit.sha_file(tmp_path/'pre_hardening_manifest.json')
    policy={'required_tpr':1.0,'required_tnr':1.0,'positive_replays':12,'negative_replays':12,'append_only':True,'checker_modified':False,'prompt_modified':False,'policy_outputs_used_for_test_selection':False}
    task={'task_id':'fixture','status':'pass','positive_replays':12,'negative_replays':12,'positive_accepted':12,'negative_rejected':12,'tpr':1.0,'tnr':1.0,'violations':[]}
    quality={'schema':'constructive-code-hardening-quality-20260921-v1','hardening_schema':audit.HARDENING_SCHEMA,'status':'pass','policy':policy,'fixed_probe_manifest_sha256':probe_hash,'source_manifest_sha256':source_hash,'tasks':[task],'admitted_problem_ids':['fixture'],'source_candidates':1,'admitted':1,'quarantined':0}
    write(tmp_path/'hardening_quality.json',quality)
    manifest={'hardening_quality_sha256':audit.sha_file(tmp_path/'hardening_quality.json'),'fixed_probe_manifest_sha256':probe_hash,'source_manifest_sha256':source_hash,'admitted_problem_ids':['fixture']}
    identity={'hardening_quality_sha256':manifest['hardening_quality_sha256'],'fixed_probe_manifest_sha256':probe_hash,'hardening_source_manifest_sha256':source_hash}
    return manifest,identity,quality


def test_hardening_global_quality_requires_exact_source_reference_policy(tmp_path,monkeypatch):
    manifest,identity,quality=quality_header(tmp_path,monkeypatch)
    assert audit.hardening_quality(tmp_path,manifest,identity)[0]['status']=='pass'
    quality['tasks'][0]['positive_accepted']=11
    write(tmp_path/'hardening_quality.json',quality)
    identity['hardening_quality_sha256']=manifest['hardening_quality_sha256']=audit.sha_file(tmp_path/'hardening_quality.json')
    with pytest.raises(audit.AuditError,match='exact reference gate'):audit.hardening_quality(tmp_path,manifest,identity)


def test_hardening_quality_cannot_hide_quarantined_task_in_primary_pool(tmp_path,monkeypatch):
    manifest,identity,quality=quality_header(tmp_path,monkeypatch)
    quality['tasks'][0]['status']='fail'
    write(tmp_path/'hardening_quality.json',quality)
    identity['hardening_quality_sha256']=manifest['hardening_quality_sha256']=audit.sha_file(tmp_path/'hardening_quality.json')
    with pytest.raises(audit.AuditError,match='admitted-task set mismatch'):audit.hardening_quality(tmp_path,manifest,identity)


def test_source_identity_relocation_preserves_module_path_not_basename():
    h='a'*64
    old={'/original/repo/src/package/helper.py':h}
    relocated={'/frozen/job/bundle/src/package/helper.py':h}
    wrong={'/frozen/job/bundle/ops/helper.py':h}
    assert audit.relocated_source_identity(old)==audit.relocated_source_identity(relocated)
    assert audit.relocated_source_identity(old)!=audit.relocated_source_identity(wrong)
    with pytest.raises(audit.AuditError,match='duplicate'):
        audit.relocated_source_identity({**old,**relocated})


def source_closure(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';root=bundle/'data/0';root.mkdir(parents=True)
    identities={}
    for relative in audit.VERIFIER_CRITICAL_PATHS:
        path=bundle/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(relative+' fixture source\n');identities[relative]=audit.sha_file(path)
    header_hash=identities['third_party/testlib/testlib.h'];monkeypatch.setattr(audit,'PINNED_TESTLIB_SHA256',header_hash)
    guard=bundle/'ops/build_constructive_code_hardened_20260921.py';guard.write_text('guarded loader fixture\n')
    q={'canonicalizer_source_files':{k:v for k,v in identities.items()if k.startswith('src/')},'verifier_support_files':{k:v for k,v in identities.items()if not k.startswith('src/')},'pinned_testlib_sha256':header_hash,'loader_guard_sha256':audit.sha_file(guard),'builder_sha256':audit.sha_file(guard)}
    return root,bundle,q


def test_verifier_source_closure_rejects_changed_transitive_checker_helper(tmp_path,monkeypatch):
    root,bundle,q=source_closure(tmp_path,monkeypatch)
    assert audit.audit_hardening_sources(root,q)['status']=='pass'
    (bundle/'ops/replay_constructive_code_review_slate.py').write_text('changed checker helper\n')
    with pytest.raises(audit.AuditError,match='support bytes differ'):
        audit.audit_hardening_sources(root,q)


def test_executed_hardener_remains_distinct_from_later_loader_guards(tmp_path,monkeypatch):
    root,bundle,q=source_closure(tmp_path,monkeypatch)
    old=root/'source_code/hardener_executed.py';old.parent.mkdir();old.write_text('original executed fixture\n')
    q['builder_sha256']=audit.sha_file(old)
    with pytest.raises(audit.AuditError,match='without explicit provenance'):
        audit.audit_hardening_sources(root,q)
    q['post_audit_provenance_verification']={'executed_hardener_source':'source_code/hardener_executed.py','executed_hardener_sha256':audit.sha_file(old),'canonicalizer_sources_byte_identical_to_full_replay':True,'support_sources_unchanged_since_before_hardener_execution':True}
    assert audit.audit_hardening_sources(root,q)['post_audit_supplement_declared']
    old.write_text('mutated original source\n')
    with pytest.raises(audit.AuditError,match='snapshot hash mismatch'):
        audit.audit_hardening_sources(root,q)


def test_qa_receipt_cannot_satisfy_coding_readiness(tmp_path):
    data=source(tmp_path);path=capability(tmp_path,data)
    verified=audit.audit_evaluation(path,Codec())
    assert verified['domain']=='qa'
    summary=audit.report({'code':[verified]}, {})
    assert summary['domains']['code']['readiness']=='fail'


def reuse_fixture(tmp_path):
    import shutil
    initial=tmp_path/'initial';initial.mkdir()
    old_dir,old,old_admitted=reference_panel(initial)
    old.update(problem_key='human:fixture',witness_family='unordered_set',limits={'time_milliseconds':1000},statement='Produce a witness.',statement_sha256=audit.sha_bytes(b'Produce a witness.'))
    old_admitted.update(audit_method='full_independent_12_positive_12_negative_replay',status='pass',violations=[],runtime={'image_sha256':'same-runtime'})
    write(old_dir/'admission_audit.json',old_admitted)
    old['admission_audit_sha256']=audit.sha_file(old_dir/'admission_audit.json');old['task_record_sha256']=audit.object_sha(old)
    write(old_dir/'task.json',old)
    sources={'src/fixture.py':'a'*64};support={'ops/fixture.py':'b'*64}
    old_quality={'canonicalizer_source_files':sources,'verifier_support_files':support}
    write(initial/'hardening_quality.json',old_quality)
    items=[{'source_problem_id':'fixture','relative_path':'reference_panel','task_record_sha256':old['task_record_sha256']}]
    write(initial/'manifest.json',{'tasks':items,'tasks_sha256':audit.object_sha(items),'hardening_quality_sha256':audit.sha_file(initial/'hardening_quality.json')})
    root=tmp_path/'larger';root.mkdir();directory=root/'reference_panel';shutil.copytree(old_dir,directory)
    contract={'problem_id':'fixture','suite_id':old['suite_id'],'suite_sha256':old['suite']['suite_sha256'],'inputs_in_order_sha256':audit.object_sha(['3\n']),'checker_sha256':old['checker_sha256'],'reference_ledger_sha256':old['references']['jsonl_sha256'],'limits':old['limits'],'statement_sha256':old['statement_sha256'],'witness_family':old['witness_family'],'canonicalizer_sources':sources,'verifier_support_sources':support}
    origin={'root':str(initial),'manifest_sha256':audit.sha_file(initial/'manifest.json'),'hardening_quality_sha256':audit.sha_file(initial/'hardening_quality.json'),'task_record_sha256':old['task_record_sha256'],'admission_audit_sha256':audit.sha_file(old_dir/'admission_audit.json'),'reference_replays_sha256':audit.sha_file(old_dir/'audit_replays.jsonl'),'original_status':'pass','verification_contract':contract,'verification_contract_sha256':audit.object_sha(contract)}
    admitted={**old_admitted,'audit_method':'reused_exact_hardened_verification_contract_24_reference_replays','replay_reuse_origin':origin}
    write(directory/'admission_audit.json',admitted)
    record={**old,'admission_audit_sha256':audit.sha_file(directory/'admission_audit.json')};record['task_record_sha256']=audit.object_sha({k:v for k,v in record.items() if k!='task_record_sha256'})
    write(directory/'task.json',record)
    reuse={'schema':'constructive-code-exact-reference-replay-reuse-v1','initial_manifest_sha256':origin['manifest_sha256'],'initial_quality_sha256':origin['hardening_quality_sha256'],'reused_problem_ids':['fixture'],'policy_model_outputs_used':False}
    write(root/'replay_reuse_manifest.json',reuse)
    review={'status':'pass','initial_manifest_sha256':origin['manifest_sha256'],'replay_reuse_manifest_sha256':audit.sha_file(root/'replay_reuse_manifest.json'),'tasks':[{'task_id':'fixture','old_task_record_sha256':old['task_record_sha256'],'new_task_record_sha256':record['task_record_sha256'],'admission_audit_sha256':audit.sha_file(directory/'admission_audit.json'),'reference_replays_sha256':admitted['replays_sha256'],'task_adapter':record['task_adapter'],'problem_key':record['problem_key'],'witness_family':record['witness_family'],'original_audit_counts_status_violations_preserved':True}]}
    write(root/'independent_replay_reuse_audit.json',review)
    quality={**old_quality,'replay_reuse_manifest_sha256':audit.sha_file(root/'replay_reuse_manifest.json'),'source_candidates':1,'post_replay_pool_sealing':{'independent_reuse_audit_sha256':audit.sha_file(root/'independent_replay_reuse_audit.json'),'reused_initial_source_panels':1,'fresh_source_panels':0}}
    return root,directory,record,admitted,quality


def test_exact_reference_replay_reuse_recomputes_contract_and_raw_origin(tmp_path):
    args=reuse_fixture(tmp_path)
    assert audit.audit_reused_hardening_task(*args)['status']=='pass'
    (Path(args[3]['replay_reuse_origin']['root'])/'reference_panel/audit_replays.jsonl').write_text('changed original verdicts\n')
    with pytest.raises(audit.AuditError,match='reference verdict bytes'):
        audit.audit_reused_hardening_task(*args)


@pytest.mark.parametrize('field,value',[('task_adapter','different-adapter'),('problem_key','different-problem'),('limits',{'time_milliseconds':99999})])
def test_replay_reuse_cannot_relabel_same_suite_as_different_task(tmp_path,field,value):
    root,directory,record,admitted,quality=reuse_fixture(tmp_path)
    record[field]=value;record['task_record_sha256']=audit.object_sha({k:v for k,v in record.items()if k!='task_record_sha256'})
    with pytest.raises(audit.AuditError,match='execution contract changed'):
        audit.audit_reused_hardening_task(root,directory,record,admitted,quality)


def test_replay_reuse_cannot_change_original_runtime_even_with_equal_counts(tmp_path):
    root,directory,record,admitted,quality=reuse_fixture(tmp_path)
    admitted['runtime']={'image_sha256':'different-runtime'}
    with pytest.raises(audit.AuditError,match='runtime or audit details'):
        audit.audit_reused_hardening_task(root,directory,record,admitted,quality)


def test_replay_reuse_requires_complete_independent_panel_bijection(tmp_path):
    root,directory,record,admitted,quality=reuse_fixture(tmp_path)
    review=audit.read_json(root/'independent_replay_reuse_audit.json');review['tasks']=[]
    write(root/'independent_replay_reuse_audit.json',review)
    quality['post_replay_pool_sealing']['independent_reuse_audit_sha256']=audit.sha_file(root/'independent_replay_reuse_audit.json')
    with pytest.raises(audit.AuditError,match='panel identity bijection'):
        audit.audit_reused_hardening_task(root,directory,record,admitted,quality)


def runtime_fixture(tmp_path,monkeypatch):
    directory,record,admitted=reference_panel(tmp_path)
    work=tmp_path/'work';work.mkdir();(work/'build/fixture').mkdir(parents=True);(work/'runtime/usr/bin').mkdir(parents=True);(work/'scratch').mkdir()
    (work/'launcher').write_bytes(b'pinned launcher');(work/'build/fixture/checker').write_bytes(b'pinned checker')
    (work/'runtime/usr/bin/python').write_bytes(b'pinned interpreter')
    image=tmp_path/'image.sqsh';image.write_bytes(b'pinned runtime image')
    pinned={'usr/bin/python':audit.sha_file(work/'runtime/usr/bin/python')}
    monkeypatch.setattr(audit,'PINNED_RUNTIME_FILES',pinned);monkeypatch.setattr(audit,'PINNED_RUNTIME_IMAGE_SHA256',audit.sha_file(image))
    admitted.update(status='pass',runtime={'launcher_sha256':audit.sha_file(work/'launcher'),'runtime':{'image_sha256':audit.sha_file(image),'critical_file_sha256':pinned}},checker_build={'binary_sha256':audit.sha_file(work/'build/fixture/checker'),'source_sha256':record['checker_sha256']})
    write(directory/'admission_audit.json',admitted)
    record['admission_audit_sha256']=audit.sha_file(directory/'admission_audit.json');record['task_record_sha256']=audit.object_sha(record);write(directory/'task.json',record)
    items=[{'source_problem_id':'fixture','relative_path':'reference_panel','task_record_sha256':record['task_record_sha256']}]
    write(tmp_path/'manifest.json',{'tasks':items,'tasks_sha256':audit.object_sha(items)})
    config={'adapter_module':'build_constructive_code_hardened_20260921','adapter_config':{'slate_root':str(tmp_path),'problem_ids':['fixture'],'image':str(image),'build_root':str(work/'build'),'runtime_root':str(work/'runtime'),'scratch_root':str(work/'scratch'),'launcher':str(work/'launcher')},'task_ids':['fixture'],'seed':7,'max_tokens':1024,'learning_rate':1e-5}
    return config,work


def test_runtime_location_normalization_requires_equal_executable_bytes(tmp_path,monkeypatch):
    import shutil
    config,work=runtime_fixture(tmp_path,monkeypatch);copy_root=tmp_path/'other-role';shutil.copytree(work,copy_root)
    other=json.loads(json.dumps(config))
    for field in audit.CODE_LOCATION_FIELDS:other['adapter_config'][field]=str(copy_root/Path(config['adapter_config'][field]).relative_to(work))
    assert audit.comparison_config(config,tmp_path/'job1')==audit.comparison_config(other,tmp_path/'job2')
    (copy_root/'build/fixture/checker').write_bytes(b'changed checker')
    with pytest.raises(audit.AuditError,match='actual checker differs'):
        audit.comparison_config(other,tmp_path/'job2')


@pytest.mark.parametrize('field,value',[('seed',99),('max_tokens',128),('learning_rate',.01)])
def test_runtime_location_normalization_preserves_semantic_config(tmp_path,monkeypatch,field,value):
    config,_=runtime_fixture(tmp_path,monkeypatch);other=json.loads(json.dumps(config));other[field]=value
    assert audit.comparison_config(config,tmp_path/'job1')!=audit.comparison_config(other,tmp_path/'job2')


@pytest.mark.parametrize('file,match',[('launcher','actual launcher'),('runtime/usr/bin/python','runtime executable')])
def test_runtime_location_normalization_rejects_launcher_or_interpreter_drift(tmp_path,monkeypatch,file,match):
    config,work=runtime_fixture(tmp_path,monkeypatch);(work/file).write_bytes(b'changed executable')
    with pytest.raises(audit.AuditError,match=match):audit.comparison_config(config,tmp_path/'job')


def test_coding_development_groups_never_become_reserved_holdout():
    assert audit.endpoint_stratum('code','development',True)=='trained_development'
    assert audit.endpoint_stratum('code','development',False)=='untrained_development'
    assert audit.endpoint_stratum('qa','train',True)=='trained_train'
    assert audit.endpoint_stratum('qa','dev',False)=='dev'
    for split in ('test','heldout','dev','unknown'):
        with pytest.raises(audit.AuditError):audit.endpoint_stratum('code',split,False)
    assert audit.endpoint_stratum('code','train',True)=='trained_train'
    assert audit.endpoint_stratum('code','validation',False)=='validation'
    with pytest.raises(audit.AuditError,match='split contradiction'):audit.endpoint_stratum('qa','dev',True)


def test_frozen_testlib_relocation_requires_unique_original_to_snapshot_binding(tmp_path,monkeypatch):
    root,bundle,quality=source_closure(tmp_path,monkeypatch)
    header=bundle/'third_party/testlib/testlib.h';relocated=bundle/'testlib/testlib.h';relocated.parent.mkdir();header.rename(relocated)
    row={'source':'/original/repo/third_party/testlib/testlib.h','snapshot':str(relocated),'sha256':audit.sha_file(relocated)}
    write(bundle.parent/'identity.json',{'files':[row]})
    assert audit.audit_hardening_sources(root,quality)['status']=='pass'
    write(bundle.parent/'identity.json',{'files':[row,row]})
    with pytest.raises(audit.AuditError,match='not uniquely byte-bound'):audit.audit_hardening_sources(root,quality)


def test_active_fresh_advantages_and_resource_diagnostics_are_recomputed():
    row={'fresh_successes':3,'adv_min':-1.0,'adv_max':16/3-1,'adv_mean':0.0,'actual_gradient_norm':.25,'generation_seconds':2.0,'verification_seconds':1.0,'learning_seconds':1.0,'total_update_seconds':4.5}
    result={'peak_gpu_allocated_bytes':100,'peak_gpu_reserved_bytes':200,'allocated_gpu_hours_during_runner':5/3600}
    diag=audit.audit_training_diagnostics([row],result,16)
    assert diag['groups_with_nonzero_recomputed_fresh_advantages']==diag['mixed_groups_with_positive_actual_gradient']==1
    changed={**row,'adv_max':0.0}
    with pytest.raises(audit.AuditError,match='advantage diagnostic mismatch'):audit.audit_training_diagnostics([changed],result,16)
    with pytest.raises(audit.AuditError,match='duration shorter'):audit.audit_training_diagnostics([row],{**result,'allocated_gpu_hours_during_runner':.0001},16)


def test_pilot_readiness_cannot_pass_without_both_external_endpoint_comparisons():
    caps={d:[{'status':'pass','domain':d,'summary':{'tasks_with_multiple_modes':1},'token_text_binding':'pass','frozen_implementation_binding':'pass','job_id':1}] for d in ('code','qa')}
    pairs={d:{'status':'pass','domain':d,'token_text_binding':'pass','frozen_implementation_binding':'pass','arms':[{'job_id':2},{'job_id':3}]} for d in ('code','qa')}
    summary=audit.report(caps,pairs)
    assert summary['readiness_gate']=='unknown'
    assert all(d['readiness']=='unknown' for d in summary['domains'].values())


def cohort_fixture(tmp_path,domain):
    source={'train':'train','train2':'train','dev':'dev' if domain=='qa' else 'validation','test1':'test','test2':'test'}
    if domain=='qa':
        p=tmp_path/'cohort.jsonl';rows(p,[{'id':t,'split':split}for t,split in source.items()])
        config={'adapter_module':'oat_drgrpo.noncoding_multi_answer_sata','adapter_config':{'records_path':str(p),'records_sha256':audit.sha_file(p),'allow_test':True,'splits':['test']}}
    else:
        items=[{'source_problem_id':t,'split':split,'status':'admitted'}for t,split in source.items()]
        write(tmp_path/'manifest.json',{'tasks':items,'tasks_sha256':audit.object_sha(items),'admitted_problem_ids':list(source),'admitted_test_problem_ids':['test1','test2']})
        config={'adapter_module':'build_constructive_code_hardened_20260921','adapter_config':{'slate_root':str(tmp_path),'allow_heldout':True}}
    return config,{t:{'split':split}for t,split in source.items()}


@pytest.mark.parametrize('domain',['qa','code'])
def test_reserved_primary_requires_complete_frozen_support_and_permission(tmp_path,domain):
    config,metadata=cohort_fixture(tmp_path,domain);selected={t:metadata[t]for t in ('test1','test2')}
    assert audit.endpoint_cohort(config,selected,{'train'},domain)=='reserved_test_primary'
    assert audit.endpoint_stratum(domain,'test',False,allow_reserved=True)=='reserved_test'
    with pytest.raises(audit.AuditError,match='full frozen test set'):
        audit.endpoint_cohort(config,{'test1':metadata['test1']},{'train'},domain)
    with pytest.raises(audit.AuditError,match='source task used for training'):
        audit.endpoint_cohort(config,selected,{'train','test1'},domain)
    with pytest.raises(audit.AuditError,match='full frozen test set'):
        audit.endpoint_cohort(config,{**selected,'train':metadata['train']},{'train'},domain)
    config['adapter_config']['allow_test' if domain=='qa' else 'allow_heldout']=False
    with pytest.raises(audit.AuditError,match='requires allow_'):
        audit.endpoint_cohort(config,selected,{'train'},domain)


@pytest.mark.parametrize('domain',['qa','code'])
def test_trained_secondary_requires_all_training_ids_and_source_labels(tmp_path,domain):
    config,metadata=cohort_fixture(tmp_path,domain)
    assert audit.endpoint_cohort(config,{'train':metadata['train']},{'train'},domain)=='trained_support_diagnostic'
    with pytest.raises(audit.AuditError,match='omits trained'):
        audit.endpoint_cohort(config,{'train':metadata['train']},{'train','train2'},domain)
    with pytest.raises(audit.AuditError,match='label differs'):
        audit.endpoint_cohort(config,{'train':{'split':'unknown'}},{'train'},domain)
    with pytest.raises(audit.AuditError,match='source task used for training'):
        audit.endpoint_cohort(config,{'dev':metadata['dev']},{'dev'},domain)


def test_reserved_primary_accepts_code_heldout_label_but_rejects_unsupported_label(tmp_path):
    config,metadata=cohort_fixture(tmp_path,'code');manifest=audit.read_json(tmp_path/'manifest.json')
    for row in manifest['tasks']:
        if row['split']=='test':row['split']='heldout';metadata[row['source_problem_id']]['split']='heldout'
    manifest['tasks_sha256']=audit.object_sha(manifest['tasks']);write(tmp_path/'manifest.json',manifest)
    assert audit.endpoint_cohort(config,{t:metadata[t]for t in ('test1','test2')},{'train'},'code')=='reserved_test_primary'
    assert audit.endpoint_stratum('code','heldout',False,allow_reserved=True)=='reserved_test'
    with pytest.raises(audit.AuditError,match='unknown stratum'):audit.endpoint_stratum('code','unknown',False,allow_reserved=True)
    with pytest.raises(audit.AuditError,match='overlaps training'):audit.endpoint_stratum('code','test',True,allow_reserved=True)


@pytest.mark.parametrize('cohort',['reserved','trained'])
def test_complete_three_endpoint_audit_supports_separate_primary_and_secondary_cohorts(tmp_path,cohort):
    data=source(tmp_path);records=audit.read_rows(data['records_path'])
    for record in records:record['split']=record['id']
    for name in ('test1','test2'):
        record=deepcopy(records[0]);record.update(id=name,split='test');records.append(record)
    rows(Path(data['records_path']),records);data['records_sha256']=audit.sha_file(data['records_path'])
    a=training(tmp_path,'maxrl',data);b=training(tmp_path,'remax',data);pair=audit.audit_pair(a,b,(Codec(),Codec()))
    paths=[endpoint(tmp_path,'base',data),endpoint(tmp_path,'maxrl',data,a),endpoint(tmp_path,'remax',data,b)]
    for path in paths:
        receipt=audit.read_json(path)
        if cohort=='reserved':
            mapping={'dev':'test1','train':'test2'};receipt['config']['adapter_config']={**data,'allow_test':True,'splits':['test']}
        else:mapping={'train':'train'}
        receipt['config']['task_ids']=list(mapping.values())
        receipt['task_prompts']=[{**p,'task_id':mapping[p['task_id']],'split':'test' if cohort=='reserved' else 'train'}for p in receipt['task_prompts'] if p['task_id']in mapping]
        receipt['task_results']=[{**r,'task_id':mapping[r['task_id']]}for r in receipt['task_results']if r['task_id']in mapping]
        for artifact in ('responses','attempts'):
            p=Path(receipt['artifacts'][artifact]['path']);rr=[]
            for row in audit.read_rows(p):
                if row['task_id']not in mapping:continue
                row['task_id']=mapping[row['task_id']]
                if cohort=='trained':row['request_seed']-=10000
                rr.append(row)
            rows(p,rr);receipt['artifacts'][artifact]['sha256']=audit.sha_file(p)
        write(path,receipt)
    result=audit.audit_endpoints(paths,pair,[Codec()]*3)
    assert result['status']=='pass'
    assert result['endpoint_cohort']==('reserved_test_primary' if cohort=='reserved' else 'trained_support_diagnostic')
    assert set(result['strata'])==({'reserved_test'}if cohort=='reserved' else {'trained_train'})
    if cohort=='reserved':
        assert result['trained_prompt_table']=={} and result['common_multi_bank_task_ids']==[]


def test_nested_freeze_normalization_requires_explicit_context_and_preserves_semantics(tmp_path):
    base=tmp_path/'base';endpoint=tmp_path/'endpoint';base.mkdir();endpoint.mkdir()
    write(base/'identity.json',{'schema':'real-domains-frozen-job-20260921-v1','request':{'freeze_data_roots':['/original/data']}})
    write(endpoint/'identity.json',{'schema':'real-domains-frozen-job-20260921-v1','request':{'freeze_data_roots':[str(base/'bundle/data/0')]}})
    config={'data':str(endpoint/'bundle/data/0/manifest.json'),'temperature':1,'seed':7}
    assert audit.normalized_config(config,endpoint)['data']==str(base/'bundle/data/0/manifest.json')
    expected={'data':'/original/data/manifest.json','temperature':1,'seed':7}
    assert audit.normalized_config(config,endpoint,chain_directories=[base])==expected
    assert audit.normalized_config({**config,'seed':9},endpoint,chain_directories=[base])!=expected
    write(base/'identity.json',{'schema':'real-domains-frozen-job-20260921-v1','request':{'freeze_data_roots':[str(endpoint/'bundle/data/0')]}})
    with pytest.raises(audit.AuditError,match='cyclic frozen-data'):
        audit.normalized_config(config,endpoint,chain_directories=[base])


def test_nested_code_comparison_checks_runtime_at_actual_receipt_locations(tmp_path,monkeypatch):
    config,_=runtime_fixture(tmp_path,monkeypatch);base=tmp_path/'base';endpoint=tmp_path/'endpoint';base.mkdir();endpoint.mkdir()
    write(base/'identity.json',{'schema':'real-domains-frozen-job-20260921-v1','request':{'freeze_data_roots':['/original/data']}})
    write(endpoint/'identity.json',{'schema':'real-domains-frozen-job-20260921-v1','request':{'freeze_data_roots':[str(base/'bundle/data/0')]}})
    config['data']=str(endpoint/'bundle/data/0/manifest.json')
    observed=[]
    real_check=audit.audit_code_runtime_locations
    def inspect(raw):
        observed.append(deepcopy(raw));return real_check(raw)
    monkeypatch.setattr(audit,'audit_code_runtime_locations',inspect)
    result=audit.comparison_config(config,endpoint,chain_directories=[base])
    assert result['data']=='/original/data/manifest.json'
    assert observed==[config]


def test_initial_manifest_without_split_uses_hash_bound_task_record(tmp_path):
    directory=tmp_path/'example';directory.mkdir()
    record={'source_problem_id':'example','split':'development'};record['task_record_sha256']=audit.object_sha(record);write(directory/'task.json',record)
    items=[{'source_problem_id':'example','status':'admitted','relative_path':'example','task_record_sha256':record['task_record_sha256']}]
    write(tmp_path/'manifest.json',{'tasks':items,'tasks_sha256':audit.object_sha(items),'admitted_problem_ids':['example']})
    config={'adapter_config':{'slate_root':str(tmp_path)}}
    assert audit.endpoint_cohort(config,{'example':{'split':'development'}},{'example'},'code')=='trained_support_diagnostic'
    record['split']='test';write(directory/'task.json',record)
    with pytest.raises(audit.AuditError,match='task record hash mismatch'):
        audit.endpoint_cohort(config,{'example':{'split':'test'}},{'example'},'code')
