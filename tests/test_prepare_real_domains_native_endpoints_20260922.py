import copy
import json
from pathlib import Path

import pytest

import prepare_real_domains_native_endpoints_20260922 as prep


PLAN = Path(__file__).resolve().parents[1] / "var/artifacts/real_domains_pilot_20260921/code_corrected128_plan_unfrozen_unsubmitted/plan.json"


@pytest.fixture
def plan():
    return json.loads(PLAN.read_text())


def test_registered_schedule_has_complete_global_denominators(plan):
    shards = prep.validate_plan(plan)
    assert len(shards) == 13
    assert sum(s["total_completions"] for s in shards) == 23040
    assert sum(s["gpu_hour_cap"] for s in shards) == 37
    assert [s["checkpoint_step"] for s in shards].count(128) == 6
    assert any("1408_A" in s["task_ids"] for s in shards)


@pytest.mark.parametrize("mutation", ["omit_hard_question", "seed", "counts", "monitor_selection", "budget", "unsafe_path", "duplicate_policy"])
def test_schedule_rejects_silent_design_changes(plan, mutation):
    evaluation = plan["native_hf_evaluation"]
    if mutation == "omit_hard_question":
        evaluation["schedule"][0]["job_shards"][0]["task_ids"].remove("1408_A")
    elif mutation == "seed":
        evaluation["seed"] += 1
    elif mutation == "counts":
        evaluation["schedule"][0]["job_shards"][0]["samples_per_task"] = 508
    elif mutation == "monitor_selection":
        evaluation["schedule"][1]["step"] = 96
    elif mutation == "budget":
        plan["budget"]["total_new_hard_envelope_gpu_hours"] = 48
    elif mutation == "unsafe_path":
        evaluation["schedule"][0]["job_shards"][0]["name"] = "../endpoint"
    else:
        evaluation["schedule"].append(copy.deepcopy(evaluation["schedule"][0]))
    with pytest.raises(ValueError):
        prep.validate_plan(plan)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoint-32"
    adapter = path / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"sealed learned weights")
    save(adapter / "adapter_config.json", {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]})
    save(path / "bank.json", {"entries": {}})
    (path / "training.pt").write_bytes(b"sealed optimizer and RNG")
    cfg = {"lora_rank": 16, "lora_alpha": 32, "lora_target_modules": ["q_proj"]}
    identity = {"config": cfg, "config_sha256": prep.canonical(cfg)}
    seal = {"schema": prep.TRAINER_SCHEMA, "arm": "maxrl", "completed_updates": 32, "config_sha256": identity["config_sha256"], "bank_sha256": prep.digest(path / "bank.json"), "training_state_sha256": prep.digest(path / "training.pt"), "adapter_files": {p.name: prep.digest(p) for p in adapter.iterdir()}}
    save(path / "complete.json", seal)
    return path, identity


def test_intermediate_seal_does_not_require_terminal_checkpoint_number(checkpoint):
    path, identity = checkpoint
    assert prep.verify_checkpoint(path, "maxrl", 32, identity)["step"] == 32


@pytest.mark.parametrize("mutation", ["wrong_arm", "wrong_step", "wrong_config", "optimizer", "extra_adapter", "architecture", "old_schema"])
def test_intermediate_seal_rejects_mixing_or_mutation(checkpoint, mutation):
    path, identity = checkpoint
    seal = json.loads((path / "complete.json").read_text())
    if mutation == "wrong_arm":
        seal["arm"] = "remax"
    elif mutation == "wrong_step":
        seal["completed_updates"] = 64
    elif mutation == "wrong_config":
        seal["config_sha256"] = "0" * 64
    elif mutation == "optimizer":
        (path / "training.pt").write_bytes(b"changed optimizer")
    elif mutation == "extra_adapter":
        (path / "adapter" / "extra.bin").write_bytes(b"unsealed file")
    elif mutation == "architecture":
        identity["config"]["lora_rank"] = 32
    else:
        seal["schema"] = "real-domain-online-maxrl-remax-pilot-20260921-v1"
    save(path / "complete.json", seal)
    with pytest.raises(ValueError):
        prep.verify_checkpoint(path, "maxrl", 32, identity)


def test_frozen_job_preserves_checkpoint_name_and_never_submits(tmp_path, monkeypatch, checkpoint):
    repo = tmp_path / "repo"
    for relative in ("ops/" + prep.ENTRYPOINT, "ops/repo_env.sh", "src/example.py", "third_party/testlib/testlib.h"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable source bytes\n")
    monkeypatch.setattr(prep, "ROOT", repo)
    source = tmp_path / "frozen_data"
    source.mkdir()
    original_checkpoint, identity = checkpoint
    import shutil
    shutil.copytree(original_checkpoint, source / original_checkpoint.name)
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"model weights")
    save(source / "model_files.json", {"files": [{"path": str(model / "weights.safetensors"), "sha256": prep.digest(model / "weights.safetensors")}]})
    request = {"entrypoint": prep.ENTRYPOINT, "time_limit_minutes": 150, "partition": "lowprio", "job_name": "native-test", "freeze_data_roots": [str(source)], "config": {"checkpoint_path": str(source / "checkpoint-32"), "model_manifest_path": str(source / "model_files.json"), "model_manifest_sha256": prep.digest(source / "model_files.json")}}
    request_path = tmp_path / "request.json"
    save(request_path, request)
    output = tmp_path / "output"
    result = prep.freeze_job(request_path, output)
    cfg = json.loads((output / "config.json").read_text())
    assert Path(cfg["checkpoint_path"]).name == "checkpoint-32"
    prep.verify_checkpoint(cfg["checkpoint_path"], "maxrl", 32, identity)
    assert result["submitted"] is False
    assert not (output / "submission.json").exists()
    assert json.loads((output / "identity.json").read_text())["job_id"] is None
    assert "--partition=lowprio" in json.loads((output / "submission_intent.json").read_text())["argv"]
    assert "native evaluator verifies full model bytes before loading" in (output / "run.slurm").read_text()
    (source / "checkpoint-32" / "training.pt").write_bytes(b"mutated original")
    prep.verify_checkpoint(cfg["checkpoint_path"], "maxrl", 32, identity)
    with pytest.raises(FileExistsError):
        prep.freeze_job(request_path, output)


def test_overlong_validation_allocation_is_rejected(tmp_path):
    request = {"entrypoint": prep.ENTRYPOINT, "time_limit_minutes": 180, "partition": "lowprio", "config": {"validation_only": True}}
    save(tmp_path / "request.json", request)
    with pytest.raises(ValueError, match="bounded allocation"):
        prep.freeze_job(tmp_path / "request.json", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_production_all_partition_is_rejected_before_output(tmp_path):
    request = {'entrypoint': prep.ENTRYPOINT, 'time_limit_minutes': 180, 'partition': 'all', 'config': {}}
    save(tmp_path/'request.json', request)
    with pytest.raises(ValueError, match='bounded allocation'):
        prep.freeze_job(tmp_path/'request.json', tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('partition', ['all', 'lowprio'])
def test_tiny_validation_partition_is_preserved_in_submission_intent(tmp_path, monkeypatch, partition):
    repo=tmp_path/'repo'
    for name in ('ops/'+prep.ENTRYPOINT, 'ops/repo_env.sh', 'third_party/testlib/testlib.h'):
        file=repo/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('source')
    monkeypatch.setattr(prep,'ROOT',repo)
    data=tmp_path/'data';data.mkdir();save(data/'model_files.json', {'files':[]})
    request={'entrypoint':prep.ENTRYPOINT,'time_limit_minutes':15,'partition':partition,'job_name':'validation','freeze_data_roots':[str(data)],'config':{'validation_only':True,'model_manifest_path':str(data/'model_files.json'),'model_manifest_sha256':prep.digest(data/'model_files.json')}}
    save(tmp_path/'request.json',request)
    prep.freeze_job(tmp_path/'request.json',tmp_path/'output')
    intent=json.loads((tmp_path/'output/submission_intent.json').read_text())
    assert '--partition='+partition in intent['argv']
    assert intent['authorized_gpu_hour_ceiling']==.25


@pytest.fixture
def paired_training(tmp_path,monkeypatch,plan):
    repo=tmp_path/'repo';repo.mkdir()
    monkeypatch.setattr(prep,'ROOT',repo)
    pins={}
    for name in prep.PRODUCTION_MODULES:
        relative='src/'+name.replace('.','/')+'.py'
        path=repo/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
        pins[name]=prep.digest(path)
    source_contract=tmp_path/'source_contract.json';save(source_contract,{})
    roots={};frozen={}
    for arm in ('maxrl','remax'):
        root=tmp_path/arm;root.mkdir();roots[arm]=str(root)
        slate=root/'data';slate.mkdir()
        save(slate/'manifest.json',{'fixed_tasks':plan['dataset']['train_ids']+plan['dataset']['untrained_development_ids']})
        save(slate/'hardening_quality.json',{'status':'pass'})
        cfg={'train_ids':plan['dataset']['train_ids'],'eval_ids':plan['dataset']['untrained_development_ids'],'updates':128,'seed':88411,'train_microbatch_size':1,'model_revision':plan['model_revision'],'model':plan['model'],'adapter_config':{'slate_root':str(slate),'build_root':str(root/'work')}}
        save(root/'config.json',cfg)
        identity={'schema':prep.TRAINER_SCHEMA,'arm':arm,'runner_sha256':prep.TRAINER_SHA256,'config':cfg,'config_sha256':prep.canonical(cfg),'input_config_sha256':prep.digest(root/'config.json'),'initial_trainable_parameters_sha256':'a'*64,'production_source_sha256':pins}
        save(root/'training/identity.json',identity)
        save(root/'training/result.json',{'status':'complete','arm':arm,'completed_updates':128,'config_sha256':identity['config_sha256']})
        frozen[arm]={'config':cfg,'root':root}
    plan['dataset']['manifest_sha256']=prep.digest(slate/'manifest.json')
    plan['dataset']['quality_sha256']=prep.digest(slate/'hardening_quality.json')
    monkeypatch.setattr(prep,'frozen_training_run',lambda root,schedule,arm,checked_external=None:frozen[arm])
    monkeypatch.setattr(prep.original_endpoints,'source',lambda run,rel:prep.digest(repo/rel))
    schedule={'training_runs':roots,'source_contract_path':str(source_contract)}
    return schedule,plan,roots


def test_baseline_accepts_initialized_pair_before_results_but_trained_endpoint_waits(paired_training):
    schedule,plan,roots=paired_training
    for root in roots.values(): (Path(root)/'training/result.json').unlink()
    assert set(prep.training_pair(schedule,plan,False))=={'maxrl','remax'}
    with pytest.raises(ValueError,match='both complete terminal128'):
        prep.training_pair(schedule,plan,True)


@pytest.mark.parametrize('mutation',['initial_hash','unfinished','wrong_terminal','uninitialized','config_sha','missing_source'])
def test_training_pair_rejects_mixing_or_incomplete_training(paired_training,mutation):
    schedule,plan,roots=paired_training
    path=Path(roots['remax'])/'training/identity.json';identity=json.loads(path.read_text())
    if mutation=='initial_hash': identity['initial_trainable_parameters_sha256']='b'*64
    elif mutation=='uninitialized': identity.pop('initial_trainable_parameters_sha256')
    elif mutation=='config_sha': identity['config_sha256']='c'*64
    elif mutation=='missing_source': identity['production_source_sha256'].pop('oat_drgrpo.args')
    else:
        result_path=path.parent/'result.json';result=json.loads(result_path.read_text())
        if mutation=='unfinished':result['status']='failed'
        else:result['completed_updates']=64
        save(result_path,result)
    save(path,identity)
    with pytest.raises(ValueError):prep.training_pair(schedule,plan,True)


@pytest.fixture
def externally_bound_training(tmp_path):
    run = tmp_path / 'run'
    bundle = run / 'bundle'
    source = bundle / 'ops' / prep.TRAINER
    source.parent.mkdir(parents=True)
    source.write_bytes(b'immutable corrected trainer')
    model = tmp_path / ('a' * 40)
    model.mkdir()
    weight = model / 'weights.safetensors'
    weight.write_bytes(b'full model weight bytes')
    manifest = tmp_path / 'model_files.json'
    save(manifest, {'files': [{'path': str(weight), 'sha256': prep.digest(weight)}]})
    request = {'entrypoint': prep.TRAINER, 'arm': 'maxrl', 'config': {'model': str(model)}, 'freeze_data_roots': []}
    save(run / 'config.json', request['config'])
    identity = {'schema': 'real-domains-frozen-job-20260921-v1', 'request': request, 'config_sha256': prep.digest(run / 'config.json'), 'files': [
        {'source': str(source), 'snapshot': str(source), 'sha256': prep.digest(source)},
        {'source': str(weight), 'snapshot': str(weight), 'sha256': prep.digest(weight), 'binding_kind': 'external_content_addressed_model_file'},
        {'source': str(manifest), 'snapshot': str(manifest), 'sha256': prep.digest(manifest), 'binding_kind': 'prospective_shared_model_manifest'},
    ]}
    save(run / 'identity.json', identity)
    schedule = {'model_manifest_path': str(manifest), 'model_manifest_sha256': prep.digest(manifest)}
    return run, schedule, identity, weight


def test_exact_external_model_manifest_is_accepted(externally_bound_training):
    run, schedule, _, weight = externally_bound_training
    result = prep.frozen_training_run(run, schedule, 'maxrl')
    assert result['files'][weight] == prep.digest(weight)


@pytest.mark.parametrize('mutation', ['unknown_external', 'missing_weight', 'weight_changed', 'manifest_changed', 'source_mismatch'])
def test_external_model_exception_stays_narrow(externally_bound_training, mutation):
    run, schedule, identity, weight = externally_bound_training
    if mutation == 'unknown_external':
        external = run.parent / 'unbound.txt'
        external.write_bytes(b'unbound dependency')
        identity['files'].append({'source': str(external), 'snapshot': str(external), 'sha256': prep.digest(external)})
    elif mutation == 'missing_weight':
        identity['files'] = [row for row in identity['files'] if row.get('binding_kind') != 'external_content_addressed_model_file']
    elif mutation == 'weight_changed':
        weight.write_bytes(b'mutated weights')
    elif mutation == 'manifest_changed':
        identity['files'][-1]['sha256'] = '0' * 64
    else:
        identity['files'][-2]['source'] = str(run.parent / 'different-source')
    save(run / 'identity.json', identity)
    with pytest.raises(ValueError):
        prep.frozen_training_run(run, schedule, 'maxrl')
