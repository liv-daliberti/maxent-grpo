import copy
import json
from pathlib import Path
import shutil
import pytest
import prepare_real_domains_endpoints_20260921 as ep


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);ep.write(path,value)


@pytest.fixture
def setup(tmp_path,monkeypatch,request):
    repo=tmp_path/'repo';(repo/'ops').mkdir(parents=True);(repo/'src/pkg').mkdir(parents=True)
    for name in ('evaluate_real_domains_20260921.py','train_real_domains_pilot_20260921.py','adapter.py'):
        (repo/'ops'/name).write_text(name+' immutable bytes\n')
    (repo/'src/pkg/core.py').write_text('production learner bytes\n')
    monkeypatch.setattr(ep,'ROOT',repo)
    model=tmp_path/('a'*40);model.mkdir();save(model/'config.json',{'vocab_size':5})
    data=tmp_path/'data';data.mkdir();save(data/'manifest.json',{'tasks':['one','two','heldout']})
    def freeze(request_path,output,submit=False):
        assert submit is False
        request=json.loads(request_path.read_text());output.mkdir(parents=True);bundle=output/'bundle';files=[]
        def cp(source,dest):
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);files.append({'source':str(source),'snapshot':str(dest),'sha256':ep.digest(dest)})
        for p in repo.rglob('*.py'):cp(p,bundle/p.relative_to(repo))
        replacements=[]
        for i,dirname in enumerate(request['freeze_data_roots']):
            directory=Path(dirname);target=bundle/'data'/str(i);replacements.append((str(directory.resolve()),str(target)))
            for p in directory.rglob('*'):
                if p.is_file():cp(p,target/p.relative_to(directory))
        save(output/'config.json',ep.launcher.rewrite_paths(request['config'],replacements))
        save(output/'identity.json',{'schema':'real-domains-frozen-job-20260921-v1','request':request,'request_sha256':ep.digest(request_path),'config_sha256':ep.digest(output/'config.json'),'files':files})
        return {'output':str(output),'job_id':None}
    basecfg={'model':str(model),'model_revision':model.name,'adapter_module':'adapter','adapter_config':{'slate_root':str(data),'problem_ids':['one','two','heldout'],'build_root':str(tmp_path/'basework/build'),'runtime_root':str(tmp_path/'basework/runtime'),'launcher':str(tmp_path/'basework/launcher'),'scratch_root':str(tmp_path/'basework/scratch')},'task_ids':['one','two','heldout'],'samples_per_task':128,'seed':888,'max_tokens':1024,'generation_batch_size':64,'max_model_len':8192}
    if getattr(request,'param',None)=='qa':
        basecfg['adapter_config']={'manifest_path':str(data/'manifest.json'),'records_path':str(data/'manifest.json'),'splits':['test'],'allow_test':True}
    base_request=tmp_path/'baseline_request.json';save(base_request,{'entrypoint':'evaluate_real_domains_20260921.py','config':basecfg,'freeze_data_roots':[str(data)],'time_limit_minutes':60})
    base_run=tmp_path/'base';freeze(base_request,base_run)
    runs={}
    for arm in ('maxrl','remax'):
        cfg={k:copy.deepcopy(basecfg[k]) for k in ('model','model_revision','adapter_module','adapter_config')}
        cfg.update({'updates':32,'seed':123,'train_ids':['one'],'eval_ids':['two'],'lora_rank':16,'lora_alpha':32,'lora_target_modules':['q_proj']})
        if getattr(request,'param',None)=='qa':
            cfg['adapter_config'].pop('allow_test');cfg['adapter_config']['splits']=['train','dev']
        else:
            cfg['adapter_config']['problem_ids']=['one','two']
        for key in ep.MUTABLE:
            if key in cfg['adapter_config']:cfg['adapter_config'][key]=str(tmp_path/(arm+'_work')/key)
        request_path=tmp_path/(arm+'_train_request.json');save(request_path,{'entrypoint':'train_real_domains_pilot_20260921.py','arm':arm,'config':cfg,'freeze_data_roots':[str(data)],'time_limit_minutes':60})
        run=tmp_path/arm;freeze(request_path,run);runs[arm]=run
        config=json.loads((run/'config.json').read_text());training=run/'training';checkpoint=training/'checkpoint-32';adapter=checkpoint/'adapter';adapter.mkdir(parents=True)
        save(adapter/'adapter_config.json',{'base_model_name_or_path':str(model),'r':16,'lora_alpha':32,'target_modules':['q_proj']});(adapter/'adapter_model.safetensors').write_bytes(b'learned weights')
        save(checkpoint/'bank.json',{'modes':[]});(checkpoint/'training.pt').write_bytes(b'optimizer state')
        identity={'arm':arm,'input_config_sha256':ep.digest(run/'config.json'),'config':config,'config_sha256':ep.canonical(config),'runner_sha256':ep.digest(run/'bundle/ops/train_real_domains_pilot_20260921.py'),'adapter_module':str(run/'bundle/ops/adapter.py'),'adapter_module_sha256':ep.digest(run/'bundle/ops/adapter.py'),'production_source_sha256':{'pkg.core':ep.digest(run/'bundle/src/pkg/core.py')},'model_config_sha256':ep.digest(model/'config.json')}
        save(training/'identity.json',identity);save(training/'result.json',{'status':'complete','arm':arm,'completed_updates':32,'config_sha256':ep.canonical(config)})
        seal={'schema':'real-domain-online-maxrl-remax-pilot-20260921-v1','arm':arm,'completed_updates':32,'config_sha256':ep.canonical(config),'training_state_sha256':ep.digest(checkpoint/'training.pt'),'bank_sha256':ep.digest(checkpoint/'bank.json'),'adapter_files':{p.name:ep.digest(p) for p in adapter.iterdir()}}
        save(checkpoint/'complete.json',seal)
    def prepare():return ep.prepare_pair(base_request,base_run,runs['maxrl'],runs['remax'],32,tmp_path/'requests',prepare_job=freeze,run_parent=tmp_path,run_prefix='endpoint')
    return locals()


def test_terminal_pair_keeps_sampling_and_freezes_entire_checkpoint(setup):
    result=setup['prepare']();root=setup['tmp_path'];baseline=setup['basecfg']
    assert result['status']=='prepared_not_submitted'
    for arm in ('maxrl','remax'):
        request=json.loads((root/'requests'/(arm+'_request.json')).read_text());config=json.loads((root/('endpoint_'+arm)/'config.json').read_text())
        for field in ('task_ids','samples_per_task','seed','max_tokens','generation_batch_size','max_model_len'):assert config[field]==baseline[field]
        checkpoint=Path(config['lora_path']).parent
        assert all((checkpoint/name).exists() for name in ('complete.json','bank.json','training.pt','adapter/adapter_model.safetensors'))
        assert config['adapter_config']['runtime_root']==str(root/('endpoint_'+arm+'_work')/'runtime_root')
        assert result['arms'][arm]['run']==str(root/('endpoint_'+arm))
    assert result['arms']['maxrl']['prepared']['job_id'] is None


def test_incomplete_training_refused_before_outputs(setup):
    p=setup['runs']['remax']/'training/result.json';r=json.loads(p.read_text());r['status']='failed';save(p,r)
    with pytest.raises(ValueError,match='unfinished'):setup['prepare']()
    assert not (setup['tmp_path']/'requests').exists()


def test_checkpoint_bytes_must_match_completion_seal(setup):
    (setup['runs']['maxrl']/'training/checkpoint-32/training.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='seal mismatch'):setup['prepare']()


def test_missing_completion_seal_refused(setup):
    (setup['runs']['maxrl']/'training/checkpoint-32/complete.json').unlink()
    with pytest.raises(FileNotFoundError):setup['prepare']()


def test_wrong_sealed_arm_refused(setup):
    p=setup['runs']['maxrl']/'training/checkpoint-32/complete.json';r=json.loads(p.read_text());r['arm']='remax';save(p,r)
    with pytest.raises(ValueError,match='selected arm'):setup['prepare']()


def test_evaluator_change_cannot_change_paired_sampling(setup):
    (setup['repo']/'ops/evaluate_real_domains_20260921.py').write_text('different sampling')
    with pytest.raises(ValueError,match='source drift'):setup['prepare']()


def test_frozen_dataset_mutation_refused(setup):
    save(setup['runs']['remax']/'bundle/data/0/manifest.json',{'tasks':['changed']})
    with pytest.raises(ValueError,match='dependency checksum'):setup['prepare']()


def test_training_configuration_identity_required(setup):
    p=setup['runs']['remax']/'training/identity.json';r=json.loads(p.read_text());r['config']['seed']=999;save(p,r)
    with pytest.raises(ValueError,match='identity/configuration'):setup['prepare']()


def test_unsealed_extra_adapter_file_refused(setup):
    save(setup['runs']['maxrl']/'training/checkpoint-32/adapter/extra.json',{})
    with pytest.raises(ValueError,match='adapter seal mismatch'):setup['prepare']()


def test_existing_flat_run_destination_refused(setup):
    (setup['tmp_path']/'endpoint_maxrl').mkdir()
    with pytest.raises(FileExistsError,match='distinct new siblings'):setup['prepare']()


@pytest.mark.parametrize('setup',['qa'],indirect=True)
def test_qa_train_dev_to_authorized_heldout_keeps_same_data(setup):
    result=setup['prepare']()
    config=json.loads((Path(result['arms']['remax']['run'])/'config.json').read_text())
    assert config['adapter_config']['splits']==['test']
    assert config['adapter_config']['allow_test'] is True


def test_resumed_checkpoint_validates_against_original_frozen_run(setup):
    root=setup['tmp_path'];resumed=root/'resumed_training'
    shutil.copytree(setup['runs']['remax']/'training',resumed)
    original_result=setup['runs']['remax']/'training/result.json'
    failed=json.loads(original_result.read_text());failed['status']='failed';save(original_result,failed)
    result=ep.prepare_pair(setup['base_request'],setup['base_run'],setup['runs']['maxrl'],setup['runs']['remax'],32,root/'requests',remax_checkpoint=resumed/'checkpoint-32',prepare_job=setup['freeze'],run_parent=root,run_prefix='endpoint')
    assert result['status']=='prepared_not_submitted'
    request=json.loads((root/'requests/remax_request.json').read_text())
    assert request['endpoint_provenance']['checkpoint']==str(resumed/'checkpoint-32')


def test_wrong_checkpoint_count_refused(setup):
    p=setup['runs']['maxrl']/'training/checkpoint-32/complete.json';r=json.loads(p.read_text());r['completed_updates']=16;save(p,r)
    with pytest.raises(ValueError,match='selected completed_updates'):setup['prepare']()


def test_transitive_verifier_change_is_rejected(setup):
    baseline=ep.frozen_run(setup['base_run'],'evaluate_real_domains_20260921.py')
    quality=Path(baseline['config']['adapter_config']['slate_root'])/'hardening_quality.json'
    save(quality,{'canonicalizer_source_files':{'src/pkg/core.py':ep.digest(setup['repo']/'src/pkg/core.py')},'verifier_support_files':{}})
    baseline['files'][quality]=ep.digest(quality)
    (setup['repo']/'src/pkg/core.py').write_text('altered canonicalizer dependency')
    with pytest.raises(ValueError,match='transitive verifier source mismatch'):
        ep.validate_critical_sources(baseline,{})
