#!/usr/bin/env python3
"""Validate and freeze paired terminal LoRA endpoints; never submit jobs.

A frozen baseline supplies evaluator bytes, tasks and the entire generation
configuration. Terminal checkpoint parents include adapter, optimizer, bank and
completion seal. Original training identities are retained in request provenance.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import prepare_real_domains_run_20260921 as launcher
import evaluate_real_domains_20260921 as evaluator

ROOT=Path(__file__).resolve().parents[1]
MUTABLE={'build_root','runtime_root','launcher','scratch_root'}
SELECTION={'problem_ids','task_ids','allow_heldout','allow_test','split','splits'}
LORA={'lora_path','lora_arm','lora_completed_updates','lora_checkpoint_config_sha256','max_lora_rank'}


def digest(path):
    return launcher.digest(Path(path))


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def frozen_run(path,entrypoint,arm=None):
    path=path.resolve();identity=json.loads((path/'identity.json').read_text());config=json.loads((path/'config.json').read_text());bundle=path/'bundle'
    if identity.get('schema')!='real-domains-frozen-job-20260921-v1' or identity['request'].get('entrypoint')!=entrypoint or (arm is not None and identity['request'].get('arm')!=arm):
        raise ValueError('frozen job entrypoint or arm mismatch')
    if identity['config_sha256']!=digest(path/'config.json'):
        raise ValueError('frozen job configuration checksum mismatch')
    files={}
    for row in identity['files']:
        snapshot=Path(row['snapshot']).resolve()
        if not snapshot.is_relative_to(bundle) or snapshot in files or digest(snapshot)!=row['sha256']:
            raise ValueError('frozen dependency checksum or custody mismatch')
        files[snapshot]=row['sha256']
    replacements=[(str(Path(source).resolve()),str(bundle/'data'/str(i))) for i,source in enumerate(identity['request'].get('freeze_data_roots',[]))]
    if config!=launcher.rewrite_paths(identity['request']['config'],replacements):
        raise ValueError('frozen request and runnable configuration differ')
    return {'root':path,'identity':identity,'config':config,'files':files}


def source(run,relative):
    path=run['root']/'bundle'/relative
    if path not in run['files']:
        raise ValueError('required source absent from frozen identity: '+relative)
    return run['files'][path]


def adapter_relative(name):
    return ('src/'+name.replace('.','/')+'.py') if '.' in name else 'ops/'+name+'.py'


def adapter_contract(config,run,include_selection=False):
    def content(value):
        if isinstance(value,dict):return {k:content(v) for k,v in value.items()}
        if isinstance(value,list):return [content(v) for v in value]
        if isinstance(value,str) and value.startswith('/'):
            path=Path(value).resolve()
            if path in run['files']:return {'file_sha256':run['files'][path]}
            members={p.relative_to(path).as_posix():sha for p,sha in run['files'].items() if p.is_relative_to(path)}
            if members:return {'frozen_tree':members}
            if path.is_file():return {'external_file_sha256':digest(path)}
            raise ValueError('adapter references an unfrozen or missing path: '+value)
        return value
    ignored=MUTABLE|(set() if include_selection else SELECTION)
    return content({k:v for k,v in config.items() if k not in ignored})


def validate_critical_sources(baseline,runs):
    """Enforce a coding adapter's declared transitive closure without imports."""
    slate=baseline['config']['adapter_config'].get('slate_root')
    if not slate:return
    path=Path(slate)/'hardening_quality.json'
    if not path.is_file():return
    if path not in baseline['files'] or digest(path)!=baseline['files'][path]:
        raise ValueError('verifier source closure is not bound to the frozen baseline')
    quality=json.loads(path.read_text())
    pins={**quality['canonicalizer_source_files'],**quality['verifier_support_files']}
    for relative,expected in pins.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('source closure must use safe project-relative identities')
        frozen_relative='testlib/testlib.h' if relative=='third_party/testlib/testlib.h' else relative
        if digest(ROOT/relative)!=expected or any(source(run,frozen_relative)!=expected for run in [baseline]+list(runs.values())):
            raise ValueError('baseline/training/current transitive verifier source mismatch')


def validate_checkpoint(run,checkpoint,arm,updates,baseline):
    checkpoint=checkpoint.resolve();identity_path=checkpoint.parent/'identity.json';result_path=checkpoint.parent/'result.json'
    training=json.loads(identity_path.read_text());result=json.loads(result_path.read_text());resolved=training['config'];cfg=run['config']
    if training.get('arm')!=arm or training.get('input_config_sha256')!=digest(run['root']/'config.json') or training.get('config_sha256')!=canonical(resolved) or any(resolved.get(k)!=v for k,v in cfg.items()):
        raise ValueError('training identity/configuration/arm mismatch')
    if type(updates) is not int or updates<1 or resolved.get('updates')!=updates or checkpoint.name!=f'checkpoint-{updates}':
        raise ValueError('endpoint must select the exact positive terminal update count')
    if result.get('status')!='complete' or result.get('arm')!=arm or result.get('completed_updates')!=updates or result.get('config_sha256')!=training['config_sha256']:
        raise ValueError('training is unfinished or its terminal result identity differs')
    if training.get('runner_sha256')!=source(run,'ops/train_real_domains_pilot_20260921.py'):
        raise ValueError('training runner source mismatch')
    module=cfg['adapter_module'];relative=adapter_relative(module)
    if Path(training['adapter_module']).resolve()!=run['root']/'bundle'/relative or training.get('adapter_module_sha256')!=source(run,relative) or source(run,relative)!=source(baseline,relative):
        raise ValueError('training/baseline adapter source mismatch')
    for name,sha in training.get('production_source_sha256',{}).items():
        if sha!=source(run,'src/'+name.replace('.','/')+'.py'):
            raise ValueError('production training source mismatch')
    base=baseline['config'];model=Path(cfg['model']).resolve()
    if cfg['model_revision']!=base['model_revision'] or model!=Path(base['model']).resolve() or len(cfg['model_revision'])!=40 or model.name!=cfg['model_revision'] or digest(model/'config.json')!=training['model_config_sha256']:
        raise ValueError('training/baseline model revision or configuration mismatch')
    if module!=base['adapter_module'] or adapter_contract(cfg['adapter_config'],run)!=adapter_contract(base['adapter_config'],baseline):
        raise ValueError('training/baseline verifier or dataset mismatch')
    lora={'lora_path':str(checkpoint/'adapter'),'lora_arm':arm,'lora_completed_updates':updates,'lora_checkpoint_config_sha256':training['config_sha256'],'max_lora_rank':resolved['lora_rank']}
    receipt=evaluator.verify_lora_checkpoint(lora,model)
    if type(receipt['seal']['completed_updates']) is not int or receipt['adapter_config'].get('r')!=resolved['lora_rank'] or receipt['adapter_config'].get('lora_alpha')!=resolved['lora_alpha'] or set(receipt['adapter_config'].get('target_modules',[]))!=set(resolved['lora_target_modules']):
        raise ValueError('checkpoint LoRA architecture differs from training configuration')
    paired=copy.deepcopy(resolved);paired['adapter_config']=adapter_contract(resolved['adapter_config'],run,include_selection=True)
    return {'lora':lora,'paired_config':paired,'training_identity':training,'training_result':result,'checkpoint':str(checkpoint),'checkpoint_seal_sha256':receipt['seal_sha256'],'original_run_identity_sha256':digest(run['root']/'identity.json'),'training_identity_sha256':digest(identity_path),'training_result_sha256':digest(result_path),'production_source_sha256':training.get('production_source_sha256',{}),'runner_sha256':training['runner_sha256']}


def prepare_pair(baseline_request,baseline_run,maxrl_run,remax_run,updates,output,maxrl_checkpoint=None,remax_checkpoint=None,prepare_job=None,run_parent=None,run_prefix=None):
    output=output.resolve()
    if output.exists():raise FileExistsError('endpoint output directory must be new')
    run_parent=(run_parent or output.parent).resolve();run_prefix=run_prefix or output.name
    if not run_prefix or Path(run_prefix).name!=run_prefix or run_prefix in ('.','..'):
        raise ValueError('run prefix must be a single nonempty directory name')
    destinations={arm:run_parent/(run_prefix+'_'+arm) for arm in ('maxrl','remax')}
    work={arm:run_parent/(run_prefix+'_'+arm+'_work') for arm in ('maxrl','remax')}
    for path in list(destinations.values())+list(work.values()):
        if path.exists() or path==output or path.is_relative_to(output):
            raise FileExistsError('endpoint run and work paths must be distinct new siblings')
    baseline=frozen_run(baseline_run,'evaluate_real_domains_20260921.py');request=json.loads(baseline_request.read_text())
    if digest(baseline_request)!=baseline['identity']['request_sha256'] or request!=baseline['identity']['request']:
        raise ValueError('baseline request is not the frozen baseline request')
    if any(k in baseline['config'] for k in LORA):raise ValueError('baseline must be the unadapted base policy')
    evaluator_sha=source(baseline,'ops/evaluate_real_domains_20260921.py')
    relative=adapter_relative(baseline['config']['adapter_module'])
    if digest(ROOT/'ops/evaluate_real_domains_20260921.py')!=evaluator_sha or digest(ROOT/relative)!=source(baseline,relative):
        raise ValueError('current evaluator/adapter differs from frozen baseline; refuse source drift')
    provenance={};runs={}
    for arm,run_path,selected in (('maxrl',maxrl_run,maxrl_checkpoint),('remax',remax_run,remax_checkpoint)):
        run=frozen_run(run_path,'train_real_domains_pilot_20260921.py',arm);runs[arm]=run
        checkpoint=selected or run['root']/'training'/f'checkpoint-{updates}'
        provenance[arm]=validate_checkpoint(run,checkpoint,arm,updates,baseline)
    validate_critical_sources(baseline,runs)
    if provenance['maxrl']['paired_config']!=provenance['remax']['paired_config'] or provenance['maxrl']['runner_sha256']!=provenance['remax']['runner_sha256'] or provenance['maxrl']['production_source_sha256']!=provenance['remax']['production_source_sha256']:
        raise ValueError('paired training configurations or learner sources differ')
    output.mkdir(parents=True)
    prepare_job=prepare_job or launcher.prepare
    generated={};generation=copy.deepcopy(baseline['config']);generation['adapter_config']={k:v for k,v in generation['adapter_config'].items() if k not in MUTABLE}
    for arm in ('maxrl','remax'):
        endpoint=copy.deepcopy(request);config=copy.deepcopy(baseline['config']);config.update(provenance[arm]['lora'])
        for key in MUTABLE:
            if key in config['adapter_config']:config['adapter_config'][key]=str(work[arm]/key)
        endpoint['config']=config
        endpoint['freeze_data_roots']=[str(baseline['root']/'bundle/data'/str(i)) for i,_ in enumerate(request.get('freeze_data_roots',[]))]+[provenance[arm]['checkpoint']]
        endpoint['job_name']='real-endpoint-'+arm
        endpoint['endpoint_provenance']={k:v for k,v in provenance[arm].items() if k not in ('lora','paired_config')}
        endpoint['endpoint_provenance'].update({'baseline_request_sha256':digest(baseline_request),'baseline_frozen_identity_sha256':digest(baseline['root']/'identity.json'),'baseline_evaluator_sha256':evaluator_sha,'generation_contract_sha256':canonical(generation),'preparer_sha256':digest(Path(__file__))})
        path=output/(arm+'_request.json');write(path,endpoint)
        job=destinations[arm];prepared=prepare_job(path,job,False)
        frozen=frozen_run(job,'evaluate_real_domains_20260921.py')
        if source(frozen,'ops/evaluate_real_domains_20260921.py')!=evaluator_sha or source(frozen,relative)!=source(baseline,relative):raise ValueError('endpoint source drift while freezing')
        validate_critical_sources(baseline,{'endpoint':frozen})
        copied=evaluator.verify_lora_checkpoint(frozen['config'],Path(config['model']))
        if copied['seal_sha256']!=provenance[arm]['checkpoint_seal_sha256']:raise ValueError('checkpoint changed while freezing')
        generated[arm]={'request':str(path),'request_sha256':digest(path),'run':str(job),'prepared':prepared,'checkpoint_seal_sha256':copied['seal_sha256']}
    result={'schema':'real-domains-paired-endpoint-preparation-20260921-v1','status':'prepared_not_submitted','completed_updates':updates,'generation_contract_sha256':canonical(generation),'baseline_evaluator_sha256':evaluator_sha,'arms':generated}
    write(output/'endpoint_preparation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-request','baseline-run','maxrl-run','remax-run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--completed-updates',type=int,required=True)
    p.add_argument('--maxrl-checkpoint',type=Path);p.add_argument('--remax-checkpoint',type=Path)
    p.add_argument('--run-parent',type=Path);p.add_argument('--run-prefix')
    a=p.parse_args();print(json.dumps(prepare_pair(a.baseline_request,a.baseline_run,a.maxrl_run,a.remax_run,a.completed_updates,a.output,a.maxrl_checkpoint,a.remax_checkpoint,run_parent=a.run_parent,run_prefix=a.run_prefix),indent=2))


if __name__=='__main__':main()
