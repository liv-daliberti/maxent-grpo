import copy
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location('benchmark_e124', Path(__file__).parents[1] / 'ops/exp_scaling/benchmark_e124_qwen7b.py')
b = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(b)


def cells():
    result = []
    for level in (1, 2, 3):
        for domain in b.DOMAINS:
            for arm in b.ARMS:
                env = dict(b.PROFILE, OAT_ZERO_NUM_SAMPLES='16', OAT_ZERO_TRAIN_BATCH_SIZE='16',
                           OAT_ZERO_ROLLOUT_BATCH_SIZE='1', OAT_ZERO_NUM_PPO_EPOCHS='1',
                           OAT_ZERO_NUM_PROMPT_EPOCH='8', OAT_ZERO_MAX_PROMPT_EPOCHS='8', OAT_ZERO_MAX_TRAIN='384',
                           OAT_ZERO_N_GPU='1', OAT_ZERO_NUM_GPUS_PER_ACTOR='1', OAT_ZERO_MAXRL_TASK_OBJECTIVE='1',
                           OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE='verified_likelihood_per_rollout',
                           OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY='0' if arm == 'replay_maxrl' else '1',
                           OAT_ZERO_PRETRAIN='/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/' + b.REVISION,
                           OAT_ZERO_SEED='70', RUN_STAMP=f'e124_l{level}_{domain}_{arm}_s70', OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS='4', OAT_ZERO_EVAL_MODE_COVERAGE_K='8',
                           OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE='1.0', OAT_ZERO_TOP_P='1.0',
                           OAT_ZERO_EVAL_MODE_COVERAGE_SEED='76299', OAT_ZERO_EVAL_PROMPT_INTERVAL='96',
                           OAT_ZERO_EVAL_BATCH_SIZE='32', OAT_ZERO_LR_SCHEDULER='cosine', OAT_ZERO_LR_WARMUP_RATIO='0.1',
                           OAT_ZERO_LEARNING_RATE='1e-7', OAT_ZERO_PROMPT_MAX_LENGTH='1024', OAT_ZERO_GENERATE_MAX_LENGTH='192',
                           OAT_ZERO_CANONICAL_ACTION_TASK='pantry_support_mask' if level == 1 and domain == 'pantry_plan' else 'none')
                result.append({'level': level, 'domain': domain, 'arm': arm, 'seed': 70, 'environment': env})
    return result


def test_exact_thirty_matrix_and_single_seed():
    assert len(b.extract_cells({'cells': cells()})) == 30
    for mutate in ('duplicate', 'seed', 'missing'):
        c = cells()
        if mutate == 'duplicate': c[-1] = copy.deepcopy(c[0])
        elif mutate == 'seed': c[-1]['seed'] = 71
        else: c.pop()
        with pytest.raises(ValueError): b.extract_cells({'cells': c})


@pytest.mark.parametrize('key,value', [('OAT_ZERO_NUM_SAMPLES','8'), ('OAT_ZERO_TRAIN_BATCH_SIZE','32'),
    ('OAT_ZERO_ROLLOUT_BATCH_SIZE','4'), ('OAT_ZERO_NUM_PROMPT_EPOCH','1'), ('OAT_ZERO_MAXRL_TASK_OBJECTIVE','0'),
    ('OAT_ZERO_ADAM_OFFLOAD','0'), ('OAT_ZERO_VLLM_GPU_RATIO','0.25'), ('OMP_NUM_THREADS','1'),
    ('OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS','1'), ('OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY','0')])
def test_reject_science_or_fixed_profile_drift(key, value):
    c = cells(); c[0]['environment'][key] = value
    with pytest.raises(ValueError): b.extract_cells({'cells': c})


def test_smoke_preserves_horizon_sampling_and_canonical_path(tmp_path):
    c = next(c for c in cells() if c['level'] == 1 and c['domain'] == 'pantry_plan')
    fresh = b.e2e_environment({}, c, tmp_path/'fresh')
    resume = b.e2e_environment({}, c, tmp_path/'resume', tmp_path/'fresh/debug/checkpoints/step_00002')
    for key, value in c['environment'].items():
        assert fresh[key] == resume[key] == value
    assert fresh['OAT_ZERO_MAX_QUERIES'] == '16' and resume['OAT_ZERO_MAX_QUERIES'] == '32'
    assert fresh['OAT_ZERO_PRUNE_RESUME_ON_SUCCESS'] == '0'
    assert fresh['RUN_STAMP'] == resume['RUN_STAMP'] == c['environment']['RUN_STAMP']
    assert 'OAT_ZERO_RESUME_DIR' not in fresh
    assert resume['OAT_ZERO_RESUME_TAG'] == 'step_00002'


def test_root_not_inferred_from_snapshot():
    env = b.runtime_environment({'workspace_root':'/workspace', 'runtime':{'source_root':'/workspace/snapshot/src'}})
    assert env['CUDA_HOME'] == '/workspace/var/cuda124_toolkit'
    assert env['MAXENT_GRPO_ROOT'] == env['OAT_ZERO_REPO_ROOT'] == '/workspace'
    assert env['PYTHONPATH'] == '/workspace/snapshot/src'


def checkpoint(path, step=2, optimizer=True, bank=True):
    path.mkdir()
    state = dict(global_steps=step, global_step=step, prompt_batches_consumed_total=step)
    if bank: state['online_canonical_bank_state'] = {'some':'actual state'}
    with zipfile.ZipFile(path/'mp_rank_00_model_states.pt','w') as z: z.writestr('archive/data.pkl',pickle.dumps(state,protocol=4))
    if optimizer:
        with zipfile.ZipFile(path/'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt','w') as z:
            z.writestr('archive/data.pkl',pickle.dumps({'optimizer_state_dict':{}}))
    return path


def test_full_checkpoint_counters_and_bank_required(tmp_path):
    good=b.checkpoint_metadata(checkpoint(tmp_path/'good'),2)
    assert good['step']==2 and good['bank_state_present'] and good['bytes']>0
    for name,kw in [('noopt',{'optimizer':False}),('nobank',{'bank':False}),('wrongstep',{'step':3})]:
        with pytest.raises(ValueError): b.checkpoint_metadata(checkpoint(tmp_path/name,**kw),2)


def draws():
    return [{'evaluation_kind':'fixed_seed_sampled_k_neutral','sample_count':8,'temperature':1.,'top_p':1.,
             'seed':76299+i,'draw_index':i,'step':2,'prompts':[{'prompt_index':j,'responses':['valid output']*8} for j in range(2)]} for i in range(4)]


def test_complete_draws_never_gate_on_score():
    d=draws(); e=cells()[0]['environment']
    assert b.validate_evaluation(d,e,2)=={'2':[0,1,2,3]}
    for mutation in ('seed','partial','duplicate','response','binary'):
        d=draws()
        if mutation=='seed':d[0]['seed']+=1
        elif mutation=='partial':d.pop()
        elif mutation=='duplicate':d.append(copy.deepcopy(d[0]))
        elif mutation=='response':d[0]['prompts'][0]['responses'].pop()
        else:d[0]['prompts'][0]['responses'][0]='bad\x00'
        with pytest.raises(ValueError): b.validate_evaluation(d,e,2)


def qualification(tmp_path):
    keys=['l1_graph_coloring_replay_maxrl']
    plan={'plan_sha256':'plan','output_root':str(tmp_path),'coverage':keys,'resources':{'host_memory_gib':256},
          'runtime':{},'model':{'weight_bytes':14*b.GIB},'datasets':{},'manifest_path':'/manifest','manifest_sha256':'manifest'}
    memory={'available':True,'sample_count':3,'events_delta':{'high':0,'oom':0,'oom_kill':0},'peak_nonreclaimable_dirty_bytes':100*b.GIB}
    stress={'status':'passed','plan_sha256':'plan','own_checkpoint_resume':True,'full_shape_arms':list(b.ARMS),
            'host_memory':copy.deepcopy(memory),'device':{'total_memory_bytes':48*b.GIB},'gpu_peak_reserved_bytes':40*b.GIB,'checkpoint_bytes':100*b.GIB}
    results={keys[0]:{'status':'passed','plan_sha256':'plan','own_checkpoint_resume':True,'host_memory':copy.deepcopy(memory),
                     'gpu':{'sample_count':3,'peak_bytes':40*b.GIB},'device_total_bytes':48*b.GIB,'checkpoint_bytes':100*b.GIB}}
    p=tmp_path/b.PROFILE_ID/'stress_result.json';b.write(p,stress)
    for key,value in results.items():b.write(tmp_path/'end_to_end'/key/'result.json',value)
    return plan,stress,results


def test_pass_profile_binds_manifest_and_immutable_evidence(tmp_path):
    p,s,c=qualification(tmp_path); q=b.qualify(p,s,c)
    assert q['status']=='passed' and q['manifest_path']=='/manifest' and q['manifest_sha256']=='manifest'
    assert q['measurements']['concurrency_tested']==1
    assert q['measurements']['required_storage_reserve_bytes']==218*b.GIB
    assert len(q['evidence']['files'])==2 and not q['outcomes_used_for_selection']
    assert b.identity({k:v for k,v in q.items() if k!='profile_sha256'})==q['profile_sha256']


@pytest.mark.parametrize('bad', ['host','gpu','pressure','coverage','resume','plan','arm','storage'])
def test_qualification_fail_closed(tmp_path,bad):
    p,s,c=qualification(tmp_path)
    if bad=='host':s['host_memory']['peak_nonreclaimable_dirty_bytes']=220*b.GIB
    elif bad=='gpu':s['gpu_peak_reserved_bytes']=47*b.GIB
    elif bad=='pressure':s['host_memory']['events_delta']['high']=1
    elif bad=='coverage':c={}
    elif bad=='resume':s['own_checkpoint_resume']=False
    elif bad=='plan':s['plan_sha256']='other'
    elif bad=='storage':s['checkpoint_bytes']=102*b.GIB
    else:s['full_shape_arms']=['replay_maxrl']
    with pytest.raises(ValueError):b.qualify(p,s,c)


def test_partial_stress_never_blindly_retries(tmp_path,monkeypatch):
    p=tmp_path/'plan.json';b.write(p,{'output_root':str(tmp_path),'plan_sha256':'p'})
    (tmp_path/b.PROFILE_ID).mkdir()
    monkeypatch.setattr(b,'verify_pins',lambda *a,**k:None)
    monkeypatch.setattr(b,'run_process',lambda *a,**k:pytest.fail('must not rerun partial stage'))
    assert b.run_suite(p)==1
    assert b.read(tmp_path/'suite_status.json')['review_required']
    assert not (tmp_path/'qualified_profile.json').exists()


def test_failed_stress_never_blindly_retries(tmp_path,monkeypatch):
    p=tmp_path/'plan.json';b.write(p,{'output_root':str(tmp_path),'plan_sha256':'p'})
    b.write(tmp_path/b.PROFILE_ID/'stress_result.json',{'plan_sha256':'p','status':'failed'})
    monkeypatch.setattr(b,'verify_pins',lambda *a,**k:None)
    monkeypatch.setattr(b,'run_process',lambda *a,**k:pytest.fail('must not rerun failed stage'))
    assert b.run_suite(p)==1
    assert b.read(tmp_path/'suite_status.json')['review_required']


def test_cleanup_refuses_science_path_and_symlink(tmp_path):
    science=tmp_path/'science';science.mkdir(); output=tmp_path/'benchmark';output.mkdir()
    with pytest.raises(ValueError):b.safe_retire_checkpoint(science,output)
    link=output/'step_00002';link.symlink_to(science,target_is_directory=True)
    with pytest.raises(ValueError):b.safe_retire_checkpoint(link,output)
    assert science.is_dir()


def test_prepare_binds_frozen_control_model_and_dataset_bytes(tmp_path):
    root=tmp_path/'workspace';snapshot=root/'snapshot';source=snapshot/'src';source.mkdir(parents=True)
    (source/'runtime.py').write_text('frozen_source = True\n')
    (snapshot/'ops').mkdir();(snapshot/'ops/train.sh').write_text('true\n');(snapshot/'ops/repo_env.sh').write_text('true\n')
    (snapshot/'control').mkdir();(snapshot/'control'/b.SOURCE.name).write_bytes(b.SOURCE.read_bytes())
    model=root/'models--Qwen--Qwen2.5-7B-Instruct'/'snapshots'/b.REVISION;model.mkdir(parents=True)
    for n in ('config.json','tokenizer.json'):(model/n).write_text('{}')
    (model/'weights.safetensors').write_bytes(b'frozen weights')
    b.write(model/'model.safetensors.index.json',{'weight_map':{'w':'weights.safetensors'}})
    data=root/'data';data.mkdir();(data/'data.arrow').write_bytes(b'frozen dataset')
    c=cells()
    for row in c:
        row['environment'].update(OAT_ZERO_REPO_ROOT=str(root),OAT_ZERO_SOURCE_ROOT=str(source),
            OAT_ZERO_OPS_SNAPSHOT_ROOT=str(snapshot/'ops'),OAT_ZERO_PRETRAIN=str(model),
            OAT_ZERO_PROMPT_DATA=str(data),OAT_ZERO_EVAL_DATA=str(data))
    manifest=root/'manifest.json';b.write(manifest,{'cells':c,'snapshot':{'root':str(snapshot),'sha256':'a'*64}})
    plan=b.prepare(manifest,root/'benchmark')
    assert len(plan['coverage'])==16 and len(plan['cells'])==30
    assert str(snapshot/'control'/b.SOURCE.name) in plan['runtime']['files']
    b.verify_pins(plan)
    with pytest.raises(ValueError):b.prepare(manifest,root/'benchmark')
    (data/'data.arrow').write_bytes(b'changed dataset')
    with pytest.raises(ValueError,match='frozen input changed'):b.verify_pins(plan)


def test_gpu_gate_rejects_unqualified_classes():
    class Cuda:
        def is_available(self):return True
        def device_count(self):return 1
        def get_device_properties(self,i):return self.props
    cuda=Cuda();torch=SimpleNamespace(cuda=cuda)
    cuda.props=SimpleNamespace(name='NVIDIA RTX A6000',total_memory=48*b.GIB)
    assert b.require_gpu(torch).total_memory==48*b.GIB
    for name,size in [('NVIDIA A100',80),('NVIDIA RTX A5000',24),('NVIDIA L40',48)]:
        cuda.props=SimpleNamespace(name=name,total_memory=size*b.GIB)
        with pytest.raises(ValueError):b.require_gpu(torch)
