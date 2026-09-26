"""Frozen-checkpoint Pantry adaptation, with immutable resumable request receipts."""
from pathlib import Path
import argparse,fcntl,importlib.metadata,json,os,sys,time
from types import SimpleNamespace
from datetime import datetime,timezone
CODE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CODE/'ops'),str(CODE/'src')]
from followup_metrics import atomic_new,file_sha,sha


def survivors(samples,spec):
    from oat_drgrpo.pantry_plan import validate_pantry_plan
    return [i for i,s in enumerate(samples) if s['verified'] and validate_pantry_plan(s['candidate'],spec) is not None]


def recovery_messages(messages,update,portfolio,attempts):
    text=messages[1]['content']+'\n\nUPDATED REQUIREMENTS:\n'+update
    text+='\n\nPreviously saved plans (some may be invalid):\n'+'\n'.join(f'{i+1}. {s["text"]}' for i,s in enumerate(portfolio))
    if attempts:text+='\n\nRecovery attempts rejected by the verifier:\n'+'\n'.join(f'{i+1}. {s["text"]}\nVerifier: invalid under the updated constraints.' for i,s in enumerate(attempts))
    text+='\n\nReturn one valid plan for the updated requirements using the required boxed format.'
    return [messages[0],{'role':'user','content':text}]


def run(planpath,index):
    plan=json.loads(planpath.read_text())
    for p,h in plan['code_sha256'].items():
        if file_sha(p)!=h:raise ValueError('frozen code changed: '+p)
    if file_sha(plan['inputs'])!=plan['inputs_sha256']:raise ValueError('frozen input changed')
    data=json.loads(Path(plan['inputs']).read_text());checkpoint=data['checkpoints'][index]
    out=Path(data['base'])/'results'/checkpoint['label'];out.mkdir(parents=True,exist_ok=True)
    identity=sha({'plan':file_sha(planpath),'checkpoint':checkpoint})
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'result.json').exists():
            r=json.loads((out/'result.json').read_text());assert r['identity']==identity and r['status']=='complete';return
        for f in checkpoint['files']:
            if file_sha(Path(checkpoint['model_path'])/f['name'])!=f['sha256']:raise ValueError('checkpoint changed')
        from frontier_modebench_contract import make_messages
        from oat_drgrpo.math_grader import _extract_modebench_candidate,validated_modebench_outcome_key
        from evaluate_modebench_level2_viability import sampling_params
        import vllm
        llm=vllm.LLM(model=checkpoint['model_path'],dtype='float16',max_model_len=8192,
                     gpu_memory_utilization=.65,swap_space=4,enable_prefix_caching=True,enforce_eager=True)
        tok=llm.get_tokenizer()
        runtime=out/'runtime.json'
        if not runtime.exists():atomic_new(runtime,{'identity':identity,'hostname':os.uname().nodename,'job':os.environ.get('SLURM_JOB_ID'),'versions':{x:importlib.metadata.version(x) for x in ('torch','vllm','transformers')},'created_at':datetime.now(timezone.utc).isoformat()})
        seeds={};receipts=[]
        def call(uid,messages,row,temperature=1.0,n=1):
            seed=1000000000+int(sha([checkpoint['label'],uid])[:15],16)%1000000000
            if seed in seeds and seeds[seed]!=uid:raise ValueError('seed collision')
            for child in range(seed,seed+n):
                if child in seeds and seeds[child]!=uid:raise ValueError('child seed collision')
                seeds[child]=uid
            rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            nt=len(tok.encode(rendered,add_special_tokens=False))
            if nt+192>8192:raise ValueError('full saved history exceeds frozen context; no silent truncation')
            req={'uid':uid,'messages':messages,'row_sha256':sha(row),'n':n,'temperature':temperature,'seed':seed,'max_tokens':192,'prompt_tokens':nt}
            path=out/'requests'/(uid+'.json');binding=sha([identity,req])
            if path.exists():
                result=json.loads(path.read_text());assert result['binding']==binding and len(result['samples'])==n
            else:
                a=SimpleNamespace(temperature=temperature,top_p=1.0,max_tokens=192,seed=seed,syntax_profile='domain_legal_v1')
                ref=sampling_params(a,'pantry',row)
                params=vllm.SamplingParams(n=n,temperature=temperature,top_p=1.0,max_tokens=192,seed=seed,guided_decoding=ref.guided_decoding)
                start=time.monotonic();outputs=llm.generate([rendered],[params],use_tqdm=False);seconds=time.monotonic()-start
                assert len(outputs)==1 and len(outputs[0].outputs)==n and outputs[0].prompt==rendered
                samples=[]
                for i,s in enumerate(sorted(outputs[0].outputs,key=lambda s:s.index)):
                    t0=time.monotonic();key=validated_modebench_outcome_key(s.text,row['answer']);vt=time.monotonic()-t0
                    candidate=_extract_modebench_candidate(s.text,row['answer'])
                    samples.append({'text':s.text,'candidate':candidate,'verified':key is not None,'canonical_key':key,'output_tokens':len(s.token_ids),'finish_reason':s.finish_reason,'seed':seed+i,'verification_seconds':vt})
                result={'binding':binding,'request':req,'samples':samples,'generation_wall_seconds':seconds,'input_tokens_per_response':nt,'logical_input_tokens':nt*n}
                atomic_new(path,result)
            receipts.append({'path':str(path),'sha256':file_sha(path)})
            return result['samples']
        tasks=data['tasks'];dev=[t for t in tasks if t['split']=='dev'];ev=[t for t in tasks if t['split']=='eval']
        temperatures=data['temperature_grid'];calibration=[]
        for temperature in temperatures:
            values=[]
            for task in dev:
                row=task['row'];ss=call(f'dev_t{temperature}_{task["id"]}',make_messages(2,'pantry',row),row,temperature,8)
                pp=[p for p in task['perturbations'] if p['feasible'] and p['kind']=='outage']
                values.append(sum(bool(survivors(ss,p['spec'])) for p in pp)/len(pp) if pp else None)
            defined=[v for v in values if v is not None]
            if not defined:raise ValueError('no development outages')
            calibration.append({'temperature':temperature,'survival':sum(defined)/len(defined),'eligible_dev_problems':len(defined),'per_problem':values})
        chosen=min(calibration,key=lambda r:(-r['survival'],abs(r['temperature']-1),r['temperature']))['temperature']
        if not (out/'calibration.json').exists():atomic_new(out/'calibration.json',{'identity':identity,'results':calibration,'chosen_temperature':chosen})
        # Only now read the original evaluation portfolios; perturbations were
        # exhaustively certified and frozen before this worker was submitted.
        savedpath=Path(data['saved_results_root'])/checkpoint['label']/'result.json'
        assert file_sha(savedpath)==plan['saved_result_sha256'][checkpoint['label']]
        saved=json.loads(savedpath.read_text());assert saved['status']=='complete'
        assert saved['identity']['checkpoint']==checkpoint
        if file_sha(saved['responses_path'])!=saved['responses_sha256']:raise ValueError('saved portfolio responses changed')
        old={r['row_index']:r for r in saved['prompt_results'] if r['level']==2 and r['domain']=='pantry' and r['arm']=='original'}
        summaries=[]
        for task in ev:
            row=task['row'];messages=make_messages(2,'pantry',row);original=[]
            for s in old[row['row_index']]['attempts']:
                candidate=_extract_modebench_candidate(s['text'],row['answer']);key=validated_modebench_outcome_key(s['text'],row['answer'])
                if s['verified']!=(key is not None) or s['canonical_key']!=key:raise ValueError('saved strict grade disagrees')
                original.append({**s,'candidate':candidate,'output_tokens':s['token_count'],'source':'previous_original_n8'})
            assert len(original)==8
            portfolios={'ordinary':original,'temperature':call('eval_temp_'+task['id'],messages,row,chosen,8)}
            diverse=[]
            for round in range(8):
                history='\n'.join(f'{i+1}. {s["text"]}' for i,s in enumerate(diverse)) or '(none yet)'
                dm=[messages[0],{'role':'user','content':messages[1]['content']+'\n\nPrevious responses (including invalid attempts):\n'+history+'\n\nGive one valid plan using a different ingredient support from all previous responses, where feasible. Use the required boxed format.'}]
                diverse+=call(f'eval_diverse_{task["id"]}_{round}',dm,row,1.0,1)
            portfolios['diversity_prompt']=diverse
            if not (out/'portfolios'/f'{task["id"]}.json').exists():atomic_new(out/'portfolios'/f'{task["id"]}.json',{'identity':identity,'task':task['id'],'portfolios':portfolios,'chosen_temperature':chosen,'ordinary_input_tokens_per_response':len(tok.encode(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),add_special_tokens=False))})
            for strategy,ss in portfolios.items():
                for pi,p in enumerate(task['perturbations']):
                    if not p['feasible']:continue
                    found=survivors(ss,p['spec']);attempts=[];recovered=bool(found)
                    revised={**row,'answer':json.dumps(p['spec'],sort_keys=True)}
                    while not recovered and len(attempts)<8:
                        uid=f'recovery_{task["id"]}_{strategy}_{pi}_{len(attempts)}'
                        rs=call(uid,recovery_messages(messages,p['update'],ss,attempts),revised,1.0,1)
                        attempts+=rs;recovered=rs[0]['verified']
                    summary={'task':task['id'],'row_index':row['row_index'],'strategy':strategy,'kind':p['kind'],'perturbation':p['id'],'initial_correct':sum(s['verified'] for s in ss),'initial_distinct':len({s['canonical_key'] for s in ss if s['verified']}),'zero_call_recovery':bool(found),'saved_usable':len(found),'recovered':recovered,'recovery_calls':len(attempts),'recovery_output_tokens':sum(s['output_tokens'] for s in attempts)}
                    summaries.append(summary)
            print(json.dumps({'event':'problem_complete','checkpoint':checkpoint['label'],'task':task['id'],'requests':len(receipts)}),flush=True)
        atomic_new(out/'result.json',{'schema':'pantry-adaptation-result-v1','status':'complete','identity':identity,'checkpoint':checkpoint,'calibration':calibration,'chosen_temperature':chosen,'records':summaries,'request_receipts':receipts,'saved_source':{'path':str(savedpath),'sha256':file_sha(savedpath)},'cost_note':'Request receipts retain logical input/output tokens and generation timing. Shared prefixes affect realized compute. Ordinary portfolio historical generation timing is unavailable; response-count equality is not a compute-cost claim.'})
        print(json.dumps({'event':'complete','checkpoint':checkpoint['label'],'requests':len(receipts),'records':len(summaries)}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args();run(a.plan.resolve(),a.index)
