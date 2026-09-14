#!/usr/bin/env python3
"""Run authenticated hosted discovery cohorts with an intended credential source."""
from __future__ import annotations
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from prepare_modebench_discovery_hosted import BASE,CONDITION,MODELS,ARMS,authenticate,sha_file,write
from run_hosted_prompt_ablation import credential,authenticated_completed

ROOT=Path(__file__).resolve().parents[1]

def inventory(base):
    base=Path(base).resolve();manifest=json.loads((base/'manifest.json').read_text());authenticate(base,manifest)
    registry=json.loads((base/'hosted_analysis_runs.json').read_text())
    assert registry['experiment_condition']==CONDITION
    assert registry['ablation_manifest_sha256']==sha_file(base/'manifest.json')
    amendment=json.loads((base/'hosted_execution.json').read_text())
    assert amendment['root_manifest_sha256']==sha_file(base/'manifest.json')
    assert amendment['registry_sha256']==sha_file(base/'hosted_analysis_runs.json')
    for path,digest in amendment['execution_code_sha256'].items():
        assert sha_file(ROOT/path)==digest, 'Execution code changed: '+path
    seen=set()
    for entry in registry['runs']:
        key=entry['model_id'],entry['arm'];assert key not in seen;seen.add(key)
        out=Path(entry['run_dir']);assert out.resolve()==base/'hosted'/key[0]/key[1]
        assert sha_file(out/'manifest.json')==entry['manifest_sha256']
        current=json.loads((out/'manifest.json').read_text());authenticate(out,current)
        assert current['experiment_condition']==CONDITION and current['sample_count']==64
        assert current['request_count']==6144 and current['prompt_count']==96
    assert seen=={(m,a) for m in MODELS for a in ARMS}
    return registry['runs']

def preflight_receipt(out,evidence):
    assert evidence
    write(out/'discovery_preflight.json',{'schema':'discovery-hosted-preflight-v1',
          'manifest_sha256':sha_file(out/'manifest.json'),'authenticated_sample_count':len(evidence),
          'evidence':evidence,'retained_in_production':True,
          'checked_at_utc':datetime.now(timezone.utc).isoformat()})

def authenticated_inventory(entries):
    """Reject a reused native sample even when it appears in different arms."""
    result={};seen={}
    for entry in entries:
        out=Path(entry['run_dir'])
        manifest=json.loads((out/'manifest.json').read_text())
        evidence=authenticated_completed(out)
        assert len(evidence)<=6144
        for item in evidence:
            provider=tuple(item['provider_sample_identity'])
            identity=(entry['model'],manifest['protocol'],*provider)
            if identity in seen:
                raise ValueError('Duplicate provider sample across hosted cohorts: '+str(identity))
            seen[identity]=(str(out),item['sample_id'])
        result[str(out)]=evidence
    return result


def run(command,base=BASE,credential_file=None):
    base=Path(base).resolve();entries=inventory(base)
    if command=='status':
        evidence=authenticated_inventory(entries)
        for entry in entries:
            print(json.dumps({'model':entry['model'],'arm':entry['arm'],
                              'authenticated_responses':len(evidence[entry['run_dir']]),'planned_responses':6144}))
        return
    with (base/'hosted_runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inventory_evidence=authenticated_inventory(entries)
        pending=[]
        for entry in entries:
            out=Path(entry['run_dir']);evidence=inventory_evidence[str(out)]
            if command=='preflight' and evidence:
                preflight_receipt(out,evidence);continue
            if command=='full':
                if not evidence:
                    raise ValueError('Run and inspect an authenticated preflight before full collection')
                marker_path=out/'discovery_preflight.json'
                if not marker_path.exists():
                    preflight_receipt(out,evidence)
                marker=json.loads(marker_path.read_text())
                assert marker['manifest_sha256']==sha_file(out/'manifest.json')
                current={x['sample_id']:x for x in evidence}
                assert marker['evidence'] and all(current.get(x['sample_id'])==x for x in marker['evidence'])
                if len(evidence)==6144:continue
            pending.append(entry)
        # Completed receipts can be recovered without any credential or HTTP.
        key=credential(credential_file) if pending else None
        processes=[]
        try:
            for entry in pending:
                out=Path(entry['run_dir']);env=os.environ.copy();env['AZURE_OPENAI_API_KEY']=key
                adapter=out/'code/ops/evaluate_native_prompt_ablation.py'
                cmd=[sys.executable,str(adapter),'run','--model',entry['model'],'--output',str(out),
                     '--workers','8','--request-timeout','300','--max-attempts','8']
                if command=='preflight':cmd+=['--max-new','1']
                log=(out/('discovery_'+command+'.log')).open('a')
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                processes.append((entry,process,log))
            failed=[]
            for entry,process,log in processes:
                code=process.wait();log.close()
                if code:failed.append({'model':entry['model'],'arm':entry['arm'],'exit_code':code})
            final_evidence=authenticated_inventory(entries)
            for entry in pending:
                out=Path(entry['run_dir']);evidence=final_evidence[str(out)]
                if (command=='full' and len(evidence)!=6144) or not evidence:
                    failed.append({'model':entry['model'],'arm':entry['arm'],'authenticated':len(evidence)})
            if failed:raise RuntimeError(json.dumps(failed))
            if command=='preflight':
                for entry in pending:
                    out=Path(entry['run_dir']);preflight_receipt(out,final_evidence[str(out)])
        except BaseException:
            for _,process,log in processes:
                if process.poll() is None:process.terminate()
            for _,process,log in processes:
                process.wait();log.close()
            raise
        finally:
            key=None
    print(json.dumps({'stage':command,'status':'complete','newly_launched_cohorts':len(pending)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('status','preflight','full'))
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--credential-file',type=Path)
    args=parser.parse_args();run(args.command,args.base,args.credential_file)
