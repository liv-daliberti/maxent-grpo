"""Write a read-only experiment census from actual completion receipts."""
from pathlib import Path
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/modebench_inference_followups_20260911'


def status():
    inputs=json.loads((BASE/'pantry/inputs.json').read_text());checkpoints=[]
    for cp in inputs['checkpoints']:
        folder=BASE/'pantry/results'/cp['label'];final=folder/'result.json';logs=list((BASE/'pantry/logs').glob('*.out'))
        checkpoints.append({'label':cp['label'],'complete':final.exists(),'portfolio_problems_saved':len(list((folder/'portfolios').glob('*.json'))),'request_groups_saved':len(list((folder/'requests').glob('*.json')))})
    job=json.loads((BASE/'pantry/submission_result.json').read_text())['job_id'];analysis_job=json.loads((BASE/'pantry/analysis_submission_result.json').read_text())['job_id']
    q=subprocess.run(['squeue','-j',job+','+analysis_job,'-h','-o','%i %T %M %R'],capture_output=True,text=True)
    hosted={}
    for model in ('gpt56sol','gpt54','grok43'):
        for arm in ('original','neutral'):
            p=ROOT/'artifacts/modebench_discovery_curves_20260911/hosted'/model/arm/'status.json'
            if p.exists():
                d=json.loads(p.read_text());hosted[model+'/'+arm]={k:d.get(k) for k in ('complete','completed_samples','expected_samples','updated_at_utc','failed_groups_this_session')};hosted[model+'/'+arm]['grading_complete']=(p.parent/'discovery_hosted_grading_audit.json').exists()
    report={'at_utc':datetime.now(timezone.utc).isoformat(),'cross_model':'complete' if (BASE/'offline/independent_audit.json').exists() else 'incomplete','coarse_frontier':'complete' if (BASE/'offline/independent_audit.json').exists() else 'incomplete','coarse_local_probes':'complete' if (BASE/'local_coarse_v2/results.json').exists() else 'incomplete','pantry_saved_portfolios':'complete' if (BASE/'pantry/saved_portfolios/results.json').exists() else 'incomplete','pantry_full_analysis':'complete' if (BASE/'pantry/analysis_complete/results.json').exists() else 'pending_all_five_workers','pantry_checkpoints':checkpoints,'pantry_array':job,'pantry_analysis_job':analysis_job,'slurm':q.stdout.strip().splitlines(),'slurm_query_returncode':q.returncode,'hosted_discovery':hosted}
    tmp=BASE/'RUN_STATUS.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,BASE/'RUN_STATUS.json');return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    while True:
        r=status();print(json.dumps(r),flush=True)
        if not a.watch or r['pantry_full_analysis']=='complete' or not r['slurm']:break
        time.sleep(30)
