#!/usr/bin/env python3
"""Launch the reviewed CPU-only timeout guard and read-only startup observer."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
import guard_pantry_timeout_31048182_20260909 as guard
import prioritize_e118_capacity_20260905 as base
ART=guard.ART

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(d):base.atomic(ART/'supervisor_launch.json',d)

def main():
 assert not (ART/'supervisor_launch.json').exists(),'Inspect an existing submission; never submit blindly twice'
 if not guard.PLAN.exists():guard.prepare()
 assert json.loads(guard.PLAN.read_text())['rows'][0]['job_id']==31048182
 script=ART/'supervisor.slurm'
 observer=ROOT/'ops/exp_scaling/observe_e119_pantry_s44_memory116_20260909.py'
 python='/usr/local/anaconda3/2024.02/bin/python3'
 script.write_text('#!/bin/bash\nset -euo pipefail\nexport PATH=/usr/bin:/bin\ncd '+str(ROOT)+'\n'+python+' '+str(observer)+' >'+str(ROOT/'var/artifacts/e119_pantry_s44_memory116_20260909/startup_observer.out')+' 2>'+str(ROOT/'var/artifacts/e119_pantry_s44_memory116_20260909/startup_observer.err')+' &\nexec '+python+' '+str(pathlib.Path(guard.__file__).resolve())+' watch --apply\n')
 command=['sbatch','--parsable','--hold','--job-name=pantry-timeout-guard-31048182','--account=mltheory','--partition=mltheory','--nodelist=node915','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=256M','--gres=none','--time=2-00:10:00','--requeue','--chdir='+str(ROOT),'--output='+str(ART/'guard-%j.out'),'--error='+str(ART/'guard-%j.err'),'--export=NONE','--comment=campaign-timeout-guard-31048182-20260909',str(script)]
 d={'created_at_utc':base.now(),'command':command,'plan_sha256':sha(guard.PLAN),'script_sha256':sha(script),'observer_sha256':sha(observer),'submission_uncertain':True};save(d)
 r=subprocess.run(command,capture_output=True,text=True,timeout=45);d.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);save(d)
 assert r.returncode==0 and r.stdout.strip().split(';')[0].isdigit()
 jid=int(r.stdout.strip().split(';')[0]);d.update(job_id=jid,submission_uncertain=False);save(d)
 held=subprocess.check_output(['scontrol','show','job','-dd','-o',str(jid)],text=True)
 for k,v in {'JobState':'PENDING','Reason':'JobHeldUser','Account':'mltheory','Partition':'mltheory','ReqNodeList':'node915','MinMemoryNode':'256M','NumCPUs':'1','TimeLimit':'2-00:10:00'}.items():assert base.field(held,k)==v,(k,base.field(held,k))
 assert 'gres/gpu' not in base.field(held,'ReqTRES')
 assert sha(guard.PLAN)==d['plan_sha256'] and sha(script)==d['script_sha256']
 d.update(held_audit=held,release_requested=True);save(d)
 subprocess.run(['scontrol','release',str(jid)],check=True)
 after=subprocess.check_output(['scontrol','show','job','-dd','-o',str(jid)],text=True)
 assert base.field(after,'JobState') in {'RUNNING','PENDING'} and base.field(after,'Reason') not in {'JobHeldUser','JobHeldAdmin'}
 d.update(released=True,after_release=after);save(d)
 print(json.dumps({'guard_job_id':jid,'target':31048182,'startup_observer':str(observer),'released':True}))

if __name__=='__main__':main()
