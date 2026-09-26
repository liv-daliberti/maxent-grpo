#!/usr/bin/env python3
"""Continue E119 Countdown Dr.GRPO seed 45 after an evaluation watchdog exit."""

import hashlib, json, subprocess
from pathlib import Path
import launch_e119_level2_qwen05b_factorial as launch

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/launch.LEDGER
CONT=ROOT/"var/artifacts/e119_level2_continuation_jobs.json"
REPAIR=ROOT/"var/artifacts/e119_countdown_drgrpo_s45_watchdog_repair_20260903.json"
OLD=31014406
PVL="node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"

def main():
 if REPAIR.exists(): raise SystemExit(f"repair exists: {REPAIR}")
 ledger=json.loads(LEDGER.read_text()); old=next(x for x in ledger["runs"] if int(x["job_id"])==OLD)
 state=subprocess.run(["sacct","-n","-X","-j",str(OLD),"--format=State","-P"],capture_output=True,text=True,check=True).stdout.strip().split("|")[0].split()[0]
 if state!="FAILED": raise RuntimeError(f"{OLD} is {state}")
 ckpt=Path(old["run_dir"])/f"debug_job{OLD}/checkpoints/step_00192"
 if not ckpt.is_dir(): raise RuntimeError(f"missing {ckpt}")
 template=next(x for x in launch.templates(ROOT) if x["domain"]==old["domain"] and int(x["seed"])==int(old["seed"]))
 env,target=launch.environment(ROOT,template,str(old["arm"]),Path(ledger["snapshot_root"]))
 if str(target)!=str(old["run_dir"]): raise RuntimeError("target drift")
 env.update({"OAT_ZERO_WATCHDOG_STALE_SECONDS":"7200","OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS":"3600",
             "OAT_ZERO_WATCHDOG_MAX_RESTARTS":"12"})
 cmd=launch.command(ROOT,template,str(old["arm"]),env)
 mapping={"--partition=mltheory":"--partition=all","--account=mltheory":"--account=allcs",
          "--nodelist=node105":"--nodelist=node203,node204,node205,node207",
          "--gres=gpu:a5000:1":"--gres=gpu:1","--mem=64G":"--mem=40G"}
 cmd=[mapping.get(x,x) for x in cmd]; cmd.insert(-1,f"--exclude={PVL}")
 submitted=[]
 try:
  p=subprocess.run(cmd,capture_output=True,text=True)
  if p.returncode: raise RuntimeError(p.stderr.strip())
  nid=int(p.stdout.strip().split(";")[0]); submitted.append(nid)
  held=subprocess.run(["scontrol","show","job","-dd","-o",str(nid)],capture_output=True,text=True,check=True).stdout
  required=("JobState=PENDING","Reason=JobHeldUser","Account=allcs",f"ExcNodeList={PVL}",
            "OAT_ZERO_WATCHDOG_STALE_SECONDS=7200",f"RUN_STAMP={old['run_stamp']}")
  missing=[x for x in required if x not in held]
  if missing: raise RuntimeError(f"{nid} missing {missing}")
  payload=json.loads(CONT.read_text())
  if payload["original_ledger_sha256"]!=hashlib.sha256(LEDGER.read_bytes()).hexdigest(): raise RuntimeError("ledger hash drift")
  if any(int(x["original_job_id"])==OLD for x in payload["continuations"]): raise RuntimeError("duplicate continuation")
  record={"original_job_id":OLD,"continuation_job_id":nid,"domain":old["domain"],"arm":old["arm"],
          "seed":old["seed"],"run_dir":old["run_dir"],"run_stamp":old["run_stamp"],
          "repair_kind":"evaluation_watchdog_continuation"}
  payload["continuations"].append(record); payload["released"]=False
  audit={"schema":"e119_countdown_watchdog_repair_v1","scientific_configuration_changed":False,
         "resume_checkpoint":str(ckpt),"pvl_exclusion":PVL,"replacement":record,"released":False}
  launch.e78.atomic_json(CONT,payload); launch.e78.atomic_json(REPAIR,audit)
  subprocess.run(["scontrol","release",str(nid)],check=True)
  payload["released"]=True; audit["released"]=True
  launch.e78.atomic_json(CONT,payload); launch.e78.atomic_json(REPAIR,audit)
 except Exception:
  if submitted: subprocess.run(["scancel",*map(str,submitted)],check=False)
  raise
 print("repaired 1 job",submitted[0])

if __name__=="__main__": main()

