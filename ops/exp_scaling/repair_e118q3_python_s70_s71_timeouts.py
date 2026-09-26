#!/usr/bin/env python3
"""Continue four timed-out E118 Qwen-3B Python cells on node302/mltheory."""

import json, re, subprocess
from pathlib import Path
import launch_e118q3_qwen3b_maxrl_extension as launch

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/launch.LEDGER
REPAIR=ROOT/"var/artifacts/e118q3_python_s70_s71_timeout_repair_20260903.json"
EXPECTED={31010902,31010903,31010904,31010905}
PVL="node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"

def state(j):
 r=subprocess.run(["sacct","-n","-X","-j",str(j),"--format=State","-P"],capture_output=True,text=True,check=True)
 return next(x.split("|")[0].strip().split()[0] for x in r.stdout.splitlines() if x.strip())

def main():
 if REPAIR.exists(): raise SystemExit(f"repair exists: {REPAIR}")
 data=json.loads(LEDGER.read_text()); chosen=[x for x in data["runs"] if int(x["job_id"]) in EXPECTED]
 if {int(x["job_id"]) for x in chosen}!=EXPECTED: raise RuntimeError("repair set drifted")
 templates={(str(x["domain"]),int(x["seed"])):x for x in launch.e80.references(ROOT)}
 snap=Path(data["snapshot_root"]); model=launch.e80.model_root(ROOT)
 submitted=[]; replacements=[]
 try:
  for old in chosen:
   oid=int(old["job_id"])
   if state(oid)!="TIMEOUT": raise RuntimeError(f"{oid} is not TIMEOUT")
   ckpt=Path(old["run_dir"])/f"debug_job{oid}/checkpoints/step_01920"
   if not ckpt.is_dir(): raise RuntimeError(f"missing {ckpt}")
   template=templates[(str(old["domain"]),int(old["seed"]))]
   env,target=launch.environment(ROOT,template,str(old["arm"]),snap,model)
   if str(target)!=str(old["run_dir"]): raise RuntimeError("target drift")
   cmd=launch.command(ROOT,template,str(old["arm"]),env)
   mapping={"--partition=all":"--partition=mltheory","--account=allcs":"--account=mltheory",
            f"--nodelist={launch.NODES}":"--nodelist=node302"}
   cmd=[mapping.get(x,x) for x in cmd]; cmd.insert(-1,f"--exclude={PVL}")
   p=subprocess.run(cmd,capture_output=True,text=True)
   if p.returncode: raise RuntimeError(p.stderr.strip())
   nid=int(p.stdout.strip().split(";")[0]); submitted.append(nid)
   held=subprocess.run(["scontrol","show","job","-dd","-o",str(nid)],capture_output=True,text=True,check=True).stdout
   required=("JobState=PENDING","Reason=JobHeldUser","Account=mltheory","Partition=mltheory",
             "ReqNodeList=node302",f"ExcNodeList={PVL}","MinMemoryNode=128G",
             f"RUN_STAMP={old['run_stamp']}",f"SAVE_PATH={old['run_dir']}")
   missing=[x for x in required if x not in held]
   if missing: raise RuntimeError(f"{nid} missing {missing}")
   replacements.append({"old_job_id":oid,"new_job_id":nid,"domain":old["domain"],"arm":old["arm"],
                        "seed":old["seed"],"resume_checkpoint":str(ckpt),"held_scheduler_record":held})
  for rep in replacements:
   run=next(x for x in data["runs"] if int(x["job_id"])==rep["old_job_id"])
   run["previous_job_ids"]=[*run.get("previous_job_ids",[]),rep["old_job_id"]]
   run["job_id"]=rep["new_job_id"]; run["held_scheduler_record"]=rep["held_scheduler_record"]
  audit={"schema":"e118q3_python_timeout_repair_v1","reason":"12-hour allocation timeout",
         "scientific_configuration_changed":False,"placement":"node302/mltheory",
         "pvl_exclusion":PVL,"replacements":replacements,"released":False}
  launch.e80.atomic_json(LEDGER,data); launch.e80.atomic_json(REPAIR,audit)
  for j in submitted: subprocess.run(["scontrol","release",str(j)],check=True)
  audit["released"]=True; launch.e80.atomic_json(REPAIR,audit)
 except Exception:
  if submitted: subprocess.run(["scancel",*map(str,submitted)],check=False)
  raise
 print("repaired",len(replacements),"jobs",",".join(map(str,submitted)))

if __name__=="__main__": main()

