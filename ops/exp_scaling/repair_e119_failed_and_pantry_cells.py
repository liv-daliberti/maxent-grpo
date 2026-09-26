#!/usr/bin/env python3
"""Repair E119 watchdog failures and invalid Level-2 Pantry plumbing."""

import hashlib, json, subprocess
from pathlib import Path
import launch_e119_level2_qwen05b_factorial as launch

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/launch.LEDGER
CONT=ROOT/"var/artifacts/e119_level2_continuation_jobs.json"
REPAIR=ROOT/"var/artifacts/e119_level2_failed_and_pantry_repair_20260903.json"
PVL="node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
TIMEOUT={31014405,31014408,31014410,31014450}
PANTRY_FAILED={31014470,31014471,31014472,31014473,31014474,31014475,31014476,31014477}

def state(j):
 r=subprocess.run(["sacct","-n","-X","-j",str(j),"--format=State","-P"],capture_output=True,text=True,check=True)
 rows=[x.split("|")[0].strip().split()[0] for x in r.stdout.splitlines() if x.strip()]
 return rows[0] if rows else "UNKNOWN"

def command(parts):
 repl={"--partition=mltheory":"--partition=all","--account=mltheory":"--account=allcs",
       "--nodelist=node105":"--nodelist=node203,node204,node205,node207",
       "--gres=gpu:a5000:1":"--gres=gpu:1","--mem=64G":"--mem=40G"}
 out=[repl.get(x,x) for x in parts]
 out.insert(-1,f"--exclude={PVL}")
 return out

def main():
 if REPAIR.exists(): raise SystemExit(f"repair already exists: {REPAIR}")
 ledger=json.loads(LEDGER.read_text())
 snap=Path(ledger["snapshot_root"])
 templates={(str(x["domain"]),int(x["seed"])):x for x in launch.templates(ROOT)}
 byid={int(x["job_id"]):x for x in ledger["runs"]}
 pantry=[x for x in ledger["runs"] if x["domain"]=="pantry_plan"]
 selected=pantry+[byid[x] for x in sorted(TIMEOUT)]
 if len(pantry)!=20 or len(selected)!=24: raise RuntimeError("repair set drifted")
 if {int(x["job_id"]) for x in pantry if state(int(x["job_id"]))=="FAILED"}!=PANTRY_FAILED:
  raise RuntimeError("Pantry failure set drifted")
 if any(state(x)!="FAILED" for x in TIMEOUT): raise RuntimeError("watchdog state drifted")
 old=json.loads(CONT.read_text())
 existing=list(old["continuations"])
 if {int(x["original_job_id"]) for x in existing}!={31014401}: raise RuntimeError("lineage drifted")
 submitted=[]; records=[]
 try:
  for row in selected:
   template=templates[(str(row["domain"]),int(row["seed"]))]
   env,target=launch.environment(ROOT,template,str(row["arm"]),snap)
   if str(target)!=str(row["run_dir"]): raise RuntimeError("run directory drifted")
   env.update({"OAT_ZERO_WATCHDOG_STALE_SECONDS":"7200",
               "OAT_ZERO_WATCHDOG_STARTUP_GRACE_SECONDS":"3600",
               "OAT_ZERO_WATCHDOG_MAX_RESTARTS":"12"})
   kind="watchdog_continuation"
   if row["domain"]=="pantry_plan":
    env.update({"OAT_ZERO_CANONICAL_ACTION_TASK":"none",
                "OAT_ZERO_CANONICAL_GRAPH_ACTIONS":"0",
                "OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT":"3",
                "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING":"0",
                "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING":"0"})
    kind="level2_pantry_policy_exclusivity"
   p=subprocess.run(command(launch.command(ROOT,template,str(row["arm"]),env)),
                    capture_output=True,text=True)
   if p.returncode: raise RuntimeError(p.stderr.strip())
   new=int(p.stdout.strip().split(";")[0]); submitted.append(new)
   held=subprocess.run(["scontrol","show","job","-dd","-o",str(new)],
                       capture_output=True,text=True,check=True).stdout
   required=["JobState=PENDING","Reason=JobHeldUser","Account=allcs",
             f"ExcNodeList={PVL}",
             "OAT_ZERO_WATCHDOG_STALE_SECONDS=7200",f"RUN_STAMP={row['run_stamp']}"]
   if row["domain"]=="pantry_plan":
    required+=["OAT_ZERO_CANONICAL_ACTION_TASK=none",
               "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=0",
               "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=0"]
   missing=[x for x in required if x not in held]
   if not any(x in held for x in (
       "ReqNodeList=node203,node204,node205,node207",
       "ReqNodeList=node[203-205,207]",
   )):
    missing.append("safe ReqNodeList")
   if missing: raise RuntimeError(f"{new} missing {missing}")
   records.append({"original_job_id":int(row["job_id"]),"continuation_job_id":new,
    "domain":row["domain"],"arm":row["arm"],"seed":int(row["seed"]),
    "run_dir":row["run_dir"],"run_stamp":row["run_stamp"],"repair_kind":kind})
  pending=[int(x["job_id"]) for x in pantry if state(int(x["job_id"]))=="PENDING"]
  if len(pending)!=12: raise RuntimeError(f"pending Pantry set drifted: {pending}")
  subprocess.run(["scancel",*map(str,pending)],check=True)
  base={"schema":"e119_level2_continuation_jobs_v1","original_ledger":str(LEDGER),
   "original_ledger_sha256":hashlib.sha256(LEDGER.read_bytes()).hexdigest(),
   "same_scientific_cells":True,"same_run_directories":True,
   "optimizer_update_changed":False,"treatment_changed":False,
   "released":False,"installed":True,"outcomes_inspected":False,
   "operational_change":"evaluation-safe watchdog and Level-2 Pantry policy-exclusivity repair",
   "continuations":existing+records}
  audit={"schema":"e119_failed_and_pantry_repair_v1","failed_cells_restarted":12,
   "invalid_pending_cells_superseded":12,"scientific_cells_changed":False,
   "outcomes_inspected":False,"pvl_exclusion":PVL,"replacements":records,"released":False}
  launch.e78.atomic_json(CONT,base); launch.e78.atomic_json(REPAIR,audit)
  for j in submitted: subprocess.run(["scontrol","release",str(j)],check=True)
  base["released"]=True; audit["released"]=True
  launch.e78.atomic_json(CONT,base); launch.e78.atomic_json(REPAIR,audit)
 except Exception:
  if submitted: subprocess.run(["scancel",*map(str,submitted)],check=False)
  raise
 print(f"repaired={len(records)} failed=12 superseded_pending=12 jobs={','.join(map(str,submitted))}")

if __name__=="__main__": main()
