#!/usr/bin/env python3
"""Transactionally recover E116 Qwen-3B after its signal-53 zero-step failure."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parent))
import direct_comparator_completion as shared  # noqa: E402

ROOT=shared.ROOT
LEDGER=ROOT/"var/artifacts/e116_sparse_rlep_qwen3b_jobs.json"
RECORD=ROOT/"var/artifacts/e116r1_qwen3b_signal53_zero_step_recovery.json"
PROTOCOL=ROOT/"paper/preregistration/e116r1_qwen3b_signal53_zero_step_recovery_20260831.md"
ORIGINAL_SMOKE=30790466
ORIGINAL_AUDIT=30790467
ORIGINAL_SCIENCE=tuple(range(30790468,30790493))
BLOCKED_AUDITS={30790437,30790439,30790443}
STAGED:list[str]=[]

def command(argv:list[str])->str:
 r=subprocess.run(argv,text=True,capture_output=True,check=False)
 if r.returncode: raise RuntimeError(f"command failed: {shlex.join(argv)}\n{r.stderr.strip()}")
 return r.stdout.strip()

def frozen_submit(record:str)->list[str]:
 if "SubmitLine=" not in record or " WorkDir=" not in record: raise RuntimeError("unbounded frozen SubmitLine")
 argv=shlex.split(record.split("SubmitLine=",1)[1].split(" WorkDir=",1)[0])
 if not argv or argv[0]!="sbatch": raise RuntimeError("frozen command is not sbatch")
 if "--wrap" in argv:
  index=argv.index("--wrap")
  argv=argv[:index+1]+[shlex.join(argv[index+1:])]
 return argv

def dependency(argv:list[str],value:str)->list[str]:
 out=[x for x in argv if not x.startswith("--dependency=")]
 if "--hold" not in out: out.insert(2,"--hold")
 if value:
  index=next((i for i,x in enumerate(out) if x.startswith("/") and (x.endswith(".slurm") or x.endswith("/python"))),len(out))
  out.insert(index,f"--dependency=afterok:{value}")
 return out

def accounting(ids:list[int]|tuple[int,...])->dict[int,dict[str,str]]:
 fields="JobIDRaw,State,ExitCode,Elapsed,Start,NodeList"
 raw=command(["sacct","-n","-P","-X","-j",",".join(map(str,ids)),f"--format={fields}"])
 names=fields.split(","); rows={}
 for line in raw.splitlines():
  values=line.split("|")
  if len(values)==len(names) and values[0].isdigit(): rows[int(values[0])]=dict(zip(names,values))
 return rows

def validate(p:dict[str,Any])->tuple[dict[int,dict[str,str]],dict[int,dict[str,str]]]:
 if p.get("schema")!="e116_sparse_rlep_completion_jobs_v1" or p.get("cohort")!="e116_qwen3b" or p.get("released") is not True: raise RuntimeError("ledger identity drifted")
 if int(p["smoke"]["job_id"])!=ORIGINAL_SMOKE or int(p["smoke"]["audit_job_id"])!=ORIGINAL_AUDIT: raise RuntimeError("smoke identity drifted")
 if tuple(int(x["job_id"]) for x in p["runs"])!=ORIGINAL_SCIENCE: raise RuntimeError("science identity drifted")
 paths=[Path(p["smoke"]["run_dir"]),*[Path(x["run_dir"]) for x in p["runs"]]]
 if any(x.exists() for x in paths): raise RuntimeError("zero-step output boundary drifted")
 boundary=accounting([ORIGINAL_SMOKE,ORIGINAL_AUDIT,*ORIGINAL_SCIENCE])
 s=boundary[ORIGINAL_SMOKE]
 if (s["State"],s["ExitCode"],s["Elapsed"])!=("FAILED","0:53","00:00:00"): raise RuntimeError(f"smoke boundary drifted: {s}")
 for jid in (ORIGINAL_AUDIT,*ORIGINAL_SCIENCE):
  row=boundary[jid]
  if not row["State"].startswith("CANCELLED") or row["Elapsed"]!="00:00:00" or row["Start"] not in {"","None","Unknown"}: raise RuntimeError(f"dependent boundary drifted: {jid}: {row}")
 records=p["collection"]["records"]
 ids=[int(x[k]) for x in records for k in ("collection_job_id","audit_job_id")]
 pools=accounting(ids)
 for x in records:
  cid,aid=int(x["collection_job_id"]),int(x["audit_job_id"]); c,a=pools[cid],pools[aid]
  if aid in BLOCKED_AUDITS:
   if c["State"]!="COMPLETED" or a["State"]!="FAILED": raise RuntimeError(f"blocked pool drifted: {cid}/{aid}")
  elif c["State"]=="COMPLETED":
   if a["State"]!="COMPLETED" or not Path(x["pool_root"]).is_dir(): raise RuntimeError(f"valid pool drifted: {cid}/{aid}")
  elif not c["State"].startswith("CANCELLED") or c["Elapsed"]!="00:00:00" or Path(x["pool_root"]).exists(): raise RuntimeError(f"retry pool drifted: {cid}: {c}")
 return boundary,pools

def held(argv:list[str],name:str,expected:tuple[str,...])->tuple[int,str]:
 jid=shared.submit_held(argv)
 STAGED.append(jid)
 return int(jid),shared.audit_held(jid,name=name,expected=expected)

def job_name(argv:list[str])->str:
 return next(x.split("=",1)[1] for x in argv if x.startswith("--job-name="))

def main()->int:
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--recover",action="store_true"); args=ap.parse_args()
 if not LEDGER.is_file() or not PROTOCOL.is_file(): raise SystemExit("missing ledger or prospective protocol")
 if RECORD.exists(): raise SystemExit(f"recovery already exists: {RECORD}")
 original=json.loads(LEDGER.read_text()); boundary,pool_boundary=validate(original)
 if not args.recover:
  print("[e116r1] zero-step boundary valid")
  print("[e116r1] pools: 14 valid/re-audit, 8 zero-step/retry, 3 hard-gate blocked")
  print("[e116r1] ready: 54 held jobs, 22 feasible science cells")
  return 0
 submitted=STAGED; updated=deepcopy(original)
 try:
  audit_by_key={}; pool_replacements=[]
  for old,new in zip(original["collection"]["records"],updated["collection"]["records"]):
   key=(str(old["domain"]),int(old["seed"])); old_aid=int(old["audit_job_id"]); old_cid=int(old["collection_job_id"])
   if old_aid in BLOCKED_AUDITS:
    new.update({"hard_gate_blocked":True,"blocked_because":"frozen pool has zero replay-eligible prompts"}); continue
   replacement_cid=None; audit_dep=""
   if pool_boundary[old_cid]["State"]!="COMPLETED":
    argv=dependency(frozen_submit(old["held_collection_scheduler_record"]),"")
    cid,record=held(argv,job_name(argv),("Reason=JobHeldUser","ReqNodeList=node302","OAT_ZERO_EVAL_ONLY=1",f"SAVE_PATH={old['pool_root']}","OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4"))
    replacement_cid=cid; audit_dep=str(cid)
    new.update({"replaces_collection_job_id":old_cid,"collection_job_id":cid,"held_collection_scheduler_record":record})
   argv=dependency(frozen_submit(old["held_audit_scheduler_record"]),audit_dep)
   expected=["Reason=JobHeldUser",str(old["pool_root"]),"--allow-sparse"]
   if audit_dep: expected.append(f"Dependency=afterok:{audit_dep}")
   aid,record=held(argv,job_name(argv),tuple(expected)); audit_by_key[key]=aid
   new.update({"replaces_audit_job_id":old_aid,"audit_job_id":aid,"held_audit_scheduler_record":record})
   pool_replacements.append({"domain":key[0],"seed":key[1],"original_collection_job_id":old_cid,"replacement_collection_job_id":replacement_cid,"original_audit_job_id":old_aid,"replacement_audit_job_id":aid,"reused_immutable_pool":replacement_cid is None})
  smoke_pool=audit_by_key[("graph_coloring",70)]
  argv=dependency(frozen_submit(original["smoke"]["held_scheduler_record"]),str(smoke_pool))
  smoke_id,smoke_record=held(argv,"e116-qwen3b-smoke",("Reason=JobHeldUser","ReqNodeList=node302",f"Dependency=afterok:{smoke_pool}","OAT_ZERO_MAX_TRAIN=32","OAT_ZERO_RLEP_SPARSE_FALLBACK=1"))
  argv=dependency(frozen_submit(original["smoke"]["held_audit_scheduler_record"]),str(smoke_id))
  smoke_audit_id,smoke_audit_record=held(argv,"e116-qwen3b-smoke-audit",(f"Dependency=afterok:{smoke_id}","--expected-terminal-step"," 32"))
  feasible=[]; blocked=[]; mappings=[]
  for old in original["runs"]:
   key=(str(old["domain"]),int(old["seed"])); old_aid=int(old["pool_audit_dependency_job_id"])
   if old_aid in BLOCKED_AUDITS:
    row=deepcopy(old); row.update({"blocked_because":"frozen sparse-RLEP pool has zero replay-eligible prompts; preregistered hard gate failed","unstarted_cancelled_job_id":int(old["job_id"])}); blocked.append(row); continue
   pool_id=audit_by_key[key]; argv=dependency(frozen_submit(old["held_scheduler_record"]),f"{pool_id}:{smoke_audit_id}")
   jid,record=held(argv,job_name(argv),(f"afterok:{pool_id}",f"afterok:{smoke_audit_id}","ReqNodeList=node302",f"OAT_ZERO_SEED={old['seed']}","OAT_ZERO_VARIANT=rlep","OAT_ZERO_RLEP_REPLAY_COUNT=2","OAT_ZERO_RLEP_SPARSE_FALLBACK=1"))
   row=deepcopy(old); row.update({"replaces_job_id":int(old["job_id"]),"job_id":jid,"pool_audit_dependency_job_id":pool_id,"smoke_audit_dependency_job_id":smoke_audit_id,"held_scheduler_record":record}); feasible.append(row)
   mappings.append({"original_job_id":int(old["job_id"]),"replacement_job_id":jid})
  updated["smoke"].update({"replaces_job_id":ORIGINAL_SMOKE,"replaces_audit_job_id":ORIGINAL_AUDIT,"job_id":smoke_id,"audit_job_id":smoke_audit_id,"pool_audit_dependency_job_id":smoke_pool,"held_scheduler_record":smoke_record,"held_audit_scheduler_record":smoke_audit_record})
  updated["runs"]=feasible; updated["blocked_runs"]=blocked
  now=datetime.now(timezone.utc).isoformat()
  recovery={"schema":"e116r1_qwen3b_signal53_zero_step_recovery_v1","created_at":now,"released":False,"protocol":str(PROTOCOL),"protocol_sha256":shared.digest(PROTOCOL),"application":str(Path(__file__).resolve()),"application_sha256":shared.digest(Path(__file__)),"original_ledger_sha256":shared.digest(LEDGER),"original_smoke_job_id":ORIGINAL_SMOKE,"original_smoke_audit_job_id":ORIGINAL_AUDIT,"replacement_smoke_job_id":smoke_id,"replacement_smoke_audit_job_id":smoke_audit_id,"science_job_mappings":mappings,"pool_replacements":pool_replacements,"blocked_cells":[{"domain":x["domain"],"seed":x["seed"],"failed_pool_audit_job_id":x["pool_audit_dependency_job_id"]} for x in blocked],"zero_step_accounting":boundary,"pool_boundary_accounting":pool_boundary,"scientific_settings_changed":False,"orchestration_changes":["fresh_job_ids","fresh_cpu_reaudits","eight_zero_runtime_collection_retries","replacement_dependencies"]}
  updated["e116r1_zero_step_recovery"]=recovery; updated["released"]=False
  shared.atomic_json(LEDGER,updated); shared.atomic_json(RECORD,recovery)
  shared.release(submitted)
  recovery.update({"released":True,"released_at":datetime.now(timezone.utc).isoformat()}); updated.update({"released":True,"e116r1_zero_step_recovery":recovery})
  shared.atomic_json(LEDGER,updated); shared.atomic_json(RECORD,recovery)
 except Exception:
  shared.cancel(submitted)
  if submitted: shared.atomic_json(LEDGER,original)
  raise
 print(f"[e116r1] released {len(submitted)} jobs: smoke={smoke_id}, audit={smoke_audit_id}, science={len(feasible)}, blocked={len(blocked)}")
 print(f"[e116r1] recovery record: {RECORD}")
 return 0
if __name__=="__main__": raise SystemExit(main())
