"""Publish the prospective Python L3 default and exact migration inventory.

Does not mutate frozen registrations, release jobs or re-label historical data.
The output binds current training/inference wording to the existing ablation.
"""
from pathlib import Path
import json,subprocess,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ops'))
from followup_metrics import atomic_new,file_sha,sha
from modebench_current_contract import CURRENT,HISTORICAL,profile_metadata,make_messages,training_environment


def main():
    out=ROOT/'artifacts/modebench_level3_neutral_default_20260911';out.mkdir(exist_ok=True)
    source=ROOT/'artifacts/modebench_prompt_ablation_20260911'
    rows=[json.loads(l) for l in (source/'rows.jsonl').read_text().splitlines() if l.strip()]
    rows=[r for r in rows if r['level']==3 and r['domain']=='python_factors'];assert len(rows)==32
    frozen=[json.loads(l) for l in (source/'prompts.jsonl').read_text().splitlines() if l.strip()]
    # Bind the default to exactly the existing neutral strings, not a new rewrite.
    expected={r['row_index']:r for r in frozen if r['level']==3 and r['domain']=='python_factors' and r['arm']=='neutral'}
    messages=[]
    for r in rows:
        m=make_messages(3,'python_factors',r);assert m==expected[r['row_index']]['messages'];messages.append({'row_index':r['row_index'],'row_sha256':sha(r),'messages':m,'messages_sha256':sha(m)})
    campaigns=[];pending=[]
    for exp in ('e122_level3_factorial','e123_level3_factorial','e124_qwen7b_three_level'):
        path=ROOT/'var/artifacts'/(exp+'_jobs.json');ledger=json.loads(path.read_text());runs=[r for r in ledger.get('runs',[]) if r['domain']=='python_factors' and r.get('level',3)==3]
        planned=[r for r in ledger.get('planned_runs',[]) if r.get('domain')=='python_factors']
        ids=[str(r['job_id']) for r in runs]
        history=subprocess.run(['sacct','-X','-j',','.join(ids),'-n','-P','--format=JobID,State,Start,ElapsedRaw,Restarts'],check=True,capture_output=True,text=True).stdout if ids else ''
        histories=[l.split('|')[:5] for l in history.splitlines() if l.strip()]
        assert len(histories)==len(ids) and all(state=='PENDING' and start=='Unknown' and elapsed=='0' and restarts=='0' for _,state,start,elapsed,restarts in histories)
        assert all(not Path(r['run_dir']).exists() for r in runs)
        campaigns.append({'campaign':exp,'ledger':str(path),'ledger_sha256':file_sha(path),'submitted_python_jobs':len(runs),'planned_python_jobs':len(planned),'no_training_evidence':True,'scheduler_history':histories,'registered_jobs_rewritten':False})
        for r in runs:pending.append({'campaign':exp,**{k:r.get(k) for k in ('job_id','level','domain','arm','seed','run_dir','run_stamp')}})
    profile=profile_metadata(3,'python_factors')
    minimal=training_environment(3,'python_factors',{'OAT_ZERO_PROMPT_TEMPLATE':'qwen_level2_python_factors'})
    registration={'schema':'modebench-python-level3-neutral-default-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'default_condition':CURRENT,'historical_condition':HISTORICAL,'scope':{'level':3,'domains':['python_factors'],'training_and_inference':True},'profile':profile,'new_training_environment':minimal,'system_prompt':messages[0]['messages'][0]['content'],'new_run_namespace_suffix':'python_level3_neutral_v1','selection_provenance':'User chose this interface after seeing the original/neutral inference and coarse-key comparisons. It is prospective for training but not a blind choice on the existing evaluation set.','difficulty_calibration':'Existing matched-difficulty evidence used registered hints. It is historical provenance, not a calibration certificate for the neutral prompt.','outcome_identity':'Original divisor-vector keys remain primary; unordered factor-pair keys remain a required sensitivity. Neither identifies algorithms.','historical_results_policy':'Preserve original wording results and the adverse coarsening finding; do not relabel them as neutral.','campaign_audit':campaigns,'existing_jobs_pending_version_migration':pending,'new_jobs_submitted':0,'source_sha256':{str(p):file_sha(p) for p in [ROOT/'src/oat_drgrpo/templates.py',ROOT/'ops/modebench_current_contract.py',Path(__file__),ROOT/'tests/test_modebench_current_contract.py',source/'prompts.jsonl',source/'rows.jsonl']},'exact_existing_neutral_prompts':messages}
    atomic_new(out/'registration.json',registration)
    atomic_new(out/'default_training_environment.json',minimal)
    (out/'README.md').write_text('# Python Level 3 neutral default\n\nThe current interface for new training and inference is `ops/modebench_current_contract.py`. Its default is `python_level3_neutral_v1`; the training template is `qwen_level3_python_factors_neutral_v1` (also exposed as `qwen_level3_python_factors`). Use `registered_hints_v1` explicitly for the historical condition.\n\nThe 32 saved neutral prompt pairs match byte for byte. The same renderer feeds new training and inference; answer format, executable tasks, verifier and syntax constraints are preserved.\n\nThe registered E122/E124 held jobs still contain the historical interface. This registration inventories their exact IDs and zero-training evidence; it does not claim that those frozen jobs have been migrated. The E123 Python jobs remain prospective. New prompt-version runs need a distinct result namespace and regenerated snapshots/commands; resuming an old run with new wording is not supported.\n\nThe prompt was chosen after inspecting existing evaluation results. Retain both conditions and coarse-key sensitivity in the paper. Original difficulty matching does not transfer automatically to the new wording.\n')
    print(json.dumps({'status':'registered','path':str(out/'registration.json'),'exact_neutral_prompt_pairs':32,'old_held_jobs':len(pending),'new_jobs_submitted':0}))

if __name__=='__main__':main()
