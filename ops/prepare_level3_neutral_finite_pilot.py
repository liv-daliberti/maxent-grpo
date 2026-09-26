"""Matched-case pilot clarifying the existing finite-input evaluation scope."""
from pathlib import Path
import json,sys,shlex,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import calibrate_modebench_level3_neutral_v2 as c
ART=ROOT/'var/artifacts/modebench_level3_neutral_finite_pilot'
PARENT=ROOT/'var/artifacts/modebench_level3_neutral_common3_pilot'
CLARIFICATION='Only the four listed inputs are evaluated; behavior on other inputs is unconstrained. '


def finite_prompt(text):
    anchor='For every one of these inputs, return an integer d'
    c.require(text.count(anchor)==1,'expected finite constraint site changed')
    return text.replace(anchor,CLARIFICATION+anchor)


def main():
    c.require(not ART.exists(),'fresh diagnostic namespace required')
    c.require((PARENT/'receipt.json').exists(),'paired parent diagnostic must be complete')
    ART.mkdir();rows=c.mixture.read_jsonl(PARENT/'rows.jsonl')
    for r in rows:
        r['problem']=finite_prompt(r['problem']);r['level3_task_wording']='neutral_finite_inputs_v4'
    source=ART/'rows.jsonl';source.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    task={'domain':c.DOMAIN,'level':'level3','split':'dev','interface':c.neutral.INTERFACE,'rows_jsonl':str(source),
          'output':str(ART/'receipt.json'),'seeds':[6827000],'batch_size':8,'row_offset':0,'row_limit':0}
    tasks=ART/'tasks.json';c.new(tasks,[task]);model=c.read(c.PLAN)['model']['path']
    files={str(p):c.digest(p) for p in [Path(__file__).resolve(),source,tasks,PARENT/'rows.jsonl',PARENT/'receipt.json']}
    for name,sha in c.neutral.code_identity().items():files[str(ROOT/name)]=sha
    c.new(ART/'plan.json',{'schema':'neutral_finite_scope_paired_development_pilot_v1','model':c.read(c.PLAN)['model'],
            'task':task,'files_sha256':files,'parent_diagnostic':str(PARENT),'added_sentence':CLARIFICATION,
            'admission_claimed':False,'case_sets_unchanged':True,'valid_program_space_unchanged':True,
            'solution_example_or_algorithm_hint_added':False,'training_started':False})
    worker=ART/'worker.py'
    worker.write_text('import sys,json,hashlib\nfrom pathlib import Path\nsys.path[:0]='+repr([str(ROOT/'ops'),str(ROOT/'src')])+'\np=json.loads(Path('+repr(str(ART/'plan.json'))+').read_text())\nfor name,sha in p["files_sha256"].items():\n assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name\nimport evaluate_modebench_level3_neutral as n\nn.evaluator.main('+repr(['--model',model,'--model-label','3b','--tasks-json',str(tasks)])+')\n')
    cmd=['sbatch','--parsable','--partition=all','--qos=normal','--gres=gpu:rtx_6000:1','--cpus-per-task=6','--mem=48G','--time=00:45:00','--exclude=node103','--chdir='+str(ROOT),'--job-name=l3-neutral-finite-pilot','--output='+str(ART/'worker-%j.out'),'--error='+str(ART/'worker-%j.err'),'--export=ALL,VLLM_USE_V1=0,VLLM_ATTENTION_BACKEND=XFORMERS,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4','--wrap=source ops/repo_env.sh; exec '+shlex.join([str(c.PYTHON),'-u','-B',str(worker)])]
    c.new(ART/'submission_intent.json',{'command':cmd,'worker_sha256':c.digest(worker),'plan_sha256':c.digest(ART/'plan.json')})
    p=subprocess.run(cmd,capture_output=True,text=True);c.new(ART/'submission_result.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    c.require(p.returncode==0,'pilot submission failed');print(p.stdout,flush=True)

if __name__=='__main__':main()
