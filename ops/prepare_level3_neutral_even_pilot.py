"""Prepare a 32-row, one-draw development diagnostic; never an admission fit."""
from pathlib import Path
import json,sys,shlex,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ops/exp_scaling'),str(ROOT/'ops'),str(ROOT/'src')]
import calibrate_modebench_level3_neutral_v2 as c
import modebench_level3_python_neutral_v5 as g
ART=ROOT/'var/artifacts/modebench_level3_neutral_even_pilot'


def main():
    c.require(not ART.exists(),'fresh pilot namespace required');ART.mkdir()
    reference=c.materializer.reference_rows(c.DOMAIN,'dev')
    selected=sorted(reference,key=lambda r:c.mixture.sha(['common3_pilot_supports_v1',r['answer_mode_count'],r['problem']]))[:32]
    quota=c.materializer.modes(selected)
    blocked,pins=c.historical_inventory();rows=[]
    parent=ROOT/'var/artifacts/modebench_level3_neutral_common3_pilot/rows.jsonl'
    blocked|=c.ids(c.mixture.read_jsonl(parent))
    for tier in [3]:
        generated=g.build_pool(c.DOMAIN,quota,blocked,14037100+1000*tier,'neutral_even_pilot',tier,1)
        rows.extend(generated);blocked|=c.ids(generated)
    source=ART/'rows.jsonl';source.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    output=ART/'receipt.json'
    task={'domain':c.DOMAIN,'level':'level3','split':'dev','interface':c.neutral.INTERFACE,'rows_jsonl':str(source),
          'output':str(output),'seeds':[7027000],'batch_size':8,'row_offset':0,'row_limit':0}
    tasks=ART/'tasks.json';c.new(tasks,[task])
    model=c.read(c.PLAN)['model']['path']
    sources=[Path(__file__).resolve(),Path(g.__file__),g.BASE,source,tasks]
    code={str(p):c.digest(p) for p in sources}
    for name,sha in c.neutral.code_identity().items():code[str(ROOT/name)]=sha
    for name,sha in c.mixture.local_dependency_sources([Path(__file__).resolve(),Path(g.__file__)]).items():code[str(ROOT/name)]=sha
    plan={'schema':'neutral_even_development_pilot_v1','rows':32,'draws':1,'samples_per_draw':8,
          'admission_or_recipe_claimed':False,'histogram_selection_outcome_independent':True,
          'informed_by':'proper-divisor pilot motivates adding an easier all-even numeric stratum',
          'files_sha256':code,'model':c.read(c.PLAN)['model'],'source':str(source),'task':task}
    c.new(ART/'plan.json',plan)
    worker=ART/'worker.py'
    worker.write_text('import sys,json,hashlib\nfrom pathlib import Path\nsys.path[:0]='+repr([str(ROOT/'ops'),str(ROOT/'src')])+'\np=json.loads(Path('+repr(str(ART/'plan.json'))+').read_text())\nfor name,sha in p["files_sha256"].items():\n assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name\nimport evaluate_modebench_level3_neutral as n\nn.evaluator.main('+repr(['--model',model,'--model-label','3b','--tasks-json',str(tasks)])+')\n')
    cmd=['sbatch','--parsable','--partition=all','--qos=normal','--gres=gpu:rtx_6000:1','--cpus-per-task=6','--mem=48G','--time=00:45:00','--exclude=node103','--chdir='+str(ROOT),'--job-name=l3-neutral-even-pilot','--output='+str(ART/'worker-%j.out'),'--error='+str(ART/'worker-%j.err'),'--export=ALL,VLLM_USE_V1=0,VLLM_ATTENTION_BACKEND=XFORMERS,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=4','--wrap=source ops/repo_env.sh; exec '+shlex.join([str(c.PYTHON),'-u','-B',str(worker)])]
    c.new(ART/'submission_intent.json',{'command':cmd,'plan_sha256':c.digest(ART/'plan.json'),'worker_sha256':c.digest(worker)})
    p=subprocess.run(cmd,capture_output=True,text=True)
    c.new(ART/'submission_result.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    c.require(p.returncode==0,'pilot submission failed')
    print(p.stdout,flush=True)

if __name__=='__main__':main()
