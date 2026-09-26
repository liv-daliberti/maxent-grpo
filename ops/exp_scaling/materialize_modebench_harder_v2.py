#!/usr/bin/env python3
"""Build support-matched harder-v2 ModeBench train/dev/evaluation data."""

from __future__ import annotations
from collections import Counter
import hashlib, importlib.util, itertools, json, random, shutil, sys, tempfile
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from datasets import Dataset, DatasetDict, load_from_disk

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/"ops",ROOT/"src",ROOT/"ops/exp_scaling"):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from make_exact_countdown_mode_data import _synthetic_countdown_rows
from make_exact_answer_mode_data import _synthetic_graph_rows
from make_python_factor_mode_data import _candidate_values,_row as _python_row
from make_mathir_action_menu_data import _build_rows as _mathir_rows,_family_support
from oat_drgrpo.python_modebench import python_factor_mode_count
from make_pantry_plan_mode_data import FAMILY_POOLS,_build_row as _pantry_row,_canonical_sha256
from materialize_e117_evaluation_reserves import _identities,_load_source_rows

SCHEMA="modebench_harder_v2_support_matched_splits"
OUTPUT_ROOT=ROOT/"var/data/modebench_harder_v2_matched"
DOMAINS=("countdown","graph_coloring","python_factors","mathir","pantry")
SEEDS={"countdown":4217100,"graph_coloring":4217200,"python_factors":4217300,"mathir":4217400,"pantry":4217500}
SPLITS={"train":(384,"train"),"dev":(128,"multi_answer"),"eval":(128,"multi_answer")}
SPLIT_SEED_OFFSETS={"train":0,"dev":10_000,"eval":20_000}
RESERVE=ROOT/"var/data/e117_evaluation_reserve_v1"
LEVEL1={
"countdown":{"train":ROOT/"var/data/exact_countdown_easy3_probe/train","dev":RESERVE/"development/countdown/eval","eval":RESERVE/"confirmation/countdown/eval"},
"graph_coloring":{"train":ROOT/"var/data/graph_coloring_modebench_v2/train","dev":RESERVE/"development/graph_coloring/eval","eval":RESERVE/"confirmation/graph_coloring/eval"},
"python_factors":{"train":ROOT/"var/data/python_factor_modebench_v1/train","dev":RESERVE/"development/python_factors/eval","eval":RESERVE/"confirmation/python_factors/eval"},
"mathir":{"train":ROOT/"var/data/mathir_action_menu_v1/train","dev":RESERVE/"development/mathir/eval","eval":RESERVE/"confirmation/mathir/eval"},
"pantry":{split:ROOT/f"var/data/pantry_plan_modebench_v2/{split}" for split in SPLITS},
}

def load_rows(path:Path,split:str)->list[dict[str,Any]]:
    return [dict(x) for x in load_from_disk(str(path))[split]]

def modes(rows): return Counter(int(x["answer_mode_count"]) for x in rows)
def row_hash(rows):
    text="\n".join(json.dumps(x,sort_keys=True,separators=(",",":")) for x in rows)
    return hashlib.sha256(text.encode()).hexdigest()
def take_hist(pool,target):
    got=Counter(); out=[]
    for row in pool:
        m=int(row["answer_mode_count"])
        if got[m]<target[m]: out.append(row); got[m]+=1
        if got==target: return out
    raise RuntimeError(f"candidate pool misses support cells: {target-got}")

def existing_ids(domain):
    if domain!="pantry":
        ids=_identities(domain,_load_source_rows(domain))
        reserve=ROOT/"var/data/e117_evaluation_reserve_v1"
        for block in ("development","confirmation"):
            ids|=_identities(domain,load_rows(reserve/block/domain/"eval","multi_answer"))
        return ids
    root=ROOT/"var/data/pantry_plan_modebench_v2"; ids=set()
    for part,split in (("train","train"),("dev","multi_answer"),("eval","multi_answer")):
        ids|={("pantry",str(x["instance_fingerprint"])) for x in load_rows(root/part,split)}
    return ids

def build_countdown(target,seed,tag,excluded):
    pool=_synthetic_countdown_rows(max(12000,sum(target.values())*24),seed=seed,split_tag=tag,number_count=4,max_value=14,min_modes=2,max_modes=64,exclude={x[1:] for x in excluded})
    def easier(row):
        spec=json.loads(row['answer']); numbers=tuple(int(x) for x in spec['numbers'])
        target_value=int(spec['target'])
        one_multiply=set(); paired=set()
        for values in set(itertools.permutations(numbers)):
            a,b,c,d=values
            one_multiply.update((a*b+c+d,a*b+c-d,a*b-c+d,a*b-c-d))
            paired.update((a*b+c*d,a*b-c*d,(a+b)*(c+d),(a+b)*(c-d)))
        family_rank=0 if target_value in paired else 1 if target_value in one_multiply else 2
        return (family_rank,abs(target_value-sum(numbers)),max(numbers),sum(numbers),abs(target_value),numbers)
    return take_hist(sorted(pool,key=easier),target)

def build_graph(target,seed,tag,excluded):
    pool=_synthetic_graph_rows(max(5000,sum(target.values())*12),seed=seed,split_tag=tag,hidden_count=4,min_completions=4,max_completions=32,min_solutions=4,max_n=6,max_edges=10,prompt_style="original",balance_hidden_color=False,exclude={x[1:] for x in excluded})
    return take_hist(pool,target)

def build_python(target,excluded,seed,tag):
    rng=random.Random(seed); values=_candidate_values(192)
    blocked={x[1] for x in excluded}; seen=set(); got=Counter(); rows=[]; attempts=0
    while got!=target:
        attempts+=1
        if attempts>1000000: raise RuntimeError(f"Python pool misses {target-got}")
        cases=tuple(sorted(rng.sample(values,4)))
        if max(cases)<=96 or cases in blocked or cases in seen: continue
        m=python_factor_mode_count(cases)
        if got[m]>=target[m]: continue
        rows.append(_python_row(cases=cases,split_tag=tag,seed=seed,index=len(rows)))
        seen.add(cases); got[m]+=1
    rng.shuffle(rows); return rows

def build_mathir(target,excluded,seed,tag):
    raw={(x[1],x[2]) for x in excluded}
    pool=_mathir_rows(max(1024,sum(target.values())*8),seed=seed,split_tag=tag,family_support=_family_support(),excluded=raw)
    pool=[x for x in pool if x["mathir_family"] in {"ax_plus_b_eq_dx_plus_c","ax_plus_b_eq_c_minus_dx"}]
    def harder(row):
        bindings={k:int(v) for k,v in json.loads(row['answer'])['bindings'].items()}
        magnitude=sum(abs(bindings[k]) for k in sorted(bindings))
        coefficient_symmetry=abs(abs(bindings['a'])-abs(bindings['d']))
        combined_coefficient=(abs(bindings['a']-bindings['d'])
                              if row['mathir_family']=='ax_plus_b_eq_dx_plus_c'
                              else abs(bindings['a']+bindings['d']))
        return (coefficient_symmetry,combined_coefficient,-magnitude,str(row['problem']))
    return take_hist(sorted(pool,key=harder),target)

def pantry_pool(count,excluded,seed,tag):
    curation=json.loads((ROOT/"var/data/pantry_plan_v1/ingredients.json").read_text())
    source={x["id"]:x for x in curation["ingredients"]}; blocked={x[1] for x in excluded}
    rng=random.Random(seed); rows=[]; attempts=0
    while len(rows)<count:
        attempts+=1
        family=rng.choice(tuple(FAMILY_POOLS))
        row=_pantry_row(family=family,split=tag,seed=seed,index=attempts,rng=rng,source_by_id=source,ingredient_table_sha256=curation["ingredient_table_sha256"])
        if row is None: continue
        spec=json.loads(row["answer"]); fp_spec=dict(spec); fp_spec.pop("instance_id",None)
        fingerprint=_canonical_sha256(fp_spec)
        if fingerprint in blocked: continue
        row["instance_fingerprint"]=fingerprint; blocked.add(fingerprint); rows.append(row)
        if len(rows)%250==0: print(f"[level2] {tag} Pantry candidates {len(rows)}/{count}",flush=True)
    return rows

def select_pantry(pool,target,family_target,fail=True):
    support=sorted(target); families=sorted(family_target); n=len(pool)
    specs=[json.loads(x["answer"]) for x in pool]
    matrix=lil_matrix((len(support)+len(families),n),dtype=float); required=[]
    for i,value in enumerate(support):
        for j,row in enumerate(pool):
            if int(row["answer_mode_count"])==value: matrix[i,j]=1
        required.append(target[value])
    for i,family in enumerate(families,start=len(support)):
        for j,row in enumerate(pool):
            if row["answer_mode_family"]==family: matrix[i,j]=1
        required.append(family_target[family])
    hard_ids={'navel_orange','banana','peanut_butter','granny_smith_apple','black_beans_canned'}
    easy_ids={'broccoli','grape_tomatoes'}
    def difficulty_cost(j,spec):
        ids={str(x['id']) for x in spec['ingredients']}
        # One exclusion dominates every possible secondary-score difference.
        return (-1_000_000.0*bool(spec['forbidden_tags'])
                -1_000.0*len(ids&hard_ids)+250.0*len(ids&easy_ids)+j/n)
    objective=np.array([difficulty_cost(j,spec) for j,spec in enumerate(specs)])
    result=milp(objective,integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(matrix.tocsr(),required,required),options={"time_limit":180})
    if not result.success:
        if fail: raise RuntimeError(f"Pantry matched selection failed from pool={n}: {result.message}")
        return None
    return [row for row,chosen in zip(pool,result.x) if chosen>.5]

def build_pantry_adaptive(target,family_target,support_families,excluded,seed,tag,initial,max_pool=5000,step=500):
    """Grow one pool, targeting any rare support that blocks exact matching."""
    curation=json.loads((ROOT/"var/data/pantry_plan_v1/ingredients.json").read_text())
    source={x["id"]:x for x in curation["ingredients"]}; blocked={x[1] for x in excluded}
    rng=random.Random(seed); pool=[]; attempts=0
    def generate(families):
        nonlocal attempts
        while True:
            attempts+=1; family=rng.choice(tuple(families))
            row=_pantry_row(family=family,split=tag,seed=seed,index=attempts,rng=rng,source_by_id=source,ingredient_table_sha256=curation["ingredient_table_sha256"])
            if row is None: continue
            spec=json.loads(row["answer"]); fp_spec=dict(spec); fp_spec.pop("instance_id",None)
            fingerprint=_canonical_sha256(fp_spec)
            if fingerprint in blocked: continue
            row["instance_fingerprint"]=fingerprint; blocked.add(fingerprint); pool.append(row)
            if len(pool)%250==0: print(f"[level2] {tag} Pantry candidates {len(pool)}",flush=True)
            return row
    milestones=list(range(initial,max_pool+1,step))
    if milestones[-1]!=max_pool: milestones.append(max_pool)
    for size in milestones:
        while len(pool)<size: generate(FAMILY_POOLS)
        coverage=Counter(int(row["answer_mode_count"]) for row in pool)
        missing={m:count-coverage[m] for m,count in target.items() if coverage[m]<count}
        for support,deficit in sorted(missing.items()):
            repair_attempts=0
            while coverage[support]<target[support]:
                repair_attempts+=1
                if repair_attempts>100_000: raise RuntimeError(f"Pantry targeted support {support} exhausted")
                row=generate(support_families[support])
                coverage[int(row["answer_mode_count"])]+=1
            print(f"[level2] {tag} repaired support={support} deficit={deficit} pool={len(pool)}",flush=True)
        selected=select_pantry(pool,target,family_target,fail=False)
        print(f"[level2] {tag} Pantry MILP pool={len(pool)} feasible={selected is not None}",flush=True)
        if selected is not None: return selected,len(pool)
    raise RuntimeError(f"Pantry exact marginal match infeasible through pool={len(pool)}")

def identity_set(domain,rows):
    if domain=="pantry": return {("pantry",x["instance_fingerprint"]) for x in rows}
    return _identities(domain,rows)

def materialize(output=OUTPUT_ROOT,pantry_pool_size=1000):
    output=output.resolve()
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True,exist_ok=True)
    built={d:{} for d in DOMAINS}; records={d:{} for d in DOMAINS}
    blocked={d:existing_ids(d) for d in DOMAINS}
    for domain in ("pantry",)+tuple(d for d in DOMAINS if d!="pantry"):
        for split,(expected,dataset_split) in SPLITS.items():
            print(f"[level2] building {domain}/{split}",flush=True)
            easy=load_rows(LEVEL1[domain][split],dataset_split)
            if expected%len(easy): raise RuntimeError(f"{domain}/{split} Level-1 size cannot scale exactly")
            histogram_scale=expected//len(easy)
            target=Counter({m:count*histogram_scale for m,count in modes(easy).items()})
            seed=SEEDS[domain]+SPLIT_SEED_OFFSETS[split]; tag=f"level2_{split}"
            family_target=Counter({family:count*histogram_scale for family,count in Counter(str(x["answer_mode_family"]) for x in easy).items()}) if domain=="pantry" else None
            support_families={m:sorted({str(x["answer_mode_family"]) for x in easy if int(x["answer_mode_count"])==m}) for m in target} if domain=="pantry" else None
            if domain=="countdown": rows=build_countdown(target,seed,tag,blocked[domain])
            elif domain=="graph_coloring": rows=build_graph(target,seed,tag,blocked[domain])
            elif domain=="python_factors": rows=build_python(target,blocked[domain],seed,tag)
            elif domain=="mathir": rows=build_mathir(target,blocked[domain],seed,tag)
            else:
                pool_size=max(pantry_pool_size,1500 if split=="train" else 1000)
                rows,pantry_selected_pool=build_pantry_adaptive(target,family_target,support_families,blocked[domain],seed,tag,pool_size)
            ids=identity_set(domain,rows)
            verifier=lambda row:(str(row.get("modebench_task","")),str(json.loads(row["answer"]).get("verifier","")))
            checks={"row_count":len(rows)==expected,"unique_identities":len(ids)==expected,"disjoint_from_all_level1_and_prior_level2":not(ids&blocked[domain]),"exact_support_histogram":modes(rows)==target,"verifier_and_canonicalization_contract":{verifier(x) for x in easy}=={verifier(x) for x in rows}}
            if domain=="pantry":
                checks["exact_family_histogram"]=Counter(x["answer_mode_family"] for x in rows)==family_target
            if not all(checks.values()): raise RuntimeError(f"{domain}/{split} admission failed: {checks}")
            if domain=="countdown" and not all(len(json.loads(x["answer"])["numbers"])==4 for x in rows): raise RuntimeError("Countdown structural drift")
            if domain=="graph_coloring" and not all(sum(v is None for v in json.loads(x["answer"])["partial_colors"])==4 for x in rows): raise RuntimeError("graph structural drift")
            if domain=="python_factors" and not all(len(json.loads(x["answer"])["cases"])==4 and max(json.loads(x["answer"])["cases"])>96 for x in rows): raise RuntimeError("Python structural drift")
            blocked[domain]|=ids; built[domain][split]=rows
            records[domain][split]={"seed":seed,"level1_reference_rows":len(easy),"histogram_scale_factor":histogram_scale,"rows":len(rows),"rows_sha256":row_hash(rows),"checks":checks,"answer_mode_count_histogram":dict(sorted(target.items()))}
            if domain=="pantry":
                records[domain][split]["family_histogram"]=dict(sorted(family_target.items()))
                records[domain][split]["selected_candidate_pool_size"]=pantry_selected_pool
                records[domain][split]["active_dietary_exclusions"]=sum(bool(json.loads(x["answer"])["forbidden_tags"]) for x in rows)
    staging=Path(tempfile.mkdtemp(prefix=".modebench_harder_v2.",dir=output.parent))
    try:
        for domain in DOMAINS:
            for split,(_,dataset_split) in SPLITS.items(): DatasetDict({dataset_split:Dataset.from_list(built[domain][split])}).save_to_disk(str(staging/domain/split))
        manifest={"schema":SCHEMA,"status":"pass","decision":"structurally_admitted_pending_frozen_base_model_viability","split_sizes":{k:v[0] for k,v in SPLITS.items()},"domains":records,"level1_reference":{d:{s:str(p.relative_to(ROOT)) for s,p in refs.items()} for d,refs in LEVEL1.items()},"fairness_contract":{"support_histogram":"exact counts when source and target sizes match; exact integer expansion otherwise (Pantry dev 64 to 128 uses factor 2)","identity_disjointness":"all Level-1 and all Level-2 splits","verifier_and_canonicalization":"unchanged","prompt_and_response_budget":"paired evaluator must use identical settings"},"viability_gate":{"metric":"pass@8","minimum":0.10,"maximum":0.90,"harder_than_level1":"level2_pass_at_8 < level1_pass_at_8","freeze":"before treatment training"},"difficulty":{"countdown":"four operands versus three; operands up to 14; within each exact support cell prefer shallow paired-product or one-multiplication targets, then smaller operands","graph_coloring":"four hidden vertices versus three","python_factors":"four cases with at least one value above 96","mathir":"variable-on-both-sides families; within each exact support cell prefer symmetric/cancelling variable coefficients, then larger binding magnitude","pantry":"exact support/family matching; lexicographically maximize active dietary exclusions, then the frozen ingredient-composition difficulty score"}}
        for name in ("identity.json","admission_fairness_report.json"): (staging/name).write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        staging.rename(output); return manifest
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True); raise

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path,default=OUTPUT_ROOT)
    parser.add_argument("--pantry-pool-size",type=int,default=1000)
    args=parser.parse_args()
    print(json.dumps(materialize(args.output_root,args.pantry_pool_size),indent=2,sort_keys=True))
