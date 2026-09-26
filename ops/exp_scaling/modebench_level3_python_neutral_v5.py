"""Neutral proper-divisor task: three common3 strata and one all-even stratum.

Mixing these prospectively defined numeric laws can adjust initial correctness
and prompt survival independently. No numeric answer or algorithm is revealed.
The verifier, four distinct cases, legal language and support histogram remain.
"""
import importlib.util
from collections import defaultdict
import json
from pathlib import Path
import modebench_level3_python_neutral_v3 as common3
from modebench_level3_python_neutral_v2 import prompt as legal_prompt
from prepare_level3_neutral_divisor_pilot import finite_prompt

BASE=Path(__file__).with_name('modebench_level3_python_v7.py')
spec=importlib.util.spec_from_file_location('_neutral_v5_even_sampler',BASE)
_even=importlib.util.module_from_spec(spec);spec.loader.exec_module(_even)
PROFILE='python_neutral_divisor_numeric_mixture_v5'
GENERATOR='modebench_level3_python_neutral_candidate_v5'
CASE_WINDOWS=((12,1000),)*4
MINIMUM_BANDS=((12,1000),)*4
COMMON_TIERS={0:0,1:2,2:3}
PRESETS={0:'common3_nearest_feasible_one_odd',1:'common3_nearest_feasible_three_odd',
         2:'common3_nearest_feasible_four_odd',3:'all_even_12_1000'}


def even_catalog(tier):
    if tier!=3:raise ValueError('the even numeric law is only tier3')
    ds={};groups=defaultdict(list)
    for n in range(12,1001,2):
        divisors=_even.proper_divisors(n)
        if len(divisors)>=2:ds[n]=divisors;groups[(len(divisors),1)].append(n)
    return ds,{k:tuple(v) for k,v in sorted(groups.items())}


for name in ('PROFILE','GENERATOR','CASE_WINDOWS','MINIMUM_BANDS','PRESETS'):setattr(_even,name,globals()[name])
_even.catalog=even_catalog


def catalog(tier): return even_catalog(tier) if tier==3 else common3.catalog(COMMON_TIERS[tier])
def available_capacity(support,tier,excluded):
    return _even.available_capacity(support,tier,excluded) if tier==3 else common3.available_capacity(support,COMMON_TIERS[tier],excluded)
def eligible_cases(cases,tier):
    return _even.eligible_cases(cases,tier) if tier==3 else common3.eligible_cases(cases,COMMON_TIERS[tier])
def prompt(cases): return finite_prompt(legal_prompt(cases))


def build_pool(domain,target,excluded,seed,tag,difficulty,multiplier=1):
    source=_even if difficulty==3 else common3
    rows=source.build_pool(domain,target,excluded,seed,tag,difficulty if difficulty==3 else COMMON_TIERS[difficulty],multiplier)
    for row in rows:
        cases=json.loads(row['answer'])['cases']
        row.update(problem=prompt(cases),level3_generator=GENERATOR,level3_generation_profile=PROFILE,level3_difficulty=difficulty,
                   level3_python_preset=PRESETS[difficulty],level3_task_wording='neutral_divisor_task_v5',
                   level3_common_factor=2 if difficulty==3 else 3,level3_odd_cases=sum(n%2 for n in cases))
    return rows
