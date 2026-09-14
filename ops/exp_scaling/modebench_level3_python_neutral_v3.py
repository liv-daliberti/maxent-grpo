"""Development candidate: common-factor-three tasks with controlled parity.

Every case set has the same exact divisor-vector support as its reference cell.
The four strata prefer one, two, three or four odd cases, clamped to the nearest feasible count within each support cell. No factor
or solution strategy is revealed in the task; legal-syntax-first wording is
inherited unchanged from neutral v2. Uniform integer tickets choose among all
eligible distinct unordered four-case sets conditional on support and parity.
"""
from collections import Counter,defaultdict
from functools import lru_cache
import importlib.util
from itertools import combinations_with_replacement
import json
from math import comb,prod
from pathlib import Path
from modebench_level3_python_neutral_v2 import prompt

BASE=Path(__file__).with_name('modebench_level3_python_v7.py')
spec=importlib.util.spec_from_file_location('_neutral_v3_sampler',BASE)
_sampler=importlib.util.module_from_spec(spec);spec.loader.exec_module(_sampler)
PROFILE='python_neutral_common3_feasible_parity_v3'
GENERATOR='modebench_level3_python_neutral_candidate_v3'
CASE_WINDOWS=((12,1000),)*4
MINIMUM_BANDS=((12,1000),)*4
PRESETS={i:f'common_factor3_nearest_feasible_to_{i+1}_odd_cases_12_1000' for i in range(4)}


@lru_cache(maxsize=4)
def catalog(tier):
    if type(tier) is not int or tier not in range(4):raise ValueError('tier must be an exact integer in0..3')
    divisors={};groups=defaultdict(list)
    for value in range(12,1001):
        if value%3:continue
        ds=_sampler.proper_divisors(value)
        if len(ds)>=2:
            divisors[value]=ds;groups[(len(ds),value%2)].append(value)
    return divisors,{k:tuple(v) for k,v in sorted(groups.items())}


@lru_cache(maxsize=512)
def support_profiles(tier,support):
    _,groups=catalog(tier);profiles=[];capacities=[]
    for profile in combinations_with_replacement(sorted(groups),4):
        if prod(k[0] for k in profile)!=support:continue
        counts=Counter(profile);capacity=prod(comb(len(groups[k]),n) for k,n in counts.items())
        if capacity:profiles.append(tuple(counts.items()));capacities.append(capacity)
    if not profiles:raise ValueError(f'common3 parity tier{tier} cannot realize support{support}')
    odd_counts=[sum(k[1]*n for k,n in profile) for profile in profiles]
    selected=min(set(odd_counts),key=lambda n:(abs(n-(tier+1)),n))
    chosen=[i for i,n in enumerate(odd_counts) if n==selected]
    return tuple(profiles[i] for i in chosen),tuple(capacities[i] for i in chosen)


def eligible_cases(cases,tier):
    divisors,_=catalog(tier)
    if len(cases)!=len(set(cases)) or len(cases)!=4 or not all(v in divisors for v in cases):return False
    profiles,_=support_profiles(tier,prod(len(divisors[v]) for v in cases))
    target=sum(k[1]*n for k,n in profiles[0])
    return sum(v%2 for v in cases)==target


for name in ('PROFILE','GENERATOR','CASE_WINDOWS','MINIMUM_BANDS','PRESETS','catalog','support_profiles','eligible_cases'):
    setattr(_sampler,name,globals()[name])
available_capacity=_sampler.available_capacity


def build_pool(*args,**kwargs):
    rows=_sampler.build_pool(*args,**kwargs)
    for row in rows:
        cases=json.loads(row['answer'])['cases']
        row['problem']=prompt(cases)
        row['level3_task_wording']='neutral_legal_syntax_first_v2'
        row['level3_odd_cases']=sum(v%2 for v in cases)
        row['level3_common_factor']=3
    return rows
