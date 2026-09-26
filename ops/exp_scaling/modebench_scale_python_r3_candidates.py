"""Prospective Python r3 prime-square laws after the saved r2 development failure.

One recognizable prime square has one forced proper divisor; three ordinary
cases retain the exact overall canonical mode support. This is a structural
hypothesis about short dispatch under the unchanged native prompt/parser. It
makes no claim about model accuracy before fresh registered development.

For each support/tier, integer capacity tickets sample ordinary three-case sets
uniformly, then condition on gcd-one and historical exclusions. Quotas only stop
independent deterministic streams. No receipts, outcomes, or scores are read.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from itertools import combinations_with_replacement
import json
from math import comb, gcd, prod
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
for directory in ('ops','ops/exp_scaling','src'):
    if str(ROOT/directory) not in sys.path:sys.path.insert(0,str(ROOT/directory))
import modebench_scale_candidates as original
from make_python_factor_mode_data import _row as certified_python_row
from oat_drgrpo.python_modebench import proper_divisors

SCHEMA='modebench_scale_python_r3_prime_square_laws_v1'
DOMAINS=('python_factors',)
PRIMES=(13,17,19,23)
ORDINARY_WINDOW=(48,1000)
MAX_PROPOSALS_PER_ROW=100_000
PROFILES={'python_factors':[
    {'cases':4,'exceptional_cases':1,'exceptional_case':prime*prime,
     'exceptional_factor':prime,'exceptional_proper_divisors':1,
     'ordinary_cases':3,'ordinary_case_minimum':ORDINARY_WINDOW[0],
     'ordinary_case_maximum':ORDINARY_WINDOW[1],'ordinary_maximum_smallest_factor':5,
     'ordinary_minimum_proper_divisors':2,'case_gcd':1,
     'sampling':'uniform_ordinary_three_case_sets_conditioned_on_exact_total_support_gcd_and_exclusions'}
    for prime in PRIMES]}
identity=original.identity


def source_paths():
    return sorted({Path(__file__).resolve(),*original.source_paths(),
        Path(sys.modules['oat_drgrpo.python_modebench'].__file__).resolve()})


def _seed(*parts):
    payload=json.dumps([SCHEMA,*parts],sort_keys=True,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(),'big')


def _tier(tier):
    if type(tier) is not int or tier not in range(4):raise ValueError('fixed Python r3 tier0..3 required')
    return PRIMES[tier]


@lru_cache(maxsize=1)
def catalog():
    groups=defaultdict(list);divisors={}
    for value in range(ORDINARY_WINDOW[0],ORDINARY_WINDOW[1]+1):
        ds=proper_divisors(value)
        if len(ds)>=2 and ds[0]<=5:
            groups[len(ds)].append(value);divisors[value]=tuple(ds)
    return divisors,{count:tuple(values) for count,values in groups.items()}


@lru_cache(maxsize=256)
def proposal_profiles(support):
    if type(support) is not int or support<2:raise ValueError('positive multi-mode support required')
    _,groups=catalog();options=[];capacities=[]
    for counts in combinations_with_replacement(sorted(groups),3):
        if prod(counts)!=support:continue
        repeats=tuple(sorted(Counter(counts).items()))
        capacity=prod(comb(len(groups[count]),repeat) for count,repeat in repeats)
        if capacity:options.append(repeats);capacities.append(capacity)
    if not options:raise ValueError('Python r3 ordinary triples cannot realize support'+str(support))
    return tuple(options),tuple(capacities)


def _valid_cases(cases,tier,support=None):
    prime=_tier(tier);divisors,_=catalog()
    if (not isinstance(cases,tuple) or len(cases)!=4 or len(set(cases))!=4
            or tuple(sorted(cases))!=cases or cases.count(prime*prime)!=1 or gcd(*cases)!=1):return False
    ordinary=[n for n in cases if n!=prime*prime]
    return (len(ordinary)==3 and all(n in divisors for n in ordinary)
            and (support is None or prod(len(divisors[n]) for n in ordinary)==support))


def capacity(tier,support,excluded=()):
    """Exact remaining semantic capacity, including gcd and excluded case sets."""
    prime=_tier(tier);_,groups=catalog();options,capacities=proposal_profiles(support)
    # With exceptional p², gcd-one fails exactly when all ordinary n share p.
    bad=sum(prod(comb(sum(n%prime==0 for n in groups[count]),repeat)
                 for count,repeat in repeats) for repeats in options)
    blocked={tuple(key[1]) for key in excluded
        if isinstance(key,(tuple,list)) and len(key)==2 and key[0]=='python_factors'
        and isinstance(key[1],(tuple,list)) and _valid_cases(tuple(key[1]),tier,support)}
    return sum(capacities)-bad-len(blocked)


def _proposal(rng,tier,support):
    prime=_tier(tier);_,groups=catalog();options,capacities=proposal_profiles(support)
    ticket=rng.randrange(sum(capacities))
    for repeats,total in zip(options,capacities):
        if ticket<total:break
        ticket-=total
    cases=[prime*prime]
    for count,repeat in repeats:cases.extend(rng.sample(groups[count],repeat))
    return tuple(sorted(cases))


def build_pool(domain,target,excluded,seed,tag,tier,multiplier=1,*,joint_target=None):
    if domain!='python_factors' or joint_target is not None:raise ValueError('Python-only exact marginal support law required')
    _tier(tier)
    if type(seed) is not int or seed<0 or type(multiplier) is not int or multiplier<1:
        raise ValueError('nonnegative fixed seed and positive integer multiplier required')
    if any(type(support) is not int or support<2 or type(count) is not int or count<0 for support,count in target.items()):
        raise ValueError('exact support quotas required')
    required=Counter({support:count*multiplier for support,count in target.items() if count})
    blocked=set(excluded);rows=[]
    for support,count in sorted(required.items()):
        if capacity(tier,support,blocked)<count:raise ValueError('insufficient exact unexcluded Python r3 capacity')
        rng=random.Random(_seed(seed,tier,support,'python_factors'))
        for index in range(count):
            for _ in range(MAX_PROPOSALS_PER_ROW):
                cases=_proposal(rng,tier,support);key=('python_factors',cases)
                if gcd(*cases)!=1 or key in blocked:continue
                blocked.add(key)
                row=certified_python_row(cases=cases,split_tag=f'{tag}-t{tier}-m{support}',seed=seed,index=index)
                row.update(scale_candidate_generator=SCHEMA,scale_candidate_tier=tier,
                    scale_candidate_profile=json.dumps(PROFILES[domain][tier],sort_keys=True,separators=(',',':')),
                    scale_origin_metadata='{}',scale_cell_index=index,scale_python_r3_law_version=SCHEMA)
                rows.append(row);break
            else:raise RuntimeError(f'Python r3 fixed proposal budget exhausted: tier{tier}/support{support}')
    rows.sort(key=lambda row:_seed(seed,tier,'display',identity(domain,row)))
    ids={identity(domain,row) for row in rows}
    if len(ids)!=len(rows) or ids & set(excluded):raise RuntimeError('Python r3 semantic overlap')
    if Counter(row['answer_mode_count'] for row in rows)!=required:raise RuntimeError('Python r3 exact support histogram drift')
    return rows


def verify_rows(domain,rows):
    if domain!='python_factors':raise ValueError('Python-only r3 row audit required')
    for row in rows:
        tier=row.get('scale_candidate_tier');_tier(tier)
        spec=json.loads(row['answer'])
        if (row.get('scale_candidate_generator')!=SCHEMA or row.get('scale_python_r3_law_version')!=SCHEMA
                or row.get('scale_candidate_profile')!=json.dumps(PROFILES[domain][tier],sort_keys=True,separators=(',',':'))
                or not _valid_cases(tuple(spec['cases']),tier,row['answer_mode_count'])):
            raise RuntimeError('Python r3 fixed structural law changed')
    return {**original.verify_rows(domain,rows),'prime_square_structural_profile':True}
