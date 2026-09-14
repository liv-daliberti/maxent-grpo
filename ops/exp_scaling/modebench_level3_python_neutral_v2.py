"""Neutral Python numeric calibration with legal-syntax-first task wording.

The neutral system prompt, permitted programs, cases, verifier and outcome
keys are unchanged. The task's existing syntax restriction is stated first,
explicitly excluding loops/comprehensions and naming common forbidden calls.
No valid program, divisor example or solution strategy is supplied.
"""
import importlib.util
import json
from pathlib import Path
BASE=Path(__file__).with_name('modebench_level3_python_v7.py')
spec=importlib.util.spec_from_file_location('_neutral_v2_sampler',BASE)
_sampler=importlib.util.module_from_spec(spec);spec.loader.exec_module(_sampler)
PROFILE='python_neutral_numeric_bands_v2'
GENERATOR='modebench_level3_python_neutral_candidate_v2'
CASE_WINDOWS=((12,240),(30,256),(50,320),(60,512))
MINIMUM_BANDS=((12,29),(30,49),(50,79),(60,119))
PRESETS={i:f'neutral_syntax_first_minimum_{a}_{b}_cases_{lo}_{hi}' for i,((a,b),(lo,hi)) in enumerate(zip(MINIMUM_BANDS,CASE_WINDOWS))}
for name in ('PROFILE','GENERATOR','CASE_WINDOWS','MINIMUM_BANDS','PRESETS'):setattr(_sampler,name,globals()[name])
available_capacity=_sampler.available_capacity
catalog=_sampler.catalog
eligible_cases=_sampler.eligible_cases


def prompt(cases):
    values=', '.join(map(str,cases))
    return ('Write a pure Python lambda expression using only the variable n, integer literals, '
            '+, -, *, //, %, comparisons, Boolean operators, and conditional expressions. '
            'Function calls and loops are forbidden. In particular, do not use next, range, '
            'comprehensions, imports, attributes, containers, or any name other than n. '
            f'An external Python tool will evaluate your function at each n in [{values}]. '
            'For every one of these inputs, return an integer d with 1 < d < n and n % d == 0. '
            'Different valid return vectors are different solution modes. '
            'Return exactly one line of the form lambda n: EXPR inside \\boxed{} and no explanation.')


def build_pool(*args,**kwargs):
    rows=_sampler.build_pool(*args,**kwargs)
    for row in rows:
        row['problem']=prompt(json.loads(row['answer'])['cases'])
        row['level3_task_wording']='neutral_legal_syntax_first_v2'
    return rows
