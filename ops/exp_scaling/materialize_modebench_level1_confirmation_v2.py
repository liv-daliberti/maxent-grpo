#!/usr/bin/env python3
"""Prepare or publish fresh Level 1 reserves without reading model outcomes.

Inspection records immutable sources, original evaluation cells, exclusions,
fixed generation seeds and finite-law capacities. It never samples rows.
Publication is a separate operation requiring that exact prospective plan.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT/'ops', ROOT/'ops/exp_scaling', ROOT/'src'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from datasets import Dataset, DatasetDict, load_from_disk
import make_exact_answer_mode_data as graph
import make_exact_countdown_mode_data as countdown
import make_python_factor_mode_data as python_factors
import make_mathir_action_menu_data as mathir
import make_pantry_plan_mode_data as pantry
from fit_modebench_level3 import atomic_new, local_dependency_sources
from oat_drgrpo.python_modebench import proper_divisors, python_factor_mode_count

SCHEMA = 'modebench_level1_fresh_confirmation_plan_v2'
DOMAINS = ('countdown', 'graph_coloring', 'python_factors', 'mathir', 'pantry')
VERIFIERS = dict(zip(DOMAINS, ('countdown', 'graph_coloring', 'python_factor_function',
                              'mathir_action_menu', 'pantry_plan')))
DOMAIN_BY_VERIFIER = {value:key for key,value in VERIFIERS.items()}
ROOT_PREFIXES = ('exact_countdown', 'graph_coloring_modebench', 'python_factor_modebench',
                 'mathir_action_menu', 'pantry_plan_modebench', 'modebench_harder',
                 'modebench_level1', 'modebench_level2', 'modebench_level3',
                 'e117_evaluation_reserve')
DEFAULT_SEEDS = dict(zip(DOMAINS, (9411700, 9411800, 9411900, 9412000, 9412100)))
DEFAULT_OUTPUT = ROOT/'var/data/modebench_level3_v2_level1_reserve'
CURATION = ROOT/'var/data/pantry_plan_v1/ingredients.json'
REFERENCES = {
    d:ROOT/f'var/data/e117_evaluation_reserve_v1/confirmation/{d}/eval'
    for d in DOMAINS if d != 'pantry'
}
REFERENCES['pantry'] = ROOT/'var/data/pantry_plan_modebench_v2/eval'
GRAPH_CHUNK = 32
PANTRY_PER_FAMILY_CHUNK = 4
PROPOSAL_LIMITS = {'graph_coloring':16384, 'pantry':16384}
ORIGINAL_LAWS = {
    'countdown': {'number_count':3,'max_value':12,'min_modes':2,'max_modes':8,
                  'sampling':'uniform original exhaustive catalogue, conditioned on exact support cells'},
    'graph_coloring': {'hidden_count':3,'min_completions':4,'max_completions':24,
                       'min_solutions':4,'max_n':6,'max_edges':8,
                       'prompt_style':'original','balance_hidden_color':False,
                       'sampling':'unchanged accepted generator proposals before exact-cell quota filtering'},
    'python_factors': {'case_count':4,'max_value':96,'min_modes':16,
                       'sampling':'uniform original eligible unordered case sets, conditioned on exact support cells'},
    'mathir': {'families':[family.name for family in mathir.FAMILIES],
               'rows_per_family':32,'sampling':'unchanged original _build_rows and binding/menu laws'},
    'pantry': {'ingredients_per_row':6,'available_g':[100,125,150],
               'support_range':[8,64],'rows_per_family':32,
               'sampling':'unchanged original accepted _build_rows proposals before support/family quota filtering'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows_sha(rows):
    return hashlib.sha256('\n'.join(json.dumps(row,sort_keys=True,separators=(',',':'))
                                    for row in rows).encode()).hexdigest()


def prompt_sha(prompt):
    require(isinstance(prompt,str), 'prompt must be a string')
    return hashlib.sha256(prompt.encode()).hexdigest()


def semantic_identity(domain, row):
    spec = json.loads(row['answer']) if isinstance(row['answer'],str) else row['answer']
    if domain == 'countdown':
        return (domain, tuple(sorted(map(int,spec['numbers']))), int(spec['target']))
    if domain == 'graph_coloring':
        return (domain, int(spec['n']), tuple(sorted(tuple(map(int,edge)) for edge in spec['edges'])),
                graph._graph_color_string(spec['partial_colors']))
    if domain == 'python_factors':
        return (domain, tuple(sorted(map(int,spec['cases']))))
    if domain == 'mathir':
        return (domain, str(spec['family']), tuple(sorted((str(k),int(v)) for k,v in spec['bindings'].items())))
    if domain == 'pantry':
        body = dict(spec); body.pop('instance_id',None)
        computed = pantry._canonical_sha256(body)
        # Preserve historical semantic keys as well as exact prompts. Some older
        # repair banks intentionally retain their original semantic fingerprint.
        return (domain, str(row.get('instance_fingerprint',computed)))
    raise ValueError('unknown domain')


def cell(domain, row):
    return ((int(row['answer_mode_count']),str(row['answer_mode_family'])) if domain == 'pantry'
            else (int(row['answer_mode_count']),))


def serialize_cells(target):
    return [{'cell':list(key),'rows':count} for key,count in sorted(target.items())]


def restore_cells(records):
    return Counter({tuple(item['cell']):int(item['rows']) for item in records})


def derived_seed(seed, *labels):
    return int(digest([int(seed),*labels]),16)


def discover_sources(data_root):
    """Only data banks; training logs/model-response JSONL files are never read."""
    found = set()
    for root in sorted(Path(data_root).iterdir()):
        if not root.is_dir() or not root.name.startswith(ROOT_PREFIXES):
            continue
        for marker in root.rglob('dataset_dict.json'):
            found.add(('dataset_dict',str(marker.parent.resolve())))
        for path in root.rglob('*.jsonl'):
            if 'pools' in path.relative_to(root).parts:
                found.add(('jsonl',str(path.resolve())))
    return [{'kind':kind,'path':path} for kind,path in sorted(found)]


def source_snapshot(source):
    path = Path(source['path'])
    files = [path] if source['kind'] == 'jsonl' else sorted(p for p in path.rglob('*') if p.is_file())
    return {**source,'files':{str(p.resolve()):file_sha(p) for p in files}}


def read_source(source):
    path = Path(source['path'])
    if source['kind'] == 'jsonl':
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                require({'problem','answer','modebench_task'} <= set(row),
                        'candidate-pool file contains non-problem data: '+str(path))
                yield row
        return
    data = load_from_disk(str(path))
    for subset in data.values():
        if not {'problem','answer','modebench_task'} <= set(subset.column_names):
            continue
        for row in subset:
            yield dict(row)


def collect_exclusions(sources):
    ids = {d:set() for d in DOMAINS}; prompts = {d:set() for d in DOMAINS}
    records = []
    for source in sources:
        snapshot = source_snapshot(source); counts = Counter()
        for row in read_source(source):
            domain = DOMAIN_BY_VERIFIER.get(row.get('modebench_task'))
            if domain is None:
                continue
            ids[domain].add(semantic_identity(domain,row))
            prompts[domain].add(prompt_sha(row['problem'])); counts[domain] += 1
            if domain == 'pantry':
                spec = json.loads(row['answer']) if isinstance(row['answer'],str) else row['answer']
                payload = dict(spec); payload.pop('instance_id',None)
                ids[domain].add((domain,pantry._canonical_sha256(payload)))
        require(snapshot == source_snapshot(source), 'exclusion source changed while reading')
        records.append({**snapshot,'rows_by_domain':dict(counts)})
    return ids,prompts,records


def exclusion_summary(ids,prompts):
    return {d:{'semantic_identities':len(ids[d]),'semantic_identities_sha256':digest(sorted(ids[d])),
               'exact_prompts':len(prompts[d]),'exact_prompt_hashes_sha256':digest(sorted(prompts[d]))}
            for d in DOMAINS}


@lru_cache(maxsize=1)
def countdown_catalog():
    """Exhaustive law inventory only: no RNG, row selection, or model calls."""
    entries = []
    for numbers in combinations(range(2,13),3):
        for target,expressions in countdown._countdown_expression_map(list(numbers)).items():
            if target <= 0 or target in numbers or abs(target) > 36 or not expressions:
                continue
            keys = countdown._canonical_expression_keys(list(numbers),target,expressions)
            if 2 <= len(keys) <= 8:
                entries.append((numbers,int(target),len(keys)))
    return tuple(entries)


def python_catalog(max_value=96):
    values = python_factors._candidate_values(max_value)
    sizes = {n:len(proper_divisors(n)) for n in values}
    for cases in combinations(values,4):
        count = 1
        for n in cases:
            count *= sizes[n]
        if count >= 16:
            yield cases,count


def finite_cells(domain,excluded,prompts):
    cells = defaultdict(list)
    if domain == 'countdown':
        for numbers,target,count in countdown_catalog():
            identity = (domain,numbers,target)
            if identity not in excluded and prompt_sha(countdown._countdown_prompt(list(numbers),target)) not in prompts:
                cells[(count,)].append((numbers,target))
    elif domain == 'python_factors':
        for cases,count in python_catalog():
            if (domain,cases) not in excluded and prompt_sha(python_factors._prompt(cases)) not in prompts:
                cells[(count,)].append(cases)
    else:
        raise ValueError('finite cell inventory supports Countdown/Python only')
    return cells


def mathir_original_identity_holds(identity):
    if len(identity) != 3 or identity[0] != 'mathir':
        return False
    family = identity[1]; b = dict(identity[2])
    names = {f.name for f in mathir.FAMILIES}
    if family not in names or set(b) != (set('abcd') if 'd' in family else set('abc')):
        return False
    if not (1 <= abs(b['a']) <= 9 and -12 <= b['b'] <= 12):
        return False
    if family == 'ax_plus_b_eq_c':
        solution = Fraction(b['c']-b['b'],b['a'])
    elif family == 'x_over_a_plus_b_eq_c':
        solution = Fraction(b['c']-b['b'])
    else:
        if not 1 <= abs(b['d']) <= 9:
            return False
        coeff = b['a']-b['d'] if family == 'ax_plus_b_eq_dx_plus_c' else b['a']+b['d']
        if not coeff:
            return False
        solution = Fraction(b['c']-b['b'],coeff)
    return solution.denominator == 1 and 1 <= abs(solution) <= 9


def capacity_report(targets,ids,prompts):
    report = {}
    for domain in ('countdown','python_factors'):
        inventory = finite_cells(domain,ids[domain],prompts[domain])
        cells = [{'cell':list(key),'required':count,'available':len(inventory.get(key,[]))}
                 for key,count in sorted(targets[domain].items())]
        report[domain] = {'kind':'exact_finite_law_after_semantic_and_prompt_exclusions',
                          'cells':cells,'capacity_pass':all(c['available'] >= c['required'] for c in cells)}
        if domain == 'countdown':
            report[domain]['original_catalog_size'] = len(countdown_catalog())
            report[domain]['original_catalog_by_support'] = dict(Counter(c for _,_,c in countdown_catalog()))
            report[domain]['fresh_remaining_total'] = sum(map(len,inventory.values()))
    blocked = Counter(i[1] for i in ids['mathir'] if mathir_original_identity_holds(i))
    cells = [{'family':f.name,'required':32,'available':(8100 if n < 2 else 137700)-blocked[f.name]}
             for n,f in enumerate(mathir.FAMILIES)]
    # Both-sided coefficients have17 alternatives for d for each nonzero a.
    report['mathir'] = {'kind':'exact_semantic_family_capacity','cells':cells,
                        'exact_prompt_exclusions_also_checked_before_publication':True,
                        'capacity_pass':all(c['available'] >= c['required'] for c in cells)}
    for domain in ('graph_coloring','pantry'):
        report[domain] = {'kind':'not_exhaustively_enumerated_no_proposals_sampled',
                          'capacity_pass':None,'fixed_accepted_proposal_budget':PROPOSAL_LIMITS[domain],
                          'on_budget_exhaustion':'fail_with_deficient_cells_without_changing_law_or_seed'}
    return report


def load_reference(path):
    return [dict(row) for row in load_from_disk(str(path))['multi_answer']]


def prepare_plan(data_root=ROOT/'var/data',references=None,seeds=None):
    references = {d:Path(p).resolve() for d,p in (references or REFERENCES).items()}
    seeds = dict(DEFAULT_SEEDS if seeds is None else seeds)
    require(set(references) == set(seeds) == set(DOMAINS), 'all five references and generation seeds required')
    require(len(set(seeds.values())) == 5 and all(type(s) is int and s >= 0 for s in seeds.values()),
            'generation seeds must be five distinct nonnegative integers')
    targets = {}; reference_records = {}
    for domain,path in references.items():
        rows = load_reference(path)
        require(len(rows) == 128, 'each original reference must contain128 rows')
        targets[domain] = Counter(cell(domain,row) for row in rows)
        if domain == 'mathir':
            require(Counter(json.loads(row['answer'])['family'] for row in rows) ==
                    Counter({f.name:32 for f in mathir.FAMILIES}), 'MathIR original family balance differs')
        reference_records[domain] = {'path':str(path),'rows_sha256':rows_sha(rows),
                                     'cells':serialize_cells(targets[domain]),
                                     'source':source_snapshot({'kind':'dataset_dict','path':str(path)})}
    sources = discover_sources(data_root)
    ids,prompts,records = collect_exclusions(sources)
    for domain,path in references.items():
        rows = load_reference(path)
        require(all(semantic_identity(domain,r) in ids[domain] and prompt_sha(r['problem']) in prompts[domain]
                    for r in rows), 'reference missing from complete exclusions')
    capacities = capacity_report(targets,ids,prompts)
    return {'schema':SCHEMA,'created_utc':datetime.now(timezone.utc).isoformat(),
            'status':'prospective_plan_no_rows_generated','data_root':str(Path(data_root).resolve()),
            'generation_seeds':seeds,'original_laws':ORIGINAL_LAWS,
            'graph_fixed_chunk_rows':GRAPH_CHUNK,'pantry_fixed_chunk_per_family':PANTRY_PER_FAMILY_CHUNK,
            'proposal_limits':PROPOSAL_LIMITS,'references':reference_records,
            'exclusion_sources':records,'exclusions':exclusion_summary(ids,prompts),
            'capacity':capacities,'source_sha256':local_dependency_sources([Path(__file__)]),
            'curation_path':str(CURATION),'curation_sha256':file_sha(CURATION),
            'model_outcomes_read':False,'rows_generated':False,'rows_published':False,
            'policy':'Original accepted laws conditioned only on preregistered cells, semantic/exact-prompt exclusions, and uniqueness; no score inputs, law expansion, fallback, or alternate seeds.'}


def validate_plan(plan):
    require(plan.get('schema') == SCHEMA and plan.get('rows_generated') is False, 'prospective plan required')
    require(plan['source_sha256'] == local_dependency_sources([Path(__file__)]), 'planned source changed')
    require(plan['original_laws'] == ORIGINAL_LAWS and plan['proposal_limits'] == PROPOSAL_LIMITS and
            plan['graph_fixed_chunk_rows'] == GRAPH_CHUNK and
            plan['pantry_fixed_chunk_per_family'] == PANTRY_PER_FAMILY_CHUNK, 'planned laws or budgets changed')
    current = discover_sources(Path(plan['data_root']))
    expected = [{k:s[k] for k in ('kind','path')} for s in plan['exclusion_sources']]
    require(current == expected, 'exclusion source inventory changed')
    ids,prompts,records = collect_exclusions(current)
    require(records == plan['exclusion_sources'] and exclusion_summary(ids,prompts) == plan['exclusions'],
            'exclusion bytes or identities changed')
    require(file_sha(plan['curation_path']) == plan['curation_sha256'], 'curation changed')
    for domain,record in plan['references'].items():
        require(source_snapshot({k:record['source'][k] for k in ('kind','path')}) == record['source'],
                'reference source changed')
        rows = load_reference(record['path'])
        require(rows_sha(rows) == record['rows_sha256'] and
                serialize_cells(Counter(cell(domain,r) for r in rows)) == record['cells'], 'reference cells changed')
    return ids,prompts


def keep_needed(domain,proposals,target,excluded,prompts,selected,seen):
    got = Counter(cell(domain,row) for row in selected)
    selected_prompts = {prompt_sha(row['problem']) for row in selected}
    for row in proposals:
        identity = semantic_identity(domain,row); key = cell(domain,row)
        if identity in excluded or identity in seen:
            continue
        seen.add(identity)
        prompt_digest = prompt_sha(row['problem'])
        if prompt_digest in prompts or prompt_digest in selected_prompts or got[key] >= target[key]:
            continue
        selected.append(row); got[key] += 1; selected_prompts.add(prompt_digest)
    return got == target


def build_domain_rows(domain,target,excluded,prompts,seed,tag='fresh_level1_confirmation_v2'):
    require(sum(target.values()) == 128 and all(type(n) is int and n > 0 for n in target.values()),
            'exact positive128-row cells required')
    selected = []
    if domain in ('countdown','python_factors'):
        inventory = finite_cells(domain,excluded,prompts)
        for key,count in sorted(target.items()):
            choices = inventory.get(key,[])
            require(len(choices) >= count, f'{domain}: finite cell {key} needs{count}, has{len(choices)}')
            chosen = random.Random(derived_seed(seed,'cell',list(key))).sample(choices,count)
            if domain == 'countdown':
                # This is precisely the original uniform catalogue law conditioned
                # on one support cell; target/operand filters are unchanged.
                blocked = {(i[1],i[2]) for i in excluded}
                chosen_set = set(chosen)
                blocked |= {(numbers,target_value) for numbers,target_value,_ in countdown_catalog()
                            if (numbers,target_value) not in chosen_set}
                rows = countdown._synthetic_countdown_rows(count,seed=derived_seed(seed,'render',list(key)),
                    split_tag=tag,number_count=3,max_value=12,min_modes=2,max_modes=8,exclude=blocked)
                require({semantic_identity(domain,r)[1:] for r in rows} == set(chosen), 'Countdown chosen catalogue differs')
                selected.extend(rows)
            else:
                base_index = len(selected)
                selected.extend(python_factors._row(cases=cases,split_tag=tag,seed=seed,index=base_index+index)
                                for index,cases in enumerate(chosen))
    elif domain == 'mathir':
        selected = mathir._build_rows(128,seed=seed,split_tag=tag,family_support=mathir._family_support(),
                                      excluded={(i[1],i[2]) for i in excluded})
    elif domain in ('graph_coloring','pantry'):
        seen = set(); proposed = 0; chunk_index = 0
        curation = json.loads(CURATION.read_text()) if domain == 'pantry' else None
        while proposed < PROPOSAL_LIMITS[domain]:
            chunk_seed = derived_seed(seed,'accepted_chunk',chunk_index)
            if domain == 'graph_coloring':
                rows = graph._synthetic_graph_rows(GRAPH_CHUNK,seed=chunk_seed,split_tag=tag,
                    hidden_count=3,min_completions=4,max_completions=24,min_solutions=4,
                    max_n=6,max_edges=8,prompt_style='original',balance_hidden_color=False,
                    exclude={i[1:] for i in excluded | seen})
            else:
                rows = pantry._build_rows(per_family=PANTRY_PER_FAMILY_CHUNK,split=tag,seed=chunk_seed,
                    curation=curation,excluded_fingerprints={i[1] for i in excluded | seen})
            proposed += len(rows); chunk_index += 1
            if keep_needed(domain,rows,target,excluded,prompts,selected,seen):
                break
        require(Counter(cell(domain,r) for r in selected) == target,
                f'{domain}: fixed accepted-proposal budget exhausted; missing{target-Counter(cell(domain,r) for r in selected)}')
    else:
        raise ValueError('unknown domain')
    random.Random(derived_seed(seed,'output_order')).shuffle(selected)
    validate_rows(domain,selected,target,excluded,prompts)
    return selected


def validate_rows(domain,rows,target,excluded,prompts):
    ids = {semantic_identity(domain,r) for r in rows}
    texts = {prompt_sha(r['problem']) for r in rows}
    require(len(rows) == len(ids) == len(texts) == 128, 'reserve must contain128 unique identities and prompts')
    require(not ids & excluded and not texts & prompts, 'reserve overlaps a historical identity or exact prompt')
    require(Counter(cell(domain,r) for r in rows) == target, 'reserve support cells differ')
    for row in rows:
        spec = json.loads(row['answer'])
        require(row['modebench_task'] == spec['verifier'] == VERIFIERS[domain], 'verifier contract changed')
        if domain == 'countdown':
            require(len(spec['numbers']) == len(set(spec['numbers'])) == 3 and min(spec['numbers']) >= 2
                    and max(spec['numbers']) <= 12 and 0 < spec['target'] <= 36 and spec['target'] not in spec['numbers'],
                    'Countdown original numeric law changed')
            require(len(countdown._canonical_expression_keys(spec['numbers'],spec['target'])) == row['answer_mode_count'],
                    'Countdown support differs from original grader')
        elif domain == 'python_factors':
            require(len(spec['cases']) == len(set(spec['cases'])) == 4 and max(spec['cases']) <= 96 and
                    set(spec['cases']) <= set(python_factors._candidate_values(96)), 'Python original value law changed')
            require(python_factor_mode_count(spec['cases']) == row['answer_mode_count'], 'Python support differs')
        elif domain == 'graph_coloring':
            require(4 <= spec['n'] <= 6 and len(spec['edges']) <= 8 and
                    sum(c is None for c in spec['partial_colors']) == 3, 'Graph original bounds changed')
            valid = graph._valid_graph_colorings(spec['n'],spec['edges'])
            count = sum(all(p is None or p == colors[i] for i,p in enumerate(spec['partial_colors'])) for colors in valid)
            require(count == row['answer_mode_count'] and 4 <= count <= 24, 'Graph original exact support differs')
            require(row['problem'] == graph._prompt_for_style(spec['n'],spec['edges'],spec['partial_colors'],prompt_style='original'),
                    'Graph original prompt changed')
        elif domain == 'mathir':
            require(mathir_original_identity_holds(semantic_identity(domain,row)), 'MathIR original binding law changed')
            family = next(f for f in mathir.FAMILIES if f.name == spec['family'])
            require(spec['initial_lhs'] == family.initial_lhs and spec['initial_rhs'] == family.initial_rhs and
                    spec['max_steps'] == 4 and set(spec['actions']) == set('ABCDEF') and
                    set(spec['actions'].values()) == set(family.commands), 'MathIR original action contract changed')
        elif domain == 'pantry':
            require(len(spec['ingredients']) == 6 and all(i['available_g'] in (100,125,150) for i in spec['ingredients'])
                    and spec['min_ingredients'] == 2 and spec['max_ingredients'] == 4, 'Pantry original bounds changed')
            require(len(pantry.enumerate_pantry_supports(spec)) == row['answer_mode_count'], 'Pantry original support differs')
    if domain == 'mathir':
        require(Counter(json.loads(r['answer'])['family'] for r in rows) == Counter({f.name:32 for f in mathir.FAMILIES}),
                'MathIR original32-per-family balance changed')


def materialize(plan_path,output_root):
    output_root = Path(output_root).resolve()
    require(not output_root.exists(), 'fresh output root required; never overwrite a reserve')
    plan_path = Path(plan_path).resolve(); plan_sha = file_sha(plan_path)
    plan = json.loads(plan_path.read_text()); ids,prompts = validate_plan(plan)
    require(all(v.get('capacity_pass') is not False for v in plan['capacity'].values()), 'planned finite capacity failed')
    rows = {d:build_domain_rows(d,restore_cells(plan['references'][d]['cells']),ids[d],prompts[d],
                               plan['generation_seeds'][d]) for d in DOMAINS}
    require(file_sha(plan_path) == plan_sha, 'plan changed while generating')
    validate_plan(plan)
    output_root.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output_root.name+'.',dir=output_root.parent))
    try:
        records = {}
        for domain in DOMAINS:
            DatasetDict({'multi_answer':Dataset.from_list(rows[domain])}).save_to_disk(str(staging/domain/'eval'))
            records[domain] = {'rows':128,'rows_sha256':rows_sha(rows[domain]),
                               'path':str(output_root/domain/'eval'),
                               'cells':serialize_cells(Counter(cell(domain,r) for r in rows[domain])),
                               'semantic_and_exact_prompt_exclusions_pass':True}
        report = {'schema':'modebench_level1_fresh_confirmation_reserve_v2','created_utc':datetime.now(timezone.utc).isoformat(),
                  'status':'structural_checks_pass_no_model_evaluation','plan_path':str(plan_path),'plan_sha256':plan_sha,
                  'domains':records,'source_sha256':plan['source_sha256'],'model_outcomes_read':False,
                  'generation_seeds':plan['generation_seeds'],'exclusions':plan['exclusions']}
        (staging/'plan.json').write_bytes(plan_path.read_bytes())
        (staging/'identity.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        require(not output_root.exists(), 'output appeared before publication')
        staging.rename(output_root)
        return report
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--inspect',action='store_true',help='prepare metadata/capacity plan without sampling (default)')
    action.add_argument('--publish-plan',type=Path,help='explicitly generate/publish from an unchanged prospective plan')
    parser.add_argument('--plan-output',type=Path,help='fresh JSON path for inspection plan')
    parser.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.publish_plan:
        require(args.plan_output is None, 'plan output is only for inspection')
        report = materialize(args.publish_plan,args.output_root)
        print(json.dumps({'output_root':str(args.output_root),'status':report['status']},indent=2))
    else:
        report = prepare_plan()
        if args.plan_output:
            atomic_new(args.plan_output,report)
        print(json.dumps({'status':report['status'],'generation_seeds':report['generation_seeds'],
                          'capacity':report['capacity'],'plan_output':str(args.plan_output) if args.plan_output else None},indent=2))

if __name__ == '__main__':
    main()
