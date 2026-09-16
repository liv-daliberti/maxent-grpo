#!/usr/bin/env python3
"""Offline, authenticated analysis of fresh 64-draw ModeBench discovery curves.

Primary curves are exact without-replacement rarefaction of each observed pool.
Nothing in this module makes generation or network calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

CONDITION = 'sampling_budget_ablation_v1'
SEED = 20260911
ROOT_MANIFEST_SHA = '1d42de03313dc571329cbc14b4339d124c0994aae36eb6d2a8f4ea9d5b6b05e3'
LOCAL_PLAN_SHA = '7973dcdcc881b8dedab242036ce48b5e32a41ce8f505fc0af6b77fc7c0a511fb'
DRAWS = 64
PROMPTS_PER_CELL = 16
GRID = (1, 2, 4, 8, 16, 32, 64)
DOMAINS = ('python_factors', 'mathir', 'pantry_plan')
LEVELS = (2, 3)
ARMS = ('original', 'neutral')
GRADINGS = ('strict', 'normalized_secondary')
CONTRAST = 'neutral_minus_original'
DOMAIN_LABELS = {'python_factors':'Python factors', 'mathir':'MathIR', 'pantry_plan':'Pantry'}
METHOD_LABELS = {'initial':'Initial Qwen 0.5B', 'drgrpo':'DrGRPO', 'replay_drgrpo':'Re:Dr'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    return {'path':str(Path(path).resolve()), 'sha256':file_sha(path)}


def bound_file(path, digest):
    require(isinstance(digest, str) and len(digest)==64 and Path(path).is_file()
            and file_sha(path)==digest, 'File digest mismatch: '+str(path))
    return binding(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def identity(row):
    require(type(row['level']) is int and type(row['row_index']) is int, 'Noninteger prompt identity')
    return row['level'], 'pantry_plan' if row['domain']=='pantry' else row['domain'], row['row_index']


def sample_identity(sample):
    draw = sample.get('sample_index', sample.get('draw_index'))
    require(type(draw) is int and 0<=draw<DRAWS, 'Invalid draw index')
    return (*identity(sample), draw)


def unique(records, key, label):
    result = {}
    for row in records:
        k = key(row)
        require(k not in result, 'Duplicate '+label+': '+str(k))
        result[k] = row
    return result


def grade_valid(grade):
    require(type(grade.get('verified')) is bool and grade['verified']==(grade.get('canonical_key') is not None),
            'Every draw requires consistent strict success and canonical-key fields')


def absence_probability(population, marked, draws):
    """Probability of no marked item in an unordered draw without replacement."""
    require(all(type(x) is int for x in (population,marked,draws))
            and 0<=marked<=population and 0<=draws<=population, 'Invalid hypergeometric parameters')
    return 0.0 if population-marked<draws else math.comb(population-marked,draws)/math.comb(population,draws)


def rarefied_distinct(counts, population, draws):
    require(all(type(n) is int and n>0 for n in counts) and sum(counts)<=population, 'Invalid canonical-mode counts')
    return sum(1-absence_probability(population,n,draws) for n in counts)


def uniform_distinct(correct_draws, support_count):
    require(type(correct_draws) is int and correct_draws>=0 and type(support_count) is int and support_count>0,
            'Invalid certified support or correct-draw count')
    if correct_draws==0:
        return 0.0
    if support_count==1:
        return 1.0
    inverse = 1/support_count
    return float(correct_draws) if inverse==0 else -math.expm1(correct_draws*math.log1p(-inverse))/inverse


def correctness_matched_uniform(population, correct, draws, support_count):
    denominator = math.comb(population,draws)
    return sum(math.comb(correct,j)*math.comb(population-correct,draws-j)/denominator
               *uniform_distinct(j,support_count)
               for j in range(max(0,draws-(population-correct)),min(correct,draws)+1))


def validate_reference(reference):
    require(reference.get('support_kind') in ('exact','certified_lower_bound'), 'Uncertified or unsupported support reference')
    require(type(reference.get('support_count')) is int and reference['support_count']>0, 'Invalid support reference count')


def prompt_statistics(row, samples, reference):
    require(len(samples)==DRAWS and {sample_identity(s)[-1] for s in samples}==set(range(DRAWS)),
            'Every prompt must contain all 64 fresh unique slots')
    validate_reference(reference)
    ordered = sorted(samples,key=lambda s:sample_identity(s)[-1])
    for sample in ordered:
        grade_valid(sample)
    keys = [sha(s['canonical_key']) if s['verified'] else None for s in ordered]
    modes = Counter(k for k in keys if k is not None)
    correct = sum(modes.values())
    if reference['support_kind']=='exact':
        require(len(modes)<=reference['support_count'], 'Observed canonical modes exceed certified exact support')
    result = {'level':row['level'], 'domain':row['domain'], 'row_index':row['row_index'], 'row_sha256':sha(row),
              'responses':DRAWS, 'correct_draws':correct, 'failed_draws':DRAWS-correct,
              'mode_counts':sorted(modes.values(),reverse=True), 'support_reference':reference,
              'correct_pairs':math.comb(correct,2), 'colliding_correct_pairs':sum(math.comb(n,2) for n in modes.values()),
              'truncated_draws':sum(s.get('stop_reason',s.get('finish_reason')) in ('length','max_tokens')
                                    or s.get('response_status')=='incomplete' for s in ordered),
              'stop_reason_counts':dict(Counter(str(s.get('stop_reason',s.get('finish_reason','unknown'))) for s in ordered)),
              'rarefaction':{}, 'prefix':{}, 'conditional':{}}
    support = reference['support_count']
    for k in GRID:
        p = 1-absence_probability(DRAWS,correct,k)
        d = rarefied_distinct(list(modes.values()),DRAWS,k)
        u = correctness_matched_uniform(DRAWS,correct,k,support)
        result['rarefaction'][str(k)] = {'pass':p, 'distinct':d, 'breadth':max(0.,d-p),
                                        'uniform_distinct':u, 'distinct_minus_uniform':d-u}
        prefix = {x for x in keys[:k] if x is not None}
        result['prefix'][str(k)] = {'pass':float(bool(prefix)), 'distinct':float(len(prefix)),
                                   'breadth':float(len(prefix)-bool(prefix))}
        result['conditional'][str(k)] = {'eligible':correct>=k,
            'distinct':rarefied_distinct(list(modes.values()),correct,k) if correct>=k else None,
            'uniform_distinct':uniform_distinct(k,support) if correct>=k else None}
    return result


def metric_names():
    names = []
    for kind,metrics in [('rarefaction',('pass','distinct','breadth','uniform_distinct','distinct_minus_uniform')),
                         ('prefix',('pass','distinct','breadth'))]:
        names += [f'{kind}/{metric}/k{k}' for k in GRID for metric in metrics]
    for kind in ('conditional_own','conditional_joint'):
        names += [f'{kind}/{metric}/m{m}' for m in GRID for metric in ('distinct','uniform_distinct')]
    for kind in ('collision_own','collision_joint'):
        names += [kind+'/'+metric for metric in ('observed','uniform_reference','excess_uniform')]
    for kind in ('rarefaction','prefix'):
        names += [f'gain8to64/{kind}/{metric}' for metric in ('pass','distinct','breadth')]
    return tuple(names)


METRICS = metric_names()


def paired_matrices(original, neutral):
    """Per-prompt numerator/denominator channels retain zero and ineligible draws."""
    require([r['row_sha256'] for r in original]==[r['row_sha256'] for r in neutral], 'Paired task identities differ')
    matrices = {}
    for arm,records,peer in [('original',original,neutral),('neutral',neutral,original)]:
        matrix = np.zeros((len(records),len(METRICS),2),dtype=float)
        for i,(r,q) in enumerate(zip(records,peer)):
            require(r['support_reference']==q['support_reference'], 'Support certificate changed across arms')
            channels = {}
            for kind,metrics in [('rarefaction',('pass','distinct','breadth','uniform_distinct','distinct_minus_uniform')),
                                 ('prefix',('pass','distinct','breadth'))]:
                for k in GRID:
                    for metric in metrics:
                        channels[f'{kind}/{metric}/k{k}'] = (r[kind][str(k)][metric],1)
            for kind in ('conditional_own','conditional_joint'):
                for m in GRID:
                    eligible = r['correct_draws']>=m and (kind=='conditional_own' or q['correct_draws']>=m)
                    for metric in ('distinct','uniform_distinct'):
                        channels[f'{kind}/{metric}/m{m}'] = (r['conditional'][str(m)][metric] if eligible else 0,int(eligible))
            for kind in ('collision_own','collision_joint'):
                eligible = r['correct_draws']>=2 and (kind=='collision_own' or q['correct_draws']>=2)
                denominator = r['correct_pairs'] if eligible else 0
                observed = r['colliding_correct_pairs'] if eligible else 0
                reference = denominator/r['support_reference']['support_count']
                for metric,value in [('observed',observed),('uniform_reference',reference),('excess_uniform',observed-reference)]:
                    channels[kind+'/'+metric] = (value,denominator)
            for kind in ('rarefaction','prefix'):
                for metric in ('pass','distinct','breadth'):
                    channels[f'gain8to64/{kind}/{metric}'] = (r[kind]['64'][metric]-r[kind]['8'][metric],1)
            matrix[i] = [channels[name] for name in METRICS]
        matrices[arm] = matrix
    return matrices


def rates(summed):
    with np.errstate(divide='ignore',invalid='ignore'):
        return summed[...,0]/summed[...,1]


def bootstrap_pair(matrices,indices):
    require(matrices['original'].shape==matrices['neutral'].shape and matrices['original'].shape[0]==indices.shape[1],
            'Paired bootstrap dimensions differ')
    points,boots = {},{}
    for arm in ARMS:
        matrix = matrices[arm]
        points[arm] = rates(matrix.sum(axis=0))
        boots[arm] = np.empty((len(indices),len(METRICS)))
        for start in range(0,len(indices),128):
            boots[arm][start:start+128] = rates(matrix[indices[start:start+128]].sum(axis=1))
    points[CONTRAST] = points['neutral']-points['original']
    boots[CONTRAST] = boots['neutral']-boots['original']
    return points,boots


def describe(points,boots):
    result = {}
    for j,name in enumerate(METRICS):
        values = boots[:,j][np.isfinite(boots[:,j])]
        point = float(points[j]) if np.isfinite(points[j]) else None
        ci = np.quantile(values,[.025,.975]).tolist() if point is not None and len(values) else None
        result[name] = {'estimate':point, 'ci95':ci, 'defined_bootstrap_replicates':len(values),
                        'degenerate_ci':bool(ci is not None and ci[0]==ci[1])}
    return result


def summarize_pair(points,boots):
    return {label:describe(points[label],boots[label]) for label in (*ARMS,CONTRAST)}


def seed_mean(points,boots,seed_indices=None):
    """Equal-seed means: undefined registered seeds propagate, never nanmean."""
    output_points,output_boots = {},{}
    require(len(points)==len(boots)>0, 'Empty or mismatched checkpoint panel')
    for label in (*ARMS,CONTRAST):
        output_points[label] = np.mean([p[label] for p in points],axis=0)
        values = np.stack([b[label] for b in boots],axis=1)
        if seed_indices is not None:
            require(seed_indices.shape==values.shape[:2], 'Seed resampling dimensions differ')
            values = values[np.arange(len(values))[:,None],seed_indices]
        output_boots[label] = np.mean(values,axis=1)
    return output_points,output_boots


def pair_counts(original,neutral):
    result = {}
    for arm,records,peer in [('original',original,neutral),('neutral',neutral,original)]:
        item = {key:sum(r[key] for r in records) for key in
                ('responses','correct_draws','failed_draws','truncated_draws','correct_pairs','colliding_correct_pairs')}
        item['prompts'] = len(records)
        item['support_kinds'] = dict(Counter(r['support_reference']['support_kind'] for r in records))
        item['conditional_own_eligible'] = {str(m):sum(r['correct_draws']>=m for r in records) for m in GRID}
        item['conditional_joint_eligible'] = {str(m):sum(r['correct_draws']>=m and q['correct_draws']>=m for r,q in zip(records,peer)) for m in GRID}
        item['joint_correct_pairs'] = sum(r['correct_pairs'] for r,q in zip(records,peer) if q['correct_draws']>=2)
        item['joint_colliding_correct_pairs'] = sum(r['colliding_correct_pairs'] for r,q in zip(records,peer) if q['correct_draws']>=2)
        result[arm] = item
    result['paired_prompt_variation'] = {}
    for kind in ('rarefaction','prefix'):
        for k in GRID:
            for metric in ('pass','distinct','breadth'):
                differences = [b[kind][str(k)][metric]-a[kind][str(k)][metric] for a,b in zip(original,neutral)]
                result['paired_prompt_variation'][f'{kind}/{metric}/k{k}'] = {
                    'zero':sum(abs(v)<1e-12 for v in differences), 'positive':sum(v>1e-12 for v in differences),
                    'negative':sum(v< -1e-12 for v in differences)}
    return result


def load_module(name,path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_design(base):
    base = Path(base).resolve()
    bound_file(base/'manifest.json',ROOT_MANIFEST_SHA)
    manifest = json.loads((base/'manifest.json').read_text())
    require(manifest.get('condition')==CONDITION and manifest.get('sample_count')==DRAWS
            and manifest.get('prompt_count')==96 and manifest.get('arm_count')==2
            and manifest.get('fresh_draws_only') is True, 'Wrong fresh 64-draw experiment')
    required = {'rows.jsonl','prompts.jsonl','selection.json','protocol.json','ANALYSIS_PLAN.md','support_reference.json'}
    require(required<=set(manifest['artifact_sha256']), 'Unbound scientific design files')
    for name,digest in manifest['artifact_sha256'].items():
        bound_file(base/name,digest)
    for name,digest in manifest['code_sha256'].items():
        bound_file(base/'code'/name,digest)
    protocol = json.loads((base/'protocol.json').read_text())
    require(protocol['condition']==CONDITION and protocol['sample_count']==DRAWS
            and protocol['prompts_per_cell']==PROMPTS_PER_CELL and protocol['fresh_draws_only'] is True
            and protocol['old_n8_samples_in_primary'] is False and tuple(protocol['k_grid'])==GRID
            and tuple(protocol['correct_draw_m_grid'])==GRID
            and protocol['bootstrap']['replicates']==20000 and protocol['bootstrap']['seed']==SEED,
            'The prospective sampling or inference contract changed')
    parent = Path(manifest['parent_manifest_path']).resolve()
    bound_file(parent,manifest['parent_manifest_sha256'])
    parent_manifest = json.loads(parent.read_text())
    for name in ('rows.jsonl','prompts.jsonl','selection.json'):
        bound_file(parent.parent/name,parent_manifest['artifact_sha256'][name])
    old_rows = unique(read_jsonl(parent.parent/'rows.jsonl'),identity,'parent row')
    old_prompts = unique(read_jsonl(parent.parent/'prompts.jsonl'),lambda r:(r['arm'],*identity(r)),'parent prompt')
    old_selection = json.loads((parent.parent/'selection.json').read_text())
    selected = {identity(r):r for r in old_selection['candidates'] if r['rank_in_cell']<=PROMPTS_PER_CELL}
    rows = unique(read_jsonl(base/'rows.jsonl'),identity,'discovery row')
    require(set(rows)==set(selected) and all(rows[k]==old_rows[k] for k in rows), 'Rows are not the fixed first16 rank selection')
    require(Counter(k[:2] for k in rows)==Counter({(l,d):PROMPTS_PER_CELL for l in LEVELS for d in DOMAINS}),
            'Missing registered domain-level cell')
    ledger = unique(json.loads((base/'selection.json').read_text())['selected'],identity,'selection entry')
    require(set(ledger)==set(rows), 'Selection ledger differs from rows')
    prompts = unique(read_jsonl(base/'prompts.jsonl'),lambda r:(r['arm'],*identity(r)),'prompt arm')
    require(set(prompts)=={(a,*k) for a in ARMS for k in rows}, 'Missing or extra prompt arm')
    for key,row in rows.items():
        require(ledger[key]['rank_in_cell']==selected[key]['rank_in_cell'] and ledger[key]['row_sha256']==sha(row),
                'Frozen selection rank or row hash changed')
        for arm in ARMS:
            prompt=prompts[(arm,*key)]
            require(prompt['messages']==old_prompts[(arm,*key)]['messages'] and prompt['row_sha256']==sha(row)
                    and prompt['messages_sha256']==sha(prompt['messages']) and prompt['answer_spec_sha256']==sha(row['answer'])
                    and prompt['problem_sha256']==hashlib.sha256(row['problem'].encode()).hexdigest(), 'Fresh campaign prompt or mathematical task changed')
    reference_doc = json.loads((base/'support_reference.json').read_text())
    require(reference_doc['status']=='complete' and reference_doc['outcomes_read'] is False
            and reference_doc['selected_rows']==96, 'Invalid prospective support certificate')
    for source in [reference_doc['discovery_rows_binding'],*reference_doc['source_bindings'].values()]:
        bound_file(source['path'],source['sha256'])
    references = {}
    require(set(reference_doc['references'])=={p['pair_id'] for p in prompts.values()}, 'Support certificate inventory differs')
    for key,row in rows.items():
        ref=reference_doc['references'][prompts[('original',*key)]['pair_id']]
        validate_reference(ref)
        require(identity(ref)==key and ref['row_sha256']==sha(row) and ref['answer_spec_sha256']==sha(row['answer']),
                'Support certificate refers to another problem')
        references[key]=ref
    bound_file(manifest['local_parent_plan_path'],manifest['local_parent_plan_sha256'])
    return {'base':base,'manifest':manifest,'manifest_sha256':file_sha(base/'manifest.json'),'protocol':protocol,
            'rows':rows,'prompts':prompts,'references':references,
            'normalizer_sha256':manifest['code_sha256']['ops/frontier_modebench_normalization.py'],
            'contract_sha256':manifest['code_sha256']['ops/frontier_modebench_contract.py'],
            'sources':{name:binding(base/name) for name in ('manifest.json',*sorted(required))}}


def validate_local_plan(design,plan_path,plan):
    bound_file(plan_path,LOCAL_PLAN_SHA)
    controls={'sample_count':8,'total_sample_count':64,'draw_blocks':8,'seed_base':911640000,'seed_stride':128,
              'max_tokens':192,'temperature':1.0,'top_p':1.0,'syntax_profile':'domain_legal_v1'}
    require(all(plan['settings'].get(k)==v for k,v in controls.items()), 'Local generation controls differ from registration')
    require(Path(plan['rows_path']).resolve()==design['base']/'rows.jsonl'
            and Path(plan['prompts_path']).resolve()==design['base']/'prompts.jsonl', 'Local plan uses another prompt cohort')
    require(plan['input_sha256'].get(str(design['base']/'manifest.json'))==design['manifest_sha256'],
            'Local plan is not bound to the new frozen manifest')
    for path,digest in {**plan['input_sha256'],**plan['code_sha256']}.items():
        bound_file(path,digest)
    for rel,digest in design['manifest']['code_sha256'].items():
        if rel.startswith('src/oat_drgrpo/'):
            bound_file(design['base']/'local/code'/rel,digest)
    parent=json.loads(Path(design['manifest']['local_parent_plan_path']).read_text())
    old=unique(parent['checkpoints'],lambda c:c['label'],'parent checkpoint')
    new=unique(plan['checkpoints'],lambda c:c['label'],'new checkpoint')
    require(len(new)==25 and set(new)==set(old), 'A registered checkpoint was omitted or replaced')
    for label,checkpoint in new.items():
        for field in ('domain','training_method','training_seed','trained_on_level','files'):
            require(checkpoint.get(field)==old[label].get(field), 'Checkpoint identity differs: '+label+'/'+field)
        expected=12288 if checkpoint['training_method']=='initial' else 4096
        require(checkpoint['expected_draws']==expected, 'Wrong fresh draw allocation')
    return plan


def expected_local_seed(level,domain,row_index,draw):
    require(type(draw) is int and 0<=draw<DRAWS, 'Invalid local draw slot')
    request=911640000+128*((level-2)*10000+DOMAINS.index(domain)*1000+row_index)+8*(draw//8)
    return request,request+draw%8


def validate_local_draw(sample,key,row,prompt,checkpoint):
    arm,level,domain,index,draw=key
    grade_valid(sample)
    require(sample['checkpoint_label']==checkpoint['label'] and
            all(sample[field]==checkpoint[field] for field in ('training_method','training_seed','trained_on_level')),
            'Draw belongs to another checkpoint')
    require(sample['row_sha256']==sha(row) and sample['messages_sha256']==prompt['messages_sha256']
            and sample['pair_id']==prompt['pair_id'], 'Draw task or prompt hash differs')
    request,child=expected_local_seed(level,domain,index,draw)
    require(sample.get('draw_index')==draw and type(sample.get('token_count')) is int and 0<=sample['token_count']<=192
            and isinstance(sample.get('text'),str), 'Local draw index, text, or token limit differs')
    require(sample['draw_block']==draw//8 and sample['block_draw_index']==draw%8
            and sample['sampling_seed']==request and sample['child_sampling_seed']==child,
            'Draw does not follow the frozen disjoint n8 child-seed schedule')


def local_samples(design,plan_path,checkpoint):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    validate_local_plan(design,plan_path,plan)
    directory=Path(plan['output_root'])/checkpoint['label']
    require(directory.resolve().is_relative_to(design['base']), 'Local result directory lies outside the fresh experiment')
    result=json.loads((directory/'result.json').read_text())
    expected_identity={'schema':plan['schema'],'plan_sha256':file_sha(plan_path),'checkpoint':checkpoint,'settings':plan['settings']}
    require(result['identity']==expected_identity and result['identity_sha256']==sha(expected_identity)
            and result['status']=='complete', 'Result identity differs from frozen local plan')
    from datetime import datetime
    runtime=json.loads((directory/'runtime.json').read_text())
    require(runtime['identity_sha256']==sha(expected_identity)
            and datetime.fromisoformat(runtime['generated_at'])>=datetime.fromisoformat(plan['created_at'])
            >=datetime.fromisoformat(design['manifest']['prepared_at_utc']), 'Runtime receipt predates the fresh frozen plan')
    require(Path(result['responses_path']).resolve()==directory.resolve()/'responses.jsonl','Result points to another response cohort')
    bound_file(directory/'responses.jsonl',result['responses_sha256'])
    domain='pantry_plan' if checkpoint['domain']=='pantry' else checkpoint['domain']
    rows={k:r for k,r in design['rows'].items() if domain is None or k[1]==domain}
    expected={(a,*k,i) for a in ARMS for k in rows for i in range(DRAWS)}
    raw=unique(read_jsonl(directory/'responses.jsonl'),lambda s:(s['arm'],*sample_identity(s)),'local response')
    require(set(raw)==expected and result['draws']==len(expected)==checkpoint['expected_draws'], 'Incomplete or unexpected fresh64 slots')
    rebuilt={}
    for key,sample in raw.items():
        require(sample.get('schema')==plan['schema'],'Raw response belongs to another collection schema')
        validate_local_draw(sample,key,rows[key[1:4]],design['prompts'][key[:4]],checkpoint)
        rebuilt[key]={**sample,'domain':key[2],'sample_index':key[-1],
                      'stop_reason':sample.get('finish_reason','unknown'),'graded_text':sample['text']}
    groups=unique(result['prompt_results'],lambda g:(g['draw_block'],g['arm'],*identity(g)),'n8 generation block')
    require(set(groups)=={(block,a,*k) for block in range(8) for a in ARMS for k in rows}, 'Missing or duplicate n8 block receipts')
    for key,group in groups.items():
        block,arm,level,domain,index=key
        require(len(group['attempts'])==8 and group['sampling_seed']==expected_local_seed(level,domain,index,8*block)[0],
                'Block receipt has wrong request seed or attempt count')
        for offset,attempt in enumerate(group['attempts']):
            require(all(raw[(arm,level,domain,index,8*block+offset)].get(field)==value for field,value in attempt.items()),
                    'Exported response differs from original n8 attempt receipt')
    batches={}
    for path in sorted(directory.glob('batch_*.json')):
        batch=json.loads(path.read_text())
        require(batch['identity_sha256']==sha(expected_identity) and batch['records_sha256']==sha(batch['records']),
                'Original batch receipt identity or content hash differs')
        for group in batch['records']:
            key=(group['draw_block'],group['arm'],*identity(group))
            require(key not in batches,'Duplicate n8 group across original batch receipts')
            batches[key]=group
    require(batches==groups, 'Final result lacks exact original durable n8 batch receipts')
    return directory.resolve(),rows,rebuilt,result


def frozen_grade_modules(root):
    for name in sys.modules:
        require(name!='frontier_modebench_contract' and not name.startswith('oat_drgrpo'),
                'Offline grading requires an isolated process with no preloaded grader')
    sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'ops'))
    import frontier_modebench_contract as contract
    normalizer=load_module('_discovery_frozen_normalizer',root/'ops/frontier_modebench_normalization.py')
    return contract,normalizer


def warm_frozen_python(contract):
    from oat_drgrpo.python_modebench_process import _SHARED_VERIFIER
    import time
    attempts = []
    for _ in range(3):
        _SHARED_VERIFIER._start()
        time.sleep(1.25)
        result = contract.grade_response(2,'python_factors',
            {'answer':{'verifier':'python_factor_function','python_version':'factor-v1','cases':[6,8]}},
            r'\boxed{lambda n: 2}')
        attempts.append(result['verified'])
        if attempts[-1]: break
    require(attempts[-1], 'Frozen serial Python verifier failed its synthetic warmup')
    return attempts


def freeze_source(directory):
    import shutil
    destination=Path(directory)/'analysis_code';destination.mkdir(exist_ok=True)
    source=Path(__file__).resolve();target=destination/('analyze_modebench_discovery_curves_'+file_sha(source)[:16]+'.py')
    if target.exists():bound_file(target,file_sha(source))
    else:shutil.copy2(source,target)
    return binding(target)


def grade_local(base,plan_path,label):
    design=authenticate_design(base);plan=json.loads(Path(plan_path).read_text())
    checkpoints=[c for c in plan['checkpoints'] if c['label']==label]
    require(len(checkpoints)==1,'Checkpoint is not uniquely registered')
    directory,rows,raw,result=local_samples(design,plan_path,checkpoints[0])
    audit_path=directory/'discovery_grading_audit.json';output=directory/'discovery_grades.jsonl'
    if audit_path.exists():
        authenticate_grades(design,directory,raw)
        return json.loads(audit_path.read_text())
    require(not output.exists(),'Unfinished exported grading cache exists; preserve it before recovery')
    contract,normalizer=frozen_grade_modules(design['base']/'code')
    python_count=sum(key[2]=='python_factors' for key in raw)
    warmup=warm_frozen_python(contract) if python_count else []
    source=freeze_source(directory)
    chunk_dir=directory/'discovery_grading_chunks';chunk_dir.mkdir(exist_ok=True)
    chunk_identity={'result_sha256':file_sha(directory/'result.json'),'responses_sha256':file_sha(directory/'responses.jsonl'),
                    'normalizer_sha256':design['normalizer_sha256'],'contract_sha256':design['contract_sha256'],
                    'analyzer_sha256':source['sha256']}
    ordered=sorted(raw.items());entries=[];chunks=[]
    for start in range(0,len(ordered),256):
        chunk_path=chunk_dir/f'part_{start:06d}.json'
        if chunk_path.exists():
            chunk=json.loads(chunk_path.read_text())
            require(chunk['identity']==chunk_identity and chunk['start']==start and chunk['entries_sha256']==sha(chunk['entries']),
                    'Resumable grading chunk changed source or raw cohort')
            require(len(chunk['entries'])==len(ordered[start:start+256]),'Incomplete grading chunk')
        else:
            records=[]
            for key,sample in ordered[start:start+256]:
                row=rows[key[1:4]]
                strict={field:sample[field] for field in ('verified','canonical_key','graded_text')}
                if key[2]=='python_factors':strict=contract.grade_response(key[1],key[2],row,sample['text'])
                grade_valid(strict)
                normalized=normalizer.normalize_and_grade(row,sample['text'],strict_grade=strict,grader=contract.grade_response)
                grade_valid(normalized)
                require(not strict['verified'] or (normalized['verified'] and normalized['canonical_key']==strict['canonical_key']),
                        'Frozen normalization changed a strict success or key')
                records.append({'arm':key[0],'level':key[1],'domain':key[2],'row_index':key[3],'sample_index':key[4],
                                'raw_sample_sha256':sha(sample),'strict':strict,'normalization':normalized})
            chunk={'identity':chunk_identity,'start':start,'entries':records,'entries_sha256':sha(records),
                   'python_synthetic_warmup':warmup,'python_rechecked_serially':sum(k[2]=='python_factors' for k,s in ordered[start:start+256])}
            temporary=chunk_path.with_suffix('.tmp');write_json(temporary,chunk);temporary.replace(chunk_path)
        for (key,sample),entry in zip(ordered[start:start+256],chunk['entries']):
            require((entry['arm'],*sample_identity(entry))==key and entry['raw_sample_sha256']==sha(sample), 'Grading chunk belongs to another draw')
        require(chunk['python_rechecked_serially']==sum(k[2]=='python_factors' for k,s in ordered[start:start+256])
                and (not chunk['python_rechecked_serially'] or chunk['python_synthetic_warmup'][-1:] == [True]),
                'Resumed Python grading chunk was not serially rechecked after warm-up')
        entries.extend(chunk['entries']);chunks.append(binding(chunk_path))
    temporary=output.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(e,sort_keys=True,allow_nan=False)+'\n' for e in entries));temporary.replace(output)
    changed=sum(any(e['strict'].get(f)!=s.get(f) for f in ('verified','canonical_key','graded_text')) for e,(k,s) in zip(entries,ordered))
    audit={'schema':'modebench-discovery-local-grading-v1','status':'complete','api_calls':0,'records':len(raw),
           **chunk_identity,'cache_sha256':file_sha(output),'analyzer_source':source,'chunks':chunks,
           'python_synthetic_warmup':warmup,'python_rechecked_serially':python_count,'strict_changed_records':changed,
           'raw_verified':sum(s['verified'] for s in raw.values()),'strict_verified':sum(e['strict']['verified'] for e in entries),
           'normalized_verified':sum(e['normalization']['verified'] for e in entries)}
    write_json(audit_path,audit)
    authenticate_grades(design,directory,raw)
    return audit


def authenticate_grades(design,directory,raw):
    audit=json.loads((directory/'discovery_grading_audit.json').read_text())
    require(audit['status']=='complete' and audit['api_calls']==0 and audit['records']==len(raw)
            and audit['normalizer_sha256']==design['normalizer_sha256'] and audit['contract_sha256']==design['contract_sha256'],
            'Wrong grading cache inventory or frozen grader')
    for name,field in [('result.json','result_sha256'),('responses.jsonl','responses_sha256'),('discovery_grades.jsonl','cache_sha256')]:
        bound_file(directory/name,audit[field])
    bound_file(audit['analyzer_source']['path'],audit['analyzer_source']['sha256'])
    entries=unique(read_jsonl(directory/'discovery_grades.jsonl'),lambda e:(e['arm'],*sample_identity(e)),'graded response')
    require(set(entries)==set(raw),'Grading cache does not contain exactly all registered responses')
    strict,normalized={},{}
    for key,sample in raw.items():
        entry=entries[key];require(entry['raw_sample_sha256']==sha(sample),'Grading cache belongs to another response')
        a,b=entry['strict'],entry['normalization'];grade_valid(a);grade_valid(b)
        if key[2]!='python_factors':require(all(a.get(f)==sample.get(f) for f in ('verified','canonical_key','graded_text')),'Python regrade changed another domain')
        require(b['original_text']==sample['text'] and (not a['verified'] or (b['verified'] and b['canonical_key']==a['canonical_key'])),
                'Normalizer changed raw text or a strict success')
        strict[key]={**sample,**a};normalized[key]={**sample,**b}
    python_count=sum(k[2]=='python_factors' for k in raw)
    require(audit['python_rechecked_serially']==python_count and (not python_count or audit['python_synthetic_warmup'][-1:]==[True]),
            'Python recheck was incomplete or not warmed')
    for name,data in [('raw_verified',raw),('strict_verified',strict),('normalized_verified',normalized)]:
        require(audit[name]==sum(s['verified'] for s in data.values()),'Grading audit total differs: '+name)
    require(audit['strict_changed_records']==sum(any(strict[k].get(f)!=s.get(f) for f in ('verified','canonical_key','graded_text')) for k,s in raw.items()),
            'Strict grading correction count differs')
    chunk_entries=[]
    require(audit['analyzer_sha256']==audit['analyzer_source']['sha256'],'Grading audit source bindings disagree')
    chunk_identity={key:audit[key] for key in ('result_sha256','responses_sha256','normalizer_sha256','contract_sha256','analyzer_sha256')}
    for item in audit['chunks']:
        require(Path(item['path']).resolve().is_relative_to((directory/'discovery_grading_chunks').resolve()),'Grading chunk lies outside the cohort')
        bound_file(item['path'],item['sha256']);chunk=json.loads(Path(item['path']).read_text())
        require(chunk['entries_sha256']==sha(chunk['entries']) and chunk['identity']==chunk_identity
                and chunk['start']==len(chunk_entries),'Grading chunk identity, order, or entry hash differs')
        count=sum(e['domain']=='python_factors' for e in chunk['entries'])
        require(chunk['python_rechecked_serially']==count and (not count or chunk['python_synthetic_warmup'][-1:]==[True]),
                'Cached chunk lacks a successful serial Python warm-up receipt')
        chunk_entries.extend(chunk['entries'])
    require(unique(chunk_entries,lambda e:(e['arm'],*sample_identity(e)),'chunk grade')==entries,'Grade export differs from resumable chunk receipts')
    return strict,normalized


def native_inventory(run, expected):
    """Use the exact new native adapter while preserving the existing auditor."""
    auditor_path=Path(__file__).resolve().parent/'audit_hosted_modebench_completion.py'
    native=load_module('_discovery_native_auditor',auditor_path)
    native.RUNNERS = dict(native.RUNNERS)
    native.RUNNERS['frontier-modebench-native-chat-responses-v1'] = 'ops/evaluate_native_prompt_ablation.py'
    return native, native.load_inventory(run, expected_samples=expected)

def validate_run_inputs(design, entry):
    from datetime import datetime
    run = Path(entry['run_dir']).resolve()
    arm = entry['arm']
    require(arm in ARMS and entry['family'] == 'frontier', 'Unregistered hosted family or arm')
    bound_file(run / 'manifest.json', entry['manifest_sha256'])
    manifest = json.loads((run / 'manifest.json').read_text())
    require(manifest.get('experiment_condition') == CONDITION and manifest.get('prompt_arm') == arm
            and manifest.get('ablation_manifest_sha256') == design['manifest_sha256']
            and manifest.get('fresh_response_cohort') is True and manifest.get('sample_count') == DRAWS,
            'Require a fresh registered arm; historical controls are inadmissible')
    require(manifest['model'] == entry['model'], 'Registry deployment differs from run')
    expected = len(design['rows']) * DRAWS
    native, inventory = native_inventory(run, expected)
    require(inventory['rows'] == design['rows'], 'Run rows differ from frozen identical tasks')
    requests = unique(inventory['requests'], sample_identity, 'run request')
    expected_keys = {(*key, i) for key in design['rows'] for i in range(DRAWS)}
    require(set(requests) == expected_keys, 'Run request slot inventory differs')
    for key, item in requests.items():
        messages = item['request'].get('input', item['request'].get('messages'))
        require(messages == design['prompts'][(arm, *key[:3])]['messages'], 'Requested prompt changed')
        require(item.get('prompt_arm') == arm and item.get('experiment_condition') == CONDITION,
                'Request lacks fresh arm identity')
    require(manifest['code_sha256']['ops/frontier_modebench_normalization.py'] == design['normalizer_sha256']
            and manifest['code_sha256']['ops/frontier_modebench_contract.py'] == design['contract_sha256'],
            'Frozen normalizer or contract differs across conditions')
    raw = unique(read_jsonl(run / 'samples.jsonl'), sample_identity, 'sample')
    require(set(raw) == expected_keys, 'Incomplete or unexpected 64-draw slots; no inferential report')
    provider_ids = [(s['response_id'], s.get('choice_index', 0)) for s in raw.values()]
    require(len(set(provider_ids)) == expected, 'A native response/choice was reused')
    raw_bodies = native.validate_native_records(inventory, list(raw.values()))
    prepared = datetime.fromisoformat(manifest['prepared_at_utc'].replace('Z', '+00:00'))
    for sample in raw.values():
        grade_valid(sample)
        atomic = run / 'sample_receipts' / (sample['sample_id'] + '.json')
        require(json.loads(atomic.read_text()) == sample, 'Atomic receipt differs from exported sample')
        receipt = raw_bodies[sample['raw_receipt']]
        started = datetime.fromisoformat(receipt['started_at_utc'].replace('Z', '+00:00'))
        require(started >= prepared, 'Historical response predates the registered fresh run')
    return native, inventory, requests, raw

def validate_payload_pair(original, neutral):
    require(original['directory'] != neutral['directory'], 'Both arms point to the same cohort')
    require(set(original['requests']) == set(neutral['requests']), 'Paired request identities differ')
    for key, first in original['requests'].items():
        second = neutral['requests'][key]
        without_prompt = lambda r: {k: v for k,v in r['request'].items() if k not in ('input', 'messages')}
        require(without_prompt(first) == without_prompt(second), 'A non-prompt generation control changed')
        require(first['row_sha256'] == second['row_sha256'], 'Mathematical task differs across arms')
    require(not {s['response_id'] for s in original['raw'].values()} &
                {s['response_id'] for s in neutral['raw'].values()}, 'A response was reused across arms')
    # Runtime grading code is the same; collector adapters can differ by provider.
    graders = lambda r: {k:v for k,v in r['manifest']['code_sha256'].items()
                         if k.startswith('src/oat_drgrpo/') or k == 'ops/frontier_modebench_contract.py'}
    require(graders(original) == graders(neutral), 'Executable graders differ across arms')

def grade_hosted(base, run_dir):
    """Serial frozen Python recheck plus one previously frozen normalizer."""
    design = authenticate_design(base)
    registry = json.loads((Path(base) / 'hosted_analysis_runs.json').read_text())
    entries = [e for e in registry['runs'] if Path(e['run_dir']).resolve() == Path(run_dir).resolve()]
    require(len(entries) == 1, 'Run not uniquely registered')
    entry = entries[0]
    native, inventory, requests, raw = validate_run_inputs(design, entry)
    run = Path(run_dir).resolve()
    output = run / 'discovery_hosted_grades.jsonl'
    audit_path = run / 'discovery_hosted_grading_audit.json'
    if audit_path.exists():
        authenticate_hosted_grades(design, run, raw)
        return json.loads(audit_path.read_text())
    require(not output.exists(), 'Unfinished grading output exists; preserve it before regeneration')
    completion = native.audit(run,expected_samples=len(raw))
    contract, normalizer = frozen_grade_modules(run / 'code')
    warmup = warm_frozen_python(contract)
    entries, changed = [], 0
    for key, sample in sorted(raw.items()):
        row = design['rows'][key[:3]]
        strict = {k: sample[k] for k in ('verified', 'canonical_key', 'graded_text')}
        if key[1] == 'python_factors':
            strict = contract.grade_response(key[0], key[1], row, sample['text'])
        grade_valid(strict)
        changed += any(strict.get(k) != sample.get(k) for k in ('verified', 'canonical_key', 'graded_text'))
        normalized = normalizer.normalize_and_grade(row, sample['text'], strict_grade=strict, grader=contract.grade_response)
        grade_valid(normalized)
        require(not strict['verified'] or (normalized['verified'] and normalized['canonical_key'] == strict['canonical_key']),
                'Normalization changed a strict success')
        entries.append({**{k:sample[k] for k in ('level','domain','row_index','sample_index')},
                        'raw_sample_sha256': sha(sample), 'strict': strict, 'normalization': normalized})
    temporary = output.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(e, sort_keys=True, allow_nan=False)+'\n' for e in entries))
    temporary.replace(output)
    source = freeze_source(run)
    audit = {'schema': 'modebench-discovery-hosted-grades-v1', 'status': 'complete', 'api_calls': 0,
             'records': len(entries), 'manifest_sha256': file_sha(run/'manifest.json'),
             'completion_audit_sha256':file_sha(run/'completion_audit.json'),
             'evidence_inventory_sha256':file_sha(run/'evidence_file_sha256.json'),
             'native_adapter':'ops/evaluate_native_prompt_ablation.py',
             'physical_attempts':completion['raw_attempts'],
             'attempt_status_counts':completion['attempt_status_counts'],
             'raw_samples_sha256': file_sha(run/'samples.jsonl'), 'rows_sha256': file_sha(run/'rows.jsonl'),
             'cache_sha256': file_sha(output), 'normalizer_sha256': design['normalizer_sha256'],
             'contract_sha256': design['contract_sha256'], 'analyzer_source': source,
             'python_rechecked_serially': sum(k[1]=='python_factors' for k in raw),
             'strict_changed_records': changed, 'python_synthetic_warmup': warmup,
             'raw_verified': sum(s['verified'] for s in raw.values()),
             'strict_verified': sum(e['strict']['verified'] for e in entries),
             'normalized_verified': sum(e['normalization']['verified'] for e in entries),
             'grader_code_sha256': {k:v for k,v in inventory['manifest']['code_sha256'].items()
                                    if k.startswith('src/oat_drgrpo/') or k=='ops/frontier_modebench_contract.py'}}
    write_json(audit_path, audit)
    return audit

def authenticate_hosted_grades(design, run, raw):
    audit = json.loads((run / 'discovery_hosted_grading_audit.json').read_text())
    require(audit.get('status') == 'complete' and audit.get('records') == len(raw)
            and audit.get('normalizer_sha256') == design['normalizer_sha256']
            and audit.get('contract_sha256') == design['contract_sha256'], 'Incomplete or changed frozen grading procedure')
    for name, key in [('manifest.json','manifest_sha256'), ('rows.jsonl','rows_sha256'),
                      ('samples.jsonl','raw_samples_sha256'), ('discovery_hosted_grades.jsonl','cache_sha256')]:
        bound_file(run/name, audit[key])
    bound_file(audit['analyzer_source']['path'], audit['analyzer_source']['sha256'])
    bound_file(run/'completion_audit.json',audit['completion_audit_sha256'])
    bound_file(run/'evidence_file_sha256.json',audit['evidence_inventory_sha256'])
    completion = json.loads((run/'completion_audit.json').read_text())
    evidence = json.loads((run/'evidence_file_sha256.json').read_text())
    require(completion['status']=='pass' and completion['expected_responses']==len(raw)
            and completion['saved_samples']==len(raw) and completion['evidence_inventory_sha256']==sha(evidence),
            'Native completion/attempt audit incomplete or changed')
    for name,digest in evidence.items():
        bound_file(run/name,digest)
    manifest = json.loads((run/'manifest.json').read_text())
    require(audit['grader_code_sha256']=={k:v for k,v in manifest['code_sha256'].items()
            if k.startswith('src/oat_drgrpo/') or k=='ops/frontier_modebench_contract.py'},
            'Serial grading audit used different executable grader sources')
    records = unique(read_jsonl(run/'discovery_hosted_grades.jsonl'), sample_identity, 'grade cache')
    require(set(records) == set(raw), 'Grading cache does not cover every sampled slot')
    strict, normalized = {}, {}
    for key, record in records.items():
        sample = raw[key]
        require(record['raw_sample_sha256'] == sha(sample), 'Grade cache belongs to another raw response')
        for field in ('strict', 'normalization'):
            grade_valid(record[field])
        a, b = record['strict'], record['normalization']
        require(b.get('original_text') == sample['text'], 'Normalization original text changed')
        require(not a['verified'] or (b['verified'] and b['canonical_key'] == a['canonical_key']),
                'Normalization changed a strict success')
        if key[1] != 'python_factors':
            require(all(a.get(k) == sample.get(k) for k in ('verified','canonical_key','graded_text')),
                    'Serial Python audit changed another domain')
        strict[key], normalized[key] = {**sample, **a}, {**sample, **b}
    require(audit.get('python_rechecked_serially')==sum(k[1]=='python_factors' for k in raw)
            and audit.get('python_synthetic_warmup',[])[-1:] == [True], 'Incomplete warmed Python recheck')
    require(audit['strict_verified'] == sum(s['verified'] for s in strict.values())
            and audit['normalized_verified'] == sum(s['verified'] for s in normalized.values())
            and audit['raw_verified']==sum(s['verified'] for s in raw.values())
            and audit['strict_changed_records']==sum(any(strict[k].get(f)!=sample.get(f) for f in ('verified','canonical_key','graded_text')) for k,sample in raw.items()), 'Grading totals differ')
    return strict, normalized

def authenticate_hosted(design, entry):
    native, inventory, requests, raw = validate_run_inputs(design, entry)
    run = Path(entry['run_dir']).resolve()
    strict, normalized = authenticate_hosted_grades(design, run, raw)
    return {'directory': run, 'model_id': entry['model_id'], 'family': 'frontier',
            'manifest': inventory['manifest'], 'rows': inventory['rows'], 'requests': requests,
            'raw': raw, 'strict': strict, 'normalized_secondary': normalized,
            'grading_audit':json.loads((run/'discovery_hosted_grading_audit.json').read_text()),
            'sources': {name: binding(run/name) for name in ('manifest.json','rows.jsonl','requests.jsonl',
                       'samples.jsonl','discovery_hosted_grades.jsonl','discovery_hosted_grading_audit.json',
                       'completion_audit.json','evidence_file_sha256.json')}}


def authenticate_local(design,plan_path,checkpoint):
    directory,rows,raw,result=local_samples(design,plan_path,checkpoint)
    strict,normalized=authenticate_grades(design,directory,raw)
    return [{'directory':directory,'model_id':checkpoint['label'],'family':'local','checkpoint':checkpoint,'rows':rows,
             'grading_audit':json.loads((directory/'discovery_grading_audit.json').read_text()),
             'strict':{k[1:]:s for k,s in strict.items() if k[0]==arm},
             'normalized_secondary':{k[1:]:s for k,s in normalized.items() if k[0]==arm},
             'sources':{name:binding(directory/name) for name in
                        ('result.json','responses.jsonl','runtime.json','discovery_grades.jsonl','discovery_grading_audit.json')}} for arm in ARMS]


def analyze_pair(design,original,neutral,indices):
    require(original['rows']==neutral['rows'] and original['model_id']==neutral['model_id'], 'Paired runs differ in task or model')
    analyses={}
    for grading in GRADINGS:
        records={arm:[prompt_statistics(row,[run[grading][(*key,i)] for i in range(DRAWS)],design['references'][key])
                      for key,row in sorted(run['rows'].items())]
                 for arm,run in [('original',original),('neutral',neutral)]}
        cells={}
        for level,domain in sorted({key[:2] for key in original['rows']}):
            subset={arm:[r for r in records[arm] if (r['level'],r['domain'])==(level,domain)] for arm in ARMS}
            point,boot=bootstrap_pair(paired_matrices(subset['original'],subset['neutral']),indices[level,domain])
            cells[f'level{level}/{domain}']={**summarize_pair(point,boot),'counts':pair_counts(subset['original'],subset['neutral'])}
            del point,boot
        analyses[grading]={'cells':cells,'prompts':records}
    return {'model_id':original['model_id'],'family':original['family'],'checkpoint':original.get('checkpoint'),
            'conditions':{arm:{'directory':str(run['directory']),'sources':run['sources']} for arm,run in [('original',original),('neutral',neutral)]},
            'grading_audits':{arm:run['grading_audit'] for arm,run in [('original',original),('neutral',neutral)]},'analyses':analyses}


def sum_seed_counts(per_seed_counts):
    result={}
    for arm in ARMS:
        rows=[c[arm] for c in per_seed_counts]
        summed={key:sum(r[key] for r in rows) for key in ('prompts','responses','correct_draws','failed_draws','truncated_draws',
                'correct_pairs','colliding_correct_pairs','joint_correct_pairs','joint_colliding_correct_pairs')}
        for kind in ('conditional_own_eligible','conditional_joint_eligible'):
            summed[kind]={str(m):sum(r[kind][str(m)] for r in rows) for m in GRID}
        summed['pooled_collision_across_checkpoints']=summed['colliding_correct_pairs']/summed['correct_pairs'] if summed['correct_pairs'] else None
        summed['pooled_joint_collision_across_checkpoints']=summed['joint_colliding_correct_pairs']/summed['joint_correct_pairs'] if summed['joint_correct_pairs'] else None
        summed['support_kinds']=dict(sum((Counter(r['support_kinds']) for r in rows),Counter()))
        result[arm]=summed
    result['interpretation']='Summed descriptive checkpoint-prompt and pair counts; pooled_collision_across_checkpoints is separate from primary equal-seed mean collision. Repeated problems across checkpoints are not independent new problems.'
    return result


def combine_local_seeds(models,indices,replicates):
    groups={}
    for model in models:
        ck=model.get('checkpoint') or {}
        if model['family']=='local' and ck.get('training_method')!='initial':
            domain='pantry_plan' if ck['domain']=='pantry' else ck['domain']
            groups.setdefault(ck['training_method']+'/'+domain,[]).append(model)
    output={}
    for key,members in sorted(groups.items()):
        members.sort(key=lambda x:x['checkpoint']['training_seed'])
        seeds=[x['checkpoint']['training_seed'] for x in members];domain=key.split('/')[1]
        require(seeds==([43,46] if domain=='pantry_plan' else list(range(43,48))), 'Missing or selected registered training seeds')
        seed_indices=np.random.default_rng(SEED).integers(0,len(seeds),(replicates,len(seeds)))
        analyses={}
        for grading in GRADINGS:
            cells={}
            for level in LEVELS:
                cell=f'level{level}/{domain}';points=[];boots=[];per_seed={};counts=[]
                for model in members:
                    records=model['analyses'][grading]['prompts']
                    rs={arm:[r for r in records[arm] if r['level']==level] for arm in ARMS}
                    point,boot=bootstrap_pair(paired_matrices(rs['original'],rs['neutral']),indices[level,domain])
                    points.append(point);boots.append(boot)
                    per_seed[str(model['checkpoint']['training_seed'])]=model['analyses'][grading]['cells'][cell]
                    counts.append(pair_counts(rs['original'],rs['neutral']))
                fixed_points,fixed_boots=seed_mean(points,boots)
                fixed=summarize_pair(fixed_points,fixed_boots);del fixed_points,fixed_boots
                hierarchical=None
                if len(seeds)>=5:
                    hp,hb=seed_mean(points,boots,seed_indices);hierarchical=summarize_pair(hp,hb);del hp,hb
                ranges={name:[min(p[CONTRAST][j] for p in points),max(p[CONTRAST][j] for p in points)]
                        for j,name in enumerate(METRICS) if all(np.isfinite(p[CONTRAST][j]) for p in points)}
                cells[cell]={'seed_mean_fixed_seed_prompt_ci':fixed,'hierarchical_seed_and_prompt_ci':hierarchical,
                             'per_seed':per_seed,'contrast_seed_range':ranges,'counts':sum_seed_counts(counts)}
                del points,boots,point,boot
            analyses[grading]={'cells':cells}
        output[key]={'training_method':members[0]['checkpoint']['training_method'],'domain':domain,
                     'training_seeds':seeds,'seed_count':len(seeds),'checkpoint_ids':[x['model_id'] for x in members],
                     'metric_weighting':'Equal checkpoint/seed means, including conditional ratios; undefined seeds propagate.', 'analyses':analyses}
    return output


def validate_hosted_registry(design,path):
    bound_file(path,'447fd260c2e769fce1c4a9877c6d53684744c5ce3538cc3235c50dab846255f9')
    registry=json.loads(Path(path).read_text())
    require(registry['experiment_condition']==CONDITION and registry['ablation_manifest_sha256']==design['manifest_sha256'],
            'Hosted registry belongs to another experiment')
    entries=unique(registry['runs'],lambda e:(e['model_id'],e['arm']),'hosted arm')
    require(set(entries)=={(model,arm) for model in ('gpt56sol','gpt54','grok43') for arm in ARMS}, 'Missing registered hosted model or fresh arm')
    return registry


def inventory_report(design,local_plan=None,hosted_registry=None):
    report={'status':'complete','expected_draws':0,'finalized_draws':0,'durable_draws':0,'runs':[]}
    if local_plan is not None:
        plan=json.loads(Path(local_plan).read_text());validate_local_plan(design,local_plan,plan)
        for ck in plan['checkpoints']:
            directory=Path(plan['output_root'])/ck['label'];path=directory/'responses.jsonl'
            raw=read_jsonl(path) if path.exists() else []
            keys=unique(raw,lambda s:(s['arm'],*sample_identity(s)),'inventory draw')
            domain='pantry_plan' if ck['domain']=='pantry' else ck['domain']
            rows={k:r for k,r in design['rows'].items() if domain is None or k[1]==domain}
            expected={(a,*k,i) for a in ARMS for k in rows for i in range(DRAWS)}
            require(set(keys)<=expected,'Unexpected raw draw in local inventory')
            durable=0
            for p in directory.glob('batch_*.json'):
                batch=json.loads(p.read_text());require(batch['records_sha256']==sha(batch['records']),'Partial batch digest differs')
                durable+=sum(len(r['attempts']) for r in batch['records'])
            report['runs'].append({'model_id':ck['label'],'family':'local','directory':str(directory),'expected_draws':len(expected),
                                  'finalized_draws':len(raw),'durable_draws':max(len(raw),durable),
                                  'complete':set(keys)==expected and (directory/'result.json').exists(),
                                  'graded':(directory/'discovery_grading_audit.json').exists()})
    if hosted_registry is not None:
        registry=validate_hosted_registry(design,hosted_registry)
        expected={(*k,i) for k in design['rows'] for i in range(DRAWS)}
        for entry in registry['runs']:
            directory=Path(entry['run_dir']);path=directory/'samples.jsonl';raw=read_jsonl(path) if path.exists() else []
            keys=unique(raw,sample_identity,'hosted inventory draw');require(set(keys)<=expected,'Unexpected hosted raw draw')
            report['runs'].append({'model_id':entry['model_id'],'family':'frontier','arm':entry['arm'],'directory':str(directory),
                                  'expected_draws':len(expected),'finalized_draws':len(raw),'durable_draws':len(raw),
                                  'complete':set(keys)==expected,'graded':(directory/'discovery_hosted_grading_audit.json').exists()})
    for run in report['runs']:
        for key in ('expected_draws','finalized_draws','durable_draws'):report[key]+=run[key]
        if not run['complete']:report['status']='incomplete'
    return report


def build_report(base,local_plan=None,hosted_registry=None,replicates=20000,scope='full'):
    require(replicates==20000,'Publication requires all 20,000 registered bootstrap replicates')
    require(scope in ('full','local','hosted'),'Unknown panel scope')
    design=authenticate_design(base);base=design['base']
    local_plan=(Path(local_plan) if local_plan else base/'local/plan.json') if scope!='hosted' else None
    hosted_registry=(Path(hosted_registry) if hosted_registry else base/'hosted_analysis_runs.json') if scope!='local' else None
    inventory=inventory_report(design,local_plan,hosted_registry)
    require(inventory['runs'] and inventory['status']=='complete' and all(r['graded'] for r in inventory['runs']),
            'Every registered model/cell/arm/draw and grading audit in the selected scope must be complete')
    rng=np.random.default_rng(SEED)
    indices={(l,d):rng.integers(0,PROMPTS_PER_CELL,(replicates,PROMPTS_PER_CELL)) for l in LEVELS for d in DOMAINS}
    models=[];registries={}
    if local_plan is not None:
        plan=json.loads(local_plan.read_text());registries['local']=binding(local_plan)
        for checkpoint in plan['checkpoints']:
            original,neutral=authenticate_local(design,local_plan,checkpoint)
            models.append(analyze_pair(design,original,neutral,indices))
    if hosted_registry is not None:
        registry=validate_hosted_registry(design,hosted_registry);registries['hosted']=binding(hosted_registry)
        entries={(e['model_id'],e['arm']):e for e in registry['runs']}
        for model in ('gpt56sol','gpt54','grok43'):
            original=authenticate_hosted(design,entries[model,'original']);neutral=authenticate_hosted(design,entries[model,'neutral'])
            validate_payload_pair(original,neutral);models.append(analyze_pair(design,original,neutral,indices))
    families=sorted({x['family'] for x in models})
    hosted_amendment=bound_file(base/'hosted_execution_revisions/v2/hosted_execution.json',
                                '80eb49a0c35c1917c1370ad92ba8df5be2ceef567f293640cdb4e181c863c6aa')
    report={'schema':'modebench-discovery-curves-analysis-v1','status':'complete',
            'experiment_status':'complete' if len(families)==2 else 'partial_panels',
            'scope':{'included_panels':families,'omitted_panels':[x for x in ('frontier','local') if x not in families],
                     'registered_panels':['frontier','local'],'included_panels_complete':True},
            'design':design['sources'],'registries':registries,'inventory':inventory,'protocol':design['protocol'],
            'prospective_analysis_plan':binding(base/'ANALYSIS_PLAN.md'),
            'prospective_amendments':{'hosted_execution.json':hosted_amendment},
            'analyzer_source':binding(__file__),
            'analysis_dependencies':{'audit_hosted_modebench_completion.py':binding(Path(__file__).resolve().parent/'audit_hosted_modebench_completion.py')},
            'metrics':list(METRICS),'models':models,'local_seed_groups':combine_local_seeds(models,indices,replicates),
            'limitations':['Sixteen problems per domain-level cell; exploratory pointwise intervals do not establish equivalence.',
                'Each k point rarefies the same 64 observed draws; it does not estimate total unseen support.',
                'Ordered prefixes use preassigned draw_index, never wall-clock response-arrival order.',
                'Joint eligibility changes across correct-draw budgets; every denominator is retained.',
                'All support references in this campaign are certified lower bounds, not exhaustive counts.',
                'Collision reference 1/L is an upper bound for full-support uniform collision; uniform breadth at L is a lower reference, not a lower bound on model breadth.',
                'Equal-seed mean collision and pooled descriptive pair counts are distinct estimands.',
                'Two-seed Pantry intervals condition on those checkpoints; the initial checkpoint has no training-seed replication.',
                'Zero-width bootstrap intervals mean zero observed resampled variation, not population equivalence.']}
    return report


def display_rows(report,figure_key,grading):
    conditional=figure_key.startswith('correct_budget_')
    family=figure_key.removeprefix('correct_budget_')
    rows=[]
    for model in report['models']:
        if model['family']!=family or (family=='local' and model['checkpoint']['training_method']!='initial'):continue
        label=METHOD_LABELS['initial'] if family=='local' else {'gpt56sol':'GPT-5.6 Sol','gpt54':'GPT-5.4','grok43':'Grok 4.3'}.get(model['model_id'],model['model_id'])
        for cell,values in model['analyses'][grading]['cells'].items():
            level,domain=cell.split('/')
            rows.append({'label':label,'method':'initial' if family=='local' else model['model_id'],'domain':domain,
                         'level':int(level[-1]),'seed_count':1,'metrics':{a:values[a] for a in (*ARMS,CONTRAST)},
                         'counts':values['counts'],'interval_kind':'paired prompts','seed_ranges':None})
    if family=='local':
        for group in report['local_seed_groups'].values():
            for cell,values in group['analyses'][grading]['cells'].items():
                rows.append({'label':METHOD_LABELS[group['training_method']],'method':group['training_method'],'domain':group['domain'],
                             'level':int(cell[5]),'seed_count':group['seed_count'],
                             'metrics':values['hierarchical_seed_and_prompt_ci'] or values['seed_mean_fixed_seed_prompt_ci'],
                             'counts':values['counts'],'interval_kind':'seed + paired prompts' if group['seed_count']>=5 else 'fixed-seed paired prompts',
                             'seed_ranges':values['contrast_seed_range'] if group['seed_count']<5 else None})
    for row in rows:
        row['grading']=grading
        if conditional:
            row['metrics']={arm:{name:value for name,value in metrics.items() if name.startswith(('conditional_joint/','collision_joint/'))}
                            for arm,metrics in row['metrics'].items()}
    return sorted(rows,key=lambda r:(DOMAINS.index(r['domain']),r['level'],r['label']))


def write_csv(report,path):
    import csv
    fields=['model_id','family','grading','cell','arm','metric','estimate','ci95_low','ci95_high','defined_bootstrap_replicates']
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for model in report['models']:
            for grading,analysis in model['analyses'].items():
                for cell,values in analysis['cells'].items():
                    for arm in (*ARMS,CONTRAST):
                        for metric,value in values[arm].items():
                            ci=value['ci95'] or [None,None]
                            writer.writerow({'model_id':model['model_id'],'family':model['family'],'grading':grading,'cell':cell,'arm':arm,
                                             'metric':metric,'estimate':value['estimate'],'ci95_low':ci[0],'ci95_high':ci[1],
                                             'defined_bootstrap_replicates':value['defined_bootstrap_replicates']})


def scalar(value,digits=3,signed=False):
    if value is None:return '--'
    return f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def interval_tex(value):
    point=scalar(value['estimate'],signed=True)
    return '$'+point+(r'\;['+','.join(scalar(x,signed=True) for x in value['ci95'])+']' if value['ci95'] else '')+'$'


def render_gain_table(report,family,grading):
    label='strict verification' if grading=='strict' else 'frozen normalization sensitivity'
    seed_note=(r'$s=1$ denotes one fixed initial checkpoint, not training-seed replication. Five-seed intervals resample seeds and paired prompts; two-seed Pantry intervals condition on the observed checkpoints and retain all paired prompt outcomes.'
               if family=='local' else 'Each row represents a fixed hosted deployment; its intervals resample paired prompts and contain no variation across independently trained model seeds.')
    lines=[r'\begin{table}[H]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',
           r'\caption{Eight-to-64 discovery gains under '+label+r'. $G_P=P_{64}-P_8$, $G_D=D_{64}-D_8$, and $G_B=B_{64}-B_8$. '
           r'Each O/N entry gives original/neutral; $\Delta G_D$ is neutral minus original with its pointwise 95\% interval. '
           +seed_note+'}',
           r'\label{tab:discovery-gains-'+family+'-'+grading.replace('_','-')+'}',
           r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrr}',r'\toprule',
           r'Model & Cell & $G_P$: O/N & $G_D$: O/N & $G_B$: O/N & $\Delta G_D$ [95\%] \\',r'\midrule']
    for row in display_rows(report,family,grading):
        values=row['metrics'];pairs=[]
        for metric in ('pass','distinct','breadth'):
            name='gain8to64/rarefaction/'+metric
            pairs.append('$'+'/'.join(scalar(values[arm][name]['estimate']) for arm in ARMS)+'$')
        lines.append(' & '.join([row['label']+(' ($s='+str(row['seed_count'])+'$)' if family=='local' else ''),DOMAIN_LABELS[row['domain']]+' L'+str(row['level']),
                                *pairs,interval_tex(values[CONTRAST]['gain8to64/rarefaction/distinct'])])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    return '\n'.join(lines)


def render_collision_table(report,family):
    lines=[r'\begin{table}[H]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',
           r'\caption{Full-64 strict correct-pair collision. $C$ pools correct pairs within a fixed checkpoint or deployment; trained local rows then average checkpoint rates equally across seeds. $U$ applies the same weighting to the certified-support uniform upper reference. '
           r'Eligible checkpoint--problem groups $E$ and correct-pair counts $Q$ are summed descriptively across checkpoints. These totals do not define the equal-seed mean; separate pooled rates and per-seed counts are retained in the report. A seed with no eligible pair leaves its all-seed mean undefined, with every individual denominator retained.}',
           r'\label{tab:discovery-collision-'+family+'}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrrrr}',r'\toprule',
           r'Model & Cell & $C_O$ & $C_N$ & $U_O$ & $U_N$ & $E_O/E_N$ & $Q_O/Q_N$ \\',r'\midrule']
    for row in display_rows(report,family,'strict'):
        metrics=row['metrics'];counts=row['counts'];vals=[]
        for name in ('collision_own/observed','collision_own/uniform_reference'):
            vals.extend('$'+scalar(metrics[a][name]['estimate'])+'$' for a in ARMS)
        vals += [str(counts['original']['conditional_own_eligible']['2'])+'/'+str(counts['neutral']['conditional_own_eligible']['2']),
                 str(counts['original']['correct_pairs'])+'/'+str(counts['neutral']['correct_pairs'])]
        lines.append(' & '.join([row['label'],DOMAIN_LABELS[row['domain']]+' L'+str(row['level']),*vals])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    return '\n'.join(lines)


def render_eligibility_table(report,family):
    lines=[r'\begin{table}[H]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{4pt}',
           r'\caption{Strict joint eligibility for a fixed correct-draw budget: both wordings must have at least $m$ correct draws. Entries count eligible checkpoint--problem groups, summing over displayed seeds; the same 16 problems recur across checkpoints. Arm-specific eligibility at every $m$ is also retained in the report for both prompt wordings and every registered checkpoint.}',
           r'\label{tab:discovery-eligibility-'+family+'}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{llrrrrrrr}',r'\toprule',
           'Model & Cell & '+' & '.join('$m='+str(k)+'$' for k in GRID)+r' \\',r'\midrule']
    for row in display_rows(report,family,'strict'):
        counts=row['counts']['original']['conditional_joint_eligible']
        lines.append(' & '.join([row['label']+(' ($s='+str(row['seed_count'])+'$)' if family=='local' else ''),DOMAIN_LABELS[row['domain']]+' L'+str(row['level']),
                                *[str(counts[str(k)]) for k in GRID]])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    return '\n'.join(lines)


def render_appendix(report):
    families=report['scope']['included_panels']
    text=[r'''\subsection{Protocol and estimands}
\label{sec:discovery-curves}
Large-budget \texttt{pass@k} evaluation can reveal differences hidden at small sampling budgets \citep{yue2025rlvrlimit}. We therefore freeze a second follow-up after observing the eight-draw prompt control: within every domain--level cell we take ranks 1--16 of its original outcome-independent SHA-256 selection. The six cells retain both prompt wordings, the same mathematical problems, and the same fixed initial and trained checkpoints. Every arm receives 64 fresh draws; none of the earlier eight-draw responses enters this analysis. The initial Qwen2.5-0.5B-Instruct checkpoint has no training-seed replication. DrGRPO and Re:Dr retain five matched seeds in Python and MathIR and two in Pantry. Trained Level-3 evaluation is transfer from Level-2 training.

For a complete 64-draw pool let $c$ be its number of correct outputs and $n_j$ its counts of distinct verified canonical keys. At $k\in\{1,2,4,8,16,32,64\}$, primary curves average all size-$k$ subsets without replacement from the complete pool:
\[
 P_k=1-\frac{\binom{64-c}{k}}{\binom{64}{k}},\qquad
 D_k=\sum_j\left[1-\frac{\binom{64-n_j}{k}}{\binom{64}{k}}\right],\qquad B_k=D_k-P_k.
\]
An infeasible numerator combination is zero. Every incorrect, empty, refused, or truncated response remains in the pool; only verifier-accepted responses contribute keys. These correlated curve points describe the retained pool, not complete unseen support. Ordered prefixes by preassigned draw index, rather than response arrival time, are a separately labeled sensitivity in the source report. $P_{64}-P_8$, $D_{64}-D_8$, and $B_{64}-B_8$ separate first-success gains from additional observed modes.

Strict verification is primary. The unchanged frozen formatter is a sensitivity, preserving every strict success and canonical key. All Python texts undergo the same serial warmed-verifier recheck; original grades and any corrections are retained. Intervals use 20,000 whole-problem bootstrap replicates (seed 20260911), paired across wordings and stratified by domain and level. Five-seed intervals additionally resample whole matched training seeds with shared prompt indices; Pantry intervals condition on its two checkpoints. Intervals are pointwise and exploratory for 16 problems per cell. Zero-width intervals reflect zero observed resampled variation, not precise evidence of equivalence.

To compare breadth at matched observed correctness, rarefy $m$ correct draws alone:
\[
 R_m=\sum_j\left[1-\frac{\binom{c-n_j}{m}}{\binom{c}{m}}\right].
\]
The primary paired conditional comparison keeps only prompts with $c\ge m$ under both wordings; eligibility changes with $m$. Every eligible denominator is retained, and an undefined registered seed is never silently omitted from an equal-seed mean. Full-64 collision pools equal-key correct pairs within each checkpoint before taking equal-seed means.

All support counts here are certified lower bounds $L\le M$, not exhaustive support sizes: Python uses its two externally certified witnesses, MathIR five certified modes, and Pantry its frozen certificate count. Accordingly $1/L$ is an upper reference for uniform collision over full support; $L[1-(1-1/L)^m]$ is a lower reference for uniform expected breadth, not a lower bound on model breadth. Observed distinct counts may exceed $L$. The source report also averages this uniform breadth over $J\sim\operatorname{Hypergeom}(64,c,k)$ to match unconditional correctness at each $k$.
''']
    if report['experiment_status']!='complete':
        omitted=', '.join(report['scope']['omitted_panels'])
        text.append('This is a partial-panel report: the '+', '.join(families)+' panel is complete, while the separately registered '+omitted+' panel is omitted and the overall experiment remains incomplete.')
    for family in families:
        for grading in GRADINGS:text.append(render_gain_table(report,family,grading))
        text += [r'\clearpage',r'\begin{figure}[H]',r'\centering',
                 r'\includegraphics[width=\linewidth,height=0.72\textheight,keepaspectratio]{figures/modebench_discovery_curves_'+family+r'.pdf}',
                 r'\caption{Fresh 64-draw discovery curves. Separate axes show $P_k$ and $D_k$ for each level and domain; pass probability always uses a 0--1 scale. Color identifies the checkpoint method or hosted model, solid/dashed lines original/neutral wording. Lines and shaded pointwise intervals use strict grading; crosses show the frozen-normalization sensitivity. Prefix estimates are separately retained in the report with pointwise intervals for every model and cell.}',
                 r'\label{fig:discovery-curves-'+family+'}',r'\end{figure}',render_collision_table(report,family),r'\clearpage',
                 r'\begin{figure}[H]',r'\centering',r'\includegraphics[width=\linewidth,height=0.66\textheight,keepaspectratio]{figures/modebench_discovery_correct_budget_'+family+r'.pdf}',
                 r'\caption{Breadth after exactly $m$ correct draws, on prompts jointly eligible under both wordings. Color and solid/dashed wording styles match the discovery figure; dotted lines give method-specific uniform lower references using the same eligible prompts. Undefined means are gaps. Crosses show frozen normalization. Eligibility varies with $m$, so curves do not hold the eligible population fixed across the horizontal axis; the following table gives all strict joint counts for every method, level, and correct-draw budget, including groups with zero eligible prompts.}',
                 r'\label{fig:discovery-correct-budget-'+family+'}',r'\end{figure}',render_eligibility_table(report,family)]
    return '\n\n'.join(text)+'\n'


def plot_figure(report,figure_key,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    conditional=figure_key.startswith('correct_budget_');family=figure_key.removeprefix('correct_budget_')
    strict=display_rows(report,figure_key,'strict');normalized=display_rows(report,figure_key,'normalized_secondary')
    methods=['initial','drgrpo','replay_drgrpo'] if family=='local' else ['gpt56sol','gpt54','grok43']
    labels={r['method']:r['label'] for r in strict};colors=dict(zip(methods,('#3977A8','#B44C36','#387F53')))
    lookup={(r['domain'],r['level'],r['method']):r for r in strict}
    secondary={(r['domain'],r['level'],r['method']):r for r in normalized}
    fig,axes=plt.subplots(3,2 if conditional else 4,figsize=(7.4,6.4 if conditional else 6.5),squeeze=False)
    for i,domain in enumerate(DOMAINS):
        for li,level in enumerate(LEVELS):
            kinds=['distinct'] if conditional else ['pass','distinct']
            for j,metric in enumerate(kinds):
                ax=axes[i,li if conditional else 2*li+j]
                for method in methods:
                    row=lookup[domain,level,method];sensitivity=secondary[domain,level,method]
                    for arm,style in [('original','-'),('neutral','--')]:
                        names=[f'conditional_joint/distinct/m{k}' if conditional else f'rarefaction/{metric}/k{k}' for k in GRID]
                        estimates=[row['metrics'][arm][name] for name in names]
                        y=np.asarray([np.nan if v['estimate'] is None else v['estimate'] for v in estimates])
                        lo=np.asarray([np.nan if not v['ci95'] else v['ci95'][0] for v in estimates]);hi=np.asarray([np.nan if not v['ci95'] else v['ci95'][1] for v in estimates])
                        ax.plot(GRID,y,style,color=colors[method],linewidth=1.35)
                        ax.fill_between(GRID,lo,hi,color=colors[method],alpha=.065,linewidth=0)
                        normalized_y=[sensitivity['metrics'][arm][name]['estimate'] for name in names]
                        ax.scatter(GRID,[np.nan if v is None else v for v in normalized_y],marker='x',s=11,linewidths=.7,color=colors[method],alpha=.8)
                    if conditional:
                        reference=[row['metrics']['original'][f'conditional_joint/uniform_distinct/m{k}']['estimate'] for k in GRID]
                        ax.plot(GRID,[np.nan if v is None else v for v in reference],':',color=colors[method],linewidth=1,alpha=.8)
                ax.set_xscale('log',base=2);ax.set_xticks([1,8,64]);ax.set_xticklabels(['1','8','64'])
                ax.tick_params(labelsize=9.5,pad=2);ax.grid(alpha=.17,linewidth=.5)
                ax.set_title(('L'+str(level)+' | '+('$R_m$' if conditional else '$P_k$' if metric=='pass' else '$D_k$')),fontsize=10.5,pad=5)
                if metric=='pass':ax.set_ylim(-.03,1.03);ax.set_yticks([0,.5,1])
                else:ax.set_ylim(bottom=0)
                if i==2:ax.set_xlabel('Correct draws m' if conditional else 'Draw budget k',fontsize=10)
                if (li==0 and j==0):ax.set_ylabel(DOMAIN_LABELS[domain],fontsize=10)
                for spine in ('top','right'):ax.spines[spine].set_visible(False)
    handles=[Line2D([0],[0],color=colors[x],label=labels[x],lw=1.6) for x in methods]
    handles += [Line2D([0],[0],color='0.2',ls='-',label='Original'),Line2D([0],[0],color='0.2',ls='--',label='Neutral'),
                Line2D([0],[0],color='0.2',ls='None',marker='x',label='Normalized')]
    if conditional:handles.append(Line2D([0],[0],color='0.4',ls=':',label='Uniform lower reference'))
    fig.legend(handles=handles,loc='upper center',ncol=3,fontsize=9.5,frameon=False,bbox_to_anchor=(.52,1.005))
    fig.tight_layout(rect=(0,0,1,.86 if conditional else .9),h_pad=1.5,w_pad=.65)
    stem='modebench_discovery_correct_budget_'+family if conditional else 'modebench_discovery_curves_'+family
    outputs={}
    for ext in ('pdf','png'):
        path=Path(output)/(stem+'.'+ext);fig.savefig(path,dpi=220,bbox_inches='tight');outputs[ext]=binding(path)
    plt.close(fig)
    metadata={'schema':'modebench-discovery-figure-v1','family':figure_key,'report_sha256':file_sha(Path(output)/'analysis.json'),
              'outputs':outputs,'plotted_records':[{ 'grading':g,**row} for g in GRADINGS for row in display_rows(report,figure_key,g)]}
    write_json(Path(output)/(stem+'.json'),metadata)
    return binding(Path(output)/(stem+'.json'))


def write_artifacts(report,output):
    output=Path(output).resolve();require(not output.exists(),'Refusing to overwrite an existing analysis directory')
    output.mkdir(parents=True)
    source=freeze_source(output)
    import shutil
    auditor=Path(__file__).resolve().parent/'audit_hosted_modebench_completion.py';target=output/'analysis_code'/auditor.name
    shutil.copy2(auditor,target)
    report['analyzer_source']=source;report['analysis_dependencies']={auditor.name:binding(target)}
    report['publication_source']={'directory':str(output),'report_path':str(output/'analysis.json')}
    write_json(output/'analysis.json',report);write_csv(report,output/'all_cells.csv');(output/'appendix.tex').write_text(render_appendix(report))
    figures={}
    for family in report['scope']['included_panels']:
        for key in (family,'correct_budget_'+family):figures[key]=plot_figure(report,key,output)
    (output/'README.md').write_text('Fresh 64-draw discovery-curve analysis.\n\nThe complete registered panel scope is recorded in analysis.json; omitted panels remain incomplete. appendix.tex and the two figure families are exact publication inputs. all_cells.csv retains every checkpoint, arm, grading sensitivity, metric and pointwise interval. Conditional eligibility, all failures, per-seed effects, support-reference bounds, and separate pooled pair-count summaries are in analysis.json. No earlier n8 draws were pooled.\n')
    manifest={'status':'complete','analyzer_source':source,'analysis_dependencies':report['analysis_dependencies'],
              'outputs':{name:binding(output/name) for name in ('analysis.json','all_cells.csv','appendix.tex','README.md')},'figures':figures}
    write_json(output/'artifact_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('artifacts/modebench_discovery_curves_20260911'))
    parser.add_argument('--local-plan',type=Path);parser.add_argument('--hosted-registry',type=Path)
    parser.add_argument('--scope',choices=('full','local','hosted'),default='full')
    parser.add_argument('--grade-local');parser.add_argument('--grade-hosted',type=Path)
    parser.add_argument('--inventory',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args();local=args.local_plan or args.base/'local/plan.json';hosted=args.hosted_registry or args.base/'hosted_analysis_runs.json'
    if args.grade_local:
        require(not args.grade_hosted,'Choose one isolated grading operation')
        result=grade_local(args.base,local,args.grade_local)
    elif args.grade_hosted:result=grade_hosted(args.base,args.grade_hosted)
    elif args.inventory:
        result=inventory_report(authenticate_design(args.base),local if args.scope!='hosted' else None,hosted if args.scope!='local' else None)
        if args.output:write_json(args.output,result)
    else:
        require(args.output is not None,'Final reporting requires a new output directory')
        result=build_report(args.base,local,hosted,scope=args.scope);write_artifacts(result,args.output)
        result={'status':'complete','output':str(args.output),'models':len(result['models'])}
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','chunks','models')},sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
