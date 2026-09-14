#!/usr/bin/env python3
"""Fresh domain revisions after failed development fits; original evidence stays in place."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
for directory in ('ops', 'ops/exp_scaling', 'src'):
    sys.path.insert(0, str(ROOT / directory))
import materialize_modebench_scale as original
import fit_modebench_scale as common_fit
import evaluate_modebench_scale as evaluator
from fit_modebench_level3 import atomic_new, file_sha, sha, allocate_cells, select_rows, cell_histogram
from materialize_modebench_harder_v2 import identity_set, load_rows, SPLITS

SCHEMA = 'modebench_scale_domain_revision_v1'
RECIPE_SCHEMA = 'modebench_scale_domain_revision_recipe_v1'
DATASET_SCHEMA = 'modebench_scale_domain_revision_dataset_v1'
AUDIT_SCHEMA = 'modebench_scale_domain_revision_confirmation_v1'
DEFAULT_PARENT = ROOT / 'var/data/modebench_scale_v1'
DEFAULT_ROOT = ROOT / 'var/data/modebench_scale_domain_revisions_v1'
read, require = original.read, original.require


def now():
    return datetime.now(timezone.utc).isoformat()


def labels(level, domain, revision, phase):
    require(level in original.LEVELS and domain in original.DOMAINS and phase in ('dev', 'eval'), 'unknown revision scope')
    require(type(revision) is int and 2 <= revision < 50, 'revision must be 2..49')
    base = 7_000_000 + (int(level[-1]) - 4) * 1_000_000 + original.DOMAINS.index(domain) * 100_000 + revision * 1_000 + (500 if phase == 'eval' else 0)
    return list(range(base, base + 4))


def paths(root):
    root = Path(root).resolve()
    p = read(root / 'protocol.json')
    level, domain = p['level'], p['domain']
    return {'root': root, 'protocol': root / 'protocol.json', 'base': root / level,
            'pools': root / level / 'pools' / domain,
            'recipe': root / level / 'recipes' / (domain + '.json'),
            'dataset': root / level / 'dataset' / domain,
            'development': root / level / 'results/development' / domain,
            'confirmation': root / level / 'results/confirmation' / (domain + '.json'),
            'audit': root / level / 'confirmation' / (domain + '.json')}


def authenticate(root):
    root = Path(root).resolve()
    p = read(root / 'protocol.json')
    require(read(root / 'protocol.sha256.json').get('sha256') == file_sha(root / 'protocol.json'), 'revision protocol changed')
    require(p.get('schema') == SCHEMA and p.get('root') == str(root), 'wrong revision protocol')
    require(p['level'] in original.LEVELS and p['domain'] in original.DOMAINS, 'wrong revision scope')
    for path, digest in p['files_sha256'].items():
        require(file_sha(path) == digest, 'registered revision input changed: ' + path)
    require(p['draw_labels'] == {phase: labels(p['level'], p['domain'], p['revision'], phase) for phase in ('dev', 'eval')}, 'revision draw labels changed')
    require(p['selection_seed'] == original.SELECTION_SEED and p['tolerances'] == original.TOLERANCES, 'fitting policy changed')
    return p


def candidate_provider(name):
    require(isinstance(name, str) and name.startswith("modebench_scale_") and name.isidentifier(), "local scale candidate module required")
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve().parent == Path(__file__).resolve().parent, "candidate provider must be local")
    return module


def register(root, level, domain, *, parent=DEFAULT_PARENT, revision=2, candidate_module="modebench_scale_bridge_candidates"):
    candidates = candidate_provider(candidate_module)
    root, parent = Path(root).resolve(), Path(parent).resolve()
    require(not root.exists(), 'fresh revision root required')
    require(domain in candidates.PROFILES and len(candidates.PROFILES[domain]) == 4, 'four registered bridge laws required')
    p = original.authenticate(parent / 'protocol.json')
    recipe_path = parent / level / 'recipes' / (domain + '.json')
    previous = read(recipe_path)
    require(previous == common_fit.fit_domain(parent, level, domain, publish=False), 'parent fit is not reproducible')
    require(previous['development_fit_pass'] is False, 'revise only a failed development fit')
    require(not (parent / level / 'results/confirmation' / (domain + '.json')).exists(), 'this revision cannot use an observed parent holdout')
    pin_paths = set(candidates.source_paths()) | {Path(__file__), Path(original.__file__), Path(common_fit.__file__),
        ROOT / 'ops/exp_scaling/modebench_scale_source_disjointness.py', parent / 'protocol.json', recipe_path}
    pins = {**p['files_sha256'], **{str(Path(f).resolve()): file_sha(Path(f)) for f in pin_paths}}
    pins.update(previous['input_sha256'])
    value = {'schema': SCHEMA, 'created_at': now(), 'root': str(root), 'parent_root': str(parent),
             'level': level, 'domain': domain, 'revision': revision,
             'parent_protocol_sha256': file_sha(parent / 'protocol.json'), 'parent_recipe_sha256': file_sha(recipe_path),
             'models': p['models'], 'targets': {domain: p['targets'][domain]}, 'histograms': {domain: p['histograms'][domain]},
             'split_sizes': p['split_sizes'], 'tolerances': p['tolerances'], 'selection_seed': original.SELECTION_SEED,
             'candidate_module': candidate_module, 'candidate_profiles': candidates.PROFILES[domain], 'sampling': p['sampling'], 'fit': p['fit'],
             'draw_labels': {phase: labels(level, domain, revision, phase) for phase in ('dev', 'eval')},
             'generation_seed_rule': 'sha256([schema, level, domain, revision, split, tier])[:12]',
             'exclusion_rule': 'Snapshot all historical and currently materialized ModeBench rows before each generation; verify cross-source disjointness before confirmation and release.',
             'information_boundary': {'parent_development_fit_failed': True, 'parent_holdout_observed': False,
                                      'revised_holdout_observed': False, 'source_choice': 'development_only'},
             'files_sha256': pins}
    root.mkdir(parents=True)
    atomic_new(root / 'protocol.json', value)
    atomic_new(root / 'protocol.sha256.json', {'sha256': file_sha(root / 'protocol.json')})
    authenticate(root)
    return value


def generation_seed(p, split, tier):
    return int(sha([SCHEMA, p['level'], p['domain'], p['revision'], split, tier])[:12], 16)


def exclusions(root, stage):
    """Structural exclusion snapshots may grow; model outcomes never select rows."""
    root = Path(root).resolve(); p = authenticate(root)
    blocked, prompts, pins = original.history(p['domain'], root)
    # Original history scans every top-level ModeBench tree, including this
    # revisions collection. Explicit local rows cover roots outside var/data.
    for path in root.glob('**/' + p['domain'] + '/*.jsonl'):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        blocked.update(identity_set(p['domain'], rows)); prompts.update(sha(row['problem']) for row in rows)
        pins[str(path.resolve())] = file_sha(path)
    value = {'schema': 'modebench_scale_revision_exclusions_v1', 'stage': stage, 'created_at': now(),
             'protocol_sha256': file_sha(root / 'protocol.json'),
             'identities': sorted(blocked, key=sha), 'prompt_sha256': sorted(prompts), 'files_sha256': pins}
    path = root / 'exclusions' / (stage + '.json')
    atomic_new(path, value)
    return blocked, prompts, path


def generate(p, target, blocked, split, tier):
    candidates = candidate_provider(p['candidate_module'])
    marginal = Counter()
    for key, count in target.items(): marginal[key[0]] += count
    extra = {'joint_target': target} if p['domain'] == 'pantry' else {}
    return candidates.build_pool(p['domain'], marginal, blocked, generation_seed(p, split, tier),
                                 p['level'] + '_revision_' + str(p['revision']) + '_' + split,
                                 tier, multiplier=1, **extra)


def verify_rows(p, rows, target, blocked, prompts):
    candidates = candidate_provider(p['candidate_module'])
    ids = identity_set(p['domain'], rows); texts = {sha(row['problem']) for row in rows}
    require(len(rows) == sum(target.values()) == len(ids) == len(texts), 'revision duplicate or row count')
    require(not ids & blocked and not texts & prompts, 'revision overlaps excluded rows')
    require(cell_histogram(p['domain'], rows) == target, 'revision quota drift')
    certificate = candidates.verify_rows(p['domain'], rows)
    return {'rows': len(rows), 'rows_sha256': sha(rows), 'semantic_disjoint': True, 'prompt_disjoint': True,
            'cells': original.serialize_cells(target), 'verification': certificate}


def materialize_pools(root):
    root = Path(root).resolve(); p = authenticate(root); q = paths(root)
    hist = {s: original.deserialize_cells(h) for s, h in p['histograms'][p['domain']].items()}
    target = original.union_histogram(hist)
    with original.domain_lock(root, p['domain']):
        require(not q['pools'].exists(), 'never overwrite a revision pool')
        blocked, prompts, snapshot = exclusions(root, 'development')
        staging = Path(tempfile.mkdtemp(prefix='.pools-', dir=root))
        try:
            tiers = {}
            for tier in range(4):
                rows = generate(p, target, blocked, 'development', tier)
                tiers[str(tier)] = verify_rows(p, rows, target, blocked, prompts)
                tiers[str(tier)]['generation_seed'] = generation_seed(p, 'development', tier)
                original.write_jsonl(staging / f'difficulty_{tier}.jsonl', rows)
                blocked.update(identity_set(p['domain'], rows)); prompts.update(sha(row['problem']) for row in rows)
            atomic_new(staging / 'identity.json', {'schema': 'modebench_scale_revision_pools_v1',
                       'protocol_sha256': file_sha(q['protocol']), 'level': p['level'], 'domain': p['domain'],
                       'exclusions': str(snapshot), 'exclusions_sha256': file_sha(snapshot), 'tiers': tiers})
            q['pools'].parent.mkdir(parents=True, exist_ok=True); staging.rename(q['pools'])
        except BaseException:
            shutil.rmtree(staging); raise
    return read(q['pools'] / 'identity.json')


def rows_from_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def receipt_scores(path, rows, p, phase, *, regrade=False):
    receipt = read(path); evaluator.validate_seed_receipt(receipt, rows)
    identity = receipt['identity']
    require(all(receipt.get(k) == v for k, v in {'level': p['level'], 'domain': p['domain'], 'split': phase,
            'model_label': original.LEVELS[p['level']]}.items()), 'wrong revision receipt scope')
    require(identity['seeds'] == p['draw_labels'][phase] and identity['source']['row_offset'] == 0
            and identity['source']['row_limit'] == 0, 'wrong revision draw labels or partial source')
    require({k:v for k,v in identity['model'].items() if k != 'vllm_version'} == p['models'][original.LEVELS[p['level']]], 'wrong revision checkpoint')
    if regrade:
        from oat_drgrpo.math_grader import validated_modebench_outcome_key
        for row, result in zip(rows, receipt['prompt_results']):
            for draw in result['draws']:
                for attempt in draw['attempts']:
                    require(sha(validated_modebench_outcome_key(attempt['text'], row['answer'])) == sha(attempt['canonical_key']), 'original grader disagrees with revision result')
    return receipt, {sha(row): {m: result[m] for m in original.TOLERANCES} for row, result in zip(rows, receipt['prompt_results'])}


def fit_domain(root, *, publish=True):
    root = Path(root).resolve(); p = authenticate(root); q = paths(root)
    certificate = read(q['pools'] / 'identity.json')
    require(certificate['protocol_sha256'] == file_sha(q['protocol']), 'wrong pool protocol')
    require(file_sha(certificate['exclusions']) == certificate['exclusions_sha256'], 'changed exclusion snapshot')
    pools, scores, runtimes, pins = {}, {}, [], {str(q['pools']/'identity.json'): file_sha(q['pools']/'identity.json')}
    for tier in range(4):
        path = q['pools'] / f'difficulty_{tier}.jsonl'; rows = rows_from_jsonl(path)
        require(certificate['tiers'][str(tier)]['rows_sha256'] == sha(rows)
                and certificate['tiers'][str(tier)]['rows'] == len(rows), 'changed revision pool')
        require(all(row['scale_candidate_tier'] == tier for row in rows), 'wrong revision tier')
        result_path = q['development'] / f'difficulty_{tier}.json'
        receipt, scores[tier] = receipt_scores(result_path, rows, p, 'dev')
        require(Path(receipt['identity']['source']['path']).resolve() == path, 'wrong revision source path')
        pools[tier] = rows; runtimes.append(receipt['identity']['runtime'])
        pins.update({str(path):file_sha(path), str(result_path):file_sha(result_path)})
    require(all(x == runtimes[0] for x in runtimes), 'revision tiers changed runtime')
    hist = {s: original.deserialize_cells(h) for s,h in p['histograms'][p['domain']].items()}
    result = common_fit.choose_mixture(p['domain'], pools, scores, hist, p['targets'][p['domain']]['metrics'])
    result.update(schema=RECIPE_SCHEMA, level=p['level'], domain=p['domain'], revision=p['revision'],
                  protocol_sha256=file_sha(q['protocol']), target=p['targets'][p['domain']],
                  runtime=runtimes[0], input_sha256=pins, fitter_sha256=file_sha(Path(__file__)))
    if publish: atomic_new(q['recipe'], result)
    return result


def freeze_dataset(root):
    from datasets import Dataset, DatasetDict
    root = Path(root).resolve(); p = authenticate(root); q = paths(root)
    recipe = read(q['recipe'])
    require(recipe['development_fit_pass'] and recipe == fit_domain(root, publish=False), 'passing reproducible revision recipe required')
    with original.domain_lock(root, p['domain']):
        require(not q['dataset'].exists(), 'never overwrite a frozen revision dataset')
        blocked, prompts, snapshot = exclusions(root, 'freeze')
        pools = {t: rows_from_jsonl(q['pools']/f'difficulty_{t}.jsonl') for t in range(4)}
        hist = {s:original.deserialize_cells(h) for s,h in p['histograms'][p['domain']].items()}
        selected = select_rows(p['domain'], pools, hist['dev'], recipe['weights'], original.SELECTION_SEED)
        built = {'dev':[item['row'] for item in selected]}
        require([item['row_sha256'] for item in selected] == recipe['selected_dev_row_sha256'], 'changed selected development rows')
        records = {'dev':{'rows':128, 'rows_sha256':sha(built['dev']), 'origin':'sole selected development rows'}}
        for split in ('train','eval'):
            allocated = allocate_cells(hist[split], recipe['weights'], original.SELECTION_SEED); rows = []
            for tier in range(4):
                target = Counter({cell:counts[tier] for cell,counts in allocated.items() if counts[tier]})
                if not target: continue
                fresh = generate(p, target, blocked, split, tier)
                verify_rows(p, fresh, target, blocked, prompts); rows.extend(fresh)
                blocked.update(identity_set(p['domain'], fresh)); prompts.update(sha(row['problem']) for row in fresh)
            rows.sort(key=lambda row:sha([p['level'],p['domain'],split,sha(row)]))
            require(cell_histogram(p['domain'], rows) == hist[split], 'frozen revision histogram drift')
            built[split] = rows; records[split] = {'rows':len(rows), 'rows_sha256':sha(rows), 'fresh':True,
                                                  'cells':original.serialize_cells(hist[split])}
        staging = Path(tempfile.mkdtemp(prefix='.dataset-',dir=root))
        try:
            for split,rows in built.items():
                original.write_jsonl(staging/(split+'.jsonl'),rows)
                DatasetDict({SPLITS[split][1]:Dataset.from_list(rows)}).save_to_disk(str(staging/split))
                require(load_rows(staging/split,SPLITS[split][1]) == rows, 'revision Arrow identity changed')
            atomic_new(staging/'identity.json', {'schema':DATASET_SCHEMA, 'status':'frozen_pending_heldout_confirmation',
                'level':p['level'],'domain':p['domain'],'protocol_sha256':file_sha(q['protocol']),
                'recipe_sha256':file_sha(q['recipe']),'exclusions':str(snapshot),'exclusions_sha256':file_sha(snapshot),
                'splits':records,'test_split':'eval','difficulty_matched':False})
            q['dataset'].parent.mkdir(parents=True,exist_ok=True);staging.rename(q['dataset'])
        except BaseException:
            shutil.rmtree(staging);raise
    return str(q['dataset'])


def confirm_domain(root, *, publish=True):
    root=Path(root).resolve();p=authenticate(root);q=paths(root)
    from modebench_scale_source_disjointness import verify_dataset
    cross_sources=verify_dataset(root,p['level'],p['domain'])
    recipe=read(q['recipe']);dataset=read(q['dataset']/'identity.json')
    require(recipe['development_fit_pass'] and recipe == fit_domain(root,publish=False), 'passing reproducible revision fit required')
    require(dataset['schema']==DATASET_SCHEMA and dataset['protocol_sha256']==file_sha(q['protocol'])
            and dataset['recipe_sha256']==file_sha(q['recipe']), 'wrong frozen revision identity')
    rows=load_rows(q['dataset']/'eval','multi_answer')
    require(len(rows)==128 and sha(rows)==dataset['splits']['eval']['rows_sha256'], 'revision test changed')
    receipt,scores=receipt_scores(q['confirmation'],rows,p,'eval',regrade=True)
    require(Path(receipt['identity']['source']['path']).resolve() in (q['dataset']/'eval',q['dataset']/'eval.jsonl'), 'wrong revision confirmation path')
    require(receipt['identity']['runtime']==recipe['runtime'], 'confirmation runtime differs from revision development')
    target=p['targets'][p['domain']]['metrics'];delta={m:receipt['metrics'][m]-target[m] for m in original.TOLERANCES}
    gates={m:abs(delta[m])<=original.TOLERANCES[m] for m in delta}
    result={'schema':AUDIT_SCHEMA,'level':p['level'],'domain':p['domain'],'revision':p['revision'],
            'cross_source_disjointness':cross_sources,'difficulty_matched':all(gates.values()),'gates':gates,'differences':delta,'target':target,'metrics':receipt['metrics'],
            'original_grader_replayed_attempts':128*32,'receipt_sha256':file_sha(q['confirmation']),
            'recipe_sha256':file_sha(q['recipe']),'dataset_identity_sha256':file_sha(q['dataset']/'identity.json'),
            'candidate_prompt_bootstrap_delta_95':{m:common_fit.bootstrap_delta([s[m] for s in scores.values()],target[m],original.SELECTION_SEED) for m in delta}}
    if publish:atomic_new(q['audit'],result)
    return result


def launch_inputs(root, phase):
    """Return source-authenticated canonical tasks and all input pins for the subset launcher."""
    root=Path(root).resolve();p=authenticate(root);q=paths(root)
    require(phase in ('dev','eval'),'unknown phase')
    pins={str(q['protocol']):file_sha(q['protocol']),str(root/'protocol.sha256.json'):file_sha(root/'protocol.sha256.json'),**p['files_sha256']};tasks=[]
    if phase=='eval':
        from modebench_scale_source_disjointness import verify_dataset
        cross_sources=verify_dataset(root,p['level'],p['domain'])
        pins.update(cross_sources['files_sha256'])
        recipe=read(q['recipe']);require(recipe['development_fit_pass'] and recipe==fit_domain(root,publish=False),'passing revision fit required before confirmation')
        identity=read(q['dataset']/'identity.json')
        require(identity['recipe_sha256']==file_sha(q['recipe']) and identity['protocol_sha256']==file_sha(q['protocol']),'wrong frozen dataset')
        pins[str(q['recipe'])]=file_sha(q['recipe'])
    for tier in range(4) if phase=='dev' else (None,):
        path=q['pools']/f'difficulty_{tier}.jsonl' if phase=='dev' else q['dataset']/'eval.jsonl'
        rows=rows_from_jsonl(path)
        identity_path=path.parent/'identity.json';identity=read(identity_path)
        expected=identity['tiers'][str(tier)] if phase=='dev' else identity['splits']['eval']
        require(identity['protocol_sha256']==file_sha(q['protocol']) and expected['rows']==len(rows) and expected['rows_sha256']==sha(rows),'wrong source certificate')
        output=q['development']/f'difficulty_{tier}.json' if phase=='dev' else q['confirmation']
        task={'level':p['level'],'domain':p['domain'],'split':phase,'interface':evaluator.INTERFACE,
              'rows_jsonl':str(path),'seeds':p['draw_labels'][phase],'batch_size':8,'row_offset':0,'row_limit':0,'output':str(output)}
        evaluator.validate_task(task,phase=='eval');tasks.append(task)
        pins.update({str(path):file_sha(path),str(identity_path):file_sha(identity_path)})
    cell={'id':p['level']+'_'+p['domain']+'_r'+str(p['revision'])+'_'+phase,'level':p['level'],'domain':p['domain'],
          'phase':phase,'source_kind':'domain_revision_v1','source_root':str(root),
          'model_label':original.LEVELS[p['level']],'tasks':tasks}
    return cell,pins


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','pools','fit','freeze','confirm'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--level',choices=tuple(original.LEVELS));parser.add_argument('--domain',choices=original.DOMAINS)
    parser.add_argument('--revision',type=int,default=2)
    parser.add_argument('--candidate-module',default='modebench_scale_bridge_candidates');args=parser.parse_args()
    if args.action=='register':result=register(args.root,args.level,args.domain,parent=args.parent,revision=args.revision,candidate_module=args.candidate_module)
    else:result={'pools':materialize_pools,'fit':fit_domain,'freeze':freeze_dataset,'confirm':confirm_domain}[args.action](args.root)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
