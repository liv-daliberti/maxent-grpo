#!/usr/bin/env python3
"""Audit Graph v8 capacity and publish only explicitly registered DEV pools.

All old identities, including failed final v2 train/dev/eval, remain excluded.
The four union-quota pools and cumulative pure-tier full-split capacity witnesses
are constructed before publication; capacity rows are ephemeral, never datasets.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for d in ('ops/exp_scaling','ops','src'):sys.path.insert(0,str(ROOT/d))
import materialize_modebench_level3 as materializer
import modebench_level3_graph_v8 as graph
import modebench_level3_v3_common as common
from materialize_modebench_level3_graph_v7 import verify_witnesses
from fit_modebench_level3 import local_dependency_sources
DOMAIN='graph_coloring';NAME='graph_v8'
POOL_ROOT=common.POOL_ROOTS[DOMAIN]
ARTIFACTS=common.CAMPAIGN/NAME
TEST=ROOT/'tests/test_modebench_level3_graph_v8.py'
DEVELOPMENT_SEED=8737100;CAPACITY_SEED=8937100;TRAIN_SEED=9137100;EVAL_SEED=9337100

def pool_paths():
    roots=set((ROOT/'var/data').glob('modebench_harder*'))|set((ROOT/'var/data').glob('modebench_level3*'))
    return sorted(p.resolve() for root in roots if root.resolve()!=POOL_ROOT.resolve() for p in (root/'pools'/DOMAIN).glob('*.jsonl'))

def verify_snapshot(snapshot):
    common.verify_pins(snapshot['files_sha256'],snapshot['directory_files'])

def verify_generation_exclusions_unchanged(snapshot):
    verify_snapshot(snapshot)
    common.require(list(map(str,pool_paths()))==snapshot['candidate_pool_paths'],'historical graph pool inventory changed')
    common.require(materializer.row_hash(sorted(materializer.historical_ids(DOMAIN),key=repr))==snapshot['historical_identity_sha256'],'historical graph identities changed during construction')

def source_snapshot():
    inherited=common.authenticate_inherited();pins=dict(inherited['files_sha256']);trees=dict(inherited['directory_files'])
    def pin(path,expected=None):
        path=str(Path(path).resolve());value=common.digest(path)
        common.require(expected is None or value==expected,'source identity changed')
        pins.update(common.merge_pins(pins,{path:value}))
    for root in set((ROOT/'var/data').glob('modebench_harder*'))|set((ROOT/'var/data').glob('modebench_level3*')):
        for split in materializer.SPLITS:
            folder=root/DOMAIN/split
            if (folder/'dataset_dict.json').is_file():trees[str(folder.resolve())]=sorted(str(p.resolve()) for p in folder.rglob('*') if p.is_file())
    for path in pool_paths():
        pin(path)
        if path.with_suffix('.identity.json').is_file():pin(path.with_suffix('.identity.json'))
    for files in trees.values():
        for path in files:pin(path)
    for name,value in local_dependency_sources([Path(graph.__file__),Path(__file__)]).items():pin(ROOT/name,value)
    pin(TEST)
    for name in ('graph_diagnostic.py','graph_features.json','graph_result.json','graph_recommendation.json'):
        pin(common.OLD_CAMPAIGN/'confirmation_v6_failure_diagnostics'/name)
    snapshot={'files_sha256':pins,'directory_files':trees,'candidate_pool_paths':list(map(str,pool_paths())),
              'historical_identity_sha256':materializer.row_hash(sorted(materializer.historical_ids(DOMAIN),key=repr))}
    verify_generation_exclusions_unchanged(snapshot)
    return snapshot

def exclusions(snapshot):
    blocked=materializer.historical_ids(DOMAIN);historical=len(blocked)
    for path in snapshot['candidate_pool_paths']:blocked|=materializer.identity_set(DOMAIN,[json.loads(line) for line in Path(path).read_text().splitlines()])
    return blocked,historical

def calibration_target():return Counter({cell[0]:count for cell,count in common.calibration_histogram(DOMAIN).items()})

def descriptor(snapshot=None):
    return {'name':NAME,'domain':DOMAIN,'tests_path':str(TEST),'tests_sha256':common.digest(TEST),'generator_path':str(Path(graph.__file__).resolve()),'generator_sha256':common.digest(graph.__file__),
            'materializer_path':str(Path(__file__).resolve()),'materializer_sha256':common.digest(__file__),'pool_root':str(POOL_ROOT),
            'development_receipts':{str(t):str(common.RESULTS/f'calibration_3b_graph_v8_d{t}.json') for t in graph.PRESETS},
            'rows_per_tier':sum(calibration_target().values()),'calibration_histogram':dict(sorted(calibration_target().items())),
            'development_seed_base':DEVELOPMENT_SEED,'capacity_seed_base':CAPACITY_SEED,
            'final_train_seed_base':TRAIN_SEED,'final_eval_seed_base':EVAL_SEED,
            'presets':graph.PRESETS,'sampler_law':graph.SAMPLER_LAW,
            'source_snapshot':source_snapshot() if snapshot is None else snapshot}

def capacity_table(blocked):
    result={}
    for tier in graph.PRESETS:
        result[str(tier)]={}
        for support in sorted(graph.SUPPORTS):
            catalog=graph.catalogue(tier,support);available=catalog-blocked;groups=Counter(graph.group_key(item) for item in available)
            result[str(tier)][str(support)]={'total':len(catalog),'excluded':len(catalog&blocked),'remaining':len(available),
                'nonempty_joint_groups':len(groups),'minimum_nonempty_joint_group':min(groups.values(),default=0),
                'maximum_joint_group':max(groups.values(),default=0)}
    return result

def verify_prefix(rows,target,blocked,seed,tag,tier):
    larger=graph.build_pool(DOMAIN,Counter({s:c+1 for s,c in target.items()}),blocked,seed,tag,tier,multiplier=1)
    for support,count in target.items():
        prefix=sorted((r for r in rows if r['answer_mode_count']==support),key=lambda r:r['level3_cell_index'])
        full=sorted((r for r in larger if r['answer_mode_count']==support),key=lambda r:r['level3_cell_index'])
        common.require(prefix==full[:count],'per-cell quota prefix changed')

def construct(blocked):
    target=calibration_target();common.require(sum(target.values())==142 and len(target)==6,'exact union calibration target required')
    pools={};records={};current=set(blocked)
    reference=materializer.reference_rows(DOMAIN,'dev')
    for tier in graph.PRESETS:
        seed=DEVELOPMENT_SEED+1000*tier;tag='level3_v3_development_pool'
        rows=graph.build_pool(DOMAIN,target,current,seed,tag,tier,multiplier=1)
        checks=materializer.verify_rows(DOMAIN,rows,reference,target,current);verify_prefix(rows,target,current,seed,tag,tier)
        witnesses=verify_witnesses(rows)
        checks.update({'exactly_three_hidden_vertices':all(sum(color is None for color in json.loads(row['answer'])['partial_colors'])==3 for row in rows),'all_five_vertices':all(json.loads(row['answer'])['n']==5 for row in rows),
                       'original_canonical_support_verified':True,'per_cell_quota_prefix_invariance':True,'union_dev_eval_support_cells':True})
        common.require(all(value is True for value in checks.values()),'candidate checks did not pass')
        pools[tier]=rows;records[tier]={'schema':'modebench_level3_development_pool_v1','domain':DOMAIN,'difficulty':tier,'seed':seed,'rows':len(rows),
            'rows_sha256':materializer.row_hash(rows),'checks':checks,'support_histogram':dict(sorted(target.items())),
            'source_sha256':common.digest(graph.__file__),'candidate_revision':NAME,'original_grader_witnesses':witnesses,
            'information_boundary':'new v3 development candidates; old complete confirmation informed prospectively registered laws; no new model outcomes'}
        current|=materializer.identity_set(DOMAIN,rows)
    after=capacity_table(current);runs=[]
    for tier in graph.PRESETS:
        for index,split in enumerate(materializer.SPLITS):
            reference=materializer.reference_rows(DOMAIN,split);target=materializer.modes(reference)
            seed=CAPACITY_SEED+1000*tier+10000*index;tag=f'graph_v8_capacity_{split}'
            rows=graph.build_pool(DOMAIN,target,current,seed,tag,tier,multiplier=1)
            checks=materializer.verify_rows(DOMAIN,rows,reference,target,current);verify_prefix(rows,target,current,seed,tag,tier)
            witnesses=verify_witnesses(rows);current|=materializer.identity_set(DOMAIN,rows)
            runs.append({'tier':tier,'split':split,'seed':seed,'rows':len(rows),'rows_sha256':materializer.row_hash(rows),
                         'support_histogram':dict(sorted(target.items())),'checks':{**checks,'per_cell_quota_prefix_invariance':True},'original_grader_witnesses':witnesses})
            print(json.dumps({'event':'graph_v8_capacity_verified','tier':tier,'split':split,'rows':len(rows)}),flush=True)
    return pools,records,{'after_four_new_pools':after,'capacity_runs':runs,'capacity_rows_verified':sum(r['rows'] for r in runs),
        'all_capacity_rows_globally_disjoint':True,'capacity_rows_are_ephemeral_not_confirmation_datasets':True}

def publish(registration_sha256):
    record=common.validate_registration(common.REGISTRATION,registration_sha256)
    description=record['candidate_revisions'][DOMAIN];snapshot=description['source_snapshot']
    common.require(description==json.loads(json.dumps(descriptor(snapshot))), 'registered Graph v8 descriptor differs')
    common.require(not POOL_ROOT.exists() and not (ARTIFACTS/'structural_audit.json').exists(),'fresh Graph v8 pool/audit outputs required')
    for output in description['development_receipts'].values():common.require(not Path(output).exists() and not Path(output+'.batches').exists(),'model work exists before pool publication')
    verify_generation_exclusions_unchanged(snapshot);blocked,historical=exclusions(snapshot)
    pools,records,capacity=construct(blocked);verify_generation_exclusions_unchanged(snapshot)
    common.validate_registration(common.REGISTRATION,registration_sha256)
    stage=Path(tempfile.mkdtemp(prefix='.'+POOL_ROOT.name+'.',dir=POOL_ROOT.parent))
    try:
        folder=stage/'pools'/DOMAIN;folder.mkdir(parents=True)
        for tier,rows in pools.items():
            path=folder/f'difficulty_{tier}.jsonl'
            with path.open('x') as f:
                for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
            records[tier].update({'registration_path':str(common.REGISTRATION),'registration_sha256':registration_sha256})
            common.atomic_new(path.with_suffix('.identity.json'),records[tier])
        verify_generation_exclusions_unchanged(snapshot);common.require(not POOL_ROOT.exists(),'Graph v8 pool root appeared during publication')
        stage.rename(POOL_ROOT)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    report={'schema':'modebench_level3_graph_v8_structural_audit_v1','registration_path':str(common.REGISTRATION),'registration_sha256':registration_sha256,
        'generator_source_sha256':common.digest(graph.__file__),'pool_root':str(POOL_ROOT),'historical_identities':historical,
        'historical_and_pool_exclusions':len(blocked),'development_pools':records,**capacity,'sealed_files_unchanged':True,
        'new_model_outcomes_loaded':False,'fitting_performed':False,'model_or_scheduler_calls':False,
        'pool_files_sha256':{str(path):common.digest(path) for path in sorted(POOL_ROOT.rglob('*')) if path.is_file()}}
    common.atomic_new(ARTIFACTS/'structural_audit.json',report)
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--materialize-development',action='store_true');parser.add_argument('--registration-sha256')
    args=parser.parse_args(argv)
    if args.materialize_development:
        common.require(args.registration_sha256 is not None,'explicit registration pin required for publication')
        report=publish(args.registration_sha256)
        print(json.dumps({'status':'registered_development_materialized','pool_root':str(POOL_ROOT),'rows':568,'capacity_rows':report['capacity_rows_verified']}));return
    common.require(args.registration_sha256 is None,'registration pin is only used for explicit publication')
    snapshot=source_snapshot();blocked,_=exclusions(snapshot);pools,records,capacity=construct(blocked);verify_generation_exclusions_unchanged(snapshot)
    print(json.dumps({'status':'structural_preview_pass','historical_and_pool_exclusions':len(blocked),'development_rows':sum(len(p) for p in pools.values()),
                      'capacity_rows':capacity['capacity_rows_verified'],'capacity':capacity,'pools':records}))

if __name__=='__main__':main()
