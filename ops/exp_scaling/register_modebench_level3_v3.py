#!/usr/bin/env python3
"""Register reviewed V3 scientific inputs before any new model outcomes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
import modebench_level3_v3_common as common
from fit_modebench_level3 import local_dependency_sources

AMENDMENT=common.CAMPAIGN/'fixed_reference_amendment.json'
AMENDMENT_SHA='2537d45996e304c6f9880cff030332267ce5981c6674ce17f0129ffb1474fdab'
COMMON_SHA='1cb2a47b7b1751da8229be7940bd932816daea09a56bae4b6d62cb2bef1e9522'
REVISIONS={'graph_coloring':'graph_v8','python_factors':'python_v7'}
REQUIRED=(
 'ops/exp_scaling/modebench_level3_v3_common.py',
 'tests/test_modebench_level3_v3_common.py',
 'ops/exp_scaling/fit_modebench_level3_fixed_reference.py',
 'tests/test_fit_modebench_level3_fixed_reference.py',
 'ops/exp_scaling/modebench_level3_v3_development.py',
 'tests/test_modebench_level3_v3_development.py',
 'tests/test_register_modebench_level3_v3.py',
)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def no_new_work():
    common.require(not common.REGISTRATION.exists(),'canonical registration already exists')
    common.require(not common.DATASET.exists() and not common.RESULTS.exists(),
                   'new candidate results or final dataset exist before registration')
    common.require(all(not path.exists() for path in common.POOL_ROOTS.values()),
                   'new calibration pool exists before registration')
    development=common.CAMPAIGN/'development'
    artifacts=('seal.json','plan.json','worker.slurm','prepare_intent.json',
               'development_execution_claim.json','execution_claim.json')
    common.require(not any((development/name).exists() for name in artifacts)
                   and not list(development.glob('*_tasks.json'))
                   and not list(development.glob('submission_*_intent.json'))
                   and not list(development.glob('submission_*_result.json'))
                   and not list(development.glob('worker_*_execution_claim.json')),
                   'new execution preparation/submission exists before scientific registration')

def build_registration():
    no_new_work()
    common.require(common.digest(AMENDMENT)==AMENDMENT_SHA,'prospective reference amendment changed')
    common.require(common.digest(common.__file__)==COMMON_SHA,'reviewed common implementation changed')
    amendment=common.read(AMENDMENT)
    common.require(amendment['contract']==common.contract()
                   and amendment['status']=='prospective_before_new_candidate_publication_or_model_outcomes',
                   'prospective method differs from the reviewed contract')
    for relative in REQUIRED:
        common.require((ROOT/relative).is_file(),'required reviewed implementation is absent: '+relative)
    inherited=common.authenticate_inherited()
    targets=common.fixed_references()
    common.require(set(targets)==set(common.DOMAINS)
                   and all(targets[d]['metrics']==amendment['benchmark_targets'][d]['fixed_metrics']
                           for d in common.DOMAINS),'all five measured reference targets must remain fixed')
    pins=dict(inherited['files_sha256']);trees=dict(inherited['directory_files'])
    descriptions={}
    initial=[ROOT/relative for relative in REQUIRED]+[Path(__file__).resolve()]
    for domain,revision in REVISIONS.items():
        path=ROOT/f'ops/exp_scaling/materialize_modebench_level3_{revision}.py'
        materializer=module(path,'register_'+revision)
        descriptor_function=(materializer.descriptor if domain=='graph_coloring'
                             else materializer.candidate_description)
        description=json.loads(json.dumps(descriptor_function()))
        common.require(description['generator_sha256']==amendment['candidate_laws'][domain]['generator_sha256'],
                       'prospectively declared generator law changed')
        descriptions[domain]=description
        snapshot=description['source_snapshot']
        materializer.verify_generation_exclusions_unchanged(snapshot)
        pins=common.merge_pins(pins,snapshot['files_sha256'])
        trees=common.merge_pins(trees,snapshot['directory_files'])
        initial.extend(Path(description[kind+'_path']) for kind in ('generator','materializer','tests'))
    for path in [AMENDMENT,*initial]:
        pins=common.merge_pins(pins,{str(path.resolve()):common.digest(path)})
    for relative,expected in local_dependency_sources(initial).items():
        pins=common.merge_pins(pins,{str((ROOT/relative).resolve()):expected})
    registration={
      'schema':common.REGISTRATION_SCHEMA,
      'created_at':datetime.now(timezone.utc).isoformat(),
      'prospective_reference_amendment':{'path':str(AMENDMENT),'sha256':AMENDMENT_SHA},
      'contract':common.contract(),'inherited_confirmation':common.inherited_metadata(),
      'benchmark_targets':common.benchmark_metadata(),'candidate_revisions':descriptions,
      'files_sha256':dict(sorted(pins.items())),'directory_files':dict(sorted(trees.items())),
      'publication_boundary':{'new_pool_files_exist':False,'new_model_outcomes_exist':False,
         'new_scheduler_jobs_submitted':False,'retained_domains_regenerated':False},
      'future_execution_policy':'New finalization/confirmation helpers require an additive execution seal before use. Every registered scientific input remains immutable.',
    }
    common.verify_pins(pins,trees)
    no_new_work()
    return registration

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    record=build_registration()
    if not args.publish:
        print(json.dumps({'status':'ready_for_scientific_registration',
          'files':len(record['files_sha256']),'inventories':len(record['directory_files']),
          'candidate_rows':{d:r['rows_per_tier'] for d,r in record['candidate_revisions'].items()},
          'registration_published':False}))
        return
    common.atomic_new(common.REGISTRATION,record)
    pin=common.digest(common.REGISTRATION)
    common.validate_registration(common.REGISTRATION,pin)
    print(json.dumps({'status':'scientific_registration_published','path':str(common.REGISTRATION),
       'sha256':pin,'files':len(record['files_sha256']),'inventories':len(record['directory_files'])}))

if __name__=='__main__':
    main()
