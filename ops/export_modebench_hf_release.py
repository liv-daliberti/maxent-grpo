#!/usr/bin/env python3
"""Export frozen ModeBench datasets without altering source rows or live training."""
from __future__ import annotations
import argparse,ast,hashlib,importlib.metadata,json,os,re,shutil,sys,tempfile
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'var/artifacts/modebench_hf_release_20260911/package'
REPO_ID='od2961/ModeBench'
L1={'graph_coloring':'graph_coloring_modebench_v2','countdown':'exact_countdown_easy3_probe','python_factors':'python_factor_modebench_v1','mathir':'mathir_action_menu_v1','pantry_plan':'pantry_plan_modebench_v2'}
L2=ROOT/'var/data/modebench_harder_v2_matched_r5'
L3=ROOT/'var/data/modebench_level3_matched_v3'
L3_SHA='890d7697af7789e0ae53c803586ec7f722685b7eb2239643175a1933fa45650d'
CONF=ROOT/'var/artifacts/modebench_level3_v3/confirmation/confirmation_report.json'
SOURCE=ROOT/'var/artifacts/source_snapshots/e122_level3_6caff9b52d509a86499f/src/oat_drgrpo'
DOMAINS=list(L1)

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def row_digest(rows,style='json_array'):
 if style=='json_array':text=json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False)
 elif style=='json_lines_no_final_newline':text='\n'.join(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False) for r in rows)
 else:raise ValueError(style)
 return hashlib.sha256(text.encode()).hexdigest()
def write_json(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def copy_file(source,target):
 require(source.is_file() and source.resolve()==source and not source.is_symlink(),f'unsafe source {source}')
 target.parent.mkdir(parents=True,exist_ok=True);require(not target.exists(),f'output exists {target}');shutil.copyfile(source,target)
 require(sha(source)==sha(target),f'copy changed {source}')
 return {'source_repository_path':str(source.relative_to(ROOT)),'path':str(target),'sha256':sha(target),'bytes':target.stat().st_size}
def source_files(path):return sorted(p for p in path.rglob('*') if p.is_file())
def roundtrip_dataset(dataset,path):
 from datasets import Dataset
 rows=list(dataset);features=dataset.features.to_dict();columns=dataset.column_names
 path.parent.mkdir(parents=True,exist_ok=True);require(not path.exists(),f'Parquet output exists {path}')
 dataset.to_parquet(str(path),compression='zstd')
 with tempfile.TemporaryDirectory(prefix='modebench_parquet_cache_') as cache:
  recovered=Dataset.from_parquet(str(path),keep_in_memory=True,cache_dir=cache)
 require(list(recovered)==rows,'Parquet changed rows or order')
 require(recovered.features.to_dict()==features,'Parquet changed original feature types')
 require(recovered.column_names==columns,'Parquet changed column order')
 return {'rows':len(rows),'features':features,'source_column_order':columns,'rows_sha256':row_digest(rows),'rows_jsonl_sha256':row_digest(rows,'json_lines_no_final_newline'),'parquet_sha256':sha(path),'parquet_bytes':path.stat().st_size}
def code_closure():
 pending={'math_grader','templates','canonical_actions','pantry_support_action','modebench_guided','python_modebench_worker','maze_modebench_worker','__init__'};seen=set()
 while pending:
  name=pending.pop()
  if name in seen:continue
  p=SOURCE/(name+'.py');require(p.is_file(),f'missing verifier dependency {name}');seen.add(name);text=p.read_text();tree=ast.parse(text)
  for node in ast.walk(tree):
   if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
    dep=node.module.split('.')[0]
    if (SOURCE/(dep+'.py')).exists():pending.add(dep)
  for dep in re.findall(r'oat_drgrpo\.([A-Za-z_][A-Za-z0-9_]*)',text):
   if (SOURCE/(dep+'.py')).exists():pending.add(dep)
 return sorted(seen)

def export(output=OUT):
 from datasets import load_from_disk,disable_progress_bar
 disable_progress_bar()
 sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'ops/exp_scaling'),str(ROOT/'src')]
 from materialize_modebench_harder_v2 import identity_set
 require(not output.exists(),f'fresh staging directory required {output}')
 require(sha(L3/'identity.json')==L3_SHA,'E122 Level3 dataset identity changed')
 confirmation=read(CONF);require(confirmation['status']=='matched_fixed_reference' and confirmation['all_five_domains_complete'] and confirmation['confirmation_match_verified'],'Level3 separate confirmation missing')
 require(confirmation['dataset']['sha256']==L3_SHA and confirmation['dataset']['dataset_root']==str(L3),'Level3 confirmation binds different dataset')
 output.mkdir(parents=True)
 identities={2:read(L2/'identity.json'),3:read(L3/'identity.json')}
 configs=defaultdict(list);splits=[];seen_ids={d:set() for d in DOMAINS};all_source_pins={};source_before={};role_rows={};validation=[]
 roots=[ROOT/'var/data'/r for r in L1.values()]+[L2,L3]
 for root in roots:
  for p in source_files(root):
   require(p.suffix in ('.arrow','.json','.md','.txt'),f'unexpected source file {p}')
   st=p.stat();source_before[str(p)]={'sha256':sha(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns};all_source_pins[str(p.relative_to(ROOT))]=source_before[str(p)]['sha256']
 for level in (1,2,3):
  for domain in DOMAINS:
   native='pantry' if domain=='pantry_plan' else domain
   root=ROOT/'var/data'/L1[domain] if level==1 else (L2 if level==2 else L3)/native
   for role in ('train','dev','eval'):
    path=root/role
    if not path.is_dir():continue
    data=load_from_disk(str(path));require(set(data.keys())<=({'train'} if role=='train' else {'multi_answer','unique_answer'}),f'unexpected serialized split {path}')
    for subset,dataset in data.items():
     primary=(subset=='train' if role=='train' else subset=='multi_answer')
     config=f'level{level}_{domain}'+('' if primary else '_'+subset)
     rows=list(dataset);require(rows and all(isinstance(r.get('problem'),str) and isinstance(r.get('answer'),str) for r in rows),'invalid row surface')
     specs=[json.loads(r['answer']) for r in rows];require(all(isinstance(s,dict) and isinstance(s.get('verifier'),str) for s in specs),'missing executable specification')
     require(all(type(r.get('answer_mode_count')) is int and r['answer_mode_count']>0 for r in rows),'invalid support count')
     expected=384 if role=='train' else 64 if level==1 and domain=='pantry_plan' and role=='dev' else 128
     require(len(rows)==expected,f'original cardinality changed {path}/{subset}')
     ids=identity_set(native,rows);require(len(ids)==len(rows),f'duplicate canonical problem identities {path}/{subset}')
     if primary:
      require(not(ids&seen_ids[domain]),f'canonical problem overlap across published primary splits {config}/{role}')
      seen_ids[domain].update(ids);role_rows[level,domain,role]=rows
     recorded=None
     if level>1:
      recorded=identities[level]['domains'][native][role]
      hashes={style:row_digest(rows,style) for style in ('json_array','json_lines_no_final_newline')}
      require(recorded['rows_sha256'] in hashes.values(),f'frozen row hash differs {config}/{role}')
      require(recorded['rows']==len(rows) and all(recorded['checks'].values()),f'frozen structural validation differs {config}/{role}')
      target=recorded.get('support_histogram',recorded.get('answer_mode_count_histogram'))
      require({str(k):v for k,v in Counter(r['answer_mode_count'] for r in rows).items()}==target,f'support histogram differs {config}/{role}')
     relative=f'data/{config}/{role}.parquet';result=roundtrip_dataset(dataset,output/relative)
     record={'config_name':config,'level':level,'domain':domain,'source_domain_directory':native if level>1 else L1[domain],'split':role,'original_datasetdict_split':subset,'source_repository_path':str(path.relative_to(ROOT)),'reported_primary_split':primary,'data_file':relative,**result,'support_histogram':dict(sorted(Counter(str(r['answer_mode_count']) for r in rows).items())),'modebench_task_values':sorted({r['modebench_task'] for r in rows}),'verifier_values':sorted({s['verifier'] for s in specs}),'canonical_problem_identities_unique':True,'frozen_identity_record':recorded}
     configs[config].append({'split':role,'path':relative});splits.append(record)
     validation.append({'config':config,'split':role,'source_rows_equal_export_rows':True,'feature_schema_equal':True,'row_order_equal':True,'canonical_problem_identities_unique':True,'frozen_row_digest_verified':level>1})
 # Level3 intentionally preserves Level2 support histograms, not equal difficulty.
 for domain in DOMAINS:
  for role in ('train','dev','eval'):
   require(Counter(r['answer_mode_count'] for r in role_rows[2,domain,role])==Counter(r['answer_mode_count'] for r in role_rows[3,domain,role]),f'L2/L3 support mismatch {domain}/{role}')
 copied=[]
 for root in roots:
  for p in source_files(root):
   if p.suffix=='.json':copied.append(copy_file(p,output/'provenance/source_data'/p.relative_to(ROOT/'var/data')))
 for rel in ['var/artifacts/modebench_level3_v3/confirmation/confirmation_report.json','paper/preregistration/e122_level3_factorial_20260909.md','paper/preregistration/e119_level2_qwen05b_factorial_20260901.md']:
  p=ROOT/rel
  require(p.is_file(),f'missing release protocol {p}');copied.append(copy_file(p,output/'provenance'/p.name))
 # Preserve existing source licenses and attribution; no new dataset terms invented.
 copied.append(copy_file(ROOT/'LICENSE',output/'licenses/source_repository_APACHE_2.0.txt'))
 ingredients=ROOT/'var/data/pantry_plan_v1/ingredients.json'
 if ingredients.is_file():copied.append(copy_file(ingredients,output/'provenance/pantry_ingredient_source.json'))
 code=[]
 for name in code_closure():
  source=SOURCE/(name+'.py');live=ROOT/'src/oat_drgrpo'/(name+'.py');expected=confirmation['files_sha256'].get(str(live))
  require(expected and sha(source)==expected,f'verifier source not bound by completed Level3 confirmation: {name}')
  code.append(copy_file(source,output/'code/oat_drgrpo'/source.name))
 for rel in ['ops/evaluate_modebench_level2_viability.py','ops/evaluate_modebench_level3.py','ops/evaluate_modebench_level3_independent.py','ops/modebench_independent_seeds.py','ops/check_paper_domain_prompts.py','ops/exp_scaling/materialize_modebench_harder_v2.py','ops/exp_scaling/materialize_modebench_level3.py','ops/exp_scaling/modebench_level3_v3_common.py']:
  copied.append(copy_file(ROOT/rel,output/'provenance/source_code'/rel))
 manifest={'schema':'modebench-hf-dataset-release-v1','prepared_at_utc':now(),'repo_id':REPO_ID,'source_data_mutated':False,'levels':[1,2,3],'primary_configs':15,'auxiliary_configs':len(configs)-15,'config_count':len(configs),'split_count':len(splits),'row_count':sum(s['rows'] for s in splits),'configs':[{'config_name':k,'data_files':v} for k,v in sorted(configs.items())],'splits':splits,'source_file_sha256':all_source_pins,'code_files':code,'provenance_files':copied,'runtime':{k:importlib.metadata.version(k) for k in ('datasets','pyarrow','sympy','math-verify','latex2sympy2_extended','pylatexenc','torch','numpy','fsspec','huggingface-hub','PyYAML')},'level3_identity':{'path':'provenance/source_data/modebench_level3_matched_v3/identity.json','sha256':L3_SHA,'original_decision':identities[3]['decision']},'level3_confirmation':{'path':'provenance/confirmation_report.json','sha256':sha(CONF),'status':confirmation['status'],'scope':'Adaptive second-round matching to fixed historical Level1 references; Graph/Python newly confirmed in v3, Countdown/MathIR/Pantry retain earlier bytes and evidence; not a statistical equivalence claim.'},'license_provenance':{'source_code':'Existing repository Apache-2.0 LICENSE copied verbatim; existing source copyright notices retained.','dataset_specific_license':'No separate dataset license found in the frozen dataset roots; no new dataset license assigned by this export.','pantry_ingredient_source':read(ingredients)['source'] if ingredients.exists() else None}}
 for p,before in source_before.items():
  s=Path(p).stat();require(s.st_size==before['size'] and s.st_mtime_ns==before['mtime_ns'] and sha(p)==before['sha256'],f'source changed during export {p}')
 write_json(output/'MANIFEST.json',manifest)
 write_json(output/'VALIDATION.json',{'schema':'modebench-hf-roundtrip-validation-v1','validated_at_utc':now(),'status':'passed','config_count':len(configs),'split_count':len(splits),'row_count':manifest['row_count'],'all_source_files_unchanged':True,'all_primary_problem_identities_disjoint_across_published_splits':True,'all_L2_L3_support_histograms_equal':True,'source_verifier_code_bound_to_completed_confirmation':True,'splits':validation,'limits':['Canonical problem identities and original structural checks are validated; this export does not regenerate every exhaustive answer catalogue or rerun model admission.','Original source split names and auxiliary Graph unique-answer rows are preserved; no test split is created.']})
 write_docs(output,manifest)
 return manifest

def write_docs(output,m):
 # These files are release documentation; original dataset features remain unmodified.
 import yaml
 header={'language':['en'],'task_categories':['text-generation'],'tags':['reasoning','reinforcement-learning','verifiable-rewards','modebench'],'pretty_name':'ModeBench','configs':m['configs']}
 yamltext=yaml.safe_dump(header,sort_keys=False,allow_unicode=True)
 intro='''# ModeBench

ModeBench measures both correctness and the variety of correct answers produced by language models. Each problem has an executable verifier and a canonical definition of answer identity, so distinct wording alone does not count as a new solution.

This release contains three frozen benchmark levels across **Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan**. It preserves every original feature, row, and row order. The `answer` field is a serialized executable specification, not a target answer to include in the model prompt.

[Level 1](#level-1) · [Level 2](#level-2) · [Level 3](#level-3) · [Loading](#loading) · [Evaluation](#evaluation) · [Manifest](MANIFEST.json) · [Validation](VALIDATION.json)

The accompanying [model and research-artifact archive](https://huggingface.co/od2961/maxent-grpo-models) contains experiment-specific model indexes and mappings to the paper results. Dataset publication does not imply completed Level-3 training results.

<a id="level-1"></a>
## Level 1

The original benchmark uses 384 training problems and 128 primary evaluation problems per domain. PantryPlan additionally has its original 64-row development split. Graph Coloring also includes a separate 128-row `unique_answer` evaluation subset; it is published as the auxiliary `level1_graph_coloring_unique_answer` configuration and is not the primary multi-answer paper evaluation.

<a id="level-2"></a>
## Level 2

Level 2 is the frozen `modebench_harder_v2_matched_r5` release. It changes problem structure while retaining the verifier and canonicalizer contracts. Each domain has 384 training, 128 development, and 128 evaluation rows. Support histograms follow the recorded matched reference construction; refer to the original identity manifest for the precise reference splits and scaling factors.

<a id="level-3"></a>
## Level 3

Level 3 is the E122-frozen `modebench_level3_matched_v3` release, with 384 training, 128 development, and 128 evaluation rows per domain. It preserves Level-2 support histograms. Admission used adaptive second-round matching to fixed historical Level-1 measurements: Graph and Python have fresh v3 confirmation, while Countdown, MathIR, and Pantry retain prior split bytes and confirmation evidence. This is not a claim of statistical equivalence, or of five fresh same-round confirmations.

The original [identity manifest](provenance/source_data/modebench_level3_matched_v3/identity.json) still records `pending_fresh_candidate_confirmation`. That historical record is preserved verbatim. The later [confirmation report](provenance/confirmation_report.json) records `matched_fixed_reference` and binds that exact dataset identity. Neither record has been rewritten for publication.

<a id="loading"></a>
## Loading

```python
from datasets import load_dataset

ds = load_dataset("od2961/ModeBench", "level2_countdown")
train = ds["train"]
development = ds["dev"]
evaluation = ds["eval"]

# Raw problem text goes to the policy. Keep the executable answer spec private
# from the policy and use it only in the verifier.
problem = evaluation[0]["problem"]
reference_spec = evaluation[0]["answer"]
```

Use `level1_`, `level2_`, or `level3_` followed by `graph_coloring`, `countdown`, `python_factors`, `mathir`, or `pantry_plan`. Only existing splits are exposed; Level 1 generally has no `dev` split. No `test` split is invented. The Parquet files load through the standard datasets loader without remote dataset code. Pin a dataset revision for reproducible experiments.

<a id="evaluation"></a>
## Evaluation

Use the original held-out `eval` rows and the frozen prompt, decoding, and verifier settings for a stated comparison. Use `dev` only where the original release provides it. Do not fit prompts, choose checkpoints, or tune methods on evaluation outcomes; training, development, and evaluation roles remain separate after public release.

The same executable validation produces acceptance and canonical identity. Report correctness and distinct correct identities separately: with K sampled answers, pass@K is whether any answer verifies, and distinct@K counts unique verified canonical keys. The reported main benchmark uses K=8; the papers specify the seeds, draws, and training checkpoints. [Evaluation and prompt guide](EVALUATION.md) documents the supplied verifier interface and its versioned source.

The support-count metadata and reference specifications are for construction, checking, and analysis. Do not supply gold support catalogues or hidden evaluation references to the policy. Original prompt strings are retained; [prompt formatting](PROMPTS.md) distinguishes raw problems from the model-specific chat and decoding interfaces.

### What `answer_mode_count` does and does not tell you

`answer_mode_count` is the number of distinct modes the verifier **certifies**, not the number a model can be expected to produce. Treating it as an achievable ceiling will overstate how much diversity a model is losing. At 512 draws per problem, one frontier deployment reaches roughly 2.5 of Graph Coloring's 6.3 certified modes, 3.5 of Countdown's 4.5, 6.9 of PantryPlan's 16.9, and 3.8 of Python Factors' several hundred. Those reached figures come from 32 problems per domain; the certified means are over the full 128-row evaluation split.

**MathIR is a special case, and the gap there is structural rather than a sampling limit.** Its canonical key is the *state trajectory* an action program executes, so two programs that pass through the same equations are one mode and a program that detours through extra equations is a different one. Every certified mode of a MathIR problem ends at the same equation with the same solution: the modes are alternative derivations of one answer, not alternative answers. Each problem certifies exactly 5, but they are not 5 equally short routes. At Level 1 they divide into 192 two-step routes, 128 three-step and 320 at the four-step maximum across the 128 evaluation problems, so half of the certified support is as long as the interface allows, and most of those longest routes apply an action and then its inverse.

No model measured so far has produced one. Across 27,450 verified MathIR draws from local checkpoints and hosted frontier deployments, none is a four-step route. Level 2 additionally leaves every problem a single shortest route, so a policy that solves the problem the short way has exactly one mode available and scores zero on any success-conditional diversity metric without having concentrated at all.

If you are measuring diversity on ModeBench, report MathIR separately or alongside a MathIR-excluded aggregate. If you are building on the benchmark, note that a MathIR variant whose modes were distinct answers rather than distinct derivations would test diversity methods that this construction cannot.

## Files and provenance

`MANIFEST.json` records every source identity, feature schema, ordered-row digest, split mapping, and Parquet digest. `VALIDATION.json` records full row and feature round trips, unchanged source files, canonical problem-identity checks, and support matching. `code/oat_drgrpo/` preserves verifier and template source files byte for byte; `provenance/` preserves original identities and construction/admission records.

## License provenance

The original repository's Apache-2.0 source-code license is preserved in [licenses/source_repository_APACHE_2.0.txt](licenses/source_repository_APACHE_2.0.txt), and copied source copyright notices remain intact. The frozen dataset roots contain no separate dataset-specific license declaration; this export does not invent new dataset license terms. PantryPlan ingredient provenance records USDA FoodData Central Foundation Foods under CC0/public-domain source terms in [pantry_ingredient_source.json](provenance/pantry_ingredient_source.json).
'''
 (output/'README.md').write_text('---\n'+yamltext+'---\n\n'+intro)
 lines=['# Configurations and original split mappings','','| Config | Split | Rows | Original serialized subset |','| --- | --- | ---: | --- |']
 for s in m['splits']:lines.append(f"| `{s['config_name']}` | `{s['split']}` | {s['rows']} | `{s['original_datasetdict_split']}` |")
 (output/'SPLITS.md').write_text('\n'.join(lines)+'\n')
 (output/'PROMPTS.md').write_text('''# Prompt formatting

Every Parquet row preserves the original `problem` string and the original serialized `answer` specification. Send the problem through the selected model/experiment's frozen template; do not concatenate the answer specification or support metadata into the prompt.

For the original Qwen Level-1 training interface, Graph, Countdown, Python, and MathIR use `qwen_boxed`; Pantry uses `qwen_pantry_support_mask`. The copied `code/oat_drgrpo/templates.py` preserves the exact template text and Qwen/Falcon role markers. `provenance/source_code/ops/check_paper_domain_prompts.py` binds the original paper examples to those strings.

Level-2/Level-3 admission uses the versioned interface in `provenance/source_code/ops/evaluate_modebench_level2_viability.py` and `evaluate_modebench_level3.py`: Graph uses the direct boxed interface, while the other domains use the registered hybrid solver interface and their legal-output constraints. That admission interface and its model-native tokenizer rendering are distinct from a training experiment's fixed template. Copying a dataset alone does not reproduce a decoder or inference-engine configuration.

The original training and evaluation launchers remain the source of model-specific precision, length limits, guided-decoding profiles, draw seeds, and selected model checkpoints. The files here preserve the shared prompt/template implementation and the dataset rows; they do not silently standardize these experiment-specific choices.
''')
 (output/'EVALUATION.md').write_text('''# Executable verification and canonical identity

For responses already decoded to the task's verifier surface, the supplied `code/oat_drgrpo/math_grader.py` entry point is `validated_modebench_outcome_key(response, row["answer"])`. It returns a canonical key only when that same response passes the executable task verifier; invalid responses return `None`.

| Domain | Executable acceptance | Canonical identity |
| --- | --- | --- |
| Graph Coloring | Parse assignments; preserve fixed colors and satisfy every edge | Complete color vector |
| Countdown | Restricted arithmetic AST, exact operand multiplicities and exact target value | Normalized expression AST with registered commutative normalization |
| Python Factors | Restricted function AST executed in the bounded external worker on the specified cases | Executed return vector |
| MathIR | Restricted action menu expanded and executed using exact equation transformations | Executed normalized state trajectory |
| PantryPlan | Bounded ingredient quantities, serving steps, mass/nutrient constraints and dietary restrictions | Ingredient support |

Install the versioned dependencies recorded in `MANIFEST.json`/`requirements-verifier.txt`; prepend the release's `code` directory to Python's import path. The code is supplied as inspectable ordinary source, not automatically executed by the datasets loader.

```python
import sys
sys.path.insert(0, "path/to/ModeBench/code")
from oat_drgrpo.math_grader import validated_modebench_outcome_key

key = validated_modebench_outcome_key(response, row["answer"])
verified = key is not None
```

Policy action adapters must run before this grader when the registered interface uses them. In particular, Pantry's six-bit support mask is not an allocation string: use `decode_pantry_support_mask(mask, json.loads(row["answer"]))` from `oat_drgrpo.pantry_support_action`, then grade the returned allocation. The frozen adapter searches only the chosen support's registered quantity grid and returns an invalid sentinel when infeasible; it does not consult a gold support catalogue. `canonical_actions.py` preserves the other registered action-to-verifier adapters.

Python Factors retains its original separate, syntax-restricted worker. Use the supplied implementation for model outputs rather than evaluating arbitrary response strings in a notebook. The code also contains imported legacy task support needed by the original module; this release's dataset configurations contain only the five listed domains.

All copied verifier/template dependencies are authenticated against the completed Level-3 confirmation's source hashes. The release validation checks canonical *problem* identity uniqueness and disjointness, row/feature preservation, and the original structural/support-histogram records. It does not regenerate every exhaustive answer catalogue, retrain models, or claim a new model-admission result.
''')
 (output/'requirements-verifier.txt').write_text('\n'.join(f'{name}=={version}' for name,version in m['runtime'].items())+'\n')
 (output/'LICENSE_PROVENANCE.md').write_text('# Existing source terms\n\n'+json.dumps(m['license_provenance'],indent=2,sort_keys=True)+'\n')
 copy_file(Path(__file__).resolve(),output/'tools/export_modebench_hf_release.py')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();m=export(a.output.resolve());print(json.dumps({'status':'prepared','output':str(a.output.resolve()),'configs':m['config_count'],'splits':m['split_count'],'rows':m['row_count'],'parquet_bytes':sum(s['parquet_bytes'] for s in m['splits'])},indent=2))
