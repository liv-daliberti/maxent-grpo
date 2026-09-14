#!/usr/bin/env python3
"""Merge ten paired viability receipts into the single final fairness report."""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile
from pathlib import Path
DOMAINS=('countdown','graph_coloring','python_factors','mathir','pantry')
MODELS=('qwen-0.5b','falcon-1b')
def atomic(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'w') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
  os.replace(tmp,path)
 except BaseException: os.unlink(tmp); raise
def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--receipts-dir',type=Path,required=True); ap.add_argument('--output',type=Path); a=ap.parse_args()
 identity_path=a.data_root/'identity.json'
 structural=json.loads(identity_path.read_text())
 identity_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest()
 receipts={}; protocols={}; shared_sampling=None
 for domain in DOMAINS:
  receipts[domain]={}
  for model in MODELS:
   path=a.receipts_dir/f'{model}__{domain}.json'
   if not path.is_file(): raise FileNotFoundError(f'missing frozen receipt: {path}')
   row=json.loads(path.read_text())
   if row.get('schema')!='modebench-level2-paired-base-viability-v1' or row.get('domain')!=domain or row.get('model_label')!=model: raise RuntimeError(f'receipt identity drift: {path}')
   if row.get('level2_identity_sha256')!=identity_sha256:
    raise RuntimeError(f'receipt dataset identity mismatch: {path}')
   if row.get('information_boundary',{}).get('calibration_only') or row['sampling'].get('row_limit') != 0:
    raise RuntimeError(f'calibration receipt cannot support final admission: {path}')
   if row.get('information_boundary',{}).get('evaluation_prompts_loaded') is not False:
    raise RuntimeError(f'evaluation information boundary violated: {path}')
   expected_rows={'level1':structural['domains'][domain]['dev']['level1_reference_rows'],'level2':structural['split_sizes']['dev']}
   if any(row['results'][level].get('rows') != expected_rows[level] for level in ('level1','level2')):
    raise RuntimeError(f'final receipt row coverage mismatch (expected {expected_rows}): {path}')
   l1=float(row['results']['level1']['pass_at_8']); l2=float(row['results']['level2']['pass_at_8'])
   expected_checks={'level2_not_effectively_zero':l2>=.10,'level2_not_too_easy':l2<=.90,'level2_harder_than_level1':l2<l1}
   expected_status='pass' if all(expected_checks.values()) else 'fail'
   if row.get('checks')!=expected_checks or row.get('status')!=expected_status:
    raise RuntimeError(f'receipt admission predicates inconsistent: {path}')
   comparable={k:row['sampling'][k] for k in ('sample_count','temperature','top_p','max_tokens','max_model_len','prompt_template','prompt_profile','row_limit','dtype','syntax_profile','seed')}
   shared={k:comparable[k] for k in ('sample_count','temperature','top_p','max_tokens','prompt_template','row_limit','dtype')}
   if shared_sampling is None: shared_sampling=shared
   if shared!=shared_sampling: raise RuntimeError(f'shared sampling or response-budget mismatch: {path}')
   protocols.setdefault(domain,{})[model]=comparable
   receipts[domain][model]={'status':row['status'],'checks':row['checks'],'model':row['model'],'model_config_sha256':row['model_config_sha256'],'level1_rows':row['results']['level1']['rows'],'level1_rows_sha256':row['results']['level1']['rows_sha256'],'level1_pass_at_8':row['results']['level1']['pass_at_8'],'level2_rows':row['results']['level2']['rows'],'level2_rows_sha256':row['results']['level2']['rows_sha256'],'level2_pass_at_8':row['results']['level2']['pass_at_8'],'receipt':str(path)}
 domain_decisions={domain:'admit' if all(receipts[domain][m]['status']=='pass' for m in MODELS) else 'reject_or_revise' for domain in DOMAINS}
 admitted=all(x=='admit' for x in domain_decisions.values())
 final=dict(structural); final.update({'schema':'modebench-level2-final-admission-fairness-v1','status':'pass' if admitted else 'fail','decision':'admit_all_domains_for_treatment_training' if admitted else 'stop_before_treatment_training_and_revise_rejected_domains','frozen_base_model_viability':receipts,'domain_decisions':domain_decisions,'comparable_prompt_and_response_budget':{'contract':'Each frozen model/domain cell uses one recorded protocol identically for its paired Level-1 and Level-2 evaluation; shared sample count, temperature, top-p, generation-token budget, chat-template mode, dtype, and full-row setting are invariant across all cells. Domain/model-specific prompt, syntax, and context-length profiles are frozen before this final run.','shared':shared_sampling,'per_cell':protocols}})
 output=a.output or a.data_root/'admission_fairness_report.json'; atomic(output,final)
 print(json.dumps({'status':final['status'],'decision':final['decision'],'domain_decisions':domain_decisions,'output':str(output)},sort_keys=True))
if __name__=='__main__': main()
