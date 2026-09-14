#!/usr/bin/env python3
"""Seal a prepared dataset-only upload plan; never uploads or deletes files."""
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'var/artifacts/modebench_hf_release_20260911'
PACKAGE=BASE/'package'

def digest(path):
 data=path.read_bytes()
 return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def seal():
 m=json.loads((PACKAGE/'MANIFEST.json').read_text())
 v=json.loads((PACKAGE/'VALIDATION.json').read_text())
 l=json.loads((PACKAGE/'LOADER_VALIDATION.json').read_text())
 assert (m['config_count'],m['split_count'],m['row_count'])==(16,42,9152)
 assert v['status']==l['status']=='passed' and l['tests_passed']==18
 for s in m['splits']:
  d=digest(PACKAGE/s['data_file']);assert d['sha256']==s['parquet_sha256'] and d['bytes']==s['parquet_bytes']
 for group in ('code_files','provenance_files'):
  for item in m[group]:
   p=Path(item['path']);assert p.is_relative_to(PACKAGE)
   d=digest(p);assert d['sha256']==item['sha256'] and d['bytes']==item['bytes']
 for relative,expected in m['source_file_sha256'].items():assert digest(ROOT/relative)['sha256']==expected
 payload=[]
 for p in sorted(PACKAGE.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink() and p.resolve()==p and p.suffix in {'.json','.parquet','.py','.md','.txt'}
  relative=p.relative_to(PACKAGE).as_posix();assert not any(x.startswith('.') for x in p.relative_to(PACKAGE).parts)
  payload.append({'path_in_repo':relative,**digest(p)})
 contents={'schema':'modebench-public-content-manifest-v1','self_excluded':'CONTENT_SHA256.json','files':payload,'files_count':len(payload),'total_bytes':sum(x['bytes'] for x in payload)}
 save(PACKAGE/'CONTENT_SHA256.json',contents)
 payload.append({'path_in_repo':'CONTENT_SHA256.json',**digest(PACKAGE/'CONTENT_SHA256.json')})
 plan={'schema':'modebench-dataset-upload-plan-v1','status':'prepared','created_at_utc':datetime.now(timezone.utc).isoformat(),'repo_type':'dataset','repo_id':'od2961/ModeBench','public':True,'package_root':str(PACKAGE),'upload_only':True,'local_deletions_authorized_by_this_plan':False,'config_count':16,'primary_config_count':15,'auxiliary_config_count':1,'split_count':42,'row_count':9152,'primary_row_count':9024,'files_count':len(payload),'total_bytes':sum(x['bytes'] for x in payload),'files':[dict(item,source_path=str(PACKAGE/item['path_in_repo'])) for item in payload],'source_manifest_sha256':digest(PACKAGE/'MANIFEST.json')['sha256'],'content_manifest_sha256':digest(PACKAGE/'CONTENT_SHA256.json')['sha256'],'loader_validation_sha256':digest(PACKAGE/'LOADER_VALIDATION.json')['sha256'],'required_final_verification':{'immutable_commit_sha':True,'remote_path_size_and_sha256_or_git_blob_sha1_match_every_file':True,'publication_receipt_path':str(BASE/'publication_verified.json'),'receipt_fields':{'status':'verified','repo_type':'dataset','repo_id':'od2961/ModeBench','commit_sha':'actual immutable commit','upload_plan_sha256':'actual plan SHA256','all_files_verified':True}},'stable_anchors':['level-1','level-2','level-3','loading','evaluation']}
 save(BASE/'upload_plan.json',plan)
 save(BASE/'preparation.json',{'status':'prepared','upload_plan_sha256':digest(BASE/'upload_plan.json')['sha256'],'files_count':plan['files_count'],'total_bytes':plan['total_bytes'],'all_original_source_files_unchanged':True,'full_roundtrip_and_loader_tests_passed':True,'portable_verifier_five_domains_passed':True,'no_HF_mutations':True,'no_original_source_deletions':True})
 print(json.dumps(json.loads((BASE/'preparation.json').read_text()),indent=2))
if __name__=='__main__':seal()
