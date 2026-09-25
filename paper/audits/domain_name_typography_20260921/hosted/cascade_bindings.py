"""Refresh only current paper sidecar hashes downstream of the comparison JSON."""
from pathlib import Path
from graphlib import TopologicalSorter
from copy import deepcopy
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[4];A=Path(__file__).resolve().parent
ANCHOR='paper/results/frontier_comparison_20260911.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(value):
 p=Path(value)
 return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)
def bindings(v,ptr=''):
 if isinstance(v,dict):
  if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str):yield ptr+'/sha256',v,'sha256',canon(v['path'])
  if isinstance(v.get('source'),str) and isinstance(v.get('source_sha256'),str):yield ptr+'/source_sha256',v,'source_sha256',canon(v['source'])
  for k,x in v.items():yield from bindings(x,ptr+'/'+str(k))
 elif isinstance(v,list):
  for i,x in enumerate(v):yield from bindings(x,ptr+'/'+str(i))
files=list((ROOT/'paper/results').glob('*.json'))+list((ROOT/'paper/figures').glob('*.json'))
records={str(p.relative_to(ROOT)):json.loads(p.read_text()) for p in files}
refs={rel:[dest for _,_,_,dest in bindings(rec)] for rel,rec in records.items()}
active={ANCHOR}
while True:
 add={rel for rel,edges in refs.items() if rel not in active and any(dep in active for dep in edges)}
 if not add:break
 active|=add
order=list(TopologicalSorter({rel:set(refs[rel])&active for rel in active}).static_order())
assert order[0]==ANCHOR
pdf_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'paper/figures').glob('*.pdf')}
changes=[];all_checks=[]
for rel in order[1:]:
 p=ROOT/rel;original=p.read_text();data=json.loads(original);comparison=deepcopy(data);updates=[]
 for loc,node,key,dep in bindings(data):
  if dep in active:
   digest=sha(ROOT/dep)
   if node[key]!=digest:updates.append({'json_pointer':loc,'dependency':dep,'old_sha256':node[key],'new_sha256':digest});node[key]=digest
 for _,node,key,dep in bindings(comparison):
  if dep in active:node[key]=sha(ROOT/dep)
 assert data==comparison
 if updates:
  dest=A/'cascade_before'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(p,dest)
  assert p.read_text()==original,'Concurrent sidecar change: '+rel
  p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
  # Reverse exactly the authorized digest edits; every other JSON value must
  # match the pre-write record, including every measurement and interval.
  reverse=deepcopy(data);by_ptr={loc:(node,key) for loc,node,key,_ in bindings(reverse)}
  for update in updates:
   node,key=by_ptr[update['json_pointer']];node[key]=update['old_sha256']
  assert reverse==json.loads(original)
  changes.append({'path':rel,'updates':updates,'after_sha256':sha(p),'only_digest_fields_changed':True})
for rel in order:
 data=json.loads((ROOT/rel).read_text())
 for loc,node,key,dep in bindings(data):
  if dep in active:
   assert node[key]==sha(ROOT/dep),(rel,loc,dep)
   all_checks.append({'path':rel,'json_pointer':loc,'dependency':dep,'current':True})
assert pdf_hashes=={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'paper/figures').glob('*.pdf')}
result={'status':'passed','anchor':ANCHOR,'anchor_sha256':sha(ROOT/ANCHOR),'dependency_order':order,'changed_files':changes,'final_bindings':all_checks,'measurements_unchanged':True,'all_figure_pdfs_unchanged':True}
(A/'metadata_cascade_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed','changed_json_paths':[x['path'] for x in changes],'current_bindings_checked':len(all_checks),'figure_pdfs_unchanged':len(pdf_hashes)},indent=2))
