from pathlib import Path
import hashlib,importlib,json,re,sys,tempfile
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));AUDIT=Path(__file__).resolve().parent
from ops.paper_domain_typography import format_domain_names
EXCEPTIONS=('Python lambda',r'Python \texttt{lambda}','Python modulo')
owned=json.loads((AUDIT/'owned_files.json').read_text())
manifest={'tex':{},'render_checks':{},'sidecars':{},'language_exceptions':{}}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(s):return re.sub(r'\\texttt\{(Graph|Countdown|Countd\.|Python|MathIR|PantryPlan|Pantry)\}',r'\1',s)
for rel in owned['tex']:
 p=ROOT/rel;before=(AUDIT/'before'/rel).read_text();old=p.read_text()
 after=format_domain_names(old,exclude_phrases=EXCEPTIONS)
 assert plain(before)==plain(after),rel
 assert format_domain_names(after,exclude_phrases=EXCEPTIONS)==after
 p.write_text(after)
 manifest['tex'][rel]={'added_typewriter_spans':after.count(r'\texttt{')-before.count(r'\texttt{'),'only_domain_typography_changed':True,'all_other_bytes_preserved':True,'before_sha256':sha(AUDIT/'before'/rel),'after_sha256':sha(p)}
 for phrase in EXCEPTIONS:
  if phrase in after:manifest['language_exceptions'].setdefault(rel,[]).append(phrase)
pairs=dict(zip(owned['sidecars'],owned['emitters']))
for stem,script in pairs.items():
 p=ROOT/'paper/results'/f'{stem}.json';before=json.loads((AUDIT/'before'/'paper/results'/f'{stem}.json').read_text());data=json.loads(p.read_text());digest=sha(ROOT/script)
 if 'builder_sha256' in data:data['builder_sha256']=digest;before['builder_sha256']=digest
 elif 'builder' in data:data['builder']['sha256']=digest;before['builder']['sha256']=digest
 else:data['renderer']['sha256']=digest;before['renderer']['sha256']=digest
 dependency_updates=[]
 for key in ('cohort_builder','validation_dependency'):
  if key in data and data[key]!=before[key]:
   assert data[key]['path']==before[key]['path']
   assert data[key]['sha256']==sha(ROOT/data[key]['path'])
   before[key]=data[key].copy();dependency_updates.append(key)
 assert before==data,(stem,'unexpected non-renderer metadata or measurement change')
 p.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
 manifest['sidecars'][str(p.relative_to(ROOT))]={'only_builder_or_renderer_and_plot_dependency_sha256_changed':True,'plot_dependency_bindings_updated':dependency_updates,'generator':script,'generator_sha256':digest}

def record(stem):return json.loads((ROOT/'paper/results'/f'{stem}.json').read_text())
def same(rel,result):
 expected=(ROOT/rel).read_text();assert result==expected,rel
 manifest['render_checks'][rel]='byte-identical rendering from existing stored record'
for module,fn,stem,suffix in [
 ('build_frontier_temperature_paper','render','frontier_temperature_20260911',''),
 ('build_paper_gpt56_all_levels32_discovery','render_tex','gpt56_all_levels32_sampling_20260913',''),
 ('build_paper_gpt56sol_python_parity','render','gpt56sol_python_parity_20260918',''),
 ('build_paper_hosted_level_averages','render_table','hosted_level_averages_20260911',''),
 ('build_paper_hosted_level_averages','render_parity_table','hosted_level_averages_20260911','_parity'),
 ('build_paper_hosted_reasoning_off','render_appendix','hosted_reasoning_off_20260912','_appendix')]:
 mod=importlib.import_module('ops.'+module)
 same('paper/results/'+stem+suffix+'.tex',getattr(mod,fn)(record(stem)))
mod=importlib.import_module('ops.build_frontier_paper_comparison')
# The export constructs text from existing records. Suppress its independent
# figure renderer so this check does not redraw figures or mutate their assets.
mod.graph_figure=lambda *args,**kwargs: None
with tempfile.TemporaryDirectory(prefix='hosted-domain-render-') as temp:
 out=Path(temp);fig=out/'figures';fig.mkdir()
 mod.export(record('frontier_comparison_20260911'),out,fig,'frontier_comparison_20260911')
 for rel in owned['tex']:
  if Path(rel).name.startswith('frontier_comparison_'):
   same(rel,(out/Path(rel).name).read_text())
manifest['summary']={'fragments':len(owned['tex']),'new_typewriter_spans':sum(x['added_typewriter_spans'] for x in manifest['tex'].values()),'emitters':len(owned['emitters']),'rendered_fragments_checked':len(manifest['render_checks']),'sidecars_checked':len(manifest['sidecars']),'experiments_or_bootstrap_runs':0}
previous=json.loads((AUDIT/'validation.json').read_text()) if (AUDIT/'validation.json').exists() else {}
if 'generator_logic_checks' in previous:manifest['generator_logic_checks']=previous['generator_logic_checks']
(AUDIT/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest['summary'],indent=2))
