"""Render typography from stored figure/report JSON, never analyzer main routines."""
from pathlib import Path
from copy import deepcopy
import hashlib,importlib,json,re,shutil,subprocess,sys,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/'ops')]
WORK=Path('/tmp/paper-domain-typography-20260921/hosted');WORK.mkdir(parents=True,exist_ok=True)
AUDIT=Path(__file__).resolve().parent
BACKUP=Path('/tmp/paper-domain-typography-20260921/before/paper/figures')
FIG=ROOT/'paper/figures'
NAMES=['hosted_verified_breadth','gpt56_all_levels32_sampling_budget','frontier_level_grid','modebench_discovery_curves_frontier','modebench_discovery_curves_local','modebench_discovery_correct_budget_frontier','modebench_discovery_correct_budget_local','modebench_prompt_ablation_local']
SAVED={n:json.loads((BACKUP/(n+'.json')).read_text()) for n in NAMES}
CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p)}
def state(fig):
 out=[]
 for ax in fig.axes:
  out.append({'lines':[[np.asarray(l.get_xdata()).tolist(),np.asarray(l.get_ydata()).tolist()] for l in ax.lines],
              'collections':[{'offsets':np.asarray(c.get_offsets()).tolist(),'paths':[p.vertices.tolist() for p in c.get_paths()]} for c in ax.collections],
              'limits':[ax.get_xlim(),ax.get_ylim()]})
 return json.dumps(out,sort_keys=True)
import ops.paper_domain_figure_typography as typography
original_apply=typography.apply_domain_typography
def tracked(fig):
 before=state(fig);n=original_apply(fig);assert state(fig)==before
 CHECKS.append({'type':'plotted_arrays_unchanged_by_typography','changed_labels':n})
 return n
typography.apply_domain_typography=tracked
# Native hosted and budget renderers consume the saved measurement records.
for name,module in [('hosted_verified_breadth','plot_paper_hosted_breadth'),('gpt56_all_levels32_sampling_budget','plot_paper_gpt56_all_levels32_sampling')]:
 mod=importlib.import_module('ops.'+module);record=deepcopy(SAVED[name]);record.pop('outputs',None)
 record['renderer']=bind(ROOT/'ops'/f'{module}.py')
 if name=='hosted_verified_breadth':record['source']=bind(ROOT/record['source']['path'])
 mod.render(record,FIG/name)
# Level grid: load existing points, use its direct plotting function, preserve
# every metadata field except output hashes and renderer/source bindings.
mod=importlib.import_module('ops.plot_paper_frontier_level_grid');mod.style.apply_domain_typography=tracked
payload,cells=mod.load();fig,deployments,kept,ladder=mod.build(cells);old=SAVED['frontier_level_grid']
assert deployments==old['deployments'] and kept==old['average_domain_sets'] and ladder==old['levels_drawn']
mod.style.save(fig,FIG/'frontier_level_grid');mod.plt.close(fig)
record=deepcopy(old);record['builder']=bind(ROOT/'ops/plot_paper_frontier_level_grid.py');record['source']=bind(ROOT/record['source']['path'])
record['outputs']={ext:bind(FIG/('frontier_level_grid.'+ext)) for ext in ('pdf','png')}
(FIG/'frontier_level_grid.json').write_text(json.dumps(record,indent=2)+'\n')
# Direct discovery plotting only. Copy the exact existing report into scratch
# because the plotter authenticates its figure metadata against analysis.json.
mod=importlib.import_module('ops.analyze_modebench_discovery_curves');out=WORK/'discovery';out.mkdir(exist_ok=True)
report_path=ROOT/'paper/results/modebench_discovery_curves_20260911.json';shutil.copy2(report_path,out/'analysis.json');report=json.loads(report_path.read_text())
with plt.rc_context(matplotlib.rcParamsDefault):
 for key in ('frontier','local','correct_budget_frontier','correct_budget_local'):mod.plot_figure(report,key,out)
# Prompt ablation: suppress the unassigned frontier branch, retaining the exact
# local display-row selection and all local point/interval values.
mod=importlib.import_module('ops.analyze_modebench_prompt_ablation');out2=WORK/'prompt';out2.mkdir(exist_ok=True)
report_path=ROOT/'paper/results/modebench_prompt_ablation_20260911.json';shutil.copy2(report_path,out2/'analysis.json');report=json.loads(report_path.read_text());display_rows=mod.display_rows
mod.display_rows=lambda report,family,grading: [] if family=='frontier' else display_rows(report,family,grading)
with plt.rc_context(matplotlib.rcParamsDefault):
 mod.make_figures(report,out2)
for name in NAMES[3:]:
 out=out2 if name=='modebench_prompt_ablation_local' else WORK/'discovery'
 record=json.loads((out/(name+'.json')).read_text());old=SAVED[name]
 assert record['plotted_records']==old['plotted_records'],name
 assert record['report_sha256']==old['report_sha256'],name
 for ext in ('pdf','png'):shutil.copy2(out/(name+'.'+ext),FIG/(name+'.'+ext))
 record['outputs']={ext:bind(FIG/(name+'.'+ext)) for ext in ('pdf','png')}
 (FIG/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
# Metadata-only downstream bindings to the two edited plotting modules.
for stem,key in [('hosted_level_averages_20260911','cohort_builder'),('gpt56_all_levels32_sampling_20260913','validation_dependency')]:
 p=ROOT/'paper/results'/f'{stem}.json';record=json.loads(p.read_text());record[key]['sha256']=sha(ROOT/record[key]['path']);p.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
# Measurements and all descriptive fields must match retained figure records.
comparisons={}
for name in NAMES:
 new=json.loads((FIG/(name+'.json')).read_text());old=deepcopy(SAVED[name]);new2=deepcopy(new)
 for key in ('outputs','renderer','builder','source'):old.pop(key,None);new2.pop(key,None)
 assert old==new2,name
 comparisons[name]={'all_measurements_and_descriptive_fields_unchanged':True,'pdf_sha256':sha(FIG/(name+'.pdf')),'png_sha256':sha(FIG/(name+'.png'))}
# Read the actual embedded font of each domain-bearing PDF text span.
fonts={};pattern=re.compile(r'\b(?:Graph|Countdown|Python|MathIR|PantryPlan|Pantry)\b')
for name in NAMES:
 xml=WORK/(name+'.xml');subprocess.run(['pdftohtml','-xml','-hidden','-i',str(FIG/(name+'.pdf')),str(xml)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 tree=ET.parse(xml);matches=[]
 for page in tree.getroot().findall('page'):
  families={node.attrib['id']:node.attrib.get('family','') for node in page.findall('fontspec')}
  for node in page.findall('text'):
   text=''.join(node.itertext())
   for token in pattern.findall(text):
    family=families[node.attrib['font']];assert 'Mono' in family,(name,text,family)
    matches.append({'domain':token,'text':text,'font':family})
 # Rotated mixed labels are split into one XML text span per letter.
 for page in tree.getroot().findall('page'):
  families={node.attrib['id']:node.attrib.get('family','') for node in page.findall('fontspec')}
  texts=page.findall('text')
  for i in range(len(texts)-5):
   run=texts[i:i+6]
   if [''.join(n.itertext()) for n in run]==list('Python'):
    assert all('Mono' in families[n.attrib['font']] for n in run),name
    matches.append({'domain':'Python','text':'Python (six rotated character spans)','font':families[run[0].attrib['font']]})
 assert matches,name
 fonts[name]=matches
manifest={'status':'passed','families':len(NAMES),'domain_font_spans':sum(map(len,fonts.values())),'figures':comparisons,'embedded_domain_fonts':fonts,'plot_array_checks':CHECKS,'reports_and_point_estimates_recomputed':False,'bootstrap_or_experimental_runs':0}
(AUDIT/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('status','families','domain_font_spans','bootstrap_or_experimental_runs')},indent=2))
