from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections,hashlib,json,re,subprocess,xml.etree.ElementTree as ET,zipfile
ROOT=Path('/n/fs/similarity/maxent-grpo'); TMP=Path('/tmp/paper-domain-typography-20260921');OUT=ROOT/'paper/audits/domain_name_typography_20260921/figures';OUT.mkdir(parents=True,exist_ok=True)
pattern=re.compile(r'\b(?:Graph|Countdown|Python|MathIR|PantryPlan|Pantry)\b')
mono=re.compile(r'mono|courier|typewriter|cmtt|lmtt|lmtk',re.I)
archive=ROOT/'paper/iclr2027_overleaf.zip'
with zipfile.ZipFile(archive) as z:
 names=[x for x in z.namelist() if x.startswith('figures/') and x.endswith('.pdf')]
 for n in names:
  p=TMP/'archive'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def refs(obj,path=''):
 result=[]
 if isinstance(obj,dict):
  for k,v in obj.items():result+=refs(v,path+'/'+k)
 elif isinstance(obj,list):
  for i,v in enumerate(obj):result+=refs(v,path+'/'+str(i))
 elif isinstance(obj,str) and ('ops/' in obj or 'scripts/' in obj) and obj.endswith('.py'):result.append({'key':path,'path':obj})
 return result
def audit(n):
 p=TMP/'archive'/n; xml=TMP/'xml'/(p.stem+'.xml');xml.parent.mkdir(exist_ok=True)
 r=subprocess.run(['pdftohtml','-xml','-hidden','-i',str(p),str(xml)],capture_output=True,text=True,check=True)
 doc=ET.parse(xml);fonts={f.attrib['id']:dict(f.attrib) for f in doc.findall('.//fontspec')};labels=[]
 for page in doc.findall('.//page'):
  for el in page.findall('text'):
   text=''.join(el.itertext()); found=pattern.findall(text)
   if not found:continue
   f=fonts.get(el.attrib['font'],{});family=f.get('family','UNKNOWN')
   labels.append({'text':text,'domain_mentions':found,'font_family':family,'monospace':bool(mono.search(family)),'page':int(page.attrib['number']),'box':{k:el.attrib[k] for k in ['top','left','width','height']}})
 live=ROOT/'paper'/n;sidecar=live.with_suffix('.json');scripts=refs(json.loads(sidecar.read_text())) if sidecar.exists() else []
 pdffonts=subprocess.run(['pdffonts',str(p)],capture_output=True,text=True,check=True).stdout
 (OUT/(p.stem+'.fonts.txt')).write_text(pdffonts)
 return {'figure':n,'archive_sha256':sha(p),'live_matches_archive':live.exists() and sha(live)==sha(p),'domain_labels':labels,'domain_mentions':sum(len(x['domain_mentions']) for x in labels),'nonmono_domain_labels':sum(not x['monospace'] for x in labels),'nonmono_domain_mentions':sum(len(x['domain_mentions']) for x in labels if not x['monospace']),'sidecar':str(sidecar.relative_to(ROOT)) if sidecar.exists() else None,'sidecar_script_references':scripts}
with ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(audit,names))
summary={'archive':str(archive.relative_to(ROOT)),'archive_sha256':sha(archive),'figures':len(rows),'figures_with_domain_labels':sum(bool(r['domain_labels']) for r in rows),'figures_with_nonmono_domains':sum(bool(r['nonmono_domain_labels']) for r in rows),'domain_label_objects':sum(len(r['domain_labels']) for r in rows),'domain_mentions':sum(r['domain_mentions'] for r in rows),'nonmono_domain_labels':sum(r['nonmono_domain_labels'] for r in rows),'nonmono_domain_mentions':sum(r['nonmono_domain_mentions'] for r in rows),'all_live_match_archive':all(r['live_matches_archive'] for r in rows)}
(OUT/'inventory.json').write_text(json.dumps({'summary':summary,'figures':rows},indent=2)+'\n')
print(json.dumps(summary,indent=2))
for r in rows:
 print(r['figure'],r['nonmono_domain_labels'],'of',len(r['domain_labels']),'label objects',','.join(sorted({x['font_family'] for x in r['domain_labels']})), 'MATCH' if r['live_matches_archive'] else 'DIVERGED')
 print('  scripts',r['sidecar_script_references'][:5])
