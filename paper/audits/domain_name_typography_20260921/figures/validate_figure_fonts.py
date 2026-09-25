from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections,hashlib,json,re,subprocess,xml.etree.ElementTree as ET
ROOT=Path('/n/fs/similarity/maxent-grpo');BASE=Path('/tmp/paper-domain-typography-20260921');OUT=ROOT/'paper/audits/domain_name_typography_20260921/figures';old=json.loads((OUT/'inventory.json').read_text())
pattern=re.compile(r'\b(?:Graph|Countdown|Python|MathIR|PantryPlan|Pantry)\b');mono=re.compile(r'mono|courier|typewriter|cmtt|lmtt|lmtk',re.I)
def inspect(row):
 p=ROOT/'paper'/row['figure'];xml=BASE/'live_xml'/(p.stem+'.xml');xml.parent.mkdir(exist_ok=True)
 subprocess.run(['pdftohtml','-xml','-hidden','-i',str(p),str(xml)],capture_output=True,check=True)
 doc=ET.parse(xml);fonts={e.attrib['id']:e.attrib['family'] for e in doc.findall('.//fontspec')};labels=[]
 nodes=doc.findall('.//text')
 for e in nodes:
  text=''.join(e.itertext())
  for match in pattern.finditer(text):labels.append({'domain':match.group(),'text':text,'font':fonts[e.attrib['font']],'mono':bool(mono.search(fonts[e.attrib['font']]))})
 # Rotated mathtext can be serialized one letter per text object. Reconstruct
 # adjacent same-font vertical runs; retain character-level font evidence.
 index=0
 while index<len(nodes):
  e=nodes[index];text=''.join(e.itertext())
  if len(text)==1 and float(e.attrib['width'])==0:
   group=[e];j=index+1
   while j<len(nodes):
    nxt=nodes[j];nt=''.join(nxt.itertext())
    if len(nt)!=1 or nxt.attrib['font']!=e.attrib['font'] or nxt.attrib['left']!=e.attrib['left'] or float(nxt.attrib['width'])!=0:break
    group.append(nxt);j+=1
   joined=''.join(''.join(x.itertext()) for x in group)
   for match in pattern.finditer(joined):labels.append({'domain':match.group(),'text':joined,'font':fonts[e.attrib['font']],'mono':bool(mono.search(fonts[e.attrib['font']])),'xml_encoding':'adjacent rotated glyphs'})
   index=j
  else:index+=1
 before=BASE/'archive'/row['figure']
 def numbers(file):
  text=subprocess.run(['pdftotext','-layout',str(file),'-'],capture_output=True,text=True,check=True).stdout
  return collections.Counter(re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',text))
 bn,an=numbers(before),numbers(p)
 def size(file):
  t=subprocess.run(['pdfinfo',str(file)],capture_output=True,text=True,check=True).stdout
  return next(l.strip().replace('Page size:','').strip() for l in t.splitlines() if 'Page size:' in l)
 r={'figure':row['figure'],'labels':labels,'nonmono':sum(not v['mono'] for v in labels),'domains_before':len(row['domain_labels']),'domains_after':len(labels),'numbers_unchanged':bn==an,'numeric_removed':list((bn-an).elements()),'numeric_added':list((an-bn).elements()),'before_size':size(before),'after_size':size(p)}
 oldside=BASE/'before/paper'/row['figure'];oldside=oldside.with_suffix('.json');side=p.with_suffix('.json')
 if oldside.exists() and side.exists():
  a=json.loads(oldside.read_text());b=json.loads(side.read_text());r['changed_metadata_topkeys']=[k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)]
 return r
with ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(inspect,old['figures']))
result={'figures':len(rows),'domain_mentions':sum(len(r['labels']) for r in rows),'remaining_nonmono':sum(r['nonmono'] for r in rows),'records':rows}
(OUT/'after_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='records'})
for r in rows:
 if r['domains_before'] or not r['numbers_unchanged']:print(r['figure'],'remaining',r['nonmono'],'counts',r['domains_before'],r['domains_after'],'NUM',r['numbers_unchanged'],'removed',r['numeric_removed'],'added',r['numeric_added'],'geometry',r['before_size'],r['after_size'],'metadata',r.get('changed_metadata_topkeys'))
