from pathlib import Path
from zipfile import ZipFile
import re,json,hashlib,collections,sys,subprocess,difflib
root=Path('/n/fs/similarity/maxent-grpo');sys.path.insert(0,str(root));p=Path('/tmp/paper-final-appendix-20260921')
from ops.check_paper_main_length import check_main_length
from ops.check_paper_line_fill import extract_blocks,without_terminal_identifier,natural_prose
b=(p/'before.tex').read_text();s=(p/'source/main.tex').read_text()
pat=r'(?<!\\)\\\[.*?(?<!\\)\\\]|\\begin\{(?P<env>equation\*?|align\*?|gather\*?)\}.*?\\end\{(?P=env)\}'
displays=lambda x:[m.group(0) for m in re.finditer(pat,x,re.S)]
assert displays(b)==displays(s)
oldlabels=re.findall(r'\\label\{([^}]+)\}',b);newlabels=re.findall(r'\\label\{([^}]+)\}',s)
assert [l for l in oldlabels if l!='tab:pinned-revisions']==newlabels
normalize=lambda x:re.sub(r'\\texttt\{(Graph|Countdown|Python|MathIR|PantryPlan|Pantry)\}',r'\1',x)
formal=r'\\begin\{(?P<env>theorem|lemma|corollary|proposition|assumption|proof)\}.*?\\end\{(?P=env)\}'
formalblocks=lambda x:[normalize(m.group(0)) for m in re.finditer(formal,x,re.S)]
assert formalblocks(b)==formalblocks(s)
initial=json.loads((p/'initial-source-manifest.json').read_text());current=json.loads((p/'final-source-manifest.json').read_text());changes=[n for n,h in current.items() if initial[n]!=h]
for n,h in current.items():
 if n not in {'README.md','latexmkrc'}:assert hashlib.sha256((root/'paper'/n).read_bytes()).hexdigest()==h,n
input_math_count=0
with ZipFile(p/'before.zip') as z:
 for n in current:
  if n.startswith('results/') and n.endswith('.tex') and n in z.namelist():
   old=z.read(n).decode();new=(p/'source'/n).read_text();assert displays(old)==displays(new),n;input_math_count+=len(displays(new))
a=(p/'build/main.aux').read_text();labels={k:(n,pg) for k,n,pg in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^{}]*)\}\{([^{}]*)\}',a)}
heads=set(re.findall(r'\\contentsline \{(?:section|subsection|subsubsection)\}\{\\numberline \{([A-Z](?:\.\d+)*)\}',a));entries=re.findall(r'\\apx\w*\{\\ref\{([^}]+)\}',s);assert not heads-{labels.get(e,('',))[0] for e in entries}
out=subprocess.run(['pdftotext','-layout',str(p/'build/main.pdf'),'-'],capture_output=True,text=True,check=True);assert not out.stderr
text=out.stdout;apx='\f'.join(text.split('\f')[14:]);assert not re.search(r'reasoning\s+modes|reasoning\s+mode\s+collapse|post[ -]?hoc|post[ -]?freeze|camera.ready|preregistered|manuscript\s+revision',apx,re.I)
assert not re.search(r'(?:Section|Sec\.|App\.|Appendix|Figure|Fig\.|Table|Theorem|Lemma|Corollary|Proposition|Equation)\s*[~\xa0 ]*\?\?',text)
operational=r'SHA.?256|sha256|requeue|job.?ID|run.?ID|deduplicat|source hashes|\bseed (?:43|44|45|46|47|55|56|57|58|59|70|71|72|73|74)\b'
operational_hits=[l for l in text.splitlines() if re.search(operational,l,re.I)];assert not operational_hits,operational_hits
log=(p/'build/main.log').read_text(errors='replace');bad=[l for l in log.splitlines() if re.search(r'undefined|multiply defined|Overfull|destination with the same identifier|Collision|Label\(s\) may have changed',l,re.I)];assert not bad,bad

def violations(pdf):
 out=[]
 for raw in extract_blocks(pdf):
  q=without_terminal_identifier(raw)
  if not natural_prose(q):continue
  frac=q.lines[-1].width/max(l.width for l in q.lines)
  if frac<.5:out.append({'page':q.page,'ending':q.lines[-1].text,'fraction':frac})
 return out
old=violations(p/'before.pdf');new=violations(p/'build/main.pdf')
inventory=json.loads((p/'figure-inventory.json').read_text());assert set(int(labels[x['label']][0]) for x in inventory)==set(range(1,41))
for row in inventory:
 assert int(labels[row['label']][0])==row['number']
 for asset in row['assets']+row['records']:assert (root/'paper'/asset).is_file(),asset
 for renderer in row['renderers']:assert (root/renderer).is_file(),renderer
v={'main_math_displays_preserved':len(displays(s)),'included_math_displays_preserved':input_math_count,'formal_statement_and_proof_blocks_preserved_ignoring_domain_font_markup':len(formalblocks(s)),'manuscript_labels_preserved':len(newlabels),'removed_unreferenced_metadata_label':'tab:pinned-revisions','changed_package_inputs':changes,'latest_live_snapshot_matches':True,'appendix_heading_count':len(heads),'broken_references_overfull_duplicate_diagnostics':bad,'operational_identifier_scan_hits':operational_hits,'figure_inventory_count':40,'manifest_paths_valid':True,'length_contract':check_main_length(p/'build/main.pdf',p/'build/main.aux'),'line_fill':{'before':len(old),'after':len(new),'remaining':new,'note':'This global typography check remains nonpassing. Some endings changed under the simultaneous domain-font formatting and the caption cuts; no clipping or overfull boxes occur. The one-symbol flag is a mathematical-star extraction artifact.'}}
(p/'final-preservation.json').write_text(json.dumps(v,indent=2)+'\n');(p/'main-changes.patch').write_text(''.join(difflib.unified_diff(b.splitlines(keepends=True),s.splitlines(keepends=True),fromfile='latest-main-before.tex',tofile='cleaned-main-after.tex')))
print('Preserved',len(displays(s)),'main math displays and',input_math_count,'included math displays;',len(formalblocks(s)),'formal blocks.')
print('Retained',len(newlabels),'labels;',len(heads),'TOCheadings; zero unresolved/overfull diagnostics.')
print('Pages',v['length_contract']['pdf_pages'],'main',v['length_contract']['main_pages'],'linefill flags',len(new))
