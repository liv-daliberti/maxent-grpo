from pathlib import Path
from zipfile import ZipFile
import subprocess, shutil, re, json, hashlib, sys
root=Path('/n/fs/similarity/maxent-grpo');sys.path.insert(0,str(root))
from ops.check_paper_main_length import check_main_length
p=Path('/tmp/paper-domain-typewriter-20260921');paper=p/'release';audit=root/'paper/audits/domain_name_typography_20260921'
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def run(args, **kw):return subprocess.run(args,check=True,**kw)
s=(paper/'main.tex').read_text();assert s==(root/'paper/main.tex').read_text()
log=(p/'release-build/main.log').read_text(errors='replace');aux=(p/'release-build/main.aux').read_text()
bad=[l for l in log.splitlines() if re.search(r'undefined|multiply defined|Overfull|destination with the same identifier|Collision|Label\(s\) may have changed',l,re.I)];assert not bad,bad
length=check_main_length(p/'release-build/main.pdf',p/'release-build/main.aux')
labels={k:(n,pg) for k,n,pg in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^{}]*)\}\{([^{}]*)\}',aux)}
heads=set(re.findall(r'\\contentsline \{(?:section|subsection|subsubsection)\}\{\\numberline \{([A-Z](?:\.\d+)*)\}',aux))
entries=re.findall(r'\\apx\w*\{\\ref\{([^}]+)\}',s);assert not heads-{labels.get(e,('',))[0] for e in entries}
full=run(['pdftotext','-layout',str(p/'release-build/main.pdf'),'-'],capture_output=True,text=True);assert not full.stderr
pages=full.stdout.split('\f')
for ext in ['pdf','aux','bbl','blg','log','out','toc']:
 src=p/f'release-build/main.{ext}'
 if src.exists():shutil.copy2(src,paper/f'main.{ext}')
shutil.copy2(p/'release-build/main.pdf',paper/'main-with-figures.pdf')
for name in ['body.pdf','body-meta.pdf','body-qdf.pdf','body-clean.pdf']:(p/name).unlink(missing_ok=True)
(p/'body-metadata.txt').write_text('InfoBegin\nInfoKey: Title\nInfoValue: Measuring and Mitigating Solution Mode Collapse in RLVR\nInfoBegin\nInfoKey: Author\nInfoValue: Anonymous Authors\n')
run(['pdftk',str(p/'release-build/main.pdf'),'cat','1-9','output',str(p/'body.pdf')])
run(['pdftk',str(p/'body.pdf'),'update_info_utf8',str(p/'body-metadata.txt'),'output',str(p/'body-meta.pdf')])
run(['qpdf','--qdf','--object-streams=disable',str(p/'body-meta.pdf'),str(p/'body-qdf.pdf')])
j=json.loads(run(['qpdf','--json',str(p/'body-qdf.pdf')],capture_output=True,text=True).stdout)
objects=j.get('objects')
if objects is None:
 objects={k.removeprefix('obj:'):v.get('value',v) for part in j.get('qpdf',[]) if isinstance(part,dict) for k,v in part.items() if k.startswith('obj:')}
badlinks=set()
for ref,v in objects.items():
 if not isinstance(v,dict) or v.get('/Subtype')!='/Link':continue
 dest=v.get('/Dest',v.get('/A',{}).get('/D'))
 if isinstance(dest,list) and dest and dest[0] is None:badlinks.add(ref)
raw=(p/'body-qdf.pdf').read_bytes();removed=0;bodypages=0
for ref,v in objects.items():
 if not isinstance(v,dict) or v.get('/Type')!='/Page':continue
 bodypages+=1;refs=[r for r in v.get('/Annots',[]) if r in badlinks]
 if not refs:continue
 m=re.search(rb'(?m)^'+ref.split()[0].encode()+rb' 0 obj\n(.*?)\nendobj',raw,re.S);assert m
 body=m.group(1);a=re.search(rb'/Annots\s*\[(.*?)\]',body,re.S);assert a;arr=a.group(1)
 for r in refs:
  arr,c=re.subn(rb'(?<!\d)'+re.escape(r.encode())+rb'(?!\d)',b'',arr);assert c==1;removed+=c
 body=body[:a.start(1)]+arr+body[a.end(1):];raw=raw[:m.start(1)]+body+raw[m.end(1):]
assert bodypages==9 and removed==len(badlinks)
(p/'body-edited.qdf').write_bytes(raw)
with (p/'body-fixed.qdf').open('wb') as out:run(['fix-qdf',str(p/'body-edited.qdf')],stdout=out)
run(['qpdf',str(p/'body-fixed.qdf'),str(p/'body-clean.pdf')])
for args in [['qpdf','--check',str(p/'body-clean.pdf')],['pdfinfo',str(p/'body-clean.pdf')],['pdftotext','-layout',str(p/'body-clean.pdf'),'-']]:
 r=run(args,capture_output=True,text=True);assert not r.stderr,r.stderr
maintext=run(['pdftotext','-f','1','-l','9','-layout',str(p/'release-build/main.pdf'),'-'],capture_output=True,text=True).stdout;assert r.stdout==maintext
for label,filename in [('before','body-meta.pdf'),('after','body-clean.pdf')]:
 r=run(['pdftoppm','-r','72','-png',str(p/filename),str(p/f'body-{label}')],capture_output=True,text=True)
 if label=='after':assert not r.stderr,r.stderr
 else:assert set(r.stderr.splitlines())<={'Syntax Warning: Bad annotation destination'},r.stderr
for i in range(1,10):assert (p/f'body-before-{i}.png').read_bytes()==(p/f'body-after-{i}.png').read_bytes(),i
shutil.copy2(p/'body-clean.pdf',paper/'main-body.pdf');print('PDF variants assembled; main length and standalone export verified.',flush=True)
with (p/'package.log').open('w') as out:run([sys.executable,'-c',"from pathlib import Path; import ops.package_iclr_overleaf as m; m.PAPER=Path('/tmp/paper-domain-typewriter-20260921/release'); m.main()",'--output',str(paper/'iclr2027_overleaf.zip')],cwd=root,stdout=out,stderr=subprocess.STDOUT)
assert 'standalone compile OK' in (p/'package.log').read_text()
checked=[]
with ZipFile(paper/'iclr2027_overleaf.zip') as z:
 for name in z.namelist():
  f=paper/name
  if f.is_file() and name not in {'README.md','PACKAGE_MANIFEST.json','latexmkrc'}:
   assert z.read(name)==f.read_bytes(),name;checked.append(name)
 assert z.read('reference/compiled-main.pdf')==(paper/'main.pdf').read_bytes()
assert (paper/'main.tex').read_text()==s
assert (paper/'main.pdf').read_bytes()==(p/'release-build/main.pdf').read_bytes()==(paper/'main-with-figures.pdf').read_bytes()
incoming_main=run(['pdftotext','-f','1','-l','9','-layout',str(p/'before.pdf'),'-'],capture_output=True,text=True).stdout
record=(p/'release-build/main.fdb_latexmk').read_text();input_count=0;input_mismatch=[]
for name,digest in re.findall(r'^\s+"([^"]+)"\s+[0-9.]+\s+\d+\s+([a-f0-9]{32})\s',record,re.M):
 f=Path(name)
 if f.is_absolute():continue
 f=paper/f
 if not f.is_file():continue
 if f.suffix in {'.aux','.out','.toc','.log','.bbl','.blg','.pdf'} and not name.startswith('figures/'):continue
 input_count+=1
 if hashlib.md5(f.read_bytes()).hexdigest()!=digest:input_mismatch.append(name)
assert input_count>80 and not input_mismatch,(input_count,input_mismatch)
v={'scope':'All rendered benchmark-domain names in the main paper and appendix, including figures', 'appendix_heading_count':len(heads),'missing_appendix_toc_entries':[],'unresolved_reference_duplicate_overfull_collision_diagnostics':bad,'length_contract':length,'full_pdf_pages':length['pdf_pages'],'overleaf_independent_compile_passed':True,'packaged_source_assets_reference_match':True,'packaged_snapshot_files_checked':len(checked),'release_uses_fixed_source_snapshot':True,'experiments_run':False,'bootstrap_reruns':False,'latexmk_recorded_input_hash_count':input_count,'latexmk_recorded_input_hashes_match_packaged_snapshot':True,'main_body_export':{'dangling_links_removed':removed,'page_count':bodypages,'text_matches_full_pdf_main':True,'all_nine_pages_pixel_identical_after_link_repair':True,'pdf_diagnostics':[]}}
files=['main.tex','main.pdf','main-with-figures.pdf','main-body.pdf','iclr2027_overleaf.zip']
v['deliverable_sha256']={name:sha(paper/name) for name in files if (paper/name).is_file()}
for ext in ['pdf','aux','bbl','blg','log','out','toc']:
 src=paper/f'main.{ext}'
 if src.exists():shutil.copy2(src,root/'paper'/src.name)
for name in ['main-with-figures.pdf','main-body.pdf','iclr2027_overleaf.zip']:shutil.copy2(paper/name,root/'paper'/name)
v['concurrent_live_source_differences_at_release']=[name for name in checked if (root/'paper'/name).is_file() and (root/'paper'/name).read_bytes()!=(paper/name).read_bytes()]
v['snapshot_source_sha256']={name:sha(paper/name) for name in checked}
(audit/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
print('Final checks passed:',len(checked),'archive files and',input_count,'compiler input hashes match.',flush=True)
print('Full PDF pages:',length['pdf_pages'],'Main pages:',length['main_pages'],'Appendix headings:',len(heads),flush=True)
