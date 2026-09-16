from pathlib import Path
import hashlib,json,shutil,subprocess,re
root=Path('/n/fs/similarity/maxent-grpo');paper=root/'paper';audit=paper/'audits/levels_7b_main_60_appendix_20260912';stage=audit/'final_build_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=Path('/tmp/levels7b_current_contract_final.log').read_text()
assert 'Current paper contract passed:' in checks,'Full scientific contract has not passed yet'
before=json.loads((stage/'source_sha256.json').read_text())
assert all(sha(root/name)==value for name,value in before.items()),'Source changed after final compilation'
assert 'Main-length contract passed' in (stage/'main_length.log').read_text()
assert 'Line-fill contract passed' in (stage/'line_fill.log').read_text()
for ext in ['pdf','aux','bbl','blg','log','out','fls']:
 shutil.copy2(stage/f'main.{ext}',paper/f'main.{ext}')
shutil.copy2(paper/'main.pdf',paper/'main-with-figures.pdf')
subprocess.run(['qpdf',str(paper/'main.pdf'),'--pages','.','1-9','--',str(paper/'main-body.pdf')],check=True)
shutil.copy2('/tmp/levels7b_current_contract_final.log',audit/'scientific_contract.log')
shutil.copy2('/tmp/levels7b_workshop_build_final.log',audit/'workshop_build.log')
shutil.copy2('/tmp/levels7b_workshop_package_final.log',audit/'workshop_package.log')
shutil.copy2('/tmp/levels7b_main_build_final.log',audit/'shared_retained_checks.log')
files=[paper/'main.pdf',paper/'main-with-figures.pdf',paper/'main-body.pdf',paper/'mathai2026/main.pdf',paper/'mathai2026/mathai2026-source.zip']
files += [paper/'figures'/f'{stem}.{ext}' for stem in ['modebench_level_construction','modebench_base_levels_appendix'] for ext in ['pdf','png','json']]
figures={}
for edition in ['main','workshop']:
 aux=(paper/'main.aux' if edition=='main' else paper/'mathai2026/main.aux').read_text()
 figures[edition]={}
 for label in ['fig:level-construction','fig:base-levels-all-scales']:
  m=re.search(r'\\newlabel\{'+re.escape(label)+r'\}\{\{([^}]*)\}\{([^}]*)\}',aux);assert m,label
  figures[edition][label]={'number':int(m.group(1)),'page':int(m.group(2))}
record={'schema':'modebench-levels-7b-main-four-scales-appendix-publication-v1','main_scope':{'model':'Qwen2.5-7B-Instruct','levels':[1,2,3],'domains':5,'complete_cells':15,'responses':61440},'appendix_scope':{'models':['0.5B','3B','7B','14B'],'levels':[1,2,3],'domains':5,'complete_cells':60,'responses':245760},'main_subset_matches_appendix':True,'native_figure_tests_passed':23,'full_scientific_contract':'pass','main_main_pages':9,'workshop_main_pages':4,'figures':figures,'final_compilation_source_sha256':before,'final_sources_unchanged':True,'outputs':{str(p.relative_to(root)):sha(p) for p in files}}
(audit/'publication_verification.json').write_text(json.dumps(record,indent=2)+'\n')
s=(audit/'README.md').read_text().replace('Full main-paper integration checks remain in progress; see `BUILD_COORDINATION.md`.','The final main-paper scientific contract passes, including native figure reconstruction and the preserved formal chain. Its isolated final compilation uses unchanged inputs and passes all page-length and line-fill checks; the verified PDF and both review copies are promoted. `publication_verification.json` binds both paper PDFs, the standalone source bundle, and both figure asset families. See `BUILD_COORDINATION.md` for the concurrent build history.')
(audit/'README.md').write_text(s)
print(json.dumps({'status':'complete','figures':figures,'native_figure_tests_passed':23},indent=2))
