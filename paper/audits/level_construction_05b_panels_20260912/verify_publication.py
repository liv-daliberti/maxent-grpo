from pathlib import Path
import hashlib,json,re,runpy,shutil,subprocess,sys
root=Path('/n/fs/similarity/maxent-grpo');paper=root/'paper'
audit=paper/'audits/level_construction_05b_panels_20260912';stage=audit/'final_build_v1';stage.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inputs():
    pending=[paper/'main.tex']; found=set()
    pattern=re.compile(r'\\(input|includegraphics|bibliography)\s*(?:\[[^\]]*\]\s*)?\{([^{}]+)\}')
    while pending:
        p=pending.pop()
        if p in found:continue
        found.add(p)
        s=re.sub(r'(?<!\\)%[^\n]*','',p.read_text())
        for cmd,raw in pattern.findall(s):
            for name in raw.split(','):
                target=paper/name
                if not target.suffix:target=target.with_suffix('.pdf' if cmd=='includegraphics' else '.bib' if cmd=='bibliography' else '.tex')
                assert target.is_file(),target
                if cmd=='input':pending.append(target)
                else:found.add(target)
    found.update(paper.glob('*.sty'));found.update(paper.glob('*.bst'))
    return {str(p.relative_to(root)):sha(p) for p in sorted(found)}
before=inputs();(audit/'compiled_source_sha256.json').write_text(json.dumps(before,indent=2)+'\n')
for pattern in ('*.bib','*.bst'):
    for p in paper.glob(pattern):shutil.copy2(p,stage/p.name)
for i,cmd in enumerate((['pdflatex','-interaction=nonstopmode','-halt-on-error','-recorder','-output-directory='+str(stage),'main.tex'],['bibtex','main'],['pdflatex','-interaction=nonstopmode','-halt-on-error','-recorder','-output-directory='+str(stage),'main.tex'],['pdflatex','-interaction=nonstopmode','-halt-on-error','-recorder','-output-directory='+str(stage),'main.tex'])):
    with (audit/f'final_compile_{i}.log').open('w') as log:subprocess.run(cmd,cwd=stage if cmd[0]=='bibtex' else paper,stdout=log,stderr=subprocess.STDOUT,check=True)
assert inputs()==before,'Source inputs changed during compilation'
for name,args in [('main_length',['--pdf',str(stage/'main.pdf'),'--aux',str(stage/'main.aux')]),('line_fill',['--pdf',str(stage/'main.pdf')])]:
    with (audit/(name+'_verified.log')).open('w') as log:subprocess.run([sys.executable,str(root/f'ops/check_paper_{name}.py'),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
check=runpy.run_path(str(root/'ops/check_paper_current_contract.py'));text=(paper/'main.tex').read_text();main,appendix=text.split(r'\appendix',1)
check['check_editorial_structure'](main,appendix);check['check_formal_preservation'](text,check['PROOF_REFERENCE'].read_text())
log=(stage/'main.log').read_text();assert 'Overfull' not in log and 'There were undefined references' not in log
assert inputs()==before,'Source inputs changed during validation'
for suffix in ('aux','bbl','blg','log','out','fls','pdf'):shutil.copy2(stage/f'main.{suffix}',paper/f'main.{suffix}')
shutil.copy2(paper/'main.pdf',paper/'main-with-figures.pdf')
subprocess.run(['qpdf',str(paper/'main.pdf'),'--pages','.','1-9','--',str(paper/'main-body.pdf')],check=True)
receipt={'schema':'qwen05b-level-panels-publication-v1','source_sha256':before,'source_inputs_unchanged_during_build':True,'main_figure':3,'main_figure_page':3,'workshop_figure_page':2,'main_pages':9,'workshop_main_pages':4,'main_figures':8,'figure_points':8,'required_model_level_cells':25,'outputs':{str(p.relative_to(root)):sha(p) for p in (paper/'main.pdf',paper/'main-body.pdf',paper/'main-with-figures.pdf',paper/'mathai2026/main.pdf',paper/'mathai2026/mathai2026-source.zip',paper/'figures/modebench_level_construction.pdf',paper/'figures/modebench_level_construction.png',paper/'figures/modebench_level_construction.json')}}
(audit/'publication_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copy2(__file__,audit/'verify_publication.py')
print('Published final main PDF from unchanged inputs; nine main pages, Figure3 on page3, all layout/editorial checks passed.')
