from pathlib import Path
import re,json,hashlib
root=Path('/n/fs/similarity/maxent-grpo');a=root/'paper/audits/theory_examples_20260905';b=a/'before'
names=['group_mean','collapse','parameterization','natural_gradient','entropy','entropy_comparison','gradient_availability','replay','exemplar_bridge','optimizer','discovery','survival']
sections=[(a/(n+'_section.tex')).read_text().rstrip()+'\n\n' for n in names]
intro=(a/'theory_intro.tex').read_text().rstrip()+'\n\n'
newtheory=intro+''.join(sections)
start=r'\section{Why On-Policy RL Can Lose Modes and Verified Replay Retains Them}';end=r'\section{Supporting Measurements and Direct Comparators}'
formal=re.compile(r'\\begin\{(theorem|lemma|corollary|proof)\}.*?\\end\{\1\}',re.S)
base=(b/'paper/main.tex').read_text();oldtheory=base[base.index(start):base.index(end,base.index(start))]
assert [m.group(0) for m in formal.finditer(oldtheory)]==[m.group(0) for m in formal.finditer(newtheory)],'Formal statement or proof changed'
oldcites=re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{[^}]+\}',oldtheory)
newcites=re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{[^}]+\}',newtheory);assert oldcites==newcites,'Citation sequence changed'
for name in ['paper/main.tex','paper/mathai2026/appendix.tex']:
 s=(b/name).read_text();left=s.index(start);right=s.index(end,left);replacement=newtheory
 if name.endswith('appendix.tex'):
  replacement=replacement.replace(r'of Section~\ref{sec:collapse}',r'defined in Appendix~\ref{app:metrics}').replace(r'Section~\ref{sec:maxrl-method}',r'Appendix~\ref{app:binary-maxrl}')
 (root/name).write_text(s[:left]+replacement+s[right:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
snap=json.loads((b/'paper/mathai2026/snapshot.json').read_text());oldsha=snap['source_sha256'];snap['source_sha256']=sha(root/'paper/main.tex')
snap['correction_history'].append({'date':'2026-09-05','kind':'worked examples and reader progression throughout the theory appendix','previous_source_sha256':oldsha,'source_sha256':snap['source_sha256'],'experimental_results_changed':False,'figures_changed':False,'bibliography_changed':False,'main_text_changed':False,'formal_statements_and_proofs_unchanged':True,'changes':['recurring three-correct-mode categorical example','worked illustrations throughout all twelve theory subsections','illustrations in all four survival-certificate subsubsections','explicit distinction between illustrative values, guaranteed floors and measured training results','reader roadmap and transitions following the proof chain'],'audit':'../audits/theory_examples_20260905.md'})
(root/'paper/mathai2026/snapshot.json').write_text(json.dumps(snap,indent=2)+'\n')
print('Integrated examples throughout 12 theory subsections; preserved',len(list(formal.finditer(oldtheory))),'formal/proof blocks and',len(oldcites),'citation commands.')
