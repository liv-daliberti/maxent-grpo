from pathlib import Path
import hashlib,json
root=Path('/n/fs/similarity/maxent-grpo'); audit=root/'paper/audits/theory_appendix_integration_20260905'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
s=json.loads((audit/'before/paper/mathai2026/snapshot.json').read_text())
oldsource=s['source_sha256']; oldbib=s['copied_files']['example_paper.bib']
for f,h in s['copied_files'].items():
 if f!='example_paper.bib': assert sha(root/'paper/mathai2026'/f)==h,f
s['source_sha256']=sha(root/'paper/main.tex')
s['copied_files']['example_paper.bib']=sha(root/'paper/mathai2026/example_paper.bib')
s['correction_history'].append({'date':'2026-09-05','kind':'integrate attributed theoretical extensions into both appendices','previous_source_sha256':oldsource,'source_sha256':s['source_sha256'],'previous_bibliography_sha256':oldbib,'bibliography_sha256':s['copied_files']['example_paper.bib'],'experimental_results_changed':False,'figures_changed':False,'workshop_main_text_changed':False,'changes':['conditional stochastic retention and sharper bounded-noise finite-horizon bound','adaptive admission hazards and retention with controlled bank changes','exact conditional-entropy and full-support replay objective comparison','exact categorical Fisher natural-gradient comparison','sharp per-mode KL certificates, entropy threshold, complete-response mapping and sampling inference','primary-source attribution and theory roadmap/scope cross-references'],'audit':'../audits/theory_appendix_integration_20260905.md'})
(root/'paper/mathai2026/snapshot.json').write_text(json.dumps(s,indent=2)+'\n')
print('Updated source/bibliography snapshot; all frozen empirical assets unchanged.')
