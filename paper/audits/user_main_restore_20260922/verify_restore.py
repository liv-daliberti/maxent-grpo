from pathlib import Path
import hashlib,json,re,runpy
root=Path('/n/fs/similarity/maxent-grpo');paper=root/'paper';audit=paper/'audits/user_main_restore_20260922'
p=paper/'main.tex';current=p.read_text();before=(audit/'paper__main.tex.before').read_text();user=(audit/'user_main_exact.tex').read_text();meta=json.loads((audit/'before.json').read_text())
assert hashlib.sha256(user.encode()).hexdigest()==meta['paste_sha256']
start='% BEGIN author-supplied main; only reference repairs and retained figure scaffolding\n';end='% END author-supplied main\n\n'
actual=current.split(start,1)[1].split(end,1)[0]
expected=user
repairs=json.loads((audit/'main_link_repairs.json').read_text())
for c in repairs:
 assert expected.count(c['old'])==c['count'],c
 expected=expected.replace(c['old'],c['new'])
user_edits=json.loads((audit/'user_requested_edits.json').read_text())
for c in user_edits:
 assert expected.count(c['old'])==c['count'],c
 expected=expected.replace(c['old'],c['new'])
fig=(audit/'retained_figure.tex').read_text().rstrip('\n').replace(r'\begin{figure}[!tb]',r'\begin{figure}[H]')
expected=expected.replace('\\clearpage\n\\subsubsection*{AI Use Statement}', '\\label{sec:main-end}\n\\clearpage\n\\subsubsection*{AI Use Statement}')
assert actual==expected,'Unexpected change to authoritative main'
assert current.split(start,1)[0]==before[:before.index('% Provider marks: one definition each')],'Preamble change'
suffix_marker='\\clearpage\n\\phantomsection\\label{sec:references}'
suffix=before[before.index(suffix_marker):]
theorem=(audit/'relocated_theorem.tex').read_text().rstrip('\n')
proof=r'\begin{proof}[Proof of Theorem~\ref{thm:objective-inertness}]'
suffix=suffix.replace(proof,theorem+'\n\n'+proof)
suffix=suffix.replace(r'\paragraph{Support thresholds.}',r'\paragraph{Support thresholds.}'+'\n'+r'\phantomsection\label{app:registered-endpoints}')
appendix_repairs=json.loads((audit/'appendix_link_repairs.json').read_text())
for c in appendix_repairs:suffix=suffix.replace(c['old'],c['new'])
suffix=suffix.replace(r'\input{results/semantic_current_summary_20260912.tex}',fig+'\n\n'+r'\input{results/semantic_current_summary_20260912.tex}')
assert current.split(end,1)[1]==suffix,'Unexpected appendix change'
normalized_current=current
for c in appendix_repairs:normalized_current=normalized_current.replace(c['new'],c['old'])
contract=runpy.run_path(str(root/'ops/check_paper_current_contract.py'))
formal=contract['formal_blocks']
assert formal(normalized_current)==formal(before),'Formal statements or proofs changed'
# Ignore commented source commands when checking the main assets.
active=re.sub(r'(?<!\\)%[^\n]*','',actual)
figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{(figures/[^{}]+)\}',active)
assert figures==[f for f in meta['main_figures'] if f!='figures/replay_key_weighting.pdf'],figures
for f,digest in meta['figure_hashes'].items():assert hashlib.sha256((paper/f).read_bytes()).hexdigest()==digest,f
bib=(paper/'example_paper.bib').read_bytes();old_bib=(audit/'paper__example_paper.bib.before').read_bytes();assert bib.startswith(old_bib)
keys=re.findall(rb'@\w+\s*\{\s*([^,\s]+)',bib)
assert len(keys)==len(set(keys)),'Duplicate bibliography keys'
report={'author_text_exact_after_documented_reference_repairs_and_user_requested_edits':True,'user_requested_edits':len(user_edits),'main_link_repairs':len(repairs),'retained_main_figures':len(figures),'main_figure_order_and_bytes_unchanged':True,'preamble_unchanged':True,'appendix_changes_limited_to_recorded_repairs_and_theorem_and_figure_relocation':True,'formal_blocks_preserved':sum(formal(before).values()),'existing_bibliography_prefix_preserved':True,'new_bibliography_entries':len(re.findall(rb'@\w+\s*\{\s*([^,\s]+)',bib[len(old_bib):])),'main_tex_sha256':hashlib.sha256(current.encode()).hexdigest(),'bib_sha256':hashlib.sha256(bib).hexdigest()}
(audit/'resolved_user_main.tex').write_text(actual)
(audit/'fidelity_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
