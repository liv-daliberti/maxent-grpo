from pathlib import Path
import hashlib,json,re
root=Path('/n/fs/similarity/maxent-grpo')
audit=root/'paper/audits/theory_appendix_integration_20260905'
before=audit/'before'
frags={k:(audit/(k+'_appendix.tex')).read_text().strip() for k in ['optimizer','discovery','entropy','natural_gradient','survival']}
oldroad=r'''identify exactly which stored modes survive. Finally,
Lemma~\ref{lem:shared-exemplar-retention} states the energy condition needed
to extend retention to complete exemplars in a shared model.'''
newroad=r'''identify exactly which stored modes survive. The natural-gradient comparison
in Section~\ref{app:theory-natural-gradient} makes the role of update geometry
explicit, while Section~\ref{app:theory-entropy-comparison} relates exact entropy
to replay through their distinct relative-entropy objectives.
Lemma~\ref{lem:shared-exemplar-retention} then states the energy condition
needed for complete exemplars in a shared model. The remaining sections extend
that condition to controlled stochastic updates and changing banks, and derive
sharp per-mode certificates and valid finite-sample interpretations
(Sections~\ref{app:theory-optimizer}--\ref{app:theory-survival-certificates}).
Each extension states its additional assumptions and the established
literature tools used in its proof.'''
oldtime=r'''Time $t$ is
continuous optimization time; $\tau$ is a reparameterized learning clock.
Neither is a wall-clock or GPU-runtime prediction.'''
newtime=r'''In the mean-flow results, $t$ is
continuous optimization time and $\tau$ is a reparameterized learning clock.
The later stochastic and admission results use explicitly defined update or
opportunity indices. None is a wall-clock or GPU-runtime prediction.'''
newremark=r'''\begin{remark}[Exact scope of the ReplayMaxRL guarantee]
The categorical replay theorem protects represented bank modes after discovery;
it is not an unconditional end-to-end guarantee for a language model.
Without full coverage it protects only $\mathcal B$, and
Lemma~\ref{lem:replay-finite-steps} shows that a frozen incomplete bank can
exclude unbanked correct modes in the categorical limit. Capacity 16 can
therefore truncate protected support. The admission results above identify
sufficient coverage and switching conditions; they do not establish those
conditions for the implemented proposal and insertion machinery.

The categorical proof uses $\log p_b$, whereas the implementation uses
length-normalized teacher-forced exemplar scores. The complete-exemplar bridge
and stochastic extension permit shared parameters under explicit joint-energy
and update-error assumptions. They do not prove uniform neural key
probabilities or verify those assumptions for the current clipped, alternating
AdamW updates. The sharper certificates require correctly normalized complete
likelihoods under the decoding law being bounded. Finally, exact entropy and
exact categorical natural gradient have their own protection guarantees;
collapse is a property of the stated update model, not of every procedure that
optimizes correctness or entropy. These results motivate direct per-mode survival measurements.
\end{remark}'''
for name in ['paper/main.tex','paper/mathai2026/appendix.tex']:
 s=(before/name).read_text()
 assert s.count(oldroad)==1 and s.count(oldtime)==1
 s=s.replace(oldroad,newroad).replace(oldtime,newtime)
 natural_anchor=r'\subsection{Exact entropy and sampled surprisal have different guarantees}'
 entropy_anchor=r'\subsection{When replay supplies a signal that fresh groups lack}'
 remark_anchor=r'\begin{remark}[Exact scope of the ReplayMaxRL guarantee]'
 assert all(s.count(a)==1 for a in [natural_anchor,entropy_anchor,remark_anchor])
 s=s.replace(natural_anchor,frags['natural_gradient']+'\n\n'+natural_anchor)
 s=s.replace(entropy_anchor,frags['entropy']+'\n\n'+entropy_anchor)
 a=s.index(remark_anchor); b=s.index(r'\end{remark}',a)+len(r'\end{remark}')
 s=s[:a]+'\n\n'.join(frags[k] for k in ['optimizer','discovery','survival'])+'\n\n'+newremark+s[b:]
 old=r'''Extending this limit to
changing banks or stochastic neural updates requires separate analysis.'''
 new=r'''The later stochastic and changing-bank results give conditional
retention bounds under additional energy assumptions; they do not extend this
fixed-bank convergence limit to arbitrary neural training.'''
 assert old in s; s=s.replace(old,new)
 if name=='paper/main.tex':
  old=r'''\textbf{Limitations.} Theory studies idealized mean flows. Retention assumes
fixed discovered banks; Python's bank cannot cover all modes.'''
  new=r'''\textbf{Limitations.} Theory studies idealized mean flows. Optimizer and
admission extensions are conditional; Python's bank cannot cover all modes.'''
  assert old in s; s=s.replace(old,new)
 if name=='paper/mathai2026/appendix.tex':
  old=r'''Appendix~\ref{app:theory}'s categorical results are
post-discovery mean-flow guarantees: replay protects banked modes, and
convergence to uniform conditional correct mass requires full verified
coverage, which Python's bank capacity precludes.'''
  new=r'''Appendix~\ref{app:theory} gives categorical collapse and retention
results, plus conditional extensions to stochastic updates and bank admission.
These require the stated geometry, energy, and coverage assumptions; they do
not certify the implemented neural optimizer. Convergence to uniform correct
mass in the categorical uniform-replay model requires full verified coverage,
which Python's bank capacity precludes.'''
  assert old in s; s=s.replace(old,new)
 (root/name).write_text(s)
basebib=(before/'paper/example_paper.bib').read_text()
assert basebib==(before/'paper/mathai2026/example_paper.bib').read_text()
newbib='\n\n% Primary-source-verified theoretical extensions, September 5, 2026.\n'+'\n'.join((audit/f).read_text().strip() for f in ['optimizer_admission_refs.bib','entropy_geometry_refs.bib','survival_refs.bib'])+'\n'
keys=re.findall(r'@\w+\{([^,]+)',basebib+newbib); assert len(keys)==len(set(keys))
for name in ['paper/example_paper.bib','paper/mathai2026/example_paper.bib']: (root/name).write_text(basebib+newbib)
print('Integrated 5 proof fragments and',len(re.findall(r'@\w+\{',newbib)),'new references into both manuscripts.')
