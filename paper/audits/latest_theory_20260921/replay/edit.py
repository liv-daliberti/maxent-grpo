from pathlib import Path
import difflib, json, hashlib
root=Path('/tmp/paper-latest-theory-20260921')
text=(root/'before.tex').read_text()
start=r'\subsection{A replay signal when fresh groups supply none}'
end=r'\subsection{Reference KL retains the reference, not the discovery}'
a=text.index(start); b=text.index(end,a)
before=text[a:b]
s=before
changes=[]
def change(old,new,why):
    global s
    assert s.count(old)==1, (s.count(old),old)
    s=s.replace(old,new)
    changes.append({'reason':why,'before':old,'after':new})
change('so a nonzero fresh task component requires a mixed group, an event\nwhose probability is at most $h_G(P)$ at every correctness level.',
       'so the probability of a nonzero fresh task component is at most $h_G(P)$.',
       'State the probability bound directly; mixed-group probability itself is exactly h_G.')
change('Together the two lemmas say only that a banked target can supply a verified\nsignal when the fresh task gradient is unavailable. Lemma~\\ref{lem:replay-gradient-availability}\nbounds an opportunity, not a gradient magnitude, and says nothing about other\nloss terms, clipping, or momentum carried across updates.',
       'A fixed verified bank can supply a replay signal when the fresh task\ngradient is unavailable. Lemma~\\ref{lem:replay-gradient-availability}\nbounds the probability of a nonzero fresh task component; it does not bound\nits magnitude or account for other loss terms, clipping, or momentum.',
       'Replace commentary about what the lemmas say with the scientific statement and its scope.')
change('For the uniform bank target $w$, this is the established likelihood-replay\nobjective $R_{\\mathcal B}=H(w)+\\mathrm{KL}(w\\|p)$',
       'For the uniform bank target $w$, the likelihood-replay objective is\n$R_{\\mathcal B}=H(w)+\\mathrm{KL}(w\\|p)$',
       'Remove novelty positioning while retaining prior-work attribution.')
change('finite alternating updates with simultaneous gradient flow. The prompt-isolated categorical mean flow for any of the three fresh\nbinary-reward objectives above plus verified replay is',
       'finite alternating updates with simultaneous gradient flow. For one\nisolated prompt, the categorical mean flow for any of the three fresh\nbinary-reward estimators with verified replay is',
       'Use direct presentation and separate the averaging assumption from the defining equation.')
change('Nothing below uses more than that. By\nTheorem~\\ref{thm:objective-inertness} and\nEquation~\\eqref{eq:inertness-polynomial}, every bounded reward-measurable\nobjective contributes $c_a(P)\\nabla_zP$ with $c_a$ a polynomial, so it admits\nthe same $\\Psi_a$ and the retention argument runs unchanged: what follows is a\nstatement about verified replay added to \\emph{any} outcome-binary fresh\nobjective, of which Re:Dr and Re:Max are the two this paper trains. Define',
       'More generally, Theorem~\\ref{thm:objective-inertness} and\nEquation~\\eqref{eq:inertness-polynomial} give\n$c_a(P)\\nabla_zP$ for every fixed, bounded reward-measurable advantage,\nwith $c_a$ a polynomial. Its integral $\\Psi_a$ is also bounded on $[0,1]$,\nso the retention argument applies even when the fresh coefficient changes\nsign. Define',
       'Remove manuscript narration and state the exact fixed-advantage class, including signed coefficients.')
change('Suppose logits are finite at time $T$ and thereafter the nonempty verified\nbank $\\mathcal B$ is fixed, fits within replay capacity, and receives a',
       'Under Assumption~\\ref{ass:categorical-mean-flow}, suppose logits are finite\nat time $T$ and thereafter the nonempty verified bank $\\mathcal B$ is fixed,\nfits within replay capacity, and receives a',
       'Make the categorical assumptions explicit in the theorem statement; no new assumption is introduced.')
change('a statement about verified replay added to', 'a statement about verified replay added to', 'UNUSED') if False else None
change('The finite-group\n$c_G$ is smooth on $[0,1]$, and softmax derivatives and the cross-entropy',
       'The fixed reward-measurable\nadvantage gives a smooth $c_G$ on $[0,1]$, and softmax derivatives and the cross-entropy',
       'Align the full-coverage proof with its theorem-wide coefficient class, beyond the three named estimators.')
change('Thus uniform conditional allocation maximizes \\texttt{distinct@}$K$ at\nfixed correctness, and more strongly maximizes every coverage tail\nprobability. The conclusion applies\n\\citet[Theorem~3, v1]{anceaume2015coupon} to a frozen policy with a fixed\ncorrectness level and a finite catalogue of verified execution keys under\nindependent sampling.',
       'Thus uniform conditional allocation maximizes expected \\texttt{distinct@}$K$\nat fixed correctness, as well as every coverage tail probability. This is\nthe coupon-collection ordering of\n\\citet[Theorem~3, v1]{anceaume2015coupon}, applied to independent draws\nfrom a fixed policy over a finite catalogue of verified execution keys.',
       'Make expected-count meaning explicit and remove frozen-policy/source-construction phrasing.')
change('so even correct modes outside the frozen bank vanish',
       'so even correct modes outside the fixed bank vanish',
       'Use the mathematical fixed-bank assumption without process terminology.')
change('The retention theorem protects only\nadmitted banked modes; it supplies no probability floor for an unbanked key.',
       'The retention theorem protects only\nbanked modes; it supplies no probability floor for an unbanked key.',
       'Remove redundant admitted-evidence language; fixed-bank coverage is the relevant condition.')
change('% Insert after the fixed-bank replay subsection and before the exemplar bridge.\n','',
       'Remove source-level insertion instruction.')
change('The restoring logit field $\\rho(\\bar w-p)$ has a shared foundation with\ninverse probability scaling: the ideal IPS field of\n\\citet[Theorem~4.1]{sinha2026expected}, normalized to a fixed target $w$, is\n$\\bar w-p$. The uniform-correct target and its forward conditional-KL\nformulation are also established by \\citet{lochab2026ucpo}. Here the target is\nrestricted to a discovered bank and evaluated from stored exemplars. The\nquestion below is how its restoring field competes with the fresh binary\nobjective over a finite interval. Neither a uniform target nor its\ncross-entropy gradient is a new contribution of this analysis.',
       'The restoring logit field $\\rho(\\bar w-p)$ is proportional to the ideal\ninverse probability scaling field of\n\\citet[Theorem~4.1, v1]{sinha2026expected}, normalized to a fixed target $w$.\nThe uniform-correct target and its forward conditional-KL formulation also\nappear in \\citet[Sections~5.2 and~6.2, v1]{lochab2026ucpo}. For replay, the target is\nsupported on the discovered bank and the loss is evaluated from stored\nexemplars. The relative strengths of this restoring field and the fresh\nbinary objective determine whether bank diversity increases over a finite\ninterval.',
       'Replace novelty and section-construction narration with attribution and the substantive finite-time question.')
change("paper's success-conditional \\pmd{} when $\\mathcal B=\\mathcal C$; otherwise",
       'success-conditional \\pmd{} when $\\mathcal B=\\mathcal C$; otherwise',
       'Remove manuscript self-reference.')
change('The bound specifies the interval on which restoration occurs. It does not\nassume that bank mass is monotone or that correctness stays above its value\nat $T$. Both enter through the displayed exposure; a bound on unconditional\nvisibility additionally needs the stated bank-mass lower bound.',
       'The recovery bound assumes a nonnegative margin $a_\\rho$ throughout\nthe interval. Bank mass need not be monotone, and correctness need not stay\nabove its value at $T$; both contribute to $S(T,t)$. The stated unconditional\nvisibility guarantee additionally uses the bank-mass lower bound $M_0$.',
       'Make the interval condition and quantities explicit without changing the recovery theorem.')
change('These discrete statements concern exact mean gradients, without clipping,\nsampling noise, optimizer state or alternating replay visits. They establish\na finite-step result for the specified update, rather than a guarantee for\nAdamW.',
       'The discrete contraction holds for simultaneous exact mean-gradient\nsteps. Clipping, sampling noise, optimizer state, and alternating replay\nvisits are outside its assumptions, so it does not establish contraction\nfor AdamW.',
       'State the update and its scientific limitation directly.')
change('Event containment provides a conditional bridge:\none complete verified response is one way of producing its key. This bridge\nalso makes the loss\'s length factor explicit. Here',
       'A complete verified response is an event contained in the event of\nproducing its execution key. This containment converts a complete-response\nprobability floor into a key-probability floor while accounting for the\nloss\'s length factor. Here',
       'Replace metaphor and section narration with the event-containment argument.')
change('weights and $J_{\\max}=\\sum_x\\nu_x\\max_{s\\in[0,1]}\\Psi_x(s)$. The lemma does not establish\nthis descent condition for stochastic, clipped, alternating AdamW updates,',
       'weights and $J_{\\max}=\\sum_x\\nu_x\\max_{s\\in[0,1]}\\Psi_x(s)$.\nThis descent condition is not established for stochastic, clipped,\nalternating AdamW updates,',
       'Improve paragraph flow while preserving the unproved optimizer premise.')
change('For a selected bank of $k$ complete exemplars, define',
       'For a fixed bank of $k$ complete exemplars, define',
       'Use the fixed target during a realized step; selection history is irrelevant.')
change('% Insert after the existing complete-exemplar/finite-visibility subsection.\n','',
       'Remove source-level insertion instruction.')
change('Lemma~\\ref{lem:shared-exemplar-retention} permits shared parameters by\nassuming descent of a joint potential. A local statement can instead expose\nthe condition under which a realized replay update helps a particular\nexemplar. This condition involves interference among scores and the optimizer\'s\nresponse, quantities that the categorical abstraction suppresses.',
       'With shared parameters, a replay update\'s effect on a particular exemplar\ndepends on interference among score gradients and on the optimizer\'s\nresponse. A lower bound on that exemplar\'s change in log probability can be\nobtained from the realized parameter step, without assuming the global\npotential descent required by Lemma~\\ref{lem:shared-exemplar-retention}.',
       'Present the local result directly and explain its distinction from the global potential condition.')
change('nor a curvature bound is established for the experimental runs here.',
       'nor a curvature bound is established for the experimental runs.',
       'Remove deictic manuscript phrasing while preserving the empirical limitation.')
out=root/'replay'
(out/'before-block.tex').write_text(before)
(out/'replacement.tex').write_text(s)
(out/'changes.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='before-block.tex',tofile='replacement.tex')))
(out/'edits.json').write_text(json.dumps(changes,indent=2)+'\n')
print(json.dumps({'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(s.encode()).hexdigest(),'edit_count':len(changes),'before_lines':len(before.splitlines()),'after_lines':len(s.splitlines())},indent=2))
