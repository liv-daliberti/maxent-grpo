from pathlib import Path
import difflib, json, hashlib
root = Path('/tmp/paper-appendix-eighth10-20260921')
before = (root/'before.tex').read_text()
start = before.index(r'\section{Mode Collapse and Verified Replay}')
end = before.index(r'\subsection{Conditional trajectories of outcome-binary mean flows}',start)
old = before[start:end]
new = old
edits = [
(r'''We separate the per-prompt mean gradient from the geometry and noise of its
updates. Outcome-binary score estimators share a correctness-gradient
direction for differentiable neural policies, while their covariance can
differ. The categorical specialization yields finite-step concentration and
sampled symmetry breaking; fixed-bank replay has an explicit restoration
threshold and recovery bound. Complete-response and local-update arguments
state conditions for neural retention. We apply established replicator,
policy-gradient, information-theoretic, and coverage results where their
assumptions match: the sharpened replay certificate follows by KL data
processing, and exact Dr.GRPO with reference KL inherits a published
convergence guarantee. Our derivations address the group estimators, their
sampling noise, and the competition with the verified bank. These results
do not certify the implemented AdamW/PPO trajectories.
Table~\ref{tab:notation} lists the symbols used here.''',r'''For a fixed prompt, the binary-reward score estimators defined below have
mean updates along the correctness gradient, while their sampling
covariances can differ. With independently optimized categorical logits,
these updates amplify differences among equally rewarded modes, and sampling
can break an initially exact symmetry. Fixed-bank replay supplies
probability floors, restoration thresholds, and recovery bounds under the
stated update assumptions. Neural retention requires additional conditions
on complete-response likelihoods and local updates. These results do not
certify the implemented AdamW/PPO trajectories.
Table~\ref{tab:notation} defines the notation.'''),
(r'''The categorical results use the following model. The neural score identities,
finite-step updates, and sampled-update results below state separately which
assumptions they replace; none identifies a neural mode with an independently
optimized logit.''',r'''The categorical model assigns one independent logit to each outcome
category. Its Euclidean mean flow and the subsequent finite-step, sampled,
and neural updates have different assumptions; a neural solution mode need
not have an independently optimized logit.'''),
(r'''with $m\ge2$ correct
execution modes''',r'''with $m\ge2$ correct
solution modes'''),
(r'''\item[(A2)] \emph{Common length normalization.} Every response shares a single
  length scale, which is absorbed into the positive coefficient of the group
  estimator, rather than a scale that varies from one response to the next
  within the same rollout group.''',r'''\item[(A2)] \emph{Common length normalization.} Every response uses the same
  fixed length-normalization factor, absorbed into the positive coefficient
  of the group estimator.'''),
(r'''We write ``under Assumption~\ref{ass:categorical-mean-flow}'' for (A1)--(A5)
together. Each limitation of the analysis can be traced to a specific item:
AdamW and PPO violate (A3), variable-length GRPO violates (A2), and entropy
regularization violates (A4).''',r'''Assumption~\ref{ass:categorical-mean-flow} includes all five conditions.
AdamW's preconditioning and momentum violate (A3); finite, repeated PPO
updates are outside its infinitesimal on-policy limit. Response-specific
length normalization violates (A2), while reference KL and entropy
regularization violate (A4).'''),
(r'''A language model induces probabilities over canonical outcomes, but this
identification of probabilities does not identify its update geometry.
Aggregating several response strings into a mode does not generally commute
with gradient flow in response logits. One independent logit per canonical
outcome is therefore an explicit abstraction, not a reduction proved for the
neural policy or its validator equivalence classes.''',r'''A language model induces probabilities over canonical outcomes, but those
probabilities alone do not determine its update geometry. Aggregating several
response strings into a mode does not generally commute with gradient flow
in response logits. The independent-category model therefore requires an
additional assumption beyond a neural policy and its validator equivalence
classes.'''),
(r'''For the Dr.GRPO abstraction, $w(R)=1$ after absorbing the shared positive
length scale. Reward-standardized GRPO with equal or common response-length
normalization takes $w(R)$ to be the reciprocal group reward standard
deviation on mixed groups (with a fixed numerical stabilizer if used).
Standard variable-length GRPO also weights each response by its own inverse
length \citep{shao2024deepseekmath,liu2025understanding}; that factor cannot
generally be absorbed into $w(R)$ and is outside the equivalence proved here.''',r'''For the Dr.GRPO abstraction \citep{liu2025understanding}, $w(R)=1$ after
absorbing the fixed positive length-normalization factor. Reward-standardized
GRPO \citep{shao2024deepseekmath} with fixed common response-length
normalization takes $w(R)$ to be the reciprocal group reward standard
deviation on mixed groups (with a fixed numerical stabilizer if used).
Standard variable-length GRPO also weights each response by its own inverse
length; that response-specific factor cannot generally be absorbed into
$w(R)$ and is outside this equivalence.'''),
(r'''This establishes the smooth, positive extension to $P=0,1$ used in the
subsequent flow and convergence arguments.''',r'''Thus $c_G$ extends smoothly and positively to $P=0,1$.'''),
(r'''The next lemma specializes the practical-estimator analysis of
\citet[Section~4.3 and App.~D, Theorem~5]{tajwar2026maxrl}
to our categorical notation. We invoke that result directly and retain the
normalization that identifies the implemented estimator.''',r'''The practical MaxRL estimator has the same mean-gradient direction
\citep[Section~4.3 and App.~D, Theorem~5]{tajwar2026maxrl}. Its all-failure
convention determines the finite-group coefficient.'''),
(r'''Apply \citet[Theorem~5]{tajwar2026maxrl} with $N=G$ and pass rate $P$.
Its dropped-baseline estimator is exactly $\widehat g_G^{\mathrm{MaxRL}}$,
so its expected coefficient is $\sum_{j=0}^{G-2}(1-P)^j$.
The finite geometric sum gives Equation~\eqref{eq:maxrl-expected-gradient}
and its endpoint values. This application uses on-policy independent draws
and the score identity, without making an optimizer convergence claim.''',r'''With $N=G$ and pass rate $P$, the estimator in
\citet[Theorem~5]{tajwar2026maxrl} equals
$\widehat g_G^{\mathrm{MaxRL}}$. Its expected coefficient is
$\sum_{j=0}^{G-2}(1-P)^j$, whose geometric sum gives
Equation~\eqref{eq:maxrl-expected-gradient} and its endpoint values.
The identity requires on-policy independent draws and the score identity.'''),
(r'''The same finite-series identity gives the potential used below:''',r'''Integrating this coefficient gives the potential'''),
(r'''In particular, $\Psi_{\mathrm{MaxRL}}(1)=\sum_{k=1}^{G-1}1/k$.
For binary rewards, best@\(k\) equals pass@\(k\), so this is a harmonic
mixture of sampling objectives for $k=1,\ldots,G-1$. The upper limit is
$G-1$, not $G$: the implemented centered estimator drops the entire update
on all-failure groups. Subtracting an unconditional score baseline even on
those groups instead gives the order-$G$ estimator. This distinction concerns
the exact finite-group expectation at the on-policy point; it does not assert
unbiasedness of subsequent clipped or adaptive-optimizer updates.''',r'''In particular, $\Psi_{\mathrm{MaxRL}}(1)=\sum_{k=1}^{G-1}1/k$.
For binary rewards, best@\(k\) equals pass@\(k\), so this is a harmonic
mixture of sampling objectives for $k=1,\ldots,G-1$. The upper limit is
$G-1$ because the implemented centered estimator sets the entire update
to zero on all-failure groups. Subtracting an unconditional score baseline
even on those groups instead gives the order-$G$ estimator. These are exact
finite-group expectations at the on-policy point; subsequent clipping and
adaptive optimization can alter the expected update.'''),
(r'''expected-update model removes
sampling noise and retains independently trained Euclidean logits. The
conditional dynamics reduce to the standard frequency-dependent replicator
system with fitness $f_c(q)=q_c$
\citep{hofbauer1998evolutionary,harper2009informationgeometry}. The result is
a specialization to the group estimators, not a new general theorem about
replicator dynamics. Related correctness-indifferent collapse analyses include
\citet{sinha2026expected,lochab2026ucpo}.''',r'''expected-update model has
no sampling noise and uses independently optimized Euclidean logits. The
conditional dynamics follow the frequency-dependent replicator system
with fitness $f_c(q)=q_c$
\citep{hofbauer1998evolutionary,harper2009informationgeometry}.
Related analyses of collapse among equally rewarded outcomes include
\citet{sinha2026expected,lochab2026ucpo}.'''),
(r'''\begin{theorem}[Standard replicator specialization for binary-reward RL]''',r'''\begin{theorem}[Winner-take-all collapse in the categorical mean flow]'''),
(r'''Thus these categorical mean flows converge to a single correct execution
mode''',r'''Thus these categorical mean flows converge to a single correct solution
mode'''),
(r'''Theorem~\ref{thm:grpo-collapse} is asymptotic and internal to (A1)--(A3): it
claims neither finite-time support loss nor convergence of the stochastic
neural optimizers used in our experiments.''',r'''Theorem~\ref{thm:grpo-collapse} is asymptotic and requires (A1)--(A5).
Finite-time support loss and convergence of the stochastic neural optimizers
used in the experiments are outside its scope.'''),
]
edits.append((r'''We define the entire update as zero on all-equal
groups without evaluating an undefined reciprocal standard deviation;''',r'''The entire update is zero on all-equal
groups, without evaluating an undefined reciprocal standard deviation;'''))
for a,b in edits:
    assert new.count(a) == 1, repr(a[:120])
    new = new.replace(a,b)
out = root/'categorical'
(out/'original.tex').write_text(old)
(out/'replacement.tex').write_text(new)
(out/'changes.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='categorical/original.tex',tofile='categorical/replacement.tex')))
(out/'manifest.json').write_text(json.dumps({'start_marker':r'\section{Mode Collapse and Verified Replay}','stop_before':r'\subsection{Conditional trajectories of outcome-binary mean flows}','old_sha256':hashlib.sha256(old.encode()).hexdigest(),'new_sha256':hashlib.sha256(new.encode()).hexdigest(),'replacement_count':len(edits),'source':str(root/'before.tex')},indent=2)+'\n')
print(f'{len(edits)} replacements; {len(old.splitlines())} -> {len(new.splitlines())} lines')
