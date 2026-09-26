# Discovery, admission, and changing-bank retention

Independent theory/literature note, 2026-09-05. No manuscript, runtime, scheduler, or experiment changes. A self-contained mathematical fragment is in `discovery.tex`; it introduces no dependency on main-paper theorem labels. This is a conditional extension of the existing analysis, not an empirical guarantee for the released runs.

## Main conclusion

A useful guarantee has three separate premises: **a key continues to have a sufficient chance of actual admission; admission preserves a positively weighted verified representative; and optimization plus bank switches keeps a common energy budget finite.** A large bank, repeated sampling, a higher temperature, or a miss-triggered replay controller does not by itself establish any of these premises.

Two rigorous additions are ready:

1. A conditional admission-hazard bound allowing history-dependent policies, proposal scheduling, validation, and capacity selection. Deterministic hazard floors give exponential finite-budget discovery bounds. A stopped exponential supermartingale handles random predictable hazards without a false independence argument.
2. A switching-energy retention bound. It charges changes in the bank objective explicitly and protects a stored complete exemplar when its coefficient remains bounded below. Preserving old coefficients makes admission costs especially transparent; fixed total replay dose with capacity K also works through a coefficient floor and explicit renormalization jumps.

Both use standard probability/energy machinery. Their application to execution-key admission and verified exemplar losses is a corollary, not a claim to originate conditional Borel–Cantelli, coupon collection, archive exploration, or switched Lyapunov analysis.

## Verified primary literature and exact applicability

- **Rick Durrett. _Probability: Theory and Examples_, fifth edition, Cambridge University Press, 2019.** The author's [book page](https://sites.math.duke.edu/~rtd/) verifies the edition/publisher/year. The [author-hosted January 11, 2019 manuscript](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf), Theorem 4.3.4, pp.225–226, states the adapted-event conditional Borel–Cantelli equivalence: events occur infinitely often exactly on the event where the sum of their past-conditional probabilities diverges, up to null sets. This is the appropriate standard result for adaptive sampling. The direct first-admission supermartingale proof below is included so no theorem transfer is hidden.
- **Emmanuelle Anceaume, Yann Busnel, Ernst Schulte-Geers, Bruno Sericola. _Optimization results for a generalized coupon collector problem_, 2015, arXiv:1504.03878v1.** [Primary metadata](https://arxiv.org/abs/1504.03878); [full text](https://arxiv.org/html/1504.03878v1). Theorem 2 shows uniform iid sampling minimizes the collection time in stochastic order; Theorem 3 handles a null coupon and fixed total non-null mass. Incorrect responses correspond to the null coupon in a fixed-policy analogy. These theorems do not directly apply to history-dependent training, rejection by a full bank, or proposal selection. Use the adaptive hazard proof for those cases. The paper's prose calls a collection CDF Schur-convex, but under the usual majorization convention its stated averaging direction is Schur-concave; use its explicit Theorems 2–3, not that terminology aside.
- **Adrien Ecoffet, Joost Huizinga, Joel Lehman, Kenneth O. Stanley, Jeff Clune. _First return, then explore_, Nature 590, 580–586, 2021. DOI:10.1038/s41586-020-03157-9.** [Publisher metadata](https://www.nature.com/articles/s41586-020-03157-9); [author-hosted published full text](https://adrien.ecoffet.com/files/go-explore-nature.pdf); [arXiv record](https://arxiv.org/abs/2004.12919). Go-Explore explicitly stores visited state cells, selects archive states, returns to them, and explores further; its robustification phase learns a policy from discovered trajectories. This is relevant precedent for separating discovery from retaining access to discoveries. The present bank stores verified response exemplars for teacher-forced replay, and does not restore simulator states or establish Go-Explore's empirical claims in a language model.
- **Michael S. Branicky. _Multiple Lyapunov Functions and Other Analysis Tools for Switched and Hybrid Systems_, IEEE Transactions on Automatic Control 43(4), 475–482, April 1998.** [Published article PDF](https://people.ece.ubc.ca/moishi/eece571m/articles/Branicky98.pdf). The paper, including Theorem 2.3, treats the need to control energy behavior across switching systems; it explicitly assumes finitely many switches on bounded time intervals. It is methodological precedent. The simple cumulative positive-jump inequality below is derived directly and does not assert that all hypotheses of Branicky's equilibrium-stability theorem hold for neural training or divergent categorical logits.
- **David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, Greg Wayne. _Experience Replay for Continual Learning_, NeurIPS 2019, arXiv:1811.11682v2.** [Primary metadata/full-text links](https://arxiv.org/abs/1811.11682). CLEAR combines replay with on/off-policy learning and behavioral cloning to reduce forgetting; its finite-memory results are empirical. It supports the broad replay/retention motivation, not a universal probability floor for every retained execution key.

Downloaded primary PDFs and extracted text are under `discovery_sources/`. The bibliography fields and source hashes are recorded in `discovery_sources.json`. Do not add all these citations mechanically: Durrett is the main probability attribution, Go-Explore the direct archive relationship; the remaining references explain adjacent applicability.

## 1. Define admission rather than only a sampled hit

Fix one prompt and a target execution key b. Let F_n contain the full history after opportunity n: policy parameters, fresh responses, proposal requests, accepted/rejected candidates, memory state, optimizer state, and controller state. An opportunity can be a global update or a prompt-local query, but the indexing convention must be fixed. Let tau_b be the first update at which a verified exemplar for b is actually inserted into active replay memory; tau_b=infinity if this never happens.

The predictable stopped hazard is

\[
 h_{b,n}=\Pr(\tau_b=n\mid\mathcal F_{n-1}).
\]

It is zero after a previous admission. On surviving histories, it is the conditional chance of insertion now. It includes all gates; it is not simply the policy's marginal key probability.

For a frozen query law that draws G independent responses with complete accepted-key probability q_{b,n}, the chance of at least one b response is `1-(1-q_{b,n})^G`. This equals admission hazard only if the key remains admissible and selection guarantees insertion whenever it appears. More generally, conditional on eligibility being decided from the past,

\[
 h_{b,n}=\mathbf1_{\rm eligible}\,
 a_{b,n}\,[1-(1-q_{b,n})^G],
\]

where a_{b,n} is the **conditional** chance of actual insertion given that b appears in the group and the past. This factorization uses the conditional-probability chain rule, not independence of selection and generation. Multiple temperatures, early termination after another novel key, transformations, and fresh/proposal interleaving are more naturally represented directly by h. If eligibility itself depends on a fresh group, either refine the filtration to put the eligibility decision before the query or integrate over that decision in h.

## 2. Finite-budget and almost-sure admission bounds

Set H_{b,N}=sum_{n<=N}h_{b,n}. Define

\[
 Z_N=\mathbf1\{\tau_b>N\}\exp(H_{b,N}).
\]

On survival through n-1, the conditional expectation is

\[
 E[Z_n\mid\mathcal F_{n-1}]
 =Z_{n-1}(1-h_{b,n})e^{h_{b,n}}\le Z_{n-1};
\]

on already admitted histories both sides vanish. Thus Z is a nonnegative supermartingale starting at 1, without assuming independent groups or a fixed policy. Markov's inequality gives the rigorous adaptive statement

\[
 \Pr(\tau_b>N,\ H_{b,N}\ge A)\le e^{-A}.
\]

If H diverges on the event of never admitting b, Fatou's lemma excludes that event: there Z_N would diverge while E[Z_N]<=1. Equivalently,

\[
 \Pr(\tau_b=\infty,\ H_{b,\infty}=\infty)=0.
\]

This is also a first-admission application of Durrett's conditional Borel–Cantelli result. The events `{tau_b=n}` can occur at most once, so their predictable conditional probabilities cannot sum to infinity on a positive-probability path.

If deterministic numbers h_lower_{b,n} lower-bound the hazard on every not-yet-admitted history, ordinary conditioning gives

\[
 \Pr(\tau_b>N)\le\prod_{n=1}^N(1-\underline h_{b,n})
 \le e^{-\sum_n\underline h_{b,n}}.
\]

The first inequality is proved recursively; no independence assumption is needed. For a finite target set S, union bound gives

\[
 \Pr(\text{some }b\in S\text{ unadmitted by }N)
 \le\sum_{b\in S}e^{-\sum_{n\le N}\underline h_{b,n}}.
\]

When every target has a constant admission floor h_*>0, `N>=log(|S|/epsilon)/h_*` suffices for all-target admission with probability at least `1-epsilon`. This statement already assumes capacity, gates and competition cannot remove that floor. In particular it cannot hold indefinitely for all correct keys when their number exceeds a non-evicting capacity.

**Common invalid shortcut:** for history-dependent hazards, do not write `Pr(tau>N)=E[product_n(1-h_n)]`. As a two-step counterexample, take h_1=1/2, then h_2=1/2 if the first admission failed and h_2=0 if it succeeded. Actual survival probability is 1/4, while the expectation of the product is 3/8. Conditioning on a complete future hazard sequence can itself reveal whether earlier admission occurred. The stopped-supermartingale formulation avoids this problem.

## 3. What is sufficient for discovery, and counterexamples

### Summable positive hazard does not suffice

Take independent opportunities with `h_n=1/(n+1)^2`, n>=1. Every opportunity has positive admission probability, but

\[
 \prod_{n=1}^N\left(1-\frac1{(n+1)^2}\right)
 =\frac{N+2}{2(N+1)}\longrightarrow\frac12.
\]

Thus the mode is never admitted with probability 1/2. Positivity of finite softmax probabilities and infinitely many opportunities are insufficient. A bounded number of additional samples merely multiplies a small per-draw hazard by a bounded factor and can leave the series summable. A bounded temperature increase is also insufficient when the relevant logit gap diverges quickly.

### A genuine exploration condition

As a conditional design example, suppose the proposal law mixes an adaptive policy with a fixed verified-key-covering reference law: `q_n=(1-epsilon_n)pi_n+epsilon_n nu`, with `nu(b)>0` for each target. If all other gates admit b whenever it appears, a single draw gives `h_{b,n}>=epsilon_n nu(b)`. A divergent sum of epsilon_n then implies almost-sure eventual admission for every member of a finite target set. The fixed proposal law need not be uniform, but every target needs positive mass and continued eligibility. This is an assumption/design example; the actual code uses policy sampling at registered temperatures, not such a proved fixed-law mixture.

### Capacity and selection are separate obstructions

A capacity-K bank without eviction cannot simultaneously store more than K keys. Even for a selected target set of size <=K, unrelated keys may consume the slots first unless the full admissible universe fits or admission reserves the required capacity. Sorted-key collision resolution can make a key's conditional insertion chance zero on some histories despite its appearance in the batch. A singleton-only proposal gate can terminate further exploration as soon as a second key is known. A finite fresh-pass schedule can stop revisiting the prompt. None of these is repaired by invoking Borel–Cantelli.

## 4. Switching-energy retention with persistent coefficients

Let switch times t_r have no finite accumulation. On interval r, define

\[
 F_r(\theta)=\sum_jd_{rj}\ell_j(\theta)-J(\theta),
 \qquad \ell_j=-\log\pi_\theta(e_j\mid x_j)\ge0,
 \quad J\le J_{\max}<\infty.
\]

Absent entries have zero coefficient. Length-normalized losses correspond to `d_{rj}=alpha_{rj}/L_j`; use the coefficient on **sequence surprisal**, not just the nominal key weight. Assume finite initial energy, continuity of parameters at switches, and nonincrease of F_r between switches. Assume also a finite positive-jump budget

\[
 A=\sum_{r\ge1}[F_r(\theta(t_r))-F_{r-1}(\theta(t_r))]_+<\infty.
\]

For every exemplar j protected after its admission interval, suppose `d_{rj}>=d_lower_j>0` thereafter. Telescoping interval descents and switch jumps gives

\[
 F_r(\theta(t))\le F_0(\theta(t_0))+A.
\]

Consequently, setting `D=F_0(theta(t_0))+A+J_max`, every nonnegative loss summand is at most D and

\[
 p_\theta(b_j\mid x_j)\ge\pi_\theta(e_j\mid x_j)
 \ge\exp(-D/d_{{\rm lower},j})>0.
\]

This permits shared parameters and prompts. It is an energy accounting corollary, not a convergence theorem. Full-response/stopping-event and matching sampling-law assumptions are essential. Prefix likelihood alone does not bound completed-key probability. Nonnegative summands make D nonnegative on all intervals with active protected exemplars.

**Preserving old coefficients.** If new entries are appended while every old d_j is unchanged, the energy jump equals the sum of new entries' weighted surprisal at admission. This cleanly preserves the old protection coefficients, though total replay budget grows unless budget was reserved. Finite weighted admission cost gives a finite all-time retention budget. Do not claim the implemented normalized mean loss has this append-only coefficient property.

**Actual uniform-bank dilution.** For fixed categorical dose rho and a size-k uniform bank,

\[
 R_{k+1}=\frac{kR_k+\ell_{new}}{k+1},\qquad
 \Delta F=\frac{\rho}{k+1}(\ell_{new}-R_k).
\]

Old coefficients fall from rho/k to rho/(k+1), but stay >=rho/K under capacity K. For a fixed finite collection of possible prompts/exemplars, finitely many admissions with positive admission-time likelihoods give finite A. The same argument therefore protects earlier keys through bank growth in this exact averaged descent model, rather than restarting the proof only after the last admission. Multiple simultaneous admissions are handled by the exact before/after objective difference. A joint multi-prompt model must also bound the effective prompt weights below for every protected prompt.

**Changing representatives.** If a stored surface is replaced while retaining the same key, the old exemplar's coefficient becomes zero and the fixed-exemplar theorem does not protect that surface. A key-level variant is immediate: require each interval to contain at least one complete verified exemplar for the key with coefficient >=d_lower_b and charge all representative changes to A. Then key mass has the same floor even if the representative varies. This does not restore protection to an evicted key with no active representative.

**Optional continuous coefficient variation.** With fixed exemplars and J, continuously changing coefficients obey

\[
 \frac{dF_t(\theta(t))}{dt}
 =-\|\nabla_\theta F_t\|^2+\sum_j\dot d_j(t)\ell_j(\theta(t))
\]

under exact instantaneous gradient flow. If d_j>=d_lower_j>0 and
`a(t)=max_j(dot d_j/d_j)_+` is integrable, then for `E=F_t+J_max>=sum_j d_j ell_j`, `E'<=a(t)E`. Gronwall gives `E(t)<=E(T)exp(int_T^t a)`, hence a positive floor. Repeated bounded changes can have infinite positive variation; bounds on instantaneous weights alone do not prove this sufficient energy condition. No such variation/energy budget has been established for adaptive priority in the repository.

## 5. Composition: a conditional discovery-to-visibility statement

Suppose a finite target set has the deterministic admission bounds above and, on paths where it is admitted by N, retained representatives satisfy deterministic energy floors delta_b from N onward. At any later checkpoint, independently draw K responses from that checkpoint's law. Conditional on training history, the chance that some banked target is not seen is at most `sum_b(1-delta_b)^K`. Therefore

\[
 \Pr(\text{some target not admitted by }N\text{ or not seen in evaluation})
 \le\sum_b e^{-\sum_{n\le N}\underline h_{b,n}}
   +\sum_b(1-\delta_b)^K.
\]

This is a clean end-to-end **conditional** bound. No independence between training and the learned checkpoint is assumed; the evaluation bound is conditional on the resulting policy. It does not supply the missing deterministic hazard or energy constants for neural experiments. Pathwise finite random switching costs imply random positive floors, but deterministic finite-budget confidence statements additionally need deterministic bounds or a separate high-probability event controlling those costs.

## 6. Implementation mapping and limits

### Admission and bank identity

- `online_canonical_bank.py::score_and_update`: only loss-active rows with positive task reward and a nonempty key qualify. Groups use pre-commit history; new keys and candidate token tuples are sorted deterministically. Fresh-only primary banks store at most one exemplar per key, no more than capacity, and do not evict existing keys. Keys seen after capacity fills can remain in historical counts without joining active replay support. The hazard must track insertion, not count-table growth.
- `admit_verified_proposals`: stages a transaction, accepts only keys absent from the relevant support, enforces capacity in the separate-proposal-support path, and admits into replay without adding proposal rows to fresh objective counts. `new_outcomes` and `stored_exemplars` are distinct diagnostics in generic modes; use active support insertion for tau.
- Generic proposal-only conversion can overwrite a proposal surface when that key later appears in fresh data (`score_and_update`, the `new_key in proposal_only` path). The main fresh-only factorial does not use that optional branch. If analyzing it, charge the representative change or use the key-level variant above.

### Proposal generation

- `learner/run.py::_generate_verified_counterfactual_proposals` requires a verified anchor from the current neutral group or an eligible prior bank. Optional singleton-only/verified-route gates can stop searching when known support changes.
- Optional validator-preserving transforms are checked before stochastic proposals. Generation attempts use a bounded budget and registered temperatures; non-route attempt temperatures can increase by 0.2. The search stops after the first attempt yielding any novel candidates, so another key's discovery can truncate opportunities for a particular target.
- Candidates must satisfy validator and positive task-reward gates, and route-specific likelihood checks when active. They are truncated in sorted-key order to remaining slots. Compute-only paths explicitly discard them. These facts invalidate replacing h with the raw policy sampling probability without additional conditions.
- `proposal_starvation.py` increases the bounded attempt budget after a streak without admission and uses finite bursts/cooldowns. It measures lack of insertion and can improve exposure; it provides no mathematical lower bound on target probability or guarantee that cumulative admission hazard diverges.

### Replay scheduling and weights

- `scheduled_global_replay_groups` includes singleton banks when min_modes=1. With a fixed finite eligible set and no priority queue, the cursor provides recurrent access to every bank. The cursor is checkpointed. This supplies scheduling recurrence, not descent of a simultaneous objective after finite alternating fresh/replay steps.
- Optional proposal priority is consumed before ordinary round-robin slots. If priority requests keep replenishing all available slots, the ordinary cursor can fail to advance, so fairness for unprioritized prompts needs a separate reserved-share condition. The primary factorial excludes this controller.
- Priority weights are normalized to preserve the group budget. With a fixed maximum multiplier M and bank capacity K, each present key's within-group normalized weight is at least 1/(KM). This is only a within-group floor: it does not ensure a positive global prompt-update frequency or a finite switching-energy budget.
- Mean-token scores divide by stored response length, so a coefficient floor on sequence surprisal must also account for maximum exemplar length. Different shared-prompt averaging weights and replay frequencies must be included.

### What the admission-retention sensor actually records

`AdmissionRetentionTracker` correctly skips the first neutral group associated with a new proposal admission because that group was sampled before admission. Later prompt-specific opportunities and hits are tracked separately from score visits. However:

- `converted_on_policy` records **ever reappearing once**; it is not a persistent or current lower probability bound.
- The score baseline is the first observed post-admission replay score, not necessarily the score at the instant of insertion. Earlier change can be missed.
- `score_retained` compares the latest mean-token score to that baseline within a fixed drop threshold. For fixed length L, a permitted mean-score drop d permits a sequence probability ratio as small as exp(-Ld). This is a relative diagnostic, not an absolute floor on a key's probability.
- `joint_retained` combines ever-converted status with a latest score criterion. It is not an all-time survival guarantee.
- Miss streaks and score drops issue bounded priority requests when enabled; they do not constrain optimizer updates or enforce a Lyapunov inequality. Finite nonappearance is not evidence of literal extinction, and a single later appearance does not prove asymptotic retention.

### Released experimental scope

The current manuscript's main factorial explicitly has no proposal model, semantic-entropy term, novelty reward, or adaptive replay coefficient. The separately described discovery comparator bundles isolated proposals with semantic regularization; it does not identify their effects separately or track individual-mode survival. Development branches for starvation fallback, adaptive priority and open-bank control are excluded. The theoretical note should not imply these optional controllers are active in every ReplayMaxRL run, or that their empirical behavior proves the hazard/energy premises.

## Recommended use

Keep this as an attributed extension note. The two propositions in `discovery.tex` are the compact core. Use Durrett to credit adaptive probability machinery; cite Go-Explore when explaining the discovery/access distinction. Cite the coupon result only for fixed-policy iid comparisons, and do not transfer its optimality assertion to adaptive training. The practically useful next measurements would be per-key admission opportunities/rejections, cumulative empirical exposure, fixed-exemplar full-response score histories, effective replay coefficients, and exact bank-objective changes. Those are proposed measurements, not newly generated results or a retrospective proof of the current runs.


## 7. Recommended combined corollary: eventual full coverage plus retention

Assume the prompt's entire correct support C is finite, with m<=bank capacity; admissions never evict a key; and at every counted opportunity, every still-missing b has admitted-key hazard at least epsilon*r_b for fixed positive epsilon and r_b. One sufficient construction is a fixed full-support proposal mixture queried at every counted opportunity, with insertion guaranteed whenever the key appears. Capacity must accommodate the whole admissible correct universe, not merely a retrospectively selected subset competing with other keys.

Then

\[
 \Pr(\mathcal B_N=\mathcal C)\ge
 1-\sum_{b\in\mathcal C}e^{-N\epsilon r_b}
 \ge1-m e^{-N\epsilon\gamma}\quad\text{if }r_b\ge\gamma>0.
\]

The failure probability tends to zero, so finite support and non-eviction imply that every correct key is banked by a finite random opportunity index almost surely. If the switching-energy premises also hold almost surely—finite initial energy, finite total positive energy jumps, interval energy nonincrease, and a persistent positive coefficient for every admitted complete exemplar—then after full coverage each correct key stays uniformly bounded away from zero. With only a pathwise finite energy budget, the resulting positive floor can be random; this does not affect the finite-horizon coverage bound, which uses deterministic hazard floors.

This is the clean conditional connection between discovery and retention. It does not prove the required hazard floors or energy descent for the implementation, does not cover a support larger than capacity, and does not imply uniform key probabilities under shared neural parameters. It is included in the optional self-contained `discovery.tex` fragment. No new experiment is proposed or launched by this note.
