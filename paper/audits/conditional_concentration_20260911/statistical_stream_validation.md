# Independent validation of the amended collision analysis

This review checks mathematical estimands and the new nominal-stream APIs using synthetic data only. The reviewer did not read real concentration estimates. The original protocol and stream amendment were frozen before the effects; these implementation checks were completed while the parent was resolving metadata admission and running the amended analysis. This receipt must not be represented as a new preregistration.

Reproduce from the repository root:

```bash
python paper/audits/conditional_concentration_20260911/statistical_stream_validation.py
```

The adjacent JSON records the exact source/test hashes, test count, and output. The script reads the analysis source and synthetic tests, and never opens the sample cache or empirical result file. The prior pre-analysis receipt remains unchanged.

## What was verified

The amended APIs select the earliest recorded `(draw_index, option_index)` representative independently within each prompt and endpoint for every nominal `parent_seed + option_index` stream. For the ordinary four parent seeds `base + d` and `n=8`, 32 saved positions yield 11 representatives and 21 repeated occurrences. Later success or key agreement cannot replace an earlier representative. Duplicate disagreement is retained diagnostically. Different prompts are never merged because they share nominal seed IDs.

The source adapter must supply draws in recorded draw-index order. The structural selector uses that established order; it does not infer an ordering from the outcomes or sort draws by numerical parent seed.

The common-ID split uses lower `floor(n/2)` IDs and the remaining upper IDs, producing 5/6 and 6/5 comparisons when there are eleven common IDs. Each orientation retains its own eligibility. Synthetic tests verify separate across-seed fixed-population intersections, unavailable endpoints, empty intersections, and zero shared-ID cases. Old first-two-versus-last-two draw groups share seven nominal stream IDs and are therefore unsuitable as disjoint 16/16 comparisons.

Collision uses the selected streams, while contextual mean@8, pass@8, distinct@8, and extra-mode metrics retain their original intact K=8 definition. The K=8 sensitivity averages eligible draws within each prompt, then weights prompts equally. The four-condition change contrast uses one population eligible in all four conditions and the sign `(replay_end-replay_start)-(control_end-control_start)`.

Five defined measured seed estimates receive the frozen df=4 Student-t interval. Partial estimates remain descriptive without that interval. The across-seed fixed-population sensitivity is unavailable when an originally admitted paired endpoint is missing. For its coverage, divide its intersection size by the original prompt population: its recomputed records use the intersection as their analysis population and therefore report internal coverage one.

Neutral request metadata is admitted both in the legacy empty form and as the exact singleton `[parent_seed]` for each corresponding draw. Mismatching singleton values, wrong draw counts, and per-option seed lists are rejected. Repeated IDs passed to `selected_metrics` are rejected so callers cannot reintroduce duplicate streams.

## Mathematical guarantees and limits

For a fixed prompt and policy, iid observations from the fixed marginal law, and `R=r>=2` correct observations,

`E[sum_c n_c(n_c-1)/(r(r-1)) | R=r] = sum_c q_c^2`.

Conditional on `R=r`, the correct labels have law `q^r`; averaging equal-label indicators over all unordered pairs proves the identity. The test enumerates all categorical observations at budgets two through six for a nonuniform `q`, so the unequal 5/6 budgets preserve this identity. They still change eligibility and variance.

That single-prompt identity does not prove unbiasedness after arbitrary joint eligibility selection. An exact shared-RNG example has `q_A=q_B=(1/2,1/2)` but a selected contrast of `1/8`. Independently sampled endpoints eliminate that example. A second exact example reuses streams across prompts and yields a random-eligible-population mean of `1/16` despite equal promptwise concentration between endpoints. A fixed-total-prompt-denominator score equals zero in that example. Thus even disjoint nominal endpoint streams do not justify claiming an unbiased random-eligible-population mean without additional joint count/label conditions. Fixed populations chosen from observed eligibility also remain selected populations.

For exact reuse of independent all-correct stream outcomes, 38 of 496 pairs in the naive 32 positions reuse one stream. Its expectation is `C+(1-C)*19/248`, giving `267/496` for fair binary correct modes instead of `1/2`. Exact enumeration verifies this and the unbiased distinct-stream result. This is an explanatory counterexample, not a correction formula for runtime-disagreeing or success-conditioned real samples.

Under the paper's conditional categorical mean flow `dq_c/dtau=q_c(q_c-C)` with `C=sum_c q_c^2`,

`dC/dtau=2[sum_c q_c^3-C^2]=2 Var_{c~q}(q_c)>=0`.

Equality holds exactly when `q` is uniform on its positive support. This includes the point mass. It is a conditional theorem in the stated geometry; the statistic can test compatible empirical concentration patterns without asserting that neural training follows the flow or that a mode's true support vanishes.

Distinct nominal seeds are an audited implementation property under the documented V0 mapping, not proof of independence under arbitrary runtime/batch effects. The same nominal IDs also recur across prompts. The primary common-eligible comparison, disjoint orientation means, and their complete-case average should remain descriptive. The nominal five-seed intervals do not include all generation, prompt, reused-initialization, or source-selection uncertainty. Agreement across the unique-stream, disjoint, fixed-population, and intact-K8 views supports a narrower empirical statement; disagreement or small eligible populations must constrain it.


## Additional appendix identities reviewed

Define `q` and `C` when `P>0`. At `P=0`, expected successful breadth is zero and a conditional success distribution is unidentified. For two independent attempts at a fixed prompt,

`B_2 = sum_c [1-(1-P*q_c)^2] = 2P-P^2*C`.

This holds for finite or countable possible modes: the nonnegative summands are bounded by `2P*q_c`, so the series converges and the termwise identity sums legitimately. At fixed positive `P`, lower `C` is exactly higher expected breadth for two attempts. At larger budgets, a collision ordering alone does not order every occupancy functional.

For a finite or countable probability vector, `C=1` holds exactly when its law is a point mass. In contrast, observed `C_hat=1` with at least two correct draws only says those observed correct draws share one key. It does not establish the underlying point-mass property.

Lower true collision alone implies neither majorization nor larger support. The vectors `(0.6,0.2,0.2)` and `(0.5,0.4,0.1)` have concentrations `.44` and `.42`, but their largest-one and largest-two partial sums cross. Changing `(0.7,0.15,0.15)` to `(0.5,0.5,0)` lowers concentration from `.535` to `.5` while reducing positive support from three modes to two.

For a fixed equally weighted prompt population with the same budget `K` and iid samples within each prompt, the expected collision-pair count at prompt `x` is `choose(K,2)*P_x^2*C_x`, and the expected correct-pair count is `choose(K,2)*P_x^2`. The ratio of these expected counts is therefore

`sum_x P_x^2*C_x / sum_x P_x^2`.

It is also the collision probability after selecting a prompt uniformly, drawing two iid attempts, and conditioning on both attempts succeeding. This statement requires a nonzero denominator. It does not make the observed ratio of random pooled counts exactly unbiased; with different budgets, the population weights also include `choose(K_x,2)`. Dependence across prompts does not alter the expected-count identity, but it can obstruct customary iid-based concentration or interval claims.

The categorical derivative proof above applies directly in the paper's finite categorical geometry. A countable extension should explicitly assume sufficient path regularity to differentiate the series, such as an `ell_1` differentiable probability path obeying the stated flow. Equality then still means equal mass on every positive-support element, which necessarily gives a finite uniform positive support for a countable probability distribution.
