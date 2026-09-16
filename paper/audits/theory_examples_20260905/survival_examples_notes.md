# A8 and A12 worked-example staging

All numbers below are hypothetical mathematical inputs, not measurements from runs. Only the five assigned staging files were edited. The complete current subsections were extracted from `paper/main.tex`; their formal theorem/lemma/corollary/proof/remark blocks and citations are preserved verbatim. The final scope remark remains inside A12. The verifier compares these blocks to the current manuscript and records staging hashes in `verify_survival_examples.json`.

## A8: replay_section.tex

- Uses the shared running distribution `(.30,.15,.05,.50)`, G=4, with the first and third correct modes banked. For illustrative rho=.30, R=2.09985254 and C=3.34985254. A banked probability of 0.0001 would contribute 4.60517019 to the loss by itself, contradicting the bound; the other term cannot offset it. The conservative floor is 0.00123127.
- Gives lengths (10,20,40), induced normalization A=7/120 and target (4/7,2/7,1/7). This explains protection without uniformity and avoids repeating A9's sequence-probability versus mean-token-score example.
- Gives one actual deterministic categorical GD step with rho=.30, G=4, L=.525, eta=1: p becomes approximately (.3861,.1643,.0632,.3864), and F falls .25495576 to .09688800. The unbanked second correct mode initially grows, yet the proved limit is (.50,0,.50,0). No neural learning-rate recommendation is made.

## A12: survival_section.tex

Each of the four subsubsections now has a concrete worked example.

1. The existing 16-mode anchor is expanded: excess cross entropy .01 nat yields sharp coordinate floor .0339712076 versus the termwise 4.61948074e-20, and bank mass at least .990049834. The explanation identifies probability normalization as the missing information in the old bound.
2. One missing mode and uniform mass over the other fifteen has H/log16=.976722649, which rounds to 97.7%; its forward-KL gap is .0645385211 nat. The strict threshold distinction is retained.
3. Distinct complete-response probabilities (.20,.20,.40) and invalid mass .20, with target (.25,.25,.50), give excess cross entropy log1.25. The first two responses share key A: combining target mass gives its certificate .20, compared with .10091909 from adding the two individual response floors; actual constructed key mass is .40. This is a worked illustration, not a proposal to change the implementation's one-exemplar-per-key bank.
4. Binomial examples now explain what each number means: zero of32 gives a one-key one-sided95% upper bound8.94%; protecting16 prespecified keys increases this to16.50%. Two of32 has empirical6.25% but lower confidence bound1.12%. Ten checkpoints times16 keys produce160 statements; equal error allocation gives22.29% for a zero-of32 upper bound. The text distinguishes fixed-policy confidence sequences from pooling changing checkpoints. An added .10 missing-mass example contrasts one unseen mode with100 modes of mass.001 each.

## Validation

Run `python paper/audits/theory_examples_20260905/verify_survival_examples.py`.

The script uses only the standard library. It re-derives binary-KL roots by bisection, inverts the exact binomial tail, calculates the deterministic logit step and potential decrease, and checks the loss contradiction, inverse-length target, aggregation equality, and missing-mass arithmetic. It checks formal blocks and original citations against the source manuscript. No cluster calls, training jobs, or model inference are involved.

Root should integrate each staging file by replacing its complete corresponding subsection. The running-example cross-reference is `ex:theory-running`, as requested; root supplies that label in A1. Main and workshop builds remain root-owned.

Final integration note: root subsequently aligned example links, clarified complete-response wording, and adjusted headings/prose for the manuscript layout. Any handoff hashes above describe that review snapshot; current integrated file hashes are recorded in `validation.json`. All formal statements/proofs and citations remain unchanged.
