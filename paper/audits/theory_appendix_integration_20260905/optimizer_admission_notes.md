# Optimizer and admission appendix integration

Both fragments are ready for inlining after the existing complete-response bridge and finite-visibility argument. They use the existing theorem environments, have no new macros, and use natbib citations. This subtask edited no original supplement or manuscript files.

## Labels and scope

- `app:theory-optimizer`: `ext:thm-stochastic-retention`, with complete drift and Ville proof; `ext:cor-bounded-noise-retention`, the fully proved sharper finite-horizon companion from the vetted supplement's optimizer.md.
- `app:theory-discovery-admission`: `ext:lem-admission-hazards`, `ext:lem-switching-retention`, and `ext:cor-discovery-retention`.
- All new equation/result labels start `ext:`. The optimizer opening references existing `lem:shared-exemplar-retention`.

The stochastic theorem separates finite and infinite horizons, permits singular predictable preconditioners, and does not assert an AdamW/PPO, stationarity, or useful evaluation-floor guarantee. The sharper corollary retains the full theorem assumptions, including conditional centering and deterministic initialization, and adds deterministic steps plus bounded gradients/noise.

Admission includes every insertion gate rather than equating a sampled hit with storage. The group-hit expression uses unconditional `p_b`; capacity is `K_{\rm cap}` to distinguish evaluation draws. The changing-bank lemma is a separate pathwise energy result. Its composition's finite-horizon probability bound controls admission; subsequent all-time floors may depend on the realized trajectory.

## Verified references

`optimizer_admission_refs.bib` contains eight new entries, all with primary-source URLs. Verified published author lists and publication years are retained. Source metadata and theorem locations were checked in the prior literature supplement.

| Key | Precise role and locator |
|---|---|
| `wang2017sqn` | Four-author SIAM Journal on Optimization 27(2), 927–956 (2017), Section 2, assumptions AS.1–4 and Lemma 2.1: predictable variable-metric drift. Do not substitute the earlier three-author preprint. |
| `howard2020timeuniform` | Probability Surveys 17, 257–317 (2020), Lemma 1: Ville; also exponential-supermartingale construction. Published title retained. |
| `robbins1971almost` | Original 1971 chapter, pp. 233–257: related almost-supermartingale framework. Only metadata/publisher summary were accessible in the prior audit; the appendix proves the special argument directly and does not claim a detailed theorem transfer from the chapter. |
| `reddi2018adam` | ICLR 2018 despite 2019 arXiv deposit: unrestricted Adam counterexamples motivate assumptions, not failure of current runs. |
| `zhang2021reinforce` | AAAI 2021, 35(12), 10887–10895; Algorithms 2–4, Theorem 6 / Corollary 11 in the full author manuscript: stochastic tabular log-barrier precedent with different regret conclusions. The BibTeX conference entry omits issue number because conventional styles warn on simultaneous volume and number. Full verified metadata is 35(12). |
| `durrett2019probability` | Fifth edition (2019), Theorem 4.3.4, pp. 225–226 in linked author manuscript: conditional Borel–Cantelli. First-admission proof included. |
| `anceaume2015coupon` | arXiv:1504.03878 (2015), Theorems 2–3: fixed iid sampling; uniform law at fixed non-null mass. No adaptive-optimality transfer. |
| `branicky1998multiple` | IEEE TAC 43(4), 475–482 (1998): energy accounting across switches. Our jump inequality is proved directly without invoking its equilibrium-stability theorem. |

Existing `ecoffet2021return` and `rolnick2019replay` entries are reused for archive/replay motivation, explicitly distinguished from execution-key admission/retention guarantees. No duplicates are needed.

## Review and verification

- Root reviewed the core proofs and sharper corollary: no remaining mathematical concerns.
- Entropy/geometry reviewer independently passed the sharper corollary's cross term, spectral bound, conditional Hoeffding construction, and Ville optimization. Requests to inherit all theorem assumptions and state the delta range were applied.
- Root's notation and main-theorem probability-quantifier clarifications were applied.
- Standalone compilation using both fragments and real bibliography entries produced five pages. Final LaTeX/BibTeX logs have no warnings, overfull boxes, undefined citations, or unresolved references. The root still builds/audits actual manuscript versions.
- Label uniqueness and prefix checks passed; no raw literature href prose remains.
- Original supplement's 216 optimizer drift cases remain arithmetic corroboration. No proof rests on those numerical checks.

Disposable standalone files are in `/tmp/theory_optimizer_admission_check/`. Stable fragment hashes:

```json
{
  "optimizer_appendix.tex": "553c16bf673cb213dc4c68779cc30586522244576fde77c6ada6dbc9fb2f1c3b",
  "discovery_appendix.tex": "2b442902168f173c000c8bb91a2b83e5956013f1212e04fd03dbdf7353a1dbd9",
  "optimizer_admission_refs.bib": "558c097ffcbc8d4166c282ccd42b8de19559255209255b8f95032fc9d0f08f0f"
}
```

Root integration note: the final manuscript wording received subsequent prose reflow and notation alignment for the existing layout checks. Handoff hashes above describe the reviewed draft; final integrated hashes and checks are in `validation.json`. No result assumptions were removed.
