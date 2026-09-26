# Worked examples and reader progression — September 5, 2026

Added worked illustrations throughout all **12 theory subsections**, including all four parts of the survival-certificate section, in the [full paper](../main.pdf) and [workshop paper](../mathai2026/main.pdf). The [workshop source bundle](../mathai2026/mathai2026-source.zip) is rebuilt.

The appendix now follows a recurring three-correct-mode policy: `p=(.30,.15,.05,.50)`, correct mass `P=.50`, and conditional correct-mode proportions `q=(.60,.30,.10)`. Readers first see why equal rewards can produce unequal increments, then why improving correctness can coexist with losing alternatives. The later examples show what changes with geometry, entropy, replay, admission, and finite measurement. Short openings, descriptive example headings, and transitions identify the distinction each result resolves.

| Theory topic | Full / workshop section | What the example reveals |
|---|---|---|
| grpo mean | A.1 / E.1 | Three correct modes, equal reward, sixfold logit increments |
| collapse | A.2 / E.2 | Local minority loss and the effect of breaking an exact tie |
| size | A.3 / E.3 | Same correctness with different remaining breadth; positive decay exponents |
| natural gradient | A.4 / E.4 | Fisher geometry preserves ratios; replay replenishes the rare mode |
| entropy | A.5 / E.5 | Exact full entropy preserves a small incorrect mass too |
| entropy comparison | A.6 / E.6 | Same conditional spread, different correctness and replay loss |
| gradient availability | A.7 / E.7 | Rare keys can receive replay despite near-certain absence from fresh groups |
| replay | A.8 / E.8 | A loss barrier, inverse-length weights, and an initially improving unbanked mode |
| exemplar bridge | A.9 / E.9 | Complete-response probability and the draw budget needed to observe protection |
| optimizer | A.10 / E.10 | Noise budgets and the stronger finite-horizon probability floor |
| discovery admission | A.11 / E.11 | Admission opportunities, capacity gates, and bank-switch costs |
| survival certificates | A.12 / E.12 | Sharp floors, entropy hiding missing modes, key aggregation, and finite observations |

The 16-mode comparison now explains why normalization strengthens the floor from approximately `4.62e−20` to `0.0339712`, how this compares with the uniform target's `6.25%`, and why it also certifies about `99%` total bank mass. The entropy example then shows that `97.7%` of maximum entropy can coexist with a completely missing mode. Complete-response aggregation and checkpoint-count examples explain what each certificate can—and cannot—measure.

All worked values are hypothetical. Exact formulas precede rounded values where needed; the matched-accuracy table explicitly distinguishes numerically integrated trajectories from theorem floors, with the displayed floors rounded down. The scalar noisy-update example specifies a fixed one-token horizon with two possible complete responses, avoiding any prefix/complete-response ambiguity.

## Verification

The calculations were checked with four saved scripts and receipts: [early categorical examples](theory_examples_20260905/verify_early_examples.json), [entropy and geometry](theory_examples_20260905/entropy_example_verification.json), [optimizer and admission](theory_examples_20260905/verify_optimizer_examples.json), and [replay and survival](theory_examples_20260905/verify_survival_examples.json). Verification includes enumeration of all 256 four-draw groups and agreement between independent full-logit and reduced-ODE integrations to within `1.8e−14` for the matched-accuracy example. An independent mathematical review confirmed the reported values, downward-rounded floors, and stated scope.

All **45 formal statement/proof blocks** and all **25 theory citation commands** are byte-preserved. Both main texts, both bibliographies, and non-theory manuscript content are unchanged. The theory matches across versions apart from the two existing document-specific cross-references.

`make -C paper` passes the manuscript/evidence contract, frozen-prompt check, and rendered layout gate (274 prose blocks). `make -C paper/mathai2026 bundle` passes the submission checks and produces a 42-file source bundle matching current files. Main texts remain nine and four pages; complete PDFs are 56 and 57 pages. Final compiler logs contain no unresolved references/citations, duplicate labels, or overfull boxes. All 28 frozen copied assets remain intact.

Representative full-paper pages 14, 18, 30, and 34 were visually reviewed. The appendix title was shortened to fit on one line. See the [coverage map](theory_examples_20260905/example_coverage.json), [source patch](theory_examples_20260905/examples.patch), and [final validation receipt](theory_examples_20260905/validation.json).
