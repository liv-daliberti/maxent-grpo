# Local prompt-hint ablation publication audit

The completed local panel contains 27,648 responses from 25 checkpoints on
matched original and neutral Python, MathIR, and Pantry prompts at Levels 2/3.
Problems, verifier, and within-model sampling settings were held fixed. The
registered frontier panel remains unstarted: 0/9,216 responses, awaiting an
authorized Azure credential. Overall status remains `partial_panels`.

## Reviewable outputs

- Four-page focused appendix: `artifacts/modebench_prompt_ablation_20260911/appendix_review/appendix_review.pdf`.
- Long paper: `paper/main.pdf`; nine main pages, references on page ten, 74 total.
- Long main only: `paper/main-body.pdf`; nine pages, no references section.
- Workshop: `paper/mathai2026/main.pdf`; four main pages, references on page five.
- Workshop source: `paper/mathai2026/mathai2026-source.zip`; 122 files.
- Both manuscripts contain the same 19 scientific figures and the complete local ablation.

`final_receipt.json` binds all outputs, inputs, exact publication copies, and
build logs by SHA256. `finalize_local_appendix.py` reproduces these final checks
and extracts the main-body PDF. `workshop_sync_final/workshop_sync.json` preserves
superseded workshop assets and binds the final synchronization.

## Validation

The final ordered parent build passed prompt authentication, the current-paper
numerical and evidential contract, official format/page checks, LaTeX, and all
303 natural-prose line-fill checks. The workshop build and source bundle passed
its independent asset, source hash, page limit, citation, style, and layout
checks. The scientific checker reconstructed all 2,592 CSV metrics and verified
25 checkpoints plus all 5,120 post-hoc Python diagnostic draws. Exact-copy
checks passed for result JSON, generated TeX, diagnostic JSON, and figure files.
The focused appendix has four pages with no unresolved references or overfull
boxes. No new API requests were made during publication validation.

The primary numerical analysis remains sealed. Editorial v2 changes only quotes
and table/figure fitting; both numerical equality receipts preserve that fact.
The Python failure taxonomy and MathIR identity diagnostic are explicitly post
hoc. No hypothetical frontier results are included or plotted.

## Concurrent work and retained history

The final build preserves the concurrent training-census, figures, concentration
analysis, and manuscript updates. The current-paper numerical checker uses
`DATA_PYTHON` (Python 3.10), matching the retained calculation environment.
`python_runtime_rounding_audit.json` records why default Python 3.11 differed
at the last bit for some Student-t intervals; no tolerances or values were
changed. Earlier paper/source snapshots, failed build logs, and prior validated
outputs are retained here. The final logs supersede earlier failures.

## Remaining frontier execution

The user has been asked to enable the intended secure environment credential
or supply a private credential-file path. Automatic approval review rejected
reading a credential from saved chat logs as an unintended credential source;
that action was not executed and must not be retried indirectly. Once an
authorized source is available, use the sealed runner's retained preflight,
inspect receipts, run all six registered cohorts, and analyze before extending
the appendices. See the experiment's `CURRENT_STATUS.md` and execution plan.
