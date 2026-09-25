# Latest KL reconciliation

- Prose: `changes.json` contains six exact old/new replacements against `main-latest-before-prose.tex`; root has integrated them. No live main.tex edits by this agent.
- Scientific corrections: MathIR beta=.3 is PCMD .099 and pass@8 .207, five seeds. Pantry beta=.3 is PCMD .931 and pass@8 .825, five seeds. The aggregate includes all six coefficients and reverses slightly from beta=.2 to .3; the main monotonic interval is now .01 through .2.
- Recordkeeping: removed stale cell-completion counts and missing-summary narration; retained estimator eligibility, seed labels in the table, and uncertainty qualifications.
- Data: latest comparison JSON is unchanged, SHA256 `4a56ac67242af8fed9372e6c5f7ed8396bc7d53a40d1f5fec405283a4eb613d0`.
- Visual correction: removed both plane background fills because the latest curve is not monotone in PCMD. All measured curve coordinates, reference rules, axes, and comparison markers are preserved. Retained the concurrent shared .2/.3 annotation, outer ring at the exact .3 coordinate, and final centered .1 label. The annotations are legible and separate from the curve in the inspected final PNG.
- Knee figure: regenerated from all 28 latest coefficient cells; all 56 PCMD/pass@8 plotted ordinates checked against JSON. Visual inspection found no clipped labels, curves, or legend items.
- `final/` is a coherent saved renderer/data/style/figure snapshot; `final/sha256.json` provides hashes. Live renderer and assets were updated once more from this snapshot; further external writes should be reconciled by root at release, not overwritten repeatedly.
- `validation.json` and `plane-geometry.json` document deterministic checks; `validate.py` validates the saved final plane renderer. The source JSON must still equal the preserved latest input.
- `live-agreement-at-handoff.json` records the last live/snapshot comparison. Parent should verify these hashes immediately before release.
