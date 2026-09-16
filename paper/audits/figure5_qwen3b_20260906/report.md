Figure 5 now includes Qwen2.5-3B's completed Dr.GRPO/ReplayDr.GRPO track as a third model row. The existing Qwen2.5-0.5B and Falcon3-1B rows, all MaxRL endpoint arrays and the Falcon Countdown seed-59 exclusion remain exactly unchanged. Qwen3B MaxRL remains incomplete and has no cross-domain aggregate.

The added row uses all 50 admitted E80-R1 terminal cells: five registered seeds (70–74), five domains and two arms. Each terminal cell has four valid draws at exact step 3072 in the frozen core endpoint audit. The E80-R1 ledger hash matches both the terminal and initial-reference source records. All 25 initial-reference control endpoints agree exactly with the admitted terminal control values.

| Metric | Untrained | Dr.GRPO | ReplayDr.GRPO | Replay minus Dr.GRPO |
|---|---:|---:|---:|---:|
| pass@8 | 0.355625 | 0.45515625 | 0.734296875 | 0.279140625 |
| distinct@8 | 0.56078125 | 0.575234375 | 1.134375 | 0.559140625 |

Each value averages domains within the same five paired seeds. These are post-hoc descriptive cross-domain summaries, with no pooled inference.

Implementation: `ops/exp_scaling/plot_paper_e118_all_scale_progress.py` adds an independent completed Qwen3B reference track, leaving the unfinished E118 MaxRL intersections intact. `ops/check_paper_current_contract.py` verifies its exact seed sets and initial/terminal averages and prohibits a Qwen3B MaxRL aggregate. Captions and relevant discussion are updated in both manuscript forms; workshop snapshot provenance and its build receipt are current.

Verification completed:

- Seven focused seed-integrity tests passed, including complete Qwen3B reference data with empty MaxRL intersections and rejection of a missing reference pair.
- Existing model cells and average arrays, all MaxRL data and original endpoint audits compare exactly with the saved baseline in `before/`.
- Full manuscript build passed: five frozen prompts match, paper contract passes, and all 281 checked prose blocks meet the final-line-fill rule. The rebuilt full manuscript has 59 pages.
- Workshop build passed: four content pages, references begin on page 5, six main figures and seven supplementary figures; all source/artifact hashes match, without unresolved references or overfull boxes.
- The standalone figure, full-paper page 8 and workshop page 4 were visually inspected for labels, clipping, overlap and page composition.

Artifacts: `paper/figures/e118_all_scale_factorial_progress.{pdf,png,json}`, mirrored workshop PDF/JSON, `paper/main.pdf`, and `paper/mathai2026/main.pdf`. `refresh_figure5.py` reproduces this targeted update from the saved figure record without rereading or refreshing ongoing E118 runs. `validation.json` records exact preservation checks, averages, sources and final artifact hashes. Final build logs are `paper-build-r3.log` and `workshop-build-final.log`.
