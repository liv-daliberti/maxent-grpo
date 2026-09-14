# Staged introductory model-icon table

**Not activated or built.** The accompanying `frontier_motivation_staged_20260911_1553.tex` is a new standalone source file. No existing manuscript, figure contract, image, or build artifact was changed while staging it.

The table contains only the completed GPT-5.6 Sol and Claude Opus 5 runs. It uses the original `paper/icons/openai.png` and `paper/icons/claude.png` assets identified by the parent agent. Their rendering has not been tested in this table.

| Deployment | Level-3 Graph verified responses | Distinct modes per eight draws | Same-mode correct pairs |
|---|---:|---:|---:|
| GPT-5.6 Sol | 1,024 / 1,024 | 148 / 128 = 1.15625 | 3,407 / 3,584 = 95.0614% |
| Claude Opus 5 | 1,024 / 1,024 | 130 / 128 = 1.015625 | 3,562 / 3,584 = 99.3862% |

The common conditional uniform-correct-coloring collision reference is 19.0039%, displayed as 19.0%. These Graph scores need no formatting normalization. Source records are the complete, audited `summary.json` files in `artifacts/frontier_modebench_gpt56sol_20260911` and `artifacts/frontier_modebench_claude_opus5_20260911`; the comparison record is `paper/results/frontier_comparison_20260911.json`.

Activation after infrastructure recovery:

1. Read the current authoritative manuscripts again. Add the shared table to the ICLR Introduction as motivation. In the workshop Introduction, replace the paragraph beginning “The loss also appears across our datasets,” which reports the 99.4%, 71.6%, and 41.5% training losses by model size.
2. Remove or shorten the later hosted-result paragraphs that the new introductory table duplicates. Check any remaining `sec:hosted-concentration` references when moving that material; this staged file defines only `tab:frontier-motivation`.
3. Keep the full cross-model appendix and its refusal disclosure. Claude's Level-2/Level-3 Python cells have no eligible correct pairs, so their collision values and five-domain collision averages remain undefined. This motivating table uses Graph, where that issue does not affect the reported scores.
4. Synchronize the original icon assets into the standalone workshop through its normal recursive asset sync. Icons are cosmetic table assets, not additional scientific figures. Do not use `gh_logo.png` for Grok; it is a GitHub icon.
5. Build and inspect both manuscripts, enforce the workshop's four content pages, and check table width, caption wrapping, references, source hashes, and figure counts. The table's exact float placement and page fit are unverified.
6. Add other deployment rows only after their complete graded summaries and provider-outcome audits are available. No pending or refusal-adjusted results have been inserted here.

The environment prevented safe updates to existing files and prevented compilation. Staging this source therefore does not establish that either current PDF contains the requested introductory table.
