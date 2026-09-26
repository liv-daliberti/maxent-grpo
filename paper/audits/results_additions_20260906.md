# Result additions — September 6, 2026 UTC

Implemented the user-authorized five-seed Qwen2.5-3B Python integration and the registered E120 primary breadth analysis in both the ICLR and MATH-AI NeurIPS workshop manuscripts.

## Scientific additions

The completed Python factorial uses all four methods and registered seeds 70–74. ReplayMaxRL minus MaxRL is +.39140625 raw distinct@8 (unadjusted paired Student-t 95% interval [.078679823, .704132677]) and +.2765625 pass@8 ([-.036915277, .590040277]). The ICLR main text links to the full appendix table; the workshop includes the same table and interpretation in its appendix. Coverage distinguishes 11 complete MaxRL pair blocks from 10 complete four-arm blocks. The Falcon Countdown Dr.GRPO track retains four admissible seeds. No Qwen3B cross-domain average or model-size claim was added.

E120 uses the unchanged September 4, 15:40:26 UTC snapshot of 25 Qwen0.5B treatment cells. The primary contrast is uniform minus frequency weighting on B=distinct@8−pass@8; co-primary correctness is uniform minus frequency on pass@8. The new standalone builder emits every seed effect, domain means, the registered five-domain mean, and nominal paired percentile bootstrap intervals. Exact enumeration covers all 3,125 ordered seed resamples; domain averaging precedes resampling so the five-domain vector stays paired within each seed. The fixed source bytes are SHA256-pinned. The percentile/interpolation and shared-seed conventions are documented analysis implementation choices, not falsely described as preregistered details.

The five-domain B mean is +.3178125 [.272265625, .3578125]; correctness is +.00984375 [−.119375, .1390625]. Both appendices explain that the latter does not establish noninferiority, equivalence, or improved accuracy. Graph, Countdown and Pantry have positive breadth intervals; Python and MathIR remain inconclusive. The original supporting raw-endpoint table is retained with its frequency-minus-uniform sign explicitly distinguished from the primary contrast. All paired seed values and interval conventions appear in the manuscripts. Both main texts reference the primary finding.

## Reproduction and integrity

Run `python ops/exp_scaling/build_paper_e120_primary_breadth.py` or `make -C paper primary-breadth` to reproduce the new analysis and two table bodies. The parent manuscript build independently recomputes these outputs and checks exact equality, source provenance and table inclusion. The E118 Python table generator now emits its own closing booktabs rule, resolving the previously unbuilt table's alignment error. Registry wording no longer mislabels the new Python result as part of the September 4 freeze.

Independent review parsed all 20 raw Python factorial endpoints and all 50 E120 arm endpoints, requiring exact step 3072, K=8 and four unique fixed-seed draws. Every endpoint matched; no conflicting terminal payload was found. A separate multinomial-weighted bootstrap enumeration verified all 18 generated summaries, 90 seed contrasts and 36 interval limits. See `results_additions_20260906/independent_e118_e120_review.md`.

## Validation

The new analysis tests passed 14/14; existing endpoint integrity and E118 seed-track tests passed 17/17. ICLR compilation includes the prompt contract, evidence gates, nine-page main-body check and rendered paragraph-ending check. Workshop validation preserves four content pages, six main figures and seven supplementary figures, reference placement, anonymous official style and source/asset hashes. Its packaging also checks referenced files, records compiled input hashes and rejects stale builds. Final build logs and the standalone-bundle validation record are stored in `results_additions_20260906/`.

The added E120 summary and seed tables and E118 Python table were visually inspected in the rendered PDFs. No new experiments were launched; live partial cohorts were not incorporated into the frozen E120 analysis. Before copies and a focused patch are retained alongside this audit.
