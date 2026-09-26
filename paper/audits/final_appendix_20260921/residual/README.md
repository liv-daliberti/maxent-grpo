# Residual appendix language and caption audit

Read-only scan of the installed `paper/main.pdf`, **PDF pages 15–118**, beginning at the appendix organization and ending after Q. Main text, main references, and the active R/S cleanup are excluded. The PDF hash and checks are in `validation.json`. The scan also follows included TeX and preamble macro inputs: **54 files/scope blocks**. No live manuscript, generator, PDF, or data files were edited.

Two actionable regions remain in the scanned PDF. Both are confined to the domain-definition material on pages 21–22. Exact replacement proposals are supplied for root integration; the remaining figure-caption style check passes.

## Important residuals

| Rendered page | Source at scan time | Phrase/problem | Proposed treatment |
| --- | --- | --- | --- |
| 21–22 | `paper/main.tex:1184` onward, B.1 | Reader-directed/rhetorical framing: “a reader may fairly ask,” “The question is decidable rather than terminological,” “The domain that settles the narrow question,” “Whatever the right name,” and “we do not claim otherwise.” | Replace the subsection prose with direct definitions and descriptive domain comparisons. Preserve all generated collision/count macros and the route/behaviour/answer distinction. |
| 22 | `paper/main.tex:1223` | “these are reasoning modes” remains despite the requested terminology update; the surrounding paragraph infers whether the model reasons and what deliberation produces from an explicit provider setting. | Describe solution modes and the tested configurations. Preserve the 1.62→0.99 and 1.85→1.18 mean counts, but identify them as raw distinct@8, distinguish conditional PCMD, and retain the limitation that explicit settings do not establish absence of hidden computation. |
| 21 | `paper/main.tex:1212`–1215, B.1 | MathIR routes are called “unreachable rather than abandoned,” inferred from absent observed outputs. | Preserve longer certified derivations and their observed absence; state that this does not establish unreachability. |
| 21 | `paper/main.tex:1128`–1143, `tab:tasks` caption | The MathIR gap is called “structural rather than a sampling limit.” The caption also describes the two columns as observations on the same prompts, although catalogue counts use all 128 evaluation rows and Reached uses a 32-prompt subset. | Remove the unsupported causal/exhaustiveness language, distinguish the two populations, and call the first column certified catalogue counts. Countdown admits valid keys outside its enumerated catalogue, as the paper already states elsewhere. |

The first three rows are one local B.1 rewrite, not separate changes elsewhere. Files:

- `b1_before.tex` and **`b1_replacement.tex`**: preserve the section label, every `MDkey` numerical macro in the same order, and the quoted hosted counts. The proposed heading is **Solution mode definitions and concentration by domain**, so the manual TOC entry should be synchronized.
- `task_caption_before.tex` and **`task_caption_replacement.tex`**: preserve all table macros and cross-references. The table body and all values remain untouched. The caption defines its certified counts and sampled coverage populations without claiming an unattainable support region.

The larger reasoning-control subsection already states the needed limits correctly: `paper/results/hosted_reasoning_off_20260912_appendix.tex`. It tabulates five validated deployments on common prompts, with lower raw success/mode counts but mixed conditional-diversity changes. The B.1 proposal agrees with that existing detailed section.

The domain-population check reads `ops/build_paper_domain_support_macros.py` and `paper/results/domain_support.json`: Valid modes uses `answer_mode_count` on the full evaluation split; Reached uses the existing 512-draw, 32-prompt hosted summary. The generator was not run. These observations do not change any measured values.

## Caption style

All **28 appendix figure captions, Figures 13–40**, begin with a bold takeaway followed by setup/encoding details. PDF text/font extraction confirms the opening uses the bold Times-family font (`NimbusRomNo9L-Medi`), rather than merely finding `textbf` in unused source. The parsed list is in `figure_captions.json`, with pages and lead text. No figure-caption style outlier was found through Q; no broad recaptioning is proposed.

## Excluded false positives

The scan distinguishes manuscript-history language from scientific content. No change is proposed for:

- fixed/frozen model weights or public task inputs;
- bank admission, retained exemplars, or historical counts that define an actual algorithm;
- conditioning on training history in a probability statement;
- checkpoint selection, missingness, bootstrap replicates, or a prespecified key needed for statistical validity;
- original/revised wording as the two conditions of a prompt comparison;
- provider reasoning effort/disabled settings;
- a statement that a retry reproduces an already observed solution mode;
- the cost-accounting limitation that the estimates are not a cumulative compute audit;
- unrendered labels, source comments, or the R/S entries in the manual TOC, which are being handled by the agents assigned those sections.

A normalized PDF scan that removes line numbers and joins hyphenated line breaks finds one remaining `reasoning modes` occurrence (B.1, page22) and no `post-hoc`, manuscript, camera-ready, or preregistered phrase through Q. Candidate scans are saved for traceability; they include intentional scientific terms and are not themselves a list of recommended deletions.

This audit covers residual language and caption style. It does not re-run the earlier numerical or theorem audits, and it does not certify every claim outside the concrete regions identified above.
