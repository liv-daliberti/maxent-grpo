# Long-paper main rewrite — September 11, 2026

The seven-page build below is preserved as the original rewrite. The
[latest verified revision](resumed_verification/README.md) includes the newer
hosted comparison and occupies eight main pages, with references on page nine.

The long scientific main now occupies **seven pages**, including **all six main
figures**, under the nine-page limit. References begin on page eight. The complete
PDF also contains the references and appendix.

- [Main text only](main-text.pdf)
- [Final full paper](final/main.pdf) and [source](final/main.tex)
- [Validation](validation.json), [compiled input hashes](compiled_inputs.json),
  [build log](build.log), and [25 passing pagination tests](tests.log)
- [Previous main source](before/paper/main.tex) and [previous PDF](before/paper/main.pdf)

The rewrite follows one argument: correctness does not identify which verified
alternatives remain sampleable; execution keys make those alternatives measurable;
the same keys organize replay memory; three controlled comparisons test its value.
Related work follows the results. Exact effect sizes, intervals, metric identities,
and the replay-objective decomposition accompany the appendix evidence. The three
relocated quantitative paragraphs preserve unique printed results, including
unfavorable or uncertain effects. Repeated numerical displays were consolidated.

All existing model rows and figure data are retained, including 3B MaxRL/ReplayMaxRL,
both training metrics, partial-cohort rules, and the current two-deployment hosted
comparison. UCPO/RLEP remain restricted to 0.5B/1B. Figure captions distinguish the
illustrative opening, paired replay contrasts, and descriptive partial averages.
The theoretical and finite-sample boundaries remain explicit.

A concurrent manuscript edit was preserved in `concurrent_main_before_merge.tex`;
its metric appendix, replay decomposition, hosted update, and disclosures were
merged. The earlier checker variant is also preserved. Compilations were moved to
an isolated output directory after concurrent builds collided. The final artifacts
were validated before atomic promotion and match the recorded input hashes.

`ops/check_paper_main_length.py` now checks the actual Figure 1–6 caption pages,
main section pages, final main-text marker, and References boundary. This rejects
both a conclusion that spills past page nine and a delayed main figure. It does
not constrain the length of the references or supplement. The normal long-paper
build invokes this gate through the current evidence checker.
