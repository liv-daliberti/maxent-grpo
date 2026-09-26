# Current experiment figure plan

Status: synchronized with `main.tex` and `FIGURE_MANIFEST.md` at the September 2
freeze. The older ten-method/750-cell display plan is retired.

## Main figures

1. **Problem:** `modecollapse_story.pdf` shows correctness increasing while
   verified support collapses, against a ReplayMaxRL trajectory from the same
   initial samples.
2. **Measurement:** `modebench_examples.pdf` explains reward plus canonical key
   across all five executable domains and notes the two-level construction.
3. **Method:** `verified_support_story.pdf` shows fresh MaxRL search plus
   key-balanced replay and the combined ReplayMaxRL update.
4. **Experiment 1:** `experiment1_retention_comparator_matrix.pdf` combines
   the complete three-scale ReplayDr.GRPO retention contrast with a fully
   matched Qwen-0.5B panel containing ReplayDr.GRPO, MaxRL without replay,
   UCPO, RLEP-Dr, fixed Semantic-MaxEnt without replay, and ordinary GRPO.
5. **Experiment 2:** `e118_all_scale_factorial_progress.pdf` reports only the
   within-seed cross-domain averages for the complete Qwen-0.5B and Falcon-1B
   factorials. `e118_scale_extensions_appendix.pdf` gives five domain rows for
   both models and both endpoints. Qwen-3B remains unplotted until all five
   blocks are complete.
6. **Experiment 3:** `modebench_level_admission.pdf` combines the completed
   harder-but-solvable admission result with the complete terminal Level-2
   Graph four-arm block (`n=5`). The other Level-2 domains remain incomplete.

## Appendix figures retained

- Qwen trajectory-AUC retention effects.
- Direct-comparator endpoint effects and trajectories.
- One fixed Semantic-MaxEnt row in the main direct-comparator panel, plus one bounded supporting comparison in the appendix.
- Replay actuation telemetry.

## Promotion gates

- `n=5` terminal paired seeds: mean plus 95% Student-t interval.
- `n<5`: raw matched prefix and descriptive mean only.
- No shallow checkpoint substitutes for a terminal seed.
- No model/domain/level pooling.
- No Qwen-3B row made of empty panels.
- No Level-2 treatment language attached to the admission plot.
- No return of the omnibus endpoint frontier or Qwen-only MaxRL snapshot.

## Next renderers

1. Extend the all-scale factorial builder to emit complete per-seed four-arm
   contrasts and interactions once Falcon/Qwen-3B finish.
2. Add the Level-2 terminal builder and a within-level factorial-effect plot.
3. Add a compact Level-1 versus Level-2 effect comparison using independent
   intervals after both level records are frozen.

See `EXPERIMENT_SECTION_PLAN.md` for execution priorities and
`FIGURE_MANIFEST.md` for the exact compiled asset list.
