# Pantry support-retention split correction

**Status: FROZEN AFTER CALIBRATION AND BEFORE FINAL-COHORT SUBMISSION — 2026-07-30**

The repair-v1 launcher identity labeled jobs 30204528–30204529
`development_only`, but the inherited comparative runtime always reads
`$DATA_ROOT/eval`. The job logs confirm that both calibration arms loaded all
128 rows from `pantry_plan_modebench_v3_repair/eval`; they did not load the
64-row `dev` directory.

The calibration gate remains a valid development selection outcome, but its
split label was wrong and the 128-row directory is not eligible as sealed
final data.

Before any five-seed final job is submitted, an immutable final view maps:

- source `train/` to final-view `train/`;
- previously untouched source `dev/` to final-view `eval/`; and
- excludes the calibration-loaded source `eval/`.

The view records all three source tree hashes and the two calibration job IDs.
The final audit must confirm the final-view tree hash and 64 evaluation rows.
No calibration endpoint is relabeled as a final result.
