# Level-3 neutral Python repair

The active calibration is registered at `var/artifacts/modebench_level3_neutral_v1/registration.json` (SHA-256 `48a5bdee7be17a16fa914b76e4f1db98c47a23b7e35e4d42c4a92ab63477b935`). Development array 31252690 evaluates frozen Qwen2.5-3B on 664 fresh Python problems, with four independent groups of eight responses per problem. Its controller is `ops/exp_scaling/calibrate_modebench_level3_neutral.py`.

`development_structure.json` independently confirms all 664 problems are unique and disjoint from 7,326 historical cases; each tier has the exact 43-cell solution-count histogram; all 1,328 executable witnesses verify; and all 2,656 request seed blocks are distinct. All 327 registered source/data hashes matched. Thirteen evaluator/generator/controller tests passed.

The target remains the measured Qwen2.5-0.5B Level-1 pass@1 of .2109375 and pass@8 of .76953125, with absolute tolerances .04 and .08. The reference keeps its original hinted wording. This is numerical calibration across explicitly specified interfaces, not evidence of a causal difficulty-only comparison or statistical equivalence.

The controller fits only development correctness. After a passing fit it creates a separate dataset at `var/data/modebench_level3_matched_neutral_v1`, with fresh 384-row training and 128-row confirmation splits and a fixed 128-row development selection. The other four domains retain their original bytes. Confirmation regrades all 4,096 responses with the external verifier, reports uncertainty, and writes `admission.json` only if both unrounded gates pass. Development failure or confirmation failure retains its evidence without selecting a new subset from those outcomes.

Both manuscript sources now distinguish the admitted historical Level-3 V3 release from this pending neutral-Python revision. No existing experiment is relabeled as having used the new dataset. `before/` preserves source versions before this disclosure; `parent_build.log` records compilation checks.

Two earlier construction-only preparations in `artifacts/modebench_level3_neutral_calibration_20260911{,_r2}` failed an exact-support capacity check before model sampling. Their preparation code is preserved under `superseded_preparation/`; the old entry point refuses duplicate submissions. The active shared controller owns collection and fitting. This independent audit did not launch a second development run.
