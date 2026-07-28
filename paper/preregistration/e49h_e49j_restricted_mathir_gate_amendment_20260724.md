# E49H second amendment: replace failed E49I with calibrated E49J MathIR

**Status: FROZEN BEFORE ANY E49H 72B REQUEST — 2026-07-24**

E49I completed all 116 frozen free-text algorithm-signature assessments but
failed its calibration. It made 16 false-new predictions among 25 manually
non-distinct or unsound pairs (false-new rate 0.64), with only 3/4 recall on
the manually distinct pairs. Its report SHA-256 is
`23a6bb08eecdef63bdb912998f8a1a933d68ac4d7d52924a7b94a7f081348f4e`.
E49I cannot authorize E49H.

This amendment replaces only E49H's automated pair-veto dependency with the
E49J restricted MathIR executable-signature gate. E49J completed all 116
frozen assessments and passed: zero false-new predictions across all 25
negative pairs, all three hidden equivalent controls rejected, and all four
manually distinct pairs recovered. Its frozen calibration-report SHA-256 is
`3400db77d6da7572395fe4d40b03ec59b792e2682c18c49ce861d32206bab3db`;
its decision-file SHA-256 is
`f457487b308ceb966d3f32d11b95e66b0902b8542bee208856a9e3e7275e9e63`.

For each E49H candidate, both routes must still independently pass both E49E
literal/adversarial execution audits. E49J then parses the executed traces
into its frozen restricted operator enum. Novelty is accepted only by
four-way unanimity of the deterministic local signature comparison.
Malformed, incomplete, unsound, empty, or disagreeing parses reject novelty.
The model's prose rationale never determines the decision.

All remaining E49H requirements are unchanged: a sealed blinded manual audit
with injected equivalent controls follows the automated gate; manual
false-new must be exactly zero; retained support must reach at least 10 train
and 10 evaluation rows; every row must retain nonzero validated support; all
prompts must fit the frozen context budget; and no policy training begins
before every gate passes.

The 22 E49H contracts remain byte-identical with SHA-256
`81746269950c6eabc092451380c4c04d175c763b4f26e875384fe1fccc805a72`.
No E49H judge request existed when this amendment was written.
