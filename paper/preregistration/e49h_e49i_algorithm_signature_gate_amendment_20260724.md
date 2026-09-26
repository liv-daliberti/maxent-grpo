# E49H amendment: replace failed E49G with calibrated E49I gate

**Status: FROZEN BEFORE ANY E49H 72B REQUEST — 2026-07-24**

E49G completed its 116 requested assessments and failed calibration.  It
produced zero false-new predictions but zero true-distinct predictions; its
minimal-consequence equivalence definition therefore provides no usable
support.  Some equivalent assessments also used empty schema fields, so the
frozen completeness check failed.  E49G cannot authorize E49H.

This amendment replaces only E49H's automated pair-veto dependency with the
E49I executable algorithm-signature veto.  E49I must first pass its own
frozen calibration with zero false-new predictions, all hidden equivalent
controls rejected, at least three of four manually distinct pairs recovered,
and every request complete.

All other E49H requirements are unchanged:

- both routes independently pass both E49E action-execution audits;
- the calibrated E49I decision requires at least three of four distinct
  votes;
- a sealed manual audit follows on the unseen curated cohort;
- manual false-new must be exactly zero;
- support must reach at least 10 train and 10 evaluation rows; and
- no policy training begins before every gate passes.

The E49H contracts remain byte-identical to the cohort frozen before this
amendment.  No E49H judge request existed when this amendment was written.
