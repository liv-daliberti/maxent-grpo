# E49E clean-relaunch and negative-control amendment

**Status: FROZEN BEFORE ANY CLEAN-RELAUNCH JUDGE REQUEST OR POLICY TRAINING — 2026-07-24**

This amendment governs the first usable E49E toy trace-bank artifact. It does
not change the learning hypothesis, cohorts, controller, matched arms, or
three-epoch schedule in the canonical E49E protocol.

## Cancelled snapshot

Preprocessing job `30074100` began from a source snapshot that did not include
the protocol-required known-invalid control manifest. The mismatch was
detected after seven soundness responses and before any row-level trace-bank
record, materialized dataset, manual-audit ledger, or policy-training job
existed. The job was cancelled after 63 seconds and its complete evidence was
moved without modification to:

`var/artifacts/e49e_trace_bank_math_toy_aborted_precontrols_20260724`

That snapshot cannot authorize training. Its seven completed deterministic
request-cache records remain terminal: the clean relaunch inherits and binds
them by count and content-addressed cache-tree hash, then validates them under
the clean implementation. They may not be resampled. The frozen identity
records the inherited count and hash and records zero new requests at freeze.

## Required falsification controls

The clean toy launch must freeze and enforce both manifests:

1. `e49e_known_invalid_controls_toy.json`, containing five previously observed
   mathematically invalid or non-self-contained routes. Each must receive both
   soundness audits and must fail double-sound acceptance.
2. `e49e_known_equivalent_controls_toy.json`, containing three previously
   observed duplicate strategy pairs. Each must receive complete soundness
   decisions and must not receive a unanimous distinct pair edge. Rejection
   at completed soundness is also a valid rejection as new.

Any accepted invalid control or false-distinct equivalent control terminates
preprocessing with no dataset and no training. The control manifests, source
snapshot, E49D input, endpoint, no-network preflight, inherited request cache,
canonical protocol, this amendment, launcher, and Slurm wrapper are all bound
into the clean frozen identity before the first new request.

## Test and artifact rule

The focused E49E/menu/canonicalizer suite must pass in the project runtime
before launch. The cancelled artifact remains immutable and the clean artifact
uses the canonical path `var/artifacts/e49e_trace_bank_math_toy_v1`.
