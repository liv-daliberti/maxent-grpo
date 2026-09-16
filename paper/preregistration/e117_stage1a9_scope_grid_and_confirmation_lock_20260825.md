# E117 Stage-1 A9: scope, exact grid, and confirmation lock

Frozen: 2026-08-25T12:35:32-04:00 while all twelve E117-R1 jobs were
pending, no E117-R1 optimizer update had occurred, and no Stage-1 seed, job,
sampled response, or outcome existed.

Status: pre-outcome inferential correction and execution lock. This amendment
does not authorize Stage-1 or change its endpoint thresholds. It removes
remaining degrees of freedom, strengthens paired identity, states the actual
scope of inference, and fixes the replication rule before development data can
influence it. PointMaze remains excluded.

## Exact Stage-1 grid

The formerly open "at least 16" and "fresh seeds" choices are now exact:

- paired training seeds: `201, 202, 203`;
- common-random-number evaluation draw labels: exactly `0, ..., 15`;
- checkpoints: exactly `0, 192, ..., 3072` (17 points, eight passes);
- actual cyclic start orders by seed rank:
  - 201: `C-P-F`;
  - 202: `P-F-C`;
  - 203: `F-C-P`.

An exact JSON audit of all registered job manifests found no prior use of
training seeds 201--203. A future execution manifest must additionally freeze
the response-free request-seed projections that realize draw labels 0--15 and
bind the v10 effective contract before submission.

Because every arm and training seed starts from the same pretrained weights,
uses the same prompt bank, and uses the same registered request surface,
step-zero primitive endpoints must agree across *all* seeds and C/P/F arms
within a sentinel/draw, not merely across arms within one seed. The analyzer
must fail closed otherwise.

## Scope of inference

The endpoint estimand is conditional on each registered finite 128-prompt
development bank. Evaluation Monte Carlo uncertainty covers generation draws;
training-seed uncertainty covers fitted-policy variation. Neither is prompt-
population uncertainty, and the analyzer must say so explicitly. No pooled
interval or p-value combines these axes.

The four contexts include two model families but do not factorially cross model
family and domain: Falcon appears only with MathIR. Therefore the broad gate is
replicated multi-context evidence observed under both registered model
families. It does not identify a model-family main effect, a domain main
effect, or their interaction. The field historically named
`cross_model_support` is retained for schema continuity but means only
"actionable contexts include both registered model-family labels."

The independently generated confirmation bank is the preregistered check of
prompt-block transfer. Even a successful confirmation supports only the
registered model--domain contexts and these two generator draws; broader
population language requires a later crossed design or additional prompt
blocks.

## Confirmation locked before selection

If Stage-1 advances either component, run the full C/P/F grid on all four
registered sentinels using the sealed `confirmation` paths. Do not select a
subset of arms or contexts from favorable Stage-1 cells. Use:

- exactly six fresh paired training seeds: `301, ..., 306`;
- exactly the same 16 draw labels and 17-checkpoint horizon;
- two complete cyclic start-order replicates:
  - 301 and 304: `C-P-F`;
  - 302 and 305: `P-F-C`;
  - 303 and 306: `F-C-P`.

An exact JSON audit of all registered job manifests found no prior use of
training seeds 301--306. Confirmation remains conditional on a passing E117
audit and on a Stage-1 component advancing; this document submits no jobs.

For each Stage-1-selected component and sentinel, retain every development
actionability condition. At both terminal and normalized AUC, additionally
require the raw-distinct estimate to exceed two training-seed SEs and to be
positive in at least five of six paired seeds. Continue to require it to exceed
two evaluation Monte Carlo SEs and +0.05, and retain both mean and every-seed
pass-safety checks versus the component denominator and versus C.

A broad claim must reproduce the broad rule: the same component actionable in
at least three of four contexts and contexts bearing both model-family labels.
A Countdown-specific candidate must reproduce on Countdown; every other
context remains mandatory and reported, but the confirmation cannot upgrade
the scope selected in Stage-1. If both components advance, evaluate both by
the same locked rule. These are deterministic replication gates, not p-values;
no multiplicity-adjusted inferential claim is implied.

Development analysis must never read confirmation responses. Confirmation
analysis may identify the already selected Stage-1 component and scope, but
must not import development endpoint values into its estimates, uncertainty,
or gate.
