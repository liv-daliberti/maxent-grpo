# E121: fixed-bank identity-level survival telemetry

Preregistered 2026-09-03 before any E121 job was submitted or any E121
training outcome existed.

## Motivation and scope

E121 addresses the empirical gap identified for Theorem A.5: the existing
immutable cohorts retained aggregate bank occupancy but not longitudinal
scores for each individual banked mode. This cohort is a direct mechanism
audit, not a new primary efficacy comparison and not an attempt to validate
the theorem's numerically tiny lower bound.

The cohort contains Qwen2.5-0.5B-Instruct ReplayDr.GRPO on Graph coloring for
seeds 43--47. Graph is fixed in advance because it has finite executable
support, is the paper's primary retention example, and admits unambiguous
canonical keys. This choice is not based on E121 outcomes. Existing E78
terminal results and partial E120-R1 progress were visible before this
registration; no claim of blindness to those earlier cohorts will be made.

## Fixed intervention and horizon

Each run inherits the E78 ReplayDr.GRPO Graph cell for the same seed: group
size 16, learning rate 2e-7, one PPO epoch, beta zero, uniform verified-
likelihood replay with coefficient 0.10, capacity 16, one checkpointed global
round-robin replay group per optimizer update, 384 training prompts per pass,
and eight passes (3,072 learner steps).

Bank admission and fresh-observation counts operate normally through learner
step 383. At learner step 384, before processing that step's fresh group, both
membership and counts become immutable for the rest of training. Replay
continues over the fixed retained exemplars. The freeze step is a constant
configuration value and never reads evaluation, aggregate outcomes, support
size, or an individual key's score.

## Identity-level telemetry

For every materialized replay row after the freeze, persist:

- the learner step and an explicit fixed-bank-active flag;
- a prompt fingerprint and canonical-outcome fingerprint whose pair identifies
  the banked mode;
- the fixed prompt-bank membership fingerprint;
- response-length-normalized teacher-forced mean log probability;
- teacher-forced sequence log probability and the existing replay token count;
- target weight, frozen fresh count, replay coefficient, objective scale, and
  applied gradient diagnostics.

The auditor must establish that each prompt's membership fingerprint is
constant after step 384 and that the outcome rows, prompt rows, mean scores,
and sequence scores are aligned and finite. Raw canonical strings are not
written to the metrics log; the content-addressed run snapshot and bank
checkpoint preserve reproducibility without exposing task answers in routine
telemetry.

## Registered estimands

The analysis population is every `(seed, prompt fingerprint, outcome
fingerprint)` present in the frozen bank and observed in at least two
post-freeze replay visits. No identity may be selected by its score trajectory.

The primary descriptive endpoints are:

1. the fraction of frozen identities with a finite score at every scheduled
   post-freeze observation;
2. the per-identity change in mean token log probability from its first to its
   final post-freeze observation, summarized by seed with median, 10th
   percentile, minimum, and the fraction decreasing by more than 0.5 nat/token;
3. the analogous sequence-log-probability change, reported separately because
   it is length dependent.

Also report visit counts, the number of prompts and identities, and the worst
intermediate drop from the first observation. All five seed summaries and the
pooled empirical distribution are shown; no seed, prompt, key, checkpoint, or
threshold is chosen after inspection. Bootstrap intervals resample prompts
within seed and then seeds, with 10,000 draws and seed 121.

These are language-model score surrogates, not exact categorical probabilities
of canonical modes. Therefore E121 can directly test whether individual fixed
exemplars remain scored and how their likelihoods evolve under replay, but it
cannot numerically verify `exp(-k C_T)` or turn Theorem A.5 into an exact
finite-network guarantee. A missing identity, changing membership fingerprint,
non-finite score, or fewer than two visits is reported as a mechanism-audit
failure, not silently excluded.

## Prior outcomes and stopping

Partial E120-R1 status and previously reported E78 results had been inspected
before this protocol. E121 analysis is nevertheless prospective: no E121 run
or metric exists at registration. There is no early stopping, checkpoint
selection, seed replacement, or outcome-conditioned rerun. Infrastructure
requeues may resume the same cell from its latest durable checkpoint and must
be recorded without changing its scientific identity.

## Scheduling and compute

E121 must not request or use PVL compute. Each job is pinned to an explicit
non-PVL CS node (`node202`, `node203`, or `node204`) under account `allcs`, with
a runtime fence that aborts if any effective scheduler or runtime field
contains the case-insensitive substring `pvl`. The cluster submit plugin may
normalize the requested `cs` partition label to effective `all`; this was
observed on a held, immediately canceled CPU barrier. Such normalization is
accepted only with the explicit non-PVL node constraint and `allcs` account.

All five E121 jobs are submitted only after source compilation and focused
regression tests pass. Slurm rejected one attempted direct 45-ID dependency
before creating any job or ledger. The registered operational implementation
therefore verifies already-terminal E120-R1 IDs as `COMPLETED`, splits every
still-live E120-R1 ID across non-GPU `afterok` barriers of at most ten inputs,
and makes all five science jobs depend `afterok` on every barrier. This bounded
fan-in is transitively equivalent to requiring successful completion of all 45
registered E120-R1 science jobs, and E121 cannot allocate a GPU earlier. The
E121 ledger and campaign row are created at submission time, while the jobs are
dependency-pending.
