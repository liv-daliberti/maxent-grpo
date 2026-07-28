# E59-MIR: executable MathIR extension of global verified replay

**Status: FROZEN AFTER THE TWO REPORTED BASE-MODEL ENGINEERING PROBES AND
BEFORE ANY E59 POLICY UPDATE — 2026-07-26.**

## Purpose and provenance

E59 is a separately identified fourth-domain extension of E58. It does not
alter, resume, or pool E58's frozen graph-coloring, Countdown, or Python
cohorts. It asks whether E58's latest verified-first global-replay algorithm
can operate on mathematical solution paths when the policy emits a bounded
program that is executed exactly.

This is an exploratory synthetic algebra environment, not a claim that
free-form MATH-500 strategies are validator-bound. E50G remains a failed
natural-language-route calibration and supplies no E59 bank entry, training
example, checkpoint, or semantic label.

## Executable task contract

Each prompt supplies:

- one exact linear equation with concrete integer bindings;
- six prompt-local action IDs, randomly permuted across instances; and
- a maximum of four actions.

The response is only a semicolon-separated action program such as `F;E`.
Every selected ID expands to one concrete MathIR transformation. The
interpreter applies that transformation to both sides of the current equation,
normalizes exact rational symbolic state, and rejects unknown IDs, illegal
operations, repeated states, nonterminal paths, or a wrong solution.

The task validator returns either failure or the ordered normalized state path
created by that same successful execution. Menu labels, whitespace, a terminal
semicolon, and boxing do not define identity. A claimed numeric answer is not
accepted. Thus task reward and canonical identity share one fail-closed
execution:

`response -> action IDs -> concrete transformations -> normalized states -> key`.

The four frozen families are:

1. `a*x + b = c`;
2. `x/a + b = c`;
3. `a*x + b = d*x + c`; and
4. `a*x + b = c - d*x`.

The bounded six-action/four-step support is exhaustively enumerated offline.
Every row has exactly five distinct executable canonical modes. This count is
used only as an evaluation denominator. Training starts with an empty bank and
receives neither the valid programs nor their keys.

## Frozen data

- root: `var/data/mathir_action_menu_v1`;
- generator seed: `5900`;
- training rows: `384`;
- held-out `multi_answer` rows: `128`;
- training row SHA-256:
  `bfc9836206b6e7ed10b42d53ea0c0615559a2a4446cd15f67476b5b0d45d3835`;
- evaluation row SHA-256:
  `df5783ddc9d2a70dd27a237450ce6467884d5ba371032ba8c2cb037bbf328b96`;
- data identity SHA-256:
  `a278cca1b222202a40616750e1e0c56799f02ded838eef51eeb3b097d75b103f`.

Train and evaluation formal-instance identities are disjoint. Every generated
row is certified to contain at least two distinct executable paths, and the
family-level exhaustive support digest is frozen in the data identity.

## Disclosed pretraining-policy probes

These engineering probes were observed before this protocol and therefore
cannot be used as prospective E59 evidence.

At temperature one, top-p one:

- the eight-row, 16-sample probe produced 8/128 verified programs, reward in
  three families, reward on four of eight prompts, and no prompt with two
  observed modes;
- the four-row, 64-sample probe produced 20/256 verified programs, reward in
  two families, reward on two of four prompts, and one prompt with two
  distinct executed modes.

The second probe failed its prewritten three-family coverage condition. E59
does not relabel that probe as passing. Together the probes establish only
that the frozen policy can enter the executable support and that sampled
multi-mode support exists. E59 therefore begins with an engineering smoke,
not a full scientific result.

## Latest-algorithm treatment

The E59 treatment is exactly E58's
`verified_first_global_replay_canonical` variant:

1. no direct token-entropy objective;
2. zero policy-changing objective on all-zero pre-discovery groups;
3. model-generated validator-positive outcomes only;
4. open-set semantic exploration with base coefficient `0.10`, 64 eligible
   warmup observations, EMA `0.90`, and projection-free inverse self-warmup
   control;
5. one-time validator-positive novelty credit `0.50`;
6. verified-mass replay with base coefficient `0.10`, 64-observation
   self-warmup surprisal-ratio control, and no projection;
7. known-mode balance with base coefficient `0.10`, 64-observation inverse
   self-warmup control, and no projection; and
8. exactly one model-discovered verified prompt bank per optimizer update,
   selected by checkpointed prompt-hash round robin, capacity 16.

No controller reads the exhaustive five-mode count, evaluation output, desired
entropy, desired success rate, or desired support size.

## Engineering smoke

Run only the treatment at seed `9010` for one 384-prompt pass. The smoke may
record evaluations, but they cannot change the objective or authorize a
behavioral claim.

The smoke passes only if:

- reward/key parity holds on every response;
- direct token-entropy objectives and controllers are absent;
- all pre-discovery all-zero groups have zero policy-changing augmentation;
- at least one model-generated verified outcome enters a bank;
- after first discovery, exactly one global replay bank is scheduled on every
  recorded optimizer update;
- verified-mass replay activates for singleton banks;
- balance activates only for a bank with at least two verified modes;
- scheduler cursor and all three controller states survive checkpointing;
- every loss, coefficient, observation, and gradient is finite; and
- there is no traceback, worker death, CUDA OOM, or validator exception.

Smoke weights, optimizer state, banks, and evaluations are discarded.

## Conditional matched cohort

Only a clean terminal smoke may authorize a fresh matched cohort:

- arms: ordinary Dr.GRPO and
  `verified_first_global_replay_canonical`;
- seeds: `43,44,45`;
- model: Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- group size: `16`;
- learning rate: `2e-7`;
- PPO epochs: `1`;
- PPO beta: `0`;
- rollout temperature/top-p: `1/1`;
- maximum generated length: `64`;
- prompt-pool passes: `50`;
- evaluation every 96 prompts;
- evaluation: deterministic pass@1 plus four fixed temperature-one `K=8`
  draws; and
- one GPU per run.

Both arms use the same prompts, action menus, validator, passive discovery
telemetry, evaluation requests, data order, and checkpoint cadence. The only
policy-objective difference is the E58 treatment.

Primary outcomes are held-out distinct-correct@8 and coverage@8. Supporting
outcomes are pass@8, mean@8, pass@1, cumulative verified discoveries, mean
bank support, invalid-program rate, replay activation, and controller
trajectories. All seeds and null or negative results remain reportable.
