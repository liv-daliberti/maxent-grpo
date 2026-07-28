# E58: verified-frontier global replay canonical control

**Status: FROZEN BEFORE E58 ENGINEERING SMOKE OR SENTINEL SUBMISSION — 2026-07-26**

## Failure being repaired

E57 repaired the invalid Python cold start by removing unconditional token
entropy. Its seed-9010 Python treatment preserved the pretrained proposal until
the model produced its first executable success at optimizer step 91, then
found additional validator-positive outputs. This demonstrated that direct
token entropy was the wrong pre-discovery actuator.

E57 nevertheless exposes a second information-timing failure. Its verified
replay bank is prompt-local *and* is materialized only when the same prompt is
in the current rollout. With 384 unique training prompts, a singleton
discovered late in one pass may receive one immediate replay update and then
wait nearly a complete pass before it can act again. The verified-mass
controller therefore receives sparse observations, the known-mode balance
controller remains ineligible until a prompt independently reveals two modes,
and correct executable structure cannot propagate promptly across the model.
This is especially damaging in the sparse-success Python domain. It also leaves
known graph and Countdown modes exposed between prompt revisits.

The repair must change actuator availability, not invent a desired entropy,
support size, or answer catalogue.

## Verified-frontier global replay

E58 keeps E57's zero-gradient verified-first cold start and its three
projection-free, self-warmup controllers. It changes only replay scheduling.

After the model creates its first validator-positive prompt-local bank, every
optimizer update materializes exactly one previously model-discovered verified
bank. Eligible banks are traversed in deterministic round-robin order over
their prompt hashes. The cursor is checkpointed. A bank contributes at most
one stored model-generated exemplar per observed canonical outcome, up to the
fixed replay capacity of 16 outcomes.

This schedule has the following information boundary:

- admission still requires the external task validator;
- the canonical key still comes from executed/constraint-checked behavior, not
  model-declared text;
- selection sees only the accumulated verified banks and a fixed compute
  budget of one bank per optimizer update;
- selection never reads exhaustive support, a gold mode count, evaluation
  output, a desired entropy, or a desired success rate.

The fixed one-bank budget is a compute measure, not a semantic target. Before
the first verified discovery there is no eligible replay bank and therefore no
replay gradient.

## Objectives and controllers

For the scheduled bank, E58 uses E57's split objective:

1. **Verified mass.** Every scheduled bank, including a singleton, receives
   uniform verified-exemplar likelihood. Its unbounded coefficient is
   `mu = 0.10 * surprisal_ema / surprisal_ref` after 64 eligible observations.
2. **Known-mode balance.** A scheduled bank with at least two observed modes
   receives `KL(U_bank || q_model)`. Its unbounded coefficient is
   `alpha = 0.10 * h_ref / h_ema` after 64 eligible observations.
3. **Open-set semantic exploration.** Validator-positive on-policy samples
   retain E57's structural unseen bucket and unbounded coefficient
   `beta = 0.10 * z_ref / z_ema` after 64 eligible observations.

Every reference is the corresponding controller's own warmup statistic.
There is no coefficient projection. The one-time validator-positive novelty
bonus remains `0.50`. Direct token entropy and its controller remain absent.

With one scheduled bank, the replay contribution remains

`(15/16) * (1/16) * (mu * L_mass + alpha * L_balance)`.

## Engineering smoke

Run executable Python factors for 256 optimizer updates from the frozen
Qwen2.5-0.5B-Instruct revision and seed `9010`. Stochastic mode-coverage
evaluation is disabled. The runner's automatic deterministic endpoint pass@1
diagnostic may execute, but it is write-only telemetry and cannot select or
modify the mechanism.

The smoke passes only if:

- all pre-discovery all-zero groups have zero task/semantic/replay gradient;
- direct MaxEnt telemetry and controller state are absent;
- a verifier-positive model output is discovered without a seeded exemplar;
- the global scheduler is configured for exactly one bank per update;
- the discovery update and every subsequent recorded update contain exactly
  one replay bank with at least one validator-positive exemplar;
- verified-mass controller observations advance on every post-discovery
  update, while balance and semantic controllers advance only when eligible;
- scheduler configuration and cursor survive checkpoint serialization;
- every loss, gradient, coefficient, controller observation, and scheduler
  diagnostic is finite;
- every coefficient reports no projection and no gold-support feedback;
- no traceback, worker death, CUDA OOM, or nonfinite event occurs.

Smoke weights, optimizer state, banks, and any incidental samples are
discarded.

## Three-domain sentinel

Only a clean terminal smoke bound to this protocol, source snapshot, execution
snapshot, launcher, auditor, and exact job manifest may authorize one fresh
seed-9010 E58 treatment in graph coloring, Countdown easy3, and executable
Python factors.

The model revision, datasets, prompt/evaluation row order, group size 16,
learning rate `2e-7`, one PPO epoch, PPO `beta=0`, max norm 1, temperature 1,
top-p 1, response limit 192, quarter-pass evaluation cadence, and 50 complete
prompt-pool passes remain matched to E57 and the frozen E53 seed-9010 Dr.GRPO
controls.

For each domain, over the final eight matched quarter-pass evaluations, E58
must:

- have higher mean distinct-correct@8 than matched Dr.GRPO;
- win distinct-correct@8 at least six of eight times;
- have higher mean `(distinct-correct@8 - pass@8)` than matched Dr.GRPO;
- win that excess-multiplicity quantity at least six of eight times;
- show positive excess multiplicity at least six of eight times;
- retain at least 75% of its own best rolling-eight distinct-correct mean;
- trail matched terminal pass@8 and mean@8 by no more than `0.03`.

The final 64 training records must remain within the matched-control
response-length mean plus 32 tokens and no-EOS mean plus one response per
group. Runtime, information-firewall, objective-isolation, scheduler,
controller, checkpoint, or domain failure rejects E58.

These gates are relative to a matched learning control and to each run's own
history. No ground-truth support size or absolute diversity target is supplied
to training or used to tune a controller.

## Conditional replication

Only a terminal three-domain approval may authorize fresh E58 treatment seeds
43, 44, and 45. Every treatment seed and the three-seed mean must independently
pass the same stability, multiplicity, task-quality, runtime, and information
firewall gates.
