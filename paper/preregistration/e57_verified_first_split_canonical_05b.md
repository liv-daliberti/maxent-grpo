# E57: verified-first open-set split canonical control

**Status: FROZEN BEFORE E57 ENGINEERING SMOKE OR SENTINEL SUBMISSION — 2026-07-26**

## Failure being repaired

E56 established that unconditional conditional-content token entropy is not a
valid cold-start signal for every domain. In executable Python factors, the
seed-9010 treatment had no verifier-positive rollout through optimizer step 96.
Over the same interval its content-token entropy rose from its warmup regime to
3.22 nats, its unprojected inverse coefficient fell from `0.000075` to
`0.0000346`, its last-64 response window exceeded both preregistered safety
limits, and its first matched evaluation remained `0/0/0`. The matched ordinary
Dr.GRPO run had already discovered a correct output.

This is an information-boundary failure, not an alpha-bound failure. Before a
validator-positive outcome exists, token entropy can distinguish concentrated
from diffuse text but cannot distinguish useful exploration from diverse
malformed programs. Adding an unconditional entropy gradient can therefore
move a competent pretrained proposal distribution away from the rare valid
outputs needed to start the canonical bank.

## Verified-first cold start

E57 removes the direct token-entropy objective and its controller. On a
16-candidate group with no validator-positive outcome:

- ordinary Dr.GRPO task advantages are exactly zero;
- semantic open-set advantages are exactly zero;
- canonical replay has no eligible group;
- the optimizer receives no policy-changing objective.

The pretrained model is therefore preserved as the proposal distribution until
its own temperature-1 sampling produces a verifier-positive outcome. This is a
zero-gradient cold start, not a seeded answer, teacher, support oracle,
domain-specific entropy target, or coefficient bound.

After model-generated verified discovery, E57 retains E56's three target-free
actuators:

1. **Open-set semantic exploration.** The support is the model's observed
   verified modes plus one structural unseen bucket. The coefficient remains
   `beta = 0.10 * z_ref / z_ema` after 64 eligible observations.
2. **Verified mass.** Every observed prompt-local verified exemplar, including
   a singleton, receives uniform likelihood replay. The coefficient remains
   `mu = 0.10 * surprisal_ema / surprisal_ref` after 64 eligible observations.
3. **Known-mode balance.** Prompt-local banks with at least two observed modes
   receive `KL(U_bank || q_model)`. The coefficient remains
   `alpha = 0.10 * h_ref / h_ema` after 64 eligible observations.

All three coefficients have no lower or upper projection. Each reference is
the controller's own warmup statistic. Training receives no gold support
count, desired mode count, desired entropy, or evaluation feedback.

The replay objective remains

`(15/16) * (1/16) * (mu * L_mass + alpha * L_balance)`.

The validator-positive novelty bonus remains `0.50`.

## Engineering smoke

First run executable Python factors for at most 128 optimizer updates from the
same pretrained revision and seed `9010` that exposed E56's cold-start failure.
Evaluation behavior is disabled and cannot select the mechanism.

The smoke passes only if:

- direct MaxEnt telemetry and controller state are absent;
- every all-zero, pre-discovery step has zero task reward, zero semantic
  augmentation, no replay group, and finite zero policy-gradient norm;
- a verifier-positive model output is discovered without a seeded exemplar;
- the first discovery activates singleton verified-mass replay;
- semantic and balance controllers advance only on their eligible events;
- all active losses, gradients, coefficients, and observations are finite;
- every coefficient reports no projection and no gold-support feedback;
- no traceback, worker death, CUDA OOM, or nonfinite event occurs.

If Python passes, run a 32-update graph-coloring smoke to prove multi-mode
semantic and balance activation under the same direct-MaxEnt-off isolation.
Smoke weights, optimizer state, banks, and any incidental evaluations are
discarded.

## Three-domain sentinel

Only a terminal machine-readable smoke approval may authorize one fresh
seed-9010 E57 treatment in graph coloring, Countdown easy3, and executable
Python factors. The model revision, prompt/evaluation pools, row order, group
size 16, learning rate `2e-7`, one PPO epoch, `beta=0`, max norm 1,
temperature 1, top-p 1, response limit 192, quarter-pass evaluation cadence,
and 50 complete prompt-pool passes remain matched to E53.

The frozen E53 seed-9010 ordinary Dr.GRPO arms are the controls. E56 is
diagnostic history and is not a control.

For each domain, over the final eight matched quarter-pass evaluations, E57
must:

- have higher mean distinct-correct@8 than matched Dr.GRPO;
- win distinct-correct@8 at least six of eight times;
- have higher mean `(distinct-correct@8 - pass@8)` than matched Dr.GRPO;
- win that excess-multiplicity quantity at least six of eight times;
- show positive excess multiplicity at least six of eight times;
- retain at least 75% of its own best rolling-eight distinct-correct mean;
- trail matched terminal pass@8 and mean@8 by no more than `0.03`.

The final 64 training records must also remain within the matched-control
response-length mean plus 32 tokens and no-EOS mean plus one response per
group. Any runtime, information-firewall, objective-isolation, controller,
checkpoint, or domain failure rejects E57.

## Conditional replication

Only a terminal three-domain approval bound to this protocol, source,
execution snapshot, launcher, auditor, manifests, and E53 identity may
authorize fresh treatment seeds 43, 44, and 45. Every treatment seed and the
three-seed mean must independently pass the same stability, multiplicity, and
task-quality gates.
