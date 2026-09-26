# E113-R4: pinned official-verl DAPO direct-recipe comparator

**Frozen:** 2026-08-20 12:47:46 EDT, after E113-R3 was canceled and after the
final SIF, import, model/tokenizer, data, reward, and Hydra-composition
compatibility gates, but before submission or observation of any E113-R4
generation or training outcome.

## Why R3 is retired

E113-R3 implemented DAPO-shaped terms inside the OAT one-prompt learner.  Its
ten-generation-batch loop retried one 16-response group until that single group
was nonconstant.  The published DAPO implementation instead generates a large
multi-prompt batch, filters every constant-reward prompt group, pools all
eligible groups in a buffer across generation batches, and updates only after
the buffer reaches the train prompt-batch size.  Those are materially different
sampling algorithms.  E113-R3 therefore remains an immutable diagnostic of the
custom adapter but is excluded from every named-DAPO efficacy analysis.

At the replacement request, the released R3 cohort comprised jobs
30790925--30790974: 42 were active (11 running and 31 pending) and eight were
already terminal.  All exact registered IDs were canceled; the external failure
reaper 30791185 then exited successfully.  No R3 job may be relaunched or
reclassified as E113-R4.


After R4 release, the 50 original E113 dependency-never-satisfied scientific
placeholders were also canceled without allocation or output; their exact
retirement record is `var/artifacts/e113_original_dependency_placeholders_retirement.json`.
## Authoritative implementation

E113-R4 executes the upstream `verl-recipe/dapo` implementation without source
patches:

- repository: `verl-project/verl`;
- commit: `4f80e465c2ec79ab9c3c30ec74b9745de61d0490` (the commit named by
  the official reproduction page);
- entry point: `recipe.dapo.src.main_dapo`;
- dynamic sampler: `recipe/dapo/src/dapo_ray_trainer.py`;
- container:
  `hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.3-flashinfer0.2.2-cxx11abi0`,
  converted once to a content-hashed Apptainer SIF;
  The source OCI manifest is
  `sha256:335ed6cd1fe73090e458409cfa4394d6abf4cd0503ca44dbafdc28ff72e5ed20`.
  Because the cluster disables user namespaces and ptrace, SIF packaging uses
  the official Apptainer `1.5.3` source release, built locally without setuid.
  Its upstream fix skips PRoot only for ownership preservation when ptrace is
  unavailable and invokes `mksquashfs -all-root` directly.  The upstream
  release-archive SHA-256 is
  `5a3bf360a5240086324aa7f7005ab7eeee91095e2091078b3f9783eaf6e7288a`;
- rollout/training stack: vLLM plus FSDP from that container;
- project-local reward-verifier dependency layer: the official image's existing
  SymPy 1.13.1 and ANTLR 4.9.3 plus `latex2sympy2-extended==1.11.0` wheel
  SHA-256 `aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234`
  and `math-verify==0.9.0` wheel SHA-256
  `3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194`.
  These pure-Python packages are prepended through a hash-checked project-local
  `PYTHONPATH`; they change neither the SIF nor upstream DAPO source.

The launcher must verify the upstream checkout commit and cleanliness, the
published entry-point hashes, the SIF and verifier-layer hashes, converted
parquet hashes, model revision paths, and an immutable runtime snapshot before releasing jobs.  A
checkout or artifact mismatch fails closed.  Local code is limited to the
ModeBench parquet conversion, exact task verifier adapter, scheduler wrapper,
and terminal receipt; it does not replace or patch DAPO's trainer.

## Final zero-science compatibility gate

The content-hashed SIF is 13,399,597,056 bytes with SHA-256
`1e978dd8f5b100d7d0214e56d694de23412f167fa417b503cc2c62d2a968969f`.
CPU-only Slurm preflights completed before this freeze:

- job 30800569 imported PyTorch 2.6.0+cu124, Ray 2.43.0, Transformers 4.51.1,
  vLLM 0.8.3, the pinned upstream DAPO entry point/trainer, and the two exact
  verifier packages;
- job 30800578 loaded both frozen model configs/tokenizers offline and read all
  five 384-row train plus five 128-row evaluation Parquet splits;
- job 30800589 obtained `(acc_correct, acc_wrong) = (1, 0)` through the actual
  reward adapter on Graph, Countdown, Python, MathIR, and Pantry; and
- job 30800607 composed the complete R4 Hydra override surface with exit code
  zero, including the upstream multi-prompt filter/buffer and 24-step horizon.

Diagnostic job 30800559 passed all imports but exited one only because its final
diagnostic requested installed-distribution metadata for the source checkout;
the corrected import gate is job 30800569.  None of these jobs generated a
response, performed an optimizer step, or entered the scientific denominator.

## Published recipe retained

The following settings reproduce the official script:

- GRPO advantage estimator with no critic and no KL reward or KL loss;
- 16 responses per prompt;
- dynamic filtering on raw binary `acc`, with constant-reward groups rejected;
- a buffer that pools eligible prompt groups across at most ten generation
  batches, as implemented upstream;
- temperature 1, top-p 1, top-k -1;
- asymmetric clipping `.20/.28`, dual-clip bound `10.0`;
- token-mean policy loss, one PPO epoch, entropy coefficient zero;
- learning rate `1e-6`, ten warmup steps, weight decay `.1`, gradient clip `1`;
- 32-prompt PPO minibatches;
- DAPO soft overlong shaping with the nearest positive integer to 20% of the
  frozen per-domain response length and penalty factor `1.0`; filtering still
  uses the unshaped binary accuracy returned separately by the verifier;
- remove-padding, dynamic token batches, gradient checkpointing, and parameter
  plus optimizer offload;
- sampled validation at temperature 1/top-p `.7`, checkpoint and validation
  frequency five, and automatic same-job checkpoint resume.

## Necessary scale and benchmark substitutions

The published run uses Qwen2.5-32B on DAPO-Math-17K with 512 train prompts,
1,536 generation prompts, 16 nodes by eight H800s, tensor parallelism four, and
Ulysses sequence parallelism eight.  Those model, corpus, and hardware choices
would not estimate the requested ModeBench two-family comparator.

E113-R4 changes only the following declared surfaces:

1. Models are the frozen Qwen2.5-0.5B-Instruct and Falcon3-1B-Instruct
   revisions used by the paired paper controls.
2. Data are the exact 384-row ModeBench train split and 128-row evaluation split
   for Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.  The
   converted chat messages must render byte-for-byte to each frozen E78/E79
   prompt under its own tokenizer, and no prompt may be truncated.
   Pantry retains the frozen six-bit support-mask policy action.  The reward
   adapter decodes the mask in the public ingredient-row order and invokes the
   existing deterministic trusted-environment transition to select the
   lexicographically first feasible quantity allocation on exactly that
   support, then calls the ordinary Pantry verifier.  It does not consult a
   reference support or allocation and does not modify DAPO's trainer.
3. The prompt batches are scaled uniformly by four: train batch 128 and
   generation batch 384.  This is the largest no-duplicate generation batch the
   384-row corpus permits.  It preserves the official 3:1 overgeneration ratio,
   group size 16, PPO minibatch 32, and ten-generation-batch cap.
4. Each job uses one 48 GB A6000, so tensor and sequence parallelism are one.
   This is a hardware scaling, not a sampler or objective rewrite.
5. Prompt/response limits remain the frozen paired-control limits.  Console
   logging replaces Weights & Biases; this changes no optimizer input.

Because the official learning rate and optimizer batching differ from E78/E79,
E113-R4 is a **method-recipe comparator**, not an isolated component ablation.
It may be compared to the frozen controls with that qualification.  R3 and R4
must never be pooled.

## Scientific matrix and horizon

- Families: Qwen seeds 43--47 and Falcon seeds 55--59.
- Domains: the five domains above.
- Full cells: `2 * 5 * 5 = 50`.
- One accepted upstream training step contains 128 nonconstant prompt groups
  and 2,048 accepted responses, split into four published-size PPO minibatches.
- Horizon: 24 accepted upstream steps = 3,072 accepted prompt groups = 49,152
  accepted responses per cell, matching the earlier accepted-prompt horizon.
- Worst-case query ceiling:
  `24 * 10 * 384 * 16 = 1,474,560` sampled responses per cell.  This is exactly
  30 sampled responses per accepted response, the official 3x generation ratio
  times the ten-batch cap.

Every result must report upstream `train/num_gen_batches`, accepted step count,
sampled-response ceiling status, family/domain/seed, source/container hashes,
and terminal state.

## Prospective operational gate and release

Before any scientific cell can start, two zero-cell Graph Coloring smokes run
the exact final batch geometry for one accepted upstream step: Qwen seed 43 and
Falcon seed 55.  They must load the pinned SIF and unmodified upstream trainer,
materialize the exact prompt surface, exercise vLLM generation, reward
adaptation, multi-prompt filtering/buffering, FSDP actor update, checkpointing,
and write a terminal receipt.  The 50 scientific jobs are submitted with an
`afterok` dependency on both smokes and are canceled automatically if that gate
cannot be satisfied.

An infrastructure or compatibility failure before an accepted optimizer step
may be repaired prospectively with a documented amendment and fresh zero-cell
smokes.  Once a scientific cell begins, no seed, model, data, batch ratio,
generation cap, reward, or optimization parameter may change.

## R4-P1 placement-only amendment — 2026-08-20 12:54:15 EDT

The initial held-audited submission used account `mltheory` with nominal
partition `all`.  The site submit plugin resolved that combination to partition
`mltheory`, whose only GPU node exposes A5000 rather than the frozen A6000 GRES,
so both zero-output smokes were pending `BadConstraints`.

All exact 52 R4 jobs were held before this amendment.  No job allocated, no
output directory existed, and no response or optimizer step was observed.  The
placement repair changes only the account to the project's established
`allcs` association and retains nominal partition `all`; the site resolves that
combination to the A6000-capable `cs` pool.  GPU type/count, CPU/memory/time,
image, snapshot, data, seeds, dependencies, and all scientific settings remain
unchanged.

The exact jobs may be amended in place.  If the scheduler refuses to release an
original smoke because it converted `BadConstraints` to an administrative hold,
only the two zero-output smokes may be replaced by fresh identical `allcs`
smokes, with all 50 science dependencies rewritten and audited before release.

## R4-P2 scheduling-only amendment — 2026-08-20 14:06:40 EDT

At this amendment all 52 authoritative jobs remained pending: the two smokes
for `Priority` and all 50 scientific jobs for their joint smoke dependency.  No
run directory existed for any job, no allocation had begun, and no outcome was
observed.

To improve ordinary scheduler priority and permit short-job backfill, all 52
jobs change from user-requested `Nice=100` to the default `Nice=0`.  The 50
scientific jobs retain their seven-day time limit.  Only the two operational
one-step smokes, whose frozen ceiling is ten generation batches or 61,440
sampled responses, reduce their scheduler time limit from seven days to one
day.  This changes the maximum allocation duration, not the one-accepted-step
smoke workload.

This is an operations-only amendment.  GPU type/count, CPU/memory, account,
partition, job IDs, dependencies, image, source snapshot, data, models, seeds,
batch geometry, generation cap, reward, optimizer, and scientific horizon are
unchanged.

The upstream ten-generation-batch exception is retained.  A cell that cannot
pool 128 eligible prompt groups from ten complete 384-prompt generation batches
is a terminal official-DAPO feasibility failure.  It is not requeued with a
higher cap, warm start, denser reward, easier prompt, or replacement seed.
Scheduler requeue of the same job and exact checkpoint after a node failure is
allowed.

Always display the denominator of 50, failed cells, and exact terminal `n` by
family and domain.  Endpoint comparisons use only terminal successful R4 cells
and their predeclared E78/E79 controls.  No complete-surface or pooled claim is
licensed if the required R4 cells are absent; feasibility failures remain a
separate reported result.

