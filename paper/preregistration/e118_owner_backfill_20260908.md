# E118 owner-node capacity amendment — September 8, 2026

The user requested more existing E118 cells running on node302 and node105.
All 26 unfinished cells are Qwen-3B. This amendment changes resource allocation
only, without selecting on efficacy outcomes or adding scientific cells.

## Node302

Continue Graph ReplayMaxRL seed71, pending job31048110, from the validated
model and optimizer checkpoint at step1536. Request one A100,16CPUs and116GiB
host memory. The node has119.035GiB unallocated with three128GiB jobs running.
The running E118 Pantry continuation has84.119GiB noncache memory,40.284GiB
clean inactive filecache,85.723GiB summed process high-water RSS, and zero
memory.high or OOM events after completing its checkpoint restore. Five
historical E118 Graph/MathIR/Countdown observations have83.58–83.76GiB noncache.
These observations support116GiB with monitoring through fresh updates and
the next checkpoint save; they do not establish every future transient peak.
All runtime exports remain unchanged apart from explicit repository roots.

## Node105 controlled admission

Continue Graph MaxRL seed74, pending job31048115, from its validated step960
checkpoint. Request one A5000,16CPUs and128GiB host memory. Change only
`OAT_ZERO_VLLM_GPU_RATIO` from0.25 to0.40. The original A5000 attempt's exact
failure was `No available memory for the cache blocks` at
`var/artifacts/logs/e118q3-graph-m-s74-31048115.out:614`; it is not evidence
that the training model inherently requires more than24GiB.

The0.25 allowance supplies about5.89GiB on this GPU, while0.40 supplies9.42GiB.
The runtime passes this value to vLLM `gpu_memory_utilization`, which controls
the cache allocation. The sampling parameters are set separately. Existing
actor sleep/wake releases the actor allocation before learner updates.
Keep the original model/revision, source snapshot, seed, objective, optimizer,
training and evaluation batches, lengths, temperature, evaluation draws,
activation/Adam offloading, run directory and3072-step target unchanged.
Verify cache initialization, checkpoint restoration, actor sleep/wake and
fresh optimizer updates before admitting more E118 cells to node105.

## Execution and provenance

Both continuations use72-hour scheduler limits to avoid repeated short
allocation timeouts while retaining the same8-pass training target.
No running job is interrupted. Preserve the frozen launcher and verify its
hash. Hold the pending predecessor and replacement, validate the complete
export difference against the explicit allowance above, promote the source
and aggregate ledgers under their lock, retire the held predecessor, and
release its replacement only after checking for other writers and completion
receipts. Preserve source/aggregate backups and exact old/new job IDs under
`var/artifacts/e118_owner_backfill_20260908/`.

Implementation: `ops/exp_scaling/backfill_e118_owner_nodes_20260908.py`.
Runtime source evidence: installed `oat/interface.py:82`,
`vllm/worker/worker.py:233–246`, `oat/actors/base.py:56–70`, and
`oat/learners/base.py:751–777` in `var/seed_paper_eval/paper310/lib/python3.10/site-packages/`.

## Conditional additional MathIR pair

After Graph MaxRL seed74 on node105 has restored step960, advanced to fresh
optimizer updates, and recorded positive actor sleep/wake timings with zero
memory.high/OOM events, admit the existing never-started MathIR seed70 pair:
MaxRL job31048133 at128GiB and ReplayMaxRL job31048134 at116GiB, both with
one A5000,16CPUs and the same0.40 vLLM allocation. The116GiB admission also
requires the node302116GiB continuation to have restored and advanced with
noncache usage below100GiB and no memory.high/OOM events. Check fresh,
current scheduler and telemetry receipts at preparation and submission.

Both MathIR cells have no prior training metrics, valid checkpoints or
completion receipts. Their384-token model limit and64-token generation
limit are below Graph's512/192, with the same batch1 and offloading settings.
The MaxRL compute-matched branch executes the same replay tensor path before
zeroing its applied gradients, so Graph admission exercises that memory path.
Preserve every original scientific export. These are existing registered
cells starting for the first time, not checkpoint continuations.

With all four admissions, node302 requests500GiB across four GPUs and node105
requests500GiB across five GPUs (two existing Falcon and three E118 jobs).
The remaining host memory cannot admit another such job. Monitor the new
allocations through first updates and their first checkpoint saves.
