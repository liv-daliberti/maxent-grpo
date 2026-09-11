# E29 matched free-form Graph/Countdown scaling at 7B on four GPUs

**Status: FROZEN USER-REQUESTED SCALE EXTENSION BEFORE E29 SUBMISSION OR
E29 OUTCOMES (2026-07-21).**

**Prospective horizon amendment (2026-07-22).** Following inspection of the
first partial E29 trajectories, the campaign training and reporting boundary
is extended from five to ten complete prompt-pool passes. This applies to new
or cleanly resumed E29 training; it does not reclassify already observed
points or repair an invalid actor-weight synchronization.

E29 scales the two matched unrestricted free-form forms from E25-v2/E28 to
Qwen2.5-7B-Instruct for graph coloring and Countdown easy3:

1. ordinary free-form Dr.GRPO; and
2. conditional-token MaxEnt with the base-preserving Haarnoja dual and the
   task-specific 125% entropy target.

Each task/arm uses seeds 43, 44, and 45. All scientific settings are inherited
from E25-v2/E28: group size 16, one PPO epoch, learning rate `2e-7`, `beta=0`,
temperature one, `top_p=1`, `qwen_boxed`, maximum response length 192, and ten
complete passes over the frozen 192-example graph or 384-example Countdown
pool. The MaxEnt arm uses `conditional_token_mean`, alpha initial/minimum
`0.000075`, maximum `0.00060`, log-alpha Adam learning rate `0.010`, graph
target `1.622718550885717`, and Countdown target `1.347109432487438`. The
Dr.GRPO arm has alpha zero and all entropy controllers disabled.

The model is Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Each job requests four compatible
GPUs on one node and 256 GiB host memory. Four learner ranks receive the same
prompt and the same single generated group of 16; group statistics are formed
from all 16 candidates, then a shared permutation assigns four disjoint
candidates to each learner rank. Thus the four-GPU implementation changes
hardware execution but does not change the one-prompt, group-16 optimizer
recipe. ZeRO-2, optimizer offload, activation offload, vLLM sleep, and a
four one-GPU actors are enabled. Each learner rank broadcasts its replicated
ZeRO-2 policy to its collocated actor concurrently; rank zero's actor generates
the single audited group used by all learners. Hardware type is not an
experimental invariant; the scheduler may place jobs on A100, A6000, A40, or
L40 nodes while retaining the four-GPU single-node topology.

The superseded v1 canaries 30041632 and 30041633 used one four-way
tensor-parallel actor and serialized every full-model update from learner rank
zero into all four actor workers. Both arms completed the exact-group backward
pass, but the first weight sync did not complete in the observed window. They
are excluded from E29 outcomes. V2 changed only this weight-copy transport.
Its canary 30041782 initialized all four local groups and reached the first
backward pass, but four level-1 vLLM sleeps copied four complete actor models
into host RAM. The allocation reached roughly 283 GB RSS and could not make
timely optimizer progress, so all v2 jobs are excluded. V3 retains the same
scientific update and local transport while using vLLM level-2 sleep: actor
weights are discarded during learning, empty weight storage is remapped after
the update, the current learner weights are broadcast, and only then is the KV
cache restored. This is an operational memory amendment, not a treatment
change.

The first v3 submission (jobs 30042287--30042298) remained held and is also
excluded: its fail-closed held-job audit detected that the generic submission
wrapper had not exported the new sleep-level argument. V4 adds that export;
no affected job started training.

The first v4 A6000 canary 30042400 is excluded after initialization reported
that ratio `0.25` left no vLLM KV-cache blocks on a 48 GB GPU. Its retry loop
was stopped before any rollout or optimizer update. The seven cells assigned
to 48 GB A6000/A40/L40 nodes are operationally resubmitted with vLLM GPU ratio
`0.40`; A100 jobs retain `0.25`. This changes cache capacity only, not model
weights, sampling, batches, objectives, or optimizer updates.

**V4 outcome invalidation and v5 repair (2026-07-22).** Released v4 graph jobs
were healthy at initialization and then produced invalid binary-only text after
their first level-2 sleep and actor-weight synchronization. The learner-side
Dr.GRPO update was an exact no-op (zero reward, policy loss, and gradient norm),
which isolated the corruption to actor restoration. Level-2 sleep had discarded
Qwen's non-persistent rotary cache along with parameter storage, while the
custom synchronization restored only named parameters. Every post-initial v4
point is excluded. V5 backs up all actor model buffers to CPU before discard,
restores and byte-verifies them after weight remapping, and fails closed on any
missing or changed buffer. V5 uses a fresh run namespace and immutable source
snapshot; no v4 metric is an eligible v5 outcome.
The final pending v4 job, `30042486`, was cancelled before allocation after the
source-snapshot audit confirmed that it still referenced the invalid v4 code.

Any v5 jobs are submitted held. The complete held cohort is audited before any
release. A single graph Dr.GRPO seed is the runtime canary; the remaining jobs
stay held until that canary completes one rollout, optimizer update, parameter
sync, scheduled evaluation, and checkpoint boundary without rank divergence,
staleness, buffer-restoration failure, or output-format corruption.
