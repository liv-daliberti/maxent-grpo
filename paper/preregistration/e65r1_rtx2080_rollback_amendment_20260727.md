# E65R1 RTX 2080 compatibility rollback

Frozen 2026-07-27 13:24:44 EDT, before requeueing or changing placement.

The five zero-runtime E65R1 jobs moved under the MLTheory RTX 2080 placement
amendment allocated together on node915:

- Graph seed 44: 30127479
- Graph seed 45: 30127481
- MathIR seed 43: 30127489
- MathIR seed 44: 30127490
- MathIR seed 45: 30127491

All five reached actor initialization and failed before an optimizer metric
was emitted. Each actor recorded the same explicit incompatibility: the
frozen Qwen/vLLM execution requests bfloat16, while the RTX 2080 Ti has CUDA
compute capability 7.5 and supports this path only in float16. Changing dtype
would change the registered numerical experiment and is forbidden.

At this boundary all five jobs had zero `train_metrics.jsonl` records, zero
checkpoint state, zero Slurm restarts, and no training or evaluation outcome.
They may be requeued directly into a user-held pending state, restored to the
original `nodelist=node105,node202,node203,node204` and
`gres=gpu:a5000:1`, audited while held, and released together. Job identity,
source, operations snapshot, protocol identity, dtype, model, data, seed,
optimizer, controller/actuator, and checkpoint settings remain unchanged.

The failed RTX 2080 signature is retained in this amendment, the campaign log,
and the Slurm restart count; an active Slurm output file may be replaced when
the same job identity is requeued. If the prior attempt remains in an appended
log, it is classified only as a zero-step hardware-compatibility interruption
when the log names compute capability 7.5 and bfloat16 incompatibility and a
later attempt continues on a compatible GPU. Any other rank traceback remains
a run failure.
