# E111 Graph preemption-cleanup traceback audit clarification

Date recorded: 2026-08-19, before the E111 terminal audit. No evaluation
endpoint was used for this clarification.

Slurm preempted original Qwen-3B Graph job `30674758`. During shutdown, the
DeepSpeed ignored Triton atexit callback and PyArrow Plasma context cleanup
each emitted a `FileNotFoundError` for a node-local `/tmp` directory that had
already been removed. These exceptions occur after SIGTERM/preemption cleanup;
they do not alter or invalidate a completed optimizer update.

The terminal auditor may remove the generic `Traceback` failure marker only
when every traceback in a stdout file terminates in one of these two exact
cleanup classes:

1. missing `/tmp/od2961`; or
2. missing `/tmp/test_plasma-*`.

If the number of tracebacks differs from the total number of those exact
terminal signatures, the generic failure marker remains fatal. This changes
only log classification, not training, checkpoints, treatment, or outcomes.
PointMaze remains excluded.
