# E106 Python smoke time-limit amendment (2026-08-17)

## Scope

This is a scheduler-only amendment for the two unreached E106 repaired-Python
mechanism cells:

- Falcon3-1B job `30640330`;
- Qwen2.5-3B job `30640331`.

Both jobs were `PENDING` with runtime `00:00:00` and no training metrics when
this amendment was written. The Qwen-0.5B E106 cell is complete and is not
changed. No E104 job is changed. PointMaze is excluded.

## Evidence available before the change

- The matched 64-step Qwen2.5-3B E104 Python cell `30637797` completed in
  `00:27:26` under a two-hour limit.
- All five matched Qwen2.5-3B E104 static-domain cells completed in at most
  `00:34:03` under two-hour limits.
- The full 3,072-step Falcon3-1B plain-GRPO Python run `30516430` completed in
  `12:27:29`; E106 requests only 64 optimizer steps.
- The matched 64-step Falcon3-1B E104 Countdown cell `30637791` completed in
  `00:20:06` under a two-hour limit.

These scheduler runtimes and optimizer-step counts are capacity evidence only.
No post-update E104 or E106 evaluation outcome was inspected.

## Frozen change

For jobs `30640330` and `30640331`, change only:

```text
TimeLimit: 08:00:00 -> 02:00:00
```

Do not change the job ID, model, seed, domain, data, prompt template, source or
ops snapshot, optimizer, objective, estimator flags, replay settings, GPU type,
memory, CPU count, output path, or outcome-blinding contract. A two-hour limit
retains a conservative margin over the observed matched smoke runtimes while
making the jobs eligible for substantially more scheduler backfill windows.

## Gate consequence

This amendment does not relax any E106 mechanism criterion. Both jobs must
still reach optimizer step 64 and pass admission, replay-gradient, v6 activity,
legacy-disablement, controller-disablement, boundedness, and centering checks.
E105 remains locked until the combined 15-cell gate is complete and passes.
