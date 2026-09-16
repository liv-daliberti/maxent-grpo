# E110 amendment: two-hour Falcon backfill limit

Frozen on 2026-08-18 at 02:03 EDT while E110 job `30647379` remained
`PENDING` with runtime `00:00:00`. The decision uses scheduler state and
previously completed runtime/mechanism records only. No E110 output exists and
no E104/E106 post-update evaluation outcome was inspected. PointMaze is
excluded.

## Runtime evidence

E110's initial three-hour request received a 09:00 EDT estimate on partition
`all`, despite having no scientific dependency on that reservation boundary.
Two exact same-family A6000 records bound the required runtime:

- E106 Falcon/Python v6 job `30640330` completed 64 steps in `00:39:03` on
  node208 with the same model, 512-token generation surface, 16 samples,
  optimizer, v6/replay stack, 8 CPUs, and 64 GiB. Tripling that elapsed time is
  `01:57:09`; E110 uses four registered evaluation points across 192 steps,
  whereas tripling E106 would include nine.
- The matched full Falcon ReplayDr/Python seed-55 job `30269053` completed
  3,072 steps in `11:45:28` on node207 with the same model, prompt surface,
  group size, and A6000/CPU/memory request. Linear scaling to 192 steps is
  approximately 44 minutes.

The two-hour limit is therefore a conservative execution bound. E110 retains
joint checkpoints every 64 steps and the frozen watchdog/requeue path.

## Authorized scheduler-only change

While job `30647379` remains pending at zero runtime, hold it transactionally,
change only `TimeLimit=03:00:00` to `TimeLimit=02:00:00`, audit the unchanged
partition/account/node/GPU/CPU/memory request and entire scientific
environment, persist a content-addressed amendment artifact, and release it.
If any check fails before release, restore the three-hour limit.

This amendment changes no model, seed, data, prompt, parser, objective,
coefficient, sampling rule, optimizer, horizon, checkpoint cadence, output
path, gate, or analysis. E110 must still reach step 192 and satisfy every
registered mechanism criterion before E105 or E109 can be released.
