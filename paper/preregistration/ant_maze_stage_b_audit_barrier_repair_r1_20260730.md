# AntMaze Stage-B audit barrier repair r1

Frozen on 2026-07-30 while all ten final AntMaze cells were still running and
before any terminal AntMaze receipt or audit.  The original audit dependency
used one colon-composed `afterany` clause.  PointMaze job 30203031 established
that this scheduler expression can release after a terminated subset: it ran
while six PointMaze cells remained active and correctly emitted a partial
failure.

The AntMaze training jobs, arms, seeds, snapshots, schedules, resources, and
outputs are unchanged.  The original pending audit job 30203042 is canceled
without running.  A fresh audit job uses the explicit conjunction
`afterany:J1,afterany:J2,...,afterany:J10`, so it cannot start until every
frozen AntMaze cell is terminal.  Identity and submission hashes are rebound
before any AntMaze training job reaches its terminal receipt.
