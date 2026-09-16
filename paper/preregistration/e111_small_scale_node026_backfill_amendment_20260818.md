# E111 small-scale node026 backfill amendment

Date frozen: 2026-08-18, before applying the change and without inspecting an
E111 endpoint result.

Exactly three nonterminal small-scale E111 jobs remain pending:

- `30674729`: Qwen-0.5B Graph Coloring
- `30674733`: Falcon-1B Graph Coloring
- `30674754`: Falcon-1B Countdown

Node026 is idle with ten RTX3090 GPUs and sufficient host memory. RTX3090 is
already present in these jobs' originally submitted allowed hardware pool, but
node026 was omitted from the node-list expression. Historical E111/E104
small-scale 64-step gates fit comfortably within 45 minutes.

For these three exact pending jobs only:

- `ReqNodeList`: `node[020-022,025,103-104,202-208,403]` -> `node026`
- `TimeLimit`: `08:00:00` -> `00:45:00`

GRES remains generic `gpu:1`; partition `lowprio`, account `mltheory`, CPU and
memory requests, job IDs, stored environments, run directories, checkpoints,
models, seeds, data, optimizer, MaxEnt/ReplayDr objectives, evaluation, and
target steps remain unchanged. No job is canceled, signaled, reset, or
duplicated. The change is scheduler-only and endpoint-blind. PointMaze remains
excluded.

The final E111 auditor must validate exact before/after scheduler records,
unchanged submit lines, ledger/protocol digests, node026 capacity evidence,
and only the frozen node/time transitions above.
