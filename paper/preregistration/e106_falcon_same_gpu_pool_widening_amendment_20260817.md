# E106 Falcon same-GPU pool-widening amendment

Frozen before application on 2026-08-17 EDT. This amendment uses scheduler
state only. No E104 or E106 post-update evaluation outcome was inspected.
PointMaze is excluded.

## Motivation

The three untouched Falcon E104 mechanism cells remain pending with zero
runtime because their node constraints are narrower than already demonstrated
same-GPU capacity:

- Graph and MathIR request A5000 on node202 only. The matched Falcon E104
  Countdown cell completed on node203 in 00:20:06 with the same model,
  source snapshot, optimizer geometry, memory request, and A5000 GPU type.
- Pantry requests A6000 on node206 only. Completed Falcon jobs demonstrate
  the existing node205--207 A6000 pool, and the pending Falcon Python cell
  already uses that same pool.

## Authorized change

Only the pending Slurm node list may change:

- jobs 30637790 and 30637793: node202 to node[202-204], retaining A5000;
- job 30637794: node206 to node[205-207], retaining A6000.

Partition cs, account allcs, one-hour limit, CPU, memory, GPU type,
scientific environment, source snapshot, seeds, outputs, and every training
and evaluation setting remain unchanged. Each job must still be PENDING with
RunTime=00:00:00 at application. The operation is transactional: all changed
node lists are restored if any post-update validation fails.

## Gate treatment

The application record is mandatory provenance for the combined E104+E106
mechanism gate and the downstream E105 release. It cannot relax any mechanism
criterion or authorize inspecting evaluation outcomes.
