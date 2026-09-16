# E117 Stage-1 A8: untouched evaluation reserves

Frozen: 2026-08-25T12:14:49-04:00, while all twelve E117-R1 jobs were
pending, no E117-R1 optimizer update had occurred, and no Stage-1 job, sampled
response, or outcome existed.

Status: outcome-blind data reservation only. This amendment closes a missing
precondition in the effective Stage-1 contract. It does not authorize a
Stage-1 launch, change any advancement rule, or inspect any model outcome.
PointMaze remains excluded.

## Gap

The four registered sentinel datasets contain a training bank and the
evaluation bank already reused throughout the historical program. Calling
that evaluation bank "untouched" would therefore be false. A future Stage-1
launcher must not use it for development advancement, and a future
confirmation must not reuse the Stage-1 development prompts.

## Fixed reservation

Before any E117 or Stage-1 outcome, materialize exactly two new blocks of 128
prompts per sentinel:

- `development`: the only evaluation block permitted for the three-seed
  Stage-1 development screen;
- `confirmation`: a sealed block reserved for the later at-least-five-fresh-
  seed confirmation, if a component advances.

The deterministic generator seeds are fixed as follows.

| Sentinel | Development | Confirmation |
|---|---:|---:|
| `qwen05b/countdown` | 117100 | 117500 |
| `qwen05b/graph_coloring` | 117200 | 117600 |
| `qwen05b/python_factors` | 117300 | 117700 |
| `falcon1b/mathir` | 117400 | 117800 |

For each sentinel, both blocks use the frozen task generator and the exact
source distribution parameters. Development excludes every identity in every
historical train and evaluation split. Confirmation excludes every historical
identity and every development identity. Identity is task-semantic, not an
instance ID or row index:

- Countdown: `(sorted numbers, target)`;
- Graph: `(n, sorted edges, partial coloring)`;
- Python factors: the sorted tuple of test cases;
- MathIR: `(family, sorted bindings)`.

The materializer must fail closed on any overlap, duplicate identity, wrong
row count, unexpected source split, source-generator drift, or an existing
output root. It records the historical source trees, generator files,
canonical rows, and all disjointness checks by SHA-256. The output path is
`var/data/e117_evaluation_reserve_v1/{development,confirmation}/<domain>/eval`
and the frozen identity record is
`var/data/e117_evaluation_reserve_v1/identity.json`.

## Distribution contract

No prompt is hand-authored or selected by its contents. The materializer
reuses the exact recovered source parameters:

- Countdown: three distinct integers from 2--12, 2--8 exact modes;
- Graph: original prompt style, 3 hidden vertices, 4--24 completions, at most
  6 vertices and 8 edges;
- Python factors: four cases through 96 and at least 16 exact modes;
- MathIR: the four frozen families, balanced by construction.

The source-contract reconstruction for Countdown and Graph reproduced the
existing train and multi-answer evaluation rows exactly before this document
was frozen. Python and MathIR parameters are already carried by their source
identity manifests and generator defaults.

## Use boundary

The existing training banks remain the Stage-1 training data. Only the
evaluation prompts are replaced. A future launcher must bind the reserve
identity digest and use the `development` paths. Development analysis and
tuning must never read the confirmation rows. If a component advances, a
separate confirmation protocol must bind the already-reserved `confirmation`
paths and at least five fresh paired training seeds.

Reservation does not cure outcome-guided selection in the historical E112
comparison and does not upgrade Stage-1 from development to confirmation. The
E117 mechanism audit and activation-readiness gate remain mandatory launch
conditions.
