# E105 policy-gradient post-freeze revalidation

Recorded: 2026-08-25T12:04:00-04:00 during an outcome-blind repository
integrity audit. No E105, E109, E112, or E117 efficacy endpoint was inspected.

## Trigger

The immutable 2026-08-18 E105 policy-gradient evidence correctly pins a
three-test file with SHA-256
`49727b4a70fd5ff37e471bcac563726ee94f2c5cc4bcc5271c8188f8602bf289`.
The current root test has since expanded to five tests and SHA-256
`bb0eb3836b2d6c2b54d139592afcf97cf183e553ae82011bf5ba02f2ea9724d3`.
Consequently, the historical release validator now reports test-digest drift.
That is a correct historical-integrity signal, but it does not establish that
the expanded behavioral checks fail.

The pinned Python 3.10 environment also requires its bundled
`libpython3.10.so.1.0` and TensorFlow framework directory on the process-local
loader path in the current shell. Without that path, collection fails before
any test executes.

## Supplemental check

Run the current five-test file with:

- the original frozen E106 source snapshot on `PYTHONPATH`;
- only the pinned environment's own `lib` and TensorFlow directories on
  `LD_LIBRARY_PATH`; and
- the pinned `paper310` Python executable.

The complete suite passes: five tests, seven environment/deprecation warnings,
zero failures. It exercises the production semantic advantage, replay joint
update, microbatch geometry, sparse-singleton v7 path, and discovered-unsampled
mode replay path.

## Boundary

This is supplemental current-behavior evidence. It neither overwrites nor
replaces `var/artifacts/e105_policy_gradient_direction_tests.json`, does not
retroactively change the E105 release gate, and does not reinterpret any
outcome. Historical provenance should continue to report the old artifact and
its old test hash; current development validation may additionally cite this
post-freeze revalidation.
