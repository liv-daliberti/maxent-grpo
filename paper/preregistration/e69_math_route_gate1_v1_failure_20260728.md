# E69 MATH route Gate 1 v1 failure record

Date recorded: 2026-07-28

The frozen `math-route-v1` base-model coverage gate completed on Slurm job
`30158805` (`COMPLETED`, exit `0:0`, node302, elapsed 00:02:13). The scientific
gate failed:

- task-correct samples: 174 / 1,024;
- trace-parse/execute samples: 0 / 1,024;
- accepted route samples: 0 / 1,024;
- coverage among task-correct samples: 0 / 174;
- multi-route prompts: 0;
- cross-prompt recurring signatures: 0;
- deterministic manual-review queue: 0 / 50.

This failure blocks Gate 2. It does not authorize inspecting MATH-500 scores.
The dominant observed failure was interface compliance: none of the 174
task-correct responses emitted a valid `math-route-v1` block. Any successor
trace language must receive a new version and a new frozen sample archive.

Artifact identities:

- `automatic_summary.json`:
  `498c7971e020c35c1ba5a2fb88a0436ceba1a0d53304e1c4d76313b9a5b19fc1`
- `sample_archive.jsonl`:
  `4dda526d8685b5fe32be66586a22b41fceec3c1f82e6392f5ae4805dd58b2f7d`
- `manual_review_queue.jsonl` (empty):
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Slurm stdout:
  `0c85a68c7aa7a6cde18e3325a86206209bbd773514e9aff956fefcfddd782351`
- Slurm stderr:
  `644596d559af60beafd9cf2052596f335c82c6f8c740168302238ab58ec43ac7`

The single logged symbolic-parser timeout failed closed and did not cause the
coverage failure.
