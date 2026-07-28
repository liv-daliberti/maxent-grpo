# E69 MATH route Gate 1 RPN-v2 failure record

Date recorded: 2026-07-28

The frozen RPN-v2 coverage rerun completed on Slurm job `30159041`
(`COMPLETED`, exit `0:0`, node302, elapsed 00:01:54). The automatic gate
failed:

- task-correct samples: 104 / 1,024;
- trace-parse/execute samples: 2 / 1,024;
- accepted route samples: 0 / 1,024;
- coverage among task-correct samples: 0 / 104;
- multi-route prompts: 0;
- cross-prompt recurring signatures: 0;
- deterministic manual-review queue: 0 / 50.

The archive contained many attempted route-like strings, but no task-correct
sample contained an opening route tag. This is evidence that asking the 0.5B
base policy to emit an inline route competes with task correctness and does not
provide a viable admission interface. Gate 2 remains blocked. The next
mechanism-design attempt may parse only bounded arithmetic equation chains
already present in natural task-correct derivations, provided every transition
is independently re-executed and no prose or unchecked literal defines route
identity.

Artifact identities:

- `automatic_summary.json`:
  `989684db732d2e0d189dc728af0762d3ab6d691728a76e348a5d689b90c7357c`
- `sample_archive.jsonl`:
  `130d7338f89369b5a6b675998ad8afc079e155f28ff9fc2a9cd30a9b11d28e13`
- `manual_review_queue.jsonl` (empty):
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Slurm stdout:
  `c108a5691f0bab08a8c63847fdf67727313ed3a6938dab1ab7caebda69c86417`
- Slurm stderr:
  `9bbd28e5e3281fc1112c8837cb8b79cefa510ffa16b11118327760bf59be600d`
