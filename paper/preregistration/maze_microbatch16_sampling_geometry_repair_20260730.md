# Maze sampling-geometry scoring repair

Frozen on 2026-07-30 after the PointMaze v2 and AntMaze v13r1 paired audits
failed, and before any replacement-cohort model request.

Both failed pairs completed 64/64 verified episodes per arm with four
multimode updates. Their independent audits stopped Stage B solely because
the policy scorer used microbatch 4 while behavior sampling used aligned
groups of 16. The immutable unchanged-GRPO diagnostics reproduced every
recorded microbatch-4 discrepancy and found exactly zero maximum absolute
sampling/rescoring error for microbatch 16 on all four updates in both
domains. Point prompts were 178--193 tokens and Ant prompts were 227 tokens.

The prospective repair changes only the policy/replay scoring microbatch from
4 to 16, matching the sampler's 16-episode group geometry. PointMaze v3 uses
fresh development seed 75303 and the unchanged v2 rows [1, 3, 5, 7]. AntMaze
v13r2 uses the unchanged seed 76313 and rows [0, 1, 2, 3] to isolate this
operational change. Models, prompts, routes, workers, objectives, learning
rate, optimizer, update count, rollouts, replay schedule, thresholds, action
support, and final five seeds do not change.

The replacement identities must bind the failed audit and immutable
diagnostic for their domain. Their independent audit continues to require
behavior/live maximum absolute log-probability error at most 1e-4. A failed
replacement stops the corresponding final ten cells. No failed artifact is
overwritten or relabeled.
