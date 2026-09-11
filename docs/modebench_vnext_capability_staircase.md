# ModeBench vNext capability staircase

Status: prospective development design. No vNext result is a paper result until
its own frozen admission, capability, and matched-comparison gates pass. The
failed one-shot gates remain immutable negative results.

## What the failed probes measured

The repaired PantryPlan probe produced 4,096 completions: 4,050 violated the
required `ingredient_id=grams` grammar, 37 failed another candidate-parser
rule, and 9 contained a missing or nested/malformed box. No completion reached
joint nutrition-and-inventory validation. It measured grammar grounding, not
allocation search.

The repaired PointMaze probe produced 256 completions: 139 violated the program
length, 64 used an unknown action token, 4 had malformed boxes, and 49 were
syntactically executable. Networkless replay of those 49 found zero goal hits
and zero approaches within the 0.5 goal radius. Mean path length was only about
0.4--0.5 units on maps needing 4--6 units of net progress. PointMaze therefore
has both a text-interface failure and a one-shot planning/horizon failure.

AntMaze job 30184769 remains failed. An exact post-outcome diagnostic using its
hash-bound source and first frozen map nevertheless reproduced the upper
fixture 5/5 times through the official worker at distance 0.430 (threshold
0.5); a raw diagnostic passed the lower fixture at 0.449. This is now a
cross-node numerical/controller robustness question. It does not retroactively
pass 30184769 and does not authorize map substitution.

## Scientific object

The policy remains Qwen2.5-0.5B. At each decision it receives a textual
observation and chooses one member of a finite public action menu. A token mask
guarantees a legal action label. The mask removes formatting as a nuisance
variable but provides no certified support, route, endpoint answer, or reward.

The endpoint verifier and ModeBench canonicalizer remain unchanged. MaxEnt and
the compute-matched control must receive identical observations, masks,
transitions, warm-start checkpoint, and terminal rewards.

### PantryPlan

A rollout alternates between choosing an unused locally permitted ingredient
and choosing one public inventory-aligned quantity. `STOP` becomes available
after the minimum ingredient count. The observation reports the original
problem, partial allocation, and running totals computed from public
per-100g attributes. Reward and identity are computed only at `STOP` by the
existing exact verifier.

### PointMaze

The model chooses one of the nine frozen actions after each observation. The
environment applies it for five simulator steps and returns position, goal,
map, and remaining horizon. The model, not a planner, chooses the route.
Endpoint identity remains the ordered verified gate sequence. The admitted
implementation uses a persistent networkless hash-bound worker.

### AntMaze

AntMaze cannot enter LM sampling until a new prospective cross-node controller
audit uses the same 12 maps, both frozen programs and seeds, at least three
nodes from the intended training partition, raw trajectories, and a frozen
robustness margin. If v5 is not robust, develop a feedback-based v6 on training
maps only; keep development/evaluation map seeds sealed.

## Capability staircase

0. Environment oracle: certify at least two modes and sufficient throughput.
1. Constrained base policy: untouched 0.5B model, no demonstrations, existing
   development success/multimode thresholds.
2. Shared train-only warm start: behavior-clone solver/planner actions on the
   training split only; both comparison arms use the same checkpoint.
3. One-seed matched smoke: require nonzero reward, at least one multimode
   prompt, bounded KL, and matched rollout count.
4. Five-seed comparison: run only after the matched smoke passes.

Potential shaping belongs only in a shared warm start or identically in both
online arms with an ablation. Confirmatory metrics always use the binary
executable endpoint verifier.

The one-shot failures remain useful negative results. vNext asks the cleaner
question: with the same legitimate action interface, does explicit online
verified MaxEnt discover and retain more executable outcome modes than a
compute-matched control?
