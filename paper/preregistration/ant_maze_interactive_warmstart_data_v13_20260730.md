# AntMaze v13 constrained-language warm-start data

**Status: FROZEN AFTER THE V12 FREE-FORM FAILURE AND BEFORE V13 MATERIALIZATION — 2026-07-30**

AntMaze v12 remains a terminal 0/256 free-form failure. V13 is a visibly
separate language-action interface: Qwen2.5-0.5B chooses exactly one of the
eight public compass tokens at each decision from restricted model logits.
The exact admitted v12 maps, v11 low-level controller, v12 grid-anchored
targeting, simulator, route identities, and split assignment are unchanged.

This stage loads only the four `train/train` rows and their two already
certified route programs per map. It replays those eight routes through a
persistent trusted simulator worker and records the public Markov prompt,
current position, planar velocity, goal, remaining horizon, and next compass
token at every decision. The prompt contains the public map and state, but no
canonical route key, verifier result, reward, answer, future action, planner
output, or certified route. Development and evaluation datasets are not
loaded; no language model is sampled or updated in this stage.

Materialization passes only if all eight exact routes replay to their frozen
canonical keys under the v12 controller/executor, every label belongs to
`N NE E SE S SW W NW`, every prompt uses the one-token public Markov
interface, and the output identity binds all train rows, examples, episodes,
protocol, source, controller, and dataset identities. A failure stops v13
before SFT or development sampling.
