# PointMaze K=16-aligned algorithm-repair pair v2

**Status: FROZEN AFTER V2 VIABILITY AND BEFORE PAIRED TRAINING — 2026-07-30**

The balanced v2 slate passed executable admission. Its frozen-model viability
sample produced 17 verified routes among 256 attempts, two multimode prompts,
and no successes in the preregistered first-eight prefixes. The eight-draw
gate therefore failed.

The online algorithm does not train on eight-draw groups: every update uses
16 rollouts for one prompt. In the already frozen viability ledger, verified
routes first occur at sample indices 9, 10, and 11 on three distinct prompts;
the actual K=16 training-sized prefix therefore contains verified signal on
three of four prompts. This v2 pair repairs the algorithm/gate mismatch. It
does not change or resample the model, maps, attempts, verifier, or action
interface.

Qualification requires:

- at least three prompts with a verified route among their first 16 frozen
  attempts;
- at least two multimode prompts among 64 attempts;
- aggregate verified rate in [4%, 25%];
- passing executable admission; and
- exact binding to viability job 30204702 and seed 76520.

The pair uses seed 76521, all eight train-only balanced maps, 12 complete
prompt passes, exactly 96 updates, and 16 rollouts per prompt. Arms are
compute-matched Dr.GRPO and verified-first online MaxEnt with semantic
coefficient 0.10, novelty 0.50, replay 0.10, the same optimizer and binary
official reward, and identical fixed-shape policy/replay work.

Development evaluation occurs at update zero and every two updates with four
K=8 draws. Request seeds are functions of arm seed, row, draw, sample, and
decision round but are invariant to checkpoint update, so checkpoint curves
use common random numbers.

The paired audit requires each arm's aggregate train verified rate in
[2%, 50%], at least 10 of 96 updates with nonzero task advantage, exact-zero
control exploration/replay derivatives, nonzero treatment
exploration/replay derivatives, matched fixed-shape traversal, all 49
evaluation coordinates, finite metrics, exact identities, and no resume.
Only a passing pair can authorize a separate five-seed repair final. Original
PointMaze results remain immutable and are reported separately.
