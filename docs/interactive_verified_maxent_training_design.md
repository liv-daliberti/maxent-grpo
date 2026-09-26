# Interactive verified-MaxEnt training design

Status: prospective implementation design. This document does not authorize a
training job or make any vNext result a paper result.

## Scientific object

Qwen2.5-0.5B is the policy. At every decision it receives the public problem
and current public state and selects one finite action label. A local mask
guarantees that the label is legal, but contains no solver answer, certified
support, route, endpoint identity, reward, or evaluation-only information.

The environment adapter applies the selected action and returns the next
public state:

- PantryPlan updates the partial allocation and public running totals.
- PointMaze applies the selected force for the frozen number of simulator
  steps.
- AntMaze, only after v6 admission, asks the admitted maze-blind controller to
  locomote toward the selected adjacent grid cell.

The controller is therefore an actuator, not the policy. It cannot choose the
next cell, route around an obstacle, access the language-model distribution,
or change the terminal reward.

## Rollout record

Each episode stores an ordered list of decision records:

1. tokenized public state;
2. ordered legal action labels and their token IDs;
3. selected label and environment action;
4. complete masked behavior-policy log probabilities; and
5. transition metadata needed to replay the public state.

The episode also stores the terminal binary verifier reward and, only for a
verified episode, its unchanged ModeBench canonical key. Injected environment
observations are context, never policy targets. Loss is evaluated only at the
model-selected action token in each decision record.

Actor and learner both renormalize logits over the same recorded legal support.
Any missing selected-token probability, support mismatch, nonfinite value, or
state-replay mismatch fails closed.

## Matched objectives

For prompt `i`, rollout `j`, and terminal reward `r_ij`, ordinary Dr.GRPO uses
the group-centered, non-variance-normalized task advantage

`A_task_ij = r_ij - mean_j(r_ij)`.

The verified-MaxEnt arm submits the same terminal rewards and verified
canonical keys to the existing online canonical bank. Its detached entropy
and novelty advantage is added after Dr.GRPO task-reward centering, matching
the E58 mechanism. Invalid or unsuccessful episodes have no canonical identity
and cannot update verified support.

Every decision in an episode receives the episode advantage. The clipped
policy loss is averaged across decisions within an episode before averaging
across episodes, so long trajectories do not receive extra statistical
weight. The two arms use identical clipping, optimizer, learning rate,
rollouts, action masks, horizons, update epochs, checkpoint schedule, and
terminal-verifier calls. MaxEnt identity bookkeeping is CPU-side; GPU forward,
backward, and optimizer work is matched.

No confirmatory online arm receives distance progress, nutrition residual,
planner agreement, controller reward, or other intermediate shaping.

## Capability staircase

1. Run the untouched 0.5B constrained-policy gate.
2. If the gate has usable verified mass, start both arms from that exact base
   checkpoint. PantryPlan has met this condition in development.
3. If the gate is zero, behavior-clone train-split-only oracle actions and give
   the identical warm-start checkpoint to both arms. The oracle, its traces,
   and its rewards are absent from online training and all development/eval
   prompts. PointMaze is expected to require this rung.
4. Run one paired-seed smoke. Require nonzero terminal reward, at least one
   prompt with two verified identities, finite loss/KL, and identical rollout
   and optimizer counts.
5. Only then launch the five paired seeds requested for the comparison.

AntMaze additionally requires: passing v6 open-plane admission, a newly frozen
fresh-maze executable gate, and a prospective cross-node determinism audit.
No Ant language-model sample precedes all three.

## Required implementation tests

- deterministic state-transition replay from every stored episode;
- selected actions always belong to their recorded legal supports;
- environment/context tokens receive exactly zero policy loss;
- episode loss is invariant to duplicating a decision record with zero weight;
- longer episodes are not upweighted;
- MaxEnt alpha zero is numerically identical to the matched Dr.GRPO update;
- permuting canonical-key names leaves the update unchanged;
- invalid and reward-zero episodes never enter verified support;
- both arms consume equal prompt, rollout, decision, forward/backward, and
  optimizer-step counts; and
- all evaluation prompts and sealed map seeds are absent from warm-start data.
