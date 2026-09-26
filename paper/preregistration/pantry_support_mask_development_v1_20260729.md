# PantryPlan fixed support-mask development gate v1

**Status: FROZEN BEFORE THE FIRST SUPPORT-MASK MODEL COMPLETION — 2026-07-29**

## Antecedent and purpose

The one-shot full-allocation v2/v3 probes remain failed. The prospectively
frozen interactive finite-action repair passed after an infrastructure-only
GPU placement retry (34/64 prefix-success prompts, 29/64 multimode prompts).
A separate support-first development run also passed (58/64 and 52/64), but
its intended prose protocol was accidentally empty; its executable identity
and independent replay audit preserve it only as a development signal.

This gate tests a new, immediately trainable serialization of that support
interface. It is development-only and is not a training result or paper result.

## Fixed policy and environment boundary

Every admitted prompt contains exactly six Pantry rows. The policy emits exactly
six one-token binary actions in printed row order. `1` includes the row and `0`
excludes it. The unmasked policy support is all 64 bit strings, including masks
with too few or too many selected ingredients. No feasible-support catalogue,
reference allocation, or solver answer is included in the prompt or action mask.

After the six actions, a deterministic trusted transition searches the public
quantity lattice on exactly the selected support and returns the lexicographically
first verifier-accepted allocation, if one exists. It reads only prompt-local
inventory, step sizes, and targets. Invalid support widths and infeasible supports
receive zero reward. Semantic identity is the selected ingredient support derived
by the existing PantryPlan endpoint verifier.

## Frozen sample

- model: immutable Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- data: all 64 rows of `pantry_plan_modebench_v2/dev`, `multi_answer`;
- no evaluation split is loaded;
- 64 rollouts per prompt, with the first 16 as the prefix;
- temperature 1.0, top-p 1.0, seed 76102;
- exactly six generated tokens, restricted only to the single-token alphabet
  `{0,1}`, with deterministic length termination.

## Frozen decision

Pass only if both hold:

- at least 32/64 prompts have one verified completion in the first 16 rollouts;
- at least 16/64 prompts have two verifier-distinct ingredient supports across
  all 64 rollouts.

On pass, the interface becomes eligible for a separately frozen one-seed
Dr.GRPO plumbing smoke. That smoke must establish actor/learner restricted-policy
parity and cannot authorize the five-seed comparison. On fail, this serialization
is ineligible and its thresholds may not be relaxed.
