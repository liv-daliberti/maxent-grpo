# Story-first deck: Verified Mode Support in RLVR

The paper contains many experiment families, but the talk should carry one
causal thread:

> Binary correctness can improve while a policy's verified solution support
> collapses. ModeBench makes that hidden state observable with execution-defined
> keys. ReplayDr.GRPO retains modes after discovery; verified-support
> Semantic-MaxEnt is a separate exploratory extension for discovery.

The evidence has two deliberately different roles. The 74-admissible-pair
retention comparison is the headline. Falcon Countdown excludes one conflicting seed and has no five-seed interval.
The 49-endpoint discovery comparison is
an exploratory bundled contrast and is never allowed to blur that core claim.

## Core talk — stop after slide 10

1. **When eight samples become eight copies** — the failure and thesis.
2. **Accuracy can look perfect while support disappears** — one concrete,
   mechanically selected Graph Coloring pair.
3. **Binary correctness hides answer identity** — ModeBench and the difference
   between `pass@8` and `distinct@8`.
4. **Why GRPO concentrates correct modes** — sampled common modes get more
   updates; absent modes get none.
5. **One execution key repairs two different failures** — uniform replay for
   retention; verified proposals and bounded rarity credit for discovery.
6. **Two questions, two matched tests** — complete confirmatory retention versus
   the smaller exploratory discovery bundle.
7. **Retention is the strongest result: 74 / 74** — raw `distinct@8` improves in
   every admissible model–domain–seed pair, without cross-domain pooling.
8. **The honest split: extra support vs. accuracy rescue** — adjusted breadth
   identifies Graph, Countdown, and Pantry as the clearest independent-support
   effects; Python adjusted effects are unresolved and MathIR effects are small and scale-dependent.
9. **Discovery adds breadth beyond replay—unevenly** — positive mean
   adjusted breadth at both analyzed scales across 49 endpoints, with the
   bundled/exploratory boundary explicit.
10. **Three takeaways—and the boundary** — measure, retain, discover; then state
    what is not established.

## Backup only

11. **Only replay is uniform; alternatives are local** — complete smaller-model
    GRPO, UCPO, and RLEP-Dr adjusted-breadth contrasts.
12. **Where excess breadth appears, it persists through training** — Qwen
    trajectory-AUC evidence.
13. **The retention mechanism is small and measurable** — bank occupancy,
    capacity hits, and descriptive optimizer-time overhead.

14. **Replay also improves MaxRL** — refreshed September 8 factorial figure; eleven complete paired blocks, with larger-model partial domains explicit.
15. **Level 2 now has three complete domain blocks** — refreshed comparison and current terminal coverage; newly completed Python/MathIR pass@8 and distinct@8 intervals include zero.

## Deliberate exclusions from the core talk

- No legacy adaptive-semantic sampling-frontier plots.
- No DAPO efficacy claim: the available runs do not have the paper's
  standardized `pass@8` / breadth endpoint.
- No cross-domain pooled confidence interval or model-size trend.
- No claim that raw breadth automatically improves downstream utility.
- No claim that replay recovers modes it never observes.
- No component-isolated discovery claim: the 49-endpoint result is the bundled
  verified proposal–pressure–replay treatment, and Qwen2.5-3B is not analyzed.
- No manuscript edits. All deck code and generated assets remain under
  `presentation/`.

## Regeneration

From the repository root:

```bash
python presentation/build_story_deck.py
```

The script reads paper JSON/PNG assets without modifying them, creates
slide-native statistical figures under `presentation/figures/`, and writes the
editable deck to `presentation/verified_mode_support_story.pptx`.
