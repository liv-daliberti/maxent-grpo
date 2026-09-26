# Submission reorganization plan — 2026-09-14

Goal: a clean-cut story for `paper/main.pdf`, with **no figure removed from the
main text**, process language gone, the proofs tidied, and the appendix
reordered so a reviewer can find things. Appendix may grow.

Current state: 9-page body (at the hard cap), 8 main figures + 1 main table,
~3,350 words of body prose, ~90 pages of appendix in 16 sections.

---

## 0. Constraints that decide what is possible

Four fail-closed gates run in `make`. They were built to stop accidental drift,
so each one below must be updated *deliberately and in the same commit* as the
change it guards. None of them blocks this plan; they only decide the order of
operations.

| Gate | What it pins | Effect on this plan |
|---|---|---|
| `ops/check_paper_main_length.py` | Body ≤ **9 pages**; References must start by p.10; `fig:story` on p.1 | **Zero slack.** Every word added to the body must be funded by a word cut. See §3. |
| `ops/check_paper_current_contract.py` → `check_editorial_structure` | The 8 main figure stems **in order**, their section→section "evidence roles", section label order, and the hosted table's role | New `\section`s and any figure re-homing require editing the `ordered` and `roles` tuples. |
| `ops/check_paper_current_contract.py` → `check_formal_preservation` | The **18 formal blocks** must match `audits/narrative_reorganization_20260911/before/paper/main.tex` exactly (normalized whitespace) | See the key finding below. |
| `ops/check_paper_line_fill.py` | No prose paragraph may end on < half a line | Every reflowed paragraph gets re-audited. Run the PDF audit after each editing pass, not once at the end. |

**Key finding about the proof gate.** `formal_blocks()` builds a `Counter` of
normalized block text, and the check is `Counter == Counter`. That is
**order-insensitive**. So:

- **Moving** theorems/lemmas/proofs between sections — including splitting the
  theory appendix in two — passes the gate untouched.
- **Rewriting** any theorem statement or proof body fails it.

The proof cleanup in §5 deliberately rewrites statements, so it must re-pin
`PROOF_REFERENCE` / `PROOF_REFERENCE_SHA256` and record why. Do the *relocation*
work first (free), then the *rewriting* work (costs one re-pin).

**Step 0 before anything else:** `paper/main.tex` has 2,747 insertions / 1,853
deletions uncommitted against HEAD. Commit or tag that state so the
reorganization is a reviewable diff rather than a merge of two rewrites.

---

## 1. Diagnosis: why the story does not flow

**A. The paper's own headline claim has no home.** The title promises "Mode
Collapse in RLVR". The evidence for it currently appears in four fragments:
Figure 1 (an illustration), one intro sentence that forward-references an
appendix, two *unnumbered* paragraphs at the head of §5 before Experiment 1,
and `concentration_story.pdf` in the appendix. The reader meets the central
finding as a preamble to somebody else's experiment. It is the only result in
the paper without a heading.

**B. §5.3 carries four unrelated results.** "Experiment 3: replay within a
second task construction" is 343 words covering (i) Level-2 replay, (ii) the
prompt-hint ablation, (iii) sampling budgets and coarser keys, (iv) Pantry
outage adaptation. Only (i) is about task construction. Items (ii)–(iv) are all
asking *what does a verified mode actually represent* — a genuinely interesting
question that is currently hidden inside a heading that denies it exists.

**C. The hosted section is placed as a fourth experiment but disclaims being
one.** §5.4 sits in the Results sequence, implying parity with the trained
comparisons, then spends its own prose explaining that it identifies no
training cause. Its role needs to be stated in the heading, not retracted in
the body.

**D. §3.3 is an abstract of the appendix, not an argument.** Three display
equations in 242 words. `eq:main-bank-decomposition` explains why replay is
uniformly weighted — it belongs beside the loss in §3.2, where the reader is
asking that question, not one subsection later.

**E. §4 Experimental Design restates what §5 and the appendix run contract both
say again.** 240 words, of which roughly a third are appendix pointers and seed
bookkeeping.

**F. Conclusion carries three labels** (`sec:conclusion`, `sec:discussion`,
`sec:limitations`) and 115 words. There is no real limitations section, which
ICLR reviewers will look for.

**G. §2.2 (Levels) mixes benchmark design with in-flight dataset status** —
admission tolerances, "Levels 4 and 5 remain under calibration", "the separate
Level-3 release does not imply completed Level-3 training results".

---

## 2. Target main-text structure

**Figure order is unchanged — all eight stay, in their current sequence.** This
is deliberate: it preserves every figure's number, keeps ~100 pages of
cross-references valid, and keeps the contract's ordering check satisfied. The
reorganization is done with *section boundaries and headings*, which is where
the actual problem is.

```
Abstract + Figure 1 (modecollapse_story)          [unchanged position]

1  Introduction
2  ModeBench: measuring successful alternatives     Figure 2
   2.1  Success and sampled breadth
   2.2  One benchmark family, several difficulties  Figure 3
3  ReplayMaxRL                                      Figure 4
   3.1  Learning from fresh rollouts
   3.2  Replaying distinct verified successes        <- bank decomposition moves HERE
   3.3  Why memory is a different lever              <- trimmed to one equation
4  Experimental design                               <- condensed ~240 -> ~150 words
5  Results
   5.1  Verifier-only training concentrates verified outputs   *** NEW HEADING ***
   5.2  Replay improves success and sampled breadth   Figure 5
   5.3  Replay complements a stronger fresh objective Figure 6
   5.4  Replay transfers to a harder construction     Figure 7   <- narrowed to Level 2
   5.5  What the breadth measure represents           *** NEW HEADING, no figure ***
   5.6  Concentration in fixed deployments            Figure 8 + hosted table
6  Related work
7  Conclusion and limitations
```

### The three structural moves

**Move 1 — promote the collapse finding to §5.1.** Take the two unnumbered
paragraphs now at the head of §5 ("Training can replace varied attempts with
one repeated answer" and "Concentration among *correct* outputs…") and give
them a numbered heading as the first result. No new figure, no new words — the
text already exists and is already the right length. This single change is the
largest flow improvement available, because it makes the results sequence read
as *phenomenon → intervention → transfer → interpretation → deployment*
instead of *preamble → intervention → grab-bag*.

Then drop the intro's forward reference ("Longitudinal saved-output comparisons
find increasing correct-key concentration in Graph and PantryPlan under Dr.GRPO
and GRPO at all three scales (Appendix…)") — with §5.1 existing, the intro can
promise the result instead of proving it in parentheses.

**Move 2 — split §5.3 into §5.4 and §5.5.** §5.4 keeps only the Level-2 replay
result and Figure 7. §5.5 takes the hint removal, coarser keys, sampling budget,
and Pantry adaptation paragraphs, under a heading that says what they are for:
these four interventions are the paper's answer to "is `distinct@8` measuring
anything real?" Right now that question is answered well and advertised nowhere.

**Move 3 — rename §5.4 → §5.6 "Concentration in fixed deployments."** Same
content, same figure, same table. The heading now states the scope, so the
three sentences currently spent retracting a causal reading can go
(one sentence in the section, full scope in `app:claim-basis`).

### Contract edits this requires

In `check_paper_current_contract.py`:

- `ordered` tuple: add `sec:results-collapse` after `sec:results`, add
  `sec:results-meaning` between `sec:results-levels` and
  `sec:hosted-concentration`.
- `roles` tuple: the `sec:results-levels` role boundary changes from
  `sec:hosted-concentration` to `sec:results-meaning`.
- `MAIN_FIGURES` / `MAIN_LABELS`: **unchanged.**

In `check_paper_main_length.py`: `MAIN_SECTION_LABELS` is a subset check, so
adding subsections needs no edit. `max_pages` stays 9.

---

## 3. Page budget — there is no slack, so here is the funding

The body is at exactly 9 pages. Two new `\subsection` headings cost ~0.05 page.
Fund it from these, in priority order:

| Source | Saving | Note |
|---|---|---|
| §4 condensed 240 → 150 words | ~0.09 pg | Cut seed lists and appendix pointers; both live in `tab:run-contract`. |
| §3.3: drop one display equation (`eq:main-bank-decomposition` moves to §3.2 inline or to appendix), 242 → ~150 words | ~0.12 pg | Keep `eq:main-collapse-flow`; it is the one the reader needs. |
| §5.6: remove the three retraction sentences | ~0.04 pg | Replaced by the heading + one scope sentence. |
| Process-language removal across body (§4) | ~0.05 pg | ~15 clauses. |
| §2.2: move calibration/status prose to appendix | ~0.04 pg | See §4 below. |

Total ~0.34 page recovered against ~0.05 page spent. The surplus should go into
§5.5 (which is currently four cramped paragraphs) and the Conclusion (§6 below),
not into new claims.

**Check after every pass:** `make` runs the line-fill audit against the rendered
PDF, so reflowed paragraphs are the most likely source of a late failure.
Re-render between passes rather than batching.

---

## 4. Process language to remove

The paper repeatedly describes *its own production process* — audits, censuses,
registrations, integrity amendments, build failures, work in flight. A reviewer
reads this as either defensiveness or an internal document. The scientific
content it protects (exact `n`, admitted seeds, unfavorable results) must all
stay; only the machinery language goes.

### Body

| Location | Current | Fix |
|---|---|---|
| `tab:core-terminal-endpoints` caption | "All source-admissible terminal pairs are included… No missing endpoint is imputed." | "Falcon Countdown has `n=4`; other blocks `n=5`." The exclusion reason belongs in `app:source-integrity`, which it already is. |
| §2.2 | "Development checks confirm lower frozen Qwen2.5-0.5B success in every domain." | State the property of the benchmark, not the check that established it. |
| §2.2 | "These construction checks precede the replay comparison in Experiment 3." | Delete — ordering of the authors' work is not a result. |
| §4 | "except where marked in the figures" | Delete; the figures mark it. |
| §4 | "Appendix Table~\ref{tab:experiment-map} gives their comparison scope" | Fold into the sentence naming the controls. |
| §4 | "These exploratory inference-only comparisons characterize deployment behavior; they do not apply replay or identify training or model-size effects." | One clause: "These are inference-only." Scope table carries the rest. |
| §5 head | "This is observed output collapse under the tested sparse-reward recipe" | "under the tested recipe" — "sparse-reward" is already established. |
| Conclusion | "A fixed-bank audit finds…" | "A fixed-bank study finds…" |

### Appendix

Delete or rewrite, in order of how badly they read to an outside reader:

1. **Internal experiment IDs in prose.** `E118`, `E119`, `E120`, `E120-R1`,
   `E80-R1` appear 9 times in running text ("All registered E118, E119, and E120
   blocks are complete"). Keep them in the HuggingFace link text and the
   artifact paths, where they are genuine identifiers; remove them from
   sentences, which should name the comparison instead.
2. **"the September 12 frozen census"**, "Terminal populations use the …
   census", "retains its original frozen population". Replace with the fact:
   what seeds, what date the results were fixed, stated once in
   `app:source-integrity`.
3. **"The integrity amendment excludes that source"** → "That source is excluded
   at every checkpoint" — the exclusion is the fact; the amendment is process.
4. **Work-in-flight prose in `app:data-levels`:** "Levels 4 and 5 remain under
   calibration. Their development tiers are candidate recipes, not additional
   benchmark levels."; "A separate dataset revision is being calibrated under
   this neutral interface…"; "the revised neutral dataset has no admitted result
   yet." Cut all of it. A paper does not report experiments that do not exist.
   One sentence: "Levels 1–3 are released; the training results here use Levels
   1 and 2."
5. **"Every observed treatment endpoint in this snapshot passes the persisted
   replay-weight telemetry audit"** (appears twice, in `tab:current-e120`'s
   caption and its body). Delete both; it asserts that the authors' logging
   worked.
6. **Reproducibility Statement:** "missing final results and aggregate drift
   cause the build to fail" describes the authors' Makefile. Replace with what
   a reader can do: the dataset, the model archive, the figure manifest.
7. **`\section*{Appendix Organization}`:** the last three sentences ("All
   original proof statements and proofs are retained. The main keeps its
   original figure sequence; additional concentration, factorial, weighting, and
   within-level effect plots support the corresponding appendix analyses") are a
   changelog addressed to a previous version of the paper. Rewrite the whole
   block as a reading guide to the new part structure (§6).

### Hedging

"inconclusive" ×11, "does not identify/establish/isolate" ~20×, "exploratory"
×6, "descriptive" ×8, "nominal" ×13. The individual statements are correct and
worth keeping — this is the paper being honest, and it should stay honest. The
problem is placement: they are inline, so every claim is immediately followed by
its own retraction and the prose never builds.

Rule to apply: **at most one scope clause per paragraph in the body**; the rest
moves to the end of its subsection or to `tab:claim-basis`, which already exists
for exactly this purpose and is the paper's best asset. In the appendix, collect
per-result caveats into a closing "Scope" paragraph per subsection rather than
after each sentence.

---

## 5. Proof cleanup

Six concrete problems. Items 1–2 are free (relocation only). Items 3–6 rewrite
block text and need the single `PROOF_REFERENCE` re-pin.

**1. Split the theory appendix (free).** It is ~28 pages sitting between the
metric definitions and the benchmark protocol, so a reviewer checking an
experimental detail pages through all of it. Split into:

- **"Idealized analysis"** (~8 pp) — the chain the main text actually cites:
  `lem:grpo-mean`, `lem:maxrl-mean`, `thm:grpo-collapse`,
  `lem:replay-gradient-availability`, `thm:replay-retention`,
  `cor:replay-no-collapse`, `lem:shared-exemplar-retention`.
- **"Extended scope of the idealized analysis"** (~20 pp, placed late) —
  `lem:sampled-score-bound`, the geometry-dependence discussion, the entropy
  comparison, the discrete-descent extension, the partial-bank limit.

Because the gate counts blocks rather than ordering them, this passes as-is.

**2. Collect the scope prose (free).** Almost every proof is followed by one to
three paragraphs of "this does not certify…". Move them into the closing scope
subsection, which already exists (`app:theory-entropy`). The mathematics then
reads as mathematics.

**3. Get commentary out of theorem bodies (costs the re-pin).** Two blocks
currently carry editorial prose inside the environment:

- `thm:grpo-collapse` ends with "This is an asymptotic statement within the
  specified update geometry, not a claim of finite-time support loss or a
  convergence guarantee for the stochastic neural optimizers used in our
  experiments."
- `lem:replay-gradient-availability` contains "This gives no lower bound on
  gradient magnitude and says nothing about other loss terms, clipping, or an
  optimizer step carrying momentum across successive updates."

Both sentences are worth keeping. Move them immediately below their block as a
`\begin{remark}` (the environment is already declared and unused) so the
statement is a statement.

**4. Split `lem:replay-gradient-availability` (costs the re-pin).** It asserts
three separable things: a bound on mixed-group probability, the replay ascent
formula `v = ρ(w̄ − p)`, and an asymptotic norm as `p → e_b`. As one lemma the
reader cannot tell which part the main text is invoking. Split into a lemma
(fresh-group starvation) and a lemma (replay ascent and its limit).

**5. Name the assumptions once (costs the re-pin) — biggest readability win.**
The same model is currently restated in different words in each result: "Under
the independent-logit mean-flow assumptions above", "Under the same categorical
and common-length assumptions", "In the same independent-logit model", "Under
the explicit categorical assumptions below". Declare one block:

```
\begin{assumption}[Categorical mean-flow model]
(A1) one independently optimized logit per category ...
(A2) common length normalization ...
(A3) Euclidean infinitesimal on-policy updates, no preconditioning/momentum ...
(A4) zero explicit reference KL ...
(A5) finite initial logits, 0 < P(0) < 1, G >= 2 ...
\end{assumption}
```

Then each result reads "Under (A1)–(A5)". This also makes the honest scope
argument *stronger*, because the reader can see exactly which assumption fails
for AdamW/PPO instead of parsing it out of prose each time.

**6. Fix the notation collisions (costs the re-pin) and add a notation table.**
`C` currently means four different things:

| Symbol | Meaning | Where |
|---|---|---|
| `C(q)` | conditional collision `Σ q_c²` | §2.1, `eq:main-collision` |
| `C` | bank size `|B_x|` | `app:replay-metric-alignment` |
| `C` | clip constant = 5 | `eq:legacy-semantic-advantage` |
| `C_T`, `C_{T,w}` | energy constant | `thm:replay-retention` |

and `\mathcal C_x` is the valid-key set. Rename bank size to `k` (already used
that way in `app:theory-replay` — so this also removes an inconsistency), and
the clip constant to `\kappa`.

Also unify the bank decomposition, which appears three times in three notations:
`eq:main-bank-decomposition` (`P_B`, `q_B`), `eq:replay-bank-decomposition`
(`q_B` for *mass*, `C` for size), `eq:replay-mass-uniformity-decomposition`
(`P`, `q`). State it once, reference it twice.

---

## 6. Appendix reorganization

Current order interleaves setup, theory, results, and hosted material, and
splits the hosted evidence across four separate sections (`app:hosted-
concentration`, `app:hosted-comparison`, parts of `app:inference-followups`,
and the reasoning control). Reorganize into six named parts with
`\part`-style dividers or clear section grouping:

```
PART I   — Reading guide and scope
  A  Claims and their evidential basis          [keep first; best asset in the supplement]
  B  Metric and objective details
  B' Notation                                    *** NEW ***

PART II  — Benchmark and method
  C  Benchmark and evaluation protocol           (was D)
  D  Exact domain prompts                        (was E)
  E  Replay algorithm                            (was F)

PART III — Training results
  F  Comparisons across scales and levels        (was G)
  G  Supporting comparators                      (was G.6 + K merged: Semantic-MaxEnt
                                                  factorials rejoin the other comparators)
  H  Collapse and replay mechanism checks        (was H)

PART IV  — What breadth represents               *** NEW GROUPING, mirrors main 5.5 ***
  I  Matched prompt-hint ablation                (was M)
  J  Sampling-budget ablation                    (was N)
  K  Coarser keys                                (from L.3)
  L  Adaptation under changed requirements       (from L.1)

PART V   — Fixed deployments                     *** NEW GROUPING, mirrors main 5.6 ***
  M  Hosted protocol and concentration           (was I)
  N  Comparison across deployments               (was J)
  O  Reasoning and temperature controls          (was parts of I/J)
  P  Complementarity across models               (from L.2)

PART VI  — Idealized analysis
  Q  Idealized analysis of collapse and replay   (was C, split per §5.1)
  R  Extended scope                              (was C, split per §5.1)

PART VII — Records
  S  Source integrity and reproducibility
  T  Disclosures
```

Two principles behind this: (i) **Parts IV and V mirror main §5.5 and §5.6
one-to-one**, so a reader who wants the detail behind a results subsection lands
in one contiguous place instead of three; (ii) **theory moves late** because
nothing empirical depends on reading it and it is 28 pages of skipping for
anyone checking a number. The claim-basis table in Part I routes readers who
want it early.

### What to add to the appendix

1. **Notation table** (B'), fixing §5.6's collisions.
2. **Assumption block** (A1)–(A5) in Part VI, per §5.5.
3. **Consolidated "Scope of the idealized analysis"** subsection collecting the
   per-proof disclaimers pulled out in §5.2.
4. **Expanded limitations** — the material the Conclusion cannot hold at 115
   words: bank capacity vs. Python's mean 229 valid modes, no matched retention
   arm, `distinct@K` not measuring utility, hosted results not identifying a
   training cause, the Level-2 population confound.
5. **Everything displaced from the body** by §3–§4: §2.2's calibration prose,
   §4's seed lists, §5.6's scope sentences.
6. **The reading guide** replacing `Appendix Organization` — one short paragraph
   per part, no editing history.

Nothing needs to be deleted from the appendix for this plan. The unused figures
in `figures/` are historical monitoring plots (`*_live`, `e21`–`e56`,
`compute_divergence_*`) and are not paper-ready; leave them where they are.

---

## 7. Other submission items

- **Conclusion** (§7) currently carries `sec:conclusion`, `sec:discussion`,
  `sec:limitations` on one 115-word section. Retitle "Conclusion and
  limitations", keep the three labels for cross-reference stability, spend the
  surplus from §3 on two real limitation sentences, and point to the expanded
  appendix limitations.
- **Abstract** is a catalogue of sections ("Harder constructions test replay
  beyond the original tasks; prompt and sampling interventions probe which
  alternatives remain accessible"). Rewrite to follow the new results order:
  phenomenon → benchmark → method → result → what breadth means → deployments.
  Keep the 3B numbers; they are the strongest concrete claim in it.
- **Figure 1 caption** already does the right job. Leave it.
- **Dead preamble macros.** `\xmode` ("legacy adaptive MaxEnt+ReplayDr.GRPO"),
  `\historicalt` / `\Historicalt` ("historical multi-component treatment") and
  `\domlabel` each appear exactly once in `main.tex` — the definition, with no
  use anywhere including `results/*.tex`. Delete all four. "Legacy" and
  "historical" name an internal lineage the reader cannot see, and leaving the
  macros invites their reintroduction.
- The `remark` and `definition` environments are declared in the preamble and
  used zero times; `remark` gets used by proof-cleanup item 3.

---

## 8. Execution order

Each step ends with a `make` and a rendered-PDF check. Steps 1–4 are free of
the proof gate; step 5 is the one that re-pins it.

1. **Checkpoint.** Commit the uncommitted `main.tex` state. Tag it.
2. **Structure** (§2). Add the two headings, split §5.3, rename §5.4→§5.6.
   Update `ordered` and `roles` in `check_paper_current_contract.py`. Rebuild;
   confirm still 9 pages. *This alone delivers most of the flow improvement.*
3. **Budget** (§3). Condense §4 and §3.3, move the bank decomposition to §3.2.
   Rebuild; confirm page count and line-fill.
4. **Process language** (§4), body then appendix. Rebuild.
5. **Proofs** (§5). Relocate first (items 1–2, free), rebuild and confirm the
   formal-block gate still passes — this validates the "Counter is
   order-insensitive" reading before anything is rewritten. Then do items 3–6
   and re-pin `PROOF_REFERENCE_SHA256`, recording the reason in the commit.
6. **Appendix reorganization** (§6). Largest mechanical diff; do it after the
   text is settled so cross-references move once.
7. **Abstract, conclusion, reading guide** (§7). Last, so they describe the
   paper that now exists.
8. **Final pass.** Full `make`, all four gates, and a read-through of the body
   for the one thing no gate can check: whether §5 now reads as a sequence.
