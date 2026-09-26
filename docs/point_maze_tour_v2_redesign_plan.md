# PointMaze Tour v2: redesigning the sixth ModeBench domain

Status: prospective design. No v2 cell is a paper result until its own frozen
admission, capability, collapse, and matched-comparison gates pass. The v1
waypoint cohort (E78pm/E79pm) is not deleted; it is retired into the paper as a
reported boundary result (§10).

This document supersedes `docs/point_maze_waypoint_v1.md` as the design of
record for the sixth domain. That file remains the accurate description of v1,
which v2 replaces.

## 1. Why v1 has to be replaced

ModeBench exists to measure correct-mode collapse under binary-reward RL.
PointMaze v1 does not exhibit correct-mode collapse, so it measures nothing
about retention.

Qwen2.5-0.5B `distinct@8`, pass 0 to pass 8, from
`paper/figures/figure4_interim_20260806.json`:

| Domain | control p0 | control p8 | x-Mode p8 |
|---|---|---|---|
| Graph coloring | 0.56 | 0.33 | 2.44 |
| Countdown | 0.11 | 0.49 | 1.64 |
| Python factors | 0.00 | 0.17 | 0.57 |
| MathIR | 0.22 | 0.51 | 0.83 |
| PantryPlan | 2.43 | **0.52** | 1.54 |
| PointMaze v1 | 1.37 | **1.39** | 1.41 |

Over 3,072 updates the v1 control moves `distinct@8` by `+0.016` and `pass@8`
by `+0.006`. Both arms wander inside a band and neither trends. Recomputed from
`var/artifacts/e78pm_point_maze_{control,replay}_s4{3..7}.metrics.jsonl`:

- mean paired difference at pass 8 is `+0.025`;
- the mean within-run standard deviation of `distinct@8` across the 17
  registered checkpoints is `0.048`, so the reported effect is roughly half of
  the domain's own checkpoint noise;
- seeds 45 and 46 give a paired difference of exactly `0.000`; the positive
  mean comes from seeds 43 (`+0.047`) and 47 (`+0.070`);
- the Falcon panel's `-0.016` is the same noise with the opposite sign, and it
  is the only negative cell in the paper.

The ten Qwen cells averaged 23.5 GPU-hours each and the observed Falcon cell
ran 42 hours (`sacct` jobs 30267879--30267889, 30276610), on the order of 650
GPU-hours across the two families, to produce a flat line.

## 2. Root cause: four structural faults

**F1. Mode identity rides on one decision out of roughly thirty.**
`_barrier_candidate` in `src/oat_drgrpo/point_maze_waypoint_data.py:109` builds
exactly one vertical wall at `size // 2` with three to five openings, start
fixed at column 1 and goal at column `size - 2`. The mode is which opening the
trajectory crosses. Every other waypoint is "move toward the goal" and is
shared by all modes.

The trainer divides the episode advantage uniformly across decisions
(`ops/train_point_maze_verified_replay_only.py:483`):

```python
slot["weight"] = float(task_advantages[episode].item()) / decision_counts[episode] / SAMPLES
```

With `max_actions = min(64, 4 * (size - 2))`, that is 28 to 44 decisions, so the
single mode-bearing decision receives about 1/30 of the pressure and the rest
of the gradient pushes on decisions every mode shares. GRPO's collapse
mechanism is competition between equally rewarded *emissions*; in v1 the mode
is an emergent property of a long shared prefix, so there is nothing to
compete.

**F2. The mode ceiling is three to five, and geometry picks the winner.**
Start row and goal row are fixed per map, so the nearest opening dominates.
Averaged over the evaluation split, 3.2 successful samples per prompt yield
1.43 distinct corridors, and that ratio never improves.

**F3. Reward is saturated and free.** `pass@8` is 0.775 at pass 0 and 0.781 at
pass 8. Dr.GRPO advantages are near zero in a group that is mostly correct, so
neither arm has a gradient to exploit.

**F4. MuJoCo is load-bearing for nothing.** Wall masking makes illegal moves
impossible and a pinned PD adapter does the control, so physics can only
produce a failure, never a distinct mode. In exchange the domain pays a
separate interactive trainer driver, which is exactly why it is excluded from
E81, E82, and E83 (`ops/exp_scaling/CAMPAIGN_LOG.md:2680`). The sixth domain
cannot participate in the paper's own semantic-MaxEnt 2x2.

v1 also needs a 72-update warm start that no static domain needs.

## 3. What the sixth domain must do

The axis v1 claims -- does the result generalize from single-turn answer
emission to sequential decision making? -- is the right axis. Only the
instantiation failed. v2 keeps the axis and fixes the instantiation against
seven requirements.

- **R1. Every decision is mode-bearing.** Decision `i` emits element `i` of the
  canonical key. No shared navigation filler.
- **R2. Mode count comfortably exceeds the sample budget.** Target a mean of at
  least 8 exact modes per map so `distinct@8` is not ceiling-bound at 8 samples,
  matching the 4.52--229 range of the static domains rather than v1's 3--5.
- **R3. Reward is unsaturated.** Target `mean@8` in `[0.20, 0.65]` at pass 0.
- **R4. The matched control must demonstrably lose breadth** on development
  maps before any scientific cell runs. This is the gate v1 never had (§8).
- **R5. Fixed decision count per prompt**, so an episode-level semantic
  advantage divides evenly with no length incentive and the domain becomes
  eligible for E81/E82/E83 (§7).
- **R6. MuJoCo determines which modes exist**, not merely how they execute.
  Otherwise drop the simulator and say so.
- **R7. No domain-specific warm start** unless stage 1 proves one is needed.

## 4. The v2 design: PointMaze Tour

**Contract.** Each map carries `K` landmarks plus a goal, and a total simulator
step budget for the whole tour. At each decision the language policy chooses one
*unvisited landmark* to travel to next. A pinned, hash-bound planner computes
the grid shortest path to that landmark and the existing PD adapter follows it;
MuJoCo remains authoritative for collisions, arrival, budget consumption, and
the continuous trajectory. After all `K` landmarks are visited the run makes one
automatic final leg to the goal.

The menu is over landmarks not yet *named*, not landmarks not yet *reached*. An
episode therefore emits exactly one permutation of the `K` landmarks and always
makes exactly `K` scored decisions, whatever happens physically. Budget
exhaustion is recorded and the tour fails, but the decision sequence is never
truncated.

An episode succeeds when every landmark and then the goal are reached inside the
total budget. **The budget is the difficulty knob**: long tours exhaust it, so
only some orderings succeed, and the set that succeeds is established by
execution.

*Measured correction to an earlier draft of this plan.* Stage 0 shows the
executed step cost is monotone in planned path length on these comb maps: within
a grid-cost tier, tours differ by only one to three steps, while tiers are about
100 steps apart, so ranking by executed cost reproduces ranking by grid cost.
The budget is therefore in practice a threshold on tour length that execution
measures and adjudicates at the margin -- not a quantity that reorders tours
against their grid length. Do not claim in the paper that the feasible set
cannot be read off the grid; claim only what is true, that the catalogue is
established by execution rather than declared (§6).

**Identity.** Each landmark's doorway gate sits between its doorway cell and its
interior cell, so a trajectory can only cross it by actually entering the room.
The canonical key is the order in which the executed trajectory first crosses
each gate. Different intra-leg paths realising the same order merge; different
orders stay distinct. Identity therefore comes from the same execution that
established correctness, as ModeBench requires.

This reuses machinery that already exists.
`find_point_waypoint_route_programs` (`src/oat_drgrpo/point_maze_waypoint.py:393`)
already performs BFS over `(cell, ordered_route, gate_memory)` and already
accumulates `next_route = route + (crossing,)`. `extract_directed_gate_route`
already returns an ordered tuple. v1 clamps both to length one:

- `"route_identity_rule": "exactly-one-directed-gate"` (`point_maze_waypoint.py:141`,
  asserted at `:207`);
- `if len(route) != 1: raise` in the verifier (`point_maze_waypoint.py:529`).

v2 sets the rule to `"ordered-landmark-gate-sequence"` and requires the crossed
sequence to be a permutation of all `K` landmark gates followed by the goal.

**Observation.** At decision `i` the prompt renders the map, all landmark
positions, the visited set in visit order, continuous position and velocity,
**remaining step budget**, and the menu of unvisited landmarks under global
labels `A..D`. It exposes no route hint, no feasibility hint, and no prediction
of what a leg will cost. Only labels in the current legal subset enter sampling,
scoring, or replay, exactly as in v1.

**Mode enumeration.** For each map, execute all `K!` orderings offline through
the pinned runtime and admit those that finish inside the budget. That certified
set is the map's exact mode catalogue, reported per map like the static domains'
exact counts. Certified tours are audit witnesses only: they never enter the
prompt, the warm start, online training, or the replay bank.

**Built parameters (measured, not proposed).** `K = 4`, 24 orderings per map,
map sizes 13 and 15, four rotations per size, budget calibrated per map to admit
8--16 orderings. Realised in `var/data/point_maze_tour_v1`: 384 train, 64 dev,
128 eval maps, **mean 11.09 certified tours per map** (min 11, max 12), every
map admitted only after all 24 of its orderings were executed. Geometry
fingerprints are deduplicated across all three splits, so evaluation maps are
disjoint from training and development by construction. A `K = 5` stratum (120
orderings) can be added later for a mode-count gradient; mixing `K` across maps
is safe because the group is a single prompt (§7).

**Failure handling, and why it matters.** An episode always emits exactly `K`
scored decisions because the menu is over unnamed landmarks. Once a leg exhausts
the budget the tour is doomed, but the remaining decisions are still made and
still scored, so `decision_counts` is constant inside every prompt group and no
episode can shorten its own gradient by failing early. This is the property R5
and §7 depend on, and the runner asserts it every update.

## 5. How v2 answers each fault

| Fault | v2 response |
|---|---|
| F1 one mode-bearing decision in ~30 | exactly `K` decisions, all mode-bearing; per-decision signal rises from `A/30` to `A/4` |
| F2 ceiling 3--5, geometry decides | 11.09 certified tours per map measured, and the policy picks the order outright |
| F3 saturated reward | budget admits 11 of 24 orderings, so a uniform chooser succeeds about .46 of the time |
| F4 physics decorative | partially answered: the catalogue is established by execution, but executed cost tracks path length (§4), so this is a weaker answer than F1--F3 |
| separate driver, E81/E82/E83 excluded | fixed `K` removes the length-incentive obstacle (§7) |
| domain-specific warm start | K-way masked choice should be in reach zero-shot; tested at stage 1 |

There is also a throughput win, though smaller than an earlier draft of this
plan claimed. v1 spends 32.7 scored language-model decisions per episode
(`evaluation_simulator_step_calls` counts transitions,
`train_point_maze_interactive_paired_smoke_v1.py:461`); v2 spends `K = 4`, a 16x
reduction in both sampled rounds and scored slots per update. Measured on an
RTX A6000, v2 runs at **4.50 s/update**, so eight passes (3,072 updates) are
about 3.8h of training. With half-pass evaluations the projection is roughly
6h per cell against v1's 23.5h -- about 4x, not the 8x the draft asserted,
because the language-model cost no longer dominates.

Profiled per update (16 samples): MuJoCo accounts for 3.74s, of which
**2.64s is `gym.make` plus `env.reset` for 16 fresh sessions** and only 1.11s is
actual driving at 11,738 physics steps/s. Caching environments per map geometry
is therefore the obvious next optimization if throughput matters.

**GPU architecture is a cohort-design constraint, not just a speed issue.** The
same probe runs at 18.64 s/update on an RTX 2080: Turing reports
`torch.cuda.is_bf16_supported()` true but emulates bf16, costing about 4x. A
paired cohort must not straddle architectures -- E78 already pins placement per
node for this reason -- so tour cells must request `--gres=gpu:a6000:1`
explicitly. The runner now prints the device and records `device_name` and
`device_capability` in its receipt.

## 6. What v2 does not claim

The intra-leg path is computed by a pinned planner, so the policy controls the
ordering, not the motor control. v2 is a sequential mode-structure test with a
physical cost model, not end-to-end robot learning. It is the same delegation v1
already makes, but now the delegated part is mechanical and the retained part is
the mode.

Nor does v2 claim the physics reorders tours against their grid length; stage 0
shows it does not (§4). The defensible MuJoCo claim is narrower and should be
written that way in the paper: **the mode catalogue of every map is established
by executing all 24 orderings in the pinned simulator, and every reported mode
is a trajectory that simulator actually produced.** That is the same standard
the five static domains meet through their own executors, which is the point --
PointMaze Tour is interactive and sequential, not a robotics result. If a
reviewer asks what MuJoCo buys over a deterministic grid, the honest answer is
continuity and an execution-grounded cost model, not an irreducible physical
fact. §13 keeps open whether that is worth the runtime.

## 7. E81/E82/E83 eligibility

The campaign log excludes PointMaze because "distributing an episode-level
semantic advantage across decisions without creating a length incentive is a
design question." Under v2 that question is answered by construction.

The group is one prompt with `samples_per_update = 16`, and every episode for a
given map emits exactly `K` scored decisions regardless of what the policy
chooses. Within a group `decision_counts` is therefore a constant, so dividing
an episode-level advantage by it is a uniform rescaling and cannot reward or
punish length. A policy cannot make its episode longer.

Implementation: form `A = A_drgrpo + A_sem` at the episode level, where `A_sem`
is E81's term verbatim -- `0.10 * z`, `z` the predictor-centered clipped
surprisal of the episode's canonical key under a prompt-local open-set predictor
with pseudocount 1, one structural unseen bucket and surprisal clip 5, applied
only to validator-positive episodes and added after Dr.GRPO's own task
centering -- then apply the existing per-decision division. The prompt-local
predictor's support is the ordered-tour key space, which is what makes the
surprisal meaningful.

Add a test asserting the PointMaze semantic objective dictionary equals the
static-domain one except for the driver key, mirroring the existing E81/E82
launcher-parity test. Mixed `K` across maps is safe because the predictor and
the advantage are both prompt-local.

## 8. Admission gates

Gates run on training and development maps only. The evaluation split stays
sealed. All thresholds are registered before measurement.

**Stage 0 -- environment oracle.** Enumerate all `K!` orderings per map through
the pinned runtime. Require mean certified tours per map `>= 8`, every map
`>= 4`, infeasible fraction `>= 0.25` so the budget genuinely binds, no geometry
stratum degenerate, and measured throughput recorded.

**Stage 1 -- base policy viability.** Untouched Qwen2.5-0.5B, no warm start, no
demonstrations, development maps: require `mean@8` in `[0.20, 0.65]`,
`pass@8 >= 0.70`, `distinct@8 >= 2.5`, and at least 60% of maps exposing two or
more distinct tours in eight samples. v1 sat at `distinct@8 = 1.37`, which this
gate rejects. If stage 1 fails only on floor effects, add the shared train-only
warm start and re-run; do not weaken the thresholds.

**Stage 2 -- collapse gate (new; the gate v1 never had).** Control arm only, two
seeds, development maps. Require the control's `distinct@8` to fall by at least
0.5 from its pass-0 value. A domain whose matched control does not lose breadth
cannot demonstrate retention, and admitting one wastes a cohort. v1 would have
failed here at `+0.016`.

*Amendment, registered before the gate ran:* four passes, not two, with an
evaluation every pass. A one-pass control probe on four maps showed `distinct@8`
rising (3.00 to 3.75) while `mean@8` rose 0.594 to 0.875 -- the policy is still
learning to succeed at all, and breadth plausibly grows before it collapses.
Two passes therefore risked rejecting the domain for being early rather than for
being inert. The threshold is unchanged; only the observation window widens, and
five evaluation points resolve the trend rather than one endpoint.

**Stage 3 -- matched smoke.** One seed, both arms, one pass: nonzero reward,
nonzero replay-bank occupancy, bounded KL, matched rollout counts, identical
observations and masks across arms.

**Stage 4 -- scientific cohort.** Five paired seeds, eight passes, half-pass
checkpoints, registered before launch: Qwen2.5-0.5B seeds 43--47, Falcon3-1B
seeds 55--59, matching the static protocol exactly.

**Selection-bias disclosure.** Gating domains on "the control collapses" biases
ModeBench toward domains where collapse occurs, and the paper must say so. Two
mitigations, both required: report v1 as the domain that failed the gate (§10),
and register every threshold before measurement. The gate governs *domain
admission on development maps*, never treatment outcomes, and never replaces a
scientific cell.

## 9. Implementation plan

Stages are ordered so each one is cheap to abandon.

1. **Identity rule generalisation.** `src/oat_drgrpo/point_maze_waypoint.py`:
   add `"ordered-landmark-gate-sequence"`, drop the `len(route) != 1` check for
   that rule, extend `parse_point_waypoint_spec` to accept it, key on the
   ordered tuple. Keep the v1 rule accepted and unchanged so E78pm receipts stay
   parseable and auditable.
2. **Map and landmark generator.** New `point_maze_tour_data.py` beside the v1
   generator; do not edit `point_maze_waypoint_data.py`, whose frozen seed
   88,104 must keep reproducing v1. Emit landmark cells, gate rings, total
   budget, and `K`, with balanced rotations and disjoint fingerprints from every
   v1 and interface-development cohort.
3. **Offline mode enumeration and budget calibration.** New
   `ops/make_point_maze_tour_data.py`: run all `K!` orderings through the pinned
   worker, calibrate the per-map budget on training maps to hit the stage-0
   band, and write `identity.json` with certified tours held out of model rows.
4. **Option-level policy surface.** New `point_maze_tour_policy.py`: observation
   with visited set and remaining budget, unvisited-landmark menu, terminal
   padding, and a Falcon twin that is a pure role-marker swap. Extend
   `tests/test_falcon_prompt_surface.py` to cover it.
5. **Leg executor.** Pinned planner plus the existing PD adapter, hash-bound;
   the planner hash joins the spec alongside `controller_sha256`.
6. **Runner.** Fork `ops/train_point_maze_verified_replay_only.py` to
   `ops/train_point_maze_tour.py`, changing the episode loop and constant
   `decision_counts`, then add the semantic-advantage path from §7.
7. **Gates 0--3**, in order, each written to `var/artifacts/` with a registered
   decision record.
8. **Preregistration** for the Qwen and Falcon cohorts, then stage 4.
9. **Paper edits** (§11).

Steps 1--5 are offline and cost no GPU time. Step 6 is the only substantial new
trainer work.

## 10. Disposition of v1

v1 is retired into the paper as a reported boundary result, not deleted and not
quietly dropped. Its artifacts, receipts, and preregistrations stay immutable,
consistent with the repository's existing norm for failed probes
(`docs/modebench_vnext_capability_staircase.md`).

The appendix reports: control `distinct@8` 1.373 at pass 0 and 1.389 at pass 8;
the paired difference of `+0.025` against a within-run checkpoint standard
deviation of `0.048`; two of five seeds at exactly zero; and the four structural
faults of §2. The framing is explicit: **this is a null result about the
environment, not about x-Mode.** When mode identity rides on one decision in
thirty and reward is saturated, GRPO has no lever to pull, so there is no
collapse for verified replay to prevent.

That section then motivates the stage-2 collapse gate and discharges the
selection-bias objection to it.

## 11. Paper changes required

- Abstract and §1: keep "six executable domains ... one interactive MuJoCo
  maze"; the maze becomes Tour. If v2 has no cohort by submission, the abstract
  must say five measured domains plus a reported boundary result -- decide this
  from the schedule, not from v2's outcome.
- Table `tab:tasks`: PointMaze row becomes "Select next unvisited landmark;
  simulate the tour" / "Ordered landmark gate sequence" / "Tours realising the
  same order" / certified feasible-tour count.
- §3 PointMaze paragraph: rewrite for the option interface, the budget, and
  physics-certified mode enumeration.
- Figure `fig:modebench-examples` panel F: two distinct *orders* over the same
  landmarks, not two paths through one wall.
- Results table and Figure 4 PointMaze panels: repoint at the v2 cohort.
- New appendix: the v1 boundary result (§10).
- Limitations: replace the v1 sentence with §6's honest statement, and add the
  domain-admission gate and its selection-bias disclosure.
- Evidence ledger: new row for the v2 question, and the v1 row restated as a
  reported null about the environment.
- New preregistrations under `paper/preregistration/`, one per family.
- `ops/check_paper_domain_prompts.py` and
  `ops/check_paper_figure1_contract.py` extended to the v2 surface.

## 12. Risks

- **v2 also fails to collapse.** Mitigated by stage 2, which costs two short
  control runs instead of a 10-cell cohort, and by v1 already being reported as
  a boundary result so the paper is not hostage to v2's outcome.
- **Budget calibration lands outside the band.** Calibrate on training maps
  only, with development maps as an unbiased check; never recalibrate against
  evaluation maps or after seeing an arm difference.
- **The task is a permutation choice with a physics cost model.** True. State it
  in limitations (§6) instead of overselling robotics; the sequential
  mode-structure claim survives it intact.
- **Reviewer objects to the admission gate as selection.** Answered by §8's
  disclosure plus the v1 appendix, which is the strongest possible evidence that
  the gate was not applied to favourable outcomes.
- **Schedule.** Steps 1--5 are offline. The cohort should be far cheaper than
  v1 per cell, but stage 0 must report measured throughput before stage 4 is
  scheduled.

## 13. Decisions still open

1. `K = 4` only, or a `K = 4` / `K = 5` mixed stratum for a mode-count gradient?
   Mixed is safe under §7 but doubles enumeration cost.
2. Should the policy also choose an approach corridor per leg, adding a second
   mode-bearing axis per decision? It multiplies the mode count but risks
   reintroducing dilution.
3. Does stage 1 clear without a warm start? If it does, drop the warm start and
   match the static protocol exactly, which removes a standing reviewer question.
4. Falcon cohort concurrently with Qwen, or gated on the Qwen cohort clearing
   its audit?
5. Is MuJoCo worth its runtime given §6? A deterministic grid executor would
   keep every scientific property this design relies on and remove the separate
   worker venv. Decide before the paper commits to "interactive MuJoCo maze".

## 14. Build status

Implemented and verified offline (no GPU required):

| Component | Path | State |
|---|---|---|
| Environment core, identity, tour extractor | `src/oat_drgrpo/point_maze_tour.py` | built; 24/24 orders yield 24 distinct keys |
| Comb map generator | `src/oat_drgrpo/point_maze_tour_data.py` | built; rotation decoupled from size, cross-split dedup |
| Leg executor worker | `src/oat_drgrpo/point_maze_tour_worker.py` | built; fixed-K contract asserted |
| Process client | `src/oat_drgrpo/point_maze_tour_process.py` | built; cross-venv round trip verified |
| Prompt surface | `src/oat_drgrpo/point_maze_tour_policy.py` | built, with the Falcon twin |
| Stage-0 materializer | `ops/make_point_maze_tour_data.py` | run; two-phase (maze venv enumerates, paper venv packs) |
| Trainer | `ops/train_point_maze_tour.py` | built; imports verified |
| Gate submission | `ops/slurm/point_maze_tour_gate.slurm` | built |

Stage 0 is **complete**: `var/data/point_maze_tour_v1` holds 384/64/128 maps,
mean 11.09 certified tours per map, every tour executed in the pinned runtime.

`extract_directed_gate_route` could not be reused directly: it rejects any
trajectory that recrosses a gate, which a two-cell room requires on exit. v2
therefore carries `extract_tour_gate_route`, identical in bounds, teleport,
hysteresis, and crossing geometry (it calls the same helpers) and differing only
in permitting repeat crossings. `point_maze_waypoint.py` is untouched, so the
E78pm/E79pm receipts stay reproducible.

Stages 1--4 are GPU work. Submit them correctly: this cluster routes by
requested walltime, and **only `--time` of one hour or less reaches the `all`
partition** (~40 nodes); anything longer is rerouted to `cs` (6 nodes), which is
saturated by this project's own registered E79/E82 cohorts. Measured with
identical hold jobs differing only in `--time`:

| `--time` | partition | QOS |
|---|---|---|
| `00:30:00`, `01:00:00` | `all` | none |
| `01:30:00` … `1-00:00:00` | `cs` | short |
| `7-00:00:00` | `cs` | long |

Stage 1 is evaluation-only and fits comfortably in an hour, so it belongs in
`all`. Stage 2 (two passes) and stage 4 (eight passes) will exceed the hour and
must be scheduled into `cs` or `lowprio`, or split into resumable chunks.
