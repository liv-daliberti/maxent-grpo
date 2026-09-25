# A non-synthetic Re:Max versus MaxRL application

Original research recommendation, 2026-09-21, written before the pilot.

**Pilot follow-up:** The authorized Coder-7B capability pilot completed for
0.1047 A6000 GPU-hours: 30/192 accepted programs, with multiple modes on
only one of three development tasks. It failed the frozen capability gate;
no training or appendix addition followed. See the
[result report](../var/artifacts/codecontests_pilot_20260921/report.md).
Before sampling, a missing public equation in 359_B was recovered from its
original image. The earlier failure on that task is therefore not clean
model-capacity evidence. The proposal below remains prospective.

## Recommendation

Use **constructive program generation on a small, audited subset of
CodeContests+**: human-authored competitive-programming problems that admit
multiple valid outputs. Start a new Coder-7B feasibility study, then compare
MaxRL with the current Re:Max uniform verified replay objective.

This is the strongest match to the present paper, but **sub-200-GPU-hour
feasibility is conditional on a pilot, not an established measurement**. The
old small-coder experiment failed. The proposed compute envelope below caps
the complete study, including failed development attempts and evaluation.

The accurate provenance claim is “human-authored contest problems with
generated verification assets.” These are external programming tasks, not
our procedurally generated ModeBench problems. They are not production
software issues or observational application data. CodeContests+ generates
tests and checkers with LLM agents; do not describe those assets as human
authored or the checkers as official contest judges.

## What the cited papers actually use

| Paper already in our bibliography | Relevant datasets/application | Assessment for this study |
|---|---|---|
| [MaxRL](https://arxiv.org/html/2602.02710v3) | ImageNet, generated mazes, GSM8K, POLARIS-53K mathematical reasoning | GSM8K is a direct comparator precedent, but checking only the final numeric answer does not identify distinct correct reasoning modes. Image classification has one correct label and mazes are synthetic. |
| [SetPO](https://arxiv.org/html/2602.01062v1) | GSM8K; synthetic Countdown | Small-model, short-response precedent; uses a representation-based diversity definition, not our execution-bound identity. |
| [DivPO](https://arxiv.org/html/2505.23433v1) | GSM8K training and mathematical reasoning evaluations | Non-model-generated questions, but same mode-definition problem. Its reported eight A6000s for three days is 576 GPU-hours for a run; cannot copy its full recipe. |
| [UCPO](https://arxiv.org/html/2605.00365v1) | DeepScaleR-derived competition mathematics | Appropriate for mathematical accuracy, but equation-level diversity requires a different mode definition. |
| [Outcome-based Exploration](https://arxiv.org/html/2509.06941v1) | MATH and DAPO mathematics | Exploration across answers includes incorrect answers, unlike diversity conditional on verified success. |
| [GAPO](https://aclanthology.org/2025.emnlp-main.1649.pdf) | Randomly generated lists for diversity training | Training data are explicitly synthetic. Downstream HumanEval evaluation is not a real-data diversity-training experiment. |
| [Beyond Mode Collapse / DMPO](https://arxiv.org/html/2605.19461v1) | NP-10K/MM-NP-10K generated optimization tasks | Excellent multiple-solution structure, but synthetic instances fail this request. |
| [DQO](https://arxiv.org/html/2509.04784v3) | CNN/DailyMail, Dolly, CommonGen, GSM8K | Real-source text applications, but learned quality rewards and representation/judge diversity would change our verifier contract. |
| [Sinha et al., inverse probability scaling](https://arxiv.org/html/2601.21669v1) | Molecular design against SARS-CoV-2 protease, PDB 7UVU | Strong actual scientific application, but chemical Mamba, docking, retrosynthesis, and existing SATURN replay/diversity mechanisms require a separate experimental pipeline. No verified runtime supports a sub-200-hour reproduction. |
| [Large Language Monkeys](https://arxiv.org/html/2407.21787v1) | CodeContests, SWE-bench Lite, MATH, GSM8K, MiniF2F | Provides the coding connection. Section 4.2.2 explicitly identifies false negatives when CodeContests expects one output for a multiple-answer task. CodeContests+ addresses this particular missing capability. |

GSM8K is [human-written](https://github.com/openai/grade-school-math), but
invented word problems are a weaker application story. Free-form derivations,
AST hashes, and judge clusters do not automatically inherit ModeBench's exact
execution identity. Useful mathematical or algorithmic diversity certainly
exists; these datasets simply do not supply the required identification rule.

## Why constructive programming fits

[CodeContests+](https://arxiv.org/html/2506.05817), section 4.3, supplies
custom checkers for tasks with many correct outputs, such as topological
orderings. The [released dataset](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus)
contains original problems, contestant submissions, generated tests,
validators, and checkers.

For each problem, generate a complete Python program and execute it on a
fixed test suite. Reward one only when every training check passes. For an
accepted program, canonicalize the actual output witnesses on fixed identity
probes and use their tuple as the bank key. Variable renaming and source
formatting then leave the key unchanged. Normalize unordered components and
interchangeable labels only when justified by that task's semantics.

This measures **observed behavior on fixed inputs**, not algorithm identity
or total solution-space coverage. The support need not be enumerable. Avoid
claiming that a different correct program text necessarily represents a
different mode.

The practical question is whether retaining alternative correct behaviors
produces more reliable solution portfolios at a fixed sampling budget.
Report pass@1 and pass@8/pass@32 on held-out problems and stronger withheld
tests, plus PCMD and distinct verified modes. A gain in PCMD alone supports
diversity preservation; a gain on withheld tests additionally supports
practical reliability. The latter is a hypothesis, not a guaranteed
consequence of the former. No test-time replay or repair should be needed.

## Existing work that changes the recommendation

The [existing extension plan](modebench_constructive_code_extension_plan.md)
already identified 395 metadata-aligned candidate problems in the
CodeContests+ / CodeContests-O intersection. That is a feasibility pool,
not 395 admitted tasks. Ten tasks eventually passed the v6 executable gate
with 960/960 replay records:
`var/artifacts/constructive_code_v6_gate_audit.json`.

However, Qwen2.5-Coder-0.5B had zero accepted candidates out of 192 development
samples; a train-only warm start also failed. The subsequent Coder-1.5B probe
again accepted zero out of 192:
`var/artifacts/constructive_code_v8_coder_15b_viability.json` and
`paper/preregistration/constructive_code_v8_coder_15b_capacity_retry_20260730.md`.
All 64 samples on the relatively simple Diverse Team problem terminated
normally. The failure is not explained by a universally inadequate token cap.

Thus, do not revive the stopped small-model protocol unchanged. Use a new
development pilot with [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).
This model choice is a capacity hypothesis to test. The old constructive
trainer also uses a historical multi-term objective; reusing it verbatim
would not implement the current Re:Max versus MaxRL comparison. Reuse its
execution infrastructure, and connect it to the maintained MaxRL/replay path.

## Proposed experiment and complete compute envelope

Assume **A100-class GPUs**, counting every allocated GPU-hour, including
actor/learner GPUs, startup, checkpointing, and CPU-checker waits while GPUs
remain allocated. A5000/A6000 hours are not interchangeable with these hours.

Target 128 training, 16 development, and 48 evaluation problem IDs, disjoint
by problem and deduplicated statement. These are proposed sizes, conditional
on enough independently audited tasks. Keep original problem statements.
Select by checker reliability, multiple-output semantics and development
viability; do not select evaluation problems by treatment performance.

Use two arms and three paired seeds. Both arms share the starting checkpoint,
fresh rollout budget, sampling settings, MaxRL estimator and replay scoring
work; only Re:Max applies the uniform verified-replay derivative. Banks begin
empty and contain policy-generated accepted solutions, never human references.
Use the existing replay coefficient rather than a wide hyperparameter search.

The initial target is eight passes, eight rollouts per prompt, four prompts
per update, and a 1,024-token output cap. This is 256 updates and 8,192 fresh
programs per run, or at most about 8.4 million newly generated training tokens
per run. That token count excludes teacher forcing and replay; all of their
actual cost belongs in timing.

| Allocation | Aggregate GPU-hours |
|---|---:|
| Development generation, integration and paired timing pilot | 12 |
| Six training runs, at most 22 GPU-hours each | 132 |
| Base, intermediate and terminal evaluation | 36 |
| Reserve, including failed attempts | 15 |
| **Hard study envelope** | **195** |

These are allocation ceilings, not measured completion times. For the target
training schedule, 280 allocated-GPU seconds per update gives 119.5 GPU-hours
across six runs, leaving 12.5 hours of the training allocation for startup and
checkpointing. For two GPUs this threshold is 140 seconds of wall time per
update. Time the actual code workload with occupied replay banks and CPU
checking; short ModeBench timings are not a valid substitute.

Suggested evaluation is 128 terminal draws per held-out problem per run,
with smaller fixed intermediate samples and one shared base evaluation.
Estimate terminal inference cost in the pilot too. For PCMD retain the
paper's minimum of 30 correct samples, disclose eligible-problem counts, and
report paired comparisons on common eligible tasks alongside full-task
accuracy. Do not score missing PCMD as zero or silently discard hard tasks.
Bootstrap over problems and show all seed pairs separately.

Before the full study, require nontrivial development success (for example,
aggregate pass@1 of at least 10%), multiple accepted modes on a substantial
fraction of development tasks, reliable checkers, and measured cost within
both the training and evaluation allocations. Freeze exact thresholds before
new samples. If the pilot fails, stop within its 12-hour allocation; do not
promise a completed study or retrospectively shorten different arms.

Checker auditing is CPU work and still requires engineering time. Preserve
historical audits, verify positive/negative submissions and symmetry handling,
and hold stronger tests out of training. CodeContests-O is an optional audited
overlay, not a guarantee of checker correctness. Download selected rows;
do not mirror the full submission archives.

## Decision

This is the first application I would try because it preserves our central
scientific contract and reuses substantial local infrastructure. It is a
concrete 195-GPU-hour proposal with a cheap stopping decision, rather than an
already-proven feasible run. If “non-synthetic” must mean production software
or measured scientific data, this contest application does not meet that
stricter interpretation; molecular design or real-bug test generation would
need a separate feasibility study.
