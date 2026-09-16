# E118-Q3: Qwen2.5-3B five-seed MaxRL extension

Date frozen: 2026-09-01, before inspecting any E118-Q3 outcomes.

## Question

Does verified replay improve semantic support retention beyond direct MaxRL
optimization at Qwen2.5-3B scale, and does the MaxRL-by-replay interaction
replicate across model scales?

## Design

- Model: Qwen2.5-3B-Instruct, frozen revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and Pantry.
- Seeds: 70, 71, 72, 73, 74.
- New arms: MaxRL and ReplayMaxRL.
- Existing matched comparators: terminal E80-R1 Dr.GRPO and ReplayGRPO cells.
- Training: 384 prompts, eight passes, 16 fresh samples per update, one PPO
  epoch, beta zero, and the frozen E80-R1 Qwen-3B optimizer contract.
- ReplayMaxRL uses verified-likelihood replay weight 0.1; MaxRL executes the
  identical replay machinery in compute-only mode with zero replay derivative.
- Semantic-MaxEnt and counterfactual/proposal objectives are disabled.
- Fixed-draw evaluation: pass@8 and distinct correct semantic modes at k=8.
- Recovery checkpoints: every 192 prompt updates, retaining one rolling
  checkpoint. Jobs use a 12-hour wall-time and are requeue/resume enabled.

## Analysis

Report each arm by domain and seed at matched checkpoints. Primary contrasts
are ReplayMaxRL minus MaxRL and the factorial interaction

`(ReplayMaxRL - MaxRL) - (ReplayGRPO - Dr.GRPO)`.

Endpoint and learning-curve/AUC summaries will use paired seed uncertainty.
Qwen-3B is a scale extension; it does not replace the Qwen-0.5B primary result.

