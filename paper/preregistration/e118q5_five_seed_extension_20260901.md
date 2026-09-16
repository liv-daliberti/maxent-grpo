# E118-Q5 five-seed extension

Frozen while E118-R2 is running, before inspecting outcomes for seeds 45--47.
Complete the existing Qwen2.5-0.5B factorial by adding MaxRL and ReplayMaxRL
for seeds 45, 46, and 47 across Countdown, Graph Coloring, Python Factors,
MathIR, and PantryPlan. Reuse the exact E118-R2 objective, optimizer, evaluation,
and 3072-update contract and the completed seed-matched E78 Dr.GRPO and
ReplayGRPO comparators. Retain one rolling resume checkpoint for storage safety,
use 12-hour limits, and allow placement on healthy Ampere-or-newer GPUs.
