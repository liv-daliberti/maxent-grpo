# Archived Training Surface

This directory keeps retired training entrypoints that are no longer part of
the canonical repository front door.

What remains active:

- `ops/run_oat_zero_exact_1p5b_upstream.sh`
- `ops/run_oat_zero_explorer_1p5b_upstream.sh`
- `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm`
- `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm`

What lives here instead:

- retired TRL and Hydra orchestration wrappers
- retired Slurm launchers for those wrappers
- older noncanonical OAT launchers that were superseded by the working
  README-flash OAT stack

Layout:

- `archive/trl/ops/`
- `archive/trl/ops/slurm/`

These files are preserved for historical comparison and reference, but they
should not be treated as the default launch path for new work in this repo.
