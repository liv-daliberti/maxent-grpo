# MaxEnt-GRPO

This repository now treats the upstream OAT stack as the canonical training
surface.

The only active training entrypoints under `ops/`
are:

- baseline DR.GRPO:
  `ops/run_oat_zero_exact_1p5b_upstream.sh`
- 7B R1-template DR.GRPO:
  `ops/run_oat_zero_math_7b_r1_upstream.sh`
- listwise maxent-explorer overlay:
  `ops/run_oat_zero_explorer_1p5b_upstream.sh`

The working Slurm wrappers are:

- baseline:
  `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm`
- 7B R1-template:
  `ops/slurm/train_understand_r1_zero_qwen2p5_math_7b_r1_readme_flash_node105.slurm`
- explorer:
  `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm`

Older TRL/Hydra orchestration and pre-canonical training launchers are kept for
reference under `archive/trl/`, but
they are no longer the active front door for this repo.

## Canonical Stack

The proven baseline path uses the upstream
`understand-r1-zero/`
checkout plus the repo-local listwise overlay in
`oat_zero_ext/`.

Canonical runtime:

- `python==3.10.20`
- `torch==2.6.0`
- `transformers==4.51.3`
- `vllm==0.8.4`
- `oat-llm==0.1.3.post1`
- `deepspeed==0.16.8`
- `flash-attn==2.7.4.post1` via the runtime overlay

Canonical environment and helpers:

- python: `var/seed_paper_eval/paper310/bin/python`
- audit: `tools/audit_oat_setup.py`
- guide: `docs/guides/oat-upstream-drgrpo.rst`

## Quick Start

Verify the canonical OAT runtime first:

```bash
python tools/audit_oat_setup.py
```

Launch the working baseline DR.GRPO run:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm
```

Launch the upstream 7B `r1`-template DR.GRPO recipe on `node105`:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_7b_r1_readme_flash_node105.slurm
```

Launch the listwise maxent-explorer overlay on the same stack:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm
```

Direct shell entrypoints are also available:

```bash
bash ops/run_oat_zero_exact_1p5b_upstream.sh
bash ops/run_oat_zero_math_7b_r1_upstream.sh
bash ops/run_oat_zero_explorer_1p5b_upstream.sh
```

## Safety Model

The repo is now organized so the active training surface stays narrow and
predictable:

- baseline DR.GRPO remains the default path
- explorer is opt-in through `objective=maxent_listwise`
- the same README-flash OAT runtime is shared by both launchers
- archived launchers are removed from `ops/`
- the audit script checks the pinned runtime before launches
- focused tests cover the active OAT wrappers and the archived path split

This reduces accidental drift back onto older TRL launchers or older OAT
variants while still preserving legacy material for reference.

## Archive

Retired launchers live under `archive/trl/`:

- `archive/trl/ops/`: retired orchestration wrappers and legacy training entrypoints
- `archive/trl/ops/slurm/`: retired Slurm launchers

The archived material is intentionally kept out of `ops/`
so the canonical surface is just the working OAT baseline plus the explorer
variant.

The older research package under
`src/maxent_grpo/` is retained
for reference and historical experiments, but it is not the active training
entrypoint anymore.

## Repository Layout

```text
.
├─ understand-r1-zero/         # canonical upstream OAT training checkout
├─ oat_zero_ext/               # local listwise maxent-explorer overlay
├─ ops/                        # active OAT launchers only
├─ archive/trl/                # retired TRL and legacy launchers
├─ tools/audit_oat_setup.py    # runtime audit for the canonical OAT stack
├─ docs/                       # OAT-first docs and archive notes
├─ src/maxent_grpo/            # retained legacy research package
└─ var/                        # runtime envs, caches, logs, and outputs
```

## Development

- Run the focused OAT checks with:

```bash
pytest -q tests/training/test_oat_zero_listwise.py tests/training/runtime/ops/test_ops_slurm_train.py
```

- Compile the patched OAT pieces with:

```bash
python -m py_compile understand-r1-zero/train_zero_math.py oat_zero_ext/listwise.py tools/audit_oat_setup.py
```

## Citation

If you use this work, please cite: "MaxEnt-GRPO: Maximum-Entropy
Group-Relative Policy Optimization (2025)."

```bibtex
@misc{MaxEntGRPO2025,
  title        = {MaxEnt-GRPO: Maximum-Entropy Group-Relative Policy Optimization},
  author       = {Liv d'Aliberti},
  year         = {2025},
  publisher    = {GitHub},
  note         = {Code: this repository}
}
```

## License

Apache 2.0 - see `LICENSE`.
