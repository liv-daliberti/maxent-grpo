# Minimal OAT 1.5B Training Tree

This repository is intentionally cut down to one thing:

- 1.5B OAT Dr.GRPO
- 1.5B OAT Dr.GRPO-Explorer

The live code is under `src/oat_drgrpo/`:

- `train_zero_math.py`: the patched OAT learner
- `listwise.py`: the Explorer / DrX prompt-group helpers
- `math_grader.py`: the verifiable-math reward/grader

The only retained launch surface is:

- `ops/run_oat_zero_exact_1p5b_upstream.sh`
- `ops/run_oat_zero_explorer_1p5b_upstream.sh`
- `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm`
- `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm`

Datasets used by those scripts live under:

- `datasets/train/math_12k`
- `datasets/evaluation_suite`

Quick checks:

```bash
python -m py_compile src/oat_drgrpo/train_zero_math.py src/oat_drgrpo/listwise.py src/oat_drgrpo/math_grader.py
bash -n ops/run_oat_zero_exact_1p5b_upstream.sh ops/run_oat_zero_explorer_1p5b_upstream.sh
```
