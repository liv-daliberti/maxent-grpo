# Operations Toolkit

Single supported training path:

- `run_dual_4plus4_single_node.sh` submits `slurm/train_dual_4plus4.slurm`.
- Layout is fixed to one 8-GPU A100 node split as:
  - GRPO: 1 vLLM GPU + 3 training GPUs
  - MaxEnt: 1 vLLM GPU + 3 training GPUs
- Default submit settings are built in:
  - `SBATCH_ACCOUNT=mltheory`
  - `SBATCH_PARTITION=mltheory`
  - `SBATCH_TIME=48:00:00`

Run from repo root:

```bash
var/repo/ops/run_dual_4plus4_single_node.sh
```
