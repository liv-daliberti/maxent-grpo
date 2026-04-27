# Repository Cleanup

Goal: make this repo a small, shareable training tree for two public paths:

- Dr.GRPO baseline
- Dr.X-GRPO semantic PPO variant

## Current Status

- Runtime: `var/seed_paper_eval/paper310`
- Dev tools installed there: `pytest==9.0.3`, `ruff==0.15.12`
- Full imports need:

```bash
export OAT_ZERO_PYTHON=var/seed_paper_eval/paper310/bin/python
export LD_LIBRARY_PATH="$PWD/var/seed_paper_eval/paper310/lib:${LD_LIBRARY_PATH:-}"
```

- Last full check after broad formatting polish: `ruff` pass, `py_compile`
  pass, `pytest -q` pass (`97 passed, 7 warnings`).
- `train_zero_math.py` is down to a thin learner assembly entry point; baseline
  GRPO update logic lives in `learner/grpo.py`, learner initialization lives in
  `learner/init.py`, and the run/eval/logging loop lives in `learner/run.py`.
- W&B logs are filtered to the Dr.GRPO-vs-Dr.X comparison surface by default;
  set `OAT_ZERO_WANDB_LOG_DEBUG_METRICS=1` to log debug diagnostics.
- Public Dr.X dashboard metrics now log under `drx/perf/*`, `drx/signal/*`,
  `drx/semantic/*`, and `drx/objective/*` for easier run monitoring.
- Accuracy/reward uptick metrics now log under `drx/progress/*`, including
  gain from start, gain from previous eval/log, best-so-far, and steps since
  best.
- `ops/` now contains only the public training launchers, the public evaluation
  grid entry point, and `repo_env.sh`; one-off sweeps, uploads, plots, and old
  launchers were removed.
- Working tree is intentionally broad and dirty from prior cleanup/experiment
  work. Do not revert unrelated changes.

## Left

- No active cleanup chunk is queued.
- Compatibility-facing Dr.X flag and W&B names now use the cleaned release
  vocabulary.
- Broad formatting polish is complete.

## Quality Gate

```bash
export OAT_ZERO_PYTHON=var/seed_paper_eval/paper310/bin/python
export LD_LIBRARY_PATH="$PWD/var/seed_paper_eval/paper310/lib:${LD_LIBRARY_PATH:-}"

$OAT_ZERO_PYTHON -m ruff check src tests
PYTHONPYCACHEPREFIX=var/tmp/pycache PYTHONPATH=src $OAT_ZERO_PYTHON -m py_compile \
  src/oat_drgrpo/train_zero_math.py \
  src/oat_drgrpo/logging_utils.py \
  src/oat_drgrpo/templates.py \
  src/oat_drgrpo/learner/grpo.py \
  src/oat_drgrpo/learner/init.py \
  src/oat_drgrpo/learner/run.py \
  src/oat_drgrpo/learner/drx.py \
  src/oat_drgrpo/learner/drx_backward.py \
  src/oat_drgrpo/learner/drx_logging.py \
  src/oat_drgrpo/learner/drx_semantics.py \
  src/oat_drgrpo/runtime.py \
  src/oat_drgrpo/listwise.py \
  src/oat_drgrpo/semantic_remix.py \
  src/oat_drgrpo/semantic_utility.py \
  src/oat_drgrpo/passk_eval.py \
  src/oat_drgrpo/math_grader.py
PYTHONPATH=src $OAT_ZERO_PYTHON -m pytest -q
```
