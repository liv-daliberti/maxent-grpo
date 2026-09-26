# PointMaze repair final v1 r1 execution amendment

**Status:** FROZEN BEFORE FINAL TRAINING  
**Date:** 2026-07-30

The generic direct trainer retained the development-pair seed allow-list.
Before any final job is submitted, this execution-only adapter replaces that
allow-list with the five preregistered final seeds `76531`–`76535`. It does not
change the model, data, splits, optimizer, rollout count, MaxEnt recipe,
evaluation schedule, or verifier. The terminal receipt explicitly records the
final cohort and untouched `eval` split.
