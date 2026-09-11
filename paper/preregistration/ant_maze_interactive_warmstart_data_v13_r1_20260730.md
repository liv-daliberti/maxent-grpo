# AntMaze v13-r1 constrained warm-start data: controller thread repair

**Status: FROZEN AFTER THE FIRST V13 MATERIALIZER STOPPED AND BEFORE R1 REPLAY — 2026-07-30**

The original v13 materializer stopped before writing its output root because
the persistent worker's default multithreaded controller inference caused the
certified lower route to diverge. The exact v12 executor reproduces the stored
upper/lower traces at 832/605 simulator steps when BLAS/Torch inference is
single-threaded; repeating the lower route three times under that boundary
was exact. This is the same determinism boundary used by the successful
route-generation execution, not a route, threshold, model, map, prompt,
controller, or label change.

R1 sets `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` in the trusted worker.
It then replays the same four train maps, two certified routes per map, and
the unchanged public one-compass-token policy prompts from v13. Development
and evaluation rows remain unloaded and no language model is sampled. All
other v13 materialization checks and information firewalls remain exact.
