# E49E answer-blind V2 node302 in-flight abort

**Status: TERMINAL ABANDONMENT — 2026-07-24**

Job `30075051` was submitted to node302 and initially reported pending with no
estimated start.  During the placement check it began running and was
cancelled after 36 seconds so the CPU-only work could move to available
mltheory capacity.

The archived evidence contains no request-cache file, repair record, or
materialized output.  Nevertheless, because the four-worker process may have
issued proposal requests before receiving and durably caching responses, this
amendment conservatively treats up to four V2 proposal slots as consumed.
Version `e49e_answer_blind_symbolic_singleton_repair_v2` and seeds
`492241`–`492244` are abandoned and must never be resumed or resampled.

The intact evidence is archived at
`var/artifacts/e49e_trace_bank_math_toy_repair_v2_aborted_node302_inflight_20260724`.
No output from that job may enter calibration, training, or later repair.
