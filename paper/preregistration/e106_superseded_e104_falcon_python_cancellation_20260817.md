# E106 execution record: superseded E104 Falcon-Python cancellation

Recorded on 2026-08-17 after the E106 ledger was durably written and all three
repair jobs were released.

E104 job `30637792` (`e104-f1-python`) was rechecked with `scontrol` immediately
before cancellation.  It was `PENDING`, assigned no node, and had runtime
`00:00:00`; its frozen source root was the parser-obsolete E104 snapshot.  E106
job `30640330` is its exact Falcon3-1B Python replacement under the versioned
`python-factor-response-v2-latex-lambda` snapshot.

Job `30637792` was cancelled at 2026-08-17 18:17 EDT.  `sacct` then reported
`CANCELLED by 363432`, runtime `00:00:00`, no start time, no assigned node, and
exit code `0:0`.  No run directory or evaluation artifact was created.  No
non-Python E104 job was cancelled or modified by this action.

This action was explicitly allowed by the frozen E106 protocol and only avoids
spending a GPU on a cell whose parser snapshot is already superseded.  The
combined release gate excludes all three E104 Python cells and uses the three
E106 replacements; it continues to require all twelve non-Python E104 cells.
