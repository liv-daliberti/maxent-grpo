"""Interactive AntMaze worker bound to the admitted cached v17 controller."""

from __future__ import annotations

from . import ant_maze_interactive_worker as base
from . import ant_maze_worker_v12 as v12
from . import ant_maze_worker_v17 as v17
from . import ant_maze_worker_v17_r1 as v17_r1


# Reuse the audited interactive state machine while rebinding every
# controller-dependent lookup to the admitted cached v17 executor.
v12._model = v17_r1._model
v12.controller_identity = v17_r1.controller_identity
v12.controller_receipt_sha256 = v17_r1.controller_receipt_sha256
v12.WAYPOINT_DISTANCE = v17.WAYPOINT_DISTANCE
v12.WAYPOINT_SUCCESS_THRESHOLD = v17.WAYPOINT_SUCCESS_THRESHOLD
v12.TARGETING_VERSION = v17.TARGETING_VERSION


if __name__ == "__main__":
    base.main()
