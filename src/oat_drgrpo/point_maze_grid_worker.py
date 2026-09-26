"""Networkless JSON-lines worker for development PointMaze grid programs."""

from __future__ import annotations

import json
import signal
import sys

from .point_maze_grid import execute_point_grid


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("Point grid execution timed out")


def main() -> None:
    signal.signal(signal.SIGALRM, _timeout)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            signal.setitimer(signal.ITIMER_REAL, 20.0)
            validation, execution = execute_point_grid(
                request["candidate"], request["spec"]
            )
            payload = {
                "valid": True,
                "canonical_key": validation.canonical_key,
                "directed_gates": list(validation.directed_gates),
                "action_tokens": list(validation.action_tokens),
                "simulator_steps": validation.simulator_steps,
                "execution": execution,
            }
        except Exception as error:
            payload = {
                "valid": False,
                "error": type(error).__name__,
                "message": str(error),
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
        sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
