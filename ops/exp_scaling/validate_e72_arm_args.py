#!/usr/bin/env python3
"""Parse an E72 arm's real launcher arguments offline, before any job is queued.

B1a's first cohort put twenty-five jobs on the cluster that all died in
argument validation seconds after start, because the variant block hardcoded
switches that the launcher's overrides then made inconsistent. Nothing about
that failure needed a GPU to discover: the argument parser and the objective
validator would have rejected it on a login node.

So this reconstructs exactly what would run --- the launcher's environment, the
variant block, and train.sh's argument assembly --- but substitutes a stub for
the training interpreter, captures the argument vector it would have been
called with, and puts that vector through the same parser and the same
validator the learner uses. It never starts a job and never touches a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_e72_b3a_replay_ablation as launcher  # noqa: E402

STUB = """#!/usr/bin/env bash
# Stands in for the training interpreter, but only for the training entry
# point: the launcher also runs helper scripts through the same interpreter to
# resolve evaluation cadence and similar, and stubbing those would both capture
# the wrong argument vector and change the configuration under test. Anything
# that is not the learner is delegated to the real interpreter untouched.
if [ "$1" = "-m" ] && [ "$2" = "oat_drgrpo.train_zero_math" ]; then
  printf '%s\\n' "$@" > "$ARGV_SINK"
  exit 0
fi
exec "$REAL_PYTHON" "$@"
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def captured_argv(root: Path, env: dict[str, str], sink: Path) -> list[str]:
    """Argument vector train.sh would hand the learner for this environment."""
    stub_dir = sink.parent / "stub"
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "python"
    stub.write_text(STUB)
    stub.chmod(0o755)

    child = dict(os.environ)
    child.update(env)
    child["OAT_ZERO_PYTHON"] = str(stub)
    child["ARGV_SINK"] = str(sink)
    child["REAL_PYTHON"] = str(
        root / "var" / "seed_paper_eval" / "paper310" / "bin" / "python"
    )
    # The variant block runs inside run_experiment.sh, which is what maps a
    # variant name onto the objective switches under test.
    result = subprocess.run(
        ["bash", str(root / "ops" / "run_experiment.sh")],
        env=child,
        capture_output=True,
        text=True,
    )
    if not sink.exists():
        raise SystemExit(
            "launcher exited before assembling learner arguments "
            f"(status {result.returncode})\n{result.stdout[-3000:]}\n"
            f"{result.stderr[-3000:]}"
        )
    # Empty lines are preserved: several flags are legitimately passed an empty
    # string, and dropping those turns a valid vector into a parse error.
    return sink.read_text().splitlines()


def validate(argv: list[str]) -> dict[str, object]:
    """Run the learner's own parser and objective validator over one vector."""
    sys.path.insert(0, str(repo_root() / "src"))
    import tyro  # noqa: PLC0415

    from oat_drgrpo.args import ZeroMathArgs, validate_zero_math_args  # noqa: PLC0415

    # train.sh calls `python -m <module> <flags...>`; drop the module selector.
    flags = argv[2:] if argv[:1] == ["-m"] else argv
    parsed = tyro.cli(ZeroMathArgs, args=flags)
    validate_zero_math_args(parsed)
    return {
        "maxent_alpha": float(parsed.maxent_alpha),
        "maxent_dual_target_ratio": float(parsed.maxent_dual_target_ratio),
        "maxent_dual_target_entropy": float(parsed.maxent_dual_target_entropy),
        "semantic_shannon_coef": float(parsed.semantic_shannon_coef),
        "online_canonical_bank_alpha": float(parsed.online_canonical_bank_alpha),
        "online_canonical_replay": bool(parsed.online_canonical_replay),
        "online_canonical_replay_compute_only": bool(
            parsed.online_canonical_replay_compute_only
        ),
        "observe_masked_mean": bool(
            getattr(parsed, "maxent_observe_masked_mean_entropy", False)
        ),
        "maxent_objective": str(getattr(parsed, "maxent_objective", "")),
        "seed": int(parsed.seed),
    }


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(launcher.ARM_SPECS))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "var" / "artifacts" / "e72_frontier_source_runs.json",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="validate this seed only"
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    runs = [
        run
        for run in manifest["runs"]
        if run["arm"] == launcher.REFERENCE_ARM
    ]
    if args.seed is not None:
        runs = [run for run in runs if int(run["seed"]) == args.seed]
    seen: set[str] = set()
    checked = 0
    with tempfile.TemporaryDirectory(prefix="e72-argv-") as scratch:
        for run in runs:
            # One seed per domain exercises every distinct objective
            # configuration; the rest differ only in the seed integer.
            if args.seed is None and run["domain"] in seen:
                continue
            seen.add(str(run["domain"]))
            target = launcher.save_path(root, run, args.arm)
            env = launcher.build_export_vars(root, run, target, args.arm)
            sink = Path(scratch) / f"argv_{run['domain']}_{run['seed']}.txt"
            argv = captured_argv(root, env, sink)
            resolved = validate(argv)
            checked += 1
            print(
                f"  {run['domain']:16} s{run['seed']} ok  "
                + "  ".join(f"{key}={value}" for key, value in resolved.items())
            )
    print(f"[e72-validate] {args.arm}: {checked} configurations parsed and validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
