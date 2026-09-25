#!/usr/bin/env python3
"""CPU controls for the versioned seeded candidate launcher; never pilot results.

The fixed seed controls Python's global random stream and string hash only.
Explicit reseeding, independent RNGs, clocks and OS entropy remain unsupported
sources of execution nondeterminism and retain the production replay hard stop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ops/constructive_code_sandbox_seeded_20260922.c"
ORIGINAL = ROOT / "ops/constructive_code_sandbox.c"
SEED_STREAM = [0.8444218515250481, 0.7579544029403025, 0.420571580830845]
SCHEMA = "constructive-code-seeded-launcher-controls-20260922-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_kernel_source_unchanged(source: Path, original: Path) -> None:
    """Require an exact restoration of original source after two approved edits."""
    revised = source.read_text()
    start = revised.index("/* Versioned execution contract:")
    end = revised.index("int main(", start)
    revised = revised[:start] + revised[end:]
    old_arguments = '        "-I",\n        "-B",\n'
    new_arguments = ('        "-s",\n        "-S",\n        "-B",\n'
                     '        "-c",\n        seeded_python_bootstrap,\n        python_home,\n')
    if revised.count(new_arguments) != 1:
        raise ValueError("unexpected seeded interpreter argument changes")
    if revised.replace(new_arguments, old_arguments) != original.read_text():
        raise ValueError("kernel isolation, limits, environment or other original source changed")


class SandboxProbe:
    def __init__(self, launcher: Path, runtime_root: Path):
        self.launcher = launcher.resolve()
        self.runtime_root = runtime_root.resolve()

    def run(self, code: str, *, stdin: str = "", hostile: bool = False,
            inherited_fd: int | None = None) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="seeded-launcher-control-") as raw:
            work = Path(raw)
            candidate = work / "candidate.py"
            candidate.write_text(code)
            before = sha256(candidate)
            environment = os.environ.copy()
            if hostile:
                for name in ("random.py", "runpy.py", "sitecustomize.py", "usercustomize.py", "json.py"):
                    (work / name).write_text("raise RuntimeError('UNTRUSTED MODULE LOADED')\n")
                environment.update({"PYTHONPATH": str(work), "PYTHONHOME": str(work),
                                    "PYTHONSTARTUP": str(work / "random.py"),
                                    "PYTHONHASHSEED": "123", "PYTHONINSPECT": "1",
                                    "LD_PRELOAD": "", "SEEDED_PROBE_SECRET": "not-visible"})
            completed = subprocess.run(
                [str(self.launcher), str(self.runtime_root), str(work), candidate.name,
                 "3", "536870912", "1048576", "64"],
                input=stdin, text=True, capture_output=True, timeout=10,
                env=environment, pass_fds=(() if inherited_fd is None else (inherited_fd,)),
            )
            if sha256(candidate) != before:
                raise AssertionError("candidate bytes changed during execution")
            return {"returncode": completed.returncode, "stdout": completed.stdout,
                    "stderr": completed.stderr, "candidate_sha256": before,
                    "candidate_path": str(candidate)}

    def value(self, code: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        row = self.run(code, **kwargs)
        if row["returncode"] != 0:
            raise AssertionError(row)
        return json.loads(row["stdout"]), row


def replay_control(first_key: str, second_key: str) -> dict[str, Any]:
    """Feed execution-derived control keys to the unchanged production replay gate.

    This fixture isolates replay-gate behavior; it is not a dataset checker run.
    """
    import evaluate_constructive_code_pilot_20260921 as pilot

    task = SimpleNamespace(problem_id="CONTROL_ONLY", problem_key="control", suite_id="control",
                           suite_sha256="suitehash", checker_sha256="checkerhash", tests=[1])
    candidate = {"row_index": 0, "sample_index": 0, "request_seed": 0,
                 "emitted_text_sha256": "sourcehash", "executed_source_sha256": "sourcehash",
                 "fence_stripped": False, "token_count": 1, "finish_reason": "stop",
                 "code": "# Control fixture; no dataset result\n"}
    def replay(key: str) -> dict[str, Any]:
        return {"source_problem_id": task.problem_id, "problem_key": task.problem_key,
                "suite_id": task.suite_id, "suite_sha256": task.suite_sha256,
                "checker_sha256": task.checker_sha256, "submission_sha256": "sourcehash",
                "released_checker_accepted": True, "wrapper_accepted": True, "behavior_key": key,
                "execution": {"candidate_invocation_wall_seconds": [0.1], "checker_wall_seconds": 0.1,
                              "executed_tests": 1, "suite_tests": 1, "first_failure": None}}
    replies = iter([replay(first_key), replay(second_key)])
    base = SimpleNamespace(Submission=lambda *args: args, _replay_submission=lambda **kwargs: next(replies))
    args = SimpleNamespace(launcher=None, runtime_root=None, scratch_root=None)
    return pilot._verify_candidate(base, args, candidate, task)


def run_controls(probe: SandboxProbe) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    code = ("import sys,json,random\n"
            "print(json.dumps({'version':sys.version.split()[0], 'hash':hash('fixed_probe'), "
            "'random':[random.random() for _ in range(3)], 'hash_randomization':sys.flags.hash_randomization, "
            "'ignore_environment':sys.flags.ignore_environment, 'no_site':sys.flags.no_site, "
            "'no_user_site':sys.flags.no_user_site, 'bytecode':sys.flags.dont_write_bytecode, "
            "'path':sys.path, 'name':__name__, 'argv':sys.argv, 'file':__file__}))\n")
    repeats = [probe.value(code, hostile=True) for _ in range(5)]
    expected_path = [str(probe.runtime_root / "usr/local/lib/python3.10"),
                     str(probe.runtime_root / "usr/local/lib/python3.10/lib-dynload")]
    for value, row in repeats:
        assert value["version"] == "3.10.20", value
        assert value["hash"] == 7080782428778635838, value
        assert value["random"] == SEED_STREAM, value
        assert [value[k] for k in ("hash_randomization", "ignore_environment", "no_site", "no_user_site", "bytecode")] == [0, 0, 1, 1, 1]
        assert value["path"] == expected_path, value
        assert value["name"] == "__main__" and value["argv"] == [row["candidate_path"]]
        assert value["file"] == row["candidate_path"]
    observations["seed_hash_module_isolation_and_language"] = [v for v, _ in repeats]

    neutral = [probe.value("import random,json\n" + assignment + "\nprint(json.dumps([random.random() for _ in range(3)]))\n")[0]
               for assignment in ("x=1", "renamed_variable=1")]
    assert neutral == [SEED_STREAM, SEED_STREAM], neutral
    observations["neutral_source_edits_keep_seed"] = neutral

    io = probe.run("import sys\nsys.stdout.write(sys.stdin.read())\nraise SystemExit(7)\n", stdin="hello\nλ 42\n")
    assert io["returncode"] == 7 and io["stdout"] == "hello\nλ 42\n" and not io["stderr"], io
    observations["stdin_stdout_systemexit"] = io

    env, _ = probe.value("import os,json\nprint(json.dumps(dict(os.environ)))\n", hostile=True)
    assert set(env) == {"PYTHONHOME", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "LC_ALL", "LANG", "TZ", "TMPDIR", "PATH"}, env
    assert env["PYTHONHASHSEED"] == "0" and env["PATH"] == "", env
    observations["clean_environment"] = env

    with tempfile.TemporaryDirectory(prefix="seeded-outside-control-") as raw:
        outside = Path(raw) / "sentinel.txt"
        outside.write_text("CONTROL_ONLY")
        security = "import os,socket,resource,json,errno\nchecks={}\n"
        security += "def denied(label, action):\n try: action()\n except OSError as error: checks[label]=(error.errno==errno.EACCES)\n else: checks[label]=False\n"
        security += f"denied('outside_read', lambda:open({str(outside)!r}).read())\n"
        security += f"denied('outside_write', lambda:open({str(outside)!r},'w'))\n"
        security += "denied('network',lambda:socket.socket())\ndenied('fork',lambda:os.fork())\n"
        security += "denied('host_exec',lambda:os.execv('/bin/true',['true']))\n"
        security += "denied('cross_process_signal',lambda:os.kill(os.getppid(),0))\n"
        security += "denied('cross_process_limits',lambda:resource.prlimit(os.getppid(),resource.RLIMIT_NOFILE))\n"
        security += "open('local.txt','w').write('ok')\nchecks['workdir_write']=open('local.txt').read()=='ok'\nprint(json.dumps(checks))\n"
        value, _ = probe.value(security)
        assert value and all(value.values()), value
        assert outside.read_text() == "CONTROL_ONLY"
        observations["kernel_isolation"] = value
        with outside.open() as handle:
            fd = handle.fileno()
            fd_code = (f"import os,json,errno\ntry: os.fstat({fd})\n"
                       "except OSError as error: print(json.dumps(error.errno==errno.EBADF))\n"
                       "else: print('false')\n")
            value, _ = probe.value(fd_code, inherited_fd=fd)
            assert value is True
            observations["inherited_fd_closed"] = value

    limits, _ = probe.value("import json,resource\nprint(json.dumps({name:list(resource.getrlimit(getattr(resource,name))) for name in ['RLIMIT_CPU','RLIMIT_AS','RLIMIT_FSIZE','RLIMIT_NOFILE','RLIMIT_NPROC','RLIMIT_CORE']}))\n")
    assert limits == {"RLIMIT_CPU": [3, 3], "RLIMIT_AS": [536870912] * 2,
                      "RLIMIT_FSIZE": [1048576] * 2, "RLIMIT_NOFILE": [64] * 2,
                      "RLIMIT_NPROC": [1, 1], "RLIMIT_CORE": [0, 0]}, limits
    observations["resource_limits"] = limits

    entropy = ("import os,random,time,json\na=random.Random().getrandbits(256)\n"
               "random.seed(None)\nb=random.getrandbits(256)\n"
               "print(json.dumps({'independent_random':a,'reseed_none':b,"
               "'system_random':random.SystemRandom().getrandbits(256),'urandom':os.urandom(32).hex(),"
               "'clock':time.time_ns(),'pid':os.getpid()}))\n")
    residual = [probe.value(entropy)[0] for _ in range(3)]
    for key in residual[0]:
        assert len({row[key] for row in residual}) > 1, (key, residual)
    observations["residual_entropy_is_uncontrolled"] = residual
    stable_gate = replay_control(json.dumps(SEED_STREAM), json.dumps(SEED_STREAM))
    changed_gate = replay_control(json.dumps(residual[0], sort_keys=True), json.dumps(residual[1], sort_keys=True))
    assert stable_gate["accepted"] and not stable_gate["hard_violations"], stable_gate
    assert not changed_gate["accepted"] and changed_gate["canonical_key"] is None, changed_gate
    assert any("changed canonical mode" in v for v in changed_gate["hard_violations"]), changed_gate
    observations["production_replay_gate_control_fixture"] = {"stable": stable_gate, "changed": changed_gate}
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--original-source", type=Path, default=ORIGINAL)
    args = parser.parse_args()
    check_kernel_source_unchanged(args.source, args.original_source)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    binary = args.output_dir / "constructive_code_sandbox_seeded"
    command = ["cc", "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(binary), str(args.source)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    receipt: dict[str, Any] = {"schema": SCHEMA, "purpose": "CPU controls only; not pilot outcomes",
                               "source_path": str(args.source.resolve()), "source_sha256": sha256(args.source),
                               "original_source_sha256": sha256(args.original_source),
                               "probe_sha256": sha256(Path(__file__)), "binary_sha256": sha256(binary),
                               "runtime_root": str(args.runtime_root.resolve()), "compiler_argv": command,
                               "contract": {"python_hash_seed": 0, "stdlib_global_random_seed": 0,
                                            "seed_scope": "every candidate/input invocation; task/source independent",
                                            "candidate_bytes": "unchanged", "stability_recheck": "unchanged hard failure",
                                            "residual_entropy": "unsupported; finite replay detects observed disagreements only"}}
    try:
        receipt["observations"] = run_controls(SandboxProbe(binary, args.runtime_root))
        receipt["status"] = "pass"
    except Exception as exc:
        receipt["status"] = "fail"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(args.output_dir / "receipt.json")}))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
