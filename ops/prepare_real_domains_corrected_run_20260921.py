#!/usr/bin/env python3
"""Freeze a corrected-scoring real-domain pilot and its bounded Slurm job.

The request specifies entrypoint, config, GPU time, and explicitly selected data
roots. Source/data are copied before submission. Existing outputs are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rewrite_paths(value, replacements):
    if isinstance(value, dict):
        return {k: rewrite_paths(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(v, replacements) for v in value]
    if isinstance(value, str):
        for source, destination in replacements:
            if value == source or value.startswith(source + "/"):
                return destination + value[len(source):]
    return value


def prepare(request_path: Path, output: Path, submit: bool) -> dict:
    root = Path(__file__).resolve().parents[1]
    request = json.loads(request_path.read_text())
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    entrypoint = request["entrypoint"]
    if entrypoint not in ("train_real_domains_pilot_20260921_v2.py",):
        raise ValueError("unsupported pilot entrypoint")
    minutes = request["time_limit_minutes"]
    if type(minutes) is not int or not 1 <= minutes <= 240:
        raise ValueError("one pilot allocation must be between 1 and 240 minutes")
    if request.get("gpu_count", 1) != 1:
        raise ValueError("this pilot launcher is limited to one GPU")
    partition = request.get("partition", "all")
    if partition not in {"all", "lowprio", "mltheory"}:
        raise ValueError("unsupported pilot partition")
    bundle = output / "bundle"
    bundle.mkdir(parents=True)
    copied = []
    def copy(source: Path, destination: Path):
        expected = digest(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(destination) != expected or digest(source) != expected:
            raise RuntimeError(f"source changed while freezing: {source}")
        copied.append({"source": str(source), "snapshot": str(destination), "sha256": expected})
    sources = list((root / "ops").glob("*.py")) + list((root / "ops").glob("*.c")) + list((root / "src").rglob("*.py"))
    for source in sorted(sources):
        copy(source, bundle / source.relative_to(root))
    copy(root / "ops/repo_env.sh", bundle / "repo_env.sh")
    copy(root / "third_party/testlib/testlib.h", bundle / "testlib/testlib.h")
    replacements = []
    for i, source_name in enumerate(request.get("freeze_data_roots", [])):
        source_root = Path(source_name).resolve()
        if not source_root.is_dir():
            raise NotADirectoryError(source_root)
        destination_root = bundle / "data" / str(i)
        replacements.append((str(source_root), str(destination_root)))
        for source in sorted(p for p in source_root.rglob("*") if p.is_file()):
            copy(source, destination_root / source.relative_to(source_root))
    config = rewrite_paths(request["config"], replacements)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    identity = {
        "schema": "real-domains-frozen-job-20260921-v1", "prepared_at": datetime.now(timezone.utc).isoformat(),
        "request": request, "request_sha256": digest(request_path), "config_sha256": digest(config_path),
        "files": copied, "job_id": None,
        "allocated_gpu_hour_ceiling": minutes / 60,
    }
    identity_path = output / "identity.json"
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    python = root / "var/seed_paper_eval/paper310/bin/python"
    q = shlex.quote
    args = [str(python), str(bundle / "ops" / entrypoint), "--config", str(config_path)]
    if entrypoint.startswith("evaluate_"):
        args += ["--output", str(output / "evaluation.json")]
    else:
        if request.get("arm") not in ("maxrl", "remax"):
            raise ValueError("training request requires maxrl or remax arm")
        args += ["--output", str(output / "training"), "--arm", request["arm"]]
    script = "\n".join([
        "#!/usr/bin/env bash", "set -euo pipefail",
        f"export OAT_ZERO_REPO_ROOT={q(str(root))}",
        f"source {q(str(bundle / 'repo_env.sh'))}",
        f"export OAT_ZERO_SOURCE_ROOT={q(str(bundle / 'src'))}",
        f"export OAT_ZERO_TESTLIB_ROOT={q(str(bundle / 'testlib'))}",
        f"export OAT_ZERO_SANDBOX_SOURCE={q(str(bundle / 'ops/constructive_code_sandbox.c'))}",
        f"export PYTHONPATH={q(str(bundle / 'ops') + ':' + str(bundle / 'src'))}",
        f"export LD_LIBRARY_PATH={q(str(python.parent.parent / 'lib'))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0",
        "export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4",
        f"{q(str(python))} - {q(str(identity_path))} {q(str(config_path))} <<'PYVERIFY'",
        "import hashlib,json,pathlib,sys",
        "identity=json.loads(pathlib.Path(sys.argv[1]).read_text())",
        "def digest(p):",
        "    h=hashlib.sha256()",
        "    with pathlib.Path(p).open('rb') as f:",
        "        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)",
        "    return h.hexdigest()",
        "assert digest(sys.argv[2])==identity['config_sha256'], 'config drift'",
        "for row in identity['files']:",
        "    assert digest(row['snapshot'])==row['sha256'], row['snapshot']",
        "print('Frozen source, data, and config verified.',flush=True)",
        "PYVERIFY",
        "exec " + shlex.join(args), "",
    ])
    script_path = output / "run.slurm"
    script_path.write_text(script)
    submission = ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1",
        "--gres=gpu:a6000:1", f"--cpus-per-task={int(request.get('cpus', 8))}",
        f"--mem={int(request.get('memory_gb', 64))}G", f"--time={minutes}",
        "--account=mltheory", f"--partition={partition}", f"--job-name={request.get('job_name', 'real-domains-pilot')}",
        f"--output={output}/slurm-%j.out", f"--error={output}/slurm-%j.err", str(script_path)]
    (output / "submission_intent.json").write_text(json.dumps({"argv": submission, "authorized_gpu_hour_ceiling": minutes / 60}, indent=2) + "\n")
    if submit:
        proc = subprocess.run(submission, text=True, capture_output=True, check=False)
        receipt = {"argv": submission, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        (output / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if proc.returncode:
            raise RuntimeError(f"sbatch failed: {proc.stderr}")
        job_id = int(proc.stdout.strip().split(";")[0])
        identity["job_id"] = job_id
        # Never rewrite the runnable identity while the job may be reading it.
        # Record the scheduler handle separately after sbatch returns.
        receipt["job_id"] = job_id
        (output / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"output": str(output), "job_id": identity["job_id"], "gpu_hour_ceiling": minutes / 60}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.request, args.output, args.submit), sort_keys=True))
