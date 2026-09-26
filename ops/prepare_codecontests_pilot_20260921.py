#!/usr/bin/env python3
"""Freeze the authorized capability pilot; does not submit a job."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "var/artifacts/codecontests_pilot_20260921"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(path: Path) -> str:
    records = [(p.relative_to(path).as_posix(), digest(p))
               for p in sorted(path.rglob("*")) if p.is_file()]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    bundle = ART / "bundle"
    if bundle.exists():
        raise FileExistsError("pilot bundle already exists; preserve it")
    copies: list[tuple[Path, Path]] = []
    for source in sorted((ROOT / "ops").glob("*constructive*.py")):
        copies.append((source, bundle / "ops" / source.name))
    for name in ("__init__.py", "constructive_code.py", "constructive_code_adapters.py",
                 "constructive_code_sandbox.py"):
        copies.append((ROOT / "src/oat_drgrpo" / name, bundle / "src/oat_drgrpo" / name))
    copies.extend([
        (ROOT / "ops/constructive_code_sandbox.c", bundle / "ops/constructive_code_sandbox.c"),
        (ROOT / "ops/repo_env.sh", bundle / "repo_env.sh"),
        (ROOT / "ops/slurm/codecontests_pilot_20260921.slurm", bundle / "pilot.slurm"),
        (ROOT / "third_party/testlib/testlib.h", bundle / "testlib/testlib.h"),
        (ROOT / "var/artifacts/constructive_code_candidate_source_index.json", bundle / "inputs/candidate_source_index.json"),
        (ROOT / "paper/preregistration/constructive_code_executable_slate_v5_20260730.md",
         bundle / "inputs/constructive_code_executable_slate_v5_20260730.md"),
    ])
    for name in ("constructive_code_v6_gate_audit.json", "constructive_code_v6_gate_identity.json",
                 "constructive_code_v5_replays.jsonl"):
        copies.append((ROOT / "var/artifacts" / name, bundle / "inputs" / name))
    for name in ("codecontests_pilot_protocol_20260921.md", "codecontests_pilot_prompt_overlay_20260921.json",
                 "codecontests_pilot_citations_20260921.md"):
        copies.append((ROOT / "docs" / name, bundle / "docs" / name))
    for source, _ in copies:
        if not source.is_file():
            raise FileNotFoundError(source)
    records = []
    for source, target in copies:
        expected = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != expected or digest(source) != expected:
            raise RuntimeError(f"source changed during snapshot: {source}")
        records.append({"source": str(source), "snapshot": str(target), "sha256": expected})
    payload = {
        "schema": "codecontests-pilot-20260921-identity-v1",
        "stage": "prepared_before_model_sampling",
        "job_id": None,
        "root": str(ROOT), "artifact_root": str(ART), "bundle": str(bundle),
        "source_hash": tree_digest(bundle / "src"),
        "execution_hash": tree_digest(bundle),
        "files": records,
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "model_revision": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
        "historical_source_snapshot_deleted": True,
        "historical_constructive_sources_match_surviving_e122_snapshot": True,
        "development_problem_ids": ["359_B", "988_A", "1399_D"],
        "evaluation_rows_loaded": False,
        "stage_gpu_hour_ceiling": 1,
        "total_pilot_gpu_hour_ceiling": 12,
        "sampling": {"samples_per_problem": 64, "seed": 77101,
                     "temperature": 1.0, "top_p": 1.0, "max_tokens": 1024},
    }
    (ART / "identity.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("artifact_root", "bundle", "source_hash", "execution_hash")}))


if __name__ == "__main__":
    main()
