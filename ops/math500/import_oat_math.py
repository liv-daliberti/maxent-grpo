#!/usr/bin/env python3
"""Import the exact MATH artifacts released with the Dr.GRPO paper.

The upstream repository commits Hugging Face Arrow datasets directly.  This
importer pins the initial paper-code commit, verifies every copied byte, and
materializes the two roots expected by this repository's OAT runner:

* ``train``: the 8,523-row MATH level-3--5 DatasetDict;
* ``eval``: a one-split DatasetDict whose ``math`` split is MATH-500.

MATH-500 remains evaluation-only.  No row is rewritten or filtered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_URL = "https://github.com/sail-sg/understand-r1-zero.git"
UPSTREAM_COMMIT = "559bcfd7a50727e7ed97f06a586da2c97236f496"
DEFAULT_OUTPUT_ROOT = ROOT / "var/data/oat_drgrpo_math_paper"

EXPECTED_FILES = {
    "datasets/evaluation_suite/math/data-00000-of-00001.arrow": (
        "d383d13c807e2904d0db6f8d98496a0574c0a1d51f40331b920c849cf226ef5a"
    ),
    "datasets/evaluation_suite/math/dataset_info.json": (
        "7585ceec6a09b310c377121f7480c73e23f3a27a4f1ef9aa2dd81340fc226cd9"
    ),
    "datasets/evaluation_suite/math/state.json": (
        "588bd2e529028b1e6ba0727e341bf989ef249a73c5cbeffbbce1106e71144cf9"
    ),
    "datasets/train/math_lvl3to5_8k/dataset_dict.json": (
        "b8bb47a2de28fcab12a8df528dd714a88959465737fb234bacbc2dfd0c324ba8"
    ),
    "datasets/train/math_lvl3to5_8k/train/data-00000-of-00001.arrow": (
        "bf8e6fbf72b17c88b35e2c60eec8a677ac31377f278bac033792a765ad5ec1e1"
    ),
    "datasets/train/math_lvl3to5_8k/train/dataset_info.json": (
        "36145e620c32246f703d4b45f1236a87d5ec228b734b8319f4bc01020248761c"
    ),
    "datasets/train/math_lvl3to5_8k/train/state.json": (
        "4f5c8f051843297d874ba467ac305793a44ed6d82bc7c575761a6d10872ea802"
    ),
    "datasets/train/math_lvl3to5_8k/eval/data-00000-of-00001.arrow": (
        "86da05d3062d8b7847b7aeec7ca6d208f386a85281877212f91637387564e487"
    ),
    "datasets/train/math_lvl3to5_8k/eval/dataset_info.json": (
        "7557a2db502054b8d61f0a5ff0cbe5a230401e85d701a015acb3041a23f40f0c"
    ),
    "datasets/train/math_lvl3to5_8k/eval/state.json": (
        "489f56de672d842a61c84f928aeb84ac92c8b9f5b76538dfdc811c93a40027cf"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_upstream(checkout: Path) -> None:
    for relative, expected in EXPECTED_FILES.items():
        path = checkout / relative
        if not path.is_file():
            raise RuntimeError(f"upstream artifact is missing: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"upstream checksum mismatch for {relative}: "
                f"expected {expected}, observed {observed}"
            )


def verify_materialized_source_bytes(root: Path) -> None:
    """Verify that runtime Arrow/metadata files remain byte-exact."""

    prefixes = {
        "datasets/evaluation_suite/math/": "eval/math/",
        "datasets/train/math_lvl3to5_8k/": "train/",
    }
    for source_relative, expected in EXPECTED_FILES.items():
        runtime_relative = None
        for source_prefix, runtime_prefix in prefixes.items():
            if source_relative.startswith(source_prefix):
                runtime_relative = runtime_prefix + source_relative.removeprefix(
                    source_prefix
                )
                break
        if runtime_relative is None:
            raise AssertionError(f"unmapped source artifact: {source_relative}")
        path = root / runtime_relative
        if not path.is_file():
            raise RuntimeError(f"materialized artifact is missing: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"materialized checksum mismatch for {path}: "
                f"expected {expected}, observed {observed}"
            )


def read_arrow_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:  # pragma: no cover - production preflight guard
        raise RuntimeError("pyarrow is required to audit the imported rows") from exc
    with pa.memory_map(str(path), "r") as source:
        return ipc.open_stream(source).read_all().to_pylist()


def audit_materialized(root: Path) -> dict[str, Any]:
    verify_materialized_source_bytes(root)
    train_path = root / "train/train/data-00000-of-00001.arrow"
    eval_path = root / "eval/math/data-00000-of-00001.arrow"
    train_rows = read_arrow_rows(train_path)
    eval_rows = read_arrow_rows(eval_path)
    train_problems = {str(row["problem"]).strip() for row in train_rows}
    eval_problems = {str(row["problem"]).strip() for row in eval_rows}
    blank_train_answers = sum(not str(row["answer"]) for row in train_rows)
    duplicate_train_problems = len(train_rows) - len(train_problems)
    overlap = train_problems & eval_problems
    if len(train_rows) != 8523:
        raise RuntimeError(f"expected 8,523 training rows; found {len(train_rows)}")
    if len(eval_rows) != 500:
        raise RuntimeError(f"expected 500 MATH-500 rows; found {len(eval_rows)}")
    if overlap:
        raise RuntimeError(
            f"MATH-500 leakage: {len(overlap)} exact problems occur in training"
        )
    # These two quirks are present in the byte-exact paper artifact.  Record
    # them rather than silently changing the training distribution.
    if blank_train_answers != 2 or duplicate_train_problems != 1:
        raise RuntimeError(
            "the pinned training-data quirks changed: expected two blank "
            "answers and one duplicate problem"
        )
    return {
        "train_rows": len(train_rows),
        "train_unique_problems": len(train_problems),
        "train_blank_answers": blank_train_answers,
        "train_duplicate_problems": duplicate_train_problems,
        "math500_rows": len(eval_rows),
        "math500_unique_problems": len(eval_problems),
        "exact_problem_overlap": len(overlap),
    }


def materialize(
    checkout: Path, output_root: Path, *, overwrite: bool
) -> dict[str, Any]:
    verify_upstream(checkout)
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"output already exists: {output_root}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        shutil.copytree(
            checkout / "datasets/train/math_lvl3to5_8k",
            staging / "train",
        )
        shutil.copytree(
            checkout / "datasets/evaluation_suite/math",
            staging / "eval/math",
        )
        (staging / "eval/dataset_dict.json").write_text(
            json.dumps({"splits": ["math"]}), encoding="utf-8"
        )
        license_path = checkout / "LICENSE.txt"
        if license_path.is_file():
            shutil.copy2(license_path, staging / "UPSTREAM_LICENSE.txt")

        audit = audit_materialized(staging)
        manifest = {
            "schema": "oat_drgrpo_math_import_v1",
            "source_repository": UPSTREAM_URL,
            "source_commit": UPSTREAM_COMMIT,
            "source_note": (
                "The dataset files are unchanged between the initial paper-code "
                "commit and the repository head audited on 2026-07-19."
            ),
            "role": {
                "train": "MATH level 3-5 RL prompts",
                "eval/math": "held-out MATH-500; never a training split",
            },
            "source_sha256": EXPECTED_FILES,
            "audit": audit,
        }
        (staging / "IMPORT_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def clone_pinned(destination: Path) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            UPSTREAM_URL,
            str(destination),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", UPSTREAM_COMMIT],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source-checkout",
        type=Path,
        help="existing checkout at the pinned commit (otherwise clone it)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="verify an existing output root without cloning or rewriting it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        audit = audit_materialized(args.output_root.resolve())
        print(json.dumps(audit, sort_keys=True))
        print(f"verified {args.output_root.resolve()}")
        return
    if args.source_checkout is not None:
        checkout = args.source_checkout.resolve()
        observed_commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed_commit != UPSTREAM_COMMIT:
            raise RuntimeError(
                f"source checkout must be at {UPSTREAM_COMMIT}; found {observed_commit}"
            )
        manifest = materialize(checkout, args.output_root, overwrite=args.overwrite)
    else:
        with tempfile.TemporaryDirectory(prefix="oat-drgrpo-math-") as temporary:
            checkout = Path(temporary) / "source"
            clone_pinned(checkout)
            manifest = materialize(checkout, args.output_root, overwrite=args.overwrite)
    print(json.dumps(manifest["audit"], sort_keys=True))
    print(f"wrote {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
