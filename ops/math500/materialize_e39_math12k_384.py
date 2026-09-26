#!/usr/bin/env python3
"""Materialize E39's frozen MATH12K-384 train and MATH-500 eval roots.

The training population is exactly source rows 0..383, in source order, from
the pinned SEED-GRPO MATH12K Arrow artifact.  The evaluation population is the
complete, already imported MATH-500 split.  No row is rewritten or filtered.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence
import unicodedata


ROOT = Path(__file__).resolve().parents[2]

SOURCE_CHECKOUT = ROOT / "var/seed_paper_eval/external/SEED-GRPO"
TRAIN_SOURCE_ROOT = SOURCE_CHECKOUT / "datasets/train/math_12k"
EVAL_IMPORT_ROOT = ROOT / "var/data/oat_drgrpo_math_paper"
EVAL_SOURCE_ROOT = EVAL_IMPORT_ROOT / "eval"
TOKENIZER_ROOT = (
    ROOT
    / "var/cache/huggingface/transformers"
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
DEFAULT_OUTPUT_ROOT = ROOT / "var/data/math12k_384_math500"
MANIFEST_NAME = "MATERIALIZATION_MANIFEST.json"

SOURCE_REPOSITORY = "https://github.com/Dreamer312/SEED-GRPO.git"
SOURCE_CHECKOUT_COMMIT = "325cb1a20bb60f8efd4cdc77a1565491c29fd289"
SOURCE_CHECKOUT_PARENT = "e8183f9b3e540afb29846b255e94b4397c2c9f75"
SOURCE_DATASET_COMMIT = "ffa64bfdeaeda20029e831a749714f68079d7f9c"
SOURCE_DATASET_PARENT = "af88cfe3b3aa22887d9a1385e7844ee0a14cf20c"
SOURCE_DATASET_RELATIVE = "datasets/train/math_12k"

MATH500_SOURCE_REPOSITORY = "https://github.com/sail-sg/understand-r1-zero.git"
MATH500_SOURCE_COMMIT = "559bcfd7a50727e7ed97f06a586da2c97236f496"
EVAL_IMPORT_MANIFEST_SHA256 = (
    "2fe5f5461be2fed5ec2a0ed5d400873ad02dc7684d3c4793afa54bd9cc4f5202"
)

TRAIN_SOURCE_FILES = {
    "dataset_dict.json": (
        "c172eebfc28c1400d6be4338ce7d00191507ffb4ae64c315f039585c894df5b7"
    ),
    "train/data-00000-of-00001.arrow": (
        "125db2efb27057f37b383d44110f3b7d49a1f55636b01f737ac5fd8cc27cf829"
    ),
    "train/dataset_info.json": (
        "294b7b9495bcaaf5950aff743a129940a85da12e7e465915be1bf83e0d5ae68b"
    ),
    "train/state.json": (
        "242bfceef66b8b1136441339dfea17b5a00b5464703215f9fb606c5ee6db9f18"
    ),
}

EVAL_SOURCE_FILES = {
    "dataset_dict.json": (
        "e98632f80c86d8d5a9ad7aa2d6b47abae15003c2c60e9d1ea09fb097d4029dca"
    ),
    "math/data-00000-of-00001.arrow": (
        "d383d13c807e2904d0db6f8d98496a0574c0a1d51f40331b920c849cf226ef5a"
    ),
    "math/dataset_info.json": (
        "7585ceec6a09b310c377121f7480c73e23f3a27a4f1ef9aa2dd81340fc226cd9"
    ),
    "math/state.json": (
        "588bd2e529028b1e6ba0727e341bf989ef249a73c5cbeffbbce1106e71144cf9"
    ),
}

TOKENIZER_FILES = {
    "tokenizer.json": (
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    ),
    "tokenizer_config.json": (
        "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"
    ),
    "merges.txt": (
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
    ),
    "vocab.json": (
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
    ),
}

TRAIN_SOURCE_ROWS = 12_000
TRAIN_SELECTED_ROWS = 384
EVAL_ROWS = 500
PROMPT_MAX_LENGTH = 1_024
TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

TRAIN_COLUMNS = (
    "problem",
    "solution",
    "answer",
    "subject",
    "level",
    "unique_id",
    "gold_solution_steps",
)
EVAL_COLUMNS = ("problem", "answer", "difficulty")

EXPECTED_SUBJECT_COUNTS = {
    "Algebra": 72,
    "Counting & Probability": 32,
    "Geometry": 46,
    "Intermediate Algebra": 81,
    "Number Theory": 46,
    "Prealgebra": 62,
    "Precalculus": 45,
}
EXPECTED_LEVEL_COUNTS = {"1": 29, "2": 67, "3": 77, "4": 87, "5": 124}

TRAIN_ORDERED_ROW_HASH = (
    "051baa5571a1865518ef200c414178f1e50decea261bcd46ec0328e5837c0f36"
)
TRAIN_ORDERED_PROBLEM_HASH = (
    "cd3da2886c255add064b5c93304ca0d33abd268b1b2270d7b08327e9f8a0aab8"
)
EVAL_ORDERED_ROW_HASH = (
    "1576fd11df21dc705a7c85000f232031212225cd9c00520faa26f6bdfc751166"
)
EVAL_ORDERED_PROBLEM_HASH = (
    "4d1ea48faf616bb7156a7979418560dd91a82a252e9ae895131bc414fb7469ab"
)
TRAIN_PROMPT_LENGTH_HASH = (
    "9a44fc9731d52e13b854f1f4540ec73947a9676a646189899b72427866c59905"
)

QWEN_MATH_PREFIX = (
    "<|im_start|>system\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN_MATH_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


class MaterializationError(RuntimeError):
    """Raised when a frozen E39 data identity no longer holds."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _text_sha256(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _verify_files(
    root: Path, expected: dict[str, str], *, label: str
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = root / relative
        if not path.is_file():
            raise MaterializationError(f"{label} file is missing: {path}")
        observed_hash = sha256(path)
        if observed_hash != expected_hash:
            raise MaterializationError(
                f"{label} hash mismatch for {relative}: "
                f"expected={expected_hash} observed={observed_hash}"
            )
        observed[relative] = observed_hash
    return observed


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MaterializationError(
            f"could not verify source Git identity: git {' '.join(arguments)}"
        ) from exc
    return result.stdout.strip()


def verify_source_git(checkout: Path = SOURCE_CHECKOUT) -> dict[str, str]:
    """Verify both the checkout and the commit that introduced MATH12K."""

    if not (checkout / ".git").exists():
        raise MaterializationError(f"source checkout has no .git directory: {checkout}")
    observed_remote = _git_output(checkout, "config", "--get", "remote.origin.url")
    if observed_remote != SOURCE_REPOSITORY:
        raise MaterializationError(
            f"source remote mismatch: expected={SOURCE_REPOSITORY} "
            f"observed={observed_remote}"
        )
    observed_head = _git_output(checkout, "rev-parse", "HEAD")
    observed_head_parent = _git_output(checkout, "rev-parse", "HEAD^")
    observed_dataset_parent = _git_output(
        checkout, "rev-parse", f"{SOURCE_DATASET_COMMIT}^"
    )
    expected_pairs = {
        "checkout_commit": (SOURCE_CHECKOUT_COMMIT, observed_head),
        "checkout_parent": (SOURCE_CHECKOUT_PARENT, observed_head_parent),
        "dataset_commit_parent": (
            SOURCE_DATASET_PARENT,
            observed_dataset_parent,
        ),
    }
    for name, (expected, observed) in expected_pairs.items():
        if observed != expected:
            raise MaterializationError(
                f"source {name} mismatch: expected={expected} observed={observed}"
            )
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "merge-base",
            "--is-ancestor",
            SOURCE_DATASET_COMMIT,
            SOURCE_CHECKOUT_COMMIT,
        ],
        check=False,
    )
    if ancestry.returncode != 0:
        raise MaterializationError(
            "the pinned MATH12K data commit is not an ancestor of the checkout"
        )
    unchanged = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "diff",
            "--quiet",
            SOURCE_DATASET_COMMIT,
            SOURCE_CHECKOUT_COMMIT,
            "--",
            SOURCE_DATASET_RELATIVE,
        ],
        check=False,
    )
    if unchanged.returncode != 0:
        raise MaterializationError(
            "MATH12K tracked files changed after the pinned data commit"
        )
    return {
        "repository": observed_remote,
        "checkout_commit": observed_head,
        "checkout_parent": observed_head_parent,
        "dataset_commit": SOURCE_DATASET_COMMIT,
        "dataset_commit_parent": observed_dataset_parent,
    }


def verify_eval_import_manifest(
    import_root: Path = EVAL_IMPORT_ROOT,
) -> dict[str, Any]:
    manifest_path = import_root / "IMPORT_MANIFEST.json"
    if not manifest_path.is_file():
        raise MaterializationError(
            f"MATH-500 import manifest is missing: {manifest_path}"
        )
    observed_hash = sha256(manifest_path)
    if observed_hash != EVAL_IMPORT_MANIFEST_SHA256:
        raise MaterializationError(
            "MATH-500 import manifest hash mismatch: "
            f"expected={EVAL_IMPORT_MANIFEST_SHA256} observed={observed_hash}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError("MATH-500 import manifest is invalid") from exc
    expected = {
        "source_repository": MATH500_SOURCE_REPOSITORY,
        "source_commit": MATH500_SOURCE_COMMIT,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise MaterializationError(
                f"MATH-500 import {key} mismatch: "
                f"expected={value!r} observed={manifest.get(key)!r}"
            )
    if manifest.get("audit", {}).get("math500_rows") != EVAL_ROWS:
        raise MaterializationError("MATH-500 import manifest row count drifted")
    return {
        **expected,
        "import_manifest": _repo_relative(manifest_path),
        "import_manifest_sha256": observed_hash,
    }


def apply_qwen_math_template(problem: str) -> str:
    return QWEN_MATH_PREFIX + str(problem) + QWEN_MATH_SUFFIX


def normalize_problem(problem: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(problem))
    return " ".join(normalized.split()).casefold()


def _row_identity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    row_hashes = [_json_sha256(row) for row in rows]
    problem_hashes = [_text_sha256(row["problem"]) for row in rows]
    return {
        "ordered_row_sha256": _json_sha256(row_hashes),
        "ordered_problem_sha256": _json_sha256(problem_hashes),
        "row_sha256": row_hashes,
        "problem_sha256": problem_hashes,
    }


def _load_dataset_roots(
    train_source_root: Path, eval_source_root: Path
) -> tuple[Any, Any]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise MaterializationError("datasets is required to materialize E39") from exc
    train_root = load_from_disk(str(train_source_root))
    eval_root = load_from_disk(str(eval_source_root))
    if not isinstance(train_root, DatasetDict) or set(train_root) != {"train"}:
        raise MaterializationError(
            "MATH12K source must be a DatasetDict containing only train"
        )
    if not isinstance(eval_root, DatasetDict) or set(eval_root) != {"math"}:
        raise MaterializationError(
            "MATH-500 source must be a DatasetDict containing only math"
        )
    return train_root["train"], eval_root["math"]


def _prompt_length_audit(
    train_rows: Sequence[dict[str, Any]],
    tokenizer_root: Path,
) -> dict[str, Any]:
    _verify_files(tokenizer_root, TOKENIZER_FILES, label="tokenizer")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise MaterializationError("transformers is required to audit E39") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), local_files_only=True
    )
    lengths = [
        len(
            tokenizer(
                apply_qwen_math_template(str(row["problem"])),
                add_special_tokens=False,
            )["input_ids"]
        )
        for row in train_rows
    ]
    ordered_hash = _json_sha256(lengths)
    if ordered_hash != TRAIN_PROMPT_LENGTH_HASH:
        raise MaterializationError(
            "pinned qwen_math prompt-token lengths drifted: "
            f"expected={TRAIN_PROMPT_LENGTH_HASH} observed={ordered_hash}"
        )
    if not lengths or max(lengths) > PROMPT_MAX_LENGTH:
        raise MaterializationError(
            f"MATH12K-384 has a qwen_math prompt over {PROMPT_MAX_LENGTH} tokens"
        )
    return {
        "template": "qwen_math",
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_files_sha256": dict(TOKENIZER_FILES),
        "prompt_max_length": PROMPT_MAX_LENGTH,
        "minimum_tokens": min(lengths),
        "maximum_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "ordered_token_lengths_sha256": ordered_hash,
        "token_lengths": lengths,
    }


def _validate_rows(
    train_dataset: Any,
    eval_dataset: Any,
    *,
    tokenizer_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(train_dataset) != TRAIN_SOURCE_ROWS:
        raise MaterializationError(
            f"expected {TRAIN_SOURCE_ROWS} MATH12K rows; found {len(train_dataset)}"
        )
    if tuple(train_dataset.column_names) != TRAIN_COLUMNS:
        raise MaterializationError(
            f"MATH12K schema drifted: {train_dataset.column_names!r}"
        )
    if len(eval_dataset) != EVAL_ROWS:
        raise MaterializationError(
            f"expected {EVAL_ROWS} MATH-500 rows; found {len(eval_dataset)}"
        )
    if tuple(eval_dataset.column_names) != EVAL_COLUMNS:
        raise MaterializationError(
            f"MATH-500 schema drifted: {eval_dataset.column_names!r}"
        )

    train_rows = [
        dict(row)
        for row in train_dataset.select(range(TRAIN_SELECTED_ROWS))
    ]
    eval_rows = [dict(row) for row in eval_dataset]
    train_identity = _row_identity(train_rows)
    eval_identity = _row_identity(eval_rows)
    expected_identity = {
        "MATH12K-384 ordered rows": (
            TRAIN_ORDERED_ROW_HASH,
            train_identity["ordered_row_sha256"],
        ),
        "MATH12K-384 ordered problems": (
            TRAIN_ORDERED_PROBLEM_HASH,
            train_identity["ordered_problem_sha256"],
        ),
        "MATH-500 ordered rows": (
            EVAL_ORDERED_ROW_HASH,
            eval_identity["ordered_row_sha256"],
        ),
        "MATH-500 ordered problems": (
            EVAL_ORDERED_PROBLEM_HASH,
            eval_identity["ordered_problem_sha256"],
        ),
    }
    for label, (expected, observed) in expected_identity.items():
        if observed != expected:
            raise MaterializationError(
                f"{label} drifted: expected={expected} observed={observed}"
            )

    normalized_train = [normalize_problem(row["problem"]) for row in train_rows]
    normalized_eval = [normalize_problem(row["problem"]) for row in eval_rows]
    if len(set(normalized_train)) != TRAIN_SELECTED_ROWS:
        raise MaterializationError("MATH12K-384 problems are not unique")
    if len(set(normalized_eval)) != EVAL_ROWS:
        raise MaterializationError("MATH-500 problems are not unique")
    blank_train_answers = sum(not str(row["answer"]).strip() for row in train_rows)
    blank_eval_answers = sum(not str(row["answer"]).strip() for row in eval_rows)
    if blank_train_answers:
        raise MaterializationError(
            f"MATH12K-384 contains {blank_train_answers} blank answers"
        )
    if blank_eval_answers:
        raise MaterializationError(
            f"MATH-500 contains {blank_eval_answers} blank answers"
        )
    overlap = set(normalized_train) & set(normalized_eval)
    if overlap:
        raise MaterializationError(
            f"MATH12K-384/MATH-500 normalized problem overlap: {len(overlap)}"
        )

    subject_counts = dict(
        sorted(Counter(str(row["subject"]) for row in train_rows).items())
    )
    level_counts = dict(
        sorted(
            Counter(str(int(row["level"])) for row in train_rows).items(),
            key=lambda item: int(item[0]),
        )
    )
    if subject_counts != EXPECTED_SUBJECT_COUNTS:
        raise MaterializationError(
            f"MATH12K-384 subject counts drifted: {subject_counts!r}"
        )
    if level_counts != EXPECTED_LEVEL_COUNTS:
        raise MaterializationError(
            f"MATH12K-384 level counts drifted: {level_counts!r}"
        )

    prompt_audit = _prompt_length_audit(train_rows, tokenizer_root)
    audit = {
        "train_source_rows": len(train_dataset),
        "train_rows": len(train_rows),
        "train_unique_normalized_problems": len(set(normalized_train)),
        "train_blank_answers": blank_train_answers,
        "train_subject_counts": subject_counts,
        "train_level_counts": level_counts,
        "train_identity": train_identity,
        "eval_split": "math",
        "eval_rows": len(eval_rows),
        "eval_unique_normalized_problems": len(set(normalized_eval)),
        "eval_blank_answers": blank_eval_answers,
        "eval_identity": eval_identity,
        "normalized_problem_overlap": len(overlap),
        "prompt_admission": prompt_audit,
    }
    return audit, train_rows, eval_rows


def _source_bundle(
    *,
    source_checkout: Path = SOURCE_CHECKOUT,
    train_source_root: Path = TRAIN_SOURCE_ROOT,
    eval_import_root: Path = EVAL_IMPORT_ROOT,
    eval_source_root: Path = EVAL_SOURCE_ROOT,
    tokenizer_root: Path = TOKENIZER_ROOT,
) -> tuple[dict[str, Any], Any, Any, list[dict[str, Any]], list[dict[str, Any]]]:
    git_identity = verify_source_git(source_checkout)
    train_files = _verify_files(
        train_source_root, TRAIN_SOURCE_FILES, label="MATH12K source"
    )
    eval_import = verify_eval_import_manifest(eval_import_root)
    eval_files = _verify_files(
        eval_source_root, EVAL_SOURCE_FILES, label="MATH-500 source"
    )
    train_dataset, eval_dataset = _load_dataset_roots(
        train_source_root, eval_source_root
    )
    audit, train_rows, eval_rows = _validate_rows(
        train_dataset, eval_dataset, tokenizer_root=tokenizer_root
    )
    source = {
        "math12k": {
            **git_identity,
            "root": _repo_relative(train_source_root),
            "files_sha256": train_files,
        },
        "math500": {
            **eval_import,
            "root": _repo_relative(eval_source_root),
            "files_sha256": eval_files,
        },
        "tokenizer": {
            "root": _repo_relative(tokenizer_root),
            "revision": TOKENIZER_REVISION,
            "files_sha256": dict(TOKENIZER_FILES),
        },
    }
    return source, train_dataset, eval_dataset, train_rows, eval_rows


def audit_sources() -> dict[str, Any]:
    """Verify the immutable sources and return the prospective manifest audit."""

    source, train_dataset, eval_dataset, _, _ = _source_bundle()
    audit, _, _ = _validate_rows(
        train_dataset, eval_dataset, tokenizer_root=TOKENIZER_ROOT
    )
    return {"source": source, "audit": audit}


def _output_file_hashes(root: Path) -> dict[str, str]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    ]
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(files)
    }


def _audit_output_rows(
    root: Path,
    expected_train_rows: Sequence[dict[str, Any]],
    expected_eval_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    train_dataset, eval_dataset = _load_dataset_roots(
        root / "train", root / "eval"
    )
    observed_train = [dict(row) for row in train_dataset]
    observed_eval = [dict(row) for row in eval_dataset]
    if observed_train != list(expected_train_rows):
        raise MaterializationError(
            "materialized train rows differ from source rows 0..383"
        )
    if observed_eval != list(expected_eval_rows):
        raise MaterializationError(
            "materialized math eval rows differ from full MATH-500"
        )
    return {
        "train_splits": ["train"],
        "train_rows": len(observed_train),
        "train_ordered_row_sha256": _row_identity(observed_train)[
            "ordered_row_sha256"
        ],
        "eval_splits": ["math"],
        "eval_rows": len(observed_eval),
        "eval_ordered_row_sha256": _row_identity(observed_eval)[
            "ordered_row_sha256"
        ],
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def materialize(output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Build and atomically publish the frozen E39 DatasetDict pair."""

    try:
        from datasets import DatasetDict
    except ImportError as exc:  # pragma: no cover - production dependency guard
        raise MaterializationError("datasets is required to materialize E39") from exc

    output_root = output_root.resolve()
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_root}; pass --overwrite to replace it"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    source, train_dataset, eval_dataset, train_rows, eval_rows = _source_bundle()
    audit, _, _ = _validate_rows(
        train_dataset, eval_dataset, tokenizer_root=TOKENIZER_ROOT
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    backup: Path | None = None
    try:
        DatasetDict(
            {"train": train_dataset.select(range(TRAIN_SELECTED_ROWS))}
        ).save_to_disk(str(staging / "train"))
        DatasetDict({"math": eval_dataset}).save_to_disk(str(staging / "eval"))
        output_audit = _audit_output_rows(staging, train_rows, eval_rows)
        manifest = {
            "schema": "e39_math12k_384_math500_materialization_v1",
            "selection": {
                "rule": "first_384_source_rows_in_source_order",
                "source_indices": [0, TRAIN_SELECTED_ROWS - 1],
                "train_split": "train",
                "eval_split": "math",
            },
            "source": source,
            "audit": audit,
            "output": {
                **output_audit,
                "files_sha256": _output_file_hashes(staging),
            },
        }
        _atomic_write_json(staging / MANIFEST_NAME, manifest)

        if output_root.exists():
            backup = output_root.parent / (
                f".{output_root.name}.backup.{os.getpid()}"
            )
            if backup.exists():
                raise FileExistsError(f"backup path already exists: {backup}")
            os.replace(output_root, backup)
        try:
            os.replace(staging, output_root)
        except Exception:
            if backup is not None and backup.exists() and not output_root.exists():
                os.replace(backup, output_root)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def audit_materialized(output_root: Path) -> dict[str, Any]:
    """Fail closed unless an existing output exactly matches all frozen inputs."""

    output_root = output_root.resolve()
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise MaterializationError(
            f"materialization manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError("materialization manifest is invalid") from exc
    if manifest.get("schema") != "e39_math12k_384_math500_materialization_v1":
        raise MaterializationError("materialization manifest schema drifted")

    source, train_dataset, eval_dataset, train_rows, eval_rows = _source_bundle()
    expected_audit, _, _ = _validate_rows(
        train_dataset, eval_dataset, tokenizer_root=TOKENIZER_ROOT
    )
    output_audit = _audit_output_rows(output_root, train_rows, eval_rows)
    observed_files = _output_file_hashes(output_root)
    expected_manifest_fields = {
        "source": source,
        "audit": expected_audit,
    }
    for key, expected in expected_manifest_fields.items():
        if manifest.get(key) != expected:
            raise MaterializationError(f"materialization manifest {key} drifted")
    manifest_output = manifest.get("output")
    if not isinstance(manifest_output, dict):
        raise MaterializationError("materialization manifest output is missing")
    if {
        key: manifest_output.get(key) for key in output_audit
    } != output_audit:
        raise MaterializationError("materialization output audit drifted")
    if manifest_output.get("files_sha256") != observed_files:
        raise MaterializationError("materialized output file hashes drifted")
    return expected_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="verify an existing materialization without writing it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_only:
        audit = audit_materialized(args.output_root)
        print(json.dumps(audit, sort_keys=True))
        print(f"verified {args.output_root.resolve()}")
        return
    manifest = materialize(args.output_root, overwrite=args.overwrite)
    print(json.dumps(manifest["audit"], sort_keys=True))
    print(f"wrote {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
