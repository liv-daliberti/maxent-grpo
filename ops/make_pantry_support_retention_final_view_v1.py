#!/usr/bin/env python3
"""Create the immutable Pantry repair final view from train + untouched dev."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "var/data/pantry_plan_modebench_v3_repair"
OUTPUT = ROOT / "var/data/pantry_plan_modebench_v3_repair_final_view"


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"fresh final view required: {OUTPUT}")
    for path in (
        SOURCE / "train/dataset_dict.json",
        SOURCE / "dev/dataset_dict.json",
        SOURCE / "eval/dataset_dict.json",
        SOURCE / "identity.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    )
    try:
        shutil.copytree(SOURCE / "train", staging / "train")
        shutil.copytree(SOURCE / "dev", staging / "eval")
        identity = {
            "schema": "pantry-support-retention-final-view-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(SOURCE),
            "source_identity_sha256": hashlib.sha256(
                (SOURCE / "identity.json").read_bytes()
            ).hexdigest(),
            "train_source": "train",
            "evaluation_source": "dev",
            "excluded_calibration_source": "eval",
            "train_tree_sha256": tree_hash(SOURCE / "train"),
            "evaluation_tree_sha256": tree_hash(SOURCE / "dev"),
            "excluded_calibration_tree_sha256": tree_hash(SOURCE / "eval"),
            "calibration_jobs_that_loaded_excluded_source": [30204528, 30204529],
            "evaluation_rows_previously_loaded_by_calibration": False,
        }
        (staging / "final_view_identity.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n"
        )
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(OUTPUT)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"[pantry-final-view] output={OUTPUT} tree={tree_hash(OUTPUT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
