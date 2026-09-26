#!/usr/bin/env python3
"""Reserve disjoint E117 development and confirmation evaluation prompts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "ops"
SRC = ROOT / "src"
for import_root in (OPS, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from make_exact_answer_mode_data import (  # noqa: E402
    _synthetic_graph_rows,
)
from make_exact_countdown_mode_data import (  # noqa: E402
    _synthetic_countdown_rows,
)
from make_mathir_action_menu_data import (  # noqa: E402
    _build_rows as _build_mathir_rows,
    _family_support,
)
from make_modebench_data import _graph_color_string  # noqa: E402
from make_python_factor_mode_data import (  # noqa: E402
    _build_rows as _build_python_rows,
)


SCHEMA = "e117_evaluation_reserve_v1"
FROZEN_AT = "2026-08-25T12:14:49-04:00"
ROWS_PER_BLOCK = 128
OUTPUT_ROOT = ROOT / "var/data/e117_evaluation_reserve_v1"
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e117_stage1a8_untouched_evaluation_reserves_20260825.md"
)
PROTOCOL_SHA256 = "d790812b94748c1a7bb6ed061b9800a8c286b18b676a7f2001a672b871f428d8"
TEST_PATH = ROOT / "tests/test_e117_evaluation_reserves.py"


DOMAIN_ORDER = (
    "countdown",
    "graph_coloring",
    "python_factors",
    "mathir",
)
SENTINELS = {
    "countdown": "qwen05b/countdown",
    "graph_coloring": "qwen05b/graph_coloring",
    "python_factors": "qwen05b/python_factors",
    "mathir": "falcon1b/mathir",
}
SOURCE_ROOTS = {
    "countdown": ROOT / "var/data/exact_countdown_easy3_probe",
    "graph_coloring": ROOT / "var/data/graph_coloring_modebench_v2",
    "python_factors": ROOT / "var/data/python_factor_modebench_v1",
    "mathir": ROOT / "var/data/mathir_action_menu_v1",
}
EXPECTED_SOURCE_SPLITS = {
    "countdown": {"train": {"train": 384}, "eval": {"multi_answer": 128}},
    "graph_coloring": {
        "train": {"train": 384},
        "eval": {"multi_answer": 128, "unique_answer": 128},
    },
    "python_factors": {
        "train": {"train": 384},
        "eval": {"multi_answer": 128},
    },
    "mathir": {"train": {"train": 384}, "eval": {"multi_answer": 128}},
}
EXPECTED_SOURCE_TREE_SHA256 = {
    "countdown": "3c3efca8b1d8911020e1826db196cb1c2a8a69608cae63cf6452d99f744f0ba9",
    "graph_coloring": "b0c76e2d7646a5bf228057dc9158849ffac7bb556a2dd7784de2f18f163331c1",
    "python_factors": "be47c9d40084da92acfc840fe581d9c06d6c6fe14e606400e8be2de5a191d5a8",
    "mathir": "4e3ec586c91f264e04ef0c63901a07f32e10b1150458c3f90b54de3564b63c55",
}
GENERATORS = {
    "countdown": ROOT / "ops/make_exact_countdown_mode_data.py",
    "graph_coloring": ROOT / "ops/make_exact_answer_mode_data.py",
    "python_factors": ROOT / "ops/make_python_factor_mode_data.py",
    "mathir": ROOT / "ops/make_mathir_action_menu_data.py",
}
EXPECTED_GENERATOR_SHA256 = {
    "countdown": "9712aa2b01f61df567cbbef569c4d7e1230aebebf34d51db641daeaccee5f583",
    "graph_coloring": "cb59292a32c1af7789676e7dd6eeffaf96a2937b01c14e4141b22ed223f66cba",
    "python_factors": "86d0f23147360bd389190eac4a0fbb99950e08287a986a01bda7dd13f811bc18",
    "mathir": "bc1122151c9abbfa5b7e3e7dbbe48535b30bb75bea3b1619c0472b2fd0ae0418",
}
SEEDS = {
    "development": {
        "countdown": 117100,
        "graph_coloring": 117200,
        "python_factors": 117300,
        "mathir": 117400,
    },
    "confirmation": {
        "countdown": 117500,
        "graph_coloring": 117600,
        "python_factors": 117700,
        "mathir": 117800,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _reserve_data_tree_sha256(root: Path) -> str:
    """Hash only the two data blocks, excluding the recursive identity file."""
    digest = hashlib.sha256()
    paths = [
        path
        for block in ("development", "confirmation")
        for path in (root / block).rglob("*")
        if path.is_file()
    ]
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_identity(domain: str, row: dict[str, Any]) -> tuple[Any, ...]:
    spec = json.loads(str(row["answer"]))
    if domain == "countdown":
        return (
            "countdown",
            tuple(sorted(int(value) for value in spec["numbers"])),
            int(spec["target"]),
        )
    if domain == "graph_coloring":
        return (
            "graph_coloring",
            int(spec["n"]),
            tuple(sorted(tuple(int(value) for value in edge) for edge in spec["edges"])),
            _graph_color_string(spec["partial_colors"]),
        )
    if domain == "python_factors":
        return (
            "python_factors",
            tuple(sorted(int(value) for value in spec["cases"])),
        )
    if domain == "mathir":
        return (
            "mathir",
            str(spec["family"]),
            tuple(
                sorted(
                    (str(key), int(value))
                    for key, value in spec["bindings"].items()
                )
            ),
        )
    raise ValueError(f"unknown E117 reserve domain: {domain}")


def _identities(domain: str, rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    return {_row_identity(domain, row) for row in rows}


def _identity_sha256(identities: set[tuple[Any, ...]]) -> str:
    payload = "\n".join(
        sorted(json.dumps(identity, separators=(",", ":")) for identity in identities)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_source_rows(domain: str) -> list[dict[str, Any]]:
    root = SOURCE_ROOTS[domain]
    actual_tree_sha256 = _tree_sha256(root)
    if actual_tree_sha256 != EXPECTED_SOURCE_TREE_SHA256[domain]:
        raise RuntimeError(
            f"{domain} source tree drift: {actual_tree_sha256} != "
            f"{EXPECTED_SOURCE_TREE_SHA256[domain]}"
        )
    rows: list[dict[str, Any]] = []
    for partition, expected_splits in EXPECTED_SOURCE_SPLITS[domain].items():
        dataset = load_from_disk(str(root / partition))
        actual_splits = {str(name): len(split) for name, split in dataset.items()}
        if actual_splits != expected_splits:
            raise RuntimeError(
                f"{domain}/{partition} split drift: {actual_splits} != "
                f"{expected_splits}"
            )
        for split_name in expected_splits:
            rows.extend(dict(row) for row in dataset[split_name])
    identities = _identities(domain, rows)
    if len(identities) != len(rows):
        raise RuntimeError(f"{domain} historical source contains identity overlap")
    return rows


def _verify_source_manifests() -> None:
    graph = json.loads((SOURCE_ROOTS["graph_coloring"] / "identity.json").read_text())
    if not (
        graph.get("schema") == "graph_coloring_modebench_v2"
        and graph.get("seed") == 0
        and graph.get("train_size") == 384
        and graph.get("eval_size") == 128
        and graph.get("generator_sha256") == EXPECTED_GENERATOR_SHA256["graph_coloring"]
    ):
        raise RuntimeError("graph source identity manifest drift")

    python = json.loads((SOURCE_ROOTS["python_factors"] / "identity.json").read_text())
    if not (
        python.get("schema") == "python_factor_modebench_v1"
        and python.get("seed") == 5100
        and python.get("train_rows") == 384
        and python.get("eval_rows") == 128
        and python.get("case_count") == 4
        and python.get("max_value") == 96
        and python.get("minimum_exact_mode_count") == 16
    ):
        raise RuntimeError("Python-factor source identity manifest drift")

    mathir = json.loads((SOURCE_ROOTS["mathir"] / "identity.json").read_text())
    if not (
        mathir.get("schema") == "mathir_action_menu_v1"
        and mathir.get("seed") == 5900
        and mathir.get("train_rows") == 384
        and mathir.get("eval_rows") == 128
        and len(mathir.get("families", [])) == 4
    ):
        raise RuntimeError("MathIR source identity manifest drift")


def _verify_recovered_source_contracts() -> None:
    countdown_train = _synthetic_countdown_rows(
        384,
        seed=0,
        split_tag="train_multi_answer",
        number_count=3,
        max_value=12,
        min_modes=2,
        max_modes=8,
    )
    countdown_train_ids = _identities("countdown", countdown_train)
    countdown_eval = _synthetic_countdown_rows(
        128,
        seed=10_000,
        split_tag="eval_multi_answer",
        number_count=3,
        max_value=12,
        min_modes=2,
        max_modes=8,
        exclude={identity[1:] for identity in countdown_train_ids},
    )
    source_countdown_train = [
        dict(row)
        for row in load_from_disk(str(SOURCE_ROOTS["countdown"] / "train"))["train"]
    ]
    source_countdown_eval = [
        dict(row)
        for row in load_from_disk(str(SOURCE_ROOTS["countdown"] / "eval"))[
            "multi_answer"
        ]
    ]
    if countdown_train != source_countdown_train or countdown_eval != source_countdown_eval:
        raise RuntimeError("recovered Countdown generator contract is not exact")

    graph_common = {
        "max_n": 6,
        "max_edges": 8,
        "prompt_style": "original",
        "balance_hidden_color": False,
    }
    graph_train = _synthetic_graph_rows(
        384,
        seed=0,
        hidden_count=3,
        min_completions=4,
        max_completions=24,
        min_solutions=4,
        split_tag="train_multi_answer",
        **graph_common,
    )
    graph_eval = _synthetic_graph_rows(
        128,
        seed=10_000,
        hidden_count=3,
        min_completions=4,
        max_completions=24,
        min_solutions=4,
        split_tag="eval_multi_answer",
        exclude={identity[1:] for identity in _identities("graph_coloring", graph_train)},
        **graph_common,
    )
    source_graph_train = [
        dict(row)
        for row in load_from_disk(str(SOURCE_ROOTS["graph_coloring"] / "train"))[
            "train"
        ]
    ]
    source_graph_eval = [
        dict(row)
        for row in load_from_disk(str(SOURCE_ROOTS["graph_coloring"] / "eval"))[
            "multi_answer"
        ]
    ]
    if graph_train != source_graph_train or graph_eval != source_graph_eval:
        raise RuntimeError("recovered Graph generator contract is not exact")


def _build_domain_rows(
    domain: str,
    *,
    count: int,
    seed: int,
    split_tag: str,
    excluded: set[tuple[Any, ...]],
    mathir_support: dict[str, tuple[int, str]],
) -> list[dict[str, Any]]:
    generator_excluded = {identity[1:] for identity in excluded}
    if domain == "countdown":
        return _synthetic_countdown_rows(
            count,
            seed=seed,
            split_tag=split_tag,
            number_count=3,
            max_value=12,
            min_modes=2,
            max_modes=8,
            exclude=generator_excluded,
        )
    if domain == "graph_coloring":
        return _synthetic_graph_rows(
            count,
            seed=seed,
            hidden_count=3,
            min_completions=4,
            max_completions=24,
            min_solutions=4,
            split_tag=split_tag,
            max_n=6,
            max_edges=8,
            prompt_style="original",
            balance_hidden_color=False,
            exclude=generator_excluded,
        )
    if domain == "python_factors":
        return _build_python_rows(
            count,
            seed=seed,
            split_tag=split_tag,
            case_count=4,
            max_value=96,
            min_modes=16,
            excluded={identity[1] for identity in excluded},
        )
    if domain == "mathir":
        return _build_mathir_rows(
            count,
            seed=seed,
            split_tag=split_tag,
            family_support=mathir_support,
            excluded={(identity[1], identity[2]) for identity in excluded},
        )
    raise ValueError(f"unknown E117 reserve domain: {domain}")


def _validate_reserved_rows(
    domain: str,
    rows: list[dict[str, Any]],
    *,
    split_tag: str,
) -> None:
    if len(rows) != ROWS_PER_BLOCK:
        raise RuntimeError(f"{domain}/{split_tag} row count is not {ROWS_PER_BLOCK}")
    if len(_identities(domain, rows)) != len(rows):
        raise RuntimeError(f"{domain}/{split_tag} contains duplicate identities")
    specs = [json.loads(str(row["answer"])) for row in rows]
    if any(str(row["answer_mode_split"]) != split_tag for row in rows):
        raise RuntimeError(f"{domain}/{split_tag} split labels drifted")
    if domain == "countdown" and not all(
        len(spec["numbers"]) == 3
        and min(spec["numbers"]) >= 2
        and max(spec["numbers"]) <= 12
        and 2 <= int(spec["num_completions"]) <= 8
        for spec in specs
    ):
        raise RuntimeError(f"{domain}/{split_tag} distribution drift")
    if domain == "graph_coloring" and not all(
        sum(value is None for value in spec["partial_colors"]) == 3
        and int(spec["n"]) <= 6
        and len(spec["edges"]) <= 8
        and 4 <= int(spec["num_completions"]) <= 24
        for spec in specs
    ):
        raise RuntimeError(f"{domain}/{split_tag} distribution drift")
    if domain == "python_factors" and not all(
        len(spec["cases"]) == 4
        and max(spec["cases"]) <= 96
        and int(spec["num_modes"]) >= 16
        for spec in specs
    ):
        raise RuntimeError(f"{domain}/{split_tag} distribution drift")
    if domain == "mathir":
        family_counts = Counter(str(spec["family"]) for spec in specs)
        if sorted(family_counts.values()) != [32, 32, 32, 32]:
            raise RuntimeError(f"{domain}/{split_tag} family balance drift")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def materialize(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing reserve: {output_root}")
    if _sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("E117 A8 protocol digest drift")
    for domain, path in GENERATORS.items():
        actual = _sha256_file(path)
        if actual != EXPECTED_GENERATOR_SHA256[domain]:
            raise RuntimeError(
                f"{domain} generator drift: {actual} != "
                f"{EXPECTED_GENERATOR_SHA256[domain]}"
            )
    _verify_source_manifests()
    _verify_recovered_source_contracts()

    source_rows = {domain: _load_source_rows(domain) for domain in DOMAIN_ORDER}
    historical_ids = {
        domain: _identities(domain, rows) for domain, rows in source_rows.items()
    }
    mathir_support = _family_support()
    reserved: dict[str, dict[str, list[dict[str, Any]]]] = {
        "development": {},
        "confirmation": {},
    }
    domain_records: dict[str, Any] = {}

    for domain in DOMAIN_ORDER:
        development_tag = "e117_stage1_development"
        development = _build_domain_rows(
            domain,
            count=ROWS_PER_BLOCK,
            seed=SEEDS["development"][domain],
            split_tag=development_tag,
            excluded=historical_ids[domain],
            mathir_support=mathir_support,
        )
        development_ids = _identities(domain, development)
        confirmation_tag = "e117_confirmation"
        confirmation = _build_domain_rows(
            domain,
            count=ROWS_PER_BLOCK,
            seed=SEEDS["confirmation"][domain],
            split_tag=confirmation_tag,
            excluded=historical_ids[domain] | development_ids,
            mathir_support=mathir_support,
        )
        confirmation_ids = _identities(domain, confirmation)
        _validate_reserved_rows(domain, development, split_tag=development_tag)
        _validate_reserved_rows(domain, confirmation, split_tag=confirmation_tag)

        overlaps = {
            "historical_vs_development": len(historical_ids[domain] & development_ids),
            "historical_vs_confirmation": len(historical_ids[domain] & confirmation_ids),
            "development_vs_confirmation": len(development_ids & confirmation_ids),
        }
        if any(overlaps.values()):
            raise RuntimeError(f"{domain} reserve identity overlap: {overlaps}")
        reserved["development"][domain] = development
        reserved["confirmation"][domain] = confirmation
        domain_records[domain] = {
            "sentinel": SENTINELS[domain],
            "source_root": _relative(SOURCE_ROOTS[domain]),
            "source_tree_sha256": EXPECTED_SOURCE_TREE_SHA256[domain],
            "source_rows": len(source_rows[domain]),
            "source_identity_sha256": _identity_sha256(historical_ids[domain]),
            "generator": _relative(GENERATORS[domain]),
            "generator_sha256": EXPECTED_GENERATOR_SHA256[domain],
            "development": {
                "seed": SEEDS["development"][domain],
                "rows": len(development),
                "rows_sha256": _rows_sha256(development),
                "identity_sha256": _identity_sha256(development_ids),
            },
            "confirmation": {
                "seed": SEEDS["confirmation"][domain],
                "rows": len(confirmation),
                "rows_sha256": _rows_sha256(confirmation),
                "identity_sha256": _identity_sha256(confirmation_ids),
            },
            "overlap_counts": overlaps,
        }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".e117_evaluation_reserve_v1.", dir=output_root.parent)
    )
    try:
        for block in ("development", "confirmation"):
            for domain in DOMAIN_ORDER:
                destination = temporary_root / block / domain / "eval"
                DatasetDict(
                    {"multi_answer": Dataset.from_list(reserved[block][domain])}
                ).save_to_disk(str(destination))

        data_tree_sha256 = _reserve_data_tree_sha256(temporary_root)
        manifest = {
            "schema": SCHEMA,
            "frozen_at": FROZEN_AT,
            "rows_per_domain_per_block": ROWS_PER_BLOCK,
            "blocks": ["development", "confirmation"],
            "domains": domain_records,
            "data_tree_sha256": data_tree_sha256,
            "protocol": _relative(PROTOCOL),
            "protocol_sha256": PROTOCOL_SHA256,
            "materializer": _relative(Path(__file__).resolve()),
            "materializer_sha256": _sha256_file(Path(__file__).resolve()),
            "tests": _relative(TEST_PATH),
            "tests_sha256": _sha256_file(TEST_PATH),
            "source_contract_reconstruction": {
                "countdown_train_and_eval_exact": True,
                "graph_train_and_multi_eval_exact": True,
            },
            "historical_training_banks_retained": True,
            "development_is_only_stage1_evaluation_block": True,
            "confirmation_reserved_before_development_outcomes": True,
            "confirmation_sealed_from_development_analysis": True,
            "stage1_launch_authorized": False,
            "stage1_outcomes_exist": False,
            "e117r1_terminal_cells_at_freeze": 0,
            "e117r1_realized_optimizer_updates_at_freeze": 0,
            "model_outcomes_inspected": False,
            "response_data_read": False,
            "prompt_contents_printed": False,
            "pointmaze": "excluded",
        }
        (temporary_root / "identity.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_root.rename(output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    manifest = materialize()
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "output_root": _relative(OUTPUT_ROOT),
                "data_tree_sha256": manifest["data_tree_sha256"],
                "identity_sha256": _sha256_file(OUTPUT_ROOT / "identity.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
