#!/usr/bin/env python3
"""Materialize exact OAT 384/MATH-500 from the first passing E50 toy."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "materialize_e49v_exact_oat_natural_menu.py"
)
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50e_exact_oat_math_full_20260726.md"
)
CONTINUITY_CERTIFIER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "certify_e50e_qwen72_continuity.py"
)
SOURCES = (
    (
        ROOT / "var/data/e50d_teacher_route_math_toy",
        ROOT
        / "var/artifacts/"
        "e50d_matched_math_toy_advancement_v1.json",
        ROOT
        / "var/data/e50d_teacher_route_math_toy/"
        "MATERIALIZATION_MANIFEST.json",
    ),
)


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "e50e_e49v_materializer", BASE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base materializer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_continuity_certifier():
    spec = importlib.util.spec_from_file_location(
        "e50e_continuity_certifier", CONTINUITY_CERTIFIER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load E50E continuity certifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_continuity() -> tuple[pathlib.Path, dict]:
    names = {
        "endpoint": "E50E_ENDPOINT_RECORD",
        "route": "E50E_ROUTE_CALIBRATION",
        "declaration": "E50E_DECLARATION_CALIBRATION",
        "certificate": "E50E_CONTINUITY_CERTIFICATE",
    }
    paths = {}
    for label, name in names.items():
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"E50E continuity input is missing: {name}")
        path = pathlib.Path(value).resolve()
        if not path.is_file():
            raise RuntimeError(f"E50E continuity input is absent: {path}")
        paths[label] = path
    certifier = _load_continuity_certifier()
    expected = certifier.build_certificate(
        endpoint_path=paths["endpoint"],
        route_path=paths["route"],
        declaration_path=paths["declaration"],
        expected_node="node302",
        expected_port=8771,
    )
    observed = json.loads(
        paths["certificate"].read_text(encoding="utf-8")
    )
    if observed != expected or expected.get("pass") is not True:
        raise RuntimeError("E50E continuity certificate failed")
    return paths["endpoint"], observed


def _select_source() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    for toy, advancement, certification in SOURCES:
        if not advancement.is_file():
            continue
        result = json.loads(advancement.read_text(encoding="utf-8"))
        if (
            result.get("schema") == "e50_matched_math_toy_advancement_v1"
            and result.get("variant") == "e50d"
            and result.get("complete_evidence") is True
            and result.get("advance_to_exact_oat_full") is True
            and all(result.get("checks", {}).values())
        ):
            return toy, advancement, certification
    raise RuntimeError("no E50 matched toy has passed exact-full advancement")


def main() -> None:
    endpoint_path, continuity = _verify_continuity()
    toy, advancement, certification = _select_source()
    base = _load_base()

    def _continuity_endpoint(record_path: pathlib.Path):
        if record_path.resolve() != endpoint_path:
            raise RuntimeError("E50E materializer endpoint path drifted")
        if _sha256(record_path) != continuity["identities"][
            "endpoint_record_sha256"
        ]:
            raise RuntimeError("E50E materializer endpoint identity drifted")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        expected = continuity["endpoint"]
        if any(record.get(key) != value for key, value in expected.items()):
            raise RuntimeError("E50E materializer endpoint config drifted")
        return (
            f"http://{record['node']}:{int(record['port'])}/v1",
            str(record["model"]),
        )

    base.TOY = toy
    base.TOY_ADVANCEMENT = advancement
    base.TOY_CERTIFICATION = certification
    base.TOY_ADVANCEMENT_SCHEMA = "e50_matched_math_toy_advancement_v1"
    base.OUTPUT_SCHEMA = "e50e_exact_oat_math_materialization_v1"
    base.OUTPUT_LABEL = "E50E"
    base.EXPECTED_MULTI_SUPPORT = {"train": 10, "eval": 1}
    base.EXPECTED_MULTI_TOTAL = 11
    base.EXPECTED_SINGLETON_TOTAL = 873
    base.PROTOCOL = PROTOCOL
    base.SCRIPT = SCRIPT
    base.BASE_SCRIPT = BASE_PATH
    base._endpoint = _continuity_endpoint
    base.main()


if __name__ == "__main__":
    main()
