from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "ops/math_strategy_calibration/"
    "certify_e50e_qwen72_continuity.py"
)
SERVER = ROOT / "ops/slurm/e50e_qwen72_continuity_node302.slurm"
FULL_WRAPPER = ROOT / "ops/exp_scaling/launch_e50e_exact_oat_math_05b.sh"
FULL_PROTOCOL = (
    ROOT / "paper/preregistration/e50e_exact_oat_math_full_20260726.md"
)
PREPARE_FULL = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e50e_exact_oat_materialization.sh"
)
TERMINAL = (
    ROOT
    / "ops/math_strategy_calibration/run_e50e_terminal_probes.sh"
)
MATERIALIZER = (
    ROOT
    / "ops/math_strategy_calibration/materialize_e50e_exact_oat_math.py"
)
SPEC = importlib.util.spec_from_file_location("e50e_continuity", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_existing_calibrated_endpoint_satisfies_continuity_contract(tmp_path):
    endpoint = (
        ROOT
        / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
    )
    route = (
        ROOT
        / "var/artifacts/e49t_route_confusion_calibration_v1/result.json"
    )
    declaration = (
        ROOT
        / "var/artifacts/e49t_declaration_mismatch_calibration_v1/result.json"
    )
    payload = MODULE.build_certificate(
        endpoint_path=endpoint,
        route_path=route,
        declaration_path=declaration,
        expected_node="node302",
        expected_port=8770,
    )
    assert payload["pass"] is True
    assert payload["route_checks"]["zero_false_new"] is True
    assert payload["declaration_checks"]["zero_mismatch_acceptance"] is True

    output = tmp_path / "certificate.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_continuity_certificate_rejects_endpoint_identity_tamper(tmp_path):
    endpoint = json.loads(
        (
            ROOT
            / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
        ).read_text(encoding="utf-8")
    )
    endpoint["checkpoint_revision"] = "tampered"
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(endpoint), encoding="utf-8")
    try:
        MODULE.build_certificate(
            endpoint_path=path,
            route_path=(
                ROOT
                / "var/artifacts/e49t_route_confusion_calibration_v1/"
                "result.json"
            ),
            declaration_path=(
                ROOT
                / "var/artifacts/e49t_declaration_mismatch_calibration_v1/"
                "result.json"
            ),
            expected_node="node302",
            expected_port=8770,
        )
    except RuntimeError as error:
        assert "configuration mismatch" in str(error)
    else:
        raise AssertionError("tampered endpoint was accepted")


def test_continuity_server_calibrates_before_submitting_full_pipeline():
    source = SERVER.read_text(encoding="utf-8")
    assert "#SBATCH --nodelist=node302" in source
    assert "#SBATCH --gres=gpu:a100:4" in source
    assert "#SBATCH --time=2-00:00:00" in source
    assert "698703eae6604af048a3d2f509995dc302088217" in source
    route = source.index("score_e49t_route_confusion_calibration.py")
    declaration = source.index(
        "score_e49t_declaration_mismatch_calibration.py"
    )
    certificate = source.index("certify_e50e_qwen72_continuity.py")
    downstream = source.index("e50e_materialize_exact_oat_node302.slurm")
    assert route < declaration < certificate < downstream


def test_full_path_is_e50d_only_and_endpoint_continuous():
    wrapper = FULL_WRAPPER.read_text(encoding="utf-8")
    prepare = PREPARE_FULL.read_text(encoding="utf-8")
    terminal = TERMINAL.read_text(encoding="utf-8")
    protocol = FULL_PROTOCOL.read_text(encoding="utf-8")
    materializer = MATERIALIZER.read_text(encoding="utf-8")
    for source in (wrapper, prepare, terminal, materializer):
        assert "e50d" in source
        assert "e50b_matched_math_toy_advancement" not in source
    assert "E49V_QWEN72_ENDPOINT_RECORD" in wrapper
    assert "E49V_ROUTE_CALIBRATION_RESULT" in wrapper
    assert "E49V_DECLARATION_CALIBRATION_RESULT" in wrapper
    assert "E49V_CONTINUITY_CERTIFICATE" in wrapper
    assert "full_control_terminal" in terminal
    assert "full_treatment_terminal" in terminal
    assert "node302 8771" in terminal
    assert "48-hour" in protocol
    assert "zero duplicate false-new" in protocol
