from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops/route_successor/audit_e69_math_route_rpn_v2_prompts.py"
MANIFEST = ROOT / "var/data/math12k_384_route_dev128_v1/RPN_V2_PROMPT_MANIFEST.json"


def _module():
    spec = spec_from_file_location("e69_rpn_v2_prompt_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rpn_v2_prompt_manifest_recomputes_over_exact_sealed_population():
    observed = _module().build_manifest()
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert observed == frozen
    assert frozen["population"]["rows"] == 128
    assert (
        frozen["population"]["ordered_problem_sha256"]
        == "2f37f517a6ab6badce6c8d2fcc4e05483449a06d3aee89d87b5c4ecb7400fa43"
    )
    assert frozen["prompt_contract"]["route_language"] == "math-route-rpn-v2"
    assert frozen["prompt_contract"]["maximum_tokens"] <= 1024
