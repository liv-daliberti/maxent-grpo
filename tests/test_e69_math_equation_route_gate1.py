from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLER = ROOT / "ops/route_successor/sample_e69_math_route_gate1.py"
PROMPT_AUDIT = (
    ROOT / "ops/route_successor/audit_e69_math_route_rpn_v2_prompts.py"
)
MANIFEST = (
    ROOT
    / "var/data/math12k_384_route_dev128_v1/"
    "EQUATION_V3_PROMPT_MANIFEST.json"
)


def _module(path: Path, name: str):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equation_v3_prompt_manifest_recomputes_over_sealed_population():
    observed = _module(PROMPT_AUDIT, "e69_prompt_audit").build_equation_manifest()
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert observed == frozen
    assert frozen["population"]["rows"] == 128
    assert frozen["prompt_contract"]["route_language"] == "math-equation-route-v3"
    assert frozen["prompt_contract"]["maximum_tokens"] <= 1024


def test_equation_v3_sampler_scores_natural_derivation_without_route_block():
    module = _module(SAMPLER, "e69_sampler")
    row = module._score_sample(
        {
            "route_interface": "equation-v3",
            "problem": "What is 5 plus 7?",
            "answer": "12",
            "response": r"\[5+7=12\]\boxed{12}",
        }
    )

    assert row["task_correct"] is True
    assert row["trace_parse_execute"] is True
    assert row["route_accepted"] is True
    assert row["route_formatting_stable"] is True

    wrong_task = module._score_sample(
        {
            "route_interface": "equation-v3",
            "problem": "What is 5 plus 7?",
            "answer": "13",
            "response": r"\[5+7=12\]\boxed{12}",
        }
    )
    assert wrong_task["trace_parse_execute"] is True
    assert wrong_task["task_correct"] is False
    assert wrong_task["route_accepted"] is False


def test_equation_v3_launcher_binds_fixed_inputs_and_one_gpu_job():
    launcher = (
        ROOT / "ops/route_successor/launch_e69_math_equation_gate1.sh"
    ).read_text(encoding="utf-8")
    slurm = (
        ROOT / "ops/slurm/e69_math_equation_gate1_node302.slurm"
    ).read_text(encoding="utf-8")

    assert "EQUATION_V3_PROMPT_MANIFEST.json" in launcher
    assert "e69_math_route_gate1_equation_v3_amendment_20260728.md" in launcher
    assert "e69_math_route_gate1_base_equation_v3" in launcher
    assert "OAT_E69_GATE1_INPUT_IDENTITY" in launcher
    assert "sbatch" in launcher
    assert "--interface equation-v3" in slurm
    assert "--gres=gpu:a100:1" in slurm
