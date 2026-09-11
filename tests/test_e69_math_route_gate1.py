from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops/route_successor/sample_e69_math_route_gate1.py"
SPEC = spec_from_file_location("e69_gate1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows(*, correct: int, accepted: int):
    rows = []
    for prompt_index in range(MODULE.PROMPT_COUNT):
        for sample_index in range(MODULE.SAMPLE_COUNT):
            index = len(rows)
            task_correct = index < correct
            route_accepted = index < accepted
            route_number = sample_index % 3
            rows.append(
                {
                    "unique_id": f"p{prompt_index}",
                    "sample_index": sample_index,
                    "response_sha256": f"r{index}",
                    "task_correct": task_correct,
                    "trace_parse_execute": route_accepted,
                    "route_accepted": route_accepted,
                    "route_signature": (
                        f"route-{route_number}" if route_accepted else None
                    ),
                    "route_formatting_stable": (True if route_accepted else None),
                }
            )
    return rows


def test_gate1_automatic_thresholds_pass_only_joint_contract():
    rows = _rows(correct=100, accepted=90)
    summary = MODULE.automatic_gate_summary(rows)

    assert summary["status"] == "pass"
    assert summary["rates"]["route_coverage_given_task_correct"] == 0.9
    assert summary["counts"]["multi_route_prompts"] >= 8
    assert summary["counts"]["cross_prompt_recurring_signatures"] == 3


def test_gate1_fails_route_coverage_below_eighty_percent():
    rows = _rows(correct=100, accepted=79)
    summary = MODULE.automatic_gate_summary(rows)

    assert summary["status"] == "fail"
    assert any("route coverage" in value for value in summary["violations"])


def test_gate1_launcher_binds_all_fixed_inputs_and_uses_one_job():
    launcher = (ROOT / "ops/route_successor/launch_e69_math_route_gate1.sh").read_text(
        encoding="utf-8"
    )
    slurm = (ROOT / "ops/slurm/e69_math_route_gate1_node302.slurm").read_text(
        encoding="utf-8"
    )

    assert "MATERIALIZATION_MANIFEST.json" in launcher
    assert "RPN_V2_PROMPT_MANIFEST.json" in launcher
    assert "e69_math_route_gate1_rpn_v2_amendment_20260728.md" in launcher
    assert "e69_math_route_gate1_base_rpn_v2" in launcher
    assert "sample_e69_math_route_gate1.py" in launcher
    assert "OAT_E69_GATE1_INPUT_IDENTITY" in launcher
    assert "sbatch" in launcher
    assert "--gres=gpu:a100:1" in slurm
    assert "VLLM_USE_V1=0" in slurm


def test_gate1_rpn_formatting_perturbation_preserves_tokens():
    response = r"\boxed{13}<route>v2 5 square 12 square add sqrt</route>"
    variant = MODULE._formatting_variant(
        response,
        "v2 5 square 12 square add sqrt",
    )

    assert "<route>\n" in variant
    assert variant.split("<route>", 1)[1].split("</route>", 1)[0].split() == [
        "v2",
        "5",
        "square",
        "12",
        "square",
        "add",
        "sqrt",
    ]
