from oat_drgrpo.templates import (
    TEMPLATE_FACTORY,
    apply_qwen_math_route_template,
)


def test_math_route_template_is_task_first_and_exposes_exact_trace_contract():
    prompt = apply_qwen_math_route_template("What is 5+7?")

    assert "\\boxed{}" in prompt
    assert "<route>JSON</route>" in prompt
    assert '"version":"math-route-v1"' in prompt
    assert "never write a result field" in prompt
    assert "omit the route block but still give the boxed answer" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert TEMPLATE_FACTORY["qwen_math_route"]("What is 5+7?") == prompt
