from oat_drgrpo.templates import (
    TEMPLATE_FACTORY,
    apply_qwen_math_route_template,
)


def test_math_route_template_is_task_first_and_exposes_exact_trace_contract():
    prompt = apply_qwen_math_route_template("What is 5+7?")

    assert "\\boxed{}" in prompt
    assert "<route>v2 2 5 mul 1 sub square 1 add</route>" in prompt
    assert "reverse-Polish program" in prompt
    assert "never put a computed result" in prompt
    assert "omit <route> but still give the boxed answer" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert TEMPLATE_FACTORY["qwen_math_route"]("What is 5+7?") == prompt
