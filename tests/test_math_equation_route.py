from oat_drgrpo.math_equation_route import (
    equation_formatting_variant,
    validate_math_equation_route_response,
)


def test_grounded_equation_chain_executes_and_has_value_free_identity():
    first = validate_math_equation_route_response(
        r"\[5^2+12^2=169\]\[\sqrt{169}=13\]\boxed{13}",
        "A right triangle has legs 5 and 12.",
    )
    second = validate_math_equation_route_response(
        r"\[8^2+15^2=289\]\[\sqrt{289}=17\]\boxed{17}",
        "A right triangle has legs 8 and 15.",
    )

    assert first is not None and second is not None
    assert first.terminal_value == 13
    assert second.terminal_value == 17
    assert first.route_signature == second.route_signature
    assert "5" not in first.route_signature
    assert "12" not in first.route_signature


def test_grounded_one_variable_equation_can_be_solved():
    validation = validate_math_equation_route_response(
        r"\[x+2=5\]\[x=3\]\boxed{3}",
        "Solve using the constants 2 and 5.",
    )

    assert validation is not None
    assert validation.terminal_value == 3
    assert "solve" in validation.operations


def test_ungrounded_literal_false_equality_and_prose_fail_closed():
    problem = "Use 2 and 5."

    assert validate_math_equation_route_response(r"\[7=7\]\boxed{7}", problem) is None
    assert (
        validate_math_equation_route_response(r"\[2+5=8\]\boxed{8}", problem)
        is None
    )
    assert (
        validate_math_equation_route_response(
            r"The answer is obviously seven. \(\boxed{7}\)",
            problem,
        )
        is None
    )


def test_equation_route_cannot_execute_code_or_parse_prose_as_an_operation():
    response = (
        r"\[\operatorname{eval}(__import__(os))=7\]"
        r"\[\text{trust me}=7\]\boxed{7}"
    )

    assert validate_math_equation_route_response(response, "Use 2 and 5.") is None


def test_equation_formatting_perturbation_preserves_signature():
    response = r"\[5^2+12^2=169\]\[\sqrt{169}=13\]\boxed{13}"
    variant = equation_formatting_variant(response)

    assert variant != response
    first = validate_math_equation_route_response(
        response,
        "A right triangle has legs 5 and 12.",
    )
    second = validate_math_equation_route_response(
        variant,
        "A right triangle has legs 5 and 12.",
    )
    assert first is not None and second is not None
    assert first.route_signature == second.route_signature
