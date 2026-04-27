from __future__ import annotations

import pytest


pytest.importorskip("latex2sympy2_extended")
pytest.importorskip("math_verify")
pytest.importorskip("pylatexenc")

from oat_drgrpo.math_grader import (
    extract_reasoning_signature_for_clustering,
    extract_reasoning_signature_from_trace,
)


def test_reasoning_signature_drops_low_information_transient_states():
    response = "<think>\nx+1\nx+2\nx+1=3\ny=4\nz=5\n</think><answer>3</answer>"

    signature = extract_reasoning_signature_for_clustering(response, template="r1")

    assert signature is not None
    assert "x+1" not in signature
    assert "x+2" not in signature
    assert "x+1=3" in signature


def test_reasoning_signature_canonicalizes_symmetric_equations_and_limits_milestones():
    response = "<think>\nx=2\n2=x\ny=4\nz=5\nw=6\nu=7\n</think><answer>2</answer>"

    signature = extract_reasoning_signature_for_clustering(response, template="r1")

    assert signature is not None
    states = signature.split(" || ")
    assert len(states) == 4
    assert "2=x" in states
    assert "x=2" not in states


def test_reasoning_signature_can_be_built_from_truncated_trace_text():
    signature = extract_reasoning_signature_from_trace("x=2\n2=x\ny=4\nz=5\nw=6\nu=7")

    assert signature is not None
    states = signature.split(" || ")
    assert len(states) == 4
    assert "2=x" in states
