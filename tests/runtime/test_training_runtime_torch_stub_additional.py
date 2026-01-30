"""Additional coverage for torch stub math fallbacks."""

from __future__ import annotations

from maxent_grpo.training.runtime import torch_stub


def test_tensor_std_and_mean_handle_empty():
    stub = torch_stub._build_torch_stub()
    tensor = stub.tensor([])
    std_val = tensor.std(unbiased=False)
    mean_val = tensor.mean()
    assert std_val.tolist() == [0.0]
    assert mean_val.tolist() == [0.0]


def test_tensor_std_and_mean_handle_non_numeric():
    stub = torch_stub._build_torch_stub()
    tensor = stub.tensor(["a", "b"])
    std_val = tensor.std(unbiased=True)
    mean_val = tensor.mean()
    assert std_val.tolist() == [0.0]
    assert mean_val.tolist() == [0.0]


def test_tensor_std_uses_sample_variance_when_unbiased():
    stub = torch_stub._build_torch_stub()
    tensor = stub.tensor([1.0, 2.0, 3.0])
    std_val = tensor.std(unbiased=True)
    assert std_val.tolist() == [1.0]


def test_stub_device_handles_target_string():
    stub = torch_stub._build_torch_stub()
    device = stub.device("cuda:1")
    assert device.type == "cuda:1"


def test_tensor_clamp_applies_max_bound():
    stub = torch_stub._build_torch_stub()
    tensor = stub.tensor([1, 5, 10])
    clamped = tensor.clamp(max=6)
    assert clamped.tolist() == [1, 5, 6]
