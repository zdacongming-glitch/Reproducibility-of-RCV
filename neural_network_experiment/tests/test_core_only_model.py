from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.data.synthetic_teacher_student import make_block_slices  # noqa: E402
from src.models.mlp import CoreOnlyRegressor  # noqa: E402


class SumModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)


def test_core_only_wrapper_slices_inputs() -> None:
    wrapper = CoreOnlyRegressor(base_model=SumModel(), block_slices=make_block_slices(2, 2, 1))
    x = torch.tensor([[1.0, 2.0, 10.0, 20.0, 30.0]])
    output = wrapper(x)
    assert output.item() == 3.0


def test_core_only_wrapper_is_invariant_to_fragile_perturbations() -> None:
    block_slices = make_block_slices(2, 2, 1)
    wrapper = CoreOnlyRegressor(base_model=SumModel(), block_slices=block_slices)
    x = torch.tensor([[1.0, 2.0, 10.0, 20.0, 30.0]])
    perturbed = x.clone()
    perturbed[:, block_slices.frag] += torch.tensor([[100.0, -100.0]])
    assert torch.allclose(wrapper(x), wrapper(perturbed))
