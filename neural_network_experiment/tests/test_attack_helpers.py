from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.attacks.restricted_attacks import (  # noqa: E402
    project_l2,
    project_linf,
    sample_stochastic_perturbations,
    zero_non_fragile,
)
from src.config import AttackConfig  # noqa: E402
from src.data.synthetic_teacher_student import make_block_slices  # noqa: E402


def test_linf_projection() -> None:
    delta = torch.tensor([[2.0, -3.0, 0.2]])
    projected = project_linf(delta, radius=0.5)
    assert torch.all(projected.abs() <= 0.5 + 1e-8)


def test_l2_projection() -> None:
    delta = torch.tensor([[3.0, 4.0]])
    projected = project_l2(delta, radius=2.0)
    assert projected.norm(p=2, dim=1).item() <= 2.0 + 1e-6


def test_fragile_block_only() -> None:
    block_slices = make_block_slices(2, 2, 1)
    x = torch.zeros(1, 5)
    delta_frag = torch.tensor([[0.1, -0.2]])
    full_delta = zero_non_fragile(delta_frag, x=x, block_slices=block_slices)
    assert torch.allclose(full_delta[:, block_slices.core], torch.zeros(1, 2))
    assert torch.allclose(full_delta[:, block_slices.frag], delta_frag)
    assert torch.allclose(full_delta[:, block_slices.noise], torch.zeros(1, 1))


def test_zero_radius_stochastic_perturbation_is_clean_endpoint() -> None:
    block_slices = make_block_slices(2, 2, 1)
    x = torch.randn(5, 5)
    perturbations = sample_stochastic_perturbations(
        x=x,
        config=AttackConfig(family="linf", radius=0.0),
        block_slices=block_slices,
        num_samples=3,
    )
    assert torch.allclose(perturbations, torch.zeros_like(perturbations))
