from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from src.config import AttackConfig
from src.data.synthetic_teacher_student import BlockSlices


@dataclass
class AttackResult:
    adv_inputs: torch.Tensor
    losses: torch.Tensor
    clean_losses: torch.Tensor
    deltas: torch.Tensor
    diagnostics: dict[str, float]


def fragile_delta(full_delta: torch.Tensor, block_slices: BlockSlices) -> torch.Tensor:
    return full_delta[:, block_slices.frag]


def zero_non_fragile(delta_frag: torch.Tensor, x: torch.Tensor, block_slices: BlockSlices) -> torch.Tensor:
    delta = torch.zeros_like(x)
    delta[:, block_slices.frag] = delta_frag
    return delta


def project_linf(delta_frag: torch.Tensor, radius: float) -> torch.Tensor:
    return delta_frag.clamp(min=-radius, max=radius)


def project_l2(delta_frag: torch.Tensor, radius: float) -> torch.Tensor:
    norms = delta_frag.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scales = torch.clamp(radius / norms, max=1.0)
    return delta_frag * scales


def sample_random_delta(x: torch.Tensor, config: AttackConfig, block_slices: BlockSlices) -> torch.Tensor:
    frag_dim = block_slices.frag.stop - block_slices.frag.start
    if config.family == "linf":
        delta_frag = (2.0 * torch.rand(x.shape[0], frag_dim, device=x.device) - 1.0) * config.radius
    elif config.family == "l2":
        delta_frag = torch.randn(x.shape[0], frag_dim, device=x.device)
        delta_frag = project_l2(delta_frag, radius=config.radius)
    else:
        raise ValueError(f"Unknown attack family: {config.family}")
    return zero_non_fragile(delta_frag, x=x, block_slices=block_slices)


def sample_stochastic_perturbations(
    x: torch.Tensor,
    config: AttackConfig,
    block_slices: BlockSlices,
    num_samples: int,
) -> torch.Tensor:
    samples = [sample_random_delta(x, config=config, block_slices=block_slices) for _ in range(num_samples)]
    return torch.stack(samples, dim=0)


class PGDAttack:
    def __init__(self, config: AttackConfig, block_slices: BlockSlices) -> None:
        self.config = config
        self.block_slices = block_slices

    def _project(self, delta_frag: torch.Tensor) -> torch.Tensor:
        if self.config.family == "linf":
            return project_linf(delta_frag, radius=self.config.radius)
        if self.config.family == "l2":
            return project_l2(delta_frag, radius=self.config.radius)
        raise ValueError(f"Unknown attack family: {self.config.family}")

    def _step(self, delta_frag: torch.Tensor, grad_frag: torch.Tensor) -> torch.Tensor:
        if self.config.family == "linf":
            updated = delta_frag + self.config.step_size * grad_frag.sign()
        elif self.config.family == "l2":
            normalized = grad_frag / grad_frag.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            updated = delta_frag + self.config.step_size * normalized
        else:
            raise ValueError(f"Unknown attack family: {self.config.family}")
        return self._project(updated)

    def run(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> AttackResult:
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            clean_losses = (model(x) - y).pow(2)

        best_losses = clean_losses.clone()
        best_adv = x.clone()
        best_delta = torch.zeros_like(x)

        for _ in range(max(1, self.config.restarts)):
            delta = sample_random_delta(x, self.config, self.block_slices) if self.config.random_start else torch.zeros_like(x)
            delta = delta.detach()
            for _ in range(self.config.steps):
                adv = (x + delta).detach().requires_grad_(True)
                losses = (model(adv) - y).pow(2)
                loss = losses.mean()
                grad = torch.autograd.grad(loss, adv, only_inputs=True)[0].detach()
                delta_frag = fragile_delta(delta, self.block_slices)
                grad_frag = fragile_delta(grad, self.block_slices)
                updated_frag = self._step(delta_frag, grad_frag)
                delta = zero_non_fragile(updated_frag, x=x, block_slices=self.block_slices).detach()

            with torch.no_grad():
                candidate = x + delta
                candidate_losses = (model(candidate) - y).pow(2)
                improved = candidate_losses > best_losses
                best_losses = torch.where(improved, candidate_losses, best_losses)
                best_adv = torch.where(improved.unsqueeze(1), candidate, best_adv)
                best_delta = torch.where(improved.unsqueeze(1), delta, best_delta)

        frag = fragile_delta(best_delta, self.block_slices)
        diagnostics = {
            "adv_loss_mean": float(best_losses.mean().item()),
            "clean_loss_mean": float(clean_losses.mean().item()),
            "delta_linf_mean": float(frag.abs().max(dim=1).values.mean().item()),
            "delta_l2_mean": float(frag.norm(p=2, dim=1).mean().item()),
            "success_rate": float((best_losses > clean_losses).float().mean().item()),
            "elapsed_sec": time.perf_counter() - start,
        }
        return AttackResult(adv_inputs=best_adv, losses=best_losses, clean_losses=clean_losses, deltas=best_delta, diagnostics=diagnostics)


def build_attack(config: AttackConfig, block_slices: BlockSlices) -> PGDAttack:
    return PGDAttack(config=config, block_slices=block_slices)
