from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.attacks.restricted_attacks import (
    build_attack,
    sample_stochastic_perturbations,
)
from src.config import AttackConfig, RootConfig, StochasticRobustConfig
from src.data.synthetic_teacher_student import BlockSlices, SyntheticArrays
from src.models.mlp import CoreOnlyRegressor, MLPRegressor


@dataclass
class FitResult:
    best_metric: float
    best_epoch: int
    history: list[dict[str, float]]
    train_seconds: float


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def arrays_to_loader(arrays: SyntheticArrays, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(arrays.x).float(), torch.from_numpy(arrays.y).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(config: RootConfig, procedure: str, block_slices: BlockSlices) -> nn.Module:
    input_dim = block_slices.total_dim
    if procedure == "core_only":
        base = MLPRegressor(input_dim=block_slices.core.stop - block_slices.core.start, config=config.model)
        return CoreOnlyRegressor(base_model=base, block_slices=block_slices)
    return MLPRegressor(input_dim=input_dim, config=config.model)


class BaseTrainer:
    def __init__(
        self,
        config: RootConfig,
        procedure: str,
        block_slices: BlockSlices,
        train_attack_config: AttackConfig | None = None,
        eval_attack_config: AttackConfig | None = None,
        sr_config: StochasticRobustConfig | None = None,
    ) -> None:
        self.config = config
        self.procedure = procedure
        self.block_slices = block_slices
        self.device = choose_device(config.experiment.device)
        self.model = build_model(config, procedure=procedure, block_slices=block_slices).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        self.train_attack_config = train_attack_config or config.attacks.train
        self.eval_attack_config = eval_attack_config or config.attacks.eval
        self.sr_config = sr_config or config.attacks.sr

    def compute_batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        predictions = self.model(x)
        return (predictions - y).pow(2).mean()

    def fit(self, train_data: SyntheticArrays, val_data: SyntheticArrays) -> FitResult:
        train_loader = arrays_to_loader(train_data, batch_size=self.config.training.batch_size, shuffle=True)
        history: list[dict[str, float]] = []
        best_metric = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        bad_epochs = 0
        started = time.perf_counter()

        for epoch in range(1, self.config.training.epochs + 1):
            epoch_loss = self._run_epoch(train_loader)
            stochastic_val = self.evaluate_stochastic(val_data, attack_config=self.train_attack_config)["stochastic_robust_mse"]
            metric = stochastic_val
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(epoch_loss),
                    "stochastic_val_loss": float(stochastic_val),
                }
            )
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.training.patience:
                    break

        self.model.load_state_dict(best_state)
        return FitResult(
            best_metric=best_metric,
            best_epoch=best_epoch,
            history=history,
            train_seconds=time.perf_counter() - started,
        )

    def _run_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        batch_losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.compute_batch_loss(x_batch, y_batch)
            loss.backward()
            self.optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        return float(np.mean(batch_losses)) if batch_losses else 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(x).float().to(self.device)
            predictions = self.model(tensor_x).detach().cpu().numpy()
        return predictions.astype(np.float32)

    def evaluate_stochastic(
        self,
        arrays: SyntheticArrays,
        attack_config: AttackConfig | None = None,
        sr_config: StochasticRobustConfig | None = None,
        common_noise: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        chosen_attack = attack_config or self.eval_attack_config
        chosen_sr = sr_config or self.sr_config
        x = torch.from_numpy(arrays.x).float().to(self.device)
        y = torch.from_numpy(arrays.y).float().to(self.device)
        if common_noise is None:
            perturbations = sample_stochastic_perturbations(
                x=x,
                config=chosen_attack,
                block_slices=self.block_slices,
                num_samples=chosen_sr.num_samples,
            )
        else:
            perturbations = common_noise.to(self.device)

        losses: list[torch.Tensor] = []
        with torch.no_grad():
            for sample_idx in range(perturbations.shape[0]):
                predictions = self.model(x + perturbations[sample_idx])
                losses.append((predictions - y).pow(2))
        stacked = torch.stack(losses, dim=0)
        return {
            "stochastic_robust_mse": float(stacked.mean().item()),
            "stochastic_loss_std": float(stacked.mean(dim=1).std(unbiased=False).item()),
        }


class ERMTrainer(BaseTrainer):
    pass


class AdversarialTrainer(BaseTrainer):
    def compute_batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        attack = build_attack(self.train_attack_config, self.block_slices)
        attacked = attack.run(self.model, x=x, y=y).adv_inputs.detach()
        predictions = self.model(attacked)
        return (predictions - y).pow(2).mean()


class StochasticRobustTrainer(BaseTrainer):
    def compute_batch_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        perturbations = sample_stochastic_perturbations(
            x=x,
            config=self.train_attack_config,
            block_slices=self.block_slices,
            num_samples=self.sr_config.num_samples,
        )
        losses: list[torch.Tensor] = []
        for sample_idx in range(perturbations.shape[0]):
            predictions = self.model(x + perturbations[sample_idx])
            losses.append((predictions - y).pow(2))
        return torch.stack(losses, dim=0).mean()


class CoreOnlyTrainer(BaseTrainer):
    pass


def build_trainer(
    config: RootConfig,
    procedure: str,
    block_slices: BlockSlices,
    train_attack_config: AttackConfig | None = None,
    eval_attack_config: AttackConfig | None = None,
) -> BaseTrainer:
    common_kwargs = {
        "config": config,
        "procedure": procedure,
        "block_slices": block_slices,
        "train_attack_config": train_attack_config,
        "eval_attack_config": eval_attack_config,
        "sr_config": config.attacks.sr,
    }
    if procedure == "erm":
        return ERMTrainer(**common_kwargs)
    if procedure == "at":
        return AdversarialTrainer(**common_kwargs)
    if procedure == "sr":
        return StochasticRobustTrainer(**common_kwargs)
    if procedure == "core_only":
        return CoreOnlyTrainer(**common_kwargs)
    raise ValueError(f"Unknown procedure: {procedure}")
