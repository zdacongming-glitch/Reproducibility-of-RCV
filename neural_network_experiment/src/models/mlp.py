from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig
from src.data.synthetic_teacher_student import BlockSlices


def build_activation(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "relu":
        return nn.ReLU()
    if lowered == "gelu":
        return nn.GELU()
    if lowered == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig) -> None:
        super().__init__()
        if config.depth < 1:
            raise ValueError("Model depth must be at least 1.")

        layers: list[nn.Module] = []
        if config.depth == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers.append(nn.Linear(input_dim, config.width))
            for _ in range(config.depth - 2):
                layers.extend(
                    [
                        build_activation(config.activation),
                        nn.Dropout(p=config.dropout),
                        nn.Linear(config.width, config.width),
                    ]
                )
            layers.extend(
                [
                    build_activation(config.activation),
                    nn.Dropout(p=config.dropout),
                    nn.Linear(config.width, 1),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class CoreOnlyRegressor(nn.Module):
    def __init__(self, base_model: nn.Module, block_slices: BlockSlices) -> None:
        super().__init__()
        self.base_model = base_model
        self.block_slices = block_slices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x[:, self.block_slices.core])
