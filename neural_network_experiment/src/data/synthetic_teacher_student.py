from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import RootConfig


@dataclass(frozen=True)
class BlockSlices:
    core: slice
    frag: slice
    noise: slice

    @property
    def total_dim(self) -> int:
        return self.noise.stop


@dataclass
class TeacherNetwork:
    core_hidden: np.ndarray
    core_out: np.ndarray
    frag_hidden: np.ndarray
    frag_out: np.ndarray

    def _relu(self, values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    def core_response(self, x_core: np.ndarray) -> np.ndarray:
        hidden = self._relu(x_core @ self.core_hidden.T)
        return hidden @ self.core_out

    def frag_response(self, x_frag: np.ndarray) -> np.ndarray:
        hidden = self._relu(x_frag @ self.frag_hidden.T)
        return hidden @ self.frag_out

    def predict(self, x_core: np.ndarray, x_frag: np.ndarray, lambda_value: float) -> np.ndarray:
        return self.core_response(x_core) + lambda_value * self.frag_response(x_frag)


@dataclass
class SyntheticArrays:
    x: np.ndarray
    y: np.ndarray
    block_slices: BlockSlices

    @property
    def x_core(self) -> np.ndarray:
        return self.x[:, self.block_slices.core]

    @property
    def x_frag(self) -> np.ndarray:
        return self.x[:, self.block_slices.frag]

    @property
    def x_noise(self) -> np.ndarray:
        return self.x[:, self.block_slices.noise]


@dataclass
class SyntheticReplication:
    teacher: TeacherNetwork
    train: SyntheticArrays
    test: SyntheticArrays
    replication_seed: int
    teacher_seed: int
    lambda_value: float
    n_total: int


def make_block_slices(d_core: int, d_frag: int, d_noise: int) -> BlockSlices:
    core = slice(0, d_core)
    frag = slice(d_core, d_core + d_frag)
    noise = slice(d_core + d_frag, d_core + d_frag + d_noise)
    return BlockSlices(core=core, frag=frag, noise=noise)


def sample_teacher(
    d_core: int,
    d_frag: int,
    m_core: int,
    m_frag: int,
    seed: int,
) -> TeacherNetwork:
    rng = np.random.default_rng(seed)
    return TeacherNetwork(
        core_hidden=rng.normal(size=(m_core, d_core)).astype(np.float32),
        core_out=rng.normal(size=(m_core,)).astype(np.float32),
        frag_hidden=rng.normal(size=(m_frag, d_frag)).astype(np.float32),
        frag_out=rng.normal(size=(m_frag,)).astype(np.float32),
    )


def standardize_blocks(x: np.ndarray, block_slices: BlockSlices) -> np.ndarray:
    standardized = x.copy()
    for block in (block_slices.core, block_slices.frag, block_slices.noise):
        values = standardized[:, block]
        mean = values.mean(axis=0, keepdims=True)
        std = values.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        standardized[:, block] = (values - mean) / std
    return standardized


def generate_covariates(n_samples: int, block_slices: BlockSlices, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, block_slices.total_dim)).astype(np.float32)


def generate_dataset(
    n_samples: int,
    d_core: int,
    d_frag: int,
    d_noise: int,
    teacher: TeacherNetwork,
    lambda_value: float,
    sigma: float,
    seed: int,
    standardize: bool = True,
) -> SyntheticArrays:
    block_slices = make_block_slices(d_core, d_frag, d_noise)
    x = generate_covariates(n_samples, block_slices, seed)
    if standardize:
        x = standardize_blocks(x, block_slices)
    rng = np.random.default_rng(seed + 17)
    signal = teacher.predict(x[:, block_slices.core], x[:, block_slices.frag], lambda_value=lambda_value)
    noise = rng.normal(loc=0.0, scale=sigma, size=n_samples).astype(np.float32)
    y = (signal + noise).astype(np.float32)
    return SyntheticArrays(x=x.astype(np.float32), y=y.astype(np.float32), block_slices=block_slices)


def generate_replication(
    config: RootConfig,
    seed: int,
    n_total: int | None = None,
    lambda_value: float | None = None,
    teacher_seed: int | None = None,
) -> SyntheticReplication:
    n_total = n_total if n_total is not None else config.data.n_total
    lambda_value = lambda_value if lambda_value is not None else config.teacher.lambda_value

    resolved_teacher_seed = teacher_seed if teacher_seed is not None else config.teacher.teacher_seed
    if config.data.regenerate_teacher_each_replication:
        resolved_teacher_seed = resolved_teacher_seed + seed

    teacher = sample_teacher(
        d_core=config.data.d_core,
        d_frag=config.data.d_frag,
        m_core=config.teacher.m_core,
        m_frag=config.teacher.m_frag,
        seed=resolved_teacher_seed,
    )
    train_dataset = generate_dataset(
        n_samples=n_total,
        d_core=config.data.d_core,
        d_frag=config.data.d_frag,
        d_noise=config.data.d_noise,
        teacher=teacher,
        lambda_value=lambda_value,
        sigma=config.teacher.sigma,
        seed=config.teacher.data_seed_offset + seed,
        standardize=config.data.standardize_blocks,
    )
    test_dataset = generate_dataset(
        n_samples=n_total,
        d_core=config.data.d_core,
        d_frag=config.data.d_frag,
        d_noise=config.data.d_noise,
        teacher=teacher,
        lambda_value=lambda_value,
        sigma=config.teacher.sigma,
        seed=config.teacher.test_seed_offset + seed,
        standardize=config.data.standardize_blocks,
    )
    return SyntheticReplication(
        teacher=teacher,
        train=train_dataset,
        test=test_dataset,
        replication_seed=seed,
        teacher_seed=resolved_teacher_seed,
        lambda_value=lambda_value,
        n_total=n_total,
    )
