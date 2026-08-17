from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from src.attacks.restricted_attacks import sample_stochastic_perturbations
from src.config import AttackConfig, RootConfig
from src.data.synthetic_teacher_student import SyntheticArrays
from src.trainers.trainers import build_trainer


@dataclass
class CVSelectionResult:
    per_radius_scores: dict[float, dict[str, dict[str, float]]]
    per_radius_test_scores: dict[float, dict[str, dict[str, float]]]
    selected_by_radius: dict[float, str]
    oracle_by_radius: dict[float, str]
    fit_summaries: dict[str, dict[str, float]]
    metadata: dict[str, float | int | str]


def subset_arrays(arrays: SyntheticArrays, indices: np.ndarray) -> SyntheticArrays:
    return SyntheticArrays(
        x=arrays.x[indices].copy(),
        y=arrays.y[indices].copy(),
        block_slices=arrays.block_slices,
    )


def make_holdout_split(n_samples: int, holdout_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_val = max(1, int(round(n_samples * holdout_ratio)))
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])
    return train_idx, val_idx


def make_disjoint_holdout_splits(n_samples: int, repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use each random fold once as train set and its complement as validation."""
    if repeats < 2:
        raise ValueError("Disjoint holdout requires repeats >= 2.")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = [np.sort(fold) for fold in np.array_split(indices, repeats)]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(repeats):
        train_idx = folds[fold_idx]
        val_idx = np.sort(np.concatenate([folds[j] for j in range(repeats) if j != fold_idx]))
        splits.append((train_idx, val_idx))
    return splits


def make_kfold_splits(n_samples: int, num_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, num_folds)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(num_folds):
        val_idx = np.sort(folds[fold_idx])
        train_idx = np.sort(np.concatenate([folds[j] for j in range(num_folds) if j != fold_idx]))
        splits.append((train_idx, val_idx))
    return splits


def compute_srcv_score(trainer, val_data: SyntheticArrays, attack_cfg: AttackConfig, common_randomness: torch.Tensor) -> float:
    return float(
        trainer.evaluate_stochastic(
            val_data,
            attack_config=attack_cfg,
            common_noise=common_randomness,
        )["stochastic_robust_mse"]
    )


def make_common_randomness(val_data: SyntheticArrays, attack_cfg: AttackConfig, num_samples: int, seed: int) -> torch.Tensor:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        x_val = torch.from_numpy(val_data.x).float()
        return sample_stochastic_perturbations(
            x=x_val,
            config=attack_cfg,
            block_slices=val_data.block_slices,
            num_samples=num_samples,
        )


_make_common_randomness = make_common_randomness


def _deterministic_argmin(scores: dict[str, float], tie_break_order: list[str]) -> str:
    order_map = {name: idx for idx, name in enumerate(tie_break_order)}
    return min(scores, key=lambda name: (scores[name], order_map.get(name, len(order_map)), name))


def _iter_splits(config: RootConfig, n_samples: int, seed: int):
    if config.cv.method == "kfold":
        yield from make_kfold_splits(n_samples=n_samples, num_folds=config.cv.num_folds, seed=seed)
        return
    disjoint_holdout_ratio = 1.0 - 1.0 / max(config.cv.repeats, 1)
    if config.cv.repeats > 1 and abs(config.cv.holdout_ratio - disjoint_holdout_ratio) < 1e-12:
        yield from make_disjoint_holdout_splits(n_samples=n_samples, repeats=config.cv.repeats, seed=seed)
        return
    for repeat in range(config.cv.repeats):
        yield make_holdout_split(n_samples=n_samples, holdout_ratio=config.cv.holdout_ratio, seed=seed + repeat)


def _attack_with_radius(base: AttackConfig, radius: float) -> AttackConfig:
    attack = copy.deepcopy(base)
    attack.radius = radius
    return attack


def select_procedure(
    train_data: SyntheticArrays,
    test_data: SyntheticArrays,
    config: RootConfig,
    procedures: list[str],
    seed: int,
    train_attack_config: AttackConfig,
    validation_attack_config: AttackConfig,
    eval_radii: list[float],
) -> CVSelectionResult:
    srcv_scores: dict[float, dict[str, list[float]]] = {
        float(radius): {procedure: [] for procedure in procedures} for radius in eval_radii
    }
    test_scores: dict[float, dict[str, list[float]]] = {
        float(radius): {procedure: [] for procedure in procedures} for radius in eval_radii
    }
    fit_times: dict[str, list[float]] = {procedure: [] for procedure in procedures}
    fit_epochs: dict[str, list[float]] = {procedure: [] for procedure in procedures}
    fit_metrics: dict[str, list[float]] = {procedure: [] for procedure in procedures}
    test_common_randomness: dict[float, torch.Tensor] = {}
    for radius_index, radius in enumerate(eval_radii):
        radius_key = float(radius)
        radius_attack = _attack_with_radius(validation_attack_config, radius=radius_key)
        test_common_randomness[radius_key] = make_common_randomness(
            val_data=test_data,
            attack_cfg=radius_attack,
            num_samples=config.attacks.sr.num_samples,
            seed=seed + 500000 + radius_index,
        )

    for split_id, (train_idx, val_idx) in enumerate(_iter_splits(config, train_data.x.shape[0], seed=seed)):
        inner_train = subset_arrays(train_data, train_idx)
        inner_val = subset_arrays(train_data, val_idx)
        split_trainers = {}
        for procedure in procedures:
            trainer = build_trainer(
                config=config,
                procedure=procedure,
                block_slices=train_data.block_slices,
                train_attack_config=train_attack_config,
                eval_attack_config=train_attack_config,
            )
            fit_result = trainer.fit(inner_train, inner_val)
            split_trainers[procedure] = trainer
            fit_times[procedure].append(float(fit_result.train_seconds))
            fit_epochs[procedure].append(float(fit_result.best_epoch))
            fit_metrics[procedure].append(float(fit_result.best_metric))

        for radius_index, radius in enumerate(eval_radii):
            radius_key = float(radius)
            radius_attack = _attack_with_radius(validation_attack_config, radius=radius_key)
            common_randomness = make_common_randomness(
                val_data=inner_val,
                attack_cfg=radius_attack,
                num_samples=config.attacks.sr.num_samples,
                seed=seed + 1000 + split_id * 10000 + radius_index,
            )
            for procedure, trainer in split_trainers.items():
                srcv_scores[radius_key][procedure].append(
                    compute_srcv_score(trainer, inner_val, radius_attack, common_randomness)
                )
                test_scores[radius_key][procedure].append(
                    float(
                        trainer.evaluate_stochastic(
                            test_data,
                            attack_config=radius_attack,
                            common_noise=test_common_randomness[radius_key],
                        )["stochastic_robust_mse"]
                    )
                )

    per_radius_scores: dict[float, dict[str, dict[str, float]]] = {}
    per_radius_test_scores: dict[float, dict[str, dict[str, float]]] = {}
    selected_by_radius: dict[float, str] = {}
    oracle_by_radius: dict[float, str] = {}
    for radius, procedure_scores in srcv_scores.items():
        means = {procedure: float(np.mean(values)) for procedure, values in procedure_scores.items()}
        test_means = {procedure: float(np.mean(test_scores[radius][procedure])) for procedure in procedures}
        selected_by_radius[radius] = _deterministic_argmin(means, config.cv.tie_break_order)
        oracle_by_radius[radius] = _deterministic_argmin(test_means, config.cv.tie_break_order)
        per_radius_scores[radius] = {
            procedure: {
                "srcv_mean": means[procedure],
                "srcv_std": float(np.std(procedure_scores[procedure])),
            }
            for procedure in procedures
        }
        per_radius_test_scores[radius] = {
            procedure: {
                "test_mean": test_means[procedure],
                "test_std": float(np.std(test_scores[radius][procedure])),
            }
            for procedure in procedures
        }

    fit_summaries = {
        procedure: {
            "training_time_sec": float(np.sum(fit_times[procedure])),
            "best_epoch": float(np.mean(fit_epochs[procedure])),
            "best_metric": float(np.mean(fit_metrics[procedure])),
            "num_split_models": float(len(fit_times[procedure])),
        }
        for procedure in procedures
    }

    return CVSelectionResult(
        per_radius_scores=per_radius_scores,
        per_radius_test_scores=per_radius_test_scores,
        selected_by_radius=selected_by_radius,
        oracle_by_radius=oracle_by_radius,
        fit_summaries=fit_summaries,
        metadata={
            "seed": seed,
            "validation_family": validation_attack_config.family,
            "train_radius": train_attack_config.radius,
            "cv_method": config.cv.method,
            "cv_repeats": config.cv.repeats,
            "cv_holdout_ratio": config.cv.holdout_ratio,
        },
    )
