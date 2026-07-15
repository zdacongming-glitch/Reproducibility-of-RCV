from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew, spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .aacv_core import HermiteVariant, stable_seed
from .experiment import FEATURE_COLUMNS, make_models


PRIMARY_REFERENCE = "HistGB"
PRIMARY_SCALE = "oof_residual_histgb"
PRIMARY_ANALYSIS_VARIANT = "primary_histgb_oof_j2_b6"


def model_slug(model: str) -> str:
    return {
        "OLS": "ols",
        "RidgeCV": "ridgecv",
        "LassoCV": "lassocv",
        "KNN": "knn",
        "RandomForest": "randomforest",
        "HistGB": "histgb",
    }[model]


@dataclass(frozen=True)
class AnalysisVariant:
    name: str
    role: str
    working_reference_learner: str
    scale_estimator: str
    hermite_degree: int
    envelope_multiplier: float

    @property
    def hermite(self) -> HermiteVariant:
        return HermiteVariant(
            self.name,
            self.hermite_degree,
            self.envelope_multiplier,
        )


def required_analysis_variants(
    candidate_order: tuple[str, ...],
) -> tuple[AnalysisVariant, ...]:
    variants = [
        AnalysisVariant(
            PRIMARY_ANALYSIS_VARIANT,
            "primary",
            PRIMARY_REFERENCE,
            PRIMARY_SCALE,
            2,
            6.0,
        )
    ]
    variants.extend(
        AnalysisVariant(
            f"reference_{model_slug(model)}_oof_j2_b6",
            "reference_sensitivity",
            model,
            f"oof_residual_{model_slug(model)}",
            2,
            6.0,
        )
        for model in candidate_order
        if model != PRIMARY_REFERENCE
    )
    variants.extend(
        [
            AnalysisVariant(
                "scale_tong_wang_j2_b6",
                "scale_sensitivity",
                PRIMARY_REFERENCE,
                "tong_wang",
                2,
                6.0,
            ),
            AnalysisVariant(
                "scale_fan_yao_j2_b6",
                "scale_sensitivity",
                PRIMARY_REFERENCE,
                "fan_yao_histgb",
                2,
                6.0,
            ),
            AnalysisVariant(
                "hermite_histgb_oof_j1_b6",
                "hermite_sensitivity",
                PRIMARY_REFERENCE,
                PRIMARY_SCALE,
                1,
                6.0,
            ),
            AnalysisVariant(
                "hermite_histgb_oof_j3_b6",
                "hermite_sensitivity",
                PRIMARY_REFERENCE,
                PRIMARY_SCALE,
                3,
                6.0,
            ),
            AnalysisVariant(
                "hermite_histgb_oof_j2_b4",
                "hermite_sensitivity",
                PRIMARY_REFERENCE,
                PRIMARY_SCALE,
                2,
                4.0,
            ),
            AnalysisVariant(
                "hermite_histgb_oof_j2_b8",
                "hermite_sensitivity",
                PRIMARY_REFERENCE,
                PRIMARY_SCALE,
                2,
                8.0,
            ),
        ]
    )
    return tuple(variants)


@dataclass(frozen=True)
class ScaleEstimate:
    name: str
    sigma2_raw: float
    sigma2: float
    sigma: float
    floor_applied: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossfitReferenceResult:
    learner: str
    predictions: np.ndarray
    residuals: np.ndarray
    fold_ids: np.ndarray
    fold_sizes: tuple[int, ...]
    fold_model_seeds: tuple[int, ...]
    scale: ScaleEstimate
    residual_mean: float
    residual_mse: float
    residual_skewness: float
    residual_excess_kurtosis: float
    jarque_bera_statistic: float
    jarque_bera_pvalue: float


@dataclass(frozen=True)
class WorkingErrorResult:
    references: dict[str, CrossfitReferenceResult]
    scales: dict[str, ScaleEstimate]
    covariate_diagnostics: pd.DataFrame
    covariate_bins: pd.DataFrame
    crossfit_seed: int


def _apply_floor(
    name: str,
    sigma2_raw: float,
    y: np.ndarray,
    floor_ratio: float,
    diagnostics: dict[str, Any] | None = None,
) -> ScaleEstimate:
    floor = max(
        float(floor_ratio) * float(np.var(y, ddof=1)),
        np.finfo(float).tiny,
    )
    sigma2 = max(float(sigma2_raw), floor)
    return ScaleEstimate(
        name=name,
        sigma2_raw=float(sigma2_raw),
        sigma2=sigma2,
        sigma=math.sqrt(sigma2),
        floor_applied=bool(sigma2_raw < floor),
        diagnostics={"variance_floor": floor, **(diagnostics or {})},
    )


def crossfit_reference_library(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    candidate_order: tuple[str, ...],
    n_splits: int,
    rf_trees: int,
    hgb_max_iter: int,
    variance_floor_ratio: float,
) -> dict[str, CrossfitReferenceResult]:
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x_train and y_train have incompatible shapes")
    if n_splits < 2 or len(y) < n_splits:
        raise ValueError("n_splits must be between 2 and the training size")

    folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x))
    predictions = {
        learner: np.full(len(y), np.nan, dtype=float) for learner in candidate_order
    }
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_sizes: list[int] = []
    model_seeds: dict[str, list[int]] = {learner: [] for learner in candidate_order}
    for fold_index, (fit_idx, holdout_idx) in enumerate(folds):
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x[fit_idx])
        x_holdout = scaler.transform(x[holdout_idx])
        fold_ids[holdout_idx] = fold_index
        fold_sizes.append(len(holdout_idx))
        for learner in candidate_order:
            model_seed = stable_seed(
                seed,
                "working_reference",
                learner,
                fold_index,
            )
            template = make_models(
                len(fit_idx),
                model_seed,
                rf_trees=rf_trees,
                hgb_max_iter=hgb_max_iter,
            )[learner]
            estimator = clone(template)
            estimator.fit(x_fit, y[fit_idx])
            predictions[learner][holdout_idx] = np.asarray(
                estimator.predict(x_holdout), dtype=float
            )
            model_seeds[learner].append(model_seed)

    if np.any(fold_ids < 0):
        raise RuntimeError("Cross-fitting did not assign every training row")

    results: dict[str, CrossfitReferenceResult] = {}
    for learner in candidate_order:
        learner_predictions = predictions[learner]
        if not np.isfinite(learner_predictions).all():
            raise RuntimeError(
                f"{learner} cross-fitting did not produce finite OOF predictions"
            )
        residuals = y - learner_predictions
        scale_name = f"oof_residual_{model_slug(learner)}"
        scale = _apply_floor(
            scale_name,
            float(np.var(residuals, ddof=1)),
            y,
            variance_floor_ratio,
        )
        jb = jarque_bera(residuals)
        results[learner] = CrossfitReferenceResult(
            learner=learner,
            predictions=learner_predictions,
            residuals=residuals,
            fold_ids=fold_ids.copy(),
            fold_sizes=tuple(fold_sizes),
            fold_model_seeds=tuple(model_seeds[learner]),
            scale=scale,
            residual_mean=float(np.mean(residuals)),
            residual_mse=float(np.mean(np.square(residuals))),
            residual_skewness=float(skew(residuals, bias=False)),
            residual_excess_kurtosis=float(
                kurtosis(residuals, fisher=True, bias=False)
            ),
            jarque_bera_statistic=float(jb.statistic),
            jarque_bera_pvalue=float(jb.pvalue),
        )
    return results


def estimate_tong_wang_scale(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    neighbors: int,
    variance_floor_ratio: float,
) -> ScaleEstimate:
    x = StandardScaler().fit_transform(np.asarray(x_train, dtype=float))
    y = np.asarray(y_train, dtype=float)
    k = min(int(neighbors), len(y) - 1)
    if k < 2:
        raise ValueError("Tong-Wang estimator requires at least two neighbors")
    search = NearestNeighbors(n_neighbors=k + 1)
    search.fit(x)
    _distances, indices = search.kneighbors(x)
    edge_set: set[tuple[int, int]] = set()
    for left, row in enumerate(indices[:, 1:]):
        for right in row:
            right_int = int(right)
            edge_set.add((min(left, right_int), max(left, right_int)))
    edges = np.asarray(sorted(edge_set), dtype=int)
    squared_distance = np.sum(np.square(x[edges[:, 0]] - x[edges[:, 1]]), axis=1)
    squared_difference = np.square(y[edges[:, 0]] - y[edges[:, 1]]) / 2.0
    design = np.column_stack([np.ones(len(edges)), squared_distance])
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        design, squared_difference, rcond=None
    )
    if rank < 2:
        raise RuntimeError("Tong-Wang local-distance regression is rank deficient")
    fitted = design @ coefficients
    total = float(np.sum(np.square(squared_difference - squared_difference.mean())))
    residual = float(np.sum(np.square(squared_difference - fitted)))
    diagnostics = {
        "neighbors": k,
        "pair_count": int(len(edges)),
        "distance2_q10": float(np.quantile(squared_distance, 0.10)),
        "distance2_median": float(np.median(squared_distance)),
        "distance2_q90": float(np.quantile(squared_distance, 0.90)),
        "distance_slope": float(coefficients[1]),
        "regression_r2": float(1.0 - residual / total) if total > 0 else 0.0,
    }
    return _apply_floor(
        "tong_wang",
        float(coefficients[0]),
        y,
        variance_floor_ratio,
        diagnostics,
    )


def estimate_fan_yao_scale(
    x_train: np.ndarray,
    y_train: np.ndarray,
    residuals: np.ndarray,
    *,
    neighbors: int,
    batch_rows: int,
    variance_floor_ratio: float,
) -> ScaleEstimate:
    x = StandardScaler().fit_transform(np.asarray(x_train, dtype=float))
    y = np.asarray(y_train, dtype=float)
    squared_residuals = np.square(np.asarray(residuals, dtype=float))
    k = min(int(neighbors), len(y) - 1)
    if k <= x.shape[1] + 1:
        raise ValueError("Fan-Yao estimator has too few local neighbors")
    search = NearestNeighbors(n_neighbors=k + 1)
    search.fit(x)
    distances, indices = search.kneighbors(x)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    local_variance = np.empty(len(y), dtype=float)
    for start in range(0, len(y), int(batch_rows)):
        stop = min(start + int(batch_rows), len(y))
        target = x[start:stop]
        neighbor_indices = indices[start:stop]
        offsets = x[neighbor_indices] - target[:, None, :]
        local_distance = distances[start:stop]
        bandwidth = np.maximum(local_distance[:, -1], np.finfo(float).eps)
        ratio = np.clip(local_distance / bandwidth[:, None], 0.0, 1.0)
        weights = np.power(1.0 - np.power(ratio, 3.0), 3.0)
        design = np.concatenate(
            [np.ones((*offsets.shape[:2], 1), dtype=float), offsets], axis=2
        )
        response = squared_residuals[neighbor_indices]
        normal = np.einsum("bki,bk,bkj->bij", design, weights, design)
        right = np.einsum("bki,bk,bk->bi", design, weights, response)
        slope_trace = np.trace(normal[:, 1:, 1:], axis1=1, axis2=2)
        ridge = np.maximum(slope_trace, 1.0) * 1e-10
        for coordinate in range(1, normal.shape[1]):
            normal[:, coordinate, coordinate] += ridge
        # Keep the batched right-hand side explicit for NumPy 1.x and 2.x.
        coefficients = np.linalg.solve(normal, right[..., None])[..., 0]
        local_variance[start:stop] = coefficients[:, 0]
    if not np.isfinite(local_variance).all():
        raise FloatingPointError("Fan-Yao local variance estimates are non-finite")
    diagnostics = {
        "neighbors": k,
        "batch_rows": int(batch_rows),
        "negative_local_rate": float(np.mean(local_variance < 0.0)),
        "local_variance_q10": float(np.quantile(local_variance, 0.10)),
        "local_variance_median": float(np.median(local_variance)),
        "local_variance_q90": float(np.quantile(local_variance, 0.90)),
    }
    return _apply_floor(
        "fan_yao_histgb",
        float(np.mean(local_variance)),
        y,
        variance_floor_ratio,
        diagnostics,
    )


def residual_covariate_tables(
    x_train: np.ndarray,
    references: dict[str, CrossfitReferenceResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = np.asarray(x_train, dtype=float)
    diagnostic_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for learner, result in references.items():
        residual = result.residuals
        squared = np.square(residual)
        for column_index, covariate in enumerate(FEATURE_COLUMNS):
            values = x[:, column_index]
            pearson_residual = float(np.corrcoef(values, residual)[0, 1])
            pearson_squared = float(np.corrcoef(values, squared)[0, 1])
            spearman_residual = float(spearmanr(values, residual).statistic)
            spearman_squared = float(spearmanr(values, squared).statistic)
            residual_slope = float(np.polyfit(values, residual, 1)[0])
            squared_slope = float(np.polyfit(values, squared, 1)[0])
            diagnostic_rows.append(
                {
                    "working_reference_learner": learner,
                    "covariate": covariate,
                    "pearson_residual": pearson_residual,
                    "pearson_squared_residual": pearson_squared,
                    "spearman_residual": spearman_residual,
                    "spearman_squared_residual": spearman_squared,
                    "residual_slope": residual_slope,
                    "squared_residual_slope": squared_slope,
                }
            )
            frame = pd.DataFrame(
                {"x": values, "residual": residual, "squared": squared}
            )
            frame["bin"] = pd.qcut(frame["x"], q=10, labels=False, duplicates="drop")
            grouped = frame.groupby("bin", observed=True)
            for bin_index, cell in grouped:
                bin_rows.append(
                    {
                        "working_reference_learner": learner,
                        "covariate": covariate,
                        "bin": int(bin_index),
                        "count": int(len(cell)),
                        "covariate_mean": float(cell["x"].mean()),
                        "residual_mean": float(cell["residual"].mean()),
                        "squared_residual_mean": float(cell["squared"].mean()),
                    }
                )
    return pd.DataFrame(diagnostic_rows), pd.DataFrame(bin_rows)


def build_working_error(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    candidate_order: tuple[str, ...],
    crossfit_folds: int,
    rf_trees: int,
    hgb_max_iter: int,
    variance_floor_ratio: float,
    tong_wang_neighbors: int,
    fan_yao_neighbors: int,
    fan_yao_batch_rows: int,
) -> WorkingErrorResult:
    references = crossfit_reference_library(
        x_train,
        y_train,
        seed=seed,
        candidate_order=candidate_order,
        n_splits=crossfit_folds,
        rf_trees=rf_trees,
        hgb_max_iter=hgb_max_iter,
        variance_floor_ratio=variance_floor_ratio,
    )
    scales = {result.scale.name: result.scale for result in references.values()}
    scales["tong_wang"] = estimate_tong_wang_scale(
        x_train,
        y_train,
        neighbors=tong_wang_neighbors,
        variance_floor_ratio=variance_floor_ratio,
    )
    scales["fan_yao_histgb"] = estimate_fan_yao_scale(
        x_train,
        y_train,
        references[PRIMARY_REFERENCE].residuals,
        neighbors=fan_yao_neighbors,
        batch_rows=fan_yao_batch_rows,
        variance_floor_ratio=variance_floor_ratio,
    )
    covariate_diagnostics, covariate_bins = residual_covariate_tables(
        x_train, references
    )
    return WorkingErrorResult(
        references=references,
        scales=scales,
        covariate_diagnostics=covariate_diagnostics,
        covariate_bins=covariate_bins,
        crossfit_seed=int(seed),
    )
