from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .aacv_core import (
    ExtremaResult,
    HermiteVariant,
    MultiPathResult,
    OptimizationBudget,
    aacv_observation_scores,
    continuous_box_path_search,
    linear_box_extrema,
    stable_seed,
    strict_stronger_box_extrema,
)
from .aacv_reference import (
    canonical_json_fingerprint,
    load_reference_spec,
)
from .aacv_working_error import (
    PRIMARY_ANALYSIS_VARIANT,
    AnalysisVariant,
    WorkingErrorResult,
    build_working_error,
    required_analysis_variants,
)
from .config import PROJECT_DIR, require_data_file, sha256_file
from .experiment import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    choose_min,
    deterministic_seed,
    load_ccpp_data,
    make_models,
    majority_vote,
    standard_error,
)


LINEAR_MODELS = frozenset({"OLS", "RidgeCV", "LassoCV"})
AACV_SCIENTIFIC_SCHEMA_VERSION = 2
ATTACK_ALGORITHM = "multipath_strict_envelope_v1"
SHARD_TABLES = (
    "inner_aacv_scores",
    "inner_test_aacv_scores",
    "inner_aacv_selections",
    "outer_aacv_summary",
    "outer_candidate_test_scores",
    "working_reference_diagnostics",
    "scale_estimator_diagnostics",
    "residual_covariate_diagnostics",
    "residual_covariate_bins",
    "attack_diagnostics",
    "attack_path_calibration",
)
ATTACK_DIAGNOSTIC_COLUMNS = (
    "outer",
    "inner",
    "stage",
    "radius",
    "model",
    "attack_algorithm",
    "primary_path_count",
    "primary_sobol_starts",
    "additional_sobol_starts",
    "inherited_paths",
    "new_paths_per_direction",
    "stronger_min_step_fraction",
    "audit_observations",
    "maximum_gain_min",
    "maximum_gain_mean",
    "maximum_gain_max",
    "minimum_gain_min",
    "minimum_gain_mean",
    "minimum_gain_max",
    "center_abs_gap_mean",
    "rho_abs_gap_mean",
    "primary_score_abs_gap_mean",
    "strict_envelope_passed",
)

PATH_CALIBRATION_COLUMNS = (
    "outer",
    "inner",
    "stage",
    "radius",
    "model",
    "analysis_variant",
    "path_count",
    "reference_path_count",
    "aacv_score",
    "reference_aacv_score",
    "score_symmetric_relative_gap",
    "maximum_outward_gap_p99",
    "minimum_outward_gap_p99",
    "response_sd",
    "maximum_outward_gap_p99_response_sd",
    "minimum_outward_gap_p99_response_sd",
    "selected_model_at_path_count",
    "selected_model_at_reference",
    "selection_matches_reference",
)


@dataclass(frozen=True)
class PathCalibrationConfig:
    enabled: bool
    path_counts: tuple[int, ...]
    radii: tuple[float, ...]
    score_relative_tolerance: float
    extrema_p99_response_sd_tolerance: float
    require_selection_agreement: bool


@dataclass(frozen=True)
class ModelAttackResult:
    extrema: ExtremaResult
    paths: MultiPathResult | None


@dataclass(frozen=True)
class AACVExperimentConfig:
    radii: tuple[float, ...]
    outer_repeats: int
    inner_repeats: int
    test_size: float
    validation_fraction_of_pool: float
    base_seed: int
    rf_trees: int
    hgb_max_iter: int
    candidate_order: tuple[str, ...]
    reference_crossfit_folds: int
    reference_rf_trees: int
    reference_hgb_max_iter: int
    variance_floor_ratio: float
    tong_wang_neighbors: int
    fan_yao_neighbors: int
    fan_yao_batch_rows: int
    analysis_variants: tuple[AnalysisVariant, ...]
    attack_algorithm: str
    retained_paths: int
    validation_budget: OptimizationBudget
    evaluation_budget: OptimizationBudget
    audit_enabled: bool
    audit_max_observations: int
    audit_sobol_multiplier: int
    audit_minimum_step_fraction: float
    path_calibration: PathCalibrationConfig
    progress: bool = True


def _budget_from_dict(config: dict[str, Any]) -> OptimizationBudget:
    return OptimizationBudget(
        sobol_starts=int(config["sobol_starts"]),
        step_fractions=tuple(float(value) for value in config["step_fractions"]),
        prediction_chunk_rows=int(config.get("prediction_chunk_rows", 2048)),
    )


def _validate_budget(
    budget: OptimizationBudget,
    *,
    name: str,
    required_paths: int,
) -> None:
    if budget.sobol_starts <= 0 or budget.sobol_starts & (budget.sobol_starts - 1):
        raise ValueError(f"{name} Sobol starts must be a positive power of two")
    if budget.sobol_starts < required_paths:
        raise ValueError(f"{name} Sobol starts must cover all retained paths")
    if budget.prediction_chunk_rows <= 0:
        raise ValueError(f"{name} prediction chunks must be positive")
    if not budget.step_fractions:
        raise ValueError(f"{name} refinement steps must not be empty")
    for index, step in enumerate(budget.step_fractions):
        if not math.isfinite(step) or step <= 0:
            raise ValueError(f"{name} refinement steps must be finite and positive")
        if index and not math.isclose(step, budget.step_fractions[index - 1] / 2.0):
            raise ValueError(f"{name} refinement steps must follow a halving schedule")


def _path_calibration_from_dict(config: dict[str, Any]) -> PathCalibrationConfig:
    return PathCalibrationConfig(
        enabled=bool(config["enabled"]),
        path_counts=tuple(int(value) for value in config["path_counts"]),
        radii=tuple(float(value) for value in config["radii"]),
        score_relative_tolerance=float(config["score_relative_tolerance"]),
        extrema_p99_response_sd_tolerance=float(
            config["extrema_p99_response_sd_tolerance"]
        ),
        require_selection_agreement=bool(config["require_selection_agreement"]),
    )


def _analysis_variant_from_dict(item: dict[str, Any]) -> AnalysisVariant:
    return AnalysisVariant(
        name=str(item["name"]),
        role=str(item["role"]),
        working_reference_learner=str(item["working_reference_learner"]),
        scale_estimator=str(item["scale_estimator"]),
        hermite_degree=int(item["hermite_degree"]),
        envelope_multiplier=float(item["envelope_multiplier"]),
    )


def aacv_config_from_dict(
    config: dict[str, Any], progress: bool = True
) -> AACVExperimentConfig:
    exp = config["aacv_experiment"]
    models = config["models"]
    optimizer = config["optimizer"]
    working_error = config["working_error"]
    reference = working_error["reference_protocol"]
    expected_candidates = (
        "OLS",
        "RidgeCV",
        "LassoCV",
        "KNN",
        "RandomForest",
        "HistGB",
    )
    candidate_order = tuple(models["candidate_order"])
    if candidate_order != expected_candidates:
        raise ValueError(f"Candidate order must remain fixed as {expected_candidates}")
    configured_variants = tuple(
        _analysis_variant_from_dict(item) for item in working_error["analysis_variants"]
    )
    expected_variants = required_analysis_variants(candidate_order)
    if configured_variants != expected_variants:
        raise ValueError(
            "Working-error analysis variants must be the predeclared "
            "one-nuisance-at-a-time variants"
        )
    if working_error.get("clipping") != "none":
        raise ValueError("Gaussian-Hermite clipping must remain disabled")
    if configured_variants[0].name != PRIMARY_ANALYSIS_VARIANT:
        raise ValueError("Primary analysis variant must be listed first")

    radii = tuple(float(value) for value in exp["radii"])
    if not radii or not math.isclose(radii[0], 0.0):
        raise ValueError("AACV radii must begin at zero")
    if any(not math.isfinite(radius) or radius < 0 for radius in radii):
        raise ValueError("AACV radii must be finite and nonnegative")
    if any(right <= left for left, right in zip(radii, radii[1:])):
        raise ValueError("AACV radii must be strictly increasing")
    attack_algorithm = str(optimizer["algorithm"])
    if attack_algorithm != ATTACK_ALGORITHM:
        raise ValueError(f"AACV attack algorithm must be {ATTACK_ALGORITHM}")
    retained_paths = int(optimizer["retained_paths"])
    if retained_paths <= 0:
        raise ValueError("retained_paths must be positive")
    path_calibration = _path_calibration_from_dict(optimizer["path_calibration"])
    if path_calibration.enabled:
        if path_calibration.path_counts != (1, 2, 3, 4):
            raise ValueError("Smoke path calibration must compare K=1,2,3,4")
        if any(radius <= 0 for radius in path_calibration.radii):
            raise ValueError("Path calibration radii must be nonzero")
        if any(
            not any(math.isclose(radius, configured) for configured in radii)
            for radius in path_calibration.radii
        ):
            raise ValueError("Every path calibration radius must be in the run grid")
        if not path_calibration.require_selection_agreement:
            raise ValueError("Strict path calibration requires selection agreement")
        if not math.isclose(path_calibration.score_relative_tolerance, 0.005):
            raise ValueError("Path calibration score tolerance must remain 0.005")
        if not math.isclose(path_calibration.extrema_p99_response_sd_tolerance, 0.01):
            raise ValueError("Path calibration extrema tolerance must remain 0.01")
    required_paths = (
        max(path_calibration.path_counts)
        if path_calibration.enabled
        else retained_paths
    )
    validation_budget = _budget_from_dict(optimizer["validation"])
    evaluation_budget = _budget_from_dict(optimizer["evaluation"])
    _validate_budget(
        validation_budget, name="validation", required_paths=required_paths
    )
    _validate_budget(
        evaluation_budget, name="evaluation", required_paths=required_paths
    )
    audit = optimizer["audit"]
    audit_sobol_multiplier = int(audit["sobol_multiplier"])
    if audit_sobol_multiplier != 2:
        raise ValueError("Stronger audit Sobol multiplier must remain two")
    audit_minimum_step_fraction = float(audit["minimum_step_fraction"])
    if not math.isclose(audit_minimum_step_fraction, 1.0 / 128.0):
        raise ValueError("Stronger audit minimum step must remain 1/128")

    return AACVExperimentConfig(
        radii=radii,
        outer_repeats=int(exp["outer_repeats"]),
        inner_repeats=int(exp["inner_repeats"]),
        test_size=float(exp["test_size"]),
        validation_fraction_of_pool=float(exp["validation_fraction_of_pool"]),
        base_seed=int(exp["base_seed"]),
        rf_trees=int(models["rf_trees"]),
        hgb_max_iter=int(models["hgb_max_iter"]),
        candidate_order=candidate_order,
        reference_crossfit_folds=int(reference["crossfit_folds"]),
        reference_rf_trees=int(reference["rf_trees"]),
        reference_hgb_max_iter=int(reference["hgb_max_iter"]),
        variance_floor_ratio=float(working_error["variance_floor_ratio"]),
        tong_wang_neighbors=int(working_error["tong_wang"]["neighbors"]),
        fan_yao_neighbors=int(working_error["fan_yao"]["neighbors"]),
        fan_yao_batch_rows=int(working_error["fan_yao"]["batch_rows"]),
        analysis_variants=configured_variants,
        attack_algorithm=attack_algorithm,
        retained_paths=retained_paths,
        validation_budget=validation_budget,
        evaluation_budget=evaluation_budget,
        audit_enabled=bool(audit["enabled"]),
        audit_max_observations=int(audit["max_observations"]),
        audit_sobol_multiplier=audit_sobol_multiplier,
        audit_minimum_step_fraction=audit_minimum_step_fraction,
        path_calibration=path_calibration,
        progress=progress,
    )


def experiment_fingerprint(
    config: dict[str, Any], reference_spec: dict[str, Any]
) -> str:
    payload = {
        "scientific_schema_version": AACV_SCIENTIFIC_SCHEMA_VERSION,
        "attack_algorithm": ATTACK_ALGORITHM,
        "config": {
            key: value for key, value in config.items() if key != "_config_path"
        },
        "reference_spec": reference_spec,
    }
    return canonical_json_fingerprint(payload)


def config_fingerprint(config: dict[str, Any]) -> str:
    _path, reference_spec = load_reference_spec(config)
    return experiment_fingerprint(config, reference_spec)


def _model_attack(
    model_name: str,
    estimator: object,
    x_scaled: np.ndarray,
    radius: float,
    budget: OptimizationBudget,
    seed: int,
    *,
    retained_paths: int,
    search_path_count: int,
) -> ModelAttackResult:
    if model_name in LINEAR_MODELS:
        return ModelAttackResult(
            extrema=linear_box_extrema(estimator, x_scaled, radius), paths=None
        )
    paths = continuous_box_path_search(
        estimator,
        x_scaled,
        radius,
        budget,
        seed=seed,
        path_count=search_path_count,
    )
    return ModelAttackResult(
        extrema=paths.extrema(retained_paths),
        paths=paths,
    )


def _score_diagnostics(
    y: np.ndarray,
    extrema: ExtremaResult,
    sigma: float,
    hermite: HermiteVariant,
) -> dict[str, float]:
    center = extrema.center
    rho = extrema.radius
    scores, psi, envelope = aacv_observation_scores(y, center, rho, sigma, hermite)
    return {
        "aacv_score": float(np.mean(scores)),
        "envelope": float(envelope),
        "psi_negative_rate": float(np.mean(psi < 0.0)),
        "rho_gt_envelope_rate": float(np.mean(rho > envelope)),
        "abs_y_minus_c_gt_envelope_rate": float(
            np.mean(np.abs(np.asarray(y, dtype=float) - center) > envelope)
        ),
        "center_mean": float(np.mean(center)),
        "rho_mean": float(np.mean(rho)),
        "rho_max": float(np.max(rho)),
    }


def _variant_columns(variant: AnalysisVariant) -> dict[str, Any]:
    return {
        "analysis_variant": variant.name,
        "analysis_role": variant.role,
        "working_reference_learner": variant.working_reference_learner,
        "scale_estimator": variant.scale_estimator,
        "hermite_degree": variant.hermite_degree,
        "envelope_multiplier": variant.envelope_multiplier,
    }


def _audit_indices(n_rows: int, maximum: int, seed: int) -> np.ndarray:
    count = min(int(maximum), int(n_rows))
    if count <= 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=count, replace=False))


def _attack_audit_row(
    *,
    estimator: object,
    model_name: str,
    x_scaled: np.ndarray,
    y: np.ndarray,
    sigma: float,
    radius: float,
    primary: ModelAttackResult,
    budget: OptimizationBudget,
    attack_seed: int,
    subset_seed: int,
    maximum_observations: int,
    stage: str,
    outer: int,
    inner: int,
    primary_hermite: HermiteVariant,
    retained_paths: int,
    sobol_multiplier: int,
    minimum_step_fraction: float,
) -> dict[str, object] | None:
    if model_name in LINEAR_MODELS or radius == 0 or maximum_observations <= 0:
        return None
    if primary.paths is None:
        raise RuntimeError("Nonlinear audit requires primary path state")
    indices = _audit_indices(len(x_scaled), maximum_observations, subset_seed)
    if len(indices) == 0:
        return None
    primary_paths = primary.paths.first_paths(retained_paths).subset(indices)
    primary_subset = primary_paths.extrema()
    np.testing.assert_allclose(
        primary_subset.maximum, primary.extrema.maximum[indices], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        primary_subset.minimum, primary.extrema.minimum[indices], rtol=0, atol=1e-12
    )
    stronger = strict_stronger_box_extrema(
        estimator,
        x_scaled[indices],
        radius,
        budget,
        primary_paths,
        seed=attack_seed,
        sobol_multiplier=sobol_multiplier,
        minimum_step_fraction=minimum_step_fraction,
    )
    primary_scores, _psi, _envelope = aacv_observation_scores(
        y[indices],
        primary_subset.center,
        primary_subset.radius,
        sigma,
        primary_hermite,
    )
    stronger_scores, _psi, _envelope = aacv_observation_scores(
        y[indices],
        stronger.center,
        stronger.radius,
        sigma,
        primary_hermite,
    )
    maximum_gain = stronger.maximum - primary_subset.maximum
    minimum_gain = primary_subset.minimum - stronger.minimum
    strict_passed = bool(
        np.min(maximum_gain) >= -1e-12 and np.min(minimum_gain) >= -1e-12
    )
    return {
        "outer": outer,
        "inner": inner,
        "stage": stage,
        "radius": radius,
        "model": model_name,
        "attack_algorithm": ATTACK_ALGORITHM,
        "primary_path_count": retained_paths,
        "primary_sobol_starts": budget.sobol_starts,
        "additional_sobol_starts": budget.sobol_starts * (sobol_multiplier - 1),
        "inherited_paths": retained_paths,
        "new_paths_per_direction": retained_paths,
        "stronger_min_step_fraction": minimum_step_fraction,
        "audit_observations": len(indices),
        "maximum_gain_min": float(np.min(maximum_gain)),
        "maximum_gain_mean": float(np.mean(maximum_gain)),
        "maximum_gain_max": float(np.max(maximum_gain)),
        "minimum_gain_min": float(np.min(minimum_gain)),
        "minimum_gain_mean": float(np.mean(minimum_gain)),
        "minimum_gain_max": float(np.max(minimum_gain)),
        "center_abs_gap_mean": float(
            np.mean(np.abs(stronger.center - primary_subset.center))
        ),
        "rho_abs_gap_mean": float(
            np.mean(np.abs(stronger.radius - primary_subset.radius))
        ),
        "primary_score_abs_gap_mean": float(
            np.mean(np.abs(stronger_scores - primary_scores))
        ),
        "strict_envelope_passed": strict_passed,
    }


def _attack_extrema_for_path_count(
    attack: ModelAttackResult,
    path_count: int,
) -> ExtremaResult:
    if attack.paths is None:
        return attack.extrema
    return attack.paths.extrema(path_count)


def _path_calibration_rows_for_stage(
    *,
    attack: ModelAttackResult,
    y: np.ndarray,
    working_error: WorkingErrorResult,
    variants: tuple[AnalysisVariant, ...],
    path_counts: tuple[int, ...],
    response_sd: float,
    outer: int,
    inner: int,
    stage: str,
    radius: float,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, float]]]:
    if not math.isfinite(response_sd) or response_sd <= 0:
        raise ValueError("Training response standard deviation must be positive")
    reference_path_count = max(path_counts)
    reference_extrema = _attack_extrema_for_path_count(attack, reference_path_count)
    reference_scores: dict[str, float] = {}
    for variant in variants:
        sigma = working_error.scales[variant.scale_estimator].sigma
        reference_scores[variant.name] = float(
            _score_diagnostics(y, reference_extrema, sigma, variant.hermite)[
                "aacv_score"
            ]
        )

    rows: list[dict[str, Any]] = []
    scores: dict[int, dict[str, float]] = {}
    for path_count in path_counts:
        extrema = _attack_extrema_for_path_count(attack, path_count)
        maximum_gap = np.maximum(reference_extrema.maximum - extrema.maximum, 0.0)
        minimum_gap = np.maximum(extrema.minimum - reference_extrema.minimum, 0.0)
        maximum_q99 = float(np.quantile(maximum_gap, 0.99))
        minimum_q99 = float(np.quantile(minimum_gap, 0.99))
        scores[path_count] = {}
        for variant in variants:
            sigma = working_error.scales[variant.scale_estimator].sigma
            score = float(
                _score_diagnostics(y, extrema, sigma, variant.hermite)["aacv_score"]
            )
            reference_score = reference_scores[variant.name]
            relative_gap = (
                2.0
                * abs(score - reference_score)
                / (abs(score) + abs(reference_score) + 1e-12)
            )
            scores[path_count][variant.name] = score
            rows.append(
                {
                    "outer": outer,
                    "inner": inner,
                    "stage": stage,
                    "radius": radius,
                    "model": model_name,
                    "analysis_variant": variant.name,
                    "path_count": path_count,
                    "reference_path_count": reference_path_count,
                    "aacv_score": score,
                    "reference_aacv_score": reference_score,
                    "score_symmetric_relative_gap": relative_gap,
                    "maximum_outward_gap_p99": maximum_q99,
                    "minimum_outward_gap_p99": minimum_q99,
                    "response_sd": response_sd,
                    "maximum_outward_gap_p99_response_sd": maximum_q99 / response_sd,
                    "minimum_outward_gap_p99_response_sd": minimum_q99 / response_sd,
                    "selected_model_at_path_count": None,
                    "selected_model_at_reference": None,
                    "selection_matches_reference": None,
                }
            )
    return rows, scores


def _working_error_rows(
    result: WorkingErrorResult,
    *,
    outer: int,
    inner: int,
    split_seed: int,
    train_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    reference_rows: list[dict[str, Any]] = []
    for learner, reference in result.references.items():
        reference_rows.append(
            {
                "outer": outer,
                "inner": inner,
                "split_seed": split_seed,
                "crossfit_seed": result.crossfit_seed,
                "train_size": train_size,
                "working_reference_learner": learner,
                "scale_estimator": reference.scale.name,
                "crossfit_folds": len(reference.fold_sizes),
                "fold_sizes": ",".join(str(value) for value in reference.fold_sizes),
                "fold_model_seeds": ",".join(
                    str(value) for value in reference.fold_model_seeds
                ),
                "sigma2_raw": reference.scale.sigma2_raw,
                "sigma2": reference.scale.sigma2,
                "sigma": reference.scale.sigma,
                "residual_mean": reference.residual_mean,
                "residual_mse": reference.residual_mse,
                "residual_skewness": reference.residual_skewness,
                "residual_excess_kurtosis": reference.residual_excess_kurtosis,
                "jarque_bera_statistic": reference.jarque_bera_statistic,
                "jarque_bera_pvalue": reference.jarque_bera_pvalue,
                "variance_floor_applied": int(reference.scale.floor_applied),
            }
        )

    primary_sigma2 = result.scales["oof_residual_histgb"].sigma2
    scale_rows: list[dict[str, Any]] = []
    for name, scale in result.scales.items():
        row = {
            "outer": outer,
            "inner": inner,
            "split_seed": split_seed,
            "scale_estimator": name,
            "sigma2_raw": scale.sigma2_raw,
            "sigma2": scale.sigma2,
            "sigma": scale.sigma,
            "sigma2_ratio_to_primary": scale.sigma2 / primary_sigma2,
            "variance_floor_applied": int(scale.floor_applied),
            "diagnostics_json": json.dumps(
                scale.diagnostics, sort_keys=True, separators=(",", ":")
            ),
        }
        scale_rows.append(row)

    covariates = result.covariate_diagnostics.copy()
    covariates.insert(0, "inner", inner)
    covariates.insert(0, "outer", outer)
    bins = result.covariate_bins.copy()
    bins.insert(0, "inner", inner)
    bins.insert(0, "outer", outer)
    return reference_rows, scale_rows, covariates, bins


def run_aacv_outer(
    frame: pd.DataFrame,
    config: AACVExperimentConfig,
    outer: int,
) -> dict[str, pd.DataFrame]:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    all_indices = np.arange(len(frame))
    outer_seed = deterministic_seed(config.base_seed, 0, outer)
    pool_idx, test_idx = train_test_split(
        all_indices,
        test_size=config.test_size,
        random_state=outer_seed,
        shuffle=True,
    )

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    covariate_frames: list[pd.DataFrame] = []
    bin_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    primary_variant = config.analysis_variants[0]
    inner_iter: Iterable[int] = tqdm(
        range(config.inner_repeats),
        total=config.inner_repeats,
        desc=f"AACV inner o={outer}",
        leave=False,
        disable=not config.progress,
    )
    for inner in inner_iter:
        split_seed = deterministic_seed(config.base_seed, 0, outer, inner)
        train_idx, val_idx = train_test_split(
            pool_idx,
            test_size=config.validation_fraction_of_pool,
            random_state=split_seed,
            shuffle=True,
        )
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_val = scaler.transform(x[val_idx])
        x_test = scaler.transform(x[test_idx])
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        response_sd = float(np.std(y_train, ddof=1))

        working_error = build_working_error(
            x[train_idx],
            y_train,
            seed=split_seed,
            candidate_order=config.candidate_order,
            crossfit_folds=config.reference_crossfit_folds,
            rf_trees=config.reference_rf_trees,
            hgb_max_iter=config.reference_hgb_max_iter,
            variance_floor_ratio=config.variance_floor_ratio,
            tong_wang_neighbors=config.tong_wang_neighbors,
            fan_yao_neighbors=config.fan_yao_neighbors,
            fan_yao_batch_rows=config.fan_yao_batch_rows,
        )
        refs, scales, covariates, bins = _working_error_rows(
            working_error,
            outer=outer,
            inner=inner,
            split_seed=split_seed,
            train_size=len(train_idx),
        )
        reference_rows.extend(refs)
        scale_rows.extend(scales)
        covariate_frames.append(covariates)
        bin_frames.append(bins)

        templates = make_models(
            n_train=len(train_idx),
            seed=split_seed,
            rf_trees=config.rf_trees,
            hgb_max_iter=config.hgb_max_iter,
        )
        fitted: dict[str, object] = {}
        clean_scores: dict[str, float] = {}
        clean_test_scores: dict[str, float] = {}
        for model_name in config.candidate_order:
            estimator = clone(templates[model_name])
            estimator.fit(x_train, y_train)
            fitted[model_name] = estimator
            clean_scores[model_name] = float(
                mean_squared_error(y_val, estimator.predict(x_val))
            )
            clean_test_scores[model_name] = float(
                mean_squared_error(y_test, estimator.predict(x_test))
            )
        clean_choice = choose_min(clean_scores, config.candidate_order)

        for radius_index, radius in enumerate(config.radii):
            validation_attack_seed = stable_seed(
                config.base_seed, "attack", outer, inner, radius_index, "validation"
            )
            evaluation_attack_seed = stable_seed(
                config.base_seed, "attack", outer, inner, radius_index, "evaluation"
            )
            calibrating = config.path_calibration.enabled and any(
                math.isclose(radius, configured)
                for configured in config.path_calibration.radii
            )
            search_path_count = (
                max(config.path_calibration.path_counts)
                if calibrating
                else config.retained_paths
            )
            variant_scores = {variant.name: {} for variant in config.analysis_variants}
            radius_calibration_rows: list[dict[str, Any]] = []
            calibration_validation_scores = {
                path_count: {variant.name: {} for variant in config.analysis_variants}
                for path_count in config.path_calibration.path_counts
            }
            for model_name in config.candidate_order:
                estimator = fitted[model_name]
                validation_attack = _model_attack(
                    model_name,
                    estimator,
                    x_val,
                    radius,
                    config.validation_budget,
                    validation_attack_seed,
                    retained_paths=config.retained_paths,
                    search_path_count=search_path_count,
                )
                test_attack = _model_attack(
                    model_name,
                    estimator,
                    x_test,
                    radius,
                    config.evaluation_budget,
                    evaluation_attack_seed,
                    retained_paths=config.retained_paths,
                    search_path_count=search_path_count,
                )
                for variant in config.analysis_variants:
                    sigma = working_error.scales[variant.scale_estimator].sigma
                    validation_diagnostics = _score_diagnostics(
                        y_val, validation_attack.extrema, sigma, variant.hermite
                    )
                    test_diagnostics = _score_diagnostics(
                        y_test, test_attack.extrema, sigma, variant.hermite
                    )
                    variant_scores[variant.name][model_name] = float(
                        validation_diagnostics["aacv_score"]
                    )
                    validation_rows.append(
                        {
                            **_variant_columns(variant),
                            "radius": radius,
                            "outer": outer,
                            "inner": inner,
                            "model": model_name,
                            "clean_cv": clean_scores[model_name],
                            "sigma": sigma,
                            "train_size": len(train_idx),
                            "validation_size": len(val_idx),
                            "test_size": len(test_idx),
                            **validation_diagnostics,
                        }
                    )
                    test_score = test_diagnostics.pop("aacv_score")
                    test_rows.append(
                        {
                            **_variant_columns(variant),
                            "radius": radius,
                            "outer": outer,
                            "inner": inner,
                            "model": model_name,
                            "test_aacv_score": test_score,
                            "clean_test_mse": clean_test_scores[model_name],
                            "sigma": sigma,
                            **test_diagnostics,
                        }
                    )

                if calibrating:
                    for stage, stage_y, stage_attack in (
                        ("validation", y_val, validation_attack),
                        ("evaluation", y_test, test_attack),
                    ):
                        rows, scores = _path_calibration_rows_for_stage(
                            attack=stage_attack,
                            y=stage_y,
                            working_error=working_error,
                            variants=config.analysis_variants,
                            path_counts=config.path_calibration.path_counts,
                            response_sd=response_sd,
                            outer=outer,
                            inner=inner,
                            stage=stage,
                            radius=radius,
                            model_name=model_name,
                        )
                        radius_calibration_rows.extend(rows)
                        if stage == "validation":
                            for path_count, by_variant in scores.items():
                                for variant_name, score in by_variant.items():
                                    calibration_validation_scores[path_count][
                                        variant_name
                                    ][model_name] = score

                if config.audit_enabled:
                    for stage, stage_x, stage_y, stage_attack, stage_budget, seed in (
                        (
                            "validation",
                            x_val,
                            y_val,
                            validation_attack,
                            config.validation_budget,
                            validation_attack_seed,
                        ),
                        (
                            "evaluation",
                            x_test,
                            y_test,
                            test_attack,
                            config.evaluation_budget,
                            evaluation_attack_seed,
                        ),
                    ):
                        audit = _attack_audit_row(
                            estimator=estimator,
                            model_name=model_name,
                            x_scaled=stage_x,
                            y=stage_y,
                            sigma=working_error.scales[
                                primary_variant.scale_estimator
                            ].sigma,
                            radius=radius,
                            primary=stage_attack,
                            budget=stage_budget,
                            attack_seed=seed,
                            subset_seed=stable_seed(
                                config.base_seed,
                                "audit_subset",
                                outer,
                                inner,
                                radius_index,
                                stage,
                            ),
                            maximum_observations=config.audit_max_observations,
                            stage=stage,
                            outer=outer,
                            inner=inner,
                            primary_hermite=primary_variant.hermite,
                            retained_paths=config.retained_paths,
                            sobol_multiplier=config.audit_sobol_multiplier,
                            minimum_step_fraction=config.audit_minimum_step_fraction,
                        )
                        if audit is not None:
                            audit_rows.append(audit)

            if calibrating:
                reference_path_count = max(config.path_calibration.path_counts)
                selection_lookup: dict[tuple[int, str], str] = {}
                for path_count in config.path_calibration.path_counts:
                    for variant in config.analysis_variants:
                        selection_lookup[(path_count, variant.name)] = choose_min(
                            calibration_validation_scores[path_count][variant.name],
                            config.candidate_order,
                        )
                for row in radius_calibration_rows:
                    if row["stage"] != "validation":
                        continue
                    key = (int(row["path_count"]), str(row["analysis_variant"]))
                    reference_key = (
                        reference_path_count,
                        str(row["analysis_variant"]),
                    )
                    selected = selection_lookup[key]
                    reference_selected = selection_lookup[reference_key]
                    row["selected_model_at_path_count"] = selected
                    row["selected_model_at_reference"] = reference_selected
                    row["selection_matches_reference"] = bool(
                        selected == reference_selected
                    )
                calibration_rows.extend(radius_calibration_rows)

            for variant in config.analysis_variants:
                aacv_choice = choose_min(
                    variant_scores[variant.name], config.candidate_order
                )
                selection_rows.extend(
                    [
                        {
                            **_variant_columns(variant),
                            "radius": radius,
                            "outer": outer,
                            "inner": inner,
                            "selector": "clean",
                            "selected_model": clean_choice,
                            "selected_validation_score": clean_scores[clean_choice],
                        },
                        {
                            **_variant_columns(variant),
                            "radius": radius,
                            "outer": outer,
                            "inner": inner,
                            "selector": "aacv",
                            "selected_model": aacv_choice,
                            "selected_validation_score": variant_scores[variant.name][
                                aacv_choice
                            ],
                        },
                    ]
                )

    validation = pd.DataFrame(validation_rows)
    test = pd.DataFrame(test_rows)
    selections = pd.DataFrame(selection_rows)
    outer_rows: list[dict[str, Any]] = []
    outer_candidate_rows: list[dict[str, Any]] = []
    for variant in config.analysis_variants:
        for radius in config.radii:
            selection_cell = selections[
                (selections["analysis_variant"] == variant.name)
                & np.isclose(selections["radius"], radius)
            ]
            validation_cell = validation[
                (validation["analysis_variant"] == variant.name)
                & np.isclose(validation["radius"], radius)
            ]
            test_cell = test[
                (test["analysis_variant"] == variant.name)
                & np.isclose(test["radius"], radius)
            ]
            mean_test_scores = {
                model: float(
                    test_cell[test_cell["model"] == model]["test_aacv_score"].mean()
                )
                for model in config.candidate_order
            }
            heldout_best = choose_min(mean_test_scores, config.candidate_order)
            heldout_best_score = mean_test_scores[heldout_best]
            for model in config.candidate_order:
                outer_candidate_rows.append(
                    {
                        **_variant_columns(variant),
                        "radius": radius,
                        "outer": outer,
                        "model": model,
                        "test_aacv_score": mean_test_scores[model],
                    }
                )
            for selector in ("clean", "aacv"):
                choices = (
                    selection_cell[selection_cell["selector"] == selector]
                    .sort_values("inner")["selected_model"]
                    .astype(str)
                    .tolist()
                )
                score_column = "clean_cv" if selector == "clean" else "aacv_score"
                tie_scores = {
                    model: float(
                        validation_cell[validation_cell["model"] == model][
                            score_column
                        ].mean()
                    )
                    for model in config.candidate_order
                }
                final_choice = majority_vote(
                    choices, tie_scores, config.candidate_order
                )
                selected_test_score = mean_test_scores[final_choice]
                outer_rows.append(
                    {
                        **_variant_columns(variant),
                        "radius": radius,
                        "outer": outer,
                        "selector": selector,
                        "selected_model": final_choice,
                        "heldout_best_choice": heldout_best,
                        "selected_test_aacv_score": selected_test_score,
                        "heldout_best_test_aacv_score": heldout_best_score,
                        "heldout_score_excess": selected_test_score
                        - heldout_best_score,
                        "heldout_best_match": int(final_choice == heldout_best),
                    }
                )

    return {
        "inner_aacv_scores": validation,
        "inner_test_aacv_scores": test,
        "inner_aacv_selections": selections,
        "outer_aacv_summary": pd.DataFrame(outer_rows),
        "outer_candidate_test_scores": pd.DataFrame(outer_candidate_rows),
        "working_reference_diagnostics": pd.DataFrame(reference_rows),
        "scale_estimator_diagnostics": pd.DataFrame(scale_rows),
        "residual_covariate_diagnostics": pd.concat(
            covariate_frames, ignore_index=True
        ),
        "residual_covariate_bins": pd.concat(bin_frames, ignore_index=True),
        "attack_diagnostics": pd.DataFrame(
            audit_rows, columns=ATTACK_DIAGNOSTIC_COLUMNS
        ),
        "attack_path_calibration": pd.DataFrame(
            calibration_rows, columns=PATH_CALIBRATION_COLUMNS
        ),
    }


def _sensitivity_tables(
    tables: dict[str, pd.DataFrame],
    variants: tuple[AnalysisVariant, ...],
) -> dict[str, pd.DataFrame]:
    outer = tables["outer_aacv_summary"]
    primary = outer[outer["analysis_variant"] == PRIMARY_ANALYSIS_VARIANT][
        ["radius", "outer", "selector", "selected_model"]
    ].rename(columns={"selected_model": "primary_selected_model"})
    switches = outer.merge(primary, on=["radius", "outer", "selector"], how="left")
    switches["selected_model_agreement"] = (
        switches["selected_model"] == switches["primary_selected_model"]
    ).astype(int)
    switches["selected_model_switched"] = 1 - switches["selected_model_agreement"]
    switches = switches[
        [
            "analysis_variant",
            "analysis_role",
            "working_reference_learner",
            "scale_estimator",
            "hermite_degree",
            "envelope_multiplier",
            "radius",
            "outer",
            "selector",
            "selected_model",
            "primary_selected_model",
            "selected_model_agreement",
            "selected_model_switched",
        ]
    ]
    summary = (
        switches.groupby(
            [
                "analysis_variant",
                "analysis_role",
                "working_reference_learner",
                "scale_estimator",
                "hermite_degree",
                "envelope_multiplier",
                "radius",
                "selector",
            ],
            as_index=False,
        )
        .agg(
            selected_model_agreement_rate=("selected_model_agreement", "mean"),
            selected_model_switch_rate=("selected_model_switched", "mean"),
            outer_repeats=("outer", "nunique"),
        )
        .sort_values(["analysis_variant", "radius", "selector"])
        .reset_index(drop=True)
    )
    scale = tables["scale_estimator_diagnostics"]
    scale_summary = (
        scale.groupby("scale_estimator", as_index=False)
        .agg(
            sigma_mean=("sigma", "mean"),
            sigma_sd=("sigma", lambda values: float(values.std(ddof=1))),
            sigma2_ratio_to_primary_mean=("sigma2_ratio_to_primary", "mean"),
            sigma2_ratio_to_primary_min=("sigma2_ratio_to_primary", "min"),
            sigma2_ratio_to_primary_max=("sigma2_ratio_to_primary", "max"),
            split_count=("sigma", "size"),
        )
        .sort_values("scale_estimator")
        .reset_index(drop=True)
    )
    return {
        "working_error_selection_switches": switches,
        "working_error_sensitivity_summary": summary,
        "scale_estimator_summary": scale_summary,
    }


def summarize_aacv_tables(
    tables: dict[str, pd.DataFrame],
    candidate_order: tuple[str, ...],
    variants: tuple[AnalysisVariant, ...],
) -> dict[str, pd.DataFrame]:
    outer = tables["outer_aacv_summary"]
    group_columns = [
        "analysis_variant",
        "analysis_role",
        "working_reference_learner",
        "scale_estimator",
        "hermite_degree",
        "envelope_multiplier",
        "radius",
        "selector",
    ]
    summary = (
        outer.groupby(group_columns, as_index=False)
        .agg(
            heldout_best_match_rate=("heldout_best_match", "mean"),
            heldout_score_excess_mean=("heldout_score_excess", "mean"),
            heldout_score_excess_se=("heldout_score_excess", standard_error),
            selected_test_aacv_mean=("selected_test_aacv_score", "mean"),
            heldout_best_test_aacv_mean=("heldout_best_test_aacv_score", "mean"),
            outer_repeats=("outer", "nunique"),
        )
        .sort_values(["analysis_variant", "radius", "selector"])
        .reset_index(drop=True)
    )

    frequency_rows: list[dict[str, Any]] = []
    for keys, cell in outer.groupby(group_columns, sort=True):
        metadata = dict(zip(group_columns, keys, strict=True))
        counts = cell["selected_model"].value_counts()
        total = int(len(cell))
        for model in candidate_order:
            count = int(counts.get(model, 0))
            frequency_rows.append(
                {
                    **metadata,
                    "model": model,
                    "count": count,
                    "total": total,
                    "frequency": count / total,
                }
            )

    candidate = tables["outer_candidate_test_scores"]
    candidate_group = [
        "analysis_variant",
        "analysis_role",
        "working_reference_learner",
        "scale_estimator",
        "hermite_degree",
        "envelope_multiplier",
        "radius",
        "model",
    ]
    candidate_summary = (
        candidate.groupby(candidate_group, as_index=False)
        .agg(
            test_aacv_mean=("test_aacv_score", "mean"),
            test_aacv_se=("test_aacv_score", standard_error),
            outer_repeats=("outer", "nunique"),
        )
        .sort_values(["analysis_variant", "radius", "model"])
        .reset_index(drop=True)
    )
    return {
        "aacv_summary_by_radius": summary,
        "aacv_selection_frequencies": pd.DataFrame(frequency_rows),
        "aacv_candidate_test_scores": candidate_summary,
        **_sensitivity_tables(tables, variants),
    }


def summarize_attack_path_calibration(
    calibration: pd.DataFrame,
    config: AACVExperimentConfig,
) -> pd.DataFrame:
    columns = [
        "calibration_enabled",
        "configured_path_count",
        "comparison_path_count",
        "calibration_radii",
        "selection_agreement_passed",
        "maximum_score_symmetric_relative_gap",
        "score_relative_tolerance",
        "maximum_outward_gap_p99_response_sd",
        "minimum_outward_gap_p99_response_sd",
        "extrema_p99_response_sd_tolerance",
        "calibration_passed",
        "recommended_path_count",
        "configured_path_count_matches_recommendation",
    ]
    path_config = config.path_calibration
    if not path_config.enabled:
        return pd.DataFrame(
            [
                {
                    "calibration_enabled": False,
                    "configured_path_count": config.retained_paths,
                    "comparison_path_count": None,
                    "calibration_radii": "",
                    "selection_agreement_passed": None,
                    "maximum_score_symmetric_relative_gap": None,
                    "score_relative_tolerance": path_config.score_relative_tolerance,
                    "maximum_outward_gap_p99_response_sd": None,
                    "minimum_outward_gap_p99_response_sd": None,
                    "extrema_p99_response_sd_tolerance": (
                        path_config.extrema_p99_response_sd_tolerance
                    ),
                    "calibration_passed": None,
                    "recommended_path_count": config.retained_paths,
                    "configured_path_count_matches_recommendation": True,
                }
            ],
            columns=columns,
        )
    if calibration.empty:
        raise ValueError("Enabled path calibration produced no rows")
    comparison_path_count = 4
    candidate = calibration[calibration["path_count"] == 3].copy()
    if candidate.empty:
        raise ValueError("Path calibration is missing K=3 rows")
    selection_values = candidate[candidate["stage"] == "validation"][
        "selection_matches_reference"
    ].dropna()
    selection_passed = bool(
        len(selection_values) > 0 and selection_values.astype(bool).all()
    )
    maximum_score_gap = float(candidate["score_symmetric_relative_gap"].max())
    maximum_extrema_gap = float(candidate["maximum_outward_gap_p99_response_sd"].max())
    minimum_extrema_gap = float(candidate["minimum_outward_gap_p99_response_sd"].max())
    calibration_passed = bool(
        selection_passed
        and maximum_score_gap <= path_config.score_relative_tolerance
        and maximum_extrema_gap <= path_config.extrema_p99_response_sd_tolerance
        and minimum_extrema_gap <= path_config.extrema_p99_response_sd_tolerance
    )
    recommended_path_count = 3 if calibration_passed else 4
    return pd.DataFrame(
        [
            {
                "calibration_enabled": True,
                "configured_path_count": config.retained_paths,
                "comparison_path_count": comparison_path_count,
                "calibration_radii": ",".join(
                    f"{radius:g}" for radius in path_config.radii
                ),
                "selection_agreement_passed": selection_passed,
                "maximum_score_symmetric_relative_gap": maximum_score_gap,
                "score_relative_tolerance": path_config.score_relative_tolerance,
                "maximum_outward_gap_p99_response_sd": maximum_extrema_gap,
                "minimum_outward_gap_p99_response_sd": minimum_extrema_gap,
                "extrema_p99_response_sd_tolerance": (
                    path_config.extrema_p99_response_sd_tolerance
                ),
                "calibration_passed": calibration_passed,
                "recommended_path_count": recommended_path_count,
                "configured_path_count_matches_recommendation": bool(
                    config.retained_paths == recommended_path_count
                ),
            }
        ],
        columns=columns,
    )


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _shard_dir(output_dir: Path, outer: int) -> Path:
    return output_dir / "partials" / f"outer_{outer:03d}"


def _validate_shard(path: Path, fingerprint: str, outer: int) -> None:
    manifest_path = path / "shard_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing shard manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("scientific_schema_version") != AACV_SCIENTIFIC_SCHEMA_VERSION:
        raise ValueError(f"Scientific schema mismatch in {path}")
    if manifest.get("attack_algorithm") != ATTACK_ALGORITHM:
        raise ValueError(f"Attack algorithm mismatch in {path}")
    if manifest.get("config_fingerprint") != fingerprint:
        raise ValueError(f"Configuration fingerprint mismatch in {path}")
    if int(manifest.get("outer", -1)) != int(outer):
        raise ValueError(f"Outer id mismatch in {path}")
    schemas = manifest.get("schemas", {})
    for table_name in SHARD_TABLES:
        table_path = path / f"{table_name}.csv"
        if not table_path.exists():
            raise ValueError(f"Missing shard table: {table_path}")
        frame = pd.read_csv(table_path)
        schema = schemas.get(table_name, {})
        if list(frame.columns) != schema.get("columns"):
            raise ValueError(f"Schema mismatch for {table_path}")
        if len(frame) != int(schema.get("rows", -1)):
            raise ValueError(f"Row-count mismatch for {table_path}")


def _write_outer_shard(
    output_dir: Path,
    outer: int,
    fingerprint: str,
    tables: dict[str, pd.DataFrame],
) -> None:
    partials = output_dir / "partials"
    partials.mkdir(parents=True, exist_ok=True)
    destination = _shard_dir(output_dir, outer)
    temporary = partials / f".outer_{outer:03d}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=False, exist_ok=False)
    schemas: dict[str, dict[str, Any]] = {}
    try:
        for table_name in SHARD_TABLES:
            frame = tables[table_name]
            frame.to_csv(temporary / f"{table_name}.csv", index=False)
            schemas[table_name] = {
                "columns": list(frame.columns),
                "rows": int(len(frame)),
            }
        with (temporary / "shard_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "outer": outer,
                    "scientific_schema_version": AACV_SCIENTIFIC_SCHEMA_VERSION,
                    "attack_algorithm": ATTACK_ALGORITHM,
                    "config_fingerprint": fingerprint,
                    "schemas": schemas,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing shard {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()


def _collect_shards(
    output_dir: Path,
    outer_repeats: int,
    fingerprint: str,
) -> dict[str, pd.DataFrame]:
    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in SHARD_TABLES}
    for outer in range(outer_repeats):
        path = _shard_dir(output_dir, outer)
        _validate_shard(path, fingerprint, outer)
        for table_name in SHARD_TABLES:
            collected[table_name].append(pd.read_csv(path / f"{table_name}.csv"))
    return {
        name: pd.concat(frames, ignore_index=True) for name, frames in collected.items()
    }


def _git_state() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    entries = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(entries),
        "status_entries": entries,
    }


def _source_hashes() -> dict[str, str]:
    paths = [
        PROJECT_DIR / "AACV_WORKING_ERROR_GUIDELINE.md",
        PROJECT_DIR / "AACV_IMPLEMENTATION_PLAN.md",
        PROJECT_DIR / "AACV_MULTIPATH_MEDIUM_PLAN.md",
        *sorted((PROJECT_DIR / "ccpp_repro").glob("aacv_*.py")),
        PROJECT_DIR / "requirements.txt",
        PROJECT_DIR / "ccpp_repro" / "__main__.py",
        PROJECT_DIR / "ccpp_repro" / "cli.py",
        PROJECT_DIR / "ccpp_repro" / "config.py",
        PROJECT_DIR / "ccpp_repro" / "experiment.py",
        PROJECT_DIR / "ccpp_repro" / "mismatch.py",
        PROJECT_DIR / "ccpp_repro" / "plotting.py",
    ]
    return {
        str(path.relative_to(PROJECT_DIR)).replace("\\", "/"): sha256_file(path)
        for path in paths
        if path.exists()
    }


def _package_versions() -> dict[str, str]:
    names = (
        "matplotlib",
        "numpy",
        "openpyxl",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _run_manifest(
    raw_config: dict[str, Any],
    experiment_config: AACVExperimentConfig,
    fingerprint: str,
    data_path: Path,
    reference_path: Path,
    reference_spec: dict[str, Any],
    calibration_summary: pd.DataFrame,
) -> dict[str, Any]:
    resolved_config = {
        key: value for key, value in raw_config.items() if key != "_config_path"
    }
    calibration_record = json.loads(calibration_summary.to_json(orient="records"))[0]
    return {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_schema_version": AACV_SCIENTIFIC_SCHEMA_VERSION,
        "attack_algorithm": ATTACK_ALGORITHM,
        "attack_path_calibration": calibration_record,
        "experiment_fingerprint": fingerprint,
        "resolved_config": resolved_config,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "reference_spec_path": str(reference_path),
        "reference_spec_fingerprint": canonical_json_fingerprint(reference_spec),
        "reference_selection": reference_spec,
        "data_reuse_disclosure": reference_spec["data_reuse_disclosure"],
        "git": _git_state(),
        "source_sha256": _source_hashes(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "experiment_config": asdict(experiment_config),
        "split_seeds": {
            "outer": [
                {
                    "outer": outer,
                    "seed": deterministic_seed(experiment_config.base_seed, 0, outer),
                }
                for outer in range(experiment_config.outer_repeats)
            ],
            "inner": [
                {
                    "outer": outer,
                    "inner": inner,
                    "seed": deterministic_seed(
                        experiment_config.base_seed, 0, outer, inner
                    ),
                }
                for outer in range(experiment_config.outer_repeats)
                for inner in range(experiment_config.inner_repeats)
            ],
        },
        "seed_derivation": {
            "outer_split": "deterministic_seed(base_seed, 0, outer)",
            "inner_split": "deterministic_seed(base_seed, 0, outer, inner)",
            "working_reference_crossfit": (
                "inner split seed; fold models use stable_seed with learner and fold"
            ),
            "attack": (
                "stable_seed(base_seed, attack, outer, inner, radius_index, stage)"
            ),
        },
        "analysis_variants": [
            asdict(variant) for variant in experiment_config.analysis_variants
        ],
        "scientific_status": {
            "known_variance_theorem_applies": False,
            "working_reference_is_true_regression_function": False,
            "working_error_is_physical_noise_law": False,
            "working_scales_use_inner_training_only": True,
            "nonlinear_extrema_are_numerical": True,
            "primary_attack_uses_multiple_independent_paths": True,
            "stronger_attack_inherits_primary_envelope": True,
            "stronger_extrema_cannot_degrade_primary": True,
            "heldout_best_uses_unknown_truth": False,
        },
    }


def run_aacv_and_write(
    config: dict[str, Any],
    output_dir: Path,
    *,
    progress: bool = True,
    resume: bool = False,
) -> dict[str, pd.DataFrame]:
    reference_path, reference_spec = load_reference_spec(config)
    experiment_config = aacv_config_from_dict(config, progress=progress)
    fingerprint = experiment_fingerprint(config, reference_spec)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"AACV output directory is nonempty; use --resume: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = require_data_file(config)
    frame = load_ccpp_data(config)

    outer_iter: Iterable[int] = tqdm(
        range(experiment_config.outer_repeats),
        total=experiment_config.outer_repeats,
        desc="AACV outer",
        disable=not progress,
    )
    for outer in outer_iter:
        shard = _shard_dir(output_dir, outer)
        if shard.exists():
            if not resume:
                raise FileExistsError(f"AACV shard already exists: {shard}")
            _validate_shard(shard, fingerprint, outer)
            continue
        tables = run_aacv_outer(frame, experiment_config, outer)
        _write_outer_shard(output_dir, outer, fingerprint, tables)

    tables = _collect_shards(output_dir, experiment_config.outer_repeats, fingerprint)
    for name, table in tables.items():
        _write_csv_atomic(output_dir / f"{name}.csv", table)
    summaries = summarize_aacv_tables(
        tables,
        experiment_config.candidate_order,
        experiment_config.analysis_variants,
    )
    calibration_summary = summarize_attack_path_calibration(
        tables["attack_path_calibration"], experiment_config
    )
    summaries["attack_path_calibration_summary"] = calibration_summary
    for name, table in summaries.items():
        _write_csv_atomic(output_dir / f"{name}.csv", table)
    reference_table = pd.DataFrame(reference_spec["candidate_summaries"])
    _write_csv_atomic(output_dir / "reference_selection.csv", reference_table)
    _write_json_atomic(output_dir / "reference_selection.json", reference_spec)
    _write_json_atomic(
        output_dir / "run_manifest.json",
        _run_manifest(
            config,
            experiment_config,
            fingerprint,
            data_path,
            reference_path,
            reference_spec,
            calibration_summary,
        ),
    )
    return {**tables, **summaries}
