from __future__ import annotations

import importlib.metadata
import json
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
    aacv_observation_scores,
    attack_set_diagnostic_rows,
    corner_attack_extrema,
    validate_attack_radii,
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


AACV_SCIENTIFIC_SCHEMA_VERSION = 3
ATTACK_ALGORITHM = "origin_plus_cube_corners_v1"
ATTACK_SET_DIMENSION = len(FEATURE_COLUMNS)
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
)
ATTACK_SET_DIAGNOSTIC_COLUMNS = (
    "radius_index",
    "radius",
    "dimension",
    "corner_count",
    "expected_unique_point_count",
    "actual_unique_point_count",
    "origin_count",
    "origin_present",
    "corners_on_current_radius",
    "within_current_radius",
    "construction_passed",
    "attack_algorithm",
)


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
    prediction_chunk_rows: int
    progress: bool = True


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
    if "optimizer" in config:
        raise ValueError(
            "Legacy AACV optimizer configuration is not supported by schema v3"
        )
    if "attack_set" not in config:
        raise ValueError("AACV configuration must define attack_set")
    attack_set = config["attack_set"]
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

    radii = validate_attack_radii(exp["radii"])
    allowed_attack_keys = {"algorithm", "prediction_chunk_rows"}
    unexpected_attack_keys = set(attack_set) - allowed_attack_keys
    if unexpected_attack_keys:
        raise ValueError(
            "Unexpected attack_set fields: " f"{sorted(unexpected_attack_keys)}"
        )
    attack_algorithm = str(attack_set["algorithm"])
    if attack_algorithm != ATTACK_ALGORITHM:
        raise ValueError(f"AACV attack algorithm must be {ATTACK_ALGORITHM}")
    prediction_chunk_rows = int(attack_set.get("prediction_chunk_rows", 2048))
    if prediction_chunk_rows <= 0:
        raise ValueError("attack_set prediction_chunk_rows must be positive")
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
        prediction_chunk_rows=prediction_chunk_rows,
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

        validation_attacks: dict[str, tuple[ExtremaResult, ...]] = {}
        test_attacks: dict[str, tuple[ExtremaResult, ...]] = {}
        for model_name in config.candidate_order:
            estimator = fitted[model_name]
            validation_attacks[model_name] = corner_attack_extrema(
                estimator,
                x_val,
                config.radii,
                prediction_chunk_rows=config.prediction_chunk_rows,
            )
            test_attacks[model_name] = corner_attack_extrema(
                estimator,
                x_test,
                config.radii,
                prediction_chunk_rows=config.prediction_chunk_rows,
            )

        for radius_index, radius in enumerate(config.radii):
            variant_scores = {variant.name: {} for variant in config.analysis_variants}
            for model_name in config.candidate_order:
                validation_extrema = validation_attacks[model_name][radius_index]
                test_extrema = test_attacks[model_name][radius_index]
                for variant in config.analysis_variants:
                    sigma = working_error.scales[variant.scale_estimator].sigma
                    validation_diagnostics = _score_diagnostics(
                        y_val, validation_extrema, sigma, variant.hermite
                    )
                    test_diagnostics = _score_diagnostics(
                        y_test, test_extrema, sigma, variant.hermite
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


def _attack_set_manifest(
    experiment_config: AACVExperimentConfig,
) -> dict[str, Any]:
    definition = {
        "algorithm": ATTACK_ALGORITHM,
        "coordinate_system": "inner-split standardized feature coordinates",
        "dimension": ATTACK_SET_DIMENSION,
        "radii": list(experiment_config.radii),
        "point_counts": [
            1 if radius_index == 0 else 1 + 2**ATTACK_SET_DIMENSION
            for radius_index in range(len(experiment_config.radii))
        ],
        "ordering": "origin, then lexicographic current-radius (-1,+1)^d corners",
        "exact_finite_enumeration": True,
        "grid_dependent": False,
        "validation_evaluation_attack_sets_identical": True,
    }
    return {
        **definition,
        "definition_fingerprint": canonical_json_fingerprint(definition),
    }


def _run_manifest(
    raw_config: dict[str, Any],
    experiment_config: AACVExperimentConfig,
    fingerprint: str,
    data_path: Path,
    reference_path: Path,
    reference_spec: dict[str, Any],
) -> dict[str, Any]:
    resolved_config = {
        key: value for key, value in raw_config.items() if key != "_config_path"
    }
    return {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_schema_version": AACV_SCIENTIFIC_SCHEMA_VERSION,
        "attack_algorithm": ATTACK_ALGORITHM,
        "attack_set": _attack_set_manifest(experiment_config),
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
        },
        "analysis_variants": [
            asdict(variant) for variant in experiment_config.analysis_variants
        ],
        "scientific_status": {
            "known_variance_theorem_applies": False,
            "working_reference_is_true_regression_function": False,
            "working_error_is_physical_noise_law": False,
            "working_scales_use_inner_training_only": True,
            "extrema_are_exact_over_configured_finite_attack_set": True,
            "attack_sets_are_radius_specific_and_grid_independent": True,
            "continuous_cube_extrema_are_computed": False,
            "validation_and_evaluation_share_attack_set": True,
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
    attack_set_diagnostics = pd.DataFrame(
        attack_set_diagnostic_rows(
            experiment_config.radii,
            n_features=ATTACK_SET_DIMENSION,
            attack_algorithm=ATTACK_ALGORITHM,
        ),
        columns=ATTACK_SET_DIAGNOSTIC_COLUMNS,
    )
    _write_csv_atomic(output_dir / "attack_set_diagnostics.csv", attack_set_diagnostics)
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
        ),
    )
    return {**tables, **summaries, "attack_set_diagnostics": attack_set_diagnostics}
