from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

import ccpp_repro.aacv_working_error as working_error
from ccpp_repro.aacv_experiment import aacv_config_from_dict
from ccpp_repro.aacv_working_error import (
    PRIMARY_ANALYSIS_VARIANT,
    PRIMARY_REFERENCE,
    PRIMARY_SCALE,
    crossfit_reference_library,
    estimate_fan_yao_scale,
    estimate_tong_wang_scale,
    required_analysis_variants,
    residual_covariate_tables,
)
from ccpp_repro.config import load_config
from ccpp_repro.experiment import FEATURE_COLUMNS


CANDIDATES = ("OLS", "RidgeCV", "LassoCV", "KNN", "RandomForest", "HistGB")


class TrackingMeanRegressor(BaseEstimator, RegressorMixin):
    fit_records: list[np.ndarray] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TrackingMeanRegressor":
        type(self).fit_records.append(np.asarray(y, dtype=float).copy())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.mean_, dtype=float)


def test_required_variants_change_one_nuisance_component_at_a_time() -> None:
    variants = required_analysis_variants(CANDIDATES)
    assert len(variants) == 12
    assert len({variant.name for variant in variants}) == len(variants)
    primary = variants[0]
    assert primary.name == PRIMARY_ANALYSIS_VARIANT
    assert primary.working_reference_learner == PRIMARY_REFERENCE
    assert primary.scale_estimator == PRIMARY_SCALE
    assert (primary.hermite_degree, primary.envelope_multiplier) == (2, 6.0)

    reference_variants = [v for v in variants if v.role == "reference_sensitivity"]
    assert {
        PRIMARY_REFERENCE,
        *(v.working_reference_learner for v in reference_variants),
    } == set(CANDIDATES)
    assert all(
        v.scale_estimator.startswith("oof_residual_") for v in reference_variants
    )
    assert all(
        (v.hermite_degree, v.envelope_multiplier) == (2, 6.0)
        for v in reference_variants
    )

    scale_variants = [v for v in variants if v.role == "scale_sensitivity"]
    assert {v.scale_estimator for v in scale_variants} == {
        "tong_wang",
        "fan_yao_histgb",
    }
    assert all(v.working_reference_learner == PRIMARY_REFERENCE for v in scale_variants)
    assert all(
        (v.hermite_degree, v.envelope_multiplier) == (2, 6.0) for v in scale_variants
    )

    hermite_variants = [v for v in variants if v.role == "hermite_sensitivity"]
    assert {(v.hermite_degree, v.envelope_multiplier) for v in hermite_variants} == {
        (1, 6.0),
        (3, 6.0),
        (2, 4.0),
        (2, 8.0),
    }
    assert all(
        v.working_reference_learner == PRIMARY_REFERENCE for v in hermite_variants
    )
    assert all(v.scale_estimator == PRIMARY_SCALE for v in hermite_variants)


def test_configs_declare_exact_required_variants() -> None:
    expected = required_analysis_variants(CANDIDATES)
    for filename in ("configs/ccpp_aacv_smoke.json", "configs/ccpp_aacv_full.json"):
        resolved = aacv_config_from_dict(load_config(filename), progress=False)
        assert resolved.analysis_variants == expected


def test_crossfit_is_honest_deterministic_and_fold_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 4))
    y = np.arange(30, dtype=float)
    monkeypatch.setattr(
        working_error,
        "make_models",
        lambda *_args, **_kwargs: {"OLS": TrackingMeanRegressor()},
    )
    TrackingMeanRegressor.fit_records.clear()
    first = crossfit_reference_library(
        x,
        y,
        seed=123,
        candidate_order=("OLS",),
        n_splits=5,
        rf_trees=1,
        hgb_max_iter=1,
        variance_floor_ratio=1e-8,
    )["OLS"]
    folds = list(KFold(5, shuffle=True, random_state=123).split(x))
    assert len(TrackingMeanRegressor.fit_records) == 5
    for fitted_y, (_fit_idx, holdout_idx) in zip(
        TrackingMeanRegressor.fit_records, folds, strict=True
    ):
        assert set(fitted_y).isdisjoint(set(y[holdout_idx]))

    TrackingMeanRegressor.fit_records.clear()
    second = crossfit_reference_library(
        x,
        y,
        seed=123,
        candidate_order=("OLS",),
        n_splits=5,
        rf_trees=1,
        hgb_max_iter=1,
        variance_floor_ratio=1e-8,
    )["OLS"]
    np.testing.assert_allclose(first.predictions, second.predictions)
    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    assert first.fold_model_seeds == second.fold_model_seeds
    assert first.scale.sigma2 == pytest.approx(second.scale.sigma2)


def test_six_reference_learners_share_folds_and_return_finite_scales() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(90, 4))
    y = 2.0 + x @ np.array([0.7, -0.2, 0.1, 0.5]) + rng.normal(size=len(x))
    references = crossfit_reference_library(
        x,
        y,
        seed=22,
        candidate_order=CANDIDATES,
        n_splits=5,
        rf_trees=3,
        hgb_max_iter=5,
        variance_floor_ratio=1e-8,
    )
    assert set(references) == set(CANDIDATES)
    primary_folds = references[PRIMARY_REFERENCE].fold_ids
    for result in references.values():
        np.testing.assert_array_equal(result.fold_ids, primary_folds)
        assert np.isfinite(result.predictions).all()
        assert np.isfinite(result.residuals).all()
        assert result.scale.sigma2 > 0
        assert len(result.fold_model_seeds) == 5

    diagnostics, bins = residual_covariate_tables(x, references)
    assert len(diagnostics) == len(CANDIDATES) * len(FEATURE_COLUMNS)
    assert set(diagnostics["covariate"]) == set(FEATURE_COLUMNS)
    assert len(bins) == len(CANDIDATES) * len(FEATURE_COLUMNS) * 10


def test_structural_scale_estimators_are_deterministic_and_positive() -> None:
    rng = np.random.default_rng(91)
    x = rng.normal(size=(160, 4))
    residual = rng.normal(scale=1.4, size=len(x))
    y = x[:, 0] + 0.2 * np.square(x[:, 1]) + residual
    tong_first = estimate_tong_wang_scale(x, y, neighbors=12, variance_floor_ratio=1e-8)
    tong_second = estimate_tong_wang_scale(
        x, y, neighbors=12, variance_floor_ratio=1e-8
    )
    fan_first = estimate_fan_yao_scale(
        x,
        y,
        residual,
        neighbors=30,
        batch_rows=23,
        variance_floor_ratio=1e-8,
    )
    fan_second = estimate_fan_yao_scale(
        x,
        y,
        residual,
        neighbors=30,
        batch_rows=80,
        variance_floor_ratio=1e-8,
    )
    assert tong_first.sigma2 == pytest.approx(tong_second.sigma2)
    assert fan_first.sigma2 == pytest.approx(fan_second.sigma2)
    assert tong_first.sigma2 > 0
    assert fan_first.sigma2 > 0
    assert tong_first.diagnostics["pair_count"] > len(x)
    assert "negative_local_rate" in fan_first.diagnostics
