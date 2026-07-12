from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ccpp_repro.aacv_core import (
    HermiteVariant,
    OptimizationBudget,
    _select_paths,
    additional_halving_steps,
    additional_normalized_sobol_starts,
    aacv_observation_scores,
    chebyshev_even_coefficients,
    continuous_box_extrema,
    continuous_box_path_search,
    gaussian_hermite_psi,
    linear_box_extrema,
    normalized_attack_starts,
    strict_stronger_box_extrema,
)


PRIMARY_HERMITE = HermiteVariant("primary_histgb_oof_j2_b6", 2, 6.0)


class QuadraticRegressor:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.square(x[:, 0]) - np.square(x[:, 1]) + 0.3 * x[:, 2]


class PiecewiseRegressor:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.where(x[:, 0] >= 0.25, 2.0, -1.0) + np.where(
            x[:, 1] <= -0.4, 1.0, 0.0
        )


class MultiBasinRegressor:
    def predict(self, x: np.ndarray) -> np.ndarray:
        local_distance = np.linalg.norm(x + 0.8, axis=1)
        global_distance = np.linalg.norm(x - 0.3, axis=1)
        local = 2.5 * np.maximum(0.0, 1.0 - local_distance / 0.5)
        global_peak = 3.0 * np.maximum(0.0, 1.0 - global_distance / 0.7)
        return np.maximum(local, global_peak)


def test_chebyshev_degree_two_coefficients() -> None:
    expected = np.array(
        [2.0 / (5.0 * math.pi), 24.0 / (5.0 * math.pi), -32.0 / (15.0 * math.pi)]
    )
    np.testing.assert_allclose(chebyshev_even_coefficients(2), expected)


def test_gaussian_hermite_scale_equivariance_and_radius_zero() -> None:
    y = np.array([-1.0, 0.5, 2.0])
    center = np.array([-0.5, 0.25, 1.0])
    sigma = 1.3
    psi, _ = gaussian_hermite_psi(y, center, sigma, PRIMARY_HERMITE)
    scaled_psi, _ = gaussian_hermite_psi(
        7.0 * y, 7.0 * center, 7.0 * sigma, PRIMARY_HERMITE
    )
    np.testing.assert_allclose(scaled_psi, 7.0 * psi)

    scores, _psi, _envelope = aacv_observation_scores(
        y, center, np.zeros_like(y), sigma, PRIMARY_HERMITE
    )
    np.testing.assert_allclose(scores, np.square(y - center))


def test_gaussian_hermite_monte_carlo_mean_matches_polynomial() -> None:
    rng = np.random.default_rng(20260710)
    sigma = 1.0
    theta = 1.5
    y = theta + rng.normal(size=150_000)
    center = np.zeros_like(y)
    psi, envelope = gaussian_hermite_psi(y, center, sigma, PRIMARY_HERMITE)
    coefficients = chebyshev_even_coefficients(PRIMARY_HERMITE.degree)
    target = sum(
        coefficient * envelope ** (1 - 2 * m) * theta ** (2 * m)
        for m, coefficient in enumerate(coefficients)
    )
    assert float(np.mean(psi)) == pytest.approx(target, abs=0.015)


def test_sobol_start_sets_are_nested_for_shared_seed() -> None:
    weak = normalized_attack_starts(4, 8, seed=123)
    strong = normalized_attack_starts(4, 16, seed=123)
    np.testing.assert_allclose(weak, strong[: len(weak)])


def test_linear_and_continuous_box_extrema_are_deterministic() -> None:
    rng = np.random.default_rng(31)
    x_train = rng.normal(size=(100, 4))
    y_train = x_train @ np.array([1.0, -2.0, 0.5, 3.0])
    model = LinearRegression().fit(x_train, y_train)
    x = rng.normal(size=(12, 4))
    radius = 0.3
    exact = linear_box_extrema(model, x, radius)
    budget = OptimizationBudget(8, (0.5, 0.25), prediction_chunk_rows=7)
    searched = continuous_box_extrema(model, x, radius, budget, seed=55)
    np.testing.assert_allclose(searched.maximum, exact.maximum, atol=1e-12)
    np.testing.assert_allclose(searched.minimum, exact.minimum, atol=1e-12)

    different_chunk = continuous_box_extrema(
        model,
        x,
        radius,
        OptimizationBudget(8, (0.5, 0.25), prediction_chunk_rows=1000),
        seed=55,
    )
    np.testing.assert_allclose(searched.maximum, different_chunk.maximum)
    np.testing.assert_allclose(searched.minimum, different_chunk.minimum)


def test_stronger_nested_budget_cannot_worsen_extrema() -> None:
    rng = np.random.default_rng(91)
    x_train = rng.normal(size=(80, 4))
    y_train = x_train[:, 0] - x_train[:, 1]
    model = LinearRegression().fit(x_train, y_train)
    x = rng.normal(size=(8, 4))
    weak = continuous_box_extrema(model, x, 0.2, OptimizationBudget(2, (0.5,)), seed=8)
    strong = continuous_box_extrema(
        model, x, 0.2, OptimizationBudget(4, (0.5, 0.25)), seed=8
    )
    assert np.all(strong.maximum >= weak.maximum - 1e-12)
    assert np.all(strong.minimum <= weak.minimum + 1e-12)


@pytest.mark.parametrize(
    ("model", "expected_minimum", "expected_maximum"),
    [
        (QuadraticRegressor(), -1.3, 1.3),
        (PiecewiseRegressor(), -1.0, 3.0),
    ],
)
def test_continuous_optimizer_known_nonlinear_extrema(
    model: object,
    expected_minimum: float,
    expected_maximum: float,
) -> None:
    result = continuous_box_extrema(
        model,
        np.zeros((3, 4)),
        1.0,
        OptimizationBudget(8, (1.0, 0.5, 0.25), prediction_chunk_rows=5),
        seed=44,
    )
    np.testing.assert_allclose(result.minimum, expected_minimum)
    np.testing.assert_allclose(result.maximum, expected_maximum)


def test_stable_top_k_ties_follow_start_order() -> None:
    starts = np.arange(12, dtype=float).reshape(4, 3)
    values = np.array([[2.0, 2.0, 1.0, 2.0]])
    maximum_points, maximum_values = _select_paths(starts, values, 3, maximize=True)
    minimum_points, minimum_values = _select_paths(starts, values, 2, maximize=False)
    np.testing.assert_array_equal(maximum_points[0], starts[[0, 1, 3]])
    np.testing.assert_array_equal(maximum_values[0], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(minimum_points[0], starts[[2, 0]])
    np.testing.assert_array_equal(minimum_values[0], [1.0, 2.0])


def test_additional_sobol_points_are_the_nested_tail() -> None:
    primary = normalized_attack_starts(4, 8, seed=121)
    stronger = normalized_attack_starts(4, 16, seed=121)
    additional = additional_normalized_sobol_starts(4, 8, seed=121)
    fixed_start_count = 1 + 16 + 8
    np.testing.assert_allclose(primary, stronger[: len(primary)])
    np.testing.assert_allclose(additional, stronger[fixed_start_count + 8 :])
    assert additional_halving_steps((0.5, 0.25), 1.0 / 16.0) == (
        0.125,
        0.0625,
    )


def test_multipath_history_and_envelopes_are_monotone() -> None:
    model = MultiBasinRegressor()
    x = np.zeros((3, 4))
    initial = continuous_box_path_search(
        model,
        x,
        1.0,
        OptimizationBudget(4, (), prediction_chunk_rows=7),
        seed=1,
        path_count=3,
    )
    refined = continuous_box_path_search(
        model,
        x,
        1.0,
        OptimizationBudget(4, (0.5, 0.25, 0.125, 0.0625), prediction_chunk_rows=7),
        seed=1,
        path_count=3,
    )
    assert np.all(refined.maximum_values >= initial.maximum_values)
    assert np.all(refined.minimum_values <= initial.minimum_values)
    for path_count in (1, 2):
        smaller = refined.extrema(path_count)
        larger = refined.extrema(path_count + 1)
        assert np.all(larger.maximum >= smaller.maximum)
        assert np.all(larger.minimum <= smaller.minimum)
    assert np.all(refined.extrema(3).maximum > refined.extrema(1).maximum)


def test_k_one_wrapper_matches_explicit_single_path_search() -> None:
    model = MultiBasinRegressor()
    x = np.zeros((2, 4))
    budget = OptimizationBudget(4, (0.5, 0.25), prediction_chunk_rows=5)
    wrapper = continuous_box_extrema(model, x, 1.0, budget, seed=9)
    explicit = continuous_box_path_search(
        model, x, 1.0, budget, seed=9, path_count=1
    ).extrema()
    np.testing.assert_allclose(wrapper.maximum, explicit.maximum, rtol=0, atol=0)
    np.testing.assert_allclose(wrapper.minimum, explicit.minimum, rtol=0, atol=0)


def test_strict_stronger_search_inherits_primary_envelope() -> None:
    model = MultiBasinRegressor()
    x = np.zeros((5, 4))
    budget = OptimizationBudget(4, (0.5, 0.25), prediction_chunk_rows=9)
    primary_paths = continuous_box_path_search(
        model, x, 1.0, budget, seed=14, path_count=3
    )
    primary = primary_paths.extrema()
    stronger = strict_stronger_box_extrema(
        model,
        x,
        1.0,
        budget,
        primary_paths,
        seed=14,
        sobol_multiplier=2,
        minimum_step_fraction=0.0625,
    )
    assert np.all(stronger.maximum >= primary.maximum)
    assert np.all(stronger.minimum <= primary.minimum)
