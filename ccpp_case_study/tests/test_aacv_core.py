from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ccpp_repro.aacv_core import (
    HermiteVariant,
    aacv_observation_scores,
    attack_set_diagnostic_rows,
    chebyshev_even_coefficients,
    corner_attack_extrema,
    corner_attack_points,
    cube_corner_signs,
    gaussian_hermite_psi,
    validate_attack_radii,
)


PRIMARY_HERMITE = HermiteVariant("primary_histgb_oof_j2_b6", 2, 6.0)


class QuadraticRegressor:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.square(x[:, 0]) - np.square(x[:, 1]) + 0.3 * x[:, 2] - 0.2 * x[:, 3]


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


def test_four_dimensional_corner_attack_set_has_origin_and_16_corners() -> None:
    signs = cube_corner_signs(4)
    assert signs.shape == (16, 4)
    assert len(np.unique(signs, axis=0)) == 16
    assert tuple(signs[0]) == (-1.0, -1.0, -1.0, -1.0)
    assert tuple(signs[-1]) == (1.0, 1.0, 1.0, 1.0)

    points = corner_attack_points(0.3, 4)
    assert points.shape == (17, 4)
    np.testing.assert_array_equal(points[0], np.zeros(4))
    np.testing.assert_allclose(np.abs(points[1:]), 0.3, rtol=0, atol=0)


def test_corner_extrema_equal_direct_finite_enumeration() -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=(9, 4))
    radii = (0.0, 0.2, 0.4)
    model = QuadraticRegressor()
    results = corner_attack_extrema(model, x, radii, prediction_chunk_rows=19)

    for radius_index, radius in enumerate(radii):
        offsets = corner_attack_points(radius, 4)
        predictions = model.predict(
            (x[:, None, :] + offsets[None, :, :]).reshape(-1, 4)
        ).reshape(len(x), len(offsets))
        np.testing.assert_allclose(
            results[radius_index].maximum, np.max(predictions, axis=1)
        )
        np.testing.assert_allclose(
            results[radius_index].minimum, np.min(predictions, axis=1)
        )


def test_corner_extrema_are_chunk_invariant_and_grid_independent() -> None:
    rng = np.random.default_rng(37)
    x = rng.normal(size=(11, 4))
    model = QuadraticRegressor()
    dense = corner_attack_extrema(model, x, (0.0, 0.2, 0.4), prediction_chunk_rows=17)
    sparse = corner_attack_extrema(model, x, (0.0, 0.4), prediction_chunk_rows=4096)
    np.testing.assert_allclose(dense[-1].maximum, sparse[-1].maximum, rtol=0, atol=0)
    np.testing.assert_allclose(dense[-1].minimum, sparse[-1].minimum, rtol=0, atol=0)


def test_radius_zero_extrema_equal_clean_predictions() -> None:
    x = np.arange(20, dtype=float).reshape(5, 4) / 10.0
    model = QuadraticRegressor()
    result = corner_attack_extrema(model, x, (0.0,), prediction_chunk_rows=3)[0]
    expected = model.predict(x)
    np.testing.assert_allclose(result.maximum, expected, rtol=0, atol=0)
    np.testing.assert_allclose(result.minimum, expected, rtol=0, atol=0)


def test_linear_corner_extrema_match_corner_analytic_formula() -> None:
    rng = np.random.default_rng(41)
    x_train = rng.normal(size=(80, 4))
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    model = LinearRegression().fit(x_train, x_train @ beta)
    x = rng.normal(size=(13, 4))
    radius = 0.3
    result = corner_attack_extrema(model, x, (0.0, radius), prediction_chunk_rows=23)[1]
    prediction = model.predict(x)
    half_range = radius * np.sum(np.abs(model.coef_))
    np.testing.assert_allclose(result.maximum, prediction + half_range)
    np.testing.assert_allclose(result.minimum, prediction - half_range)


def test_attack_set_diagnostics_record_one_or_17_points() -> None:
    rows = attack_set_diagnostic_rows(
        (0.0, 0.1, 0.5),
        n_features=4,
        attack_algorithm="origin_plus_cube_corners_v1",
    )
    assert [row["actual_unique_point_count"] for row in rows] == [1, 17, 17]
    assert [row["corner_count"] for row in rows] == [0, 16, 16]
    assert all(bool(row["construction_passed"]) for row in rows)


@pytest.mark.parametrize(
    "radii",
    [
        (),
        (0.1,),
        (0.0, 0.2, 0.1),
        (0.0, -0.1),
        (0.0, float("nan")),
    ],
)
def test_invalid_radius_grids_are_rejected(radii: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        validate_attack_radii(radii)
