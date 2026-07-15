from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.special import eval_hermitenorm


@dataclass(frozen=True)
class HermiteVariant:
    name: str
    degree: int
    envelope_sigma_multiplier: float


@dataclass(frozen=True)
class ExtremaResult:
    maximum: np.ndarray
    minimum: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.maximum + self.minimum) / 2.0

    @property
    def radius(self) -> np.ndarray:
        return np.maximum((self.maximum - self.minimum) / 2.0, 0.0)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def chebyshev_even_coefficients(degree: int) -> np.ndarray:
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    series = Chebyshev([2.0 / math.pi])
    for j in range(1, degree + 1):
        coefficient = (4.0 / math.pi) * ((-1.0) ** (j + 1)) / (4.0 * j * j - 1.0)
        term = np.zeros(2 * j + 1, dtype=float)
        term[-1] = coefficient
        series = series + Chebyshev(term)
    power = series.convert(kind=Polynomial).coef
    return np.asarray([power[2 * m] for m in range(degree + 1)], dtype=float)


def gaussian_hermite_psi(
    y: np.ndarray,
    center: np.ndarray,
    sigma: float,
    variant: HermiteVariant,
) -> tuple[np.ndarray, float]:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    y_array = np.asarray(y, dtype=float)
    center_array = np.asarray(center, dtype=float)
    if y_array.shape != center_array.shape:
        raise ValueError("y and center must have identical shapes")
    envelope = float(variant.envelope_sigma_multiplier * sigma)
    z = (y_array - center_array) / sigma
    coefficients = chebyshev_even_coefficients(variant.degree)
    psi = np.zeros_like(z, dtype=float)
    for m, coefficient in enumerate(coefficients):
        psi += (
            coefficient
            * envelope ** (1 - 2 * m)
            * sigma ** (2 * m)
            * eval_hermitenorm(2 * m, z)
        )
    if not np.isfinite(psi).all():
        raise FloatingPointError(
            f"Non-finite Gaussian-Hermite values for {variant.name}"
        )
    return psi, envelope


def aacv_observation_scores(
    y: np.ndarray,
    center: np.ndarray,
    robust_radius: np.ndarray,
    sigma: float,
    variant: HermiteVariant,
) -> tuple[np.ndarray, np.ndarray, float]:
    y_array = np.asarray(y, dtype=float)
    center_array = np.asarray(center, dtype=float)
    rho = np.asarray(robust_radius, dtype=float)
    if y_array.shape != center_array.shape or y_array.shape != rho.shape:
        raise ValueError("y, center, and robust_radius must have identical shapes")
    if np.any(rho < -1e-12):
        raise ValueError("robust_radius must be nonnegative")
    psi, envelope = gaussian_hermite_psi(y_array, center_array, sigma, variant)
    scores = np.square(y_array - center_array) + np.square(rho) + 2.0 * rho * psi
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"Non-finite AACV scores for {variant.name}")
    return scores, psi, envelope


def predict_in_chunks(
    estimator: object,
    points: np.ndarray,
    chunk_rows: int,
) -> np.ndarray:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    values: list[np.ndarray] = []
    for start in range(0, len(points), chunk_rows):
        prediction = estimator.predict(points[start : start + chunk_rows])
        values.append(np.asarray(prediction, dtype=float).reshape(-1))
    result = np.concatenate(values) if values else np.empty(0, dtype=float)
    if not np.isfinite(result).all():
        raise FloatingPointError("Model produced non-finite attack predictions")
    return result


def validate_attack_radii(radii: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(radius) for radius in radii)
    if not values or not math.isclose(values[0], 0.0, abs_tol=1e-15):
        raise ValueError("AACV radii must begin at zero")
    if any(not math.isfinite(radius) or radius < 0 for radius in values):
        raise ValueError("AACV radii must be finite and nonnegative")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("AACV radii must be strictly increasing")
    return values


def cube_corner_signs(n_features: int) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    return np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=int(n_features))),
        dtype=float,
    )


def cube_corner_shell(radius: float, n_features: int) -> np.ndarray:
    value = float(radius)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("corner-shell radius must be finite and positive")
    return value * cube_corner_signs(n_features)


def corner_attack_points(radius: float, n_features: int) -> np.ndarray:
    value = float(radius)
    if not math.isfinite(value) or value < 0:
        raise ValueError("attack radius must be finite and nonnegative")
    origin = np.zeros((1, n_features), dtype=float)
    if math.isclose(value, 0.0, abs_tol=1e-15):
        return origin
    return np.vstack([origin, cube_corner_shell(value, n_features)])


def corner_attack_extrema(
    estimator: object,
    x_scaled: np.ndarray,
    radii: Iterable[float],
    *,
    prediction_chunk_rows: int = 2048,
) -> tuple[ExtremaResult, ...]:
    grid = validate_attack_radii(radii)
    x = np.asarray(x_scaled, dtype=float)
    if x.ndim != 2:
        raise ValueError("x_scaled must be a two-dimensional array")
    if x.shape[1] <= 0:
        raise ValueError("x_scaled must contain at least one feature")
    if not np.isfinite(x).all():
        raise ValueError("x_scaled must contain only finite values")
    if prediction_chunk_rows <= 0:
        raise ValueError("prediction_chunk_rows must be positive")

    center = predict_in_chunks(estimator, x, prediction_chunk_rows)
    if center.shape != (len(x),):
        raise ValueError("Estimator predictions must contain one value per row")
    results = [ExtremaResult(maximum=center.copy(), minimum=center.copy())]

    signs = cube_corner_signs(x.shape[1])
    corners_per_shell = len(signs)
    sample_chunk_rows = max(1, prediction_chunk_rows // corners_per_shell)
    for radius in grid[1:]:
        shell_maximum = np.full(len(x), -np.inf, dtype=float)
        shell_minimum = np.full(len(x), np.inf, dtype=float)
        offsets = float(radius) * signs
        for start in range(0, len(x), sample_chunk_rows):
            stop = min(start + sample_chunk_rows, len(x))
            points = (x[start:stop, None, :] + offsets[None, :, :]).reshape(
                -1, x.shape[1]
            )
            predictions = predict_in_chunks(
                estimator, points, prediction_chunk_rows
            ).reshape(stop - start, corners_per_shell)
            shell_maximum[start:stop] = np.max(predictions, axis=1)
            shell_minimum[start:stop] = np.min(predictions, axis=1)
        maximum = np.maximum(center, shell_maximum)
        minimum = np.minimum(center, shell_minimum)
        if np.any(minimum > maximum):
            raise RuntimeError("Finite attack-set minimum exceeds maximum")
        results.append(ExtremaResult(maximum=maximum, minimum=minimum))
    return tuple(results)


def attack_set_diagnostic_rows(
    radii: Iterable[float],
    *,
    n_features: int,
    attack_algorithm: str,
) -> list[dict[str, object]]:
    grid = validate_attack_radii(radii)
    rows: list[dict[str, object]] = []
    expected_corner_count = 2**n_features
    for radius_index, radius in enumerate(grid):
        points = corner_attack_points(radius, n_features)
        point_set = {tuple(row) for row in points.tolist()}
        origin = tuple(0.0 for _ in range(n_features))
        origin_count = int(np.sum(np.all(points == 0.0, axis=1)))
        corner_count = 0 if radius_index == 0 else expected_corner_count
        expected_count = 1 + corner_count
        within_radius = bool(np.all(np.abs(points) <= radius + 1e-12))
        on_current_shell = bool(
            radius_index == 0
            or np.all(np.isclose(np.abs(points[1:]), radius, rtol=0, atol=1e-12))
        )
        construction_passed = bool(
            len(point_set) == expected_count
            and len(points) == expected_count
            and origin_count == 1
            and origin in point_set
            and within_radius
            and on_current_shell
        )
        rows.append(
            {
                "radius_index": radius_index,
                "radius": radius,
                "dimension": n_features,
                "corner_count": corner_count,
                "expected_unique_point_count": expected_count,
                "actual_unique_point_count": len(point_set),
                "origin_count": origin_count,
                "origin_present": origin in point_set,
                "corners_on_current_radius": on_current_shell,
                "within_current_radius": within_radius,
                "construction_passed": construction_passed,
                "attack_algorithm": attack_algorithm,
            }
        )
    return rows
