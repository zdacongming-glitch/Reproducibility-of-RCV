from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from scipy.special import eval_hermitenorm
from scipy.stats import qmc


@dataclass(frozen=True)
class HermiteVariant:
    name: str
    degree: int
    envelope_sigma_multiplier: float


@dataclass(frozen=True)
class OptimizationBudget:
    sobol_starts: int
    step_fractions: tuple[float, ...]
    prediction_chunk_rows: int = 2048


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


@dataclass(frozen=True)
class MultiPathResult:
    maximum_points: np.ndarray
    maximum_values: np.ndarray
    minimum_points: np.ndarray
    minimum_values: np.ndarray

    def __post_init__(self) -> None:
        maximum_points = np.asarray(self.maximum_points, dtype=float)
        minimum_points = np.asarray(self.minimum_points, dtype=float)
        maximum_values = np.asarray(self.maximum_values, dtype=float)
        minimum_values = np.asarray(self.minimum_values, dtype=float)
        if maximum_points.ndim != 3 or minimum_points.ndim != 3:
            raise ValueError("Attack path points must have shape (n, K, d)")
        if maximum_values.ndim != 2 or minimum_values.ndim != 2:
            raise ValueError("Attack path values must have shape (n, K)")
        if maximum_points.shape != minimum_points.shape:
            raise ValueError("Maximum and minimum path point shapes must match")
        if maximum_values.shape != minimum_values.shape:
            raise ValueError("Maximum and minimum path value shapes must match")
        if maximum_points.shape[:2] != maximum_values.shape:
            raise ValueError("Attack path point/value shapes are inconsistent")
        if maximum_values.shape[1] <= 0:
            raise ValueError("At least one attack path is required")
        arrays = (maximum_points, minimum_points, maximum_values, minimum_values)
        if not all(np.isfinite(array).all() for array in arrays):
            raise FloatingPointError("Attack path state contains non-finite values")

    @property
    def path_count(self) -> int:
        return int(self.maximum_values.shape[1])

    def extrema(self, path_count: int | None = None) -> ExtremaResult:
        count = self.path_count if path_count is None else int(path_count)
        if count <= 0 or count > self.path_count:
            raise ValueError(
                f"path_count must be in [1, {self.path_count}], got {count}"
            )
        maximum = np.max(self.maximum_values[:, :count], axis=1)
        minimum = np.min(self.minimum_values[:, :count], axis=1)
        if np.any(minimum > maximum + 1e-10):
            raise RuntimeError("Numerical optimizer returned minimum above maximum")
        return ExtremaResult(maximum=maximum, minimum=minimum)

    def first_paths(self, path_count: int) -> MultiPathResult:
        count = int(path_count)
        if count <= 0 or count > self.path_count:
            raise ValueError(
                f"path_count must be in [1, {self.path_count}], got {count}"
            )
        return MultiPathResult(
            maximum_points=self.maximum_points[:, :count].copy(),
            maximum_values=self.maximum_values[:, :count].copy(),
            minimum_points=self.minimum_points[:, :count].copy(),
            minimum_values=self.minimum_values[:, :count].copy(),
        )

    def subset(self, indices: np.ndarray) -> MultiPathResult:
        selected = np.asarray(indices, dtype=int)
        return MultiPathResult(
            maximum_points=self.maximum_points[selected].copy(),
            maximum_values=self.maximum_values[selected].copy(),
            minimum_points=self.minimum_points[selected].copy(),
            minimum_values=self.minimum_values[selected].copy(),
        )


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


def _normalized_sobol_points(
    n_features: int,
    sobol_starts: int,
    seed: int,
) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if sobol_starts <= 0 or sobol_starts & (sobol_starts - 1):
        raise ValueError("sobol_starts must be a positive power of two")
    sampler = qmc.Sobol(d=n_features, scramble=True, seed=int(seed))
    return 2.0 * sampler.random_base2(int(math.log2(sobol_starts))) - 1.0


def normalized_attack_starts(
    n_features: int,
    sobol_starts: int,
    seed: int,
) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    center = np.zeros((1, n_features), dtype=float)
    corners = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=n_features)), dtype=float
    )
    axes = np.vstack([np.eye(n_features), -np.eye(n_features)])
    sobol = _normalized_sobol_points(n_features, sobol_starts, seed)
    return np.vstack([center, corners, axes, sobol])


def additional_normalized_sobol_starts(
    n_features: int,
    primary_sobol_starts: int,
    seed: int,
    *,
    multiplier: int = 2,
) -> np.ndarray:
    if multiplier < 2 or multiplier & (multiplier - 1):
        raise ValueError("Sobol multiplier must be a power of two of at least two")
    total = int(primary_sobol_starts) * int(multiplier)
    points = _normalized_sobol_points(n_features, total, seed)
    return points[int(primary_sobol_starts) :]


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
        values.append(np.asarray(prediction, dtype=float))
    result = np.concatenate(values) if values else np.empty(0, dtype=float)
    if not np.isfinite(result).all():
        raise FloatingPointError("Model produced non-finite attack predictions")
    return result


def linear_box_extrema(
    estimator: object,
    x_scaled: np.ndarray,
    radius: float,
) -> ExtremaResult:
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    coefficients = np.asarray(getattr(estimator, "coef_"), dtype=float).reshape(-1)
    x = np.asarray(x_scaled, dtype=float)
    if coefficients.shape[0] != x.shape[1]:
        raise ValueError("Linear coefficient dimension does not match x_scaled")
    prediction = np.asarray(estimator.predict(x), dtype=float)
    half_range = float(radius) * float(np.sum(np.abs(coefficients)))
    return ExtremaResult(prediction + half_range, prediction - half_range)


def _coordinate_refine(
    estimator: object,
    x: np.ndarray,
    radius: float,
    normalized: np.ndarray,
    values: np.ndarray,
    *,
    step: float,
    maximize: bool,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_u = normalized.copy()
    current_values = values.copy()
    for coordinate in range(x.shape[1]):
        for direction in (-1.0, 1.0):
            proposal_u = current_u.copy()
            proposal_u[:, coordinate] = np.clip(
                proposal_u[:, coordinate] + direction * step, -1.0, 1.0
            )
            proposal_values = predict_in_chunks(
                estimator, x + radius * proposal_u, chunk_rows
            )
            better = (
                proposal_values > current_values
                if maximize
                else proposal_values < current_values
            )
            current_u[better] = proposal_u[better]
            current_values[better] = proposal_values[better]
    return current_u, current_values


def _select_paths(
    starts: np.ndarray,
    start_values: np.ndarray,
    path_count: int,
    *,
    maximize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(path_count)
    if count <= 0 or count > len(starts):
        raise ValueError(f"path_count must be in [1, {len(starts)}], got {count}")
    ordered_values = -start_values if maximize else start_values
    order = np.argsort(ordered_values, axis=1, kind="stable")[:, :count]
    values = np.take_along_axis(start_values, order, axis=1)
    points = starts[order]
    return points.copy(), values.copy()


def _refine_paths(
    estimator: object,
    x: np.ndarray,
    radius: float,
    points: np.ndarray,
    values: np.ndarray,
    steps: tuple[float, ...],
    *,
    maximize: bool,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, path_count, n_features = points.shape
    repeated_x = np.repeat(x, path_count, axis=0)
    current_points = points.reshape(n_rows * path_count, n_features).copy()
    current_values = values.reshape(n_rows * path_count).copy()
    for step in steps:
        if not math.isfinite(step) or step <= 0:
            raise ValueError("step fractions must be finite and positive")
        current_points, current_values = _coordinate_refine(
            estimator,
            repeated_x,
            radius,
            current_points,
            current_values,
            step=float(step),
            maximize=maximize,
            chunk_rows=chunk_rows,
        )
    return (
        current_points.reshape(n_rows, path_count, n_features),
        current_values.reshape(n_rows, path_count),
    )


def continuous_box_path_search(
    estimator: object,
    x_scaled: np.ndarray,
    radius: float,
    budget: OptimizationBudget,
    *,
    seed: int,
    path_count: int,
) -> MultiPathResult:
    x = np.asarray(x_scaled, dtype=float)
    if x.ndim != 2:
        raise ValueError("x_scaled must be a two-dimensional array")
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    count = int(path_count)
    if count <= 0:
        raise ValueError("path_count must be positive")
    if radius == 0:
        prediction = predict_in_chunks(estimator, x, budget.prediction_chunk_rows)
        points = np.zeros((len(x), count, x.shape[1]), dtype=float)
        values = np.repeat(prediction[:, None], count, axis=1)
        return MultiPathResult(
            points.copy(), values.copy(), points.copy(), values.copy()
        )

    starts = normalized_attack_starts(x.shape[1], budget.sobol_starts, seed)
    attack_points = (x[:, None, :] + radius * starts[None, :, :]).reshape(
        -1, x.shape[1]
    )
    start_values = predict_in_chunks(
        estimator, attack_points, budget.prediction_chunk_rows
    ).reshape(len(x), len(starts))
    maximum_points, maximum_values = _select_paths(
        starts, start_values, count, maximize=True
    )
    minimum_points, minimum_values = _select_paths(
        starts, start_values, count, maximize=False
    )
    maximum_points, maximum_values = _refine_paths(
        estimator,
        x,
        radius,
        maximum_points,
        maximum_values,
        budget.step_fractions,
        maximize=True,
        chunk_rows=budget.prediction_chunk_rows,
    )
    minimum_points, minimum_values = _refine_paths(
        estimator,
        x,
        radius,
        minimum_points,
        minimum_values,
        budget.step_fractions,
        maximize=False,
        chunk_rows=budget.prediction_chunk_rows,
    )
    result = MultiPathResult(
        maximum_points=maximum_points,
        maximum_values=maximum_values,
        minimum_points=minimum_points,
        minimum_values=minimum_values,
    )
    result.extrema()
    return result


def additional_halving_steps(
    primary_steps: tuple[float, ...],
    minimum_step_fraction: float,
) -> tuple[float, ...]:
    if not primary_steps:
        raise ValueError("Primary refinement steps must not be empty")
    minimum = float(minimum_step_fraction)
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("minimum_step_fraction must be finite and positive")
    last = float(primary_steps[-1])
    if last < minimum and not math.isclose(last, minimum):
        raise ValueError("Primary steps already pass the stronger minimum step")
    if math.isclose(last, minimum):
        return ()
    result: list[float] = []
    current = last / 2.0
    while current > minimum and not math.isclose(current, minimum):
        result.append(current)
        current /= 2.0
    if not math.isclose(current, minimum):
        raise ValueError("minimum_step_fraction must continue the halving schedule")
    result.append(minimum)
    return tuple(result)


def strict_stronger_box_extrema(
    estimator: object,
    x_scaled: np.ndarray,
    radius: float,
    budget: OptimizationBudget,
    primary_paths: MultiPathResult,
    *,
    seed: int,
    sobol_multiplier: int = 2,
    minimum_step_fraction: float = 1.0 / 128.0,
) -> ExtremaResult:
    x = np.asarray(x_scaled, dtype=float)
    if x.ndim != 2:
        raise ValueError("x_scaled must be a two-dimensional array")
    if len(x) != primary_paths.maximum_values.shape[0]:
        raise ValueError("Primary path state does not match x_scaled")
    primary = primary_paths.extrema()
    if radius == 0:
        return primary

    fine_steps = additional_halving_steps(budget.step_fractions, minimum_step_fraction)
    inherited_max_points, inherited_max_values = _refine_paths(
        estimator,
        x,
        radius,
        primary_paths.maximum_points,
        primary_paths.maximum_values,
        fine_steps,
        maximize=True,
        chunk_rows=budget.prediction_chunk_rows,
    )
    inherited_min_points, inherited_min_values = _refine_paths(
        estimator,
        x,
        radius,
        primary_paths.minimum_points,
        primary_paths.minimum_values,
        fine_steps,
        maximize=False,
        chunk_rows=budget.prediction_chunk_rows,
    )
    inherited = MultiPathResult(
        inherited_max_points,
        inherited_max_values,
        inherited_min_points,
        inherited_min_values,
    ).extrema()

    extra_starts = additional_normalized_sobol_starts(
        x.shape[1],
        budget.sobol_starts,
        seed,
        multiplier=sobol_multiplier,
    )
    if len(extra_starts) < primary_paths.path_count:
        raise ValueError("Additional Sobol starts must cover every retained path")
    extra_points = (x[:, None, :] + radius * extra_starts[None, :, :]).reshape(
        -1, x.shape[1]
    )
    extra_values = predict_in_chunks(
        estimator, extra_points, budget.prediction_chunk_rows
    ).reshape(len(x), len(extra_starts))
    new_max_points, new_max_values = _select_paths(
        extra_starts,
        extra_values,
        primary_paths.path_count,
        maximize=True,
    )
    new_min_points, new_min_values = _select_paths(
        extra_starts,
        extra_values,
        primary_paths.path_count,
        maximize=False,
    )
    stronger_steps = (*budget.step_fractions, *fine_steps)
    new_max_points, new_max_values = _refine_paths(
        estimator,
        x,
        radius,
        new_max_points,
        new_max_values,
        stronger_steps,
        maximize=True,
        chunk_rows=budget.prediction_chunk_rows,
    )
    new_min_points, new_min_values = _refine_paths(
        estimator,
        x,
        radius,
        new_min_points,
        new_min_values,
        stronger_steps,
        maximize=False,
        chunk_rows=budget.prediction_chunk_rows,
    )
    new_extrema = MultiPathResult(
        new_max_points,
        new_max_values,
        new_min_points,
        new_min_values,
    ).extrema()

    maximum = np.maximum(primary.maximum, inherited.maximum)
    maximum = np.maximum(maximum, new_extrema.maximum)
    minimum = np.minimum(primary.minimum, inherited.minimum)
    minimum = np.minimum(minimum, new_extrema.minimum)
    if np.any(maximum < primary.maximum) or np.any(minimum > primary.minimum):
        raise RuntimeError("Strict stronger envelope degraded a primary extremum")
    return ExtremaResult(maximum=maximum, minimum=minimum)


def continuous_box_extrema(
    estimator: object,
    x_scaled: np.ndarray,
    radius: float,
    budget: OptimizationBudget,
    *,
    seed: int,
    path_count: int = 1,
) -> ExtremaResult:
    return continuous_box_path_search(
        estimator,
        x_scaled,
        radius,
        budget,
        seed=seed,
        path_count=path_count,
    ).extrema()
