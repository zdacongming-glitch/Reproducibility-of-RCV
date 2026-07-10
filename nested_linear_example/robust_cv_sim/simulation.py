from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import blake2b
import math
from typing import Callable, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
from numpy.polynomial import hermite_e as npherme
from numpy.polynomial.chebyshev import Chebyshev


SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
SQRT_2 = math.sqrt(2.0)
ASYMPTOTIC_ADV_THRESHOLD = 1.0
ASYMPTOTIC_STO_THRESHOLD = math.sqrt(3.0)
GH_K_MIN = 24
LOCAL_GH_C0 = 32.0 ** 0.25
DEFAULT_SPLIT_GRID = tuple(round(i / 10.0, 1) for i in range(1, 10))
DEFAULT_EXP1_ADV_RADII = (
    0.0,
    0.25,
    0.5,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    ASYMPTOTIC_ADV_THRESHOLD,
    1.05,
    1.1,
    1.15,
    1.2,
    1.25,
    1.5,
    1.75,
    2.0,
)
DEFAULT_EXP1_STO_RADII = (
    0.0,
    0.5,
    1.0,
    1.25,
    1.5,
    1.55,
    1.6,
    1.65,
    1.7,
    ASYMPTOTIC_STO_THRESHOLD,
    1.75,
    1.8,
    1.85,
    1.9,
    2.0,
    2.5,
    3.0,
)
DEFAULT_EXP2_ADV_RADII = (0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.2)
DEFAULT_EXP2_STO_RADII = (0.5, 0.8, 1.0, 1.5, ASYMPTOTIC_STO_THRESHOLD, 2.0, 2.2)
DEFAULT_EXP1_N_GRID = (1000, 2000)
DEFAULT_EXP2_N_GRID = (100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000)

RAW_RESULT_COLUMNS = [
    "experiment",
    "criterion",
    "n",
    "train_ratio",
    "n1",
    "n2",
    "beta1",
    "beta2",
    "sigma",
    "r",
    "rep",
    "seed",
    "b1_a1",
    "b1_a2",
    "b2_a2",
    "cv_a1",
    "cv_a2",
    "selected_model",
    "loss_a1",
    "loss_a2",
    "optimal_model",
    "threshold_adv",
    "threshold_sto",
    "delta_cv",
    "selected_is_optimal",
    "selected_model_is_2",
    "optimal_model_is_2",
    "selected_tie",
    "optimal_tie",
]

READ_DTYPES = {
    "experiment": "category",
    "criterion": "category",
    "n": "int64",
    "train_ratio": "float64",
    "n1": "int64",
    "n2": "int64",
    "beta1": "float64",
    "beta2": "float64",
    "sigma": "float64",
    "r": "float64",
    "rep": "int64",
    "seed": "int64",
    "b1_a1": "float64",
    "b1_a2": "float64",
    "b2_a2": "float64",
    "cv_a1": "float64",
    "cv_a2": "float64",
    "selected_model": "int8",
    "loss_a1": "float64",
    "loss_a2": "float64",
    "optimal_model": "int8",
    "threshold_adv": "float64",
    "threshold_sto": "float64",
    "delta_cv": "float64",
    "selected_is_optimal": "int8",
    "selected_model_is_2": "int8",
    "optimal_model_is_2": "int8",
    "selected_tie": "int8",
    "optimal_tie": "int8",
}


@dataclass(frozen=True)
class SimulationConfig:
    beta1: float = 1.0
    beta2: float = 1.0
    sigma: float = 0.5
    n_grid: tuple[int, ...] = DEFAULT_EXP1_N_GRID
    split_grid: tuple[float, ...] = DEFAULT_SPLIT_GRID
    radius_grid_adv: tuple[float, ...] = DEFAULT_EXP1_ADV_RADII
    radius_grid_sto: tuple[float, ...] = DEFAULT_EXP1_STO_RADII
    reps: int = 1000
    seed: int = 20260319
    tie_tol: float = 1e-12
    exp2_n_grid: tuple[int, ...] = DEFAULT_EXP2_N_GRID
    exp2_radius_grid_adv: tuple[float, ...] = DEFAULT_EXP2_ADV_RADII
    exp2_radius_grid_sto: tuple[float, ...] = DEFAULT_EXP2_STO_RADII

    def with_updates(self, **kwargs: object) -> "SimulationConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ChunkSpec:
    experiment: str
    n: int
    train_ratio: float
    beta1: float
    beta2: float
    sigma: float
    chunk_id: str

    def describe(self) -> str:
        return (
            f"{self.experiment} | beta2={self.beta2:g} sigma={self.sigma:g} "
            f"n={self.n} split={self.train_ratio:.1f}"
        )


@dataclass(frozen=True)
class GHTuning:
    k: int
    m: float
    truncation_cap: float
    split_threshold: float
    hermite_coeffs: tuple[float, ...]


def make_rng(base_seed: int, *keys: object) -> tuple[np.random.Generator, int]:
    payload = "|".join(str(key) for key in (base_seed,) + keys).encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    seed = int.from_bytes(digest, "little", signed=False)
    return np.random.default_rng(seed), seed


def generate_data(
    n: int,
    beta1: float,
    beta2: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, 2))
    eps = rng.normal(scale=sigma, size=n)
    y = beta1 * x[:, 0] + beta2 * x[:, 1] + eps
    return x, y


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    n = x.shape[0]
    n1 = int(round(n * train_ratio))
    if n1 <= 0 or n1 >= n:
        raise ValueError(f"train_ratio={train_ratio} leads to invalid split for n={n}.")
    return (x[:n1], y[:n1]), (x[n1:], y[n1:])


def fit_a1(train: tuple[np.ndarray, np.ndarray], tie_tol: float = 1e-12) -> tuple[float, float]:
    x, y = train
    x1 = x[:, 0]
    denom = float(np.dot(x1, x1))
    if abs(denom) <= tie_tol:
        return 0.0, 0.0
    b1 = float(np.dot(x1, y) / denom)
    return b1, 0.0


def fit_a2(train: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    x, y = train
    beta_hat, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(beta_hat[0]), float(beta_hat[1])


def aligned_adv_center_radius(
    fit: tuple[float, float],
    x: np.ndarray,
    r: float,
) -> tuple[np.ndarray, np.ndarray]:
    b1, b2 = fit
    center = b1 * x[:, 0] + b2 * x[:, 1]
    radius = np.full(x.shape[0], r * abs(b2), dtype=float)
    return center, radius


@lru_cache(maxsize=None)
def gh_tuning(n2: int) -> GHTuning:
    if n2 <= 1:
        raise ValueError(f"n2 must be greater than 1, got {n2}.")
    log_n2 = math.log(n2)
    m_n2 = 8.0 * math.sqrt(log_n2)
    # The asymptotic cutoff floor(log n / 12) degenerates to 0 over our validation
    # sizes. We keep the paper's scaling but enforce a finite-sample floor calibrated
    # to avoid the constant-only estimator regime.
    k_n2 = max(GH_K_MIN, int(math.floor(log_n2 / 12.0)))

    cheb_coeffs = np.zeros(2 * k_n2 + 1, dtype=float)
    cheb_coeffs[0] = 2.0 / math.pi
    for j in range(1, k_n2 + 1):
        cheb_coeffs[2 * j] = (4.0 / math.pi) * ((-1) ** (j + 1)) / (4 * j * j - 1)

    g_poly = Chebyshev(cheb_coeffs).convert(kind=np.polynomial.Polynomial)
    p_coeffs = np.array(
        [coef * (m_n2 ** (1 - degree)) for degree, coef in enumerate(g_poly.coef)],
        dtype=float,
    )

    hermite_coeffs = np.zeros(2 * k_n2 + 1, dtype=float)
    for m in range(k_n2 + 1):
        degree = 2 * m
        if degree < len(p_coeffs):
            hermite_coeffs[degree] = p_coeffs[degree]
    return GHTuning(
        k=k_n2,
        m=m_n2,
        truncation_cap=float(n2),
        split_threshold=2.0 * math.sqrt(2.0 * log_n2),
        hermite_coeffs=tuple(float(value) for value in hermite_coeffs),
    )


def _build_gh_tuning(n2: int, m_value: float, k_value: int) -> GHTuning:
    if n2 <= 1:
        raise ValueError(f"n2 must be greater than 1, got {n2}.")
    if m_value <= 0.0:
        raise ValueError(f"m_value must be positive, got {m_value}.")
    if k_value < 1:
        raise ValueError(f"k_value must be at least 1, got {k_value}.")

    cheb_coeffs = np.zeros(2 * k_value + 1, dtype=float)
    cheb_coeffs[0] = 2.0 / math.pi
    for j in range(1, k_value + 1):
        cheb_coeffs[2 * j] = (4.0 / math.pi) * ((-1) ** (j + 1)) / (4 * j * j - 1)

    g_poly = Chebyshev(cheb_coeffs).convert(kind=np.polynomial.Polynomial)
    p_coeffs = np.array(
        [coef * (m_value ** (1 - degree)) for degree, coef in enumerate(g_poly.coef)],
        dtype=float,
    )

    hermite_coeffs = np.zeros(2 * k_value + 1, dtype=float)
    for m in range(k_value + 1):
        degree = 2 * m
        if degree < len(p_coeffs):
            hermite_coeffs[degree] = p_coeffs[degree]

    log_n2 = math.log(max(n2, 3))
    return GHTuning(
        k=k_value,
        m=m_value,
        truncation_cap=float(n2),
        split_threshold=2.0 * math.sqrt(2.0 * log_n2),
        hermite_coeffs=tuple(float(value) for value in hermite_coeffs),
    )


def localized_gh_tuning(
    n2: int,
    c0: float = LOCAL_GH_C0,
) -> GHTuning:
    if n2 <= 1:
        raise ValueError(f"n2 must be greater than 1, got {n2}.")
    if c0 <= 0.0:
        raise ValueError(f"c0 must be positive, got {c0}.")

    return _build_gh_tuning(n2=n2, m_value=c0 * (n2 ** -0.25), k_value=1)


def gh_series_estimator(
    x: np.ndarray,
    tuning: GHTuning,
    mode: Literal["full", "sparse"] = "full",
) -> np.ndarray:
    coeffs = np.asarray(tuning.hermite_coeffs, dtype=float).copy()
    if mode == "sparse":
        coeffs[0] = 0.0
    return npherme.hermeval(np.asarray(x, dtype=float), coeffs)


def _hybrid_gaussian_hermite_xi(
    x1: np.ndarray,
    x2: np.ndarray,
    tuning: GHTuning,
    mode: Literal["full", "sparse"] = "full",
) -> np.ndarray:
    s_values = gh_series_estimator(x1, tuning, mode=mode)
    s_tilde = np.minimum(s_values, tuning.truncation_cap)
    if mode == "sparse":
        s_tilde = np.maximum(0.0, s_tilde)
    return np.where(np.abs(x2) <= tuning.split_threshold, s_tilde, np.abs(x1))


def gaussian_hermite_absmean_estimator(
    z: np.ndarray,
    sigma: float | np.ndarray,
    n2: int,
    rng: np.random.Generator,
    tuning: GHTuning | None = None,
    mode: Literal["full", "sparse"] = "full",
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), z.shape)
    tuning = gh_tuning(n2) if tuning is None else tuning
    u = rng.normal(size=z.shape)
    x1 = (z + u) / SQRT_2
    x2 = (z - u) / SQRT_2
    xi_hat = _hybrid_gaussian_hermite_xi(x1, x2, tuning, mode=mode)
    return SQRT_2 * sigma_arr * xi_hat


def sparse_aligned_gaussian_hermite_absmean_estimator(
    z: np.ndarray,
    sigma: float | np.ndarray,
    n2: int,
    rng: np.random.Generator,
    tuning: GHTuning | None = None,
) -> np.ndarray:
    return gaussian_hermite_absmean_estimator(z, sigma, n2, rng, tuning=tuning, mode="sparse")


def localized_gaussian_hermite_absmean_estimator(
    z: np.ndarray,
    sigma: float | np.ndarray,
    n2: int,
    rng: np.random.Generator,
    tuning: GHTuning | None = None,
    c0: float = LOCAL_GH_C0,
) -> np.ndarray:
    tuning = localized_gh_tuning(n2=n2, c0=c0) if tuning is None else tuning
    return gaussian_hermite_absmean_estimator(z, sigma, n2, rng, tuning=tuning, mode="full")


def estimate_tau_hat_from_validation(w: np.ndarray, sigma: float) -> float:
    tau_sq_hat = float(np.mean(np.asarray(w, dtype=float) ** 2) - sigma**2)
    return math.sqrt(max(tau_sq_hat, 0.0))


def estimate_zero_signal_bias(
    n2: int,
    sigma: float,
    rng: np.random.Generator,
    draws: int = 20_000,
    mode: Literal["full", "sparse"] = "sparse",
) -> float:
    z = np.zeros(draws, dtype=float)
    estimate = gaussian_hermite_absmean_estimator(z, sigma=sigma, n2=n2, rng=rng, mode=mode)
    return float(np.mean(estimate))


def estimate_signal_response(
    theta_grid: Iterable[float],
    n2: int,
    sigma: float,
    rng: np.random.Generator,
    draws: int = 20_000,
    mode: Literal["full", "sparse"] = "sparse",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for theta in theta_grid:
        z = rng.normal(loc=theta / sigma, scale=1.0, size=draws)
        estimate = gaussian_hermite_absmean_estimator(z, sigma=sigma, n2=n2, rng=rng, mode=mode)
        rows.append(
            {
                "theta": float(theta),
                "mean_psi_hat": float(np.mean(estimate)),
                "sd_psi_hat": float(np.std(estimate, ddof=0)),
                "mean_bias": float(np.mean(estimate) - abs(theta)),
            }
        )
    return pd.DataFrame(rows)


def cv_adv(fit: tuple[float, float], val: tuple[np.ndarray, np.ndarray], r: float) -> float:
    b1, b2 = fit
    x, y = val
    residuals = y - (b1 * x[:, 0] + b2 * x[:, 1])
    pointwise = (np.abs(residuals) + r * abs(b2)) ** 2
    return float(np.mean(pointwise))


def cv_sto(
    fit: tuple[float, float],
    val: tuple[np.ndarray, np.ndarray],
    r: float,
    xi_val: np.ndarray,
) -> float:
    b1, b2 = fit
    x, y = val
    residuals = y - (b1 * x[:, 0] + b2 * (x[:, 1] + xi_val))
    return float(np.mean(residuals**2))


def cv_adv_aligned_gaussian(
    fit: tuple[float, float],
    val: tuple[np.ndarray, np.ndarray],
    sigma: float | np.ndarray,
    r: float,
    rng: np.random.Generator,
    beta1: float | None = None,
    beta2: float | None = None,
    tau: float | None = None,
) -> float:
    b1, b2 = fit
    x, y = val
    center, radius = aligned_adv_center_radius(fit, x, r)
    w_a = y - center
    if np.all(radius == 0.0):
        return float(np.mean(w_a**2))
    psi_hat = localized_gaussian_hermite_absmean_estimator(
        w_a / np.asarray(sigma, dtype=float),
        sigma,
        x.shape[0],
        rng=rng,
    )
    pointwise = (w_a**2) + (radius**2) + 2.0 * radius * psi_hat
    return float(np.mean(pointwise))


def closed_form_losses(
    fit_model_1: tuple[float, float],
    fit_model_2: tuple[float, float],
    beta1: float,
    beta2: float,
    r: float,
    tie_tol: float = 1e-12,
) -> dict[str, float]:
    b1_a1, _ = fit_model_1
    b1_a2, b2_a2 = fit_model_2
    tau_sq = (b1_a2 - beta1) ** 2 + (b2_a2 - beta2) ** 2
    tau = math.sqrt(max(tau_sq, 0.0))
    loss_a1 = (b1_a1 - beta1) ** 2 + beta2**2
    loss_a2_sto = tau_sq + (r**2 / 3.0) * (b2_a2**2)
    loss_a2_adv = tau_sq + 2.0 * r * abs(b2_a2) * tau * SQRT_2_OVER_PI + (r**2) * (b2_a2**2)

    threshold_sto = math.nan
    threshold_adv = math.nan
    if abs(b2_a2) > tie_tol:
        sto_gap = loss_a1 - tau_sq
        if sto_gap > tie_tol:
            threshold_sto = math.sqrt(3.0 * sto_gap / (b2_a2**2))

        adv_discriminant = loss_a1 - (1.0 - 2.0 / math.pi) * tau_sq
        if adv_discriminant >= -tie_tol:
            threshold_adv = (-tau * SQRT_2_OVER_PI + math.sqrt(max(adv_discriminant, 0.0))) / abs(b2_a2)

    return {
        "loss_a1_adv": loss_a1,
        "loss_a2_adv": loss_a2_adv,
        "loss_a1_sto": loss_a1,
        "loss_a2_sto": loss_a2_sto,
        "threshold_adv": threshold_adv,
        "threshold_sto": threshold_sto,
        "tau_sq": tau_sq,
        "tau": tau,
    }


def _choose_model(value_a1: float, value_a2: float, tie_tol: float) -> tuple[int, int]:
    if math.isclose(value_a1, value_a2, rel_tol=0.0, abs_tol=tie_tol):
        return 1, 1
    return (1, 0) if value_a1 < value_a2 else (2, 0)


def _base_row_context(
    experiment: str,
    n: int,
    train_ratio: float,
    beta1: float,
    beta2: float,
    sigma: float,
    rep: int,
    seed: int,
    fit_model_1: tuple[float, float],
    fit_model_2: tuple[float, float],
    thresholds: dict[str, float],
) -> dict[str, float | int | str]:
    n1 = int(round(n * train_ratio))
    return {
        "experiment": experiment,
        "n": n,
        "train_ratio": train_ratio,
        "n1": n1,
        "n2": n - n1,
        "beta1": beta1,
        "beta2": beta2,
        "sigma": sigma,
        "rep": rep,
        "seed": seed,
        "b1_a1": fit_model_1[0],
        "b1_a2": fit_model_2[0],
        "b2_a2": fit_model_2[1],
        "threshold_adv": thresholds["threshold_adv"],
        "threshold_sto": thresholds["threshold_sto"],
    }


def _iter_rows_for_base_config(
    *,
    experiment: str,
    n: int,
    train_ratio: float,
    beta1: float,
    beta2: float,
    sigma: float,
    rep: int,
    config: SimulationConfig,
    adv_radii: tuple[float, ...],
    sto_radii: tuple[float, ...],
) -> Iterator[dict[str, float | int | str]]:
    data_rng, rep_seed = make_rng(
        config.seed,
        experiment,
        f"n={n}",
        f"split={train_ratio:.1f}",
        f"beta2={beta2}",
        f"sigma={sigma}",
        f"rep={rep}",
        "data",
    )
    x, y = generate_data(n=n, beta1=beta1, beta2=beta2, sigma=sigma, rng=data_rng)
    train, val = split_data(x, y, train_ratio=train_ratio)
    fit_model_1 = fit_a1(train, tie_tol=config.tie_tol)
    fit_model_2 = fit_a2(train)
    thresholds = closed_form_losses(
        fit_model_1,
        fit_model_2,
        beta1=beta1,
        beta2=beta2,
        r=0.0,
        tie_tol=config.tie_tol,
    )
    base = _base_row_context(
        experiment,
        n,
        train_ratio,
        beta1,
        beta2,
        sigma,
        rep,
        rep_seed,
        fit_model_1,
        fit_model_2,
        thresholds,
    )

    for r in adv_radii:
        losses = closed_form_losses(
            fit_model_1,
            fit_model_2,
            beta1=beta1,
            beta2=beta2,
            r=r,
            tie_tol=config.tie_tol,
        )
        cv_a1_value = cv_adv(fit_model_1, val, r=r)
        cv_a2_value = cv_adv(fit_model_2, val, r=r)
        selected_model, selected_tie = _choose_model(cv_a1_value, cv_a2_value, config.tie_tol)
        optimal_model, optimal_tie = _choose_model(losses["loss_a1_adv"], losses["loss_a2_adv"], config.tie_tol)
        yield {
            **base,
            "criterion": "adversarial",
            "r": r,
            "cv_a1": cv_a1_value,
            "cv_a2": cv_a2_value,
            "selected_model": selected_model,
            "loss_a1": losses["loss_a1_adv"],
            "loss_a2": losses["loss_a2_adv"],
            "optimal_model": optimal_model,
            "delta_cv": cv_a2_value - cv_a1_value,
            "selected_is_optimal": int(selected_model == optimal_model),
            "selected_model_is_2": int(selected_model == 2),
            "optimal_model_is_2": int(optimal_model == 2),
            "selected_tie": selected_tie,
            "optimal_tie": optimal_tie,
        }

        aligned_rng_model_1, _ = make_rng(
            config.seed,
            experiment,
            f"n={n}",
            f"split={train_ratio:.1f}",
            f"beta2={beta2}",
            f"sigma={sigma}",
            f"rep={rep}",
            "criterion=aligned_adversarial",
            "model=1",
            f"r={r:.12f}",
        )
        aligned_rng_model_2, _ = make_rng(
            config.seed,
            experiment,
            f"n={n}",
            f"split={train_ratio:.1f}",
            f"beta2={beta2}",
            f"sigma={sigma}",
            f"rep={rep}",
            "criterion=aligned_adversarial",
            "model=2",
            f"r={r:.12f}",
        )
        cv_a1_aligned = cv_adv_aligned_gaussian(
            fit_model_1,
            val,
            sigma=sigma,
            r=r,
            rng=aligned_rng_model_1,
            beta1=beta1,
            beta2=beta2,
        )
        cv_a2_aligned = cv_adv_aligned_gaussian(
            fit_model_2,
            val,
            sigma=sigma,
            r=r,
            rng=aligned_rng_model_2,
            beta1=beta1,
            beta2=beta2,
        )
        selected_model, selected_tie = _choose_model(cv_a1_aligned, cv_a2_aligned, config.tie_tol)
        optimal_model, optimal_tie = _choose_model(losses["loss_a1_adv"], losses["loss_a2_adv"], config.tie_tol)
        yield {
            **base,
            "criterion": "aligned_adversarial",
            "r": r,
            "cv_a1": cv_a1_aligned,
            "cv_a2": cv_a2_aligned,
            "selected_model": selected_model,
            "loss_a1": losses["loss_a1_adv"],
            "loss_a2": losses["loss_a2_adv"],
            "optimal_model": optimal_model,
            "delta_cv": cv_a2_aligned - cv_a1_aligned,
            "selected_is_optimal": int(selected_model == optimal_model),
            "selected_model_is_2": int(selected_model == 2),
            "optimal_model_is_2": int(optimal_model == 2),
            "selected_tie": selected_tie,
            "optimal_tie": optimal_tie,
        }

    x_val = val[0]
    n2 = x_val.shape[0]
    for r in sto_radii:
        xi_rng, _ = make_rng(
            config.seed,
            experiment,
            f"n={n}",
            f"split={train_ratio:.1f}",
            f"beta2={beta2}",
            f"sigma={sigma}",
            f"rep={rep}",
            "criterion=stochastic",
            f"r={r:.12f}",
        )
        xi_val = xi_rng.uniform(-r, r, size=n2)
        losses = closed_form_losses(
            fit_model_1,
            fit_model_2,
            beta1=beta1,
            beta2=beta2,
            r=r,
            tie_tol=config.tie_tol,
        )
        cv_a1_value = cv_sto(fit_model_1, val, r=r, xi_val=xi_val)
        cv_a2_value = cv_sto(fit_model_2, val, r=r, xi_val=xi_val)
        selected_model, selected_tie = _choose_model(cv_a1_value, cv_a2_value, config.tie_tol)
        optimal_model, optimal_tie = _choose_model(losses["loss_a1_sto"], losses["loss_a2_sto"], config.tie_tol)
        yield {
            **base,
            "criterion": "stochastic",
            "r": r,
            "cv_a1": cv_a1_value,
            "cv_a2": cv_a2_value,
            "selected_model": selected_model,
            "loss_a1": losses["loss_a1_sto"],
            "loss_a2": losses["loss_a2_sto"],
            "optimal_model": optimal_model,
            "delta_cv": cv_a2_value - cv_a1_value,
            "selected_is_optimal": int(selected_model == optimal_model),
            "selected_model_is_2": int(selected_model == 2),
            "optimal_model_is_2": int(optimal_model == 2),
            "selected_tie": selected_tie,
            "optimal_tie": optimal_tie,
        }


def _chunk_id(
    experiment: str,
    n: int,
    train_ratio: float,
    beta1: float,
    beta2: float,
    sigma: float,
) -> str:
    return (
        f"{experiment}__n{n}__split{train_ratio:.1f}__beta1_{beta1:g}"
        f"__beta2_{beta2:g}__sigma_{sigma:g}"
    )


def iter_experiment_chunks(
    experiment: str,
    config: SimulationConfig,
) -> Iterator[ChunkSpec]:
    if experiment == "experiment_1":
        for n in config.n_grid:
            for train_ratio in config.split_grid:
                yield ChunkSpec(
                    experiment=experiment,
                    n=n,
                    train_ratio=train_ratio,
                    beta1=config.beta1,
                    beta2=config.beta2,
                    sigma=config.sigma,
                    chunk_id=_chunk_id(experiment, n, train_ratio, config.beta1, config.beta2, config.sigma),
                )
        return

    if experiment == "experiment_2":
        for n in config.exp2_n_grid:
            for train_ratio in config.split_grid:
                yield ChunkSpec(
                    experiment=experiment,
                    n=n,
                    train_ratio=train_ratio,
                    beta1=config.beta1,
                    beta2=config.beta2,
                    sigma=config.sigma,
                    chunk_id=_chunk_id(experiment, n, train_ratio, config.beta1, config.beta2, config.sigma),
                )
        return

    raise ValueError(f"Unknown experiment '{experiment}'.")


def _radii_for_experiment(
    experiment: str,
    config: SimulationConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if experiment == "experiment_1":
        return config.radius_grid_adv, config.radius_grid_sto
    if experiment == "experiment_2":
        return config.exp2_radius_grid_adv, config.exp2_radius_grid_sto
    raise ValueError(f"Unknown experiment '{experiment}'.")


def _iter_chunk_rows(chunk: ChunkSpec, config: SimulationConfig) -> Iterator[dict[str, float | int | str]]:
    adv_radii, sto_radii = _radii_for_experiment(chunk.experiment, config)
    for rep in range(config.reps):
        yield from _iter_rows_for_base_config(
            experiment=chunk.experiment,
            n=chunk.n,
            train_ratio=chunk.train_ratio,
            beta1=chunk.beta1,
            beta2=chunk.beta2,
            sigma=chunk.sigma,
            rep=rep,
            config=config,
            adv_radii=adv_radii,
            sto_radii=sto_radii,
        )


def run_chunk(
    chunk: ChunkSpec,
    config: SimulationConfig,
    progress_callback: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    adv_radii, sto_radii = _radii_for_experiment(chunk.experiment, config)
    rows: list[dict[str, float | int | str]] = []
    for rep in range(config.reps):
        rows.extend(
            _iter_rows_for_base_config(
                experiment=chunk.experiment,
                n=chunk.n,
                train_ratio=chunk.train_ratio,
                beta1=chunk.beta1,
                beta2=chunk.beta2,
                sigma=chunk.sigma,
                rep=rep,
                config=config,
                adv_radii=adv_radii,
                sto_radii=sto_radii,
            )
        )
        if progress_callback is not None:
            progress_callback(rep + 1)
    return _rows_to_dataframe(rows)


def iter_experiment_rows(
    experiment: str,
    config: SimulationConfig,
) -> Iterator[dict[str, float | int | str]]:
    for chunk in iter_experiment_chunks(experiment, config):
        yield from _iter_chunk_rows(chunk, config)


def _rows_to_dataframe(rows: Iterable[dict[str, float | int | str]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=RAW_RESULT_COLUMNS)
    if frame.empty:
        return frame
    for column, dtype in READ_DTYPES.items():
        frame[column] = frame[column].astype(dtype)
    return frame


def run_experiment_1(config: SimulationConfig) -> pd.DataFrame:
    return _rows_to_dataframe(iter_experiment_rows("experiment_1", config))


def run_experiment_2(config: SimulationConfig) -> pd.DataFrame:
    return _rows_to_dataframe(iter_experiment_rows("experiment_2", config))
