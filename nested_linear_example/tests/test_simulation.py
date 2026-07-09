import math

import numpy as np

from robust_cv_sim.reporting import summarize_results
from robust_cv_sim.simulation import (
    ASYMPTOTIC_STO_THRESHOLD,
    LOCAL_GH_C0,
    SimulationConfig,
    aligned_adv_center_radius,
    closed_form_losses,
    cv_adv,
    cv_adv_aligned_gaussian,
    cv_sto,
    estimate_signal_response,
    estimate_tau_hat_from_validation,
    estimate_zero_signal_bias,
    fit_a1,
    fit_a2,
    gaussian_hermite_absmean_estimator,
    generate_data,
    gh_series_estimator,
    gh_tuning,
    localized_gaussian_hermite_absmean_estimator,
    localized_gh_tuning,
    run_experiment_1,
    run_experiment_2,
    sparse_aligned_gaussian_hermite_absmean_estimator,
    split_data,
)


def test_cv_adv_matches_closed_form_pointwise_expression() -> None:
    fit = (1.2, -0.7)
    x = np.array([[0.0, 1.0], [1.0, -1.5], [2.0, 0.5]])
    y = np.array([1.0, 0.2, 3.4])
    r = 0.8
    residuals = y - (fit[0] * x[:, 0] + fit[1] * x[:, 1])
    expected = np.mean((np.abs(residuals) + r * abs(fit[1])) ** 2)
    actual = cv_adv(fit, (x, y), r)
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def test_experiment_1_default_radii_are_dense_near_switching_thresholds() -> None:
    config = SimulationConfig()
    assert {0.95, 1.0, 1.05}.issubset(config.radius_grid_adv)
    assert {1.65, 1.7, ASYMPTOTIC_STO_THRESHOLD, 1.75, 1.8}.issubset(config.radius_grid_sto)


def test_experiment_2_default_adversarial_radii_include_added_midpoints() -> None:
    config = SimulationConfig()
    assert config.exp2_radius_grid_adv == (0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.2)


def test_cv_sto_uses_supplied_validation_perturbations() -> None:
    fit = (0.9, 0.4)
    x = np.array([[1.0, -0.5], [0.3, 1.5], [-2.0, 0.1]])
    y = np.array([0.8, 2.1, -1.5])
    xi = np.array([0.2, -0.1, 0.4])
    expected = np.mean((y - (fit[0] * x[:, 0] + fit[1] * (x[:, 1] + xi))) ** 2)
    actual = cv_sto(fit, (x, y), r=0.5, xi_val=xi)
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def test_aligned_adv_center_radius_matches_linear_x2_attack_structure() -> None:
    fit = (1.1, -0.4)
    x = np.array([[1.0, 2.0], [-1.0, 0.5]])
    center, radius = aligned_adv_center_radius(fit, x, r=0.8)
    expected_center = fit[0] * x[:, 0] + fit[1] * x[:, 1]
    expected_radius = np.full(x.shape[0], 0.8 * abs(fit[1]))
    assert np.allclose(center, expected_center)
    assert np.allclose(radius, expected_radius)


def test_aligned_adv_score_reduces_to_squared_error_when_radius_is_zero() -> None:
    fit = (0.9, 0.0)
    x = np.array([[1.0, -0.5], [0.3, 1.5], [-2.0, 0.1]])
    y = np.array([0.8, 2.1, -1.5])
    expected = np.mean((y - fit[0] * x[:, 0]) ** 2)
    rng = np.random.default_rng(123)
    actual = cv_adv_aligned_gaussian(fit, (x, y), sigma=0.5, r=1.7, rng=rng)
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def test_aligned_adv_score_allows_nonzero_radius_without_oracle_tau() -> None:
    fit = (0.9, 0.4)
    x = np.array([[1.0, -0.5], [0.3, 1.5], [-2.0, 0.1]])
    y = np.array([0.8, 2.1, -1.5])
    rng = np.random.default_rng(123)
    actual = cv_adv_aligned_gaussian(fit, (x, y), sigma=0.5, r=1.7, rng=rng)
    assert math.isfinite(actual)


def test_gaussian_hermite_absmean_estimator_returns_finite_values() -> None:
    z = np.array([-1.0, 0.0, 2.0])
    rng = np.random.default_rng(2026)
    estimate = gaussian_hermite_absmean_estimator(z, sigma=0.5, n2=200, rng=rng)
    assert estimate.shape == z.shape
    assert np.isfinite(estimate).all()


def test_sparse_gh_series_excludes_constant_term() -> None:
    tuning = gh_tuning(3000)
    x = np.array([-1.5, -0.5, 0.0, 0.75, 1.25])
    full = gh_series_estimator(x, tuning, mode="full")
    sparse = gh_series_estimator(x, tuning, mode="sparse")
    assert np.allclose(full - sparse, tuning.hermite_coeffs[0])
    assert not np.allclose(full, sparse)


def test_gh_tuning_stays_out_of_constant_only_regime() -> None:
    for n2 in (60, 100, 300, 1000, 3000, 4500):
        tuning = gh_tuning(n2)
        assert tuning.k >= 1
        assert tuning.k == 24
        assert tuning.m > 0.0
        assert tuning.split_threshold > 0.0


def test_localized_gh_tuning_uses_rate_optimal_degree_two_rule() -> None:
    small = localized_gh_tuning(n2=100)
    large = localized_gh_tuning(n2=4500)
    assert small.k == 1
    assert large.k == 1
    assert math.isclose(small.m, LOCAL_GH_C0 * (100 ** -0.25), rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(large.m, LOCAL_GH_C0 * (4500 ** -0.25), rel_tol=0.0, abs_tol=1e-12)
    assert large.m < small.m


def test_localized_gaussian_hermite_absmean_estimator_returns_finite_values() -> None:
    z = np.array([-1.0, 0.0, 2.0])
    rng = np.random.default_rng(2031)
    estimate = localized_gaussian_hermite_absmean_estimator(z, sigma=0.5, n2=200, rng=rng)
    assert estimate.shape == z.shape
    assert np.isfinite(estimate).all()
    assert localized_gh_tuning(n2=200).k == 1


def test_tau_hat_from_validation_matches_moment_formula() -> None:
    w = np.array([-1.0, 0.0, 1.0, 2.0])
    sigma = 0.5
    expected = math.sqrt(max(float(np.mean(w**2) - sigma**2), 0.0))
    assert math.isclose(estimate_tau_hat_from_validation(w, sigma), expected, rel_tol=0.0, abs_tol=1e-12)


def test_gaussian_hermite_absmean_estimator_zero_signal_bias_is_not_huge() -> None:
    rng = np.random.default_rng(2027)
    z = np.zeros(20_000)
    estimate = gaussian_hermite_absmean_estimator(z, sigma=0.5, n2=3000, rng=rng)
    assert float(np.mean(estimate)) < 0.3


def test_sparse_aligned_gaussian_hermite_reduces_zero_signal_bias() -> None:
    rng_full = np.random.default_rng(2028)
    rng_sparse = np.random.default_rng(2028)
    full = estimate_zero_signal_bias(n2=3000, sigma=0.5, rng=rng_full, mode="full")
    sparse = estimate_zero_signal_bias(n2=3000, sigma=0.5, rng=rng_sparse, mode="sparse")
    assert sparse >= 0.0
    assert sparse != full


def test_gaussian_hermite_absmean_estimator_mean_increases_with_signal() -> None:
    sigma = 0.5
    n2 = 3000
    tuning = gh_tuning(n2)
    theta_grid = (0.0, 1.0, 2.0)
    means: list[float] = []
    for idx, theta in enumerate(theta_grid):
        rng = np.random.default_rng(3000 + idx)
        z = rng.normal(loc=theta / sigma, scale=1.0, size=20_000)
        estimate = gaussian_hermite_absmean_estimator(z, sigma=sigma, n2=n2, rng=rng, tuning=tuning)
        means.append(float(np.mean(estimate)))
    assert means[0] < means[1] < means[2]


def test_sparse_aligned_estimator_is_nonnegative_and_increases_with_signal() -> None:
    sigma = 0.5
    n2 = 3000
    rng = np.random.default_rng(2040)
    response = estimate_signal_response((0.0, 0.25, 0.5, 1.0, 2.0), n2=n2, sigma=sigma, rng=rng, mode="sparse")
    assert (response["mean_psi_hat"] >= 0.0).all()
    means = response["mean_psi_hat"].to_list()
    assert all(left < right for left, right in zip(means, means[1:]))


def test_sparse_aligned_wrapper_is_nonnegative() -> None:
    z = np.array([-1.0, 0.0, 2.0])
    rng = np.random.default_rng(2029)
    estimate = sparse_aligned_gaussian_hermite_absmean_estimator(z, sigma=0.5, n2=200, rng=rng)
    assert estimate.shape == z.shape
    assert np.isfinite(estimate).all()
    assert (estimate >= 0.0).all()


def test_closed_form_losses_match_proposition_formulas() -> None:
    fit_a1_model = (0.8, 0.0)
    fit_a2_model = (1.1, 0.7)
    losses = closed_form_losses(fit_a1_model, fit_a2_model, beta1=1.0, beta2=1.0, r=0.5)
    tau_sq = (1.1 - 1.0) ** 2 + (0.7 - 1.0) ** 2
    tau = math.sqrt(tau_sq)
    expected_a1 = (0.8 - 1.0) ** 2 + 1.0**2
    expected_a2_sto = tau_sq + (0.5**2 / 3.0) * (0.7**2)
    expected_a2_adv = tau_sq + 2.0 * 0.5 * abs(0.7) * tau * math.sqrt(2.0 / math.pi) + 0.5**2 * 0.7**2
    assert math.isclose(losses["loss_a1_adv"], expected_a1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(losses["loss_a1_sto"], expected_a1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(losses["loss_a2_sto"], expected_a2_sto, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(losses["loss_a2_adv"], expected_a2_adv, rel_tol=0.0, abs_tol=1e-12)


def test_r_zero_recovers_nonrobust_risks() -> None:
    fit_a1_model = (0.9, 0.0)
    fit_a2_model = (1.2, 0.8)
    losses = closed_form_losses(fit_a1_model, fit_a2_model, beta1=1.0, beta2=1.0, r=0.0)
    assert math.isclose(losses["loss_a1_adv"], losses["loss_a1_sto"], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(losses["loss_a2_adv"], losses["loss_a2_sto"], rel_tol=0.0, abs_tol=1e-12)


def test_beta2_zero_yields_nan_thresholds() -> None:
    losses = closed_form_losses((1.0, 0.0), (1.0, 0.0), beta1=1.0, beta2=0.0, r=0.7)
    assert math.isnan(losses["threshold_adv"])
    assert math.isnan(losses["threshold_sto"])


def test_generate_data_and_fit_pipeline_runs() -> None:
    rng = np.random.default_rng(2026)
    x, y = generate_data(120, beta1=1.0, beta2=1.0, sigma=0.5, rng=rng)
    train, val = split_data(x, y, train_ratio=0.4)
    fit1 = fit_a1(train)
    fit2 = fit_a2(train)
    assert len(fit1) == 2
    assert len(fit2) == 2
    assert val[0].shape[0] == 72


def test_experiment_1_reproducibility_and_schema() -> None:
    config = SimulationConfig(
        reps=2,
        n_grid=(100,),
        split_grid=(0.5,),
        radius_grid_adv=(0.0, 1.0),
        radius_grid_sto=(0.0, ASYMPTOTIC_STO_THRESHOLD),
    )
    df1 = run_experiment_1(config)
    df2 = run_experiment_1(config)
    assert df1.equals(df2)
    assert "aligned_adversarial" in set(df1["criterion"].astype(str))
    assert {
        "experiment",
        "criterion",
        "n",
        "train_ratio",
        "selected_model",
        "optimal_model",
        "threshold_adv",
        "threshold_sto",
        "delta_cv",
    }.issubset(df1.columns)


def test_experiment_2_aligned_consistency_no_longer_collapses_below_threshold() -> None:
    config = SimulationConfig(
        reps=80,
        split_grid=(0.1,),
        exp2_n_grid=(100, 500, 2000),
        exp2_radius_grid_adv=(0.5, 0.8, 1.0, 1.2),
        exp2_radius_grid_sto=(0.5,),
    )
    df = run_experiment_2(config)
    aligned = df[df["criterion"] == "aligned_adversarial"]
    consistency = (
        aligned.groupby(["r", "n"], observed=True)["selected_is_optimal"]
        .mean()
        .reset_index()
    )
    r05_n2000 = consistency[(consistency["r"] == 0.5) & (consistency["n"] == 2000)]
    r08_n2000 = consistency[(consistency["r"] == 0.8) & (consistency["n"] == 2000)]
    r12_n2000 = consistency[(consistency["r"] == 1.2) & (consistency["n"] == 2000)]
    assert not r05_n2000.empty
    assert not r08_n2000.empty
    assert not r12_n2000.empty
    assert float(r05_n2000["selected_is_optimal"].iloc[0]) > 0.8
    assert float(r08_n2000["selected_is_optimal"].iloc[0]) > 0.8
    assert float(r12_n2000["selected_is_optimal"].iloc[0]) > 0.8


def test_summary_probabilities_are_bounded() -> None:
    config = SimulationConfig(
        reps=3,
        n_grid=(100,),
        split_grid=(0.5,),
        radius_grid_adv=(0.0, 0.5),
        radius_grid_sto=(0.0, 0.5),
    )
    df = run_experiment_1(config)
    summary = summarize_results(df)
    assert ((summary["prob_selected_model_2"] >= 0.0) & (summary["prob_selected_model_2"] <= 1.0)).all()
    assert ((summary["prob_selected_is_optimal"] >= 0.0) & (summary["prob_selected_is_optimal"] <= 1.0)).all()
