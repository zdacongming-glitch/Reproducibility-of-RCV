from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .config import require_data_file


FEATURE_COLUMNS = ["AT", "V", "AP", "RH"]
TARGET_COLUMN = "PE"


@dataclass(frozen=True)
class ExperimentConfig:
    radii: tuple[float, ...]
    outer_repeats: int
    inner_repeats: int
    test_size: float
    validation_fraction_of_pool: float
    base_seed: int
    val_draws: int
    test_draws: int
    rf_trees: int
    hgb_max_iter: int
    candidate_order: tuple[str, ...]
    progress: bool = True
    inner_progress: bool = False


def experiment_config_from_dict(config: dict, progress: bool = True) -> ExperimentConfig:
    exp = config["experiment"]
    models = config["models"]
    return ExperimentConfig(
        radii=tuple(float(value) for value in exp["radii"]),
        outer_repeats=int(exp["outer_repeats"]),
        inner_repeats=int(exp["inner_repeats"]),
        test_size=float(exp["test_size"]),
        validation_fraction_of_pool=float(exp["validation_fraction_of_pool"]),
        base_seed=int(exp["base_seed"]),
        val_draws=int(exp["val_draws"]),
        test_draws=int(exp["test_draws"]),
        rf_trees=int(models["rf_trees"]),
        hgb_max_iter=int(models["hgb_max_iter"]),
        candidate_order=tuple(models["candidate_order"]),
        progress=progress,
        inner_progress=False,
    )


def load_ccpp_data(config: dict) -> pd.DataFrame:
    path = require_data_file(config)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported CCPP data file type: {path}")
    frame.columns = [str(col).strip().upper() for col in frame.columns]
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required CCPP columns: {missing}")
    frame = frame[required].dropna().copy()
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    expected_rows = int(config["data"].get("expected_rows", len(frame)))
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    return frame


def deterministic_seed(base_seed: int, radius_index: int, outer: int, inner: int = 0) -> int:
    return int(base_seed + 1_000_003 * radius_index + 10_007 * outer + 101 * inner)


def make_models(n_train: int, seed: int, rf_trees: int, hgb_max_iter: int) -> dict[str, object]:
    n_features = len(FEATURE_COLUMNS)
    knn_neighbors = int(np.clip(math.ceil(n_train ** (4.0 / (4.0 + n_features))), 25, 150))
    min_leaf_rf = max(5, math.ceil(0.005 * n_train))
    min_leaf_hgb = max(20, math.ceil(0.005 * n_train))
    ridge_alphas = n_train * np.logspace(-6, 2, 49)
    return {
        "OLS": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=ridge_alphas, cv=5, scoring="neg_mean_squared_error"),
        "LassoCV": LassoCV(
            cv=5,
            n_alphas=100,
            eps=1e-4,
            max_iter=20_000,
            random_state=seed,
            n_jobs=-1,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=knn_neighbors, weights="distance"),
        "RandomForest": RandomForestRegressor(
            n_estimators=rf_trees,
            min_samples_leaf=min_leaf_rf,
            max_features=1.0,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_iter=hgb_max_iter,
            learning_rate=0.06,
            max_leaf_nodes=15,
            min_samples_leaf=max(30, min_leaf_hgb),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        ),
    }


def perturbations(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    radius: float,
    draws: int,
) -> np.ndarray:
    if radius == 0:
        return np.zeros((draws, n_samples, n_features), dtype=float)
    return rng.uniform(low=-radius, high=radius, size=(draws, n_samples, n_features))


def perturbed_mse(estimator: object, x_scaled: np.ndarray, y: np.ndarray, delta: np.ndarray) -> float:
    return float(np.mean([mean_squared_error(y, estimator.predict(x_scaled + draw)) for draw in delta]))


def choose_min(scores: dict[str, float], candidate_order: tuple[str, ...]) -> str:
    ordered = sorted(scores.items(), key=lambda item: (item[1], candidate_order.index(item[0])))
    return ordered[0][0]


def majority_vote(
    selections: list[str], tie_scores: dict[str, float], candidate_order: tuple[str, ...]
) -> str:
    counts = Counter(selections)
    max_count = max(counts.values())
    tied = [name for name, count in counts.items() if count == max_count]
    return sorted(tied, key=lambda name: (tie_scores[name], candidate_order.index(name)))[0]


def run_experiment(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    all_indices = np.arange(len(frame))
    validation_rows: list[dict] = []
    test_rows: list[dict] = []
    selection_rows: list[dict] = []
    outer_rows: list[dict] = []

    radius_iter: Iterable[tuple[int, float]] = tqdm(
        enumerate(config.radii),
        total=len(config.radii),
        desc="radii",
        disable=not config.progress,
    )
    for radius_index, radius in radius_iter:
        outer_iter: Iterable[int] = tqdm(
            range(config.outer_repeats),
            total=config.outer_repeats,
            desc=f"outer r={radius:g}",
            leave=False,
            disable=not config.progress,
        )
        for outer in outer_iter:
            outer_seed = deterministic_seed(config.base_seed, 0, outer)
            pool_idx, test_idx = train_test_split(
                all_indices, test_size=config.test_size, random_state=outer_seed, shuffle=True
            )
            inner_clean_choices: list[str] = []
            inner_sr_choices: list[str] = []
            clean_score_sums = {name: 0.0 for name in config.candidate_order}
            sr_score_sums = {name: 0.0 for name in config.candidate_order}
            test_score_sums = {name: 0.0 for name in config.candidate_order}

            inner_iter: Iterable[int] = tqdm(
                range(config.inner_repeats),
                total=config.inner_repeats,
                desc=f"inner o={outer}",
                leave=False,
                disable=(not config.progress) or (not config.inner_progress),
            )
            for inner in inner_iter:
                split_seed = deterministic_seed(config.base_seed, 0, outer, inner)
                perturb_seed = deterministic_seed(config.base_seed, radius_index + 1, outer, inner)
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
                val_delta = perturbations(
                    np.random.default_rng(perturb_seed + 33),
                    x_val.shape[0],
                    x_val.shape[1],
                    radius,
                    config.val_draws,
                )
                test_delta = perturbations(
                    np.random.default_rng(perturb_seed + 77),
                    x_test.shape[0],
                    x_test.shape[1],
                    radius,
                    config.test_draws,
                )
                models = make_models(
                    n_train=len(train_idx),
                    seed=split_seed,
                    rf_trees=config.rf_trees,
                    hgb_max_iter=config.hgb_max_iter,
                )
                clean_scores: dict[str, float] = {}
                sr_scores: dict[str, float] = {}
                test_scores: dict[str, float] = {}
                for name in config.candidate_order:
                    estimator = clone(models[name])
                    estimator.fit(x_train, y_train)
                    clean_scores[name] = float(mean_squared_error(y_val, estimator.predict(x_val)))
                    sr_scores[name] = perturbed_mse(estimator, x_val, y_val, val_delta)
                    test_scores[name] = perturbed_mse(estimator, x_test, y_test, test_delta)
                    clean_score_sums[name] += clean_scores[name]
                    sr_score_sums[name] += sr_scores[name]
                    test_score_sums[name] += test_scores[name]
                    validation_rows.append(
                        {
                            "radius": radius,
                            "outer": outer,
                            "inner": inner,
                            "model": name,
                            "clean_cv": clean_scores[name],
                            "sr_cv": sr_scores[name],
                            "train_size": len(train_idx),
                            "validation_size": len(val_idx),
                            "test_size": len(test_idx),
                        }
                    )
                    test_rows.append(
                        {
                            "radius": radius,
                            "outer": outer,
                            "inner": inner,
                            "model": name,
                            "test_sr": test_scores[name],
                        }
                    )
                clean_choice = choose_min(clean_scores, config.candidate_order)
                sr_choice = choose_min(sr_scores, config.candidate_order)
                inner_clean_choices.append(clean_choice)
                inner_sr_choices.append(sr_choice)
                selection_rows.append(
                    {
                        "radius": radius,
                        "outer": outer,
                        "inner": inner,
                        "clean_choice": clean_choice,
                        "sr_choice": sr_choice,
                    }
                )

            mean_clean_scores = {
                name: clean_score_sums[name] / config.inner_repeats for name in config.candidate_order
            }
            mean_sr_scores = {
                name: sr_score_sums[name] / config.inner_repeats for name in config.candidate_order
            }
            mean_test_scores = {
                name: test_score_sums[name] / config.inner_repeats for name in config.candidate_order
            }
            final_clean = majority_vote(inner_clean_choices, mean_clean_scores, config.candidate_order)
            final_sr = majority_vote(inner_sr_choices, mean_sr_scores, config.candidate_order)
            oracle = choose_min(mean_test_scores, config.candidate_order)
            oracle_score = mean_test_scores[oracle]
            outer_rows.append(
                {
                    "radius": radius,
                    "outer": outer,
                    "clean_final_choice": final_clean,
                    "sr_final_choice": final_sr,
                    "oracle_choice": oracle,
                    "clean_test_sr": mean_test_scores[final_clean],
                    "sr_test_sr": mean_test_scores[final_sr],
                    "oracle_test_sr": oracle_score,
                    "clean_regret": mean_test_scores[final_clean] - oracle_score,
                    "sr_regret": mean_test_scores[final_sr] - oracle_score,
                    "clean_oracle_match": int(final_clean == oracle),
                    "sr_oracle_match": int(final_sr == oracle),
                    **{f"test_sr_{name}": mean_test_scores[name] for name in config.candidate_order},
                    **{f"clean_vote_{name}": inner_clean_choices.count(name) for name in config.candidate_order},
                    **{f"sr_vote_{name}": inner_sr_choices.count(name) for name in config.candidate_order},
                }
            )
    return {
        "inner_validation_scores": pd.DataFrame(validation_rows),
        "inner_test_scores": pd.DataFrame(test_rows),
        "inner_selections": pd.DataFrame(selection_rows),
        "outer_summary": pd.DataFrame(outer_rows),
    }


def standard_error(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def summarize_outputs(tables: dict[str, pd.DataFrame], candidate_order: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    outer = tables["outer_summary"]
    by_radius = (
        outer.groupby("radius")
        .agg(
            clean_oracle_match_rate=("clean_oracle_match", "mean"),
            sr_oracle_match_rate=("sr_oracle_match", "mean"),
            clean_regret_mean=("clean_regret", "mean"),
            sr_regret_mean=("sr_regret", "mean"),
            clean_regret_se=("clean_regret", standard_error),
            sr_regret_se=("sr_regret", standard_error),
            clean_test_sr_mean=("clean_test_sr", "mean"),
            sr_test_sr_mean=("sr_test_sr", "mean"),
            oracle_test_sr_mean=("oracle_test_sr", "mean"),
            outer_repeats=("outer", "nunique"),
        )
        .reset_index()
    )
    selection_rows = []
    for selector, column in [("clean", "clean_final_choice"), ("sr", "sr_final_choice")]:
        counts = outer.groupby(["radius", column]).size().reset_index(name="count")
        totals = outer.groupby("radius").size().rename("total").reset_index()
        counts = counts.merge(totals, on="radius", how="left")
        counts["frequency"] = counts["count"] / counts["total"]
        counts = counts.rename(columns={column: "model"})
        counts["selector"] = selector
        selection_rows.append(counts[["radius", "selector", "model", "count", "total", "frequency"]])
    selection_frequencies = pd.concat(selection_rows, ignore_index=True)
    risk_rows = []
    for name in candidate_order:
        column = f"test_sr_{name}"
        temp = (
            outer.groupby("radius")[column]
            .agg(test_sr_mean="mean", test_sr_se=standard_error)
            .reset_index()
        )
        temp["model"] = name
        risk_rows.append(temp)
    return {
        "summary_by_radius": by_radius,
        "selection_frequencies": selection_frequencies,
        "candidate_test_risk_by_radius": pd.concat(risk_rows, ignore_index=True),
    }


def write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def run_and_write(config: dict, output_dir: Path, progress: bool = True) -> None:
    exp_config = experiment_config_from_dict(config, progress=progress)
    frame = load_ccpp_data(config)
    tables = run_experiment(frame, exp_config)
    summaries = summarize_outputs(tables, exp_config.candidate_order)
    write_tables(output_dir, {**tables, **summaries})
