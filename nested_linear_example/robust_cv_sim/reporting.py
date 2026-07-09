from __future__ import annotations

import csv
from functools import lru_cache
import gzip
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from .simulation import (
    ASYMPTOTIC_ADV_THRESHOLD,
    ASYMPTOTIC_STO_THRESHOLD,
    RAW_RESULT_COLUMNS,
    READ_DTYPES,
    SimulationConfig,
    iter_experiment_rows,
)


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "gray": "#6C6C6C",
    "black": "#000000",
}

OKABE_ITO_SEQUENCE = [
    OKABE_ITO["blue"],
    OKABE_ITO["green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky"],
    OKABE_ITO["pink"],
    OKABE_ITO["yellow"],
    OKABE_ITO["gray"],
]

MODEL_COLORS = {
    "A1": OKABE_ITO["blue"],
    "A2": OKABE_ITO["vermillion"],
}
MODEL_MARKERS = {
    "A1": "D",
    "A2": "s",
}

SELECTION_COLOR = MODEL_COLORS["A2"]
THRESHOLD_COLOR = OKABE_ITO["sky"]
BOXPLOT_COLOR = OKABE_ITO["green"]
ASYMPTOTIC_DENSITY_COLOR = OKABE_ITO["vermillion"]

CRITERION_BOXPLOT_COLORS = {
    "adversarial": OKABE_ITO["orange"],
    "aligned_adversarial": OKABE_ITO["sky"],
    "stochastic": OKABE_ITO["green"],
}

PAPER_GRID_FIGSIZE = (12.0, 10.0)
PAPER_SINGLE_FIGSIZE = (7.8, 4.8)
PAPER_BOXPLOT_FIGSIZE = (7.2, 4.5)
PAPER_DPI = 300
PAPER_LINEWIDTH = 3.1
PAPER_MARKER_SIZE = 8.0
PAPER_REF_LINEWIDTH = 2.0
PAPER_BOX_LINEWIDTH = 2.0
PAPER_GRID_COLOR = "#E5E5E5"
PAPER_SPINE_COLOR = "#202020"
PAPER_TICK_SIZE = 13
PAPER_LABEL_SIZE = 16
PAPER_TITLE_SIZE = 17
PAPER_SUBPLOT_TITLE_SIZE = 14
PAPER_LEGEND_SIZE = 14
PAPER_LEGEND_TITLE_SIZE = 14
THRESHOLD_HIST_TICK_SIZE = 18
THRESHOLD_HIST_LABEL_SIZE = 24
THRESHOLD_HIST_SUBPLOT_TITLE_SIZE = 20
THRESHOLD_HIST_LEGEND_SIZE = 20

sns.set_theme(style="whitegrid", palette=OKABE_ITO_SEQUENCE)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": PAPER_TICK_SIZE,
        "axes.titlesize": PAPER_TITLE_SIZE,
        "axes.labelsize": PAPER_LABEL_SIZE,
        "axes.linewidth": 1.25,
        "axes.titlepad": 8.0,
        "axes.labelpad": 7.0,
        "xtick.labelsize": PAPER_TICK_SIZE,
        "ytick.labelsize": PAPER_TICK_SIZE,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "legend.fontsize": PAPER_LEGEND_SIZE,
        "legend.title_fontsize": PAPER_LEGEND_TITLE_SIZE,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "lines.linewidth": PAPER_LINEWIDTH,
        "lines.markersize": PAPER_MARKER_SIZE,
        "savefig.dpi": PAPER_DPI,
        "figure.dpi": PAPER_DPI,
        "grid.color": PAPER_GRID_COLOR,
        "grid.linewidth": 1.0,
        "grid.alpha": 1.0,
    }
)

BASE_GROUP_COLUMNS = [
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
]

REPLICATION_GROUP_COLUMNS = [
    "experiment",
    "n",
    "train_ratio",
    "n1",
    "n2",
    "beta1",
    "beta2",
    "sigma",
    "rep",
    "seed",
]

CRITERION_SPECS = [
    ("adversarial", "threshold_adv", ASYMPTOTIC_ADV_THRESHOLD),
    ("aligned_adversarial", "threshold_adv", ASYMPTOTIC_ADV_THRESHOLD),
    ("stochastic", "threshold_sto", ASYMPTOTIC_STO_THRESHOLD),
]

EXPERIMENT_1_LOSS_THRESHOLD_CRITERIA = {"adversarial", "stochastic"}


def write_raw_results(
    experiment: str,
    config: SimulationConfig,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment}_raw.csv.gz"
    with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_RESULT_COLUMNS)
        writer.writeheader()
        for row in iter_experiment_rows(experiment, config):
            writer.writerow(row)
    return raw_path


def write_chunk(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, compression="gzip")
    tmp_path.replace(path)
    return path


def merge_chunks(experiment_dir: Path) -> Path:
    chunks_dir = experiment_dir / "chunks"
    raw_path = experiment_dir / f"{experiment_dir.name}_raw.csv.gz"
    chunk_paths = sorted(chunks_dir.glob("*.csv.gz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")

    with gzip.open(raw_path, "wt", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(RAW_RESULT_COLUMNS)
        for chunk_path in chunk_paths:
            chunk_df = pd.read_csv(chunk_path, dtype=READ_DTYPES)
            chunk_df = chunk_df[RAW_RESULT_COLUMNS]
            chunk_df.to_csv(target, index=False, header=False)
    return raw_path


def load_results(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=READ_DTYPES, usecols=usecols)


def _normal_ci(series: pd.Series) -> tuple[float, float]:
    n = int(series.count())
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        value = float(series.iloc[0])
        return value, value
    mean = float(series.mean())
    sem = float(series.std(ddof=1) / (n**0.5))
    half_width = 1.96 * sem
    return mean - half_width, mean + half_width


def _binomial_ci(series: pd.Series) -> tuple[float, float]:
    n = int(series.count())
    if n == 0:
        return float("nan"), float("nan")
    p_hat = float(series.mean())
    half_width = 1.96 * ((p_hat * (1.0 - p_hat) / n) ** 0.5)
    return p_hat - half_width, p_hat + half_width


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(BASE_GROUP_COLUMNS, observed=True, sort=True)
    summary = grouped.agg(
        reps=("rep", "count"),
        mean_cv_a1=("cv_a1", "mean"),
        mean_cv_a2=("cv_a2", "mean"),
        mean_loss_a1=("loss_a1", "mean"),
        mean_loss_a2=("loss_a2", "mean"),
        mean_delta_cv=("delta_cv", "mean"),
        prob_selected_model_2=("selected_model_is_2", "mean"),
        prob_optimal_model_2=("optimal_model_is_2", "mean"),
        prob_selected_is_optimal=("selected_is_optimal", "mean"),
        mean_selected_tie=("selected_tie", "mean"),
        mean_optimal_tie=("optimal_tie", "mean"),
    ).reset_index()

    summary["ci_selected_model_2_low"] = grouped["selected_model_is_2"].apply(lambda s: _binomial_ci(s)[0]).to_numpy()
    summary["ci_selected_model_2_high"] = grouped["selected_model_is_2"].apply(lambda s: _binomial_ci(s)[1]).to_numpy()
    summary["ci_selected_is_optimal_low"] = grouped["selected_is_optimal"].apply(lambda s: _binomial_ci(s)[0]).to_numpy()
    summary["ci_selected_is_optimal_high"] = grouped["selected_is_optimal"].apply(lambda s: _binomial_ci(s)[1]).to_numpy()
    summary["ci_mean_loss_a1_low"] = grouped["loss_a1"].apply(lambda s: _normal_ci(s)[0]).to_numpy()
    summary["ci_mean_loss_a1_high"] = grouped["loss_a1"].apply(lambda s: _normal_ci(s)[1]).to_numpy()
    summary["ci_mean_loss_a2_low"] = grouped["loss_a2"].apply(lambda s: _normal_ci(s)[0]).to_numpy()
    summary["ci_mean_loss_a2_high"] = grouped["loss_a2"].apply(lambda s: _normal_ci(s)[1]).to_numpy()
    summary["ci_mean_delta_cv_low"] = grouped["delta_cv"].apply(lambda s: _normal_ci(s)[0]).to_numpy()
    summary["ci_mean_delta_cv_high"] = grouped["delta_cv"].apply(lambda s: _normal_ci(s)[1]).to_numpy()
    return summary


def summarize_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    base = df.drop_duplicates(subset=REPLICATION_GROUP_COLUMNS).copy()
    records: list[dict[str, float | int | str]] = []
    for criterion, threshold_column, _ in CRITERION_SPECS:
        grouped = base.groupby(
            ["experiment", "n", "train_ratio", "n1", "n2", "beta1", "beta2", "sigma"],
            observed=True,
            sort=True,
        )[threshold_column]
        for keys, series in grouped:
            clean = series.dropna()
            ci_low, ci_high = _normal_ci(clean) if not clean.empty else (float("nan"), float("nan"))
            records.append(
                {
                    "experiment": keys[0],
                    "criterion": criterion,
                    "n": keys[1],
                    "train_ratio": keys[2],
                    "n1": keys[3],
                    "n2": keys[4],
                    "beta1": keys[5],
                    "beta2": keys[6],
                    "sigma": keys[7],
                    "count_defined": int(clean.count()),
                    "mean_threshold": float(clean.mean()) if not clean.empty else float("nan"),
                    "median_threshold": float(clean.median()) if not clean.empty else float("nan"),
                    "q10_threshold": float(clean.quantile(0.1)) if not clean.empty else float("nan"),
                    "q90_threshold": float(clean.quantile(0.9)) if not clean.empty else float("nan"),
                    "ci_mean_threshold_low": ci_low,
                    "ci_mean_threshold_high": ci_high,
                }
            )
    return pd.DataFrame.from_records(records)


def save_summaries(raw_path: Path, output_dir: Path) -> tuple[Path, Path]:
    df = load_results(raw_path)
    summary = summarize_results(df)
    threshold_summary = summarize_thresholds(df)
    summary_path = output_dir / raw_path.name.replace("_raw.csv.gz", "_summary.csv")
    threshold_path = output_dir / raw_path.name.replace("_raw.csv.gz", "_threshold_summary.csv")
    summary.to_csv(summary_path, index=False)
    threshold_summary.to_csv(threshold_path, index=False)
    return summary_path, threshold_path


def _save_figure(
    fig: plt.Figure,
    path: Path,
    tight_rect: tuple[float, float, float, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_rect)
    fig.savefig(path, dpi=PAPER_DPI, bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=PAPER_GRID_COLOR, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PAPER_SPINE_COLOR)
    ax.spines["bottom"].set_color(PAPER_SPINE_COLOR)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.tick_params(axis="both", which="major", colors=PAPER_SPINE_COLOR)


def _hide_unused_axes(axes: np.ndarray, used_count: int) -> None:
    for ax in list(axes.flat)[used_count:]:
        ax.set_visible(False)


def _add_grid_labels(
    fig: plt.Figure,
    xlabel: str,
    ylabel: str,
    *,
    fontsize: int = PAPER_LABEL_SIZE,
    fontweight: str = "normal",
    x_y: float = 0.025,
    y_x: float = 0.015,
) -> None:
    fig.supxlabel(xlabel, fontsize=fontsize, fontweight=fontweight, y=x_y)
    fig.supylabel(ylabel, fontsize=fontsize, fontweight=fontweight, x=y_x)


def _angle_x_ticklabels(ax: plt.Axes, rotation: float = 30.0) -> None:
    ax.tick_params(axis="x", labelrotation=rotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _train_ratio_title(split_label: str) -> str:
    return f"Train ratio = {split_label}"


def _sorted_split_labels(df: pd.DataFrame) -> list[str]:
    values = sorted(df["train_ratio"].unique())
    return [f"{value:.1f}" for value in values]


def _sorted_r_values(df: pd.DataFrame) -> list[float]:
    return sorted(float(value) for value in df["r"].dropna().unique())


def _format_r_value(value: float) -> str:
    rounded_one_decimal = round(value, 1)
    if math.isclose(value, rounded_one_decimal, rel_tol=0.0, abs_tol=1e-10):
        return f"{rounded_one_decimal:.1f}"
    return f"{value:.3f}"


def _okabe_ito_palette(n_colors: int) -> list[str]:
    return [OKABE_ITO_SEQUENCE[index % len(OKABE_ITO_SEQUENCE)] for index in range(n_colors)]


def _consistency_x_offsets(n_values: list[float], r_values: list[float]) -> dict[float, float]:
    if len(r_values) <= 1 or len(n_values) <= 1:
        return {float(r): 0.0 for r in r_values}
    min_gap = min(np.diff(sorted(float(n) for n in n_values)))
    total_span = float(min_gap) * 0.24
    step = total_span / max(len(r_values) - 1, 1)
    midpoint = (len(r_values) - 1) / 2.0
    return {
        float(r): (index - midpoint) * step
        for index, r in enumerate(r_values)
    }


def _consistency_y_limits(criterion: str) -> tuple[float, float]:
    if criterion in {"aligned_adversarial", "stochastic"}:
        return 0.4, 1.03
    return -0.05, 1.05


def _consistency_n_ticks(n_values: list[float]) -> list[float]:
    if len(n_values) <= 7:
        return n_values

    preferred = {100, 500, 1000, 2000, 3000, 4000, 5000}
    selected = [value for value in n_values if int(value) in preferred]
    if len(selected) < 4:
        indices = np.linspace(0, len(n_values) - 1, 7, dtype=int)
        selected = [n_values[int(index)] for index in indices]

    selected.extend([n_values[0], n_values[-1]])
    return sorted(set(float(value) for value in selected))


def _threshold_bins(series: pd.Series, reps: int) -> int:
    unique_count = int(series.nunique())
    if reps < 30:
        return min(10, max(3, unique_count))
    return 30


def _criterion_title(criterion: str) -> str:
    return criterion.replace("_", " ").title()


def _normal_pdf(x: np.ndarray, loc: float, scale: float) -> np.ndarray:
    if scale <= 0.0:
        return np.full_like(x, np.nan, dtype=float)
    standardized = (x - loc) / scale
    return np.exp(-0.5 * standardized**2) / (scale * math.sqrt(2.0 * math.pi))


@lru_cache(maxsize=32)
def _adversarial_switching_limit_samples(
    beta2: float,
    sigma: float,
    sample_size: int = 40_000,
) -> np.ndarray:
    if beta2 == 0.0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(20260616)
    zeta = rng.normal(loc=0.0, scale=sigma, size=(sample_size, 2))
    radius = np.sqrt(zeta[:, 0] ** 2 + zeta[:, 1] ** 2)
    beta2_sign = 1.0 if beta2 > 0.0 else -1.0
    limit = -(math.sqrt(2.0 / math.pi) * radius + beta2_sign * zeta[:, 1]) / abs(beta2)
    return limit.astype(float, copy=False)


def _kde_pdf(x_grid: np.ndarray, samples: np.ndarray) -> np.ndarray:
    clean = np.asarray(samples, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return np.full_like(x_grid, np.nan, dtype=float)

    sample_sd = float(np.std(clean, ddof=1))
    if sample_sd <= 0.0 or not math.isfinite(sample_sd):
        return np.full_like(x_grid, np.nan, dtype=float)

    bandwidth = 1.06 * sample_sd * (clean.size ** (-1.0 / 5.0))
    bandwidth = max(bandwidth, np.finfo(float).eps)
    density = np.empty_like(x_grid, dtype=float)
    chunk_size = 128
    normalizing = clean.size * bandwidth * math.sqrt(2.0 * math.pi)
    for start in range(0, x_grid.size, chunk_size):
        stop = min(start + chunk_size, x_grid.size)
        scaled = (x_grid[start:stop, None] - clean[None, :]) / bandwidth
        density[start:stop] = np.exp(-0.5 * scaled**2).sum(axis=1) / normalizing
    return density


def _asymptotic_threshold_density_curve(
    criterion: str,
    n1: int,
    beta2: float,
    sigma: float,
    observed: pd.Series | np.ndarray | None = None,
    points: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    if n1 <= 0 or beta2 == 0.0 or sigma <= 0.0:
        return np.array([], dtype=float), np.array([], dtype=float)

    observed_values = np.asarray([] if observed is None else observed, dtype=float)
    observed_values = observed_values[np.isfinite(observed_values)]

    if criterion == "stochastic":
        center = ASYMPTOTIC_STO_THRESHOLD
        scale = math.sqrt(3.0) * sigma / (abs(beta2) * math.sqrt(n1))
        support_low = center - 4.0 * scale
        support_high = center + 4.0 * scale
        if observed_values.size:
            support_low = min(support_low, float(observed_values.min()))
            support_high = max(support_high, float(observed_values.max()))
        x_grid = np.linspace(support_low, support_high, points)
        return x_grid, _normal_pdf(x_grid, center, scale)

    if criterion == "adversarial":
        limit_samples = _adversarial_switching_limit_samples(float(beta2), float(sigma))
        if limit_samples.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        threshold_samples = ASYMPTOTIC_ADV_THRESHOLD + limit_samples / math.sqrt(n1)
        support_low, support_high = np.quantile(threshold_samples, [0.005, 0.995])
        if observed_values.size:
            support_low = min(float(support_low), float(observed_values.min()))
            support_high = max(float(support_high), float(observed_values.max()))
        if math.isclose(support_low, support_high, rel_tol=0.0, abs_tol=1e-12):
            margin = max(0.05, abs(float(support_low)) * 0.05)
            support_low -= margin
            support_high += margin
        x_grid = np.linspace(float(support_low), float(support_high), points)
        return x_grid, _kde_pdf(x_grid, threshold_samples)

    return np.array([], dtype=float), np.array([], dtype=float)


def _criterion_boxplot_color(criterion: str) -> str:
    return CRITERION_BOXPLOT_COLORS.get(criterion, BOXPLOT_COLOR)


def plot_experiment_1(df: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize_results(df)
    thresholds = df.drop_duplicates(subset=REPLICATION_GROUP_COLUMNS).copy()
    thresholds["train_ratio_label"] = thresholds["train_ratio"].map(lambda value: f"{value:.1f}")
    summary["train_ratio_label"] = summary["train_ratio"].map(lambda value: f"{value:.1f}")
    reps = int(df["rep"].nunique())

    for criterion, threshold_column, asymptote in CRITERION_SPECS:
        criterion_summary = summary.loc[summary["criterion"] == criterion].copy()
        if criterion_summary.empty:
            continue
        criterion_thresholds = thresholds[["n", "train_ratio_label", threshold_column]].copy()
        criterion_thresholds = criterion_thresholds.rename(columns={threshold_column: "threshold"}).dropna()

        for n_value in sorted(criterion_summary["n"].unique()):
            subset = criterion_summary.loc[criterion_summary["n"] == n_value].copy()
            split_labels = _sorted_split_labels(subset)
            if criterion in EXPERIMENT_1_LOSS_THRESHOLD_CRITERIA:
                fig, axes = plt.subplots(3, 3, figsize=PAPER_GRID_FIGSIZE, sharex=True, sharey=True)
                for ax, split_label in zip(axes.flat, split_labels):
                    split_subset = subset.loc[subset["train_ratio_label"] == split_label]
                    ax.plot(
                        split_subset["r"],
                        split_subset["mean_loss_a1"],
                        label="A1",
                        color=MODEL_COLORS["A1"],
                        marker=MODEL_MARKERS["A1"],
                        linewidth=PAPER_LINEWIDTH,
                        markersize=PAPER_MARKER_SIZE,
                    )
                    ax.plot(
                        split_subset["r"],
                        split_subset["mean_loss_a2"],
                        label="A2",
                        color=MODEL_COLORS["A2"],
                        marker=MODEL_MARKERS["A2"],
                        linewidth=PAPER_LINEWIDTH,
                        markersize=PAPER_MARKER_SIZE,
                    )
                    ax.axvline(
                        asymptote,
                        color=OKABE_ITO["black"],
                        linestyle="--",
                        linewidth=PAPER_REF_LINEWIDTH,
                    )
                    ax.set_title(_train_ratio_title(split_label), fontsize=PAPER_SUBPLOT_TITLE_SIZE)
                    _style_axis(ax)
                _hide_unused_axes(axes, len(split_labels))
                _add_grid_labels(fig, "Robust radius $r$", "Mean robust loss")
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.995),
                    ncol=2,
                    frameon=False,
                )
                _save_figure(
                    fig,
                    output_dir / f"exp1_{criterion}_n{n_value}_loss_curves.png",
                    tight_rect=(0.05, 0.06, 1.0, 0.93),
                )

            fig, axes = plt.subplots(3, 3, figsize=PAPER_GRID_FIGSIZE, sharex=True, sharey=True)
            for ax, split_label in zip(axes.flat, split_labels):
                split_subset = subset.loc[subset["train_ratio_label"] == split_label]
                ax.plot(
                    split_subset["r"],
                    split_subset["prob_selected_model_2"],
                    color=SELECTION_COLOR,
                    marker="o",
                    linewidth=PAPER_LINEWIDTH,
                    markersize=PAPER_MARKER_SIZE,
                )
                ax.fill_between(
                    split_subset["r"],
                    split_subset["ci_selected_model_2_low"],
                    split_subset["ci_selected_model_2_high"],
                    alpha=0.16,
                    color=SELECTION_COLOR,
                )
                ax.axvline(
                    asymptote,
                    color=OKABE_ITO["black"],
                    linestyle="--",
                    linewidth=PAPER_REF_LINEWIDTH,
                )
                ax.set_title(_train_ratio_title(split_label), fontsize=PAPER_SUBPLOT_TITLE_SIZE)
                ax.set_ylim(-0.05, 1.05)
                _style_axis(ax)
            _hide_unused_axes(axes, len(split_labels))
            _add_grid_labels(fig, "Robust radius $r$", "P(select A2)")
            _save_figure(
                fig,
                output_dir / f"exp1_{criterion}_n{n_value}_selection_prob.png",
                tight_rect=(0.05, 0.06, 1.0, 0.98),
            )

            if criterion in EXPERIMENT_1_LOSS_THRESHOLD_CRITERIA:
                fig, axes = plt.subplots(3, 3, figsize=PAPER_GRID_FIGSIZE, sharey=True)
                threshold_subset = criterion_thresholds.loc[criterion_thresholds["n"] == n_value]
                for ax, split_label in zip(axes.flat, split_labels):
                    split_subset = subset.loc[subset["train_ratio_label"] == split_label]
                    split_thresholds = threshold_subset.loc[
                        threshold_subset["train_ratio_label"] == split_label,
                        "threshold",
                    ]
                    ax.hist(
                        split_thresholds,
                        bins=_threshold_bins(split_thresholds, reps=reps),
                        density=True,
                        color=THRESHOLD_COLOR,
                        alpha=0.72,
                        edgecolor="white",
                        linewidth=0.8,
                        label="Finite-sample thresholds",
                    )
                    split_meta = split_subset.iloc[0]
                    density_x, density_y = _asymptotic_threshold_density_curve(
                        criterion=criterion,
                        n1=int(split_meta["n1"]),
                        beta2=float(split_meta["beta2"]),
                        sigma=float(split_meta["sigma"]),
                        observed=split_thresholds,
                    )
                    if density_x.size and density_y.size:
                        ax.plot(
                            density_x,
                            density_y,
                            color=ASYMPTOTIC_DENSITY_COLOR,
                            linewidth=PAPER_LINEWIDTH,
                            label="Asymptotic density",
                        )
                    ax.axvline(
                        asymptote,
                        color=OKABE_ITO["black"],
                        linestyle="--",
                        linewidth=PAPER_REF_LINEWIDTH,
                        label="Asymptotic threshold",
                    )
                    ax.set_title(
                        _train_ratio_title(split_label),
                        fontsize=THRESHOLD_HIST_SUBPLOT_TITLE_SIZE,
                    )
                    ax.tick_params(axis="both", which="major", labelsize=THRESHOLD_HIST_TICK_SIZE)
                    _style_axis(ax)
                _hide_unused_axes(axes, len(split_labels))
                fig.supylabel("Density", fontsize=THRESHOLD_HIST_LABEL_SIZE, x=0.03)
                handles, labels = axes.flat[0].get_legend_handles_labels()
                if handles:
                    fig.legend(
                        handles,
                        labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.995),
                        ncol=3,
                        frameon=False,
                        prop={"size": THRESHOLD_HIST_LEGEND_SIZE, "weight": "normal"},
                    )
                _save_figure(
                    fig,
                    output_dir / f"exp1_{criterion}_n{n_value}_threshold_hist.png",
                    tight_rect=(0.06, 0.05, 1.0, 0.89),
                )


def plot_experiment_2(df: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize_results(df)
    summary["train_ratio_label"] = summary["train_ratio"].map(lambda value: f"{value:.1f}")
    df = df.copy()
    df["train_ratio_label"] = df["train_ratio"].map(lambda value: f"{value:.1f}")
    df["n_label"] = df["n"].astype(str)
    df["r_label"] = df["r"].map(_format_r_value)
    summary["r_label"] = summary["r"].map(_format_r_value)

    for criterion in ("adversarial", "aligned_adversarial", "stochastic"):
        criterion_summary = summary.loc[summary["criterion"] == criterion]
        if criterion_summary.empty:
            continue
        criterion_raw = df.loc[df["criterion"] == criterion]
        for split_label in _sorted_split_labels(criterion_summary):
            split_summary = criterion_summary.loc[criterion_summary["train_ratio_label"] == split_label].copy()
            r_values = _sorted_r_values(split_summary)
            n_values = sorted(float(value) for value in split_summary["n"].unique())
            offsets = _consistency_x_offsets(n_values, r_values)
            palette = _okabe_ito_palette(len(r_values))
            markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
            fig, ax = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE)
            for index, r_value in enumerate(r_values):
                line_data = split_summary.loc[split_summary["r"] == r_value].sort_values("n").copy()
                x_plot = line_data["n"].astype(float) + offsets[r_value]
                ax.plot(
                    x_plot,
                    line_data["prob_selected_is_optimal"],
                    label=_format_r_value(r_value),
                    color=palette[index],
                    marker=markers[index % len(markers)],
                    linewidth=PAPER_LINEWIDTH,
                    markersize=PAPER_MARKER_SIZE,
                )
            ax.set_title(f"{_criterion_title(criterion)} consistency, train ratio = {split_label}")
            ax.set_ylabel("P(select robust-optimal)")
            ax.set_xlabel("n")
            ax.set_ylim(*_consistency_y_limits(criterion))
            n_ticks = _consistency_n_ticks(n_values)
            ax.set_xticks(n_ticks)
            ax.set_xticklabels([str(int(value)) for value in n_ticks])
            _style_axis(ax)
            _angle_x_ticklabels(ax)
            ax.legend(
                title="Robust radius $r$",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                borderaxespad=0.0,
            )
            _save_figure(fig, output_dir / f"exp2_{criterion}_trainratio_{split_label}_consistency.png")

            split_raw = criterion_raw.loc[criterion_raw["train_ratio_label"] == split_label].copy()
            boxplot_color = _criterion_boxplot_color(criterion)
            for r_value in r_values:
                r_label = _format_r_value(r_value)
                r_raw = split_raw.loc[split_raw["r_label"] == r_label].copy()
                if r_raw.empty:
                    continue
                fig, ax = plt.subplots(figsize=PAPER_BOXPLOT_FIGSIZE)
                sns.boxplot(
                    data=r_raw,
                    x="n_label",
                    y="delta_cv",
                    color=boxplot_color,
                    order=[str(int(value)) for value in n_values],
                    fliersize=3.5,
                    linewidth=PAPER_BOX_LINEWIDTH,
                    saturation=0.9,
                    width=0.62,
                    ax=ax,
                )
                ax.axhline(
                    0.0,
                    color=OKABE_ITO["gray"],
                    linestyle="--",
                    linewidth=PAPER_REF_LINEWIDTH,
                    alpha=0.85,
                )
                ax.set_title(f"{_criterion_title(criterion)} Delta CV, train ratio = {split_label}, r = {r_label}")
                ax.set_xlabel("n")
                ax.set_ylabel("Delta CV")
                _style_axis(ax)
                _angle_x_ticklabels(ax)
                ax.legend(
                    handles=[
                        Patch(
                            facecolor=boxplot_color,
                            edgecolor=boxplot_color,
                            label=_criterion_title(criterion),
                        )
                    ],
                    title="CV criterion",
                    loc="best",
                    frameon=False,
                )
                _save_figure(
                    fig,
                    output_dir / f"exp2_{criterion}_trainratio_{split_label}_r_{r_label}_delta_cv_boxplot.png",
                )


def plot_experiment_3(df: pd.DataFrame, output_dir: Path) -> None:
    summary = summarize_results(df)
    summary["train_ratio_label"] = summary["train_ratio"].map(lambda value: f"{value:.1f}")

    for criterion in ("adversarial", "aligned_adversarial", "stochastic"):
        criterion_summary = summary.loc[summary["criterion"] == criterion]
        if criterion_summary.empty:
            continue
        for n_value in sorted(criterion_summary["n"].unique()):
            for split_label in _sorted_split_labels(criterion_summary):
                subset = criterion_summary.loc[
                    (criterion_summary["n"] == n_value)
                    & (criterion_summary["train_ratio_label"] == split_label)
                ].copy()
                if subset.empty:
                    continue

                loss_long = pd.concat(
                    [
                        subset.assign(model="A1", mean_loss=subset["mean_loss_a1"]),
                        subset.assign(model="A2", mean_loss=subset["mean_loss_a2"]),
                    ],
                    ignore_index=True,
                )
                g = sns.relplot(
                    data=loss_long,
                    x="r",
                    y="mean_loss",
                    hue="model",
                    kind="line",
                    col="beta2",
                    row="sigma",
                    palette=MODEL_COLORS,
                    height=4.0,
                    aspect=1.25,
                    facet_kws={"sharey": False},
                )
                g.figure.suptitle(
                    f"Experiment 3 {_criterion_title(criterion)} losses, n={n_value}, train_ratio={split_label}",
                    y=1.02,
                )
                plot_path = output_dir / f"exp3_{criterion}_n{n_value}_trainratio_{split_label}_loss.png"
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                g.savefig(plot_path, dpi=220, bbox_inches="tight")
                plt.close(g.figure)

                g = sns.relplot(
                    data=subset,
                    x="r",
                    y="prob_selected_model_2",
                    kind="line",
                    color=SELECTION_COLOR,
                    col="beta2",
                    row="sigma",
                    height=4.0,
                    aspect=1.25,
                    facet_kws={"sharey": True},
                )
                g.figure.suptitle(
                    f"Experiment 3 {_criterion_title(criterion)} selection, n={n_value}, train_ratio={split_label}",
                    y=1.02,
                )
                plot_path = output_dir / f"exp3_{criterion}_n{n_value}_trainratio_{split_label}_selection.png"
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                g.savefig(plot_path, dpi=220, bbox_inches="tight")
                plt.close(g.figure)


def generate_outputs(raw_path: Path, experiment: str, output_dir: Path, skip_plots: bool = False) -> None:
    df = load_results(raw_path)
    save_summaries(raw_path, output_dir)
    if skip_plots:
        return
    plots_dir = output_dir / "plots"
    if experiment == "experiment_1":
        plot_experiment_1(df, plots_dir)
        return
    if experiment == "experiment_2":
        plot_experiment_2(df, plots_dir)
        return
    if experiment == "experiment_3":
        plot_experiment_3(df, plots_dir)
        return
    raise ValueError(f"Unknown experiment '{experiment}'.")
