from __future__ import annotations

from pathlib import Path

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd

from src.config import RootConfig
from src.utils import ensure_dir


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
}

OKABE_ITO_SEQUENCE = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["sky"],
    OKABE_ITO["pink"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]

PROCEDURE_COLORS = {
    "erm": OKABE_ITO["blue"],
    "sr": OKABE_ITO["green"],
    "at": OKABE_ITO["vermillion"],
    "core_only": OKABE_ITO["pink"],
}

PROCEDURE_STYLE = {
    "erm": {"marker": "o"},
    "at": {"marker": "s"},
    "sr": {"marker": "^"},
    "core_only": {"marker": "D"},
}
CURVE_MARKER_SIZE = 3.5
SCATTER_MARKER_SIZE = 18


def _procedure_color(procedure: str, index: int = 0) -> str:
    return PROCEDURE_COLORS.get(str(procedure), OKABE_ITO_SEQUENCE[index % len(OKABE_ITO_SEQUENCE)])


def _save_figure(fig, output_dir: Path, name: str, config: RootConfig) -> None:
    for extension in config.plots.file_formats:
        fig.savefig(output_dir / f"{name}.{extension}", dpi=config.plots.dpi, bbox_inches="tight")
    plt.close(fig)


def _group_signature(frame: pd.DataFrame, columns: list[str]) -> str:
    row = frame.iloc[0]
    return "_".join(f"{column}-{row[column]}" for column in columns)


def _sparse_tick_positions(num_ticks: int, max_labels: int = 10) -> list[int]:
    if num_ticks <= max_labels:
        return list(range(num_ticks))
    step = max(1, round((num_ticks - 1) / (max_labels - 1)))
    positions = list(range(0, num_ticks, step))
    if positions[-1] != num_ticks - 1:
        positions.append(num_ticks - 1)
    return positions


def plot_stochastic_robust_risk_curves(aggregated_df: pd.DataFrame, output_dir: Path, config: RootConfig) -> None:
    group_cols = ["n_total", "lambda_value", "r_train", "threat_model"]
    for _, frame in aggregated_df.groupby(group_cols, dropna=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for index, (procedure, proc_frame) in enumerate(frame.groupby("procedure", dropna=False)):
            ordered = proc_frame.sort_values("r_eval")
            ax.plot(
                ordered["r_eval"],
                ordered["stochastic_robust_mse_mean"],
                label=procedure,
                color=_procedure_color(procedure, index),
                marker=PROCEDURE_STYLE.get(procedure, {}).get("marker", "o"),
                markersize=CURVE_MARKER_SIZE,
            )
        ax.set_xlabel("Evaluation radius r")
        ax.set_ylabel("Stochastic robust MSE")
        ax.set_title("Stochastic robust risk vs r")
        ax.legend()
        signature = _group_signature(frame, group_cols)
        _save_figure(fig, output_dir, f"stoch_robust_risk_{signature}", config)


def plot_selection_curves(selection_df: pd.DataFrame, output_dir: Path, config: RootConfig) -> None:
    group_cols = ["criterion", "n_total", "lambda_value", "r_train", "threat_model"]
    for _, frame in selection_df.groupby(group_cols, dropna=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        criterion = str(frame.iloc[0]["criterion"])
        for index, (procedure, proc_frame) in enumerate(frame.groupby("procedure", dropna=False)):
            ordered = proc_frame.sort_values("r_eval")
            ax.plot(
                ordered["r_eval"],
                ordered["frequency"],
                label=procedure,
                color=_procedure_color(procedure, index),
                marker=PROCEDURE_STYLE.get(procedure, {}).get("marker", "o"),
                markersize=CURVE_MARKER_SIZE,
            )
        ax.set_xlabel("Evaluation radius r")
        ax.set_ylabel("Selection frequency")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{criterion} vs r")
        ax.legend()
        signature = _group_signature(frame, group_cols)
        _save_figure(fig, output_dir, f"selection_{signature}", config)


def plot_phase_diagrams(selection_df: pd.DataFrame, output_dir: Path, config: RootConfig) -> None:
    group_cols = ["criterion", "n_total", "r_train", "threat_model"]
    for _, frame in selection_df.groupby(group_cols, dropna=False):
        pivot_source = (
            frame.sort_values(["lambda_value", "r_eval", "frequency"], ascending=[True, True, False])
            .groupby(["lambda_value", "r_eval"], dropna=False)
            .first()
            .reset_index()
        )
        procedures = sorted(pivot_source["procedure"].unique())
        mapping = {name: idx for idx, name in enumerate(procedures)}
        pivot = pivot_source.pivot(index="lambda_value", columns="r_eval", values="procedure")
        numeric = pivot.replace(mapping).astype(float)
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = ListedColormap([_procedure_color(name, index) for index, name in enumerate(procedures)])
        boundaries = [index - 0.5 for index in range(len(procedures) + 1)]
        norm = BoundaryNorm(boundaries, cmap.N)
        ax.imshow(numeric.values, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        xtick_positions = _sparse_tick_positions(len(pivot.columns))
        ytick_positions = _sparse_tick_positions(len(pivot.index))
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f"{pivot.columns[position]:g}" for position in xtick_positions], rotation=45, ha="right")
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([f"{pivot.index[position]:g}" for position in ytick_positions])
        ax.set_xlabel("Evaluation radius r")
        ax.set_ylabel("Lambda")
        ax.set_title("Phase diagram (SR-CV)")
        legend_handles = [
            Patch(facecolor=_procedure_color(name, index), edgecolor="black", label=name)
            for index, name in enumerate(procedures)
        ]
        ax.legend(handles=legend_handles, title="Procedure", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
        signature = _group_signature(frame, group_cols)
        _save_figure(fig, output_dir, f"phase_{signature}", config)


def plot_tradeoff_curves(aggregated_df: pd.DataFrame, output_dir: Path, config: RootConfig) -> None:
    group_cols = ["n_total", "lambda_value", "r_train", "threat_model"]
    key_cols = ["procedure", *group_cols]
    clean_endpoints = (
        aggregated_df[aggregated_df["r_eval"] == 0.0][[*key_cols, "stochastic_robust_mse_mean"]]
        .rename(columns={"stochastic_robust_mse_mean": "clean_endpoint_mse_mean"})
        .drop_duplicates(key_cols)
    )
    merged = aggregated_df.merge(clean_endpoints, on=key_cols, how="left")
    for _, frame in merged.groupby(group_cols, dropna=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for index, (procedure, proc_frame) in enumerate(frame.groupby("procedure", dropna=False)):
            ax.scatter(
                proc_frame["clean_endpoint_mse_mean"],
                proc_frame["stochastic_robust_mse_mean"],
                label=procedure,
                color=_procedure_color(procedure, index),
                marker=PROCEDURE_STYLE.get(procedure, {}).get("marker", "o"),
                s=SCATTER_MARKER_SIZE,
            )
        ax.set_xlabel("Stochastic robust MSE at r=0")
        ax.set_ylabel("Stochastic robust MSE")
        ax.set_title("Clean endpoint vs stochastic robust risk")
        ax.legend()
        signature = _group_signature(frame, group_cols)
        _save_figure(fig, output_dir, f"tradeoff_{signature}", config)


def plot_agreement_heatmaps(aggregated_df: pd.DataFrame, output_dir: Path, config: RootConfig) -> None:
    metric = "selected_srcv_matches_oracle_mean"
    if metric not in aggregated_df.columns:
        return

    group_cols = ["n_total", "r_train", "threat_model"]
    heatmap_source = (
        aggregated_df.groupby([*group_cols, "lambda_value", "r_eval"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    for _, frame in heatmap_source.groupby(group_cols, dropna=False):
        pivot = frame.pivot(index="lambda_value", columns="r_eval", values=metric).sort_index().sort_index(axis=1)
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("SRCV-oracle agreement")

        xtick_positions = _sparse_tick_positions(len(pivot.columns))
        ytick_positions = _sparse_tick_positions(len(pivot.index))
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels([f"{pivot.columns[position]:g}" for position in xtick_positions], rotation=45, ha="right")
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels([f"{pivot.index[position]:g}" for position in ytick_positions])
        ax.set_xlabel("Evaluation radius r")
        ax.set_ylabel("Lambda")
        ax.set_title("SRCV-oracle agreement")
        signature = _group_signature(frame, group_cols)
        _save_figure(fig, output_dir, f"agreement_heatmap_{signature}", config)


def make_all_plots(
    aggregated_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_dir: str | Path,
    config: RootConfig,
) -> None:
    del raw_df
    target = ensure_dir(output_dir)
    if aggregated_df.empty:
        return
    plot_stochastic_robust_risk_curves(aggregated_df, target, config)
    plot_tradeoff_curves(aggregated_df, target, config)
    plot_agreement_heatmaps(aggregated_df, target, config)
    if not selection_df.empty:
        plot_selection_curves(selection_df, target, config)
        plot_phase_diagrams(selection_df, target, config)
