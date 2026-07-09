from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
}
MODEL_STYLES = {
    "OLS": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "RidgeCV": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "LassoCV": {"color": "#2ca02c", "marker": "^", "linestyle": ":"},
    "KNN": {"color": OKABE_ITO["green"], "marker": "D", "linestyle": "-"},
    "RandomForest": {"color": OKABE_ITO["vermillion"], "marker": "P", "linestyle": "-"},
    "HistGB": {"color": OKABE_ITO["blue"], "marker": "X", "linestyle": "-"},
    "Best linear": {"color": "#666666", "marker": "o", "linestyle": "--"},
}
MODEL_LABELS = {"RandomForest": "RF"}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "figure.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.7, length=3)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str, formats: list[str]) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(figure_dir / f"{stem}.{ext}")
    plt.close(fig)


def best_linear_reference(risk: pd.DataFrame) -> pd.DataFrame:
    linear = risk[risk["model"].isin(["OLS", "RidgeCV", "LassoCV"])].copy()
    idx = linear.groupby("radius")["test_sr_mean"].idxmin()
    out = linear.loc[idx, ["radius", "test_sr_mean", "test_sr_se"]].copy()
    out["model"] = "Best linear"
    return out.sort_values("radius")


def draw_case_study_main(run_dir: Path, figure_dir: Path, formats: list[str], stem: str) -> None:
    risk = pd.read_csv(run_dir / "candidate_test_risk_by_radius.csv")
    summary = pd.read_csv(run_dir / "summary_by_radius.csv").sort_values("radius")
    risk_main = pd.concat(
        [risk[risk["model"].isin(["HistGB", "KNN", "RandomForest"])], best_linear_reference(risk)],
        ignore_index=True,
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.85))
    ax_full, ax_zoom, ax_match, ax_regret = axes.ravel()
    for model in ["HistGB", "KNN", "RandomForest", "Best linear"]:
        sub = risk_main[risk_main["model"] == model].sort_values("radius")
        style = MODEL_STYLES[model]
        x = sub["radius"].to_numpy(float)
        y = sub["test_sr_mean"].to_numpy(float)
        se = sub["test_sr_se"].to_numpy(float)
        ax_full.plot(x, y, label=MODEL_LABELS.get(model, model), linewidth=1.35, markersize=2.5, **style)
        ax_full.fill_between(x, y - se, y + se, color=style["color"], alpha=0.10, linewidth=0)
    ax_full.set_xlabel("Robust radius $r$")
    ax_full.set_ylabel("Mean robust test risk")
    ax_full.set_title("Candidate risk over the full radius range", pad=5)
    ax_full.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax_full.legend(frameon=False, ncol=2, loc="upper left")
    panel_label(ax_full, "A")
    despine(ax_full)

    zoom = risk_main[(risk_main["radius"].between(0.17, 0.24)) & (risk_main["model"].isin(["HistGB", "KNN", "RandomForest"]))]
    for model in ["HistGB", "KNN", "RandomForest"]:
        sub = zoom[zoom["model"] == model].sort_values("radius")
        if sub.empty:
            continue
        style = MODEL_STYLES[model]
        x = sub["radius"].to_numpy(float)
        y = sub["test_sr_mean"].to_numpy(float)
        se = sub["test_sr_se"].to_numpy(float)
        ax_zoom.plot(x, y, label=MODEL_LABELS.get(model, model), linewidth=1.4, markersize=3, **style)
        ax_zoom.fill_between(x, y - se, y + se, color=style["color"], alpha=0.13, linewidth=0)
    ax_zoom.set_xlabel("Robust radius $r$")
    ax_zoom.set_ylabel("Mean robust test risk")
    ax_zoom.set_title("Transition region", pad=5)
    ax_zoom.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    handles, _labels = ax_zoom.get_legend_handles_labels()
    if handles:
        ax_zoom.legend(frameon=False, loc="upper left")
    panel_label(ax_zoom, "B")
    despine(ax_zoom)

    radius = summary["radius"].to_numpy(float)
    ax_match.plot(radius, summary["clean_oracle_match_rate"], color=OKABE_ITO["orange"], linewidth=1.45, marker="o", markersize=2.8, label="Clean CV")
    ax_match.plot(radius, summary["sr_oracle_match_rate"], color=OKABE_ITO["green"], linewidth=1.45, marker="s", markersize=2.8, label="SRCV")
    ax_match.set_ylim(-0.03, 1.04)
    ax_match.set_xlabel("Robust radius $r$")
    ax_match.set_ylabel("Oracle match rate")
    ax_match.set_title("Agreement with the robust-risk oracle", pad=5)
    ax_match.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax_match.legend(frameon=False, loc="lower left")
    panel_label(ax_match, "C")
    despine(ax_match)

    for mean_col, se_col, label, color, marker in [
        ("clean_regret_mean", "clean_regret_se", "Clean CV", OKABE_ITO["orange"], "o"),
        ("sr_regret_mean", "sr_regret_se", "SRCV", OKABE_ITO["green"], "s"),
    ]:
        y = summary[mean_col].to_numpy(float)
        se = summary[se_col].to_numpy(float)
        ax_regret.plot(radius, y, color=color, linewidth=1.45, marker=marker, markersize=2.8, label=label)
        ax_regret.fill_between(radius, y - se, y + se, color=color, alpha=0.13, linewidth=0)
    ax_regret.set_xlabel("Robust radius $r$")
    ax_regret.set_ylabel("Mean robust regret")
    ax_regret.set_title("Regret relative to the radius-specific oracle", pad=5)
    ax_regret.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax_regret.legend(frameon=False, loc="upper left")
    panel_label(ax_regret, "D")
    despine(ax_regret)
    fig.subplots_adjust(hspace=0.44, wspace=0.32)
    save_figure(fig, figure_dir, stem, formats)


def format_radius(value: float) -> str:
    if abs(value - round(value)) < 1e-10:
        return str(int(round(value)))
    return f"{value:g}"


def draw_radius_mismatch(run_dir: Path, figure_dir: Path, formats: list[str], stem: str) -> None:
    posthoc = run_dir / "radius_mismatch_posthoc"
    summary = pd.read_csv(posthoc / "radius_mismatch_summary.csv")
    frequencies = pd.read_csv(posthoc / "radius_mismatch_selection_frequencies.csv")
    srcv = summary[summary["selector"] == "SRCV"].copy()
    clean = summary[summary["selector"] == "clean"].copy()
    r_val_grid = sorted(float(value) for value in srcv["r_val"].unique())
    r_eval_grid = sorted(float(value) for value in srcv["r_eval"].unique())
    x_labels = [format_radius(value) for value in r_val_grid]
    y_labels = [format_radius(value) for value in r_eval_grid]
    regret = srcv.pivot(index="r_eval", columns="r_val", values="regret_mean").loc[r_eval_grid, r_val_grid].to_numpy(float)
    match = srcv.pivot(index="r_eval", columns="r_val", values="oracle_match_rate").loc[r_eval_grid, r_val_grid].to_numpy(float)

    fig = plt.figure(figsize=(7.25, 6.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.03], hspace=0.42, wspace=0.34)
    ax_regret = fig.add_subplot(gs[0, 0])
    ax_match = fig.add_subplot(gs[0, 1])
    ax_lines = fig.add_subplot(gs[1, 0])
    ax_freq = fig.add_subplot(gs[1, 1])
    im = ax_regret.imshow(regret, aspect="auto", origin="lower", cmap="viridis")
    ax_regret.set_title("SRCV mismatch regret")
    ax_regret.set_xlabel(r"selection radius $r_{\mathrm{val}}$")
    ax_regret.set_ylabel(r"evaluation radius $r_{\mathrm{eval}}$")
    ax_regret.set_xticks(range(len(x_labels)))
    ax_regret.set_xticklabels(x_labels, rotation=65, ha="right")
    ax_regret.set_yticks(range(len(y_labels)))
    ax_regret.set_yticklabels(y_labels)
    fig.colorbar(im, ax=ax_regret, fraction=0.046, pad=0.03)
    panel_label(ax_regret, "A")
    im = ax_match.imshow(match, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    ax_match.set_title("SRCV oracle-match rate")
    ax_match.set_xlabel(r"selection radius $r_{\mathrm{val}}$")
    ax_match.set_ylabel(r"evaluation radius $r_{\mathrm{eval}}$")
    ax_match.set_xticks(range(len(x_labels)))
    ax_match.set_xticklabels(x_labels, rotation=65, ha="right")
    ax_match.set_yticks(range(len(y_labels)))
    ax_match.set_yticklabels(y_labels)
    fig.colorbar(im, ax=ax_match, fraction=0.046, pad=0.03)
    panel_label(ax_match, "B")

    line_drawn = False
    for i, r_eval in enumerate([0.20, 0.24, 0.32, 0.50]):
        if r_eval not in r_eval_grid:
            continue
        line_drawn = True
        cur = srcv[np.isclose(srcv["r_eval"], r_eval)].sort_values("r_val")
        color = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"], OKABE_ITO["vermillion"]][i]
        ax_lines.plot(cur["r_val"], cur["regret_mean"], marker="o", markersize=2.8, linewidth=1.15, color=color, label=rf"SRCV, $r_{{eval}}={format_radius(r_eval)}$")
        lower = np.maximum(cur["regret_mean"].to_numpy(float) - 1.96 * cur["regret_mcse"].to_numpy(float), 0.0)
        upper = cur["regret_mean"].to_numpy(float) + 1.96 * cur["regret_mcse"].to_numpy(float)
        ax_lines.fill_between(cur["r_val"].to_numpy(float), lower, upper, color=color, alpha=0.08, linewidth=0)
        clean_level = float(clean[np.isclose(clean["r_eval"], r_eval)].sort_values("r_val")["regret_mean"].iloc[0])
        if clean_level > 1e-10:
            ax_lines.axhline(clean_level, color=color, linestyle=":", linewidth=0.8)
    ax_lines.set_title("Regret over the selection radius")
    ax_lines.set_xlabel(r"selection radius $r_{\mathrm{val}}$")
    ax_lines.set_ylabel("mean regret")
    numeric_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ax_lines.set_xticks(numeric_ticks)
    ax_lines.set_xticklabels([format_radius(value) for value in numeric_ticks])
    ax_lines.set_xlim(-0.01, 0.51)
    ax_lines.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    if line_drawn:
        ax_lines.legend(frameon=False, ncol=1, loc="upper right", handlelength=1.5)
    despine(ax_lines)
    panel_label(ax_lines, "C")

    freq = frequencies[(frequencies["selector"] == "SRCV") & np.isclose(frequencies["r_eval"], r_eval_grid[0])]
    for model, color in {"KNN": OKABE_ITO["green"], "RandomForest": OKABE_ITO["vermillion"], "HistGB": OKABE_ITO["blue"]}.items():
        cur = freq[freq["model"] == model].sort_values("r_val")
        if cur.empty or cur["frequency"].max() <= 0:
            continue
        ax_freq.plot(cur["r_val"], cur["frequency"], marker="o", markersize=2.8, linewidth=1.25, color=color, label=MODEL_LABELS.get(model, model))
    ax_freq.set_title("SRCV selected-model frequency")
    ax_freq.set_xlabel(r"selection radius $r_{\mathrm{val}}$")
    ax_freq.set_ylabel("frequency")
    ax_freq.set_ylim(-0.03, 1.03)
    ax_freq.set_xticks(numeric_ticks)
    ax_freq.set_xticklabels([format_radius(value) for value in numeric_ticks])
    ax_freq.set_xlim(-0.01, 0.51)
    ax_freq.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_freq.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax_freq.legend(frameon=False, ncol=1, loc="center left", handlelength=1.5)
    despine(ax_freq)
    panel_label(ax_freq, "D")
    save_figure(fig, figure_dir, stem, formats)


def draw_all(config: dict, run_dir: Path, figure_dir: Path) -> None:
    apply_style()
    formats = list(config["figures"].get("formats", ["pdf", "png"]))
    draw_case_study_main(run_dir, figure_dir, formats, config["figures"]["case_study_stem"])
    draw_radius_mismatch(run_dir, figure_dir, formats, config["figures"]["mismatch_stem"])
