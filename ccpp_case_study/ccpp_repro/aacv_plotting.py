from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aacv_working_error import PRIMARY_ANALYSIS_VARIANT, PRIMARY_REFERENCE
from .plotting import (
    MODEL_LABELS,
    MODEL_STYLES,
    OKABE_ITO,
    apply_style,
    despine,
    format_radius,
    panel_label,
    save_figure,
)


def _sparse_axis_ticks(
    labels: list[str], maximum_ticks: int = 11
) -> tuple[np.ndarray, list[str]]:
    if len(labels) <= maximum_ticks:
        positions = np.arange(len(labels), dtype=int)
    else:
        positions = np.unique(
            np.rint(np.linspace(0, len(labels) - 1, maximum_ticks)).astype(int)
        )
    return positions, [labels[index] for index in positions]


def _best_linear(scores: pd.DataFrame) -> pd.DataFrame:
    linear = scores[scores["model"].isin(["OLS", "RidgeCV", "LassoCV"])]
    index = linear.groupby("radius")["test_aacv_mean"].idxmin()
    result = linear.loc[index, ["radius", "test_aacv_mean", "test_aacv_se"]].copy()
    result["model"] = "Best linear"
    return result


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["analysis_variant"].astype(str) == PRIMARY_ANALYSIS_VARIANT
    ].copy()


def _plot_candidate_curves(ax: plt.Axes, scores: pd.DataFrame) -> None:
    for model in ["HistGB", "KNN", "RandomForest", "Best linear"]:
        cell = scores[scores["model"] == model].sort_values("radius")
        if cell.empty:
            continue
        style = MODEL_STYLES[model]
        x = cell["radius"].to_numpy(float)
        y = cell["test_aacv_mean"].to_numpy(float)
        se = cell["test_aacv_se"].fillna(0).to_numpy(float)
        ax.plot(
            x,
            y,
            label=MODEL_LABELS.get(model, model),
            linewidth=1.35,
            markersize=2.5,
            **style,
        )
        ax.fill_between(
            x, y - se, y + se, color=style["color"], alpha=0.10, linewidth=0
        )


def draw_aacv_case_study_main(
    run_dir: Path,
    figure_dir: Path,
    formats: list[str],
    stem: str,
    transition_window: tuple[float, float],
) -> None:
    candidate = _primary(pd.read_csv(run_dir / "aacv_candidate_test_scores.csv"))
    summary = _primary(pd.read_csv(run_dir / "aacv_summary_by_radius.csv"))
    shown = pd.concat(
        [
            candidate[candidate["model"].isin(["HistGB", "KNN", "RandomForest"])],
            _best_linear(candidate),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.85))
    ax_full, ax_transition_panel, ax_match, ax_excess = axes.ravel()
    _plot_candidate_curves(ax_full, shown)
    ax_full.set_xlabel("Robust radius $r$")
    ax_full.set_ylabel("Mean held-out AACV score")
    ax_full.set_title("Candidate score over the full radius range", pad=5)
    ax_full.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax_full.legend(frameon=False, ncol=2, loc="upper left")
    ax_full.set_xlim(float(shown["radius"].min()), float(shown["radius"].max()))
    panel_label(ax_full, "A")
    despine(ax_full)

    transition_min, transition_max = transition_window
    ax_transition_panel.set_axis_off()
    transition_specs = (
        ("Early", 0.0, 0.10, (0.00, 0.08, 0.46, 0.78)),
        ("Late", transition_min, transition_max, (0.54, 0.08, 0.46, 0.78)),
    )
    for label, radius_min, radius_max, bounds in transition_specs:
        transition = shown[
            shown["radius"].between(radius_min, radius_max)
            & shown["model"].isin(["HistGB", "KNN", "RandomForest"])
        ]
        transition_ax = ax_transition_panel.inset_axes(bounds)
        _plot_candidate_curves(transition_ax, transition)
        transition_ax.set_xlim(radius_min, radius_max)
        transition_ax.set_xticks(
            [radius_min, (radius_min + radius_max) / 2.0, radius_max]
        )
        transition_ax.set_xticklabels(
            [
                format_radius(radius_min),
                format_radius((radius_min + radius_max) / 2.0),
                format_radius(radius_max),
            ]
        )
        transition_ax.set_title(
            f"{label}: {format_radius(radius_min)}-{format_radius(radius_max)}",
            pad=3,
        )
        transition_ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
        despine(transition_ax)
    ax_transition_panel.text(
        0.5,
        -0.10,
        "Robust radius $r$",
        transform=ax_transition_panel.transAxes,
        ha="center",
        va="top",
    )
    ax_transition_panel.text(
        -0.16,
        0.47,
        "Mean held-out AACV score",
        transform=ax_transition_panel.transAxes,
        rotation=90,
        ha="center",
        va="center",
    )
    ax_transition_panel.text(
        0.5,
        1.02,
        "Transition regions",
        transform=ax_transition_panel.transAxes,
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    panel_label(ax_transition_panel, "B")

    colors = {"clean": OKABE_ITO["orange"], "aacv": OKABE_ITO["green"]}
    labels = {"clean": "Clean CV", "aacv": "AACV"}
    markers = {"clean": "o", "aacv": "s"}
    for selector in ("clean", "aacv"):
        cell = summary[summary["selector"] == selector].sort_values("radius")
        ax_match.plot(
            cell["radius"],
            cell["heldout_best_match_rate"],
            color=colors[selector],
            marker=markers[selector],
            markersize=2.8,
            linewidth=1.45,
            label=labels[selector],
        )
    ax_match.set_ylim(-0.03, 1.04)
    ax_match.set_xlabel("Robust radius $r$")
    ax_match.set_ylabel("Held-out-best match rate")
    ax_match.set_title("Agreement with the held-out best candidate", pad=5)
    ax_match.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax_match.legend(frameon=False, loc="lower left")
    panel_label(ax_match, "C")
    despine(ax_match)

    for selector in ("clean", "aacv"):
        cell = summary[summary["selector"] == selector].sort_values("radius")
        x = cell["radius"].to_numpy(float)
        y = cell["heldout_score_excess_mean"].to_numpy(float)
        se = cell["heldout_score_excess_se"].fillna(0).to_numpy(float)
        ax_excess.plot(
            x,
            y,
            color=colors[selector],
            marker=markers[selector],
            markersize=2.8,
            linewidth=1.45,
            label=labels[selector],
        )
        ax_excess.fill_between(
            x,
            np.maximum(y - se, 0),
            y + se,
            color=colors[selector],
            alpha=0.13,
            linewidth=0,
        )
    ax_excess.set_xlabel("Robust radius $r$")
    ax_excess.set_ylabel("Mean held-out score excess")
    ax_excess.set_title("Excess relative to the radius-specific held-out best", pad=5)
    ax_excess.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    ax_excess.legend(frameon=False, loc="upper left")
    panel_label(ax_excess, "D")
    despine(ax_excess)
    fig.subplots_adjust(hspace=0.44, wspace=0.32)
    save_figure(fig, figure_dir, stem, formats)


def draw_aacv_mismatch(
    run_dir: Path,
    figure_dir: Path,
    formats: list[str],
    stem: str,
    representative_eval_radii: list[float],
) -> None:
    posthoc = run_dir / "aacv_radius_mismatch"
    summary = pd.read_csv(posthoc / "radius_mismatch_summary.csv")
    frequencies = pd.read_csv(posthoc / "radius_mismatch_selection_frequencies.csv")
    aacv = summary[summary["selector"] == "aacv"].copy()
    clean = summary[summary["selector"] == "clean"].copy()
    r_val_grid = sorted(float(value) for value in aacv["r_val"].unique())
    r_eval_grid = sorted(float(value) for value in aacv["r_eval"].unique())
    x_labels = [format_radius(value) for value in r_val_grid]
    y_labels = [format_radius(value) for value in r_eval_grid]
    excess = (
        aacv.pivot(index="r_eval", columns="r_val", values="heldout_score_excess_mean")
        .loc[r_eval_grid, r_val_grid]
        .to_numpy(float)
    )
    match = (
        aacv.pivot(index="r_eval", columns="r_val", values="heldout_best_match_rate")
        .loc[r_eval_grid, r_val_grid]
        .to_numpy(float)
    )

    fig = plt.figure(figsize=(7.25, 6.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.03], hspace=0.42, wspace=0.34)
    ax_excess = fig.add_subplot(grid[0, 0])
    ax_match = fig.add_subplot(grid[0, 1])
    ax_lines = fig.add_subplot(grid[1, 0])
    ax_frequency = fig.add_subplot(grid[1, 1])

    image = ax_excess.imshow(excess, aspect="auto", origin="lower", cmap="viridis")
    ax_excess.set_title("AACV radius-mismatch score excess")
    ax_excess.set_xlabel(r"selection radius $r_{\mathrm{val}}$")
    ax_excess.set_ylabel(r"evaluation radius $r_{\mathrm{eval}}$")
    x_tick_positions, x_tick_labels = _sparse_axis_ticks(x_labels)
    y_tick_positions, y_tick_labels = _sparse_axis_ticks(y_labels)
    ax_excess.set_xticks(x_tick_positions, x_tick_labels, rotation=65, ha="right")
    ax_excess.set_yticks(y_tick_positions, y_tick_labels)
    fig.colorbar(image, ax=ax_excess, fraction=0.046, pad=0.03)
    panel_label(ax_excess, "A")

    image = ax_match.imshow(
        match, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1
    )
    ax_match.set_title("AACV held-out-best match rate")
    ax_match.set_xlabel(r"selection radius $r_{\mathrm{val}}$")
    ax_match.set_ylabel(r"evaluation radius $r_{\mathrm{eval}}$")
    ax_match.set_xticks(x_tick_positions, x_tick_labels, rotation=65, ha="right")
    ax_match.set_yticks(y_tick_positions, y_tick_labels)
    fig.colorbar(image, ax=ax_match, fraction=0.046, pad=0.03)
    panel_label(ax_match, "B")

    palette = [
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["vermillion"],
    ]
    for index, requested in enumerate(representative_eval_radii):
        if not any(np.isclose(r_eval_grid, requested)):
            continue
        cell = aacv[np.isclose(aacv["r_eval"], requested)].sort_values("r_val")
        color = palette[index % len(palette)]
        ax_lines.plot(
            cell["r_val"],
            cell["heldout_score_excess_mean"],
            marker="o",
            markersize=2.8,
            linewidth=1.15,
            color=color,
            label=rf"AACV, $r_{{eval}}={format_radius(requested)}$",
        )
        clean_level = float(
            clean[np.isclose(clean["r_eval"], requested)]
            .sort_values("r_val")["heldout_score_excess_mean"]
            .iloc[0]
        )
        ax_lines.axhline(clean_level, color=color, linestyle=":", linewidth=0.8)
    ax_lines.set_title("Score excess over the selection radius")
    ax_lines.set_xlabel(r"selection radius $r_{\mathrm{val}}$")
    ax_lines.set_ylabel("Mean held-out score excess")
    minimum_radius = float(min(r_val_grid))
    maximum_radius = float(max(r_val_grid))
    ax_lines.set_xticks(np.linspace(minimum_radius, maximum_radius, 6))
    margin = max((maximum_radius - minimum_radius) * 0.02, 0.01)
    ax_lines.set_xlim(minimum_radius - margin, maximum_radius + margin)
    ax_lines.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    if ax_lines.lines:
        ax_lines.legend(frameon=False, loc="upper right", handlelength=1.5)
    despine(ax_lines)
    panel_label(ax_lines, "C")

    selected = frequencies[
        (frequencies["selector"] == "aacv")
        & np.isclose(frequencies["r_eval"], r_eval_grid[0])
    ]
    for model in ["KNN", "RandomForest", "HistGB", "OLS", "RidgeCV", "LassoCV"]:
        cell = selected[selected["model"] == model].sort_values("r_val")
        if cell.empty or cell["frequency"].max() <= 0:
            continue
        style = MODEL_STYLES[model]
        ax_frequency.plot(
            cell["r_val"],
            cell["frequency"],
            label=MODEL_LABELS.get(model, model),
            linewidth=1.25,
            markersize=2.6,
            **style,
        )
    ax_frequency.set_title("AACV selected-model frequency")
    ax_frequency.set_xlabel(r"selection radius $r_{\mathrm{val}}$")
    ax_frequency.set_ylabel("Frequency")
    ax_frequency.set_ylim(-0.03, 1.03)
    ax_frequency.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    if ax_frequency.lines:
        ax_frequency.legend(frameon=False, loc="best")
    despine(ax_frequency)
    panel_label(ax_frequency, "D")
    save_figure(fig, figure_dir, stem, formats)


def draw_working_error_diagnostics(
    run_dir: Path,
    figure_dir: Path,
    formats: list[str],
    stem: str,
) -> None:
    references = pd.read_csv(run_dir / "working_reference_diagnostics.csv")
    scales = pd.read_csv(run_dir / "scale_estimator_diagnostics.csv")
    covariates = pd.read_csv(run_dir / "residual_covariate_diagnostics.csv")
    attack_sets = pd.read_csv(run_dir / "attack_set_diagnostics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0))
    ax_scale, ax_gaussian, ax_covariate, ax_attack = axes.ravel()

    scale_order = [
        "oof_residual_histgb",
        "oof_residual_randomforest",
        "oof_residual_knn",
        "tong_wang",
        "fan_yao_histgb",
        "oof_residual_ols",
        "oof_residual_ridgecv",
        "oof_residual_lassocv",
    ]
    values = [
        scales[scales["scale_estimator"] == name]["sigma"].to_numpy(float)
        for name in scale_order
    ]
    positions = np.arange(len(scale_order))
    means = [float(np.mean(value)) for value in values]
    ax_scale.bar(
        positions,
        means,
        color=[OKABE_ITO["blue"]] + ["#888888"] * (len(scale_order) - 1),
        width=0.72,
    )
    ax_scale.set_xticks(
        positions,
        [
            "HGB",
            "RF",
            "KNN",
            "TW",
            "FY",
            "OLS",
            "Ridge",
            "Lasso",
        ],
        rotation=35,
        ha="right",
    )
    ax_scale.set_title("Cross-fitted working scales")
    ax_scale.set_ylabel(r"Mean $\widehat{\sigma}$")
    ax_scale.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    despine(ax_scale)
    panel_label(ax_scale, "A")

    primary = references[references["working_reference_learner"] == PRIMARY_REFERENCE]
    scatter = ax_gaussian.scatter(
        primary["residual_skewness"],
        primary["residual_excess_kurtosis"],
        c=primary["jarque_bera_pvalue"],
        cmap="viridis",
        s=20,
        vmin=0.0,
        vmax=1.0,
        edgecolors="none",
    )
    ax_gaussian.axvline(0, color="#AAAAAA", linewidth=0.7)
    ax_gaussian.axhline(0, color="#AAAAAA", linewidth=0.7)
    ax_gaussian.set_title("Primary OOF residual diagnostics")
    ax_gaussian.set_xlabel("Skewness")
    ax_gaussian.set_ylabel("Excess kurtosis")
    fig.colorbar(scatter, ax=ax_gaussian, fraction=0.046, pad=0.03, label="JB p-value")
    despine(ax_gaussian)
    panel_label(ax_gaussian, "B")

    relationship = (
        covariates.assign(
            maximum_absolute_relationship=lambda frame: frame[
                [
                    "pearson_residual",
                    "pearson_squared_residual",
                    "spearman_residual",
                    "spearman_squared_residual",
                ]
            ]
            .abs()
            .max(axis=1)
        )
        .groupby("working_reference_learner", as_index=False)[
            "maximum_absolute_relationship"
        ]
        .max()
    )
    relationship["rank"] = relationship["working_reference_learner"].map(
        {name: index for index, name in enumerate(MODEL_STYLES)}
    )
    relationship = relationship.sort_values("rank")
    ax_covariate.bar(
        np.arange(len(relationship)),
        relationship["maximum_absolute_relationship"],
        color=[
            OKABE_ITO["blue"] if name == PRIMARY_REFERENCE else "#888888"
            for name in relationship["working_reference_learner"]
        ],
        width=0.72,
    )
    ax_covariate.set_xticks(
        np.arange(len(relationship)),
        [
            MODEL_LABELS.get(name, name)
            for name in relationship["working_reference_learner"]
        ],
        rotation=35,
        ha="right",
    )
    ax_covariate.set_title("Residual-covariate relationships")
    ax_covariate.set_ylabel("Maximum absolute correlation")
    ax_covariate.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    despine(ax_covariate)
    panel_label(ax_covariate, "C")

    attack_sets = attack_sets.sort_values("radius")
    ax_attack.step(
        attack_sets["radius"],
        attack_sets["actual_unique_point_count"],
        where="mid",
        color=OKABE_ITO["orange"],
        linewidth=1.3,
    )
    ax_attack.scatter(
        attack_sets["radius"],
        attack_sets["actual_unique_point_count"],
        color=OKABE_ITO["orange"],
        s=12,
        zorder=3,
    )
    ax_attack.set_title("Finite attack-set size")
    ax_attack.set_xlabel("Robust radius $r$")
    ax_attack.set_ylabel("Attack points per observation")
    ax_attack.set_ylim(bottom=0)
    ax_attack.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    despine(ax_attack)
    panel_label(ax_attack, "D")
    fig.subplots_adjust(hspace=0.52, wspace=0.36)
    save_figure(fig, figure_dir, stem, formats)


def _agreement_by_variant(summary: pd.DataFrame, role: str) -> pd.DataFrame:
    return (
        summary[(summary["analysis_role"] == role) & (summary["selector"] == "aacv")]
        .groupby("analysis_variant", as_index=False)["selected_model_agreement_rate"]
        .mean()
        .sort_values("selected_model_agreement_rate", ascending=False)
    )


def draw_aacv_sensitivity(
    run_dir: Path,
    figure_dir: Path,
    formats: list[str],
    stem: str,
) -> None:
    reference = pd.read_csv(run_dir / "reference_selection.csv").sort_values("rank")
    scale = pd.read_csv(run_dir / "scale_estimator_summary.csv")
    sensitivity = pd.read_csv(run_dir / "working_error_sensitivity_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0))
    ax_reference, ax_scale, ax_reference_agreement, ax_other_agreement = axes.ravel()

    y = np.arange(len(reference))
    ax_reference.barh(
        y,
        reference["mean_clean_mse"],
        xerr=reference["clean_mse_mcse"],
        color=[
            OKABE_ITO["blue"] if model == PRIMARY_REFERENCE else "#888888"
            for model in reference["model"]
        ],
        height=0.68,
    )
    ax_reference.set_yticks(
        y, [MODEL_LABELS.get(model, model) for model in reference["model"]]
    )
    ax_reference.invert_yaxis()
    ax_reference.set_title("Global clean-risk reference selection")
    ax_reference.set_xlabel("Mean clean validation MSE")
    ax_reference.grid(axis="x", color="#E5E5E5", linewidth=0.55)
    despine(ax_reference)
    panel_label(ax_reference, "A")

    scale = scale.sort_values("sigma2_ratio_to_primary_mean")
    ax_scale.barh(
        np.arange(len(scale)),
        scale["sigma2_ratio_to_primary_mean"],
        color=[
            OKABE_ITO["blue"] if value == "oof_residual_histgb" else "#888888"
            for value in scale["scale_estimator"]
        ],
        height=0.68,
    )
    ax_scale.axvline(1.0, color="#666666", linestyle="--", linewidth=0.8)
    ax_scale.set_yticks(
        np.arange(len(scale)),
        [value.replace("oof_residual_", "") for value in scale["scale_estimator"]],
    )
    ax_scale.set_title("Scale-estimator sensitivity")
    ax_scale.set_xlabel(r"Mean $\widehat{\sigma}^2/\widehat{\sigma}^2_{primary}$")
    ax_scale.grid(axis="x", color="#E5E5E5", linewidth=0.55)
    despine(ax_scale)
    panel_label(ax_scale, "B")

    reference_agreement = _agreement_by_variant(sensitivity, "reference_sensitivity")
    ax_reference_agreement.bar(
        np.arange(len(reference_agreement)),
        reference_agreement["selected_model_agreement_rate"],
        color="#888888",
        width=0.7,
    )
    ax_reference_agreement.set_xticks(
        np.arange(len(reference_agreement)),
        [
            value.removeprefix("reference_").removesuffix("_oof_j2_b6")
            for value in reference_agreement["analysis_variant"]
        ],
        rotation=35,
        ha="right",
    )
    ax_reference_agreement.set_ylim(-0.03, 1.03)
    ax_reference_agreement.set_title("Reference-learner selection agreement")
    ax_reference_agreement.set_ylabel("Agreement with primary AACV")
    ax_reference_agreement.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    despine(ax_reference_agreement)
    panel_label(ax_reference_agreement, "C")

    other = pd.concat(
        [
            _agreement_by_variant(sensitivity, "scale_sensitivity"),
            _agreement_by_variant(sensitivity, "hermite_sensitivity"),
        ],
        ignore_index=True,
    )
    ax_other_agreement.bar(
        np.arange(len(other)),
        other["selected_model_agreement_rate"],
        color=[OKABE_ITO["orange"]] * 2 + ["#888888"] * max(len(other) - 2, 0),
        width=0.7,
    )
    labels = [
        value.replace("scale_", "").replace("hermite_histgb_oof_", "")
        for value in other["analysis_variant"]
    ]
    ax_other_agreement.set_xticks(
        np.arange(len(other)), labels, rotation=35, ha="right"
    )
    ax_other_agreement.set_ylim(-0.03, 1.03)
    ax_other_agreement.set_title("Scale and Hermite selection agreement")
    ax_other_agreement.set_ylabel("Agreement with primary AACV")
    ax_other_agreement.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    despine(ax_other_agreement)
    panel_label(ax_other_agreement, "D")
    fig.subplots_adjust(hspace=0.55, wspace=0.40)
    save_figure(fig, figure_dir, stem, formats)


def draw_aacv_all(config: dict, run_dir: Path, figure_dir: Path) -> None:
    apply_style()
    formats = list(config["figures"].get("formats", ["pdf", "png"]))
    transition_window = tuple(
        float(value) for value in config["figures"]["transition_window"]
    )
    if len(transition_window) != 2 or transition_window[0] >= transition_window[1]:
        raise ValueError("AACV transition window must contain two increasing values")
    draw_aacv_case_study_main(
        run_dir,
        figure_dir,
        formats,
        config["figures"]["main_stem"],
        transition_window,
    )
    draw_aacv_mismatch(
        run_dir,
        figure_dir,
        formats,
        config["figures"]["mismatch_stem"],
        [float(value) for value in config["mismatch"]["representative_eval_radii"]],
    )
    draw_working_error_diagnostics(
        run_dir, figure_dir, formats, config["figures"]["diagnostics_stem"]
    )
    draw_aacv_sensitivity(
        run_dir, figure_dir, formats, config["figures"]["sensitivity_stem"]
    )
