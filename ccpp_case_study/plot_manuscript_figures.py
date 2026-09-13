"""Generate the official CCPP manuscript and supplementary figures from saved CSVs.

This is a presentation-only entry point. It neither fits models nor recomputes
scores or Monte Carlo summaries. See README.md for the input/output mapping.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ccpp_repro.plotting import (
    MODEL_LABELS, MODEL_STYLES, OKABE_ITO, apply_style, despine, format_radius,
)

ROOT = Path(__file__).resolve().parent
PRIMARY = "primary_histgb_oof_j2_b6"
NONLINEAR = ["HistGB", "KNN", "RandomForest"]
LINEAR = ["OLS", "RidgeCV", "LassoCV"]
STEMS = {
    "heldout": "fig_ccpp_manuscript_heldout",
    "mismatch": "fig_ccpp_manuscript_mismatch",
    "heldout_excess": "fig_ccpp_supp_heldout_excess",
    "mismatch_excess": "fig_ccpp_supp_mismatch_excess",
}
SINGLE_PANELS = {
    "fig_ccpp_srcv_heldout_regret": ("SRCV", "heldout", "D"),
    "fig_ccpp_aacv_heldout_excess": ("AACV", "heldout", "D"),
    "fig_ccpp_srcv_mismatch_regret": ("SRCV", "mismatch", "A"),
    "fig_ccpp_srcv_mismatch_slices": ("SRCV", "mismatch", "C"),
    "fig_ccpp_aacv_mismatch_excess": ("AACV", "mismatch", "A"),
    "fig_ccpp_aacv_mismatch_slices": ("AACV", "mismatch", "C"),
}


@dataclass
class Results:
    method: str
    candidate: pd.DataFrame
    summary: pd.DataFrame
    mismatch: pd.DataFrame
    frequencies: pd.DataFrame
    transition: tuple[float, float]
    slices: list[float]
    files: list[Path]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete(frame: pd.DataFrame, keys: dict, numeric: list[str], name: str) -> None:
    """Reject incomplete grids, duplicates and invalid plotted quantities."""
    missing = set(keys).union(numeric).difference(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    index = pd.MultiIndex.from_frame(frame[list(keys)])
    expected = pd.MultiIndex.from_product(keys.values(), names=keys)
    if index.has_duplicates or len(index) != len(expected) or len(expected.difference(index)):
        raise ValueError(f"{name}: incomplete or duplicate configured grid")
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError(f"{name}: nonfinite plotted values")
    for col in numeric:
        if ("se" == col or "mcse" == col) and (frame[col] < 0).any():
            raise ValueError(f"{name}: negative uncertainty")
        if col in ("match", "frequency") and not frame[col].between(0, 1).all():
            raise ValueError(f"{name}: {col} outside [0, 1]")


def load_results(run_dir: Path, method: str, config: dict) -> Results:
    """Normalize column names only; preserve every published numerical value."""
    aacv = method == "AACV"
    experiment = config["aacv_experiment" if aacv else "experiment"]
    radii = experiment["radii"]
    models = config["models"]["candidate_order"]
    subdir = config["mismatch"]["output_subdir"]
    names = (
        ["aacv_candidate_test_scores.csv", "aacv_summary_by_radius.csv"]
        if aacv else ["candidate_test_risk_by_radius.csv", "summary_by_radius.csv"]
    )
    files = [run_dir / name for name in names] + [
        run_dir / subdir / "radius_mismatch_summary.csv",
        run_dir / subdir / "radius_mismatch_selection_frequencies.csv",
    ]
    frames = [pd.read_csv(path) for path in files]
    if aacv:
        primary = [v["name"] for v in config["working_error"]["analysis_variants"]
                   if v["role"] == "primary"]
        if primary != [PRIMARY]:
            raise ValueError("The official figures require the declared primary AACV variant")
        for i, frame in enumerate(frames):
            frames[i] = frame.loc[frame["analysis_variant"] == PRIMARY].copy()
            if frames[i].empty:
                raise ValueError(f"{files[i].name}: missing primary AACV variant")
    candidate, summary, mismatch, frequencies = frames
    for frame in frames:
        if "outer_repeats" in frame and not frame["outer_repeats"].eq(experiment["outer_repeats"]).all():
            raise ValueError(f"{method}: outer-repeat count differs from the full configuration")
    candidate = candidate.rename(columns={
        "test_aacv_mean" if aacv else "test_sr_mean": "mean",
        "test_aacv_se" if aacv else "test_sr_se": "se",
    })
    if aacv:
        summary = summary.rename(columns={
            "heldout_best_match_rate": "match",
            "heldout_score_excess_mean": "mean",
            "heldout_score_excess_se": "se",
        })
    else:
        summary = pd.concat([
            summary[["radius", f"{prefix}_oracle_match_rate",
                     f"{prefix}_regret_mean", f"{prefix}_regret_se"]].rename(columns={
                         f"{prefix}_oracle_match_rate": "match",
                         f"{prefix}_regret_mean": "mean",
                         f"{prefix}_regret_se": "se",
                     }).assign(selector=selector)
            for prefix, selector in [("clean", "clean"), ("sr", "SRCV")]
        ], ignore_index=True)
    selector = "aacv" if aacv else "SRCV"
    mismatch = mismatch.rename(columns={
        "heldout_best_match_rate" if aacv else "oracle_match_rate": "match",
        "heldout_score_excess_mean" if aacv else "regret_mean": "mean",
        "heldout_score_excess_mcse" if aacv else "regret_mcse": "mcse",
    })
    eval_radii = radii if aacv else config["mismatch"]["eval_radii"]
    _complete(candidate, {"radius": radii, "model": models}, ["mean", "se"], f"{method} candidates")
    _complete(summary, {"radius": radii, "selector": ["clean", selector]},
              ["mean", "se", "match"], f"{method} summary")
    grid = {"selector": ["clean", selector], "r_val": radii, "r_eval": eval_radii}
    _complete(mismatch, grid, ["mean", "mcse", "match"], f"{method} mismatch")
    _complete(frequencies, {**grid, "model": models}, ["frequency"], f"{method} frequencies")
    totals = frequencies.groupby(list(grid))["frequency"].sum().to_numpy()
    if not np.allclose(totals, 1, rtol=0, atol=1e-10):
        raise ValueError(f"{method}: selected-model frequencies do not sum to one")
    if (frequencies.groupby(["selector", "r_val", "model"])["frequency"].nunique() > 1).any():
        raise ValueError(f"{method}: selections unexpectedly depend on the evaluation radius")
    return Results(
        method, candidate, summary, mismatch, frequencies,
        tuple(config["figures"]["transition_window"]) if aacv else (0.17, 0.24),
        config["mismatch"]["representative_eval_radii"] if aacv else [0.20, 0.24, 0.32, 0.50],
        files,
    )


def _finish(ax: plt.Axes, xlabel: str, ylabel: str, title: str, label: str) -> None:
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, pad=7)
    ax.text(-0.14, 1.09, label, transform=ax.transAxes,
            fontsize=10, weight="bold", va="top")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.55)
    despine(ax)


def _curves(ax: plt.Axes, frame: pd.DataFrame, zoom_srcv: bool = False) -> None:
    for model in NONLINEAR + ["Best linear"]:
        cell = frame.loc[frame["model"] == model].sort_values("radius")
        if cell.empty:
            continue
        x, y, se = (cell[col].to_numpy(float) for col in ["radius", "mean", "se"])
        style = MODEL_STYLES[model]
        ax.plot(x, y, label=MODEL_LABELS.get(model, model), **style,
                linewidth=1.4 if zoom_srcv else 1.35, markersize=3 if zoom_srcv else 2.5)
        ax.fill_between(x, y-se, y+se, color=style["color"],
                        alpha=0.13 if zoom_srcv else 0.10, linewidth=0)


def draw_heldout(ax: plt.Axes, data: Results, panel: str) -> None:
    aacv = data.method == "AACV"
    xlabel = "Robust radius $r$"
    ylabel = "Mean held-out AACV score" if aacv else "Mean held-out SRCV loss"
    if panel in ("A", "B"):
        candidate = data.candidate
        nonlinear = candidate.loc[candidate["model"].isin(NONLINEAR)]
        if panel == "A":
            linear = candidate.loc[candidate["model"].isin(LINEAR)]
            best = linear.loc[linear.groupby("radius")["mean"].idxmin()].assign(model="Best linear")
            _curves(ax, pd.concat([nonlinear, best], ignore_index=True))
            ax.legend(frameon=False, ncol=2, loc="upper left")
            if aacv:
                ax.set_xlim(candidate["radius"].min(), candidate["radius"].max())
            _finish(ax, xlabel, ylabel, "Full radius range", panel)
        elif not aacv:
            _curves(ax, nonlinear.loc[nonlinear["radius"].between(*data.transition)], True)
            ax.legend(frameon=False, loc="upper left")
            _finish(ax, xlabel, ylabel, "Transition region", panel)
        else:
            ax.set_axis_off()
            for name, limits, bounds in [
                ("Early", (0.0, 0.10), (0.00, 0.08, 0.43, 0.78)),
                ("Late", data.transition, (0.59, 0.08, 0.41, 0.78)),
            ]:
                child = ax.inset_axes(bounds)
                _curves(child, nonlinear.loc[nonlinear["radius"].between(*limits)])
                ticks = [limits[0], sum(limits)/2, limits[1]]
                child.set_xlim(*limits)
                child.set_xticks(ticks, [format_radius(t) for t in ticks])
                child.set_title(f"{name}: {format_radius(limits[0])}-{format_radius(limits[1])}", pad=3)
                child.grid(axis="y", color="#E5E5E5", linewidth=0.55)
                despine(child)
            ax.text(0.5, -0.10, xlabel, transform=ax.transAxes, ha="center", va="top")
            ax.text(-0.16, 0.47, ylabel, transform=ax.transAxes, rotation=90,
                    ha="center", va="center")
            ax.text(0.5, 1.04, "Transition regions", transform=ax.transAxes,
                    ha="center", fontsize=mpl.rcParams["axes.titlesize"])
            ax.text(-0.14, 1.09, panel, transform=ax.transAxes, fontsize=10, weight="bold", va="top")
        return
    for selector, name, color, marker in [
        ("clean", "Clean CV", OKABE_ITO["orange"], "o"),
        ("aacv" if aacv else "SRCV", data.method, OKABE_ITO["green"], "s"),
    ]:
        cell = data.summary.loc[data.summary["selector"] == selector].sort_values("radius")
        x = cell["radius"].to_numpy(float)
        y = cell["match" if panel == "C" else "mean"].to_numpy(float)
        ax.plot(x, y, color=color, linewidth=1.45, marker=marker, markersize=2.8, label=name)
        if panel == "D":
            se = cell["se"].to_numpy(float)
            lower = np.maximum(y-se, 0) if aacv else y-se
            ax.fill_between(x, lower, y+se, color=color, alpha=0.13, linewidth=0)
    if panel == "C":
        ax.set_ylim(-0.03, 1.04)
        _finish(ax, xlabel, "Held-out-best match rate", "Agreement with held-out best", panel)
    else:
        _finish(ax, xlabel, "Mean held-out score excess" if aacv else "Mean robust regret",
                f"{data.method}: matched-radius performance", panel)
    ax.legend(frameon=False, loc="lower left" if panel == "C" else "upper left")


def draw_mismatch(ax: plt.Axes, data: Results, panel: str) -> None:
    aacv = data.method == "AACV"
    selector = "aacv" if aacv else "SRCV"
    robust = data.mismatch.loc[data.mismatch["selector"] == selector]
    clean = data.mismatch.loc[data.mismatch["selector"] == "clean"]
    r_val = sorted(robust["r_val"].unique())
    r_eval = sorted(robust["r_eval"].unique())
    xlabel = r"Selection radius $r_{\mathrm{val}}$"
    if panel in ("A", "B"):
        quantity = "mean" if panel == "A" else "match"
        matrix = robust.pivot(index="r_eval", columns="r_val", values=quantity).loc[r_eval, r_val]
        kwargs = {} if panel == "A" else {"vmin": 0, "vmax": 1}
        im = ax.imshow(matrix.to_numpy(float), aspect="auto", origin="lower",
                       cmap="viridis" if panel == "A" else "magma", **kwargs)
        for axis, values in [(ax.xaxis, r_val), (ax.yaxis, r_eval)]:
            ticks = (np.unique(np.rint(np.linspace(0, len(values)-1, 11)).astype(int))
                     if aacv and len(values) > 11 else np.arange(len(values)))
            axis.set_ticks(ticks)
            axis.set_ticklabels([format_radius(values[i]) for i in ticks])
        plt.setp(ax.get_xticklabels(), rotation=65, ha="right")
        ax.set(xlabel=xlabel, ylabel=r"Evaluation radius $r_{\mathrm{eval}}$")
        title = ("score excess" if aacv else "regret") if panel == "A" else "held-out-best match rate"
        ax.set_title(f"{data.method}: {title}", pad=7)
        ax.text(-0.17, 1.09, panel, transform=ax.transAxes, fontsize=10, weight="bold", va="top")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        return
    if panel == "C":
        palette = [OKABE_ITO[c] for c in ["blue", "orange", "green", "vermillion"]]
        for i, requested in enumerate(data.slices):
            cell = robust.loc[np.isclose(robust["r_eval"], requested)].sort_values("r_val")
            if cell.empty:
                continue
            color = palette[i % len(palette)]
            x, y = (cell[col].to_numpy(float) for col in ["r_val", "mean"])
            ax.plot(x, y, marker="o", markersize=2.8, linewidth=1.15, color=color,
                    label=rf"{data.method}, $r_{{eval}}={format_radius(requested)}$")
            if not aacv:
                mcse = cell["mcse"].to_numpy(float)
                ax.fill_between(x, np.maximum(y-1.96*mcse, 0), y+1.96*mcse,
                                color=color, alpha=0.08, linewidth=0)
            level = float(clean.loc[np.isclose(clean["r_eval"], requested)].sort_values("r_val")["mean"].iloc[0])
            if aacv or level > 1e-10:
                ax.axhline(level, color=color, linestyle=":", linewidth=0.8)
        _finish(ax, xlabel, "Mean held-out score excess" if aacv else "Mean regret",
                f"{data.method}: representative slices", panel)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.95,
                  loc="upper right", handlelength=1.5)
    else:
        freq = data.frequencies
        freq = freq.loc[(freq["selector"] == selector) & np.isclose(freq["r_eval"], r_eval[0])]
        for model in ["KNN", "RandomForest", "HistGB"] + (LINEAR if aacv else []):
            cell = freq.loc[freq["model"] == model].sort_values("r_val")
            if cell.empty or cell["frequency"].max() <= 0:
                continue
            style = MODEL_STYLES[model] if aacv else {
                "color": MODEL_STYLES[model]["color"], "marker": "o", "linestyle": "-"}
            ax.plot(cell["r_val"], cell["frequency"], **style, linewidth=1.25,
                    markersize=2.6 if aacv else 2.8, label=MODEL_LABELS.get(model, model))
        ax.set_ylim(-0.03, 1.03)
        _finish(ax, xlabel, "Frequency", f"{data.method}: selected-model frequency", panel)
        ax.legend(frameon=False, loc="best" if aacv else "center left", handlelength=1.5)
        if not aacv:
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    if not aacv:
        ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        ax.set_xticks(ticks, [format_radius(t) for t in ticks])
        ax.set_xlim(-0.01, 0.51)
    elif panel == "C":
        ax.set_xticks(np.linspace(min(r_val), max(r_val), 6))
        margin = max((max(r_val)-min(r_val))*0.02, 0.01)
        ax.set_xlim(min(r_val)-margin, max(r_val)+margin)


def make_figures(srcv: Results, aacv: Results):
    """Yield composites and six standalones from the same panel functions."""
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 7.7))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.07, top=0.92, wspace=0.48, hspace=0.62)
    for col, data in enumerate([srcv, aacv]):
        for row, panel in enumerate("ABC"):
            draw_heldout(axes[row, col], data, panel)
        x = sum(axes[0, col].get_position().intervalx)/2
        fig.text(x, 0.975, data.method, ha="center", weight="bold", fontsize=11)
    yield STEMS["heldout"], fig
    for kind, panels in [("mismatch", "BD"), ("mismatch_excess", "AC")]:
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.9))
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.93, wspace=0.50, hspace=0.65)
        for row, data in enumerate([srcv, aacv]):
            for col, panel in enumerate(panels):
                draw_mismatch(axes[row, col], data, panel)
        yield STEMS[kind], fig
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.87, wspace=0.48)
    for ax, data in zip(axes, [srcv, aacv]):
        draw_heldout(ax, data, "D")
    yield STEMS["heldout_excess"], fig
    for stem, (method, family, panel) in SINGLE_PANELS.items():
        fig, ax = plt.subplots(figsize=(3.9, 3.0))
        fig.subplots_adjust(left=0.20, right=0.96, bottom=0.23, top=0.86)
        draw = draw_heldout if family == "heldout" else draw_mismatch
        draw(ax, srcv if method == "SRCV" else aacv, panel)
        yield stem, fig


def generate(srcv_run_dir: Path, aacv_run_dir: Path, output_dir: Path) -> dict:
    configs = [ROOT / "configs" / name for name in ["ccpp_full.json", "ccpp_aacv_full.json"]]
    config_data = [json.loads(path.read_text(encoding="utf-8")) for path in configs]
    srcv = load_results(srcv_run_dir, "SRCV", config_data[0])
    aacv = load_results(aacv_run_dir, "AACV", config_data[1])
    input_files = srcv.files + aacv.files + configs
    hashes = {str(path.resolve()): sha256(path) for path in input_files}
    output_dir = output_dir.resolve()
    if output_dir in [p.parent.resolve() for p in srcv.files + aacv.files]:
        raise ValueError("Use a separate figure directory, not an input CSV directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    with mpl.rc_context():
        apply_style()
        # Bundle-supplied font avoids platform-specific Arial/Helvetica fallback.
        mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        for stem, fig in make_figures(srcv, aacv):
            try:
                for ext in ("pdf", "png"):
                    path = output_dir / f"{stem}.{ext}"
                    metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else None
                    fig.savefig(path, dpi=400, metadata=metadata)
                    outputs[path.name] = sha256(path)
            finally:
                plt.close(fig)
    if hashes != {str(path.resolve()): sha256(path) for path in input_files}:
        raise RuntimeError("An input file changed during plotting; rerun from a stable snapshot")
    manifest = {
        "purpose": "Official manuscript and supplementary CCPP figures; no refitting or rescoring",
        "aacv_variant": PRIMARY,
        "input_sha256": hashes,
        "script_sha256": sha256(Path(__file__)),
        "shared_style_sha256": sha256(ROOT / "ccpp_repro" / "plotting.py"),
        "versions": {"python": platform.python_version(), "matplotlib": mpl.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "font": "DejaVu Sans", "png_dpi": 400,
        "layouts": {
            STEMS["heldout"]: "3x2: columns SRCV/AACV; rows A/B/C; AACV B includes Early and Late",
            STEMS["mismatch"]: "2x2: rows SRCV/AACV; columns B/D",
            STEMS["heldout_excess"]: "1x2: SRCV D / AACV D",
            STEMS["mismatch_excess"]: "2x2: rows SRCV/AACV; columns A/C",
        },
        "standalone_panels": SINGLE_PANELS,
        "output_sha256": outputs,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srcv-run-dir", type=Path, default=ROOT / "outputs" / "srcv_full")
    parser.add_argument("--aacv-run-dir", type=Path, default=ROOT / "outputs" / "aacv_full")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "manuscript_figures")
    args = parser.parse_args()
    try:
        manifest = generate(args.srcv_run_dir, args.aacv_run_dir, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot generate manuscript figures: {exc}\n")
    print(f"Generated {len(manifest['output_sha256'])} PDF/PNG files in {args.output_dir.resolve()}")
    print("Input CSVs unchanged; provenance recorded in figure_manifest.json")


if __name__ == "__main__":
    main()
