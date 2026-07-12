from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aacv_working_error import PRIMARY_ANALYSIS_VARIANT
from .mismatch import binary_mcse, entropy, mcse, modal_value


def _load_primary_tables(
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = pd.read_csv(run_dir / "outer_aacv_summary.csv")
    candidates = pd.read_csv(run_dir / "outer_candidate_test_scores.csv")
    validation = pd.read_csv(run_dir / "inner_aacv_scores.csv")
    frames = []
    for frame in (outer, candidates, validation):
        primary = frame[
            frame["analysis_variant"].astype(str) == PRIMARY_ANALYSIS_VARIANT
        ].copy()
        primary["radius"] = primary["radius"].astype(float)
        primary["outer"] = primary["outer"].astype(int)
        frames.append(primary)
    return frames[0], frames[1], frames[2]


def _validate_outer_grid(outer: pd.DataFrame) -> tuple[list[float], list[int]]:
    radii = sorted(float(value) for value in outer["radius"].unique())
    outers = sorted(int(value) for value in outer["outer"].unique())
    expected = len(radii) * len(outers) * 2
    if len(outer) != expected:
        raise ValueError(
            f"Expected {expected} primary matched rows, found {len(outer)}"
        )
    duplicates = outer.groupby(
        ["analysis_variant", "radius", "outer", "selector"], as_index=False
    ).size()
    if not duplicates["size"].eq(1).all():
        raise ValueError("Duplicate or missing primary AACV matched cells")
    return radii, outers


def build_radius_mismatch(
    outer: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_order: tuple[str, ...],
) -> pd.DataFrame:
    radii, outers = _validate_outer_grid(outer)
    choice_index = {
        (float(row.radius), int(row.outer), str(row.selector)): str(row.selected_model)
        for row in outer.itertuples(index=False)
    }
    score_index = {
        (float(row.radius), int(row.outer), str(row.model)): float(row.test_aacv_score)
        for row in candidates.itertuples(index=False)
    }
    model_rank = {model: index for index, model in enumerate(candidate_order)}
    best_index: dict[tuple[float, int], tuple[str, float]] = {}
    for (radius, outer_id), cell in candidates.groupby(["radius", "outer"], sort=False):
        ordered = cell.assign(_model_rank=cell["model"].map(model_rank)).sort_values(
            ["test_aacv_score", "_model_rank"], kind="stable"
        )
        best = ordered.iloc[0]
        best_index[(float(radius), int(outer_id))] = (
            str(best["model"]),
            float(best["test_aacv_score"]),
        )

    rows: list[dict[str, object]] = []
    for selector in ("clean", "aacv"):
        for r_val in radii:
            for r_eval in radii:
                for outer_id in outers:
                    selected = choice_index[(r_val, outer_id, selector)]
                    selected_score = score_index[(r_eval, outer_id, selected)]
                    best_model, best_score = best_index[(r_eval, outer_id)]
                    rows.append(
                        {
                            "analysis_variant": PRIMARY_ANALYSIS_VARIANT,
                            "selector": selector,
                            "r_val": r_val,
                            "r_eval": r_eval,
                            "outer": outer_id,
                            "selected_model": selected,
                            "heldout_best_choice_at_eval": best_model,
                            "selected_test_aacv_at_eval": selected_score,
                            "heldout_best_test_aacv_at_eval": best_score,
                            "heldout_score_excess": selected_score - best_score,
                            "heldout_best_match": int(selected == best_model),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_radius_mismatch(mismatch: pd.DataFrame) -> pd.DataFrame:
    grouped = mismatch.groupby(
        ["analysis_variant", "selector", "r_val", "r_eval"], sort=True
    )
    summary = grouped.agg(
        outer_repeats=("outer", "nunique"),
        heldout_score_excess_mean=("heldout_score_excess", "mean"),
        heldout_score_excess_sd=(
            "heldout_score_excess",
            lambda values: float(values.std(ddof=1)),
        ),
        heldout_score_excess_mcse=("heldout_score_excess", mcse),
        heldout_best_match_rate=("heldout_best_match", "mean"),
        selected_test_aacv_mean=("selected_test_aacv_at_eval", "mean"),
        heldout_best_test_aacv_mean=("heldout_best_test_aacv_at_eval", "mean"),
        selected_model_mode=("selected_model", modal_value),
        selected_model_entropy=("selected_model", entropy),
    ).reset_index()
    summary["heldout_best_match_mcse"] = [
        binary_mcse(rate, count)
        for rate, count in zip(
            summary["heldout_best_match_rate"],
            summary["outer_repeats"],
            strict=True,
        )
    ]
    summary["excess_ci95_halfwidth"] = 1.96 * summary["heldout_score_excess_mcse"]
    return summary


def mismatch_selection_frequencies(
    mismatch: pd.DataFrame,
    candidate_order: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["analysis_variant", "selector", "r_val", "r_eval"]
    for keys, cell in mismatch.groupby(group_columns, sort=True):
        counts = cell["selected_model"].value_counts()
        total = len(cell)
        for model in candidate_order:
            count = int(counts.get(model, 0))
            rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "model": model,
                    "count": count,
                    "frequency": count / total,
                }
            )
    return pd.DataFrame(rows)


def mismatch_stability(mismatch: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    r_val_grid = sorted(float(value) for value in mismatch["r_val"].unique())
    r_eval_grid = sorted(float(value) for value in mismatch["r_eval"].unique())
    for selector in ("clean", "aacv"):
        cell = mismatch[
            (mismatch["selector"] == selector)
            & np.isclose(mismatch["r_eval"], r_eval_grid[0])
        ]
        choices = cell.pivot(index="outer", columns="r_val", values="selected_model")
        for left, right in zip(r_val_grid[:-1], r_val_grid[1:], strict=True):
            same = (choices[left] == choices[right]).astype(float)
            for r_eval in r_eval_grid:
                rate = float(same.mean())
                rows.append(
                    {
                        "analysis_variant": PRIMARY_ANALYSIS_VARIANT,
                        "selector": selector,
                        "r_eval": r_eval,
                        "r_val_left": left,
                        "r_val_right": right,
                        "outer_repeats": len(same),
                        "same_selection_rate": rate,
                        "same_selection_mcse": binary_mcse(rate, len(same)),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_validation_scores(validation: pd.DataFrame) -> pd.DataFrame:
    return (
        validation.groupby(["analysis_variant", "radius", "model"], as_index=False)
        .agg(
            cells=("aacv_score", "size"),
            clean_cv_mean=("clean_cv", "mean"),
            clean_cv_sd=("clean_cv", lambda values: float(values.std(ddof=1))),
            clean_cv_mcse=("clean_cv", mcse),
            aacv_score_mean=("aacv_score", "mean"),
            aacv_score_sd=("aacv_score", lambda values: float(values.std(ddof=1))),
            aacv_score_mcse=("aacv_score", mcse),
        )
        .sort_values(["radius", "model"])
        .reset_index(drop=True)
    )


def low_excess_intervals(
    summary: pd.DataFrame,
    representative_eval_radii: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    aacv = summary[summary["selector"] == "aacv"]
    for requested in representative_eval_radii:
        cell = aacv[np.isclose(aacv["r_eval"], requested)].sort_values("r_val")
        if cell.empty:
            continue
        best = cell.loc[cell["heldout_score_excess_mean"].idxmin()]
        threshold = float(
            best["heldout_score_excess_mean"] + best["heldout_score_excess_mcse"]
        )
        low = cell[cell["heldout_score_excess_mean"] <= threshold]
        rows.append(
            {
                "analysis_variant": PRIMARY_ANALYSIS_VARIANT,
                "selector": "aacv",
                "r_eval": float(requested),
                "best_r_val": float(best["r_val"]),
                "best_excess_mean": float(best["heldout_score_excess_mean"]),
                "best_excess_mcse": float(best["heldout_score_excess_mcse"]),
                "one_mcse_threshold": threshold,
                "low_excess_r_val_min": float(low["r_val"].min()),
                "low_excess_r_val_max": float(low["r_val"].max()),
                "low_excess_r_val_values": ",".join(
                    f"{float(value):g}" for value in low["r_val"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _check_clean_stability(validation: pd.DataFrame, tolerance: float) -> None:
    spans = validation.groupby(["outer", "inner", "model"])["clean_cv"].agg(
        lambda values: float(np.max(values) - np.min(values))
    )
    maximum = float(spans.max())
    if maximum > tolerance:
        raise ValueError(f"Clean CV is not radius invariant: maximum span {maximum}")


def _check_matched_diagonal(
    mismatch: pd.DataFrame,
    outer: pd.DataFrame,
    tolerance: float,
) -> None:
    diagonal = mismatch[np.isclose(mismatch["r_val"], mismatch["r_eval"])].copy()
    diagonal = diagonal.rename(columns={"r_val": "radius"})
    keys = ["analysis_variant", "selector", "radius", "outer"]
    merged = diagonal.merge(
        outer,
        on=keys,
        how="outer",
        suffixes=("_mismatch", "_matched"),
        validate="one_to_one",
    )
    if merged.isna().any().any():
        raise ValueError("Matched diagonal and primary outer summary keys differ")
    if not np.allclose(
        merged["heldout_score_excess_mismatch"],
        merged["heldout_score_excess_matched"],
        atol=tolerance,
        rtol=0,
    ):
        raise ValueError("Matched diagonal excess does not equal primary summary")
    if not np.array_equal(
        merged["heldout_best_match_mismatch"].to_numpy(int),
        merged["heldout_best_match_matched"].to_numpy(int),
    ):
        raise ValueError("Matched diagonal best-match flags differ")


def run_aacv_mismatch(config: dict, run_dir: Path) -> Path:
    output_dir = run_dir / config["mismatch"]["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    outer, candidates, validation = _load_primary_tables(run_dir)
    tolerance = float(config["mismatch"]["clean_cv_tolerance"])
    _check_clean_stability(validation, tolerance)
    candidate_order = tuple(config["models"]["candidate_order"])
    mismatch = build_radius_mismatch(outer, candidates, candidate_order)
    _check_matched_diagonal(mismatch, outer, tolerance)
    summary = summarize_radius_mismatch(mismatch)
    outputs = {
        "radius_mismatch_outer.csv": mismatch,
        "radius_mismatch_summary.csv": summary,
        "radius_mismatch_selection_frequencies.csv": mismatch_selection_frequencies(
            mismatch, candidate_order
        ),
        "radius_mismatch_stability.csv": mismatch_stability(mismatch),
        "radius_mismatch_validation_scores.csv": aggregate_validation_scores(
            validation
        ),
        "radius_mismatch_low_excess_intervals.csv": low_excess_intervals(
            summary,
            [float(value) for value in config["mismatch"]["representative_eval_radii"]],
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    return output_dir
