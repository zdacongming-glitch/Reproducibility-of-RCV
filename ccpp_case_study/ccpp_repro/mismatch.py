from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


SELECTORS = {"clean": "clean_final_choice", "SRCV": "sr_final_choice"}


def radius_key(value: float) -> str:
    return f"{float(value):.12g}"


def mcse(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def binary_mcse(rate: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    return float(math.sqrt(max(rate * (1.0 - rate), 0.0) / count))


def entropy(values: pd.Series) -> float:
    probs = values.value_counts(normalize=True).to_numpy(dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return float("nan")
    return float(-(probs * np.log(probs)).sum())


def modal_value(values: pd.Series) -> str:
    counts = values.value_counts()
    if counts.empty:
        return ""
    max_count = counts.max()
    return str(sorted(counts[counts == max_count].index)[0])


def resolve_requested_radii(available: list[float], requested: list[float], tol: float = 1e-10) -> list[float]:
    resolved: list[float] = []
    missing: list[float] = []
    for request in requested:
        matches = [value for value in available if abs(value - request) <= tol]
        if not matches:
            missing.append(request)
        elif matches[0] not in resolved:
            resolved.append(matches[0])
    if missing:
        raise ValueError(f"Requested evaluation radii are absent from the run grid: {missing}")
    return resolved


def load_outer_summary(run_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    path = run_dir / "outer_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    outer = pd.read_csv(path)
    outer["radius"] = outer["radius"].astype(float)
    outer["outer"] = outer["outer"].astype(int)
    models = [col.removeprefix("test_sr_") for col in outer.columns if col.startswith("test_sr_")]
    if not models:
        raise ValueError("outer_summary.csv has no test_sr_<model> columns")
    return outer, models


def validate_outer_grid(outer: pd.DataFrame) -> dict[str, object]:
    radii = sorted(float(value) for value in outer["radius"].unique())
    reference = set(outer.loc[np.isclose(outer["radius"], radii[0]), "outer"])
    for radius in radii:
        cur = set(outer.loc[np.isclose(outer["radius"], radius), "outer"])
        if cur != reference:
            raise ValueError(f"Radius {radius:g} does not share the same outer replications")
    duplicates = outer.groupby(["radius", "outer"], as_index=False).size().query("size != 1")
    if not duplicates.empty:
        raise ValueError(f"Duplicate or missing radius/outer cells: {duplicates.to_dict('records')}")
    return {"radii": radii, "outer_repeats": len(reference), "rows": int(outer.shape[0])}


def check_clean_cv_stability(run_dir: Path, tolerance: float) -> dict[str, object]:
    path = run_dir / "inner_validation_scores.csv"
    if not path.exists():
        return {"available": False, "passed": None, "max_clean_cv_span": None}
    df = pd.read_csv(path)
    spans = (
        df.groupby(["outer", "inner", "model"])["clean_cv"]
        .agg(lambda x: float(np.nanmax(x) - np.nanmin(x)))
        .reset_index(name="span")
    )
    max_span = float(spans["span"].max())
    return {"available": True, "passed": max_span <= tolerance, "max_clean_cv_span": max_span}


def build_mismatch_outer(
    outer: pd.DataFrame,
    models: list[str],
    r_val_grid: list[float],
    r_eval_grid: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    indexed = {(radius_key(row.radius), int(row.outer)): row for row in outer.itertuples(index=False)}
    outer_values = sorted(outer["outer"].unique())
    for selector, choice_col in SELECTORS.items():
        for r_val in r_val_grid:
            for r_eval in r_eval_grid:
                for outer_id in outer_values:
                    val_row = indexed[(radius_key(r_val), int(outer_id))]
                    eval_row = indexed[(radius_key(r_eval), int(outer_id))]
                    selected = str(getattr(val_row, choice_col))
                    selected_score = float(getattr(eval_row, f"test_sr_{selected}"))
                    oracle_score = float(eval_row.oracle_test_sr)
                    oracle_model = str(eval_row.oracle_choice)
                    rows.append(
                        {
                            "selector": selector,
                            "r_val": r_val,
                            "r_eval": r_eval,
                            "outer": int(outer_id),
                            "selected_model": selected,
                            "oracle_model_at_eval": oracle_model,
                            "selected_test_sr_at_eval": selected_score,
                            "oracle_test_sr_at_eval": oracle_score,
                            "regret": selected_score - oracle_score,
                            "oracle_match": int(selected == oracle_model),
                        }
                    )
    out = pd.DataFrame(rows)
    out["selected_model"] = pd.Categorical(out["selected_model"], categories=models)
    out["oracle_model_at_eval"] = pd.Categorical(out["oracle_model_at_eval"], categories=models)
    return out


def summarize_mismatch(mismatch_outer: pd.DataFrame) -> pd.DataFrame:
    grouped = mismatch_outer.groupby(["selector", "r_val", "r_eval"], observed=False)
    summary = grouped.agg(
        outer_repeats=("outer", "nunique"),
        regret_mean=("regret", "mean"),
        regret_sd=("regret", lambda x: float(x.std(ddof=1))),
        regret_mcse=("regret", mcse),
        oracle_match_rate=("oracle_match", "mean"),
        selected_test_sr_mean=("selected_test_sr_at_eval", "mean"),
        oracle_test_sr_mean=("oracle_test_sr_at_eval", "mean"),
        selected_model_mode=("selected_model", modal_value),
        selected_model_entropy=("selected_model", entropy),
    ).reset_index()
    summary["oracle_match_mcse"] = [
        binary_mcse(rate, count)
        for rate, count in zip(summary["oracle_match_rate"], summary["outer_repeats"], strict=True)
    ]
    summary["regret_ci95_halfwidth"] = 1.96 * summary["regret_mcse"]
    return summary


def selection_frequencies(mismatch_outer: pd.DataFrame) -> pd.DataFrame:
    counts = (
        mismatch_outer.groupby(["selector", "r_val", "r_eval", "selected_model"], observed=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"selected_model": "model"})
    )
    totals = mismatch_outer.groupby(["selector", "r_val", "r_eval"], observed=False).size()
    counts["frequency"] = [
        row["count"] / totals.loc[(row["selector"], row["r_val"], row["r_eval"])]
        for _, row in counts.iterrows()
    ]
    return counts


def adjacent_stability(mismatch_outer: pd.DataFrame, r_val_grid: list[float], r_eval_grid: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for selector in SELECTORS:
        sub = mismatch_outer[mismatch_outer["selector"] == selector]
        reference = (
            sub[np.isclose(sub["r_eval"], r_eval_grid[0])]
            .pivot(index="outer", columns="r_val", values="selected_model")
            .sort_index(axis=1)
        )
        for left, right in zip(r_val_grid[:-1], r_val_grid[1:], strict=True):
            same = (reference[left] == reference[right]).astype(float)
            for r_eval in r_eval_grid:
                rows.append(
                    {
                        "selector": selector,
                        "r_eval": r_eval,
                        "r_val_left": left,
                        "r_val_right": right,
                        "outer_repeats": int(same.shape[0]),
                        "same_selection_rate": float(same.mean()),
                        "same_selection_mcse": binary_mcse(float(same.mean()), int(same.shape[0])),
                    }
                )
    return pd.DataFrame(rows)


def low_regret_table(summary: pd.DataFrame, selector: str = "SRCV") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for r_eval, cur in summary[summary["selector"] == selector].groupby("r_eval", sort=True):
        cur = cur.sort_values("r_val")
        best = cur.loc[cur["regret_mean"].idxmin()]
        threshold = float(best["regret_mean"] + best["regret_mcse"])
        low = cur[cur["regret_mean"] <= threshold]
        rows.append(
            {
                "selector": selector,
                "r_eval": float(r_eval),
                "best_r_val": float(best["r_val"]),
                "best_regret_mean": float(best["regret_mean"]),
                "best_regret_mcse": float(best["regret_mcse"]),
                "one_mcse_threshold": threshold,
                "low_regret_r_val_min": float(low["r_val"].min()),
                "low_regret_r_val_max": float(low["r_val"].max()),
                "low_regret_r_val_values": ",".join(f"{float(value):g}" for value in low["r_val"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_validation_scores(run_dir: Path, output_path: Path) -> None:
    path = run_dir / "inner_validation_scores.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    out = df.groupby(["radius", "model"], as_index=False).agg(
        cells=("clean_cv", "size"),
        clean_cv_mean=("clean_cv", "mean"),
        clean_cv_sd=("clean_cv", lambda x: float(x.std(ddof=1))),
        clean_cv_mcse=("clean_cv", mcse),
        sr_cv_mean=("sr_cv", "mean"),
        sr_cv_sd=("sr_cv", lambda x: float(x.std(ddof=1))),
        sr_cv_mcse=("sr_cv", mcse),
    )
    out.to_csv(output_path, index=False)


def run_mismatch(config: dict, run_dir: Path) -> Path:
    mismatch_config = config["mismatch"]
    output_dir = run_dir / mismatch_config["output_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    outer, _models = load_outer_summary(run_dir)
    grid = validate_outer_grid(outer)
    clean_cv_check = check_clean_cv_stability(run_dir, float(mismatch_config["clean_cv_tolerance"]))
    if clean_cv_check["available"] and not clean_cv_check["passed"]:
        raise ValueError(f"clean_cv stability check failed: {clean_cv_check}")
    r_val_grid = [float(value) for value in grid["radii"]]
    r_eval_grid = resolve_requested_radii(r_val_grid, [float(value) for value in mismatch_config["eval_radii"]])
    mismatch_outer = build_mismatch_outer(outer, _models, r_val_grid, r_eval_grid)
    summary = summarize_mismatch(mismatch_outer)
    mismatch_outer.to_csv(output_dir / "radius_mismatch_outer.csv", index=False)
    summary.to_csv(output_dir / "radius_mismatch_summary.csv", index=False)
    selection_frequencies(mismatch_outer).to_csv(output_dir / "radius_mismatch_selection_frequencies.csv", index=False)
    adjacent_stability(mismatch_outer, r_val_grid, r_eval_grid).to_csv(output_dir / "radius_mismatch_stability.csv", index=False)
    low_regret_table(summary).to_csv(output_dir / "radius_mismatch_low_regret_intervals.csv", index=False)
    aggregate_validation_scores(run_dir, output_dir / "radius_mismatch_validation_scores.csv")
    return output_dir
