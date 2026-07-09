from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


def _match_rows(df: pd.DataFrame, match: dict[str, Any], tol: float) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, value in match.items():
        if isinstance(value, (int, float)):
            mask &= np.isclose(df[col].astype(float), float(value), atol=tol, rtol=0)
        else:
            mask &= df[col].astype(str).eq(str(value))
    return df[mask]


def verify_run(config: dict, run_dir: Path) -> list[str]:
    errors: list[str] = []
    verification = config["verification"]
    for rel_path, expected_rows in verification["required_rows"].items():
        path = run_dir / rel_path
        if not path.exists():
            errors.append(f"missing CSV: {rel_path}")
            continue
        rows = len(pd.read_csv(path))
        if rows != int(expected_rows):
            errors.append(f"{rel_path}: expected {expected_rows} rows, found {rows}")
    tol = float(verification["float_tolerance"])
    for signature in verification.get("numeric_signatures", []):
        path = run_dir / signature["file"]
        if not path.exists():
            errors.append(f"missing signature file: {signature['file']}")
            continue
        df = pd.read_csv(path)
        rows = _match_rows(df, signature["match"], tol)
        if len(rows) != 1:
            errors.append(f"{signature['file']}: match {signature['match']} selected {len(rows)} rows")
            continue
        row = rows.iloc[0]
        for col, expected in signature["values"].items():
            actual = float(row[col])
            if not np.isclose(actual, float(expected), atol=tol, rtol=0):
                errors.append(f"{signature['file']} {signature['match']} {col}: expected {expected}, found {actual}")
    return errors


def verify_figures(config: dict, figure_dir: Path, reference_figure_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    stems = [config["figures"]["case_study_stem"], config["figures"]["mismatch_stem"]]
    for stem in stems:
        for ext in config["figures"].get("formats", ["pdf", "png"]):
            path = figure_dir / f"{stem}.{ext}"
            if not path.exists():
                errors.append(f"missing figure: {path}")
            elif path.stat().st_size <= 1024:
                errors.append(f"figure is unexpectedly small: {path}")
    if reference_figure_dir is not None:
        tol = float(config["verification"].get("image_mean_abs_tolerance", 0.015))
        for stem in stems:
            actual = figure_dir / f"{stem}.png"
            reference = reference_figure_dir / f"{stem}.png"
            if actual.exists() and reference.exists():
                actual_img = mpimg.imread(actual).astype(float)
                ref_img = mpimg.imread(reference).astype(float)
                if actual_img.shape != ref_img.shape:
                    errors.append(f"{stem}.png shape mismatch: {actual_img.shape} vs {ref_img.shape}")
                    continue
                mad = float(np.mean(np.abs(actual_img - ref_img)))
                if mad > tol:
                    errors.append(f"{stem}.png mean absolute difference {mad:.6g} exceeds {tol}")
    return errors


def verify_all(config: dict, run_dir: Path, figure_dir: Path, reference_figure_dir: Path | None = None) -> None:
    errors = verify_run(config, run_dir) + verify_figures(config, figure_dir, reference_figure_dir)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"Verification failed:\n{joined}")
    print("Verification passed.")
