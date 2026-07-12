from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from .aacv_experiment import (
    AACV_SCIENTIFIC_SCHEMA_VERSION,
    ATTACK_ALGORITHM,
    _source_hashes,
    aacv_config_from_dict,
    experiment_fingerprint,
)
from .aacv_reference import (
    canonical_json_fingerprint,
    load_reference_spec,
    validate_reference_spec,
)
from .aacv_working_error import (
    PRIMARY_ANALYSIS_VARIANT,
    PRIMARY_SCALE,
    required_analysis_variants,
)
from .experiment import FEATURE_COLUMNS, deterministic_seed


def _load_csv(run_dir: Path, relative: str, errors: list[str]) -> pd.DataFrame:
    path = run_dir / relative
    if not path.exists():
        errors.append(f"missing CSV: {relative}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        errors.append(f"cannot read {relative}: {exc}")
        return pd.DataFrame()


def _expect_rows(
    frame: pd.DataFrame,
    expected: int,
    label: str,
    errors: list[str],
) -> None:
    if len(frame) != expected:
        errors.append(f"{label}: expected {expected} rows, found {len(frame)}")


def _check_unique(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
    errors: list[str],
) -> None:
    if frame.empty:
        return
    missing = set(columns) - set(frame.columns)
    if missing:
        errors.append(f"{label}: missing key columns {sorted(missing)}")
        return
    duplicates = int(frame.duplicated(columns).sum())
    if duplicates:
        errors.append(f"{label}: {duplicates} duplicate keys for {columns}")


def _variant_metadata(config: dict[str, Any]) -> pd.DataFrame:
    variants = required_analysis_variants(tuple(config["models"]["candidate_order"]))
    return pd.DataFrame(
        [
            {
                "analysis_variant": variant.name,
                "analysis_role": variant.role,
                "working_reference_learner": variant.working_reference_learner,
                "scale_estimator": variant.scale_estimator,
                "hermite_degree": variant.hermite_degree,
                "envelope_multiplier": variant.envelope_multiplier,
            }
            for variant in variants
        ]
    )


def _check_variant_metadata(
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    label: str,
    errors: list[str],
) -> None:
    columns = list(expected.columns)
    if frame.empty:
        return
    missing = set(columns) - set(frame.columns)
    if missing:
        errors.append(f"{label}: missing variant columns {sorted(missing)}")
        return
    actual = frame[columns].drop_duplicates().sort_values("analysis_variant")
    target = expected.sort_values("analysis_variant")
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            target.reset_index(drop=True),
            check_dtype=False,
            atol=1e-12,
            rtol=0,
        )
    except AssertionError as exc:
        errors.append(f"{label}: analysis-variant metadata mismatch: {exc}")


def _check_output_terminology(run_dir: Path, errors: list[str]) -> None:
    forbidden = ("pseudo_truth", "pseudo-true", "pseudo-noise", "oracle", "regret")
    for path in run_dir.glob("*.csv"):
        header = path.open("r", encoding="utf-8").readline().lower()
        for term in forbidden:
            if term in header:
                errors.append(f"{path.name}: forbidden AACV output term {term!r}")
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        text = manifest.read_text(encoding="utf-8").lower()
        for term in forbidden:
            if term in text:
                errors.append(f"run_manifest.json: forbidden AACV term {term!r}")


def verify_aacv_run(config: dict[str, Any], run_dir: Path) -> list[str]:
    errors: list[str] = []
    experiment = aacv_config_from_dict(config, progress=False)
    reference_path, reference_spec = load_reference_spec(config)
    try:
        validate_reference_spec(reference_spec, aacv_config=config)
    except ValueError as exc:
        errors.append(f"invalid frozen reference spec: {exc}")

    files = {
        name: _load_csv(run_dir, f"{name}.csv", errors)
        for name in (
            "inner_aacv_scores",
            "inner_test_aacv_scores",
            "inner_aacv_selections",
            "outer_aacv_summary",
            "outer_candidate_test_scores",
            "working_reference_diagnostics",
            "scale_estimator_diagnostics",
            "residual_covariate_diagnostics",
            "residual_covariate_bins",
            "attack_diagnostics",
            "attack_path_calibration",
            "attack_path_calibration_summary",
            "aacv_summary_by_radius",
            "aacv_selection_frequencies",
            "aacv_candidate_test_scores",
            "working_error_selection_switches",
            "working_error_sensitivity_summary",
            "scale_estimator_summary",
            "reference_selection",
        )
    }
    radius_count = len(experiment.radii)
    outer_count = experiment.outer_repeats
    inner_count = experiment.inner_repeats
    model_count = len(experiment.candidate_order)
    variant_count = len(experiment.analysis_variants)
    reference_count = model_count
    scale_count = model_count + 2
    nonzero_radii = sum(not np.isclose(radius, 0.0) for radius in experiment.radii)
    calibration_rows = (
        outer_count
        * inner_count
        * len(experiment.path_calibration.radii)
        * 2
        * model_count
        * variant_count
        * len(experiment.path_calibration.path_counts)
        if experiment.path_calibration.enabled
        else 0
    )

    expected_rows = {
        "inner_aacv_scores": radius_count
        * outer_count
        * inner_count
        * model_count
        * variant_count,
        "inner_test_aacv_scores": radius_count
        * outer_count
        * inner_count
        * model_count
        * variant_count,
        "inner_aacv_selections": radius_count
        * outer_count
        * inner_count
        * variant_count
        * 2,
        "outer_aacv_summary": radius_count * outer_count * variant_count * 2,
        "outer_candidate_test_scores": radius_count
        * outer_count
        * model_count
        * variant_count,
        "working_reference_diagnostics": outer_count * inner_count * reference_count,
        "scale_estimator_diagnostics": outer_count * inner_count * scale_count,
        "residual_covariate_diagnostics": outer_count
        * inner_count
        * reference_count
        * len(FEATURE_COLUMNS),
        "residual_covariate_bins": outer_count
        * inner_count
        * reference_count
        * len(FEATURE_COLUMNS)
        * 10,
        "attack_diagnostics": (
            outer_count * inner_count * nonzero_radii * 3 * 2
            if experiment.audit_enabled
            else 0
        ),
        "attack_path_calibration": calibration_rows,
        "attack_path_calibration_summary": 1,
        "aacv_summary_by_radius": radius_count * variant_count * 2,
        "aacv_selection_frequencies": radius_count * variant_count * 2 * model_count,
        "aacv_candidate_test_scores": radius_count * variant_count * model_count,
        "working_error_selection_switches": radius_count
        * outer_count
        * variant_count
        * 2,
        "working_error_sensitivity_summary": radius_count * variant_count * 2,
        "scale_estimator_summary": scale_count,
        "reference_selection": model_count,
    }
    for name, expected in expected_rows.items():
        _expect_rows(files[name], expected, name, errors)

    variant_metadata = _variant_metadata(config)
    for name in (
        "inner_aacv_scores",
        "inner_test_aacv_scores",
        "inner_aacv_selections",
        "outer_aacv_summary",
        "outer_candidate_test_scores",
        "aacv_summary_by_radius",
        "aacv_selection_frequencies",
        "aacv_candidate_test_scores",
    ):
        _check_variant_metadata(files[name], variant_metadata, name, errors)

    _check_unique(
        files["inner_aacv_scores"],
        ["analysis_variant", "radius", "outer", "inner", "model"],
        "inner_aacv_scores",
        errors,
    )
    _check_unique(
        files["inner_test_aacv_scores"],
        ["analysis_variant", "radius", "outer", "inner", "model"],
        "inner_test_aacv_scores",
        errors,
    )
    _check_unique(
        files["inner_aacv_selections"],
        ["analysis_variant", "radius", "outer", "inner", "selector"],
        "inner_aacv_selections",
        errors,
    )
    _check_unique(
        files["outer_aacv_summary"],
        ["analysis_variant", "radius", "outer", "selector"],
        "outer_aacv_summary",
        errors,
    )
    _check_unique(
        files["attack_path_calibration"],
        [
            "outer",
            "inner",
            "stage",
            "radius",
            "model",
            "analysis_variant",
            "path_count",
        ],
        "attack_path_calibration",
        errors,
    )

    validation = files["inner_aacv_scores"]
    outer = files["outer_aacv_summary"]
    selections = files["inner_aacv_selections"]
    tolerance = float(config["verification"]["float_tolerance"])
    if not validation.empty:
        radius_zero = validation[np.isclose(validation["radius"], 0.0)]
        if not np.allclose(
            radius_zero["aacv_score"],
            radius_zero["clean_cv"],
            atol=tolerance,
            rtol=0,
        ):
            errors.append("AACV does not reduce to clean CV at radius zero")
        scale_spans = validation.groupby(
            ["analysis_variant", "radius", "outer", "inner"]
        )["sigma"].nunique()
        if not scale_spans.eq(1).all():
            errors.append(
                "candidate models do not share one scale within a split/variant"
            )
    if not outer.empty and (outer["heldout_score_excess"] < -tolerance).any():
        errors.append("outer summary contains negative held-out score excess")
    if not selections.empty:
        clean = selections[selections["selector"] == "clean"]
        spans = clean.groupby(["outer", "inner"])["selected_model"].nunique()
        if not spans.eq(1).all():
            errors.append("clean selections vary across radius or analysis variant")

    references = files["working_reference_diagnostics"]
    if not references.empty:
        expected_learners = set(experiment.candidate_order)
        actual_learners = set(references["working_reference_learner"].astype(str))
        if actual_learners != expected_learners:
            errors.append("working-reference diagnostics do not cover all six learners")
        if (
            references[
                [
                    "sigma2_raw",
                    "sigma2",
                    "sigma",
                    "residual_mean",
                    "residual_mse",
                    "residual_skewness",
                    "residual_excess_kurtosis",
                    "jarque_bera_statistic",
                    "jarque_bera_pvalue",
                ]
            ]
            .isna()
            .any()
            .any()
        ):
            errors.append("working-reference diagnostics contain missing values")
        if references["fold_model_seeds"].astype(str).str.len().eq(0).any():
            errors.append("working-reference diagnostics omit fold model seeds")

    scales = files["scale_estimator_diagnostics"]
    expected_scales = {
        variant.scale_estimator for variant in experiment.analysis_variants
    }
    if not scales.empty:
        if set(scales["scale_estimator"].astype(str)) != expected_scales:
            errors.append("scale diagnostics do not cover all predeclared estimators")
        primary_scale = scales[scales["scale_estimator"] == PRIMARY_SCALE]
        if not np.allclose(
            primary_scale["sigma2_ratio_to_primary"], 1.0, atol=1e-12, rtol=0
        ):
            errors.append("primary scale ratio is not one")
        if not np.isfinite(
            scales[
                ["sigma2_raw", "sigma2", "sigma", "sigma2_ratio_to_primary"]
            ].to_numpy(float)
        ).all():
            errors.append("scale diagnostics contain non-finite values")

    covariates = files["residual_covariate_diagnostics"]
    if not covariates.empty:
        if set(covariates["covariate"].astype(str)) != set(FEATURE_COLUMNS):
            errors.append("residual diagnostics do not cover every covariate")
        relationship_columns = [
            "pearson_residual",
            "pearson_squared_residual",
            "spearman_residual",
            "spearman_squared_residual",
            "residual_slope",
            "squared_residual_slope",
        ]
        if not np.isfinite(covariates[relationship_columns].to_numpy(float)).all():
            errors.append("residual-covariate diagnostics contain non-finite values")

    attacks = files["attack_diagnostics"]
    if not attacks.empty:
        if not attacks["strict_envelope_passed"].astype(bool).all():
            errors.append("one or more strict-envelope attack audits failed")
        if (attacks[["maximum_gain_min", "minimum_gain_min"]] < -tolerance).any().any():
            errors.append("stronger attack audit worsened a primary extremum")
        if set(attacks["attack_algorithm"].astype(str)) != {ATTACK_ALGORITHM}:
            errors.append("attack diagnostics record the wrong algorithm")
        if not attacks["primary_path_count"].eq(experiment.retained_paths).all():
            errors.append("attack diagnostics record the wrong primary path count")

    calibration = files["attack_path_calibration"]
    calibration_summary = files["attack_path_calibration_summary"]
    if experiment.path_calibration.enabled:
        if calibration.empty:
            errors.append("enabled path calibration has no detail rows")
        else:
            if set(calibration["path_count"].astype(int)) != {1, 2, 3, 4}:
                errors.append("path calibration does not cover K=1,2,3,4")
            if set(calibration["stage"].astype(str)) != {"validation", "evaluation"}:
                errors.append("path calibration does not cover both stages")
            actual_radii = sorted(calibration["radius"].astype(float).unique())
            if len(actual_radii) != len(
                experiment.path_calibration.radii
            ) or not np.allclose(
                actual_radii,
                experiment.path_calibration.radii,
                atol=tolerance,
                rtol=0,
            ):
                errors.append("path calibration radii do not match the configuration")
            gap_columns = [
                "score_symmetric_relative_gap",
                "maximum_outward_gap_p99_response_sd",
                "minimum_outward_gap_p99_response_sd",
            ]
            if (calibration[gap_columns] < -tolerance).any().any():
                errors.append("path calibration contains negative gap diagnostics")
    elif not calibration.empty:
        errors.append("disabled path calibration produced detail rows")
    if not calibration_summary.empty:
        row = calibration_summary.iloc[0]
        if int(row["configured_path_count"]) != experiment.retained_paths:
            errors.append("path calibration summary records the wrong configured K")
        if not bool(row["configured_path_count_matches_recommendation"]):
            errors.append("configured K does not match the strict calibration decision")

    output_reference_json = run_dir / "reference_selection.json"
    if not output_reference_json.exists():
        errors.append("missing reference_selection.json")
    else:
        output_reference = json.loads(output_reference_json.read_text(encoding="utf-8"))
        if canonical_json_fingerprint(output_reference) != canonical_json_fingerprint(
            reference_spec
        ):
            errors.append("output reference selection differs from frozen spec")

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        errors.append("missing run_manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            errors.append("run manifest status is not complete")
        if manifest.get("scientific_schema_version") != AACV_SCIENTIFIC_SCHEMA_VERSION:
            errors.append("run manifest scientific schema is stale")
        if manifest.get("attack_algorithm") != ATTACK_ALGORITHM:
            errors.append("run manifest attack algorithm is incorrect")
        expected_fingerprint = experiment_fingerprint(config, reference_spec)
        if manifest.get("experiment_fingerprint") != expected_fingerprint:
            errors.append("run manifest experiment fingerprint mismatch")
        if manifest.get("reference_spec_fingerprint") != canonical_json_fingerprint(
            reference_spec
        ):
            errors.append("run manifest reference-spec fingerprint mismatch")
        recorded_source_hashes = manifest.get("source_sha256")
        if not recorded_source_hashes:
            errors.append("run manifest omits source hashes")
        elif recorded_source_hashes != _source_hashes():
            errors.append("run manifest source hashes differ from the current code")
        expected_split_seeds = {
            "outer": [
                {
                    "outer": outer_id,
                    "seed": deterministic_seed(experiment.base_seed, 0, outer_id),
                }
                for outer_id in range(experiment.outer_repeats)
            ],
            "inner": [
                {
                    "outer": outer_id,
                    "inner": inner_id,
                    "seed": deterministic_seed(
                        experiment.base_seed, 0, outer_id, inner_id
                    ),
                }
                for outer_id in range(experiment.outer_repeats)
                for inner_id in range(experiment.inner_repeats)
            ],
        }
        if manifest.get("split_seeds") != expected_split_seeds:
            errors.append("run manifest split seeds are missing or inconsistent")
        if set(manifest.get("seed_derivation", {})) != {
            "outer_split",
            "inner_split",
            "working_reference_crossfit",
            "attack",
        }:
            errors.append("run manifest seed derivation is incomplete")
        git = manifest.get("git", {})

        if "dirty" not in git or "status_entries" not in git:
            errors.append("run manifest omits Git worktree state")
        expected_status = {
            "known_variance_theorem_applies": False,
            "working_reference_is_true_regression_function": False,
            "working_error_is_physical_noise_law": False,
            "working_scales_use_inner_training_only": True,
            "nonlinear_extrema_are_numerical": True,
            "primary_attack_uses_multiple_independent_paths": True,
            "stronger_attack_inherits_primary_envelope": True,
            "stronger_extrema_cannot_degrade_primary": True,
            "heldout_best_uses_unknown_truth": False,
        }
        if manifest.get("scientific_status") != expected_status:
            errors.append("run manifest scientific-status declaration is incorrect")
        manifest_calibration = manifest.get("attack_path_calibration", {})
        if int(manifest_calibration.get("configured_path_count", -1)) != (
            experiment.retained_paths
        ):
            errors.append("run manifest omits the configured attack path count")
        if manifest.get("data_reuse_disclosure") != reference_spec.get(
            "data_reuse_disclosure"
        ):
            errors.append("run manifest omits the frozen data-reuse disclosure")

    mismatch_dir = run_dir / config["mismatch"]["output_subdir"]
    mismatch_files = {
        name: _load_csv(mismatch_dir, name, errors)
        for name in (
            "radius_mismatch_outer.csv",
            "radius_mismatch_summary.csv",
            "radius_mismatch_selection_frequencies.csv",
            "radius_mismatch_stability.csv",
            "radius_mismatch_validation_scores.csv",
            "radius_mismatch_low_excess_intervals.csv",
        )
    }
    mismatch_expected = {
        "radius_mismatch_outer.csv": radius_count * radius_count * outer_count * 2,
        "radius_mismatch_summary.csv": radius_count * radius_count * 2,
        "radius_mismatch_selection_frequencies.csv": radius_count
        * radius_count
        * 2
        * model_count,
        "radius_mismatch_stability.csv": 2 * max(radius_count - 1, 0) * radius_count,
        "radius_mismatch_validation_scores.csv": radius_count * model_count,
        "radius_mismatch_low_excess_intervals.csv": len(
            config["mismatch"]["representative_eval_radii"]
        ),
    }
    for name, expected in mismatch_expected.items():
        _expect_rows(mismatch_files[name], expected, name, errors)
        frame = mismatch_files[name]
        if not frame.empty and set(frame["analysis_variant"].astype(str)) != {
            PRIMARY_ANALYSIS_VARIANT
        }:
            errors.append(f"{name}: mismatch analysis is not primary-only")

    _check_output_terminology(run_dir, errors)
    if not reference_path.exists():
        errors.append("configured frozen reference spec is missing")
    return errors


def verify_aacv_figures(config: dict[str, Any], figure_dir: Path) -> list[str]:
    errors: list[str] = []
    stems = [
        config["figures"]["main_stem"],
        config["figures"]["mismatch_stem"],
        config["figures"]["diagnostics_stem"],
        config["figures"]["sensitivity_stem"],
    ]
    for stem in stems:
        for extension in config["figures"].get("formats", ["pdf", "png"]):
            path = figure_dir / f"{stem}.{extension}"
            if not path.exists():
                errors.append(f"missing figure: {path}")
            elif path.stat().st_size <= 1024:
                errors.append(f"figure is unexpectedly small: {path}")
        png = figure_dir / f"{stem}.png"
        if png.exists():
            image = np.asarray(mpimg.imread(png), dtype=float)
            if image.size == 0 or not np.isfinite(image).all():
                errors.append(f"figure has invalid pixels: {png}")
            elif float(np.std(image[..., :3])) < 0.01:
                errors.append(f"figure appears blank: {png}")
    return errors


def verify_aacv_all(config: dict[str, Any], run_dir: Path, figure_dir: Path) -> None:
    errors = verify_aacv_run(config, run_dir) + verify_aacv_figures(config, figure_dir)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"AACV verification failed:\n{joined}")
    print("AACV verification passed.")
