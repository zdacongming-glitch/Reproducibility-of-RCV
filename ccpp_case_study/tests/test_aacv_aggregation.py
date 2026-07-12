from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from ccpp_repro.aacv_experiment import (
    AACV_SCIENTIFIC_SCHEMA_VERSION,
    ATTACK_ALGORITHM,
    SHARD_TABLES,
    _validate_shard,
    _write_outer_shard,
    aacv_config_from_dict,
    experiment_fingerprint,
    run_aacv_and_write,
    summarize_attack_path_calibration,
)
from ccpp_repro.aacv_mismatch import (
    _check_matched_diagonal,
    build_radius_mismatch,
    low_excess_intervals,
    mismatch_selection_frequencies,
    mismatch_stability,
    summarize_radius_mismatch,
)
from ccpp_repro.aacv_reference import load_reference_spec, validate_reference_spec
from ccpp_repro.aacv_working_error import (
    PRIMARY_ANALYSIS_VARIANT,
    required_analysis_variants,
)
from ccpp_repro.config import load_config
from ccpp_repro.experiment import make_histgb_model, make_models


CANDIDATES = ("OLS", "RidgeCV", "LassoCV", "KNN", "RandomForest", "HistGB")


def _synthetic_primary_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_rows = []
    candidate_rows = []
    for radius in (0.0, 0.2):
        for model_index, model in enumerate(CANDIDATES):
            candidate_rows.append(
                {
                    "analysis_variant": PRIMARY_ANALYSIS_VARIANT,
                    "radius": radius,
                    "outer": 0,
                    "model": model,
                    "test_aacv_score": float(model_index + radius),
                }
            )
        for selector, selected in (("clean", "OLS"), ("aacv", "RidgeCV")):
            selected_score = radius if selected == "OLS" else 1.0 + radius
            outer_rows.append(
                {
                    "analysis_variant": PRIMARY_ANALYSIS_VARIANT,
                    "radius": radius,
                    "outer": 0,
                    "selector": selector,
                    "selected_model": selected,
                    "heldout_best_choice": "OLS",
                    "selected_test_aacv_score": selected_score,
                    "heldout_best_test_aacv_score": radius,
                    "heldout_score_excess": selected_score - radius,
                    "heldout_best_match": int(selected == "OLS"),
                }
            )
    return pd.DataFrame(outer_rows), pd.DataFrame(candidate_rows)


def test_mismatch_is_primary_only_and_has_full_radius_product() -> None:
    outer, candidates = _synthetic_primary_tables()
    mismatch = build_radius_mismatch(outer, candidates, CANDIDATES)
    assert len(mismatch) == 2 * 2 * 2
    assert set(mismatch["analysis_variant"]) == {PRIMARY_ANALYSIS_VARIANT}
    summary = summarize_radius_mismatch(mismatch)
    assert len(summary) == 2 * 2 * 2
    frequencies = mismatch_selection_frequencies(mismatch, CANDIDATES)
    assert len(frequencies) == 2 * 2 * 2 * len(CANDIDATES)
    assert len(mismatch_stability(mismatch)) == 2 * 1 * 2
    assert len(low_excess_intervals(summary, [0.2])) == 1
    _check_matched_diagonal(mismatch, outer, tolerance=1e-12)


def test_shard_schema_and_fingerprint_are_enforced(tmp_path: Path) -> None:
    tables = {
        name: pd.DataFrame({"value": [index]})
        for index, name in enumerate(SHARD_TABLES)
    }
    _write_outer_shard(tmp_path, 0, "fingerprint", tables)
    path = tmp_path / "partials" / "outer_000"
    _validate_shard(path, "fingerprint", 0)
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_shard(path, "different", 0)
    manifest_path = path / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scientific_schema_version"] = AACV_SCIENTIFIC_SCHEMA_VERSION - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Scientific schema"):
        _validate_shard(path, "fingerprint", 0)


def test_nonempty_output_requires_resume_without_modification(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    config = load_config("configs/ccpp_aacv_smoke.json")
    with pytest.raises(FileExistsError, match="nonempty"):
        run_aacv_and_write(config, output, progress=False, resume=False)
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_aacv_config_and_srcv_histgb_factory_remain_fixed() -> None:
    config = load_config("configs/ccpp_aacv_smoke.json")
    resolved = aacv_config_from_dict(config, progress=False)
    assert resolved.candidate_order == CANDIDATES
    assert resolved.analysis_variants == required_analysis_variants(CANDIDATES)
    assert resolved.attack_algorithm == ATTACK_ALGORITHM
    assert resolved.retained_paths == 4
    assert resolved.path_calibration.enabled
    assert resolved.path_calibration.path_counts == (1, 2, 3, 4)
    direct = make_histgb_model(1000, 17, 30)
    from_factory = make_models(1000, 17, rf_trees=10, hgb_max_iter=30)["HistGB"]
    assert direct.get_params() == from_factory.get_params()


def test_checked_in_reference_spec_is_frozen_and_valid() -> None:
    config = load_config("configs/ccpp_aacv_full.json")
    _path, spec = load_reference_spec(config)
    validate_reference_spec(spec, aacv_config=config)
    assert spec["selection_status"] == "frozen_before_aacv_full"
    assert spec["primary_working_reference_learner"] == "HistGB"
    assert spec["runner_up_working_reference_learner"] == "RandomForest"
    assert spec["source"]["split_count"] == 25 * 15
    assert {row["model"] for row in spec["candidate_summaries"]} == set(CANDIDATES)
    mismatched = copy.deepcopy(config)
    mismatched["working_error"]["analysis_variants"][0][
        "working_reference_learner"
    ] = "RandomForest"
    with pytest.raises(ValueError, match="frozen winner"):
        validate_reference_spec(spec, aacv_config=mismatched)


@pytest.mark.parametrize(
    (
        "config_name",
        "radius_count",
        "outer",
        "inner",
        "expected_scores",
        "expected_selections",
        "expected_outer_rows",
    ),
    [
        ("configs/ccpp_aacv_smoke.json", 5, 1, 1, 360, 120, 120),
        ("configs/ccpp_aacv_medium.json", 15, 1, 1, 1080, 360, 360),
        ("configs/ccpp_aacv_full.json", 15, 20, 3, 64800, 21600, 7200),
    ],
)
def test_protocol_row_formulas(
    config_name: str,
    radius_count: int,
    outer: int,
    inner: int,
    expected_scores: int,
    expected_selections: int,
    expected_outer_rows: int,
) -> None:
    config = load_config(config_name)
    resolved = aacv_config_from_dict(config, progress=False)
    variants = len(resolved.analysis_variants)
    models = len(CANDIDATES)
    assert variants == 12
    assert radius_count * outer * inner * models * variants == expected_scores
    assert radius_count * outer * inner * variants * 2 == expected_selections
    assert radius_count * outer * variants * 2 == expected_outer_rows


def test_medium_and_full_share_the_approved_radius_grid() -> None:
    medium = aacv_config_from_dict(
        load_config("configs/ccpp_aacv_medium.json"), progress=False
    )
    full = aacv_config_from_dict(
        load_config("configs/ccpp_aacv_full.json"), progress=False
    )
    assert medium.radii == full.radii
    assert medium.radii == (
        0.0,
        0.0025,
        0.005,
        0.01,
        0.05,
        0.08,
        0.1,
        0.125,
        0.15,
        0.3,
        0.45,
        0.475,
        0.5,
        0.525,
        0.6,
    )
    assert medium.outer_repeats == medium.inner_repeats == 1
    assert full.outer_repeats == 20
    assert full.inner_repeats == 3
    assert medium.rf_trees == full.rf_trees == 300
    assert medium.hgb_max_iter == full.hgb_max_iter == 300
    assert medium.validation_budget == full.validation_budget
    assert medium.evaluation_budget == full.evaluation_budget
    assert medium.audit_max_observations == 16
    assert full.audit_max_observations == 64


def test_scientific_schema_and_attack_config_change_fingerprint() -> None:
    config = load_config("configs/ccpp_aacv_smoke.json")
    _path, reference = load_reference_spec(config)
    baseline = experiment_fingerprint(config, reference)
    changed = copy.deepcopy(config)
    changed["optimizer"]["retained_paths"] = 3
    assert experiment_fingerprint(changed, reference) != baseline


def test_strict_path_calibration_gate_selects_three_or_four() -> None:
    config = aacv_config_from_dict(
        load_config("configs/ccpp_aacv_smoke.json"), progress=False
    )
    passing = pd.DataFrame(
        [
            {
                "stage": "validation",
                "path_count": 3,
                "selection_matches_reference": True,
                "score_symmetric_relative_gap": 0.001,
                "maximum_outward_gap_p99_response_sd": 0.005,
                "minimum_outward_gap_p99_response_sd": 0.004,
            },
            {
                "stage": "evaluation",
                "path_count": 3,
                "selection_matches_reference": None,
                "score_symmetric_relative_gap": 0.002,
                "maximum_outward_gap_p99_response_sd": 0.006,
                "minimum_outward_gap_p99_response_sd": 0.003,
            },
        ]
    )
    summary = summarize_attack_path_calibration(passing, config).iloc[0]
    assert bool(summary["calibration_passed"])
    assert int(summary["recommended_path_count"]) == 3
    failing = passing.copy()
    failing.loc[0, "score_symmetric_relative_gap"] = 0.006
    summary = summarize_attack_path_calibration(failing, config).iloc[0]
    assert not bool(summary["calibration_passed"])
    assert int(summary["recommended_path_count"]) == 4
