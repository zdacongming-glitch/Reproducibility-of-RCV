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
    assert resolved.prediction_chunk_rows == 2048
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
        ("configs/ccpp_aacv_medium.json", 54, 1, 1, 3888, 1296, 1296),
        ("configs/ccpp_aacv_full.json", 37, 30, 3, 239760, 79920, 26640),
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


def test_medium_and_full_radius_protocols() -> None:
    medium = aacv_config_from_dict(
        load_config("configs/ccpp_aacv_medium.json"), progress=False
    )
    full = aacv_config_from_dict(
        load_config("configs/ccpp_aacv_full.json"), progress=False
    )
    expected_medium = (
        0.0,
        0.0025,
        0.005,
        0.01,
        *(0.02 * index for index in range(1, 51)),
    )
    expected_full = (
        0.0,
        0.0025,
        0.005,
        0.0075,
        0.0085,
        0.01,
        0.02,
        0.04,
        0.05,
        0.055,
        0.06,
        0.065,
        0.07,
        0.075,
        0.08,
        0.10,
        0.15,
        0.22,
        0.30,
        0.40,
        0.50,
        0.60,
        0.68,
        0.70,
        0.705,
        0.71,
        0.715,
        0.72,
        0.725,
        0.73,
        0.74,
        0.75,
        0.76,
        0.78,
        0.82,
        0.86,
        0.90,
    )
    assert len(medium.radii) == 54
    assert medium.radii == pytest.approx(expected_medium, rel=0, abs=1e-12)
    assert medium.radii[-1] == 1.0
    assert len(full.radii) == 37
    assert full.radii == pytest.approx(expected_full, rel=0, abs=1e-12)
    assert full.radii[-1] == 0.9
    assert medium.outer_repeats == medium.inner_repeats == 1
    assert full.outer_repeats == 30
    assert full.inner_repeats == 3
    assert medium.rf_trees == full.rf_trees == 300
    assert medium.hgb_max_iter == full.hgb_max_iter == 300
    assert medium.attack_algorithm == full.attack_algorithm == ATTACK_ALGORITHM
    assert medium.prediction_chunk_rows == full.prediction_chunk_rows == 2048


def test_scientific_schema_and_attack_config_change_fingerprint() -> None:
    config = load_config("configs/ccpp_aacv_smoke.json")
    _path, reference = load_reference_spec(config)
    baseline = experiment_fingerprint(config, reference)
    changed = copy.deepcopy(config)
    changed["attack_set"]["prediction_chunk_rows"] = 1024
    assert experiment_fingerprint(changed, reference) != baseline
    assert AACV_SCIENTIFIC_SCHEMA_VERSION == 3


def test_legacy_optimizer_configuration_is_rejected() -> None:
    config = load_config("configs/ccpp_aacv_smoke.json")
    config["optimizer"] = {"algorithm": "multipath_strict_envelope_v1"}
    with pytest.raises(ValueError, match="Legacy AACV optimizer"):
        aacv_config_from_dict(config, progress=False)
