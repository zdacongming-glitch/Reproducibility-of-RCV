from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_DIR


REFERENCE_SCHEMA_VERSION = 1
REFERENCE_TABLE = "inner_validation_scores.csv"


def canonical_json_fingerprint(payload: object) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "_config_path"}
    return canonical_json_fingerprint(payload)


def normalized_reference_table_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["radius", "outer", "inner", "model", "clean_cv"]
    ordered = frame[columns].copy()
    ordered["radius"] = ordered["radius"].astype(float)
    ordered["outer"] = ordered["outer"].astype(int)
    ordered["inner"] = ordered["inner"].astype(int)
    ordered["model"] = ordered["model"].astype(str)
    ordered["clean_cv"] = ordered["clean_cv"].astype(float)
    ordered = ordered.sort_values(["outer", "inner", "model"]).reset_index(drop=True)
    records = [
        {
            "radius": format(float(row.radius), ".17g"),
            "outer": int(row.outer),
            "inner": int(row.inner),
            "model": str(row.model),
            "clean_cv": format(float(row.clean_cv), ".17g"),
        }
        for row in ordered.itertuples(index=False)
    ]
    return canonical_json_fingerprint(records)


def _standard_error(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def build_reference_spec(
    srcv_config: dict[str, Any],
    validation_scores: pd.DataFrame,
) -> dict[str, Any]:
    required_columns = {"radius", "outer", "inner", "model", "clean_cv"}
    missing = required_columns - set(validation_scores.columns)
    if missing:
        raise ValueError(f"Reference table is missing columns: {sorted(missing)}")

    candidate_order = tuple(srcv_config["models"]["candidate_order"])
    experiment = srcv_config["experiment"]
    expected_splits = int(experiment["outer_repeats"]) * int(
        experiment["inner_repeats"]
    )
    zero = validation_scores[
        np.isclose(validation_scores["radius"].astype(float), 0.0)
    ].copy()
    if zero.empty:
        raise ValueError("Reference table has no radius-zero rows")
    if zero.duplicated(["outer", "inner", "model"]).any():
        raise ValueError("Reference table has duplicate radius-zero split/model rows")
    actual_models = set(zero["model"].astype(str))
    if actual_models != set(candidate_order):
        raise ValueError(
            f"Reference models {actual_models} do not match {set(candidate_order)}"
        )
    split_count = zero[["outer", "inner"]].drop_duplicates().shape[0]
    if split_count != expected_splits:
        raise ValueError(
            f"Expected {expected_splits} reference splits, found {split_count}"
        )
    expected_rows = expected_splits * len(candidate_order)
    if len(zero) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} radius-zero rows, found {len(zero)}"
        )

    pivot = zero.pivot(index=["outer", "inner"], columns="model", values="clean_cv")
    pivot = pivot.loc[:, list(candidate_order)].sort_index()
    means = pivot.mean(axis=0)
    winner = min(
        candidate_order,
        key=lambda model: (means[model], candidate_order.index(model)),
    )
    ranked = sorted(
        candidate_order,
        key=lambda model: (means[model], candidate_order.index(model)),
    )
    wins = {model: 0 for model in candidate_order}
    for row in pivot.itertuples(index=False, name=None):
        scores = dict(zip(candidate_order, row, strict=True))
        selected = min(
            candidate_order,
            key=lambda model: (scores[model], candidate_order.index(model)),
        )
        wins[selected] += 1

    summaries: list[dict[str, Any]] = []
    for rank, model in enumerate(ranked, start=1):
        values = pivot[model]
        paired_difference = values - pivot[winner]
        summaries.append(
            {
                "model": model,
                "rank": rank,
                "mean_clean_mse": float(values.mean()),
                "clean_mse_mcse": _standard_error(values),
                "difference_from_winner": float(paired_difference.mean()),
                "difference_mcse": _standard_error(paired_difference),
                "split_wins": int(wins[model]),
                "split_win_frequency": float(wins[model] / expected_splits),
            }
        )

    source_config = {
        key: value for key, value in srcv_config.items() if key != "_config_path"
    }
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "selection_status": "frozen_before_aacv_full",
        "selection_protocol": {
            "source_config": "configs/ccpp_full.json",
            "source_table": REFERENCE_TABLE,
            "radius": 0.0,
            "criterion": "clean_cv",
            "aggregation": "mean_over_predeclared_outer_inner_splits",
            "tie_breaking": "candidate_order",
            "split_keys": ["outer", "inner"],
        },
        "source": {
            "srcv_config_fingerprint": canonical_json_fingerprint(source_config),
            "reference_table_fingerprint": normalized_reference_table_fingerprint(zero),
            "data_sha256": str(srcv_config["data"]["sha256"]).upper(),
            "outer_repeats": int(experiment["outer_repeats"]),
            "inner_repeats": int(experiment["inner_repeats"]),
            "split_count": expected_splits,
        },
        "candidate_order": list(candidate_order),
        "candidate_protocol": {
            "rf_trees": int(srcv_config["models"]["rf_trees"]),
            "hgb_max_iter": int(srcv_config["models"]["hgb_max_iter"]),
            "factory": "ccpp_repro.experiment.make_models",
        },
        "primary_working_reference_learner": winner,
        "runner_up_working_reference_learner": ranked[1],
        "candidate_summaries": summaries,
        "data_reuse_disclosure": (
            "The canonical SRCV clean-risk analysis and AACV use the same CCPP "
            "case-study dataset. This frozen design decision is not independent "
            "external validation and does not identify the true regression function."
        ),
    }


def validate_reference_spec(
    spec: dict[str, Any],
    *,
    aacv_config: dict[str, Any] | None = None,
) -> None:
    if int(spec.get("schema_version", -1)) != REFERENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported AACV reference-spec schema")
    if spec.get("selection_status") != "frozen_before_aacv_full":
        raise ValueError("AACV reference selection is not frozen before the full run")

    order = tuple(spec.get("candidate_order", ()))
    summaries = spec.get("candidate_summaries", [])
    if len(order) < 2:
        raise ValueError("Reference spec must contain at least two candidates")
    if len(summaries) != len(order):
        raise ValueError("Reference-spec summaries do not match candidate_order")

    ranked_summaries = sorted(summaries, key=lambda item: item["rank"])
    if set(item.get("model") for item in ranked_summaries) != set(order):
        raise ValueError("Reference-spec ranking does not cover candidate_order")
    expected_ranks = list(range(1, len(order) + 1))
    if [int(item["rank"]) for item in ranked_summaries] != expected_ranks:
        raise ValueError("Reference-spec ranks are not consecutive")

    means = {item["model"]: float(item["mean_clean_mse"]) for item in summaries}
    if not all(math.isfinite(value) for value in means.values()):
        raise ValueError("Reference-spec clean risks are not finite")
    expected = min(order, key=lambda model: (means[model], order.index(model)))
    if spec.get("primary_working_reference_learner") != expected:
        raise ValueError("Reference-spec primary learner is not the clean-risk winner")
    ranked = sorted(order, key=lambda model: (means[model], order.index(model)))
    if spec.get("runner_up_working_reference_learner") != ranked[1]:
        raise ValueError("Reference-spec runner-up is inconsistent with ranking")

    split_count = int(spec["source"]["split_count"])
    if split_count <= 0:
        raise ValueError("Reference-spec split count must be positive")
    if sum(int(item["split_wins"]) for item in summaries) != split_count:
        raise ValueError("Reference-spec split wins do not sum to the split count")
    for item in summaries:
        wins = int(item["split_wins"])
        frequency = float(item["split_win_frequency"])
        if not 0 <= wins <= split_count:
            raise ValueError("Reference-spec split wins are out of range")
        if not math.isclose(frequency, wins / split_count, abs_tol=1e-12):
            raise ValueError("Reference-spec split win frequency is inconsistent")
    if not str(spec.get("data_reuse_disclosure", "")).strip():
        raise ValueError("Reference spec omits the data-reuse disclosure")

    if aacv_config is None:
        return

    configured_order = tuple(aacv_config["models"]["candidate_order"])
    if configured_order != order:
        raise ValueError("AACV candidate order differs from frozen reference spec")
    if (
        str(aacv_config["data"]["sha256"]).upper()
        != str(spec["source"]["data_sha256"]).upper()
    ):
        raise ValueError("AACV data hash differs from frozen reference spec")

    primary_variants = [
        item
        for item in aacv_config["working_error"]["analysis_variants"]
        if item.get("role") == "primary"
    ]
    if len(primary_variants) != 1 or primary_variants[0].get(
        "working_reference_learner"
    ) != spec.get("primary_working_reference_learner"):
        raise ValueError("AACV primary reference differs from the frozen winner")

    protocol = spec["candidate_protocol"]
    reference = aacv_config["working_error"]["reference_protocol"]
    if int(reference["rf_trees"]) != int(protocol["rf_trees"]):
        raise ValueError("Working-reference RF protocol differs from frozen spec")
    if int(reference["hgb_max_iter"]) != int(protocol["hgb_max_iter"]):
        raise ValueError("Working-reference HistGB protocol differs from frozen spec")


def load_reference_spec(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(config["working_error"]["reference_spec"])
    if not path.is_absolute():
        path = PROJECT_DIR / path
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    validate_reference_spec(spec, aacv_config=config)
    return path, spec


def write_reference_spec(path: Path, spec: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def freeze_reference_from_run(
    srcv_config: dict[str, Any],
    srcv_run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    table_path = Path(srcv_run_dir) / REFERENCE_TABLE
    if not table_path.exists():
        raise FileNotFoundError(f"Missing canonical SRCV table: {table_path}")
    spec = build_reference_spec(srcv_config, pd.read_csv(table_path))
    validate_reference_spec(spec)
    write_reference_spec(output_path, spec)
    return spec
