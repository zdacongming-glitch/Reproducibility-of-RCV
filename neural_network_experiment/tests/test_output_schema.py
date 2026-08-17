from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")

from src.experiments.pipeline import aggregate_results, compute_selection_frequencies, compute_tradeoff_table


def test_output_schema_columns_present() -> None:
    raw = pd.DataFrame(
        [
            {
                "replication_id": 0,
                "procedure": "erm",
                "n_total": 100,
                "lambda_value": 0.5,
                "r_train": 0.1,
                "r_eval": 0.0,
                "threat_model": "linf",
                "train_attack_family": "linf",
                "eval_attack_family": "linf",
                "stochastic_robust_mse": 1.0,
                "cv_srcv_score": 1.1,
                "training_time_sec": 0.5,
                "best_epoch": 2,
                "selected_srcv_matches_oracle": 1,
                "selected_procedure_srcv": "erm",
            },
            {
                "replication_id": 0,
                "procedure": "erm",
                "n_total": 100,
                "lambda_value": 0.5,
                "r_train": 0.1,
                "r_eval": 0.2,
                "threat_model": "linf",
                "train_attack_family": "linf",
                "eval_attack_family": "linf",
                "stochastic_robust_mse": 1.5,
                "cv_srcv_score": 1.3,
                "training_time_sec": 0.5,
                "best_epoch": 2,
                "selected_srcv_matches_oracle": 1,
                "selected_procedure_srcv": "erm",
            },
        ]
    )
    aggregated = aggregate_results(raw)
    selection = compute_selection_frequencies(raw)
    _, tradeoff = compute_tradeoff_table(raw, aggregated)
    assert "stochastic_robust_mse_mean" in aggregated.columns
    assert "frequency" in selection.columns
    assert "clean_endpoint_mse_mean" in tradeoff.columns
