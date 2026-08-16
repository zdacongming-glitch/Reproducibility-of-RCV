from __future__ import annotations

import pandas as pd
import pytest

from scripts.runner import build_parser
from src.experiments.multi_gpu import merge_partial_raw, parse_devices, split_point_indices


def test_parse_devices_accepts_comma_separated_ids() -> None:
    assert parse_devices("0,1") == ["0", "1"]
    assert parse_devices(" 0 , 1 ") == ["0", "1"]


def test_parse_devices_rejects_empty_or_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_devices("")
    with pytest.raises(ValueError, match="duplicate"):
        parse_devices("0,1,0")


def test_split_point_indices_round_robin_without_gaps() -> None:
    assignments = split_point_indices(total_points=7, num_workers=2)
    assert assignments == [[0, 2, 4, 6], [1, 3, 5]]
    flattened = sorted(point_index for assignment in assignments for point_index in assignment)
    assert flattened == list(range(7))


def test_merge_partial_raw_sorts_and_drops_internal_columns(tmp_path) -> None:
    first = tmp_path / "worker_00_raw_metrics.csv"
    second = tmp_path / "worker_01_raw_metrics.csv"
    pd.DataFrame(
        [
            {
                "_point_index": 2,
                "_worker_rank": 0,
                "_cuda_visible_device": "0",
                "replication_id": 0,
                "procedure": "erm",
                "n_total": 200,
                "lambda_value": 1.0,
                "r_train": 0.1,
                "r_eval": 0.0,
                "threat_model": "linf",
                "train_attack_family": "linf",
                "eval_attack_family": "linf",
                "stochastic_robust_mse": 2.0,
                "cv_srcv_score": 2.1,
                "training_time_sec": 0.5,
                "best_epoch": 3,
                "selected_srcv_matches_oracle": 1,
                "selected_procedure_srcv": "erm",
            }
        ]
    ).to_csv(first, index=False)
    pd.DataFrame(
        [
            {
                "_point_index": 1,
                "_worker_rank": 1,
                "_cuda_visible_device": "1",
                "replication_id": 0,
                "procedure": "core_only",
                "n_total": 100,
                "lambda_value": 0.0,
                "r_train": 0.1,
                "r_eval": 0.0,
                "threat_model": "linf",
                "train_attack_family": "linf",
                "eval_attack_family": "linf",
                "stochastic_robust_mse": 1.0,
                "cv_srcv_score": 1.1,
                "training_time_sec": 0.4,
                "best_epoch": 2,
                "selected_srcv_matches_oracle": 1,
                "selected_procedure_srcv": "core_only",
            }
        ]
    ).to_csv(second, index=False)

    merged = merge_partial_raw([first, second])

    assert "_point_index" not in merged.columns
    assert "_worker_rank" not in merged.columns
    assert merged["procedure"].tolist() == ["core_only", "erm"]

    expected_metric_columns = {
        "stochastic_robust_mse",
        "cv_srcv_score",
        "training_time_sec",
        "best_epoch",
        "selected_srcv_matches_oracle",
    }
    assert expected_metric_columns.issubset(set(merged.columns))


def test_runner_parser_adds_multi_gpu_without_changing_existing_main() -> None:
    parser = build_parser()
    multi_gpu_args = parser.parse_args(["multi-gpu", "main", "--scale", "lite", "--devices", "0,1"])
    assert multi_gpu_args.command == "multi-gpu"
    assert multi_gpu_args.multi_gpu_command == "main"
    assert multi_gpu_args.scale == "lite"
    assert multi_gpu_args.devices == "0,1"
    assert multi_gpu_args.workers_per_gpu == 1

    main_args = parser.parse_args(["main", "--scale", "lite"])
    assert main_args.command == "main"
    assert main_args.scale == "lite"
