from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from src.config import load_config  # noqa: E402
from src.cv.selection import CVSelectionResult  # noqa: E402
from src.experiments import pipeline  # noqa: E402


def test_run_single_replication_uses_split_averaged_cv_result(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiments" / "smoke.yaml")
    config.grid.procedures = ["erm", "core_only"]
    config.grid.eval_radii = [0.0, 0.2]
    point = pipeline.ExperimentPoint(n_total=240, lambda_value=0.5, train_radius=0.1, threat_model="linf")

    sentinel_train = object()
    sentinel_test = object()
    fake_replication = SimpleNamespace(train=sentinel_train, test=sentinel_test, teacher_seed=1234)
    seen: dict[str, object] = {}

    def fake_generate_replication(*args, **kwargs):
        return fake_replication

    def fake_select_procedure(**kwargs):
        seen["train_data"] = kwargs["train_data"]
        seen["test_data"] = kwargs["test_data"]
        return CVSelectionResult(
            per_radius_scores={
                0.0: {
                    "erm": {"srcv_mean": 1.0, "srcv_std": 0.1},
                    "core_only": {"srcv_mean": 1.5, "srcv_std": 0.2},
                },
                0.2: {
                    "erm": {"srcv_mean": 2.0, "srcv_std": 0.3},
                    "core_only": {"srcv_mean": 1.0, "srcv_std": 0.4},
                },
            },
            per_radius_test_scores={
                0.0: {
                    "erm": {"test_mean": 3.0, "test_std": 0.0},
                    "core_only": {"test_mean": 2.0, "test_std": 0.0},
                },
                0.2: {
                    "erm": {"test_mean": 4.0, "test_std": 0.0},
                    "core_only": {"test_mean": 5.0, "test_std": 0.0},
                },
            },
            selected_by_radius={0.0: "erm", 0.2: "core_only"},
            oracle_by_radius={0.0: "core_only", 0.2: "erm"},
            fit_summaries={
                "erm": {"training_time_sec": 6.0, "best_epoch": 7.0, "best_metric": 0.7},
                "core_only": {"training_time_sec": 5.0, "best_epoch": 8.0, "best_metric": 0.8},
            },
            metadata={},
        )

    monkeypatch.setattr(pipeline, "generate_replication", fake_generate_replication)
    monkeypatch.setattr(pipeline, "select_procedure", fake_select_procedure)

    rows = pipeline.run_single_replication(config, point=point, replication_id=0, seed=2026)

    assert seen["train_data"] is sentinel_train
    assert seen["test_data"] is sentinel_test
    assert len(rows) == 4
    erm_clean = next(row for row in rows if row["procedure"] == "erm" and row["r_eval"] == 0.0)
    assert erm_clean["stochastic_robust_mse"] == pytest.approx(3.0)
    assert erm_clean["training_time_sec"] == pytest.approx(6.0)
    assert erm_clean["best_epoch"] == pytest.approx(7.0)
    assert erm_clean["oracle_best_procedure"] == "core_only"
