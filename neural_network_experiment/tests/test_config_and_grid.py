from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.config import load_config
from src.experiments.pipeline import expand_experiment_grid


def test_load_smoke_config_and_expand_grid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiments" / "smoke.yaml")
    assert config.experiment.run_name == "smoke"
    points = expand_experiment_grid(config)
    assert len(points) == 2
    assert all(point.threat_model == "linf" for point in points)


def test_main_grid_is_training_grid_without_radius_filtering() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiments" / "main.yaml")
    points = expand_experiment_grid(config)
    assert len(points) == 48
    assert {point.threat_model for point in points} == {"linf", "l2"}
    assert config.experiment.parallel_workers == 4
    assert 0.0 in config.grid.eval_radii
    assert min(config.grid.eval_radii) < max(config.grid.train_radii)


def test_scaled_training_grid_sizes() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_sizes = {
        "main_lite.yaml": 16,
        "main_medium.yaml": 48,
        "main_full.yaml": 600,
    }
    for filename, expected_size in expected_sizes.items():
        config = load_config(root / "configs" / "experiments" / filename)
        points = expand_experiment_grid(config)
        assert len(points) == expected_size, filename
        assert config.cv.repeats == 5
        assert config.cv.holdout_ratio == pytest.approx(0.8)
