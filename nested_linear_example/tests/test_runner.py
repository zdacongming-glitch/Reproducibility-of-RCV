from pathlib import Path

import numpy as np
import pandas as pd

from robust_cv_sim.reporting import (
    _asymptotic_threshold_density_curve,
    _consistency_x_offsets,
    _consistency_y_limits,
    _criterion_boxplot_color,
    _format_r_value,
    _threshold_bins,
    generate_outputs,
    load_results,
    merge_chunks,
    plot_experiment_1,
    plot_experiment_2,
    plot_experiment_3,
    write_chunk,
)
from robust_cv_sim.runner import _chunk_path
from robust_cv_sim.simulation import (
    ASYMPTOTIC_STO_THRESHOLD,
    SimulationConfig,
    iter_experiment_chunks,
    run_chunk,
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
)


def test_iter_experiment_chunks_counts_match_expected_grids() -> None:
    config = SimulationConfig(
        n_grid=(100, 200),
        split_grid=(0.3, 0.7),
        exp2_n_grid=(50, 60, 70),
        exp3_beta2_grid=(0.5, 1.0),
        exp3_sigma_grid=(0.2,),
    )
    assert len(list(iter_experiment_chunks("experiment_1", config))) == 4
    assert len(list(iter_experiment_chunks("experiment_2", config))) == 6
    assert len(list(iter_experiment_chunks("experiment_3", config))) == 8


def test_run_chunk_matches_schema_and_full_experiment_rows() -> None:
    config = SimulationConfig(
        reps=2,
        n_grid=(100,),
        split_grid=(0.5,),
        radius_grid_adv=(0.0, 1.0),
        radius_grid_sto=(0.0, ASYMPTOTIC_STO_THRESHOLD),
    )
    chunk = next(iter(iter_experiment_chunks("experiment_1", config)))
    chunk_df = run_chunk(chunk, config)
    full_df = run_experiment_1(config)
    pd.testing.assert_frame_equal(chunk_df.reset_index(drop=True), full_df.reset_index(drop=True))


def test_write_chunk_and_merge_chunks_round_trip(tmp_path: Path) -> None:
    config = SimulationConfig(
        reps=2,
        n_grid=(100,),
        split_grid=(0.5,),
        radius_grid_adv=(0.0, 1.0),
        radius_grid_sto=(0.0, ASYMPTOTIC_STO_THRESHOLD),
    )
    experiment_dir = tmp_path / "experiment_1"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    chunk = next(iter(iter_experiment_chunks("experiment_1", config)))
    chunk_df = run_chunk(chunk, config)
    chunk_path = _chunk_path(experiment_dir, chunk)
    write_chunk(chunk_df, chunk_path)

    raw_path = merge_chunks(experiment_dir)
    merged_df = load_results(raw_path)
    pd.testing.assert_frame_equal(merged_df.reset_index(drop=True), chunk_df.reset_index(drop=True))


def test_resume_detection_uses_completed_chunk_file(tmp_path: Path) -> None:
    config = SimulationConfig(reps=1, n_grid=(100,), split_grid=(0.5,))
    experiment_dir = tmp_path / "experiment_1"
    chunk = next(iter(iter_experiment_chunks("experiment_1", config)))
    chunk_path = _chunk_path(experiment_dir, chunk)
    chunk_df = run_chunk(chunk, config)
    write_chunk(chunk_df, chunk_path)
    assert chunk_path.exists()
    assert chunk_path.stat().st_size > 0
    assert not chunk_path.with_suffix(chunk_path.suffix + ".tmp").exists()


def test_generate_outputs_skip_plots_only_writes_summaries(tmp_path: Path) -> None:
    config = SimulationConfig(
        reps=2,
        n_grid=(100,),
        split_grid=(0.5,),
        radius_grid_adv=(0.0, 1.0),
        radius_grid_sto=(0.0, ASYMPTOTIC_STO_THRESHOLD),
    )
    experiment_dir = tmp_path / "experiment_1"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    raw_path = experiment_dir / "experiment_1_raw.csv.gz"
    run_experiment_1(config).to_csv(raw_path, index=False, compression="gzip")

    generate_outputs(raw_path, "experiment_1", experiment_dir, skip_plots=True)
    assert (experiment_dir / "experiment_1_summary.csv").exists()
    assert (experiment_dir / "experiment_1_threshold_summary.csv").exists()
    assert not (experiment_dir / "plots").exists()


def test_asymptotic_threshold_density_curves_are_finite() -> None:
    stochastic_x, stochastic_y = _asymptotic_threshold_density_curve(
        "stochastic",
        n1=100,
        beta2=1.0,
        sigma=0.5,
        observed=pd.Series([1.7, ASYMPTOTIC_STO_THRESHOLD, 1.8]),
    )
    assert stochastic_x.size == stochastic_y.size == 300
    assert np.isfinite(stochastic_y).all()
    assert (stochastic_y >= 0.0).all()
    assert abs(float(stochastic_x[np.argmax(stochastic_y)]) - ASYMPTOTIC_STO_THRESHOLD) < 0.02

    adversarial_x, adversarial_y = _asymptotic_threshold_density_curve(
        "adversarial",
        n1=100,
        beta2=1.0,
        sigma=0.5,
        observed=pd.Series([0.9, 1.0, 1.1]),
    )
    assert adversarial_x.size == adversarial_y.size == 300
    assert np.isfinite(adversarial_y).all()
    assert (adversarial_y >= 0.0).all()
    assert float(adversarial_x.min()) < 1.0 < float(adversarial_x.max())


def test_experiment_1_plot_outputs_skip_aligned_loss_and_threshold(tmp_path: Path) -> None:
    config = SimulationConfig(
        reps=3,
        n_grid=(100,),
        split_grid=(0.5,),
        radius_grid_adv=(0.5, 1.0),
        radius_grid_sto=(1.0, ASYMPTOTIC_STO_THRESHOLD),
    )
    df = run_experiment_1(config)

    plot_experiment_1(df, tmp_path)

    files = {path.name for path in tmp_path.glob("*.png")}
    assert len(files) == 7
    assert "exp1_adversarial_n100_loss_curves.png" in files
    assert "exp1_adversarial_n100_threshold_hist.png" in files
    assert "exp1_stochastic_n100_loss_curves.png" in files
    assert "exp1_stochastic_n100_threshold_hist.png" in files
    assert "exp1_aligned_adversarial_n100_selection_prob.png" in files
    assert "exp1_aligned_adversarial_n100_loss_curves.png" not in files
    assert "exp1_aligned_adversarial_n100_threshold_hist.png" not in files


def test_experiment_2_delta_cv_boxplots_are_split_by_r_and_colored(tmp_path: Path) -> None:
    criterion_colors = [
        _criterion_boxplot_color("adversarial"),
        _criterion_boxplot_color("aligned_adversarial"),
        _criterion_boxplot_color("stochastic"),
    ]
    assert len(set(criterion_colors)) == 3

    config = SimulationConfig(
        reps=2,
        split_grid=(0.5,),
        exp2_n_grid=(80, 100),
        exp2_radius_grid_adv=(0.5, 1.0),
        exp2_radius_grid_sto=(0.5, ASYMPTOTIC_STO_THRESHOLD),
    )
    df = run_experiment_2(config)

    plot_experiment_2(df, tmp_path)

    files = {path.name for path in tmp_path.glob("*.png")}
    assert "exp2_adversarial_trainratio_0.5_consistency.png" in files
    assert "exp2_aligned_adversarial_trainratio_0.5_consistency.png" in files
    assert "exp2_stochastic_trainratio_0.5_consistency.png" in files
    assert not any(name.endswith("_delta_cv_boxplots.png") for name in files)

    for criterion in ("adversarial", "aligned_adversarial"):
        for r_label in ("0.5", "1.0"):
            assert f"exp2_{criterion}_trainratio_0.5_r_{r_label}_delta_cv_boxplot.png" in files
    for r_label in ("0.5", _format_r_value(ASYMPTOTIC_STO_THRESHOLD)):
        assert f"exp2_stochastic_trainratio_0.5_r_{r_label}_delta_cv_boxplot.png" in files


def test_experiment_3_plot_file_pattern_is_unchanged(tmp_path: Path) -> None:
    config = SimulationConfig(
        reps=1,
        n_grid=(80,),
        split_grid=(0.5,),
        radius_grid_adv=(0.5,),
        radius_grid_sto=(1.0,),
        exp3_beta2_grid=(1.0,),
        exp3_sigma_grid=(0.5,),
    )
    df = run_experiment_3(config)

    plot_experiment_3(df, tmp_path)

    files = {path.name for path in tmp_path.glob("*.png")}
    assert len(files) == 6
    for criterion in ("adversarial", "aligned_adversarial", "stochastic"):
        assert f"exp3_{criterion}_n80_trainratio_0.5_loss.png" in files
        assert f"exp3_{criterion}_n80_trainratio_0.5_selection.png" in files
    assert not any("delta_cv_boxplot" in name for name in files)


def test_small_rep_threshold_bins_are_not_fixed_to_thirty() -> None:
    series = pd.Series([0.95, 0.95, 1.01])
    assert _threshold_bins(series, reps=2) == 3
    assert _threshold_bins(series, reps=1000) == 30


def test_format_r_value_preserves_exact_discrete_labels() -> None:
    assert _format_r_value(0.5) == "0.5"
    assert _format_r_value(1.0) == "1.0"
    assert _format_r_value(2.2) == "2.2"
    assert _format_r_value(1.7320508075688772) == "1.732"


def test_consistency_offsets_separate_close_lines() -> None:
    offsets = _consistency_x_offsets([100, 200, 500, 1000], [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.2])
    assert len(offsets) == 8
    assert len(set(offsets.values())) == 8
    assert offsets[0.5] < 0.0
    assert offsets[2.2] > 0.0


def test_consistency_y_limits_use_zoomed_baseline_for_non_naive_methods() -> None:
    assert _consistency_y_limits("adversarial") == (-0.05, 1.05)
    assert _consistency_y_limits("aligned_adversarial") == (0.4, 1.03)
    assert _consistency_y_limits("stochastic") == (0.4, 1.03)
