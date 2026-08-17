from __future__ import annotations

import copy
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.config import AttackConfig, RootConfig
from src.cv.selection import select_procedure
from src.data.synthetic_teacher_student import generate_replication
from src.plots.make_plots import make_all_plots
from src.utils import ensure_dir, to_serializable


@dataclass(frozen=True)
class ExperimentPoint:
    n_total: int
    lambda_value: float
    train_radius: float
    threat_model: str


def clone_with_point(config: RootConfig, point: ExperimentPoint) -> RootConfig:
    updated = copy.deepcopy(config)
    updated.data.n_total = point.n_total
    updated.teacher.lambda_value = point.lambda_value
    updated.attacks.train.family = point.threat_model
    updated.attacks.train.radius = point.train_radius
    updated.attacks.eval.family = point.threat_model
    return updated


def expand_experiment_grid(config: RootConfig) -> list[ExperimentPoint]:
    points: list[ExperimentPoint] = []
    for n_total, lambda_value, train_radius, threat_model in itertools.product(
        config.grid.n_total,
        config.grid.lambda_values,
        config.grid.train_radii,
        config.grid.threat_models,
    ):
        points.append(
            ExperimentPoint(
                n_total=n_total,
                lambda_value=lambda_value,
                train_radius=train_radius,
                threat_model=threat_model,
            )
        )
    return points


def configure_torch_parallelism(config: RootConfig) -> None:
    if config.experiment.parallel_workers <= 1:
        return
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except ImportError:
        return


def make_attack_config(base: AttackConfig, family: str, radius: float) -> AttackConfig:
    config = copy.deepcopy(base)
    config.family = family
    config.radius = radius
    return config


def run_single_replication(point_config: RootConfig, point: ExperimentPoint, replication_id: int, seed: int) -> list[dict[str, Any]]:
    replication = generate_replication(
        point_config,
        seed=seed,
        n_total=point.n_total,
        lambda_value=point.lambda_value,
    )
    train_attack = make_attack_config(point_config.attacks.train, point.threat_model, point.train_radius)
    eval_template = make_attack_config(point_config.attacks.eval, point.threat_model, radius=0.0)
    cv_result = select_procedure(
        train_data=replication.train,
        test_data=replication.test,
        config=point_config,
        procedures=point_config.grid.procedures,
        seed=seed + 100,
        train_attack_config=train_attack,
        validation_attack_config=eval_template,
        eval_radii=point_config.grid.eval_radii,
    )

    rows: list[dict[str, Any]] = []
    for radius in point_config.grid.eval_radii:
        radius_key = float(radius)
        selected = cv_result.selected_by_radius[radius_key]
        oracle_best = cv_result.oracle_by_radius[radius_key]
        for procedure in point_config.grid.procedures:
            fit_summary = cv_result.fit_summaries[procedure]
            row = {
                "replication_id": replication_id,
                "seed": seed,
                "teacher_seed": replication.teacher_seed,
                "procedure": procedure,
                "selected_procedure_srcv": selected,
                "cv_srcv_score": cv_result.per_radius_scores[radius_key][procedure]["srcv_mean"],
                "cv_srcv_score_std": cv_result.per_radius_scores[radius_key][procedure]["srcv_std"],
                "n_total": point.n_total,
                "lambda_value": point.lambda_value,
                "r_train": point.train_radius,
                "r_eval": radius_key,
                "threat_model": point.threat_model,
                "train_attack_family": train_attack.family,
                "eval_attack_family": eval_template.family,
                "stochastic_robust_mse": cv_result.per_radius_test_scores[radius_key][procedure]["test_mean"],
                "training_time_sec": fit_summary["training_time_sec"],
                "best_epoch": fit_summary["best_epoch"],
                "best_metric": fit_summary["best_metric"],
                "cv_method": point_config.cv.method,
                "oracle_best_procedure": oracle_best,
                "selected_srcv_matches_oracle": int(selected == oracle_best),
            }
            rows.append(row)
    return rows


def _replication_seed(base_seed: int, point_index: int, replication_id: int) -> int:
    return base_seed + 10000 * point_index + replication_id


def _run_replications_for_point(
    base_config: RootConfig,
    point_config: RootConfig,
    point: ExperimentPoint,
    point_index: int,
    total_points: int,
    show_progress: bool,
    progress_position: int | None = None,
    progress_desc_prefix: str = "",
) -> list[dict[str, Any]]:
    workers = max(1, int(base_config.experiment.parallel_workers))
    replication_ids = list(range(base_config.grid.num_replications))
    rows: list[dict[str, Any]] = []
    desc = f"{progress_desc_prefix}Replications[{point_index + 1}/{total_points}|w={workers}]"
    progress = (
        tqdm(
            total=len(replication_ids),
            desc=desc,
            leave=False,
            position=progress_position,
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )

    try:
        if workers == 1:
            for replication_id in replication_ids:
                rows.extend(
                    run_single_replication(
                        point_config,
                        point=point,
                        replication_id=replication_id,
                        seed=_replication_seed(base_config.experiment.seed, point_index, replication_id),
                    )
                )
                if progress is not None:
                    progress.update(1)
            return rows

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="robustcv") as executor:
            future_map = {
                executor.submit(
                    run_single_replication,
                    point_config,
                    point,
                    replication_id,
                    _replication_seed(base_config.experiment.seed, point_index, replication_id),
                ): replication_id
                for replication_id in replication_ids
            }
            for future in as_completed(future_map):
                rows.extend(future.result())
                if progress is not None:
                    progress.update(1)
        return rows
    finally:
        if progress is not None:
            progress.close()


def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "procedure",
        "n_total",
        "lambda_value",
        "r_train",
        "r_eval",
        "threat_model",
        "train_attack_family",
        "eval_attack_family",
    ]
    metric_cols = [
        "stochastic_robust_mse",
        "cv_srcv_score",
        "training_time_sec",
        "best_epoch",
        "selected_srcv_matches_oracle",
    ]
    grouped = raw_df.groupby(group_cols, dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, frame in grouped:
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_se"] = float(frame[metric].std(ddof=0) / np.sqrt(max(len(frame), 1)))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_selection_frequencies(raw_df: pd.DataFrame) -> pd.DataFrame:
    dedup_cols = [
        "replication_id",
        "n_total",
        "lambda_value",
        "r_train",
        "r_eval",
        "threat_model",
        "selected_procedure_srcv",
    ]
    dedup = raw_df[dedup_cols].drop_duplicates()
    rows: list[dict[str, Any]] = []
    group_cols = ["n_total", "lambda_value", "r_train", "r_eval", "threat_model"]
    for keys, frame in dedup.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        total = len(frame)
        for procedure, count in frame["selected_procedure_srcv"].dropna().value_counts().items():
            rows.append(
                {
                    **base,
                    "criterion": "selected_procedure_srcv",
                    "procedure": procedure,
                    "count": int(count),
                    "frequency": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def compute_tradeoff_table(raw_df: pd.DataFrame, aggregated_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_tradeoff = raw_df.copy()
    raw_key_cols = ["replication_id", "procedure", "n_total", "lambda_value", "r_train", "threat_model"]
    raw_clean = (
        raw_tradeoff[raw_tradeoff["r_eval"] == 0.0][[*raw_key_cols, "stochastic_robust_mse"]]
        .rename(columns={"stochastic_robust_mse": "clean_endpoint_mse"})
        .drop_duplicates(raw_key_cols)
    )
    raw_tradeoff = raw_tradeoff.merge(raw_clean, on=raw_key_cols, how="left")
    raw_tradeoff["stochastic_gap"] = raw_tradeoff["stochastic_robust_mse"] - raw_tradeoff["clean_endpoint_mse"]
    raw_tradeoff["stochastic_ratio"] = raw_tradeoff["stochastic_robust_mse"] / raw_tradeoff["clean_endpoint_mse"].clip(lower=1e-12)

    summary = aggregated_df.copy()
    summary_key_cols = ["procedure", "n_total", "lambda_value", "r_train", "threat_model"]
    summary_clean = (
        summary[summary["r_eval"] == 0.0][[*summary_key_cols, "stochastic_robust_mse_mean"]]
        .rename(columns={"stochastic_robust_mse_mean": "clean_endpoint_mse_mean"})
        .drop_duplicates(summary_key_cols)
    )
    summary = summary.merge(summary_clean, on=summary_key_cols, how="left")
    summary["stochastic_gap_mean"] = summary["stochastic_robust_mse_mean"] - summary["clean_endpoint_mse_mean"]
    summary["stochastic_ratio_mean"] = summary["stochastic_robust_mse_mean"] / summary["clean_endpoint_mse_mean"].clip(lower=1e-12)
    columns = [
        "procedure",
        "n_total",
        "lambda_value",
        "r_train",
        "r_eval",
        "threat_model",
        "train_attack_family",
        "eval_attack_family",
        "clean_endpoint_mse_mean",
        "stochastic_robust_mse_mean",
        "stochastic_robust_mse_se",
        "stochastic_gap_mean",
        "stochastic_ratio_mean",
    ]
    return raw_tradeoff, summary[columns]


def compute_winner_switch_points(selection_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["criterion", "n_total", "lambda_value", "r_train", "threat_model"]
    for keys, frame in selection_df.groupby(group_cols, dropna=False):
        winners = (
            frame.sort_values(["r_eval", "frequency", "procedure"], ascending=[True, False, True])
            .groupby("r_eval", dropna=False)
            .first()
            .reset_index()
            .sort_values("r_eval")
        )
        previous: str | None = None
        for _, row in winners.iterrows():
            current = str(row["procedure"])
            if previous is not None and current != previous:
                payload = dict(zip(group_cols, keys))
                payload["switch_radius"] = float(row["r_eval"])
                payload["from_procedure"] = previous
                payload["to_procedure"] = current
                rows.append(payload)
            previous = current
    return pd.DataFrame(rows)


def save_run_manifest(config: RootConfig, output_dir: Path) -> None:
    manifest_dir = ensure_dir(output_dir / "manifests")
    with (manifest_dir / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_serializable(asdict(config)), handle, sort_keys=False)


def run_experiment_grid(config: RootConfig, output_dir: str | Path, show_progress: bool = True) -> dict[str, pd.DataFrame]:
    output_path = ensure_dir(output_dir)
    save_run_manifest(config, output_path)
    configure_torch_parallelism(config)
    points = expand_experiment_grid(config)
    raw_rows: list[dict[str, Any]] = []
    point_progress = tqdm(total=len(points), desc="Training grid points", leave=True) if show_progress else None
    for point_index, point in enumerate(points):
        point_config = clone_with_point(config, point)
        raw_rows.extend(
            _run_replications_for_point(
                base_config=config,
                point_config=point_config,
                point=point,
                point_index=point_index,
                total_points=len(points),
                show_progress=show_progress,
            )
        )
        if point_progress is not None:
            point_progress.update(1)
    if point_progress is not None:
        point_progress.close()

    raw_df = pd.DataFrame(raw_rows)
    aggregate_df = aggregate_results(raw_df)
    selection_df = compute_selection_frequencies(raw_df)
    tradeoff_raw_df, tradeoff_summary_df = compute_tradeoff_table(raw_df, aggregate_df)
    switch_df = compute_winner_switch_points(selection_df)

    raw_df.to_csv(output_path / "raw_metrics.csv", index=False)
    aggregate_df.to_csv(output_path / "aggregated_metrics.csv", index=False)
    selection_df.to_csv(output_path / "selection_frequencies.csv", index=False)
    tradeoff_raw_df.to_csv(output_path / "tradeoff_raw.csv", index=False)
    tradeoff_summary_df.to_csv(output_path / "tradeoff_summary.csv", index=False)
    switch_df.to_csv(output_path / "winner_switch_points.csv", index=False)

    make_all_plots(
        aggregated_df=aggregate_df,
        selection_df=selection_df,
        raw_df=raw_df,
        output_dir=output_path / "plots",
        config=config,
    )
    return {
        "raw_metrics": raw_df,
        "aggregated_metrics": aggregate_df,
        "selection_frequencies": selection_df,
        "tradeoff_raw": tradeoff_raw_df,
        "tradeoff_summary": tradeoff_summary_df,
        "winner_switch_points": switch_df,
    }
