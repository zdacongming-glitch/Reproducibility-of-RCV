from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml
from tqdm.auto import tqdm

from src.config import load_config
from src.utils import ensure_dir, to_serializable


INTERNAL_COLUMNS = ["_point_index", "_worker_rank", "_cuda_visible_device"]


@dataclass(frozen=True)
class WorkerSpec:
    rank: int
    device: str
    point_indices: list[int]
    partial_path: Path
    status_path: Path


def parse_devices(devices: str | Sequence[str]) -> list[str]:
    if isinstance(devices, str):
        parsed = [device.strip() for device in devices.split(",")]
    else:
        parsed = [str(device).strip() for device in devices]
    if not parsed or any(device == "" for device in parsed):
        raise ValueError("--devices must contain at least one non-empty device id.")
    if len(set(parsed)) != len(parsed):
        raise ValueError("--devices must not contain duplicate device ids.")
    return parsed


def split_point_indices(total_points: int, num_workers: int) -> list[list[int]]:
    if total_points < 0:
        raise ValueError("total_points must be non-negative.")
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1.")
    assignments = [[] for _ in range(num_workers)]
    for point_index in range(total_points):
        assignments[point_index % num_workers].append(point_index)
    return assignments


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_serializable(payload), handle, sort_keys=False)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": f"Missing worker status file: {path}"}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {"ok": False, "error": f"Invalid worker status file: {path}"}


def _point_indices_arg(point_indices: Sequence[int]) -> str:
    return ",".join(str(point_index) for point_index in point_indices)


def parse_point_indices(value: str) -> list[int]:
    if value.strip() == "":
        return []
    indices: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if token == "":
            raise ValueError("--point-indices contains an empty entry.")
        indices.append(int(token))
    return indices


def make_worker_specs(output_dir: Path, devices: Sequence[str], total_points: int) -> list[WorkerSpec]:
    partial_dir = ensure_dir(output_dir / "partials")
    assignments = split_point_indices(total_points=total_points, num_workers=len(devices))
    specs: list[WorkerSpec] = []
    for rank, (device, point_indices) in enumerate(zip(devices, assignments)):
        specs.append(
            WorkerSpec(
                rank=rank,
                device=device,
                point_indices=point_indices,
                partial_path=partial_dir / f"worker_{rank:02d}_raw_metrics.csv",
                status_path=partial_dir / f"worker_{rank:02d}_status.yaml",
            )
        )
    return specs


def _worker_command(
    runner_path: Path,
    config_path: Path,
    spec: WorkerSpec,
    workers_per_gpu: int,
    total_points: int,
    show_progress: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(runner_path),
        "_multi-gpu-worker",
        "--config",
        str(config_path),
        "--partial-path",
        str(spec.partial_path),
        "--status-path",
        str(spec.status_path),
        "--point-indices",
        _point_indices_arg(spec.point_indices),
        "--rank",
        str(spec.rank),
        "--device",
        spec.device,
        "--workers-per-gpu",
        str(workers_per_gpu),
        "--total-points",
        str(total_points),
    ]
    if show_progress:
        command.append("--show-progress")
    return command


def _status_payload(specs: Sequence[WorkerSpec], processes: dict[int, subprocess.Popen] | None = None) -> list[dict[str, Any]]:
    process_map = processes or {}
    workers: list[dict[str, Any]] = []
    for spec in specs:
        process = process_map.get(spec.rank)
        workers.append(
            {
                "rank": spec.rank,
                "device": spec.device,
                "point_indices": spec.point_indices,
                "point_count": len(spec.point_indices),
                "partial_path": str(spec.partial_path),
                "status_path": str(spec.status_path),
                "returncode": None if process is None else process.returncode,
                "status": _read_yaml(spec.status_path) if spec.status_path.exists() else None,
            }
        )
    return workers


def _write_multi_gpu_manifest(
    output_dir: Path,
    devices: Sequence[str],
    workers_per_gpu: int,
    specs: Sequence[WorkerSpec],
    processes: dict[int, subprocess.Popen] | None = None,
    state: str = "running",
) -> None:
    _write_yaml(
        output_dir / "manifests" / "multi_gpu.yaml",
        {
            "state": state,
            "created_at": _now(),
            "devices": list(devices),
            "workers_per_gpu": workers_per_gpu,
            "workers": _status_payload(specs, processes=processes),
        },
    )


def _launch_workers(
    runner_path: Path,
    repo_root: Path,
    config_path: Path,
    specs: Sequence[WorkerSpec],
    workers_per_gpu: int,
    total_points: int,
    show_progress: bool,
) -> dict[int, subprocess.Popen]:
    processes: dict[int, subprocess.Popen] = {}
    for spec in specs:
        if not spec.point_indices:
            _write_yaml(
                spec.status_path,
                {
                    "ok": True,
                    "skipped": True,
                    "rank": spec.rank,
                    "device": spec.device,
                    "point_indices": [],
                    "started_at": _now(),
                    "finished_at": _now(),
                },
            )
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = spec.device
        env["PYTHONUNBUFFERED"] = "1"
        command = _worker_command(
            runner_path=runner_path,
            config_path=config_path,
            spec=spec,
            workers_per_gpu=workers_per_gpu,
            total_points=total_points,
            show_progress=show_progress,
        )
        processes[spec.rank] = subprocess.Popen(command, cwd=str(repo_root), env=env)
    return processes


def _wait_for_workers(processes: dict[int, subprocess.Popen], show_progress: bool) -> None:
    remaining = set(processes)
    progress = tqdm(total=len(remaining), desc="GPU workers", leave=True) if show_progress and remaining else None
    try:
        while remaining:
            finished = [rank for rank in remaining if processes[rank].poll() is not None]
            for rank in finished:
                remaining.remove(rank)
                if progress is not None:
                    progress.update(1)
            if remaining:
                time.sleep(1.0)
    finally:
        if progress is not None:
            progress.close()


def _raise_if_workers_failed(specs: Sequence[WorkerSpec], processes: dict[int, subprocess.Popen]) -> None:
    failures: list[str] = []
    for spec in specs:
        process = processes.get(spec.rank)
        status = _read_yaml(spec.status_path)
        returncode = 0 if process is None else process.returncode
        if returncode == 0 and status.get("ok") is True:
            continue
        error = status.get("error") or status.get("traceback") or "No worker error detail was written."
        failures.append(
            f"worker rank={spec.rank} device={spec.device} points={spec.point_indices} "
            f"returncode={returncode}: {error}"
        )
    if failures:
        raise RuntimeError("One or more multi-GPU workers failed:\n" + "\n\n".join(failures))


def _sort_and_drop_internal_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        "_point_index",
        "replication_id",
        "r_eval",
        "procedure",
        "n_total",
        "lambda_value",
        "r_train",
        "threat_model",
    ]
    present_sort_cols = [column for column in sort_cols if column in raw_df.columns]
    if present_sort_cols:
        raw_df = raw_df.sort_values(present_sort_cols, kind="mergesort").reset_index(drop=True)
    public_df = raw_df.drop(columns=[column for column in INTERNAL_COLUMNS if column in raw_df.columns])
    return public_df.reset_index(drop=True)


def merge_partial_raw(partial_paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for partial_path in partial_paths:
        if not partial_path.exists():
            continue
        frame = pd.read_csv(partial_path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No partial raw metrics were produced by multi-GPU workers.")
    return _sort_and_drop_internal_columns(pd.concat(frames, ignore_index=True))


def write_outputs_from_raw(config, output_dir: str | Path, raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from src.experiments.pipeline import (
        aggregate_results,
        compute_selection_frequencies,
        compute_tradeoff_table,
        compute_winner_switch_points,
    )
    from src.plots.make_plots import make_all_plots

    output_path = ensure_dir(output_dir)
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


def run_multi_gpu_grid(
    config,
    output_dir: str | Path,
    devices: str | Sequence[str],
    workers_per_gpu: int = 1,
    show_progress: bool = True,
    runner_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    from src.experiments.pipeline import expand_experiment_grid, save_run_manifest

    parsed_devices = parse_devices(devices)
    if workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1.")

    run_config = copy.deepcopy(config)
    run_config.experiment.parallel_workers = workers_per_gpu
    output_path = ensure_dir(output_dir)
    save_run_manifest(run_config, output_path)

    repo_root = Path(__file__).resolve().parents[2]
    resolved_runner = Path(runner_path) if runner_path is not None else repo_root / "scripts" / "runner.py"
    config_path = output_path / "manifests" / "run_config.yaml"
    points = expand_experiment_grid(run_config)
    specs = make_worker_specs(output_path, parsed_devices, total_points=len(points))

    _write_multi_gpu_manifest(
        output_dir=output_path,
        devices=parsed_devices,
        workers_per_gpu=workers_per_gpu,
        specs=specs,
        state="running",
    )
    processes = _launch_workers(
        runner_path=resolved_runner,
        repo_root=repo_root,
        config_path=config_path,
        specs=specs,
        workers_per_gpu=workers_per_gpu,
        total_points=len(points),
        show_progress=show_progress,
    )
    _wait_for_workers(processes, show_progress=False)
    _write_multi_gpu_manifest(
        output_dir=output_path,
        devices=parsed_devices,
        workers_per_gpu=workers_per_gpu,
        specs=specs,
        processes=processes,
        state="finished",
    )
    _raise_if_workers_failed(specs, processes)

    raw_df = merge_partial_raw([spec.partial_path for spec in specs])
    results = write_outputs_from_raw(run_config, output_path, raw_df)
    _write_multi_gpu_manifest(
        output_dir=output_path,
        devices=parsed_devices,
        workers_per_gpu=workers_per_gpu,
        specs=specs,
        processes=processes,
        state="merged",
    )
    return results


def run_multi_gpu_worker(
    config_path: str | Path,
    partial_path: str | Path,
    status_path: str | Path,
    point_indices: Sequence[int],
    rank: int,
    device: str,
    workers_per_gpu: int,
    total_points: int,
    show_progress: bool = False,
) -> None:
    from src.experiments.pipeline import (
        _run_replications_for_point,
        clone_with_point,
        configure_torch_parallelism,
        expand_experiment_grid,
    )

    partial = Path(partial_path)
    status = Path(status_path)
    started_at = _now()
    try:
        config = load_config(config_path)
        config.experiment.parallel_workers = workers_per_gpu
        configure_torch_parallelism(config)
        points = expand_experiment_grid(config)
        invalid = [point_index for point_index in point_indices if point_index < 0 or point_index >= len(points)]
        if invalid:
            raise ValueError(f"Invalid point indices for worker {rank}: {invalid}")

        raw_rows: list[dict[str, Any]] = []
        grid_position = rank * 2
        replication_position = grid_position + 1
        grid_progress = (
            tqdm(
                total=len(point_indices),
                desc=f"GPU {device} grid points",
                position=grid_position,
                leave=True,
                dynamic_ncols=True,
            )
            if show_progress
            else None
        )
        try:
            for point_index in point_indices:
                point = points[point_index]
                if grid_progress is not None:
                    grid_progress.set_description(f"GPU {device} Grid[{point_index + 1}/{total_points}]")
                    grid_progress.set_postfix(
                        n=point.n_total,
                        lam=point.lambda_value,
                        r_train=point.train_radius,
                        threat=point.threat_model,
                    )
                point_config = clone_with_point(config, point)
                rows = _run_replications_for_point(
                    base_config=config,
                    point_config=point_config,
                    point=point,
                    point_index=point_index,
                    total_points=total_points,
                    show_progress=show_progress,
                    progress_position=replication_position,
                    progress_desc_prefix=f"GPU {device} ",
                )
                for row in rows:
                    row["_point_index"] = point_index
                    row["_worker_rank"] = rank
                    row["_cuda_visible_device"] = device
                raw_rows.extend(rows)
                if grid_progress is not None:
                    grid_progress.update(1)
        finally:
            if grid_progress is not None:
                grid_progress.close()

        ensure_dir(partial.parent)
        pd.DataFrame(raw_rows).to_csv(partial, index=False)
        _write_yaml(
            status,
            {
                "ok": True,
                "rank": rank,
                "device": device,
                "point_indices": list(point_indices),
                "point_count": len(point_indices),
                "partial_path": str(partial),
                "workers_per_gpu": workers_per_gpu,
                "started_at": started_at,
                "finished_at": _now(),
            },
        )
    except BaseException as exc:
        _write_yaml(
            status,
            {
                "ok": False,
                "rank": rank,
                "device": device,
                "point_indices": list(point_indices),
                "workers_per_gpu": workers_per_gpu,
                "started_at": started_at,
                "finished_at": _now(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
