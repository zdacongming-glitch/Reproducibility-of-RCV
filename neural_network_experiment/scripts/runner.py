from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.plots.make_plots import make_all_plots
from src.utils import ensure_dir


DEFAULT_CONFIGS = {
    "smoke": REPO_ROOT / "configs" / "experiments" / "smoke.yaml",
}

SCALED_CONFIGS = {
    "main": {
        "lite": REPO_ROOT / "configs" / "experiments" / "main_lite.yaml",
        "medium": REPO_ROOT / "configs" / "experiments" / "main_medium.yaml",
        "full": REPO_ROOT / "configs" / "experiments" / "main_full.yaml",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run robust CV regression experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("smoke", "main"):
        subparser = subparsers.add_parser(command)
        default_config = DEFAULT_CONFIGS.get(command)
        subparser.add_argument("--config", type=Path, default=default_config)
        subparser.add_argument("--output-dir", type=Path, default=None)
        subparser.add_argument("--run-name", type=str, default=None)
        subparser.add_argument("--workers", type=int, default=None)
        if command in SCALED_CONFIGS:
            subparser.add_argument("--scale", choices=["lite", "medium", "full"], default="medium")

    plots_parser = subparsers.add_parser("plots")
    plots_parser.add_argument("--config", type=Path, default=SCALED_CONFIGS["main"]["medium"])
    plots_parser.add_argument("--input-dir", type=Path, required=True)
    plots_parser.add_argument("--output-dir", type=Path, default=None)

    multi_gpu_parser = subparsers.add_parser("multi-gpu", help="Run an experiment grid across multiple GPUs.")
    multi_gpu_subparsers = multi_gpu_parser.add_subparsers(dest="multi_gpu_command", required=True)
    for command in ("smoke", "main"):
        subparser = multi_gpu_subparsers.add_parser(command)
        default_config = DEFAULT_CONFIGS.get(command)
        subparser.add_argument("--config", type=Path, default=default_config)
        subparser.add_argument("--output-dir", type=Path, default=None)
        subparser.add_argument("--run-name", type=str, default=None)
        subparser.add_argument("--devices", type=str, required=True, help="Comma-separated CUDA device ids, e.g. 0,1.")
        subparser.add_argument("--workers-per-gpu", type=int, default=1)
        if command in SCALED_CONFIGS:
            subparser.add_argument("--scale", choices=["lite", "medium", "full"], default="medium")

    worker_parser = subparsers.add_parser("_multi-gpu-worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--config", type=Path, required=True)
    worker_parser.add_argument("--partial-path", type=Path, required=True)
    worker_parser.add_argument("--status-path", type=Path, required=True)
    worker_parser.add_argument("--point-indices", type=str, required=True)
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--device", type=str, required=True)
    worker_parser.add_argument("--workers-per-gpu", type=int, required=True)
    worker_parser.add_argument("--total-points", type=int, required=True)
    worker_parser.add_argument("--show-progress", action="store_true")
    return parser


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_output_dir(config, override: Path | None) -> Path:
    if override is not None:
        return ensure_dir(override)
    run_name = config.experiment.run_name or config.experiment.name
    return ensure_dir(REPO_ROOT / config.experiment.output_root / f"{run_name}_{_timestamp()}")


def resolve_config_path(command: str, config_path: Path | None, scale: str | None) -> Path:
    if config_path is not None:
        return config_path
    if command in SCALED_CONFIGS:
        assert scale is not None
        return SCALED_CONFIGS[command][scale]
    return DEFAULT_CONFIGS[command]


def run_command(
    command: str,
    config_path: Path | None,
    output_dir: Path | None,
    run_name: str | None,
    workers: int | None,
    scale: str | None,
) -> None:
    from src.experiments.pipeline import run_experiment_grid

    config = load_config(resolve_config_path(command, config_path, scale))
    if run_name:
        config.experiment.run_name = run_name
    if workers is not None:
        config.experiment.parallel_workers = workers
    target_dir = resolve_output_dir(config, output_dir)
    run_experiment_grid(config=config, output_dir=target_dir, show_progress=True)
    print(f"Finished '{command}' experiment. Outputs written to: {target_dir}")


def run_multi_gpu_command(
    command: str,
    config_path: Path | None,
    output_dir: Path | None,
    run_name: str | None,
    scale: str | None,
    devices: str,
    workers_per_gpu: int,
) -> None:
    from src.experiments.multi_gpu import parse_devices, run_multi_gpu_grid

    parsed_devices = parse_devices(devices)
    if workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be at least 1.")

    config = load_config(resolve_config_path(command, config_path, scale))
    if run_name:
        config.experiment.run_name = run_name
    target_dir = resolve_output_dir(config, output_dir)
    run_multi_gpu_grid(
        config=config,
        output_dir=target_dir,
        devices=parsed_devices,
        workers_per_gpu=workers_per_gpu,
        show_progress=True,
        runner_path=Path(__file__).resolve(),
    )
    print(f"Finished multi-GPU '{command}' experiment. Outputs written to: {target_dir}")


def run_plot_command(config_path: Path, input_dir: Path, output_dir: Path | None) -> None:
    config = load_config(config_path)
    target_dir = ensure_dir(output_dir or input_dir / "plots")
    aggregated_df = pd.read_csv(input_dir / "aggregated_metrics.csv")
    selection_path = input_dir / "selection_frequencies.csv"
    raw_path = input_dir / "raw_metrics.csv"
    selection_df = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()
    raw_df = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    make_all_plots(aggregated_df=aggregated_df, selection_df=selection_df, raw_df=raw_df, output_dir=target_dir, config=config)
    print(f"Plots written to: {target_dir}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plots":
        run_plot_command(config_path=args.config, input_dir=args.input_dir, output_dir=args.output_dir)
        return
    if args.command == "multi-gpu":
        try:
            run_multi_gpu_command(
                command=args.multi_gpu_command,
                config_path=args.config,
                output_dir=args.output_dir,
                run_name=args.run_name,
                scale=getattr(args, "scale", None),
                devices=args.devices,
                workers_per_gpu=args.workers_per_gpu,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.command == "_multi-gpu-worker":
        from src.experiments.multi_gpu import parse_point_indices, run_multi_gpu_worker

        run_multi_gpu_worker(
            config_path=args.config,
            partial_path=args.partial_path,
            status_path=args.status_path,
            point_indices=parse_point_indices(args.point_indices),
            rank=args.rank,
            device=args.device,
            workers_per_gpu=args.workers_per_gpu,
            total_points=args.total_points,
            show_progress=args.show_progress,
        )
        return
    run_command(
        command=args.command,
        config_path=args.config,
        output_dir=args.output_dir,
        run_name=args.run_name,
        workers=getattr(args, "workers", None),
        scale=getattr(args, "scale", None),
    )


if __name__ == "__main__":
    main()
