from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import time

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .reporting import generate_outputs, merge_chunks, write_chunk
from .simulation import ChunkSpec, SimulationConfig, iter_experiment_chunks, run_chunk


console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chunked robust CV simulations with progress and resume.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["experiment_1", "experiment_2", "experiment_3"],
        choices=["experiment_1", "experiment_2", "experiment_3"],
        help="Experiments to run.",
    )
    parser.add_argument("--reps", type=int, default=1000, help="Monte Carlo replications per configuration.")
    parser.add_argument("--seed", type=int, default=20260319, help="Base seed for deterministic replication seeds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for raw results, summaries, and plots.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation and only write raw/summary outputs.")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep chunk CSV files after merging the final raw file.")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Resume from existing completed chunk files.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore existing chunk files and rerun everything.")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _chunk_path(experiment_dir: Path, chunk: ChunkSpec) -> Path:
    return experiment_dir / "chunks" / f"{chunk.chunk_id}.csv.gz"


def _is_completed_chunk(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _cleanup_stale_tmp(path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()


def _run_experiment(
    experiment: str,
    config: SimulationConfig,
    experiment_dir: Path,
    progress: Progress,
    overall_task: int,
    skip_plots: bool,
    keep_chunks: bool,
    resume: bool,
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    chunks = list(iter_experiment_chunks(experiment, config))
    chunk_paths = [_chunk_path(experiment_dir, chunk) for chunk in chunks]
    completed = sum(1 for path in chunk_paths if resume and _is_completed_chunk(path))

    experiment_task = progress.add_task(f"[cyan]{experiment}", total=len(chunks), completed=completed)
    current_chunk_task = progress.add_task("[yellow]current chunk", total=config.reps, completed=0)

    for chunk, chunk_path in zip(chunks, chunk_paths):
        if resume and _is_completed_chunk(chunk_path):
            progress.advance(overall_task, 1)
            continue

        _cleanup_stale_tmp(chunk_path)
        progress.update(
            current_chunk_task,
            total=config.reps,
            completed=0,
            description=f"[yellow]{chunk.describe()}",
        )

        chunk_df = run_chunk(
            chunk,
            config,
            progress_callback=lambda completed_reps: progress.update(current_chunk_task, completed=completed_reps),
        )
        write_chunk(chunk_df, chunk_path)
        progress.advance(experiment_task, 1)
        progress.advance(overall_task, 1)

    progress.update(current_chunk_task, description=f"[green]{experiment} chunk loop complete", completed=config.reps)
    raw_path = merge_chunks(experiment_dir)
    generate_outputs(raw_path, experiment, experiment_dir, skip_plots=skip_plots)
    if not keep_chunks:
        shutil.rmtree(experiment_dir / "chunks", ignore_errors=True)
    progress.remove_task(current_chunk_task)
    progress.remove_task(experiment_task)


def main() -> None:
    args = parse_args()
    config = SimulationConfig(reps=args.reps, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chunk_counts = {experiment: len(list(iter_experiment_chunks(experiment, config))) for experiment in args.experiments}
    total_chunks = sum(chunk_counts.values())

    started = time.perf_counter()
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        overall_task = progress.add_task("[bold green]overall", total=total_chunks, completed=0)
        for experiment in args.experiments:
            experiment_dir = args.output_dir / experiment
            _run_experiment(
                experiment=experiment,
                config=config,
                experiment_dir=experiment_dir,
                progress=progress,
                overall_task=overall_task,
                skip_plots=args.skip_plots,
                keep_chunks=args.keep_chunks,
                resume=args.resume,
            )

    elapsed = time.perf_counter() - started
    console.print(f"Completed {', '.join(args.experiments)} in {elapsed:.2f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
