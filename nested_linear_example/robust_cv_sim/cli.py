from __future__ import annotations

import argparse
from pathlib import Path
import time

from .reporting import generate_outputs, write_raw_results
from .simulation import SimulationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the robust CV illustrative simulations.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(reps=args.reps, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for experiment in args.experiments:
        started = time.perf_counter()
        experiment_dir = args.output_dir / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
        raw_path = write_raw_results(experiment, config, experiment_dir)
        generate_outputs(raw_path, experiment, experiment_dir)
        elapsed = time.perf_counter() - started
        print(f"{experiment}: completed in {elapsed:.2f}s -> {experiment_dir}")


if __name__ == "__main__":
    main()
