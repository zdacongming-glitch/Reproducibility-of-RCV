from __future__ import annotations

import argparse

from .aacv_experiment import run_aacv_and_write
from .aacv_mismatch import run_aacv_mismatch
from .aacv_plotting import draw_aacv_all
from .aacv_reference import freeze_reference_from_run
from .aacv_verify import verify_aacv_all
from .config import load_config, project_path
from .experiment import run_and_write
from .mismatch import run_mismatch
from .plotting import draw_all
from .verify import verify_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CCPP case-study reproducibility pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the CCPP experiment from raw data.")
    run.add_argument("--config", default="configs/ccpp_full.json")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--figure-dir", default=None)
    run.add_argument("--no-progress", action="store_true")
    run.add_argument("--skip-figures", action="store_true")

    plot = sub.add_parser(
        "plot-only", help="Regenerate manuscript figures from existing CSV outputs."
    )
    plot.add_argument("--config", default="configs/ccpp_full.json")
    plot.add_argument("--run-dir", required=True)
    plot.add_argument("--figure-dir", required=True)

    verify = sub.add_parser(
        "verify", help="Verify CSV signatures and manuscript figure outputs."
    )
    verify.add_argument("--config", default="configs/ccpp_full.json")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--figure-dir", required=True)
    verify.add_argument("--reference-figure-dir", default=None)

    freeze_reference = sub.add_parser(
        "freeze-aacv-reference",
        help="Freeze the global AACV working-reference selection from canonical SRCV.",
    )
    freeze_reference.add_argument(
        "--srcv-config", default="configs/ccpp_full.json"
    )
    freeze_reference.add_argument("--srcv-run-dir", required=True)
    freeze_reference.add_argument(
        "--output", default="configs/ccpp_aacv_reference.json"
    )

    run_aacv = sub.add_parser(
        "run-aacv", help="Run aligned adversarial CV from raw CCPP data."
    )
    run_aacv.add_argument("--config", default="configs/ccpp_aacv_full.json")
    run_aacv.add_argument("--output-dir", required=True)
    run_aacv.add_argument("--figure-dir", default=None)
    run_aacv.add_argument("--no-progress", action="store_true")
    run_aacv.add_argument("--skip-figures", action="store_true")
    run_aacv.add_argument("--resume", action="store_true")

    plot_aacv = sub.add_parser(
        "plot-aacv", help="Regenerate AACV figures from existing CSV outputs."
    )
    plot_aacv.add_argument("--config", default="configs/ccpp_aacv_full.json")
    plot_aacv.add_argument("--run-dir", required=True)
    plot_aacv.add_argument("--figure-dir", required=True)

    verify_aacv = sub.add_parser(
        "verify-aacv", help="Verify AACV tables, invariants, and figures."
    )
    verify_aacv.add_argument("--config", default="configs/ccpp_aacv_full.json")
    verify_aacv.add_argument("--run-dir", required=True)
    verify_aacv.add_argument("--figure-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-aacv-reference":
        srcv_config = load_config(args.srcv_config)
        output = project_path(args.output)
        spec = freeze_reference_from_run(
            srcv_config,
            project_path(args.srcv_run_dir),
            output,
        )
        print(
            "Froze AACV working reference "
            f"{spec['primary_working_reference_learner']} to {output}"
        )
        return 0

    config = load_config(args.config)
    if args.command == "run":
        output_dir = project_path(args.output_dir)
        run_and_write(config, output_dir, progress=not args.no_progress)
        run_mismatch(config, output_dir)
        if not args.skip_figures:
            figure_dir = (
                project_path(args.figure_dir)
                if args.figure_dir
                else output_dir / "manuscript_figures"
            )
            draw_all(config, output_dir, figure_dir)
            print(f"Wrote figures to {figure_dir}")
        print(f"Wrote run outputs to {output_dir}")
        return 0
    if args.command == "plot-only":
        run_dir = project_path(args.run_dir)
        figure_dir = project_path(args.figure_dir)
        run_mismatch(config, run_dir)
        draw_all(config, run_dir, figure_dir)
        print(f"Wrote figures to {figure_dir}")
        return 0
    if args.command == "verify":
        reference_dir = (
            project_path(args.reference_figure_dir)
            if args.reference_figure_dir
            else None
        )
        verify_all(
            config,
            project_path(args.run_dir),
            project_path(args.figure_dir),
            reference_dir,
        )
        return 0
    if args.command == "run-aacv":
        output_dir = project_path(args.output_dir)
        run_aacv_and_write(
            config,
            output_dir,
            progress=not args.no_progress,
            resume=args.resume,
        )
        run_aacv_mismatch(config, output_dir)
        if not args.skip_figures:
            figure_dir = (
                project_path(args.figure_dir)
                if args.figure_dir
                else output_dir / "figures"
            )
            draw_aacv_all(config, output_dir, figure_dir)
            print(f"Wrote AACV figures to {figure_dir}")
        print(f"Wrote AACV run outputs to {output_dir}")
        return 0
    if args.command == "plot-aacv":
        run_dir = project_path(args.run_dir)
        figure_dir = project_path(args.figure_dir)
        run_aacv_mismatch(config, run_dir)
        draw_aacv_all(config, run_dir, figure_dir)
        print(f"Wrote AACV figures to {figure_dir}")
        return 0
    if args.command == "verify-aacv":
        verify_aacv_all(
            config, project_path(args.run_dir), project_path(args.figure_dir)
        )
        return 0
    raise AssertionError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
