"""Run the Poisson PINN experiment repeatedly with different base seeds."""
#  python .\examples\run_poisson_pinn_seeds.py --runs <N> --start-seed <SEED>
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from poisson_pinn import SEED, build_config, main as run_poisson_pinn
from ssam import plot_poisson_forcing_sensitivity_average


FORCING_MODES = ("scaled", "additive")


def seed_sequence(runs: int, start_seed: int, seed_step: int = 1) -> tuple[int, ...]:
    """Return the base seed used by each sequential experiment run."""

    if runs < 1:
        raise ValueError("runs must be positive")
    if seed_step == 0:
        raise ValueError("seed_step must be non-zero")
    return tuple(start_seed + index * seed_step for index in range(runs))


def _sample_summary(values) -> dict[str, float]:
    values = tuple(float(value) for value in values)
    mean = sum(values) / len(values)
    standard_deviation = (
        math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        if len(values) > 1
        else 0.0
    )
    return {"mean": mean, "standard_deviation": standard_deviation}


def write_sensitivity_summary(
    output_directories: tuple[Path, ...],
    seeds: tuple[int, ...],
) -> Path:
    """Write separate scaled/additive averages over all completed seed runs."""

    if not output_directories:
        raise ValueError("At least one output directory is required")
    if len(output_directories) != len(seeds):
        raise ValueError("Every output directory must have one corresponding seed")

    sensitivity_runs: dict[str, dict[str, list[dict]]] = {
        mode: {} for mode in FORCING_MODES
    }
    for output_directory in output_directories:
        metrics_path = output_directory / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for algorithm, algorithm_metrics in metrics["algorithms"].items():
            by_mode = algorithm_metrics.get("forcing_sensitivities", {})
            for mode in FORCING_MODES:
                if mode not in by_mode:
                    raise ValueError(
                        f"{metrics_path} does not contain {mode} sensitivity"
                    )
                sensitivity_runs[mode].setdefault(algorithm, []).append(
                    by_mode[mode]
                )

    expected_runs = len(output_directories)
    for mode, algorithms in sensitivity_runs.items():
        for algorithm, runs in algorithms.items():
            if len(runs) != expected_runs:
                raise ValueError(
                    f"Algorithm {algorithm!r} has {len(runs)}/{expected_runs} "
                    f"runs for {mode} sensitivity"
                )

    summary_directory = output_directories[-1] / "multi_seed_summary"
    summary_directory.mkdir()
    summary_metrics = {
        "seeds": list(seeds),
        "source_run_directories": [
            str(output_directory.resolve())
            for output_directory in output_directories
        ],
        "forcing_sensitivities": {},
    }
    metric_names = (
        "boundary_condition_rmse",
        "nominal_normalized_l2_error",
        "pde_residual_rmse",
    )
    from matplotlib import pyplot as plt

    for mode, algorithms in sensitivity_runs.items():
        figure = plot_poisson_forcing_sensitivity_average(
            algorithms,
            summary_directory / f"forcing_sensitivity_{mode}_average.png",
        )
        plt.close(figure)
        mode_summary = {}
        for algorithm, runs in algorithms.items():
            sorted_evaluations = [
                sorted(
                    run["evaluations"],
                    key=lambda evaluation: evaluation["forcing_perturbation"],
                )
                for run in runs
            ]
            perturbation_grids = [
                tuple(
                    float(evaluation["forcing_perturbation"])
                    for evaluation in evaluations
                )
                for evaluations in sorted_evaluations
            ]
            if any(grid != perturbation_grids[0] for grid in perturbation_grids[1:]):
                raise ValueError(
                    "All forcing sensitivities must use the same perturbation grid"
                )
            aggregate_evaluations = []
            for index, perturbation in enumerate(perturbation_grids[0]):
                aggregate_evaluations.append(
                    {
                        "forcing_perturbation": perturbation,
                        **{
                            metric_name: _sample_summary(
                                evaluations[index][metric_name]
                                for evaluations in sorted_evaluations
                            )
                            for metric_name in metric_names
                        },
                    }
                )
            mode_summary[algorithm] = {
                "run_count": len(runs),
                "nominal_forcing_perturbation": float(
                    runs[0]["nominal_forcing_perturbation"]
                ),
                "evaluations": aggregate_evaluations,
            }
        summary_metrics["forcing_sensitivities"][mode] = mode_summary

    (summary_directory / "metrics.json").write_text(
        json.dumps(summary_metrics, indent=2),
        encoding="utf-8",
    )
    print(
        "\nMulti-seed sensitivity averages written to "
        f"{summary_directory.resolve()}"
    )
    return summary_directory


def run_many(
    runs: int,
    *,
    start_seed: int = SEED,
    seed_step: int = 1,
    run_name_prefix: str | None = "multi-seed",
) -> tuple[Path, ...]:
    """Run several seeds, then average both RHS sensitivity families."""

    seeds = seed_sequence(runs, start_seed, seed_step)
    output_directories = []
    for index, seed in enumerate(seeds, start=1):
        config = build_config(seed)
        config["pinn"]["evaluation"]["forcing_uncertainty"]["modes"] = list(
            FORCING_MODES
        )
        if run_name_prefix:
            config["run"]["name"] = f"{run_name_prefix}-seed-{seed}"
        print(f"\n=== Poisson seed run {index}/{len(seeds)}: base seed {seed} ===")
        try:
            output_directories.append(run_poisson_pinn(config))
        finally:
            # The individual example returns figures for interactive use. Close
            # them between repetitions so large sweeps do not retain every plot.
            from matplotlib import pyplot as plt

            plt.close("all")
    completed_directories = tuple(output_directories)
    write_sensitivity_summary(completed_directories, seeds)
    return completed_directories


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        required=True,
        help="number of Poisson runs",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=SEED,
        help=f"base seed for the first run (default: {SEED})",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="integer increment between base seeds (default: 1)",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="multi-seed",
        help="artifact run-name prefix; pass an empty string for no prefix",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        outputs = run_many(
            args.runs,
            start_seed=args.start_seed,
            seed_step=args.seed_step,
            run_name_prefix=args.run_name_prefix or None,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("\nCompleted run directories:")
    for output in outputs:
        print(output.resolve())
    print("Final averaged sensitivity figures:")
    print((outputs[-1] / "multi_seed_summary").resolve())


if __name__ == "__main__":
    main()
