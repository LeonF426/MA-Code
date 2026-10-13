"""Run the Poisson PINN experiment repeatedly with different base seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from poisson_pinn import SEED, build_config, main as run_poisson_pinn


def seed_sequence(runs: int, start_seed: int, seed_step: int = 1) -> tuple[int, ...]:
    """Return the base seed used by each sequential experiment run."""

    if runs < 1:
        raise ValueError("runs must be positive")
    if seed_step == 0:
        raise ValueError("seed_step must be non-zero")
    return tuple(start_seed + index * seed_step for index in range(runs))


def run_many(
    runs: int,
    *,
    start_seed: int = SEED,
    seed_step: int = 1,
    run_name_prefix: str | None = "multi-seed",
) -> tuple[Path, ...]:
    """Run one otherwise-identical Poisson config for several base seeds."""

    seeds = seed_sequence(runs, start_seed, seed_step)
    output_directories = []
    for index, seed in enumerate(seeds, start=1):
        config = build_config(seed)
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
    return tuple(output_directories)


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


if __name__ == "__main__":
    main()
