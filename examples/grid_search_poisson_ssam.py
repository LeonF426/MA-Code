"""Grid-search S-SAM on the zero-boundary Poisson PINN.

This search intentionally reuses the model, data, loss, optimizer-momentum,
batch-size, and learning-rate setup from ``grid_search_poisson_sgd.py``.  The
only additional search dimension is the S-SAM sharpness scale ``eta`` schedule.

The explicit ``piecewise_constant`` candidate holds eta fixed for long
plateaus.  It changes eta only after 1,000
updates, giving Adam's moment estimates time to adapt to one smoothed objective
before the next objective is introduced.

Examples (run from the repository root)::

    python examples/grid_search_poisson_ssam.py --smoke-test
    python examples/grid_search_poisson_ssam.py --write-default-config search.json
    python examples/grid_search_poisson_ssam.py --config search.json
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from examples.grid_search_poisson_sgd import (
        DEFAULT_GRID_SEARCH_CONFIG as SGD_GRID_SEARCH_CONFIG,
        SEED,
        _deep_merge,
        _smoke_config as _base_smoke_config,
        _write_json,
        run_grid_search,
    )
except ModuleNotFoundError:  # Direct execution from the examples directory.
    from grid_search_poisson_sgd import (  # type: ignore[no-redef]
        DEFAULT_GRID_SEARCH_CONFIG as SGD_GRID_SEARCH_CONFIG,
        SEED,
        _deep_merge,
        _smoke_config as _base_smoke_config,
        _write_json,
        run_grid_search,
    )


ETA_SCHEDULES: list[dict[str, Any]] = [
    {"name": "constant", "value": 0.1},
    {"name": "inverse_time", "initial": 1.0, "power": 0.5, "floor": 0.0},
    {
        "name": "piecewise_constant",
        "boundaries": [1_000],
        "values": [1.0, 0.1],
    },
]


def build_config() -> dict[str, Any]:
    """Return the S-SAM search while preserving the matched SGD setup."""

    config = copy.deepcopy(SGD_GRID_SEARCH_CONFIG)
    config["output_dir"] = "outputs/poisson_ssam_grid_search"
    config["training"].update(
        algorithm="s_sam",
        sharpness_scale=copy.deepcopy(ETA_SCHEDULES[1]),
        perturbation={
            "distribution": "gaussian",
            "samples": 20,
            "normalized": False,
            "antithetic": True,
            "preserve_buffers": True,
            "seed": SEED + 6,
        },
    )
    config["grid"]["training.sharpness_scale"] = copy.deepcopy(ETA_SCHEDULES)
    return config


DEFAULT_GRID_SEARCH_CONFIG = build_config()


def _smoke_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = _base_smoke_config(config)
    result["training"]["perturbation"]["samples"] = 2
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON overrides for the default config")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny two-step trial")
    parser.add_argument(
        "--write-default-config",
        type=Path,
        metavar="PATH",
        help="Write an editable JSON config and exit",
    )
    return parser.parse_args()


def main() -> Path | None:
    args = parse_args()
    if args.write_default_config is not None:
        args.write_default_config.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.write_default_config, DEFAULT_GRID_SEARCH_CONFIG)
        print(f"Wrote {args.write_default_config}")
        return None
    config = copy.deepcopy(DEFAULT_GRID_SEARCH_CONFIG)
    if args.config is not None:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(overrides, Mapping):
            raise TypeError("The config JSON root must be an object")
        config = _deep_merge(config, overrides)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.smoke_test:
        config = _smoke_config(config)
    return run_grid_search(config)


if __name__ == "__main__":
    main()
