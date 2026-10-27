import json

import torch

from examples.grid_search_poisson_sgd import (
    DEFAULT_GRID_SEARCH_CONFIG,
    expand_grid,
    manufactured_sine_problem,
    poisson_residual,
    run_grid_search,
)


def test_manufactured_solution_satisfies_configured_rhs_and_boundary():
    setup = {
        "name": "nonzero_boundary",
        "problem": {
            "amplitude": 0.75,
            "modes": [2, 1],
            "affine": [0.1, 0.2, -0.1],
        },
    }
    problem = manufactured_sine_problem(setup)

    class ExactModel(torch.nn.Module):
        def forward(self, points):
            return problem.exact_solution(points).unsqueeze(1)

    points = torch.tensor([[0.2, 0.3], [0.6, 0.7]], dtype=torch.float64)
    residual = poisson_residual(
        ExactModel(), points, problem.forcing, create_graph=False
    )

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)
    boundary = torch.tensor([[0.0, 0.4], [1.0, 0.6]], dtype=torch.float64)
    assert torch.allclose(
        problem.exact_solution(boundary),
        problem.boundary_value(boundary),
    )


def test_expand_grid_sets_nested_training_and_model_values():
    config = {
        "model": {"width": 8},
        "training": {"learning_rate": 1e-3, "optimizer": {"momentum": 0.0}},
        "loss_weights": {"pde": 1.0, "boundary": 1.0},
        "grid": {
            "training.learning_rate": [1e-3, 1e-2],
            "model.width": [8, 16],
        },
    }

    candidates = expand_grid(config)

    assert len(candidates) == 4
    assert {candidate[1]["model"]["width"] for candidate in candidates} == {8, 16}
    assert {
        candidate[1]["training"]["learning_rate"] for candidate in candidates
    } == {1e-3, 1e-2}


def test_grid_search_writes_optimal_and_complete_results(tmp_path):
    config = {
        **DEFAULT_GRID_SEARCH_CONFIG,
        "output_dir": str(tmp_path),
        "setups": [
            {
                "name": "tiny",
                "problem": {
                    "amplitude": 1.0,
                    "modes": [1, 1],
                    "affine": [0.0, 0.0, 0.0],
                },
                "data": {
                    "train_interior": 4,
                    "train_boundary": 4,
                    "validation_interior": 4,
                    "validation_boundary": 4,
                    "evaluation_resolution": 3,
                    "dtype": "float32",
                    "seed": 3,
                },
            }
        ],
        "model": {
            **DEFAULT_GRID_SEARCH_CONFIG["model"],
            "depth": 1,
            "width": 4,
        },
        "training": {
            **DEFAULT_GRID_SEARCH_CONFIG["training"],
            "steps": 1,
            "batch_size": 4,
            "device": "cpu",
        },
        "grid": {"training.learning_rate": [1e-3]},
        "repeat_seeds": [7],
    }

    output_dir = run_grid_search(config)

    optimal = json.loads(
        (output_dir / "optimal_sgd_config.json").read_text(encoding="utf-8")
    )
    results = json.loads(
        (output_dir / "grid_search_results.json").read_text(encoding="utf-8")
    )
    assert optimal["best_overall"]["candidate_id"] == 0
    assert optimal["best_per_setup"]["tiny"]["candidate_id"] == 0
    assert results["candidates"][0]["status"] == "complete"
    assert (output_dir / "optimal_optimizer_config.json").is_file()
    assert (output_dir / "trials.csv").is_file()


def test_grid_search_compares_sgd_adam_and_adagrad(tmp_path):
    config = {
        **DEFAULT_GRID_SEARCH_CONFIG,
        "output_dir": str(tmp_path),
        "setups": [
            {
                "name": "optimizer_comparison",
                "problem": {
                    "amplitude": 1.0,
                    "modes": [1, 1],
                    "boundary_condition": {"name": "constant", "value": 0.1},
                },
                "data": {
                    "train_interior": 4,
                    "train_boundary": 4,
                    "validation_interior": 4,
                    "validation_boundary": 4,
                    "evaluation_resolution": 3,
                    "dtype": "float32",
                    "seed": 3,
                },
            }
        ],
        "model": {
            **DEFAULT_GRID_SEARCH_CONFIG["model"],
            "depth": 2,
            "width": 4,
        },
        "training": {
            **DEFAULT_GRID_SEARCH_CONFIG["training"],
            "steps": 1,
            "batch_size": 4,
            "device": "cpu",
        },
        "grid": {
            "training.learning_rate": [1e-3],
            "training.optimizer": [
                {"name": "sgd", "momentum": 0.0},
                {"name": "adam", "betas": [0.9, 0.999]},
                {"name": "adagrad", "lr_decay": 0.0},
            ],
        },
        "repeat_seeds": [7],
    }

    output_dir = run_grid_search(config)
    results = json.loads(
        (output_dir / "grid_search_results.json").read_text(encoding="utf-8")
    )

    optimizer_names = {
        candidate["resolved_config"]["training"]["optimizer"]["name"]
        for candidate in results["candidates"]
    }
    assert optimizer_names == {
        "sgd",
        "adam",
        "adagrad",
    }
    assert all(candidate["status"] == "complete" for candidate in results["candidates"])
