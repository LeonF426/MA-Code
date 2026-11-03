import copy
import json

from examples.grid_search_poisson_sgd import (
    DEFAULT_GRID_SEARCH_CONFIG as SGD_GRID_SEARCH_CONFIG,
)
from examples.grid_search_poisson_ssam import (
    DEFAULT_GRID_SEARCH_CONFIG,
    ETA_SCHEDULES,
    run_grid_search,
)


def test_ssam_search_reuses_sgd_model_problem_and_optimizer_grid():
    ssam = DEFAULT_GRID_SEARCH_CONFIG
    sgd = SGD_GRID_SEARCH_CONFIG

    assert ssam["model"] == sgd["model"]
    assert ssam["setups"] == sgd["setups"]
    assert ssam["loss_weights"] == sgd["loss_weights"]
    for path in (
        "training.batch_size",
        "training.learning_rate",
        "training.optimizer",
    ):
        assert ssam["grid"][path] == sgd["grid"][path]
    assert ssam["training"]["algorithm"] == "s_sam"
    assert ssam["setups"][0]["problem"]["boundary_condition"] == {
        "name": "zero"
    }


def test_ssam_search_includes_multiple_eta_schedule_families():
    schedule_names = {schedule["name"] for schedule in ETA_SCHEDULES}

    assert schedule_names == {
        "constant",
        "inverse_time",
        "piecewise_constant",
    }
    assert sum(
        schedule["name"] == "piecewise_constant" for schedule in ETA_SCHEDULES
    ) == 1
    assert DEFAULT_GRID_SEARCH_CONFIG["grid"]["training.sharpness_scale"] == (
        ETA_SCHEDULES
    )
    candidate_count = 1
    for values in DEFAULT_GRID_SEARCH_CONFIG["grid"].values():
        candidate_count *= len(values)
    assert candidate_count == 36
    assert candidate_count <= DEFAULT_GRID_SEARCH_CONFIG["max_candidates"]
    assert len(DEFAULT_GRID_SEARCH_CONFIG["repeat_seeds"]) == 1


def test_ssam_grid_search_runs_adam_with_piecewise_constant_eta(tmp_path):
    config = copy.deepcopy(DEFAULT_GRID_SEARCH_CONFIG)
    config["output_dir"] = str(tmp_path)
    config["setups"][0]["data"].update(
        train_interior=4,
        train_boundary=4,
        validation_interior=4,
        validation_boundary=4,
        evaluation_resolution=3,
    )
    config["model"].update(depth=1, width=4)
    # Two updates cross the eta boundary while retaining Adam's moment state.
    config["training"].update(steps=2, batch_size=4, device="cpu")
    config["training"]["perturbation"].update(samples=2, antithetic=True)
    config["grid"] = {
        "training.batch_size": [4],
        "training.learning_rate": [1e-3],
        "training.optimizer": [
            {
                "name": "adam",
                "betas": [0.9, 0.99],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
            }
        ],
        "training.sharpness_scale": [
            {
                "name": "piecewise_constant",
                "boundaries": [1],
                "values": [0.01, 0.001],
            }
        ],
    }
    config["repeat_seeds"] = [7]

    output_dir = run_grid_search(config)

    results = json.loads(
        (output_dir / "grid_search_results.json").read_text(encoding="utf-8")
    )
    assert results["candidates"][0]["status"] == "complete"
    assert (output_dir / "optimal_ssam_config.json").is_file()
    assert (output_dir / "optimal_optimizer_config.json").is_file()
