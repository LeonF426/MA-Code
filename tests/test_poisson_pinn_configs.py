import copy
import json

from examples import poisson_pinn


def test_grid_search_winner_is_resolved_as_poisson_experiment_config():
    winner = {
        "best_overall": {
            "candidate_id": 7,
            "resolved_config": {
                "model": {
                    "width": 32,
                    "parameter_init": {
                        "type": "normal",
                        "mean": 0.0,
                        "std": 0.1,
                    },
                },
                "training": {
                    "algorithm": "s_sam",
                    "batch_size": 64,
                    "learning_rate": 1e-3,
                    "optimizer": {
                        "name": "adam",
                        "betas": [0.9, 0.99],
                        "eps": 1e-8,
                    },
                },
                "loss_weights": {"pde": 2.0, "boundary": 3.0},
            },
        }
    }

    resolved = poisson_pinn._resolve_experiment_config(winner)

    assert resolved["model"]["width"] == 32
    assert resolved["model"]["parameter_init"] == {
        "type": "normal",
        "mean": 0.0,
        "std": 0.1,
    }
    assert resolved["training"]["batch_size"] == 64
    assert resolved["training"]["optimizer"] == {
        "name": "adam",
        "betas": [0.9, 0.99],
        "eps": 1e-8,
    }
    assert resolved["pinn"]["pde_weight"] == 2.0
    assert resolved["pinn"]["boundary_weight"] == 3.0
    assert resolved["pinn"]["algorithms"] == ["s_sam"]
    assert resolved["run"]["name"] == "grid-s_sam-candidate-7"


def test_full_poisson_config_is_copied_without_modification():
    config = poisson_pinn.build_config(123)

    resolved = poisson_pinn._resolve_experiment_config(config)

    assert resolved == config
    assert resolved is not config


def test_grid_search_config_can_be_loaded_from_json_path(tmp_path):
    path = tmp_path / "optimal_ssam_config.json"
    path.write_text(
        json.dumps(
            {
                "best_overall": {
                    "candidate_id": 2,
                    "resolved_config": {
                        "training": {
                            "algorithm": "s_sam",
                            "batch_size": 25,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    resolved = poisson_pinn._resolve_experiment_config(path)

    assert resolved["training"]["batch_size"] == 25
    assert resolved["pinn"]["algorithms"] == ["s_sam"]


def test_grid_search_config_can_be_pasted_as_raw_json():
    raw_json = json.dumps(
        {
            "resolved_config": {
                "training": {"algorithm": "sgd", "batch_size": 50}
            }
        }
    )

    resolved = poisson_pinn._resolve_experiment_config(raw_json)

    assert resolved["training"]["batch_size"] == 50
    assert resolved["training"]["algorithm"] == "sgd"
    assert resolved["pinn"]["algorithms"] == ["sgd"]


def test_grid_search_algorithm_overrides_inherited_algorithm_comparison():
    winner = {
        "resolved_config": {
            "training": {"algorithm": "ssam"},
            "pinn": {"algorithms": ["sgd", "s_sam"]},
        }
    }

    resolved = poisson_pinn._resolve_experiment_config(winner)

    assert resolved["training"]["algorithm"] == "s_sam"
    assert resolved["pinn"]["algorithms"] == ["s_sam"]


def test_run_configs_runs_every_list_item(monkeypatch, tmp_path):
    configs = [
        {"run": {"name": "first"}},
        {"run": {"name": "second"}},
    ]
    seen = []

    def fake_main(config):
        seen.append(copy.deepcopy(config))
        return tmp_path / config["run"]["name"]

    monkeypatch.setattr(poisson_pinn, "main", fake_main)

    outputs = poisson_pinn.run_configs(configs)

    assert seen == configs
    assert outputs == (tmp_path / "first", tmp_path / "second")
