from pathlib import Path
import json
import runpy
import sys

import pytest

from ssam import build_learning_rate_policy, build_schedule, normalize_config


def _experiment_configs(namespace):
    for name, value in namespace.items():
        if (
            (name.startswith("CONFIG") or name.startswith("BASE_CONFIG"))
            and isinstance(value, dict)
            and "model" in value
            and "training" in value
        ):
            yield name, value


def test_every_example_obeys_the_reproducibility_config_contract():
    root = Path(__file__).parents[1]
    example_dir = root / "examples"
    sys.path.insert(0, str(example_dir))
    try:
        paths = [root / "main.py", root / "model_configs.py"]
        paths.extend(sorted(example_dir.glob("*.py")))
        seen = 0
        for path in paths:
            namespace = runpy.run_path(str(path))
            for name, config in _experiment_configs(namespace):
                seen += 1
                initialization = config["model"]["parameter_init"]
                assert "seed" in initialization, f"{path.name}:{name} init seed"
                training = config["training"]
                assert "seed" in training, f"{path.name}:{name} training seed"
                assert "batch_seed" in training, f"{path.name}:{name} batch seed"
                if "perturbation" in training:
                    assert "seed" in training["perturbation"], (
                        f"{path.name}:{name} perturbation seed"
                    )
                assert "type" not in training["learning_rate"], (
                    f"{path.name}:{name} vestigial learning-rate type"
                )

                resolved = normalize_config(config)
                build_schedule(resolved["training"]["sharpness_scale"])
                build_learning_rate_policy(
                    resolved["training"]["learning_rate"],
                    default_dimension=int(resolved["model"].get("input_dim", 1)),
                    default_depth=max(
                        1,
                        int(
                            resolved["model"].get(
                                "depth",
                                len(resolved["model"].get("layers", [])) or 1,
                            )
                        ),
                    ),
                )
        assert seen >= 10
    finally:
        sys.path.remove(str(example_dir))


def test_poisson_config_derives_every_rng_seed_from_one_base_seed():
    path = Path(__file__).parents[1] / "examples" / "poisson_pinn.py"
    namespace = runpy.run_path(str(path))
    first = namespace["build_config"](100)
    second = namespace["build_config"](110)

    def seeds(config):
        return (
            config["model"]["parameter_init"]["rescaling"]["seed"],
            config["model"]["parameter_init"]["seed"],
            config["data"]["seed"],
            config["pinn"]["evaluation"]["sharpness_seed"],
            config["pinn"]["evaluation"]["laplace"]["seed"],
            config["training"]["perturbation"]["seed"],
            config["training"]["seed"],
            config["training"]["batch_seed"],
        )

    assert seeds(first) == tuple(range(101, 109))
    assert seeds(second) == tuple(value + 10 for value in seeds(first))


def test_poisson_multi_seed_runner_builds_requested_sequence():
    example_dir = Path(__file__).parents[1] / "examples"
    sys.path.insert(0, str(example_dir))
    try:
        namespace = runpy.run_path(str(example_dir / "run_poisson_pinn_seeds.py"))
    finally:
        sys.path.remove(str(example_dir))

    assert namespace["seed_sequence"](4, 20, 3) == (20, 23, 26, 29)


def test_poisson_multi_seed_runner_requests_both_modes(monkeypatch, tmp_path):
    example_dir = Path(__file__).parents[1] / "examples"
    sys.path.insert(0, str(example_dir))
    try:
        namespace = runpy.run_path(str(example_dir / "run_poisson_pinn_seeds.py"))
    finally:
        sys.path.remove(str(example_dir))

    seen_modes = []
    seen_summary = []

    def fake_run(config):
        seen_modes.append(
            config["pinn"]["evaluation"]["forcing_uncertainty"]["modes"]
        )
        output = tmp_path / f"run-{len(seen_modes)}"
        output.mkdir()
        return output

    def fake_summary(outputs, seeds):
        seen_summary.append((outputs, seeds))
        return outputs[-1] / "multi_seed_summary"

    run_many = namespace["run_many"]
    monkeypatch.setitem(run_many.__globals__, "run_poisson_pinn", fake_run)
    monkeypatch.setitem(
        run_many.__globals__,
        "write_sensitivity_summary",
        fake_summary,
    )

    outputs = run_many(2, start_seed=10, seed_step=2)

    assert seen_modes == [["scaled", "additive"], ["scaled", "additive"]]
    assert seen_summary == [(outputs, (10, 12))]


def test_poisson_multi_seed_runner_averages_both_sensitivity_families(tmp_path):
    example_dir = Path(__file__).parents[1] / "examples"
    sys.path.insert(0, str(example_dir))
    try:
        namespace = runpy.run_path(str(example_dir / "run_poisson_pinn_seeds.py"))
    finally:
        sys.path.remove(str(example_dir))

    def sensitivity(mode, offset):
        perturbations = [0.9, 1.0, 1.1] if mode == "scaled" else [-0.1, 0.0, 0.1]
        return {
            "forcing_mode": mode,
            "nominal_forcing_perturbation": 1.0 if mode == "scaled" else 0.0,
            "evaluations": [
                {
                    "forcing_perturbation": perturbation,
                    "boundary_condition_rmse": offset + 0.1,
                    "nominal_normalized_l2_error": offset + index,
                    "pde_residual_rmse": offset + 2.0 * index,
                }
                for index, perturbation in enumerate(perturbations)
            ],
        }

    outputs = []
    for run_index, offset in enumerate((1.0, 3.0)):
        output = tmp_path / f"run-{run_index}"
        output.mkdir()
        payload = {
            "algorithms": {
                "sgd": {
                    "forcing_sensitivities": {
                        mode: sensitivity(mode, offset)
                        for mode in ("scaled", "additive")
                    }
                }
            }
        }
        (output / "metrics.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        outputs.append(output)

    summary = namespace["write_sensitivity_summary"](
        tuple(outputs),
        (101, 102),
    )
    aggregated = json.loads((summary / "metrics.json").read_text(encoding="utf-8"))

    assert (summary / "forcing_sensitivity_scaled_average.png").is_file()
    assert (summary / "forcing_sensitivity_additive_average.png").is_file()
    scaled = aggregated["forcing_sensitivities"]["scaled"]["sgd"]
    assert scaled["run_count"] == 2
    assert scaled["evaluations"][0]["nominal_normalized_l2_error"] == {
        "mean": 2.0,
        "standard_deviation": pytest.approx(2.0**0.5),
    }
