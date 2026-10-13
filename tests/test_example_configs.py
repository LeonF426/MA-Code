from pathlib import Path
import runpy
import sys

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
