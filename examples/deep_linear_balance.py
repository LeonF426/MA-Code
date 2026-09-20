"""Show when Gaussian S-SAM usefully balances an equivalent factorization.

The scalar two-layer model predicts ``u * v * x`` and is initialized at
``(u, v) = (10, 0.1)``. It therefore represents the target ``y = x`` exactly,
but is far more sensitive to parameter noise than the balanced representation.

For iid perturbations with variance ``eta**2``, the smoothed loss is exactly

    E[((u + e_u) * (v + e_v) - 1)**2]
      = (u*v - 1)**2 + eta**2 * (u**2 + v**2) + eta**4.

Thus SGD has zero gradient at the initialization, while S-SAM has an explicit
incentive to reduce the imbalance without changing the represented function
very much.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib
import torch
from torch.utils.data import TensorDataset

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from ssam import (
    build_model,
    evaluate_average_sharpness,
    evaluate_average_sharpness_interpolation,
    plot_sharpness_interpolation,
    train,
)


INITIAL_FACTORS = (10.0, 0.1)
TARGET_PRODUCT = 1.0

CONFIG = {
    "model": {
        "name": "two_factor_scalar",
        "type": "mixed_linear",
        "input_dim": 1,
        "layers": [
            {"type": "diag", "out_dim": 1, "activation": "identity"},
            {"type": "diag", "out_dim": 1},
        ],
        "output_activation": "identity",
        "output_reduction": "none",
        "bias": False,
        "parameter_init": {"type": "ones"},
    },
    "training": {
        "steps": 5000,
        "batch_size": 64,
        "learning_rate": {"name": "constant", "value": 1e-2},
        "sharpness_scale": {"name": "constant", "value": 0.2},
        "perturbation": {
            "distribution": "gaussian",
            "samples": 16,
            "normalized": False,
            "antithetic": True,
            "preserve_buffers": True,
        },
        "optimizer": {"name": "sgd", "momentum": 0.0},
        "loss": "mse",
        "checkpoint_every": 0,
        "seed": 37,
        "device": "cpu",
    },
    "visualization": {
        "output_dir": "outputs/deep_linear_balance",
        "evaluation": {
            "sharpness_scale": 0.2,
            "sharpness_samples": 4096,
            "sharpness_seed": 10023,
            "interpolation_points": 21,
            "interpolation_samples": 512,
        },
    },
}


def factor_values(model: torch.nn.Module) -> tuple[float, float]:
    """Return the two scalar diagonal weights."""

    return (
        float(model.layers[0].weight.detach().item()),
        float(model.layers[1].weight.detach().item()),
    )


def exact_clean_loss(first: float, second: float) -> float:
    """Return ``(first * second - 1)**2`` for the configured data."""

    return (first * second - TARGET_PRODUCT) ** 2


def exact_average_sharpness(first: float, second: float, scale: float) -> float:
    """Return the exact Gaussian average sharpness of the two-factor loss."""

    return scale**2 * (first**2 + second**2) + scale**4


def exact_regularized_loss(first: float, second: float, scale: float) -> float:
    """Return the exact Gaussian-smoothed loss."""

    return exact_clean_loss(first, second) + exact_average_sharpness(
        first,
        second,
        scale,
    )


def build_unbalanced_model(config=CONFIG) -> torch.nn.Module:
    """Construct the exactly fitting but unbalanced initial model."""

    model = build_model(config)
    with torch.no_grad():
        model.layers[0].weight.fill_(INITIAL_FACTORS[0])
        model.layers[1].weight.fill_(INITIAL_FACTORS[1])
    return model


def _trajectory_callback(storage, *, every: int, final_step: int, scale: float):
    def callback(step, model, record):
        del record
        if step % every and step != final_step:
            return
        first, second = factor_values(model)
        storage["step"].append(step + 1)
        storage["first_factor"].append(first)
        storage["second_factor"].append(second)
        storage["product"].append(first * second)
        storage["clean_loss"].append(exact_clean_loss(first, second))
        storage["exact_sharpness"].append(
            exact_average_sharpness(first, second, scale)
        )
        storage["imbalance"].append(abs(first**2 - second**2))

    return callback


def _plot_trajectories(trajectories, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = (
        ("clean_loss", "Clean loss", True),
        ("exact_sharpness", "Exact Gaussian sharpness", True),
        ("imbalance", "|u² - v²|", True),
        ("product", "Represented coefficient u*v", False),
    )
    for algorithm, trajectory in trajectories.items():
        for axis, (key, title, logarithmic) in zip(axes.flat, panels):
            axis.plot(trajectory["step"], trajectory[key], label=algorithm)
            if logarithmic and all(value > 0.0 for value in trajectory[key]):
                axis.set_yscale("log")
            axis.set(xlabel="Step", ylabel=title, title=title)
            axis.grid(alpha=0.25)
    axes[1, 1].axhline(TARGET_PRODUCT, color="black", linestyle=":", alpha=0.7)
    for axis in axes.flat:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    visualization = CONFIG["visualization"]
    output_dir = Path(visualization["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training = CONFIG["training"]
    evaluation = visualization["evaluation"]
    scale = float(evaluation["sharpness_scale"])
    steps = int(training["steps"])

    inputs = torch.ones((64, 1))
    targets = torch.ones((64, 1))
    dataset = TensorDataset(inputs, targets)
    initial_model = build_unbalanced_model()
    initial_state = copy.deepcopy(initial_model.state_dict())

    results = {}
    trajectories = {}
    for algorithm in ("sgd", "s_sam"):
        config = copy.deepcopy(CONFIG)
        config["training"]["algorithm"] = algorithm
        model = build_model(config)
        model.load_state_dict(initial_state, strict=True)
        trajectory = {
            "step": [0],
            "first_factor": [INITIAL_FACTORS[0]],
            "second_factor": [INITIAL_FACTORS[1]],
            "product": [INITIAL_FACTORS[0] * INITIAL_FACTORS[1]],
            "clean_loss": [exact_clean_loss(*INITIAL_FACTORS)],
            "exact_sharpness": [exact_average_sharpness(*INITIAL_FACTORS, scale)],
            "imbalance": [abs(INITIAL_FACTORS[0] ** 2 - INITIAL_FACTORS[1] ** 2)],
        }
        results[algorithm] = train(
            model,
            dataset,
            config,
            callbacks=[
                _trajectory_callback(
                    trajectory,
                    every=10,
                    final_step=steps - 1,
                    scale=scale,
                )
            ],
        )
        trajectories[algorithm] = trajectory

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    loss_function = torch.nn.MSELoss()
    metrics = {}
    for algorithm, result in results.items():
        first, second = factor_values(result.model)
        measured = evaluate_average_sharpness(
            result.model,
            loader,
            loss_function,
            scale,
            samples=int(evaluation["sharpness_samples"]),
            seed=int(evaluation["sharpness_seed"]),
            antithetic=True,
        )
        metrics[algorithm] = {
            "first_factor": first,
            "second_factor": second,
            "product": first * second,
            "imbalance": abs(first**2 - second**2),
            "clean_loss_exact": exact_clean_loss(first, second),
            "average_sharpness_exact": exact_average_sharpness(first, second, scale),
            "regularized_loss_exact": exact_regularized_loss(first, second, scale),
            "average_sharpness_monte_carlo": asdict(measured),
        }

    interpolation = evaluate_average_sharpness_interpolation(
        results["sgd"].model,
        results["s_sam"].model,
        loader,
        loss_function,
        scale,
        interpolation_points=int(evaluation["interpolation_points"]),
        samples=int(evaluation["interpolation_samples"]),
        seed=int(evaluation["sharpness_seed"]),
        antithetic=True,
    )

    theoretical_factor = math.sqrt(max(0.0, TARGET_PRODUCT - scale**2))
    report = {
        "initial_factors": INITIAL_FACTORS,
        "target_product": TARGET_PRODUCT,
        "sharpness_scale": scale,
        "theoretical_smoothed_optimum": {
            "first_factor": theoretical_factor,
            "second_factor": theoretical_factor,
            "product": theoretical_factor**2,
        },
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "sharpness_interpolation.json").write_text(
        json.dumps(asdict(interpolation), indent=2),
        encoding="utf-8",
    )
    _plot_trajectories(trajectories, output_dir / "balance_trajectory.png")
    plot_sharpness_interpolation(
        interpolation,
        output_dir / "sharpness_interpolation.png",
        endpoint_labels=("SGD", "S-SAM"),
    )

    for algorithm, values in metrics.items():
        print(
            f"{algorithm:>5} | factors=({values['first_factor']:.5f}, "
            f"{values['second_factor']:.5f}) | product={values['product']:.6f} | "
            f"clean={values['clean_loss_exact']:.3e} | "
            f"sharpness={values['average_sharpness_exact']:.3e}"
        )
    print(f"Artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
