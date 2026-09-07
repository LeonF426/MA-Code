"""Train one or more repository algorithms on California Housing."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ssam import build_dataset, build_model, plot_training_history, train
from model_config import BASE_CONFIG, config_for, parse_args, model_config


def regression_metrics(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Evaluate mean squared error, RMSE, MAE, and R-squared."""

    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    count = 0
    squared_error = 0.0
    absolute_error = 0.0
    target_sum = 0.0
    target_squared_sum = 0.0
    try:
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                if predictions.ndim == targets.ndim + 1 and predictions.shape[-1] == 1:
                    predictions = predictions.squeeze(-1)
                errors = predictions - targets
                count += targets.numel()
                squared_error += float(errors.square().sum().item())
                absolute_error += float(errors.abs().sum().item())
                target_sum += float(targets.sum().item())
                target_squared_sum += float(targets.square().sum().item())
    finally:
        model.train(was_training)

    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    mse = squared_error / count
    total_target_variation = target_squared_sum - target_sum**2 / count
    r2 = (
        1.0 - squared_error / total_target_variation
        if total_target_variation > 0.0
        else float("nan")
    )
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": absolute_error / count,
        "r2": r2,
    }


def main() -> None:
    # args = parse_args()
    # if "s_sam" in args.algorithms and args.perturbation_samples % 2:
    #     raise ValueError("--perturbation-samples must be even for antithetic S-SAM")
    #
    # data_config = {
    #     "name": "california_housing",
    #     "root": str(args.data_dir),
    #     "test_fraction": args.test_fraction,
    #     "standardize": True,
    #     "standardize_target": False,
    #     "download": not args.no_download,
    #     "seed": args.seed,
    # }
    # training_data = build_dataset(data_config, train=True)
    # test_data = build_dataset(data_config, train=False)
    # base_model_config = model_config(args.model)
    # args.output_dir.mkdir(parents=True, exist_ok=True)
    #
    # summaries: dict[str, dict[str, float]] = {}
    # for algorithm in args.algorithms:
    #     config = {
    #         "model": copy.deepcopy(base_model_config),
    #         "data": copy.deepcopy(data_config),
    #         "training": {
    #             "algorithm": algorithm,
    #             "steps": args.steps,
    #             "batch_size": args.batch_size,
    #             "learning_rate": {
    #                 "name": "constant",
    #                 "value": args.learning_rate,
    #             },
    #             "sharpness_scale": {
    #                 "name": "constant",
    #                 "value": args.sharpness_scale if algorithm == "s_sam" else 0.0,
    #             },
    #             "perturbation": {
    #                 "distribution": "gaussian",
    #                 "samples": args.perturbation_samples,
    #                 "normalized": True,
    #                 "antithetic": True,
    #             },
    #             "optimizer": {"name": "sgd", "momentum": 0.0},
    #             "loss": "mse",
    #             "checkpoint_every": 0,
    #             "seed": args.seed,
    #             "device": args.device,
    #         },
    #     }
    #
    #     # Resetting the seed gives every algorithm the same initialization.
    #     torch.manual_seed(args.seed)
    #     result = train(build_model(config), training_data, config)
    #     train_scores = regression_metrics(result.model, training_data)
    #     test_scores = regression_metrics(result.model, test_data)
    #     summaries[algorithm] = {
    #         **{f"train_{key}": value for key, value in train_scores.items()},
    #         **{f"test_{key}": value for key, value in test_scores.items()},
    #     }
    #     plot_training_history(
    #         result,
    #         args.output_dir / f"{args.model}_{algorithm}_history.png",
    #     )
    #     print(
    #         f"{algorithm:>5} | test RMSE={test_scores['rmse']:.4f} "
    #         f"(${test_scores['rmse'] * 100_000:,.0f}) | "
    #         f"MAE={test_scores['mae']:.4f} | R2={test_scores['r2']:.4f}"
    #     )
    #
    # summary_path = args.output_dir / f"{args.model}_metrics.json"
    # summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    # print(f"Metrics and plots written to {args.output_dir.resolve()}")


    output_dir = Path("outputs/california")

    training_data = build_dataset(
        BASE_CONFIG["data"],
        train=True
    )

    test_data = build_dataset(
        BASE_CONFIG["data"],
        train=False
    )

    DEPTHS=[2,3,4]
    ALGORITHMS = ["sgd", "s_sam"]

    results = {}


    torch.manual_seed(BASE_CONFIG["seed"])

    for depth in DEPTHS:
        reference_config = config_for(depth, "sgd")

        torch.manual_seed(BASE_CONFIG["training"]["seed"])
        reference_model = build_model(reference_config)
        initial_state = copy.deepcopy(reference_model.state_dict())

        for algorithm in ALGORITHMS:
            config = config_for(depth, algorithm)
            model = build_model(config)
            model.load_state_dict(initial_state, strict=True)

            result = train(
                model,
                training_data,
                config,
            )

            results[(depth, algorithm)] = result


        train_scores = regression_metrics(result.model, training_data)
        test_scores = regression_metrics(result.model, test_data)
        plot_training_history(
            result,
            output_dir / f"{model}_{algorithm}_history.png",
        )
        print(
            f"{algorithm:>5} | test RMSE={test_scores['rmse']:.4f} "
            f"(${test_scores['rmse'] * 100_000:,.0f}) | "
            f"MAE={test_scores['mae']:.4f} | R2={test_scores['r2']:.4f}"
        )


if __name__ == "__main__":
    main()

