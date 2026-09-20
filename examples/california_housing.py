"""Compare SGD and Gaussian S-SAM on California Housing models."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ssam import (
    AverageSharpnessInterpolationResult,
    build_dataset,
    build_model,
    evaluate_average_sharpness_interpolation,
    plot_sharpness_interpolation,
    plot_training_history,
    train,
)
from model_config import (
    BASE_CONFIG,
    BASE_CONFIG_DENSE,
    BASE_CONFIG_LINEAR,
    BASE_CONFIG_RELU,
    SEED,
    config_for,
)


def scalar_regression_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE for scalar predictions without implicit broadcasting."""

    if predictions.ndim == targets.ndim + 1 and predictions.shape[-1] == 1:
        predictions = predictions.squeeze(-1)
    if predictions.shape != targets.shape:
        raise ValueError(
            "Regression predictions and targets must have matching shapes, "
            f"received {tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    return torch.nn.functional.mse_loss(predictions, targets)


def regression_metrics(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Evaluate regression errors and the prediction/target means."""

    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    count = 0
    squared_error = 0.0
    absolute_error = 0.0
    prediction_sum = 0.0
    target_sum = 0.0
    target_squared_sum = 0.0
    try:
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                if predictions.ndim == targets.ndim + 1 and predictions.shape[-1] == 1:
                    predictions = predictions.squeeze(-1)
                if predictions.shape != targets.shape:
                    raise ValueError(
                        "Regression predictions and targets must have matching "
                        f"shapes, received {tuple(predictions.shape)} and "
                        f"{tuple(targets.shape)}"
                    )
                predictions = predictions.double()
                targets = targets.double()
                errors = predictions - targets
                count += targets.numel()
                squared_error += float(errors.square().sum().item())
                absolute_error += float(errors.abs().sum().item())
                prediction_sum += float(predictions.sum().item())
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
        "prediction_mean": prediction_sum / count,
        "target_mean": target_sum / count,
    }


def _print_interpolation_table(
    label: str,
    result: AverageSharpnessInterpolationResult,
) -> None:
    print(f"\n{label}: held-out interpolation (t=0 SGD, t=1 S-SAM)")
    print(
        f"{'t':>6} {'clean':>14} {'Gaussian mean':>14} "
        f"{'sharpness':>14} {'95% radius':>14}"
    )
    for point in result.points:
        print(
            f"{point.coefficient:6.2f} "
            f"{point.clean_loss:14.6e} "
            f"{point.regularized_loss:14.6e} "
            f"{point.average_sharpness:14.6e} "
            f"{1.96 * point.standard_error:14.6e}"
        )


def _evaluate_interpolation(
    label: str,
    sgd_model: torch.nn.Module,
    ssam_model: torch.nn.Module,
    evaluation_loader: DataLoader,
    config: dict,
    output_dir: Path,
) -> AverageSharpnessInterpolationResult:
    options = config["visualization"]["interpolation"]
    result = evaluate_average_sharpness_interpolation(
        sgd_model,
        ssam_model,
        evaluation_loader,
        scalar_regression_mse,
        sharpness_scale=float(options["sharpness_scale"]),
        interpolation_points=int(options["interpolation_points"]),
        samples=int(options["sharpness_samples"]),
        seed=int(options["sharpness_seed"]),
        antithetic=bool(options.get("antithetic", True)),
        normalized=bool(options.get("normalized", False)),
    )
    plot_sharpness_interpolation(
        result,
        output_dir / f"{label}_sharpness_interpolation.png",
        endpoint_labels=("SGD", "S-SAM"),
    )
    (output_dir / f"{label}_sharpness_interpolation.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    _print_interpolation_table(label, result)
    return result


def _train_pair(
    label: str,
    configs: dict[str, dict],
    training_data: Dataset,
    test_data: Dataset,
    evaluation_loader: DataLoader,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, dict[str, float]], AverageSharpnessInterpolationResult]:
    torch.manual_seed(SEED)
    reference = build_model(configs["sgd"])
    initial_state = copy.deepcopy(reference.state_dict())
    results = {}
    scores = {}

    for algorithm in ("sgd", "s_sam"):
        print("----------------------------Next Model----------------------------------")
        config = configs[algorithm]
        model = build_model(config)
        model.load_state_dict(initial_state, strict=True)
        result = train(model, training_data, config)
        results[algorithm] = result

        train_scores = regression_metrics(result.model, training_data)
        test_scores = regression_metrics(result.model, test_data)
        scores[algorithm] = {
            **{f"train_{key}": value for key, value in train_scores.items()},
            **{f"test_{key}": value for key, value in test_scores.items()},
        }
        plot_training_history(
            result,
            output_dir / f"{label}_{algorithm}_history.png",
        )
        print(
            f"{algorithm:>5} | test RMSE={test_scores['rmse']:.4f} "
            f"(${test_scores['rmse'] * 100_000:,.0f}) | "
            f"MAE={test_scores['mae']:.4f} | R2={test_scores['r2']:.4f}"
        )

    interpolation = _evaluate_interpolation(
        label,
        results["sgd"].model,
        results["s_sam"].model,
        evaluation_loader,
        configs["s_sam"],
        output_dir,
    )
    return results, scores, interpolation


def _algorithm_configs(base_config: dict) -> dict[str, dict]:
    """Return matched SGD/S-SAM configs for one model family."""

    configs = {}
    for algorithm in ("sgd", "s_sam"):
        config = copy.deepcopy(base_config)
        config["training"]["algorithm"] = algorithm
        config["model"]["name"] = f"{base_config['model']['name']}_{algorithm}"
        if algorithm == "sgd":
            config["training"]["sharpness_scale"] = {
                "name": "constant",
                "value": 0.0,
            }
        configs[algorithm] = config
    return configs


def main() -> None:
    output_dir = Path(BASE_CONFIG["visualization"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training_data = build_dataset(BASE_CONFIG["data"], train=True)
    test_data = build_dataset(BASE_CONFIG["data"], train=False)
    evaluation_loader = DataLoader(
        test_data,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
    )

    summaries = {}

    _, linear_scores, _ = _train_pair(
        "california_linear",
        _algorithm_configs(BASE_CONFIG_LINEAR),
        training_data,
        test_data,
        evaluation_loader,
        output_dir,
    )
    summaries["california_linear"] = linear_scores

    for depth in (3,):
        label = f"california_{depth}L_diag"
        configs = {
            algorithm: config_for(depth, algorithm)
            for algorithm in ("sgd", "s_sam")
        }
        _, scores, _ = _train_pair(
            label,
            configs,
            training_data,
            test_data,
            evaluation_loader,
            output_dir,
        )
        summaries[label] = scores

    _, dense_scores, _ = _train_pair(
        "california_dense",
        _algorithm_configs(BASE_CONFIG_DENSE),
        training_data,
        test_data,
        evaluation_loader,
        output_dir,
    )
    summaries["california_dense"] = dense_scores

    (output_dir / "metrics.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    print(f"\nMetrics and plots written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
