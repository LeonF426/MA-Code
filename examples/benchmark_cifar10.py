"""Minimal benchmark example using the same model and training APIs."""

import json

from ssam import (
    build_dataset,
    build_model,
    create_run_artifacts,
    plot_training_history,
    train,
)


CONFIG = {
    # Optional human-readable prefix for the timestamped run directory.
    "run": {"name": None},
    "model": {
        "name": "mnist",
        "type": "torchvision/resnet18",
        "num_classes": 10,
        "parameter_init": {"name": "default"},
    },
    "data": {
        "name": "cifar10",
        "root": "data",
        "download": True,
        "image_size": 224,
    },
    "training": {
        "algorithm": "s_sam",
        "steps": 100,
        "batch_size": 64,
        "learning_rate": {"name": "cosine", "initial": 0.03, "final": 0.001, "duration": 100},
        "sharpness_scale": {"name": "inverse_time", "initial": 0.02, "power": 0.5},
        "perturbation": {"samples": 1, "normalized": True},
        "optimizer": {"name": "sgd", "momentum": 0.9, "weight_decay": 0.0005},
        "loss": "cross_entropy",
        "device": "auto",
    },
    "visualization": {"output_dir": "outputs/cifar10"},
}


if __name__ == "__main__":
    run = create_run_artifacts(
        CONFIG,
        CONFIG["visualization"]["output_dir"],
        "benchmark_cifar10",
    )
    model = build_model(CONFIG)
    dataset = build_dataset(CONFIG["data"])
    result = train(model, dataset, CONFIG)
    plot_training_history(result, run.output_dir / "training_history.png")
    (run.output_dir / "metrics.json").write_text(
        json.dumps({"final_loss": result.history["loss"][-1]}, indent=2),
        encoding="utf-8",
    )
    run.complete()
    print(f"Final loss: {result.history['loss'][-1]:.4f}")
    print(f"Artifacts written to {run.output_dir.resolve()}")
