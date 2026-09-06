# S-SAM experiment toolkit

This repository provides one small, configuration-driven path for custom networks,
full-batch gradient descent (GD), mini-batch stochastic gradient descent (SGD), and
stochastically perturbed S-SAM updates. The former root-level experiment scripts
have been replaced by reusable modules; [`main.py`](main.py) is now only a concrete
mixed-linear example.

## Install and run

```bash
python -m pip install -e ".[visualization]"
python main.py
```

For torchvision models and datasets, install the benchmark extra:

```bash
python -m pip install -e ".[visualization,benchmarks]"
python examples/benchmark_cifar10.py
```

For the California Housing regression benchmark, install the tabular extra:

```bash
python -m pip install -e ".[tabular]"
python examples/california_housing.py --algorithms sgd s_sam --model mlp
```

## One dictionary controls an experiment

```python
from ssam import build_dataset, build_model, train

config = {
    "model": {
        "name": "my_experiment",   # free-form label
        "type": "mlp",             # implementation selector
        "input_dim": 20,
        "output_dim": 1,
        "depth": 4,                 # total number of trainable layers
        "width": 128,               # or a list with depth - 1 entries
        "activation": "gelu",      # identity is supported
        "output_activation": "identity",
        "bias": True,
        "parameter_init": {"type": "xavier_uniform"},
    },
    "training": {
        "algorithm": "s_sam",      # gd, sgd, or s_sam
        "steps": 500,
        "batch_size": 64,
        "learning_rate": {"name": "constant", "value": 0.01},
        "sharpness_scale": {
            "name": "inverse_time", "initial": 0.2, "power": 0.5, "floor": 0.01
        },
        "perturbation": {
            "samples": 2,
            "normalized": False,
            "antithetic": False,
            "preserve_buffers": True,
        },
        "optimizer": {"name": "sgd", "momentum": 0.9},
        "loss": "mse",
        "checkpoint_every": 10,
    },
}

model = build_model(config)
result = train(model, dataset, config)
```

`mixed_linear` additionally accepts explicit dense and diagonal layers:

```python
"model": {
    "name": "diagonal_test_1",
    "type": "mixed_linear",
    "input_dim": 8,
    "layers": [
        {"type": "diag", "out_dim": 8, "activation": "identity"},
        {"type": "dense", "out_dim": 32, "activation": "tanh"},
        {"type": "dense", "out_dim": 1},
    ],
    "bias": False,
    "parameter_init": {"type": "xavier_uniform"},
}
```

Identity initialization works for diagonal or square weights. Available activations
are `identity`, `relu`, `tanh`, `sigmoid`, `gelu`, `silu`, and `leaky_relu`.
Initialization options are `default`, `identity`, `zeros`, `ones`, `uniform`,
`normal`, `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`,
and `orthogonal`.

## Algorithms and schedules

- `gd` builds one full-dataset batch per update.
- `sgd` builds shuffled mini-batches.
- `s_sam` samples random Gaussian parameter perturbations, computes gradients at
  those perturbed parameters, restores the clean parameters, and applies the
  averaged gradient. `normalized: true` makes the sharpness value a global radius;
  otherwise it is the per-coordinate Gaussian standard deviation.

Both learning rate and sharpness scale accept `constant`, `inverse_time`, `linear`,
`cosine`, or `piecewise` schedules. For S-SAM, the theorem-inspired adaptive policy
can use the current Monte Carlo objective estimate:

```python
"learning_rate": {
    "name": "strong_descent_diag",
    "delta": 0.5,
    "safety": 0.95,
    "max_lr": 0.1,
    "loss_floor": 1e-12,
    # "dimension": 2,  # optional; otherwise inferred
    # "depth": 3,      # optional; otherwise inferred
}
```

At each S-SAM step the implementation computes the online means
`mean(loss(theta + xi))` and `mean(gradient(loss(theta + xi)))` in one loop. Thus
the regularized-objective schedule does not perform a second set of perturbed
forward passes, and memory use does not grow with `samples`. The clean and estimated
regularized losses are available as `result.history["clean_loss"]` and
`result.history["regularized_loss"]`.

For a separate diagnostic evaluation, use
`estimate_regularized_objective(model, closure, scale, samples=...)`. It also uses
an online mean and restores parameters and buffers afterwards. `antithetic: true`
reuses each sampled direction with both signs and therefore requires an even sample
count. `max_grad_norm` may be set under `perturbation` as an additional stability
guard.

The exact bound in Theorem 6.1.1 applies under its diagonal-model and exact-objective
assumptions. For general neural networks and finitely many perturbation samples,
`strong_descent_diag` is an approximate adaptive heuristic; set `dimension` and
`depth` explicitly when the theorem's quantities do not match the automatic model
inference.

A custom update can be added without editing the trainer:

```python
from ssam import register_update_rule
register_update_rule("my_algorithm", MyUpdateRule)
```

The same registry pattern is available through `register_schedule` and
`register_activation`.

## Visualization

```python
from ssam import plot_checkpoint_embedding, plot_loss_landscape, plot_training_history

plot_training_history(result, "outputs/history.png")
plot_checkpoint_embedding(result, method="tsne", path="outputs/trajectory.png")
plot_checkpoint_embedding(result, method="pca", path="outputs/trajectory_pca.png")
```

The t-SNE plot embeds saved parameter checkpoints and colors them by loss. It is
useful for discovering clusters, but t-SNE distorts distance and therefore is not a
literal loss landscape. `plot_loss_landscape` is the more faithful alternative: it
evaluates the model on a random two-dimensional parameter slice. For large benchmark
networks, use a modest resolution and a representative evaluation batch because a
grid of size `n` requires `n²` forward passes.

## Benchmarks

Any torchvision constructor can be selected as `torchvision/<name>`, including
`resnet18`, `resnet50`, `mobilenet_v3_small`, and `vit_b_16`. Classifier heads are
adapted with `num_classes`; `parameter_init.type: pretrained` requests default
weights. MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 are available through
`build_dataset`.

All three update algorithms technically work for benchmark models. In practice,
full-batch GD and multi-sample S-SAM can be prohibitively expensive. Mini-batch SGD
or one-sample normalized S-SAM is the sensible benchmark default, while checkpoint
PCA/t-SNE is cheaper than a dense loss-surface grid.

### California Housing

California Housing is available through the same dataset dictionary:

```python
data_config = {
    "name": "california_housing",
    "root": "data",
    "test_fraction": 0.2,
    "standardize": True,
    "standardize_target": False,
    "download": True,
    "seed": 7,
}

training_data = build_dataset(data_config, train=True)
test_data = build_dataset(data_config, train=False)
```

The dataset has 20,640 observations and eight numerical input features. Targets
remain in the original scikit-learn unit of $100,000 by default. Splitting is
deterministic, and all normalization statistics are fitted only on the training
partition, so the test set does not leak information into training.

[`examples/california_housing.py`](examples/california_housing.py) supports a
single linear layer or a small MLP and any selection of `gd`, `sgd`, and `s_sam`.
It gives every algorithm the same parameter initialization, evaluates held-out
MSE, RMSE, MAE, and R², and writes JSON metrics plus training-history plots.

## Repository layout

```text
src/ssam/
  config.py          dictionary defaults and validation
  models.py          custom and torchvision model builders
  data.py            synthetic and benchmark datasets
  schedules.py       scalar schedules and objective-dependent LR policies
  objectives.py      efficient perturbed-objective estimators
  update_rules.py    GD/SGD and S-SAM parameter updates
  trainers.py        shared training loop and result object
  visualization.py  history, embedding, and loss-slice plots
examples/
  benchmark_cifar10.py
  california_housing.py
main.py              mixed-linear special-case example
tests/               focused behavior tests
```


