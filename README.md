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
python examples/california_housing.py
```

The configurable Poisson PINN comparison only needs the base dependencies. Edit
the single `CONFIG` dictionary at the top of the example, then run it:

```bash
python examples/poisson_pinn.py
```

Three diagonal-linear PINN examples use polynomial feature maps instead of
learned nonlinear activations:

```bash
python examples/advection_pinn.py
python examples/polynomial_poisson_pinn.py
python examples/polynomial_heat_pinn.py
```

An analytically tractable two-factor example starts from the exactly fitting but
highly unbalanced representation `(10, 0.1)`. It demonstrates a setting where
Gaussian S-SAM can reduce parameter-noise sensitivity while SGD has zero clean
gradient:

```bash
python examples/deep_linear_balance.py
```

The example compares measured sharpness with its exact formula and writes factor,
loss, balance, and interpolation diagnostics beneath
`outputs/deep_linear_balance/`.

Select constant or tamed learning rates directly in `CONFIG["training"]`.

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

Sequential dense, diagonal, or mixed networks with identity, ReLU, or LeakyReLU
hidden activations can be made deliberately imbalanced without changing their
represented function. Set a log-normal rescaling beneath `parameter_init`:

```python
"parameter_init": {
    "type": "xavier_uniform",
    "rescaling": {
        "mode": "neuronwise",       # or "layerwise"
        "log_scale_std": 2.0,
        "seed": 19,
    },
}
```

At each hidden interface this applies a positive diagonal change of coordinates
and its inverse to the following layer. It supports changing dense widths,
diagonal layers, mixed dense/diagonal chains, and biases. The same operation is
available as `apply_function_preserving_linear_rescaling`; custom models can pass
their ordered linear layers explicitly. ReLU and LeakyReLU work because they are
positively homogeneous. Non-homogeneous activations such as GELU, tanh, sigmoid,
and SiLU are rejected because the same transformation would change the model's
function. Apply it before creating an optimizer.

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

Loss closures may also return a mapping containing `loss` plus named scalar
components. S-SAM reports clean components and Gaussian means computed from the
same perturbations as its gradient. `training.loss_requires_grad` is `false` by
default, preserving the supervised path; derivative-based objectives opt in so
coordinate autograd remains available during clean-loss logging.

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

### Sharpness along an endpoint interpolation

To determine whether two trained solutions occupy regions with genuinely
different curvature, evaluate the straight parameter path between them. The
closure must evaluate the first model, which is used as coefficient zero:

```python
from ssam import (
    evaluate_average_sharpness_interpolation,
    evaluate_average_sharpness_interpolation_closure,
    plot_sharpness_interpolation,
)

def loss_closure():
    return loss_function(sgd_model(inputs), targets)

diagnostic = evaluate_average_sharpness_interpolation_closure(
    sgd_model,
    ssam_model,
    loss_closure,
    sharpness_scale=1e-3,
    interpolation_points=11,
    samples=64,
    seed=10023,
    antithetic=True,
)
plot_sharpness_interpolation(
    diagnostic,
    "outputs/sharpness_interpolation.png",
    endpoint_labels=("SGD", "S-SAM"),
)
```

Every interpolation point uses the same Gaussian directions, reducing Monte
Carlo noise in comparisons along the path. The evaluator restores the first
model after completion. The paired PINN examples additionally write
`sharpness_interpolation.json` and `sharpness_interpolation.png`; their
derivative-based closures set `requires_grad=True`.

For ordinary supervised datasets,
`evaluate_average_sharpness_interpolation` accepts a `DataLoader` and loss
function directly. `main.py` uses this wrapper and writes the same JSON and PNG
artifacts for its noisy linear-regression comparison.

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
    "impute_missing": True,
    "clip_quantiles": [0.005, 0.995],
    "standardize": True,
    "standardize_target": False,
    "download": True,
    "seed": 7,
}

training_data = build_dataset(data_config, train=True)
test_data = build_dataset(data_config, train=False)
```

The dataset has 20,640 observations and eight numerical input features. The
canonical copy has no missing values, but non-finite features are median-imputed
by default so locally modified copies fail safely; non-finite targets are always
rejected. `clip_quantiles` optionally winsorizes each feature, which is useful for
the extreme values in the ratio and population columns. Imputation medians,
clipping bounds, feature means, feature standard deviations, and optional target
statistics are fitted only on the training partition, so the test set does not
leak information into preprocessing. Set `impute_missing: false` to reject
non-finite inputs, or omit `clip_quantiles` to disable clipping.

Targets remain in the original scikit-learn unit of $100,000 by default. Models
trained on centered features must therefore include a bias/intercept. Without
one, their mean prediction is constrained near zero while the target mean is
near two, producing the previously observed R-squared near -2.6 even if the
feature coefficients are otherwise reasonable. Target standardization is
supported, but the example deliberately leaves it off so RMSE and MAE retain
their original units.

[`examples/california_housing.py`](examples/california_housing.py) first runs a
genuine one-layer 8-to-1 affine regression baseline, then compares SGD and S-SAM
on two-, three-, and four-layer diagonal models, a dense linear factorization,
and a dense ReLU network. All models have an intercept and scalar output; the
factorized model uses Xavier initialization instead of the former all-ones
8-vector output. Every pair starts from identical parameters. Multi-layer pairs
retain a moderate function-preserving rescaling, while the normalized S-SAM
radius is 0.05 rather than the former radius of 2.0. The example evaluates
held-out MSE, RMSE, MAE, R-squared, prediction mean, and target mean; prints
held-out interpolation tables; and writes metrics, training histories,
and SGD-to-S-SAM sharpness interpolation JSON/PNG artifacts beneath
`outputs/california/`. Rescaling severity and interpolation sampling are
configured in [`examples/model_config.py`](examples/model_config.py).

### 2D Poisson PINN

[`examples/poisson_pinn.py`](examples/poisson_pinn.py) solves the unit-square
problem

```text
u*(x,y) = sin(pi*x) sin(pi*y)
-Delta u = 2*pi^2 sin(pi*x) sin(pi*y),   u|boundary = 0.
```

It uses a configurable tanh MLP and the complete weighted PDE-residual plus
boundary loss. Collocation coordinates are created once and captured by the loss
closure, so every parameter perturbation in an S-SAM update sees the same points.
SGD and S-SAM start from an identical state and share collocation/evaluation
points, evaluation scale, and Gaussian evaluation noise. The example writes
relative L2 error, PDE-residual RMSE, boundary RMSE, and average sharpness with a
Monte Carlo confidence interval to JSON. It also plots exact/predicted/error
fields and clean versus Gaussian-averaged loss histories. Missing Gaussian curves
(for example for SGD) are skipped.

The reusable API is `build_poisson_points`, `train_poisson_pinn`,
`evaluate_poisson_pinn`, `plot_poisson_solutions`, and
`plot_pinn_training_history`. `evaluate_average_sharpness_closure` provides the
same coordinate-derivative-safe sharpness path for other physics-informed losses.

### Diagonal-linear PINNs

The three additional PINN examples keep every trainable layer diagonal and every
activation equal to the identity. With sum reduction, a depth-`L` model predicts

```text
u(z) = sum_i (product_l w[l, i]) z_i.
```

[`examples/advection_pinn.py`](examples/advection_pinn.py) uses the feature vector
`(1, x, t)` for `u_t - u_x = 0` and the exact solution `2 + 3x + 3t`.
[`examples/polynomial_poisson_pinn.py`](examples/polynomial_poisson_pinn.py) uses
`(1, x, x^2)` for `-u_xx = -4` with exact solution `1 + 3x + 2x^2`.
[`examples/polynomial_heat_pinn.py`](examples/polynomial_heat_pinn.py) uses
`(x^2, t)` for `u_t - u_xx = 0` with exact solution `x^2 + 2t`.

The feature maps are evaluated from raw coordinate tensors inside each residual,
so autograd retains the coordinate derivatives. As in the original Poisson
example, each script compares SGD and S-SAM from the same initialization and
writes solution plots, loss histories, and JSON metrics beneath `outputs/`.
Their identity-initialized layers receive a deterministic, function-preserving
layerwise rescaling with log-scale standard deviation `0.5`. This makes the
starting factorization artificially imbalanced without changing its initial
input-output function; both algorithms still receive the exact same rescaled
parameters.

## Repository layout

```text
src/ssam/
  config.py          dictionary defaults and validation
  models.py          custom and torchvision model builders
  data.py            synthetic and benchmark datasets
  schedules.py       scalar schedules and objective-dependent LR policies
  objectives.py      efficient perturbed-objective estimators
  pinn.py            Poisson residuals, fixed points, training, and metrics
  update_rules.py    GD/SGD and S-SAM parameter updates
  trainers.py        shared training loop and result object
  visualization.py  history, embedding, and loss-slice plots
examples/
  benchmark_cifar10.py
  california_housing.py
  advection_pinn.py
  poisson_pinn.py
  polynomial_heat_pinn.py
  polynomial_poisson_pinn.py
main.py              mixed-linear special-case example
tests/               focused behavior tests
```


