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

## One dictionary controls an experiment

```python
from ssam import build_dataset, build_model, train

config = {
    "model": {
        "name": "mlp",
        "input_dim": 20,
        "output_dim": 1,
        "depth": 4,                 # total number of trainable layers
        "width": 128,               # or a list with depth - 1 entries
        "activation": "gelu",      # identity is supported
        "output_activation": "identity",
        "bias": True,
        "parameter_init": {"name": "xavier_uniform"},
    },
    "training": {
        "algorithm": "s_sam",      # gd, sgd, or s_sam
        "steps": 500,
        "batch_size": 64,
        "learning_rate": {"name": "constant", "value": 0.01},
        "sharpness_scale": {
            "name": "inverse_time", "initial": 0.2, "power": 0.5, "floor": 0.01
        },
        "perturbation": {"samples": 2, "normalized": False},
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
    "name": "mixed_linear",
    "input_dim": 8,
    "layers": [
        {"type": "diag", "out_dim": 8, "activation": "identity"},
        {"type": "dense", "out_dim": 32, "activation": "tanh"},
        {"type": "dense", "out_dim": 1},
    ],
    "bias": False,
    "parameter_init": {"name": "xavier_uniform"},
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
`cosine`, or `piecewise` schedules. A custom update can be added without editing the
trainer:

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
adapted with `num_classes`; `parameter_init.name: pretrained` requests default
weights. MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 are available through
`build_dataset`.

All three update algorithms technically work for benchmark models. In practice,
full-batch GD and multi-sample S-SAM can be prohibitively expensive. Mini-batch SGD
or one-sample normalized S-SAM is the sensible benchmark default, while checkpoint
PCA/t-SNE is cheaper than a dense loss-surface grid.

## Repository layout

```text
src/ssam/
  config.py          dictionary defaults and validation
  models.py          custom and torchvision model builders
  data.py            synthetic and benchmark datasets
  schedules.py       reusable scalar schedules
  update_rules.py    GD/SGD and S-SAM parameter updates
  trainers.py        shared training loop and result object
  visualization.py  history, embedding, and loss-slice plots
examples/
  benchmark_cifar10.py
main.py              mixed-linear special-case example
tests/               focused behavior tests
```
