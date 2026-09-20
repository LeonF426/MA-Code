import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from ssam import build_model, train


def _load_example_config_module():
    path = Path(__file__).parents[1] / "examples" / "model_config.py"
    spec = importlib.util.spec_from_file_location("california_model_config", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_example_module(monkeypatch):
    examples = Path(__file__).parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    path = examples / "california_housing.py"
    spec = importlib.util.spec_from_file_location("california_housing_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_california_models_have_an_intercept_and_scalar_output():
    configs = _load_example_config_module()

    linear = configs.BASE_CONFIG_LINEAR["model"]
    assert len(linear["layers"]) == 1
    assert linear["layers"][0]["out_dim"] == 1
    assert linear["layers"][0]["bias"] is True
    assert linear["output_reduction"] == "none"

    factorized = configs.BASE_CONFIG_DENSE["model"]
    assert factorized["layers"][-1]["out_dim"] == 1
    assert factorized["layers"][-1]["bias"] is True

    diagonal = configs.config_for(3, "sgd")["model"]
    assert diagonal["layers"][-1]["bias"] is True


def test_linear_example_can_fit_a_nonzero_target_mean():
    configs = _load_example_config_module()
    generator = torch.Generator().manual_seed(41)
    inputs = torch.randn(320, 8, generator=generator)
    weights = torch.tensor([0.8, -0.5, 0.2, 0.0, 0.3, -0.1, 0.6, -0.4])
    targets = 3.25 + inputs @ weights
    training = TensorDataset(inputs[:256], targets[:256])

    config = copy.deepcopy(configs.BASE_CONFIG_LINEAR)
    config["training"].update(
        algorithm="sgd",
        steps=200,
        batch_size=64,
        learning_rate={"name": "constant", "value": 0.05},
        device="cpu",
    )
    torch.manual_seed(configs.SEED)
    result = train(build_model(config), training, config)

    with torch.no_grad():
        predictions = result.model(inputs[256:]).squeeze(-1)
    held_out_targets = targets[256:]
    residual = (predictions - held_out_targets).square().sum()
    total = (held_out_targets - held_out_targets.mean()).square().sum()
    assert float(1.0 - residual / total) > 0.99


def test_california_mse_aligns_scalar_output_without_broadcasting(monkeypatch):
    example = _load_example_module(monkeypatch)
    targets = torch.arange(32, dtype=torch.float32)

    loss = example.scalar_regression_mse(targets[:, None], targets)

    assert loss.item() == 0.0
    with pytest.raises(ValueError, match="matching shapes"):
        example.scalar_regression_mse(torch.zeros((32, 2)), targets)
