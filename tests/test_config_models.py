import pytest
import torch

from ssam import build_model, normalize_config


def test_config_normalizes_ssam_alias_and_defaults():
    config = normalize_config({
        "model": {"name": "mlp", "input_dim": 2, "depth": 1},
        "training": {"algorithm": "s-sam", "steps": 2, "batch_size": 1},
    })
    assert config["training"]["algorithm"] == "s_sam"
    assert config["training"]["perturbation"]["samples"] == 1


def test_mlp_shape_and_identity_activation():
    model = build_model({
        "name": "mlp",
        "input_dim": 3,
        "output_dim": 2,
        "depth": 3,
        "width": [5, 4],
        "activation": "identity",
        "bias": False,
        "parameter_init": {"name": "ones"},
    })
    assert model(torch.ones(7, 3)).shape == (7, 2)


def test_diagonal_identity_initialization():
    model = build_model({
        "name": "mixed_linear",
        "input_dim": 2,
        "layers": [{"type": "diag", "out_dim": 2}],
        "activation": "identity",
        "bias": False,
        "parameter_init": {"name": "identity"},
    })
    inputs = torch.tensor([[2.0, -3.0]])
    assert torch.equal(model(inputs), inputs)


def test_unknown_activation_is_clear():
    with pytest.raises(ValueError, match="Unknown activation"):
        build_model({"name": "mlp", "input_dim": 2, "depth": 2, "activation": "magic"})
