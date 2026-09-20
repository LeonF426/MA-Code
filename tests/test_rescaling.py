import pytest
import torch
from torch import nn

from ssam import (
    DiagLinear,
    apply_function_preserving_linear_rescaling,
    build_model,
)


def test_neuronwise_rescaling_preserves_general_dense_affine_network():
    torch.manual_seed(12)
    model = nn.Sequential(
        nn.Linear(3, 5, bias=True),
        nn.Identity(),
        nn.Linear(5, 4, bias=True),
        nn.Identity(),
        nn.Linear(4, 2, bias=True),
    ).double()
    inputs = torch.randn(16, 3, dtype=torch.float64)
    expected = model(inputs).detach().clone()
    original_weights = [layer.weight.detach().clone() for layer in model if isinstance(layer, nn.Linear)]

    result = apply_function_preserving_linear_rescaling(
        model,
        log_scale_std=2.0,
        seed=7,
        mode="neuronwise",
    )

    torch.testing.assert_close(model(inputs), expected, atol=1e-10, rtol=1e-10)
    assert result.mode == "neuronwise"
    assert len(result.hidden_log_scales) == 2
    assert any(
        not torch.equal(layer.weight, original)
        for layer, original in zip(
            (module for module in model if isinstance(module, nn.Linear)),
            original_weights,
        )
    )


def test_rescaling_preserves_mixed_dense_and_diagonal_chain_with_biases():
    torch.manual_seed(8)
    layers = [
        DiagLinear(3, bias=True),
        nn.Linear(3, 4, bias=True),
        nn.Linear(4, 1, bias=True),
    ]

    class MixedChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(layers)
            self.activations = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

        def forward(self, inputs):
            for layer in self.layers:
                inputs = layer(inputs)
            return inputs

    model = MixedChain().double()
    inputs = torch.randn(10, 3, dtype=torch.float64)
    expected = model(inputs).detach().clone()

    apply_function_preserving_linear_rescaling(
        model,
        log_scale_std=1.5,
        seed=4,
        mode="layerwise",
    )

    torch.testing.assert_close(model(inputs), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("activation", ("relu", "leaky_relu"))
def test_rescaling_preserves_dense_homogeneous_nonlinear_network(activation):
    torch.manual_seed(21)
    config = {
        "model": {
            "name": f"rescaled_{activation}",
            "type": "mlp",
            "input_dim": 4,
            "output_dim": 2,
            "depth": 3,
            "width": [7, 5],
            "activation": activation,
            "bias": True,
            "parameter_init": {"type": "xavier_uniform", "bias": 0.2},
        }
    }
    model = build_model(config).double()
    inputs = torch.randn(13, 4, dtype=torch.float64)
    expected = model(inputs).detach().clone()

    apply_function_preserving_linear_rescaling(
        model,
        log_scale_std=1.5,
        seed=5,
        mode="neuronwise",
    )

    torch.testing.assert_close(model(inputs), expected, atol=1e-10, rtol=1e-10)


def test_build_model_applies_reproducible_configured_rescaling():
    config = {
        "model": {
            "name": "rescaled_diag",
            "type": "mixed_linear",
            "input_dim": 3,
            "layers": [
                {"type": "diag", "out_dim": 3, "activation": "identity"},
                {"type": "diag", "out_dim": 3, "activation": "identity"},
                {"type": "diag", "out_dim": 3},
            ],
            "bias": False,
            "parameter_init": {
                "type": "identity",
                "rescaling": {
                    "mode": "coordinatewise",
                    "log_scale_std": 2.0,
                    "seed": 19,
                },
            },
        }
    }
    first = build_model(config)
    second = build_model(config)
    inputs = torch.randn(7, 3)

    torch.testing.assert_close(first(inputs), inputs)
    torch.testing.assert_close(second(inputs), inputs)
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)
    assert first.rescaling_result.mode == "neuronwise"
    assert not torch.equal(first.layers[0].weight, torch.ones(3))


def test_configured_rescaling_rejects_nonhomogeneous_hidden_activations():
    config = {
        "model": {
            "type": "mlp",
            "input_dim": 2,
            "output_dim": 1,
            "depth": 2,
            "width": 3,
            "activation": "tanh",
            "parameter_init": {
                "type": "xavier_uniform",
                "rescaling": {"log_scale_std": 1.0},
            },
        }
    }

    with pytest.raises(ValueError, match="positively homogeneous hidden activations"):
        build_model(config)
