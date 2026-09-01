import pytest
import torch
from torch import nn

from ssam import estimate_regularized_objective


def test_zero_scale_matches_clean_loss_and_restores_model_state():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Tanh())
    model.train()
    inputs = torch.tensor([[1.0, -1.0], [2.0, 0.5]])
    targets = torch.zeros(2, 2)

    def closure():
        return nn.functional.mse_loss(model(inputs), targets)

    clean = float(closure().detach())
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    buffers_before = [buffer.detach().clone() for buffer in model.buffers()]
    estimated = estimate_regularized_objective(
        model, closure, 0.0, samples=4, antithetic=True
    )
    assert estimated == pytest.approx(clean)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(model.parameters(), parameters_before)
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(model.buffers(), buffers_before)
    )


def test_antithetic_sampling_requires_even_samples():
    model = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="even sample count"):
        estimate_regularized_objective(
            model,
            lambda: model(torch.ones(1, 1)).square().mean(),
            0.1,
            samples=3,
            antithetic=True,
        )

