import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ssam import (
    evaluate_average_sharpness_interpolation,
    evaluate_average_sharpness_interpolation_closure,
    plot_sharpness_interpolation,
)


def _scalar_model(weight: float) -> nn.Linear:
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


def test_interpolation_uses_common_noise_and_restores_endpoints():
    start = _scalar_model(0.0)
    end = _scalar_model(2.0)
    inputs = torch.ones((1, 1))
    targets = torch.zeros((1, 1))

    def loss_closure():
        return (start(inputs) - targets).square().mean()

    result = evaluate_average_sharpness_interpolation_closure(
        start,
        end,
        loss_closure,
        0.1,
        interpolation_points=3,
        samples=10,
        seed=7,
        antithetic=True,
    )

    assert [point.coefficient for point in result.points] == [0.0, 0.5, 1.0]
    assert [point.clean_loss for point in result.points] == pytest.approx([0.0, 1.0, 4.0])
    assert [point.average_sharpness for point in result.points] == pytest.approx(
        [result.points[0].average_sharpness] * 3,
        abs=1e-7,
    )
    assert start.weight.item() == pytest.approx(0.0)
    assert end.weight.item() == pytest.approx(2.0)


def test_interpolation_accepts_explicit_coefficients():
    start = _scalar_model(0.0)
    end = _scalar_model(1.0)

    result = evaluate_average_sharpness_interpolation_closure(
        start,
        end,
        lambda: start.weight.square().sum(),
        0.0,
        interpolation_points=(0.25, 0.75),
        samples=2,
    )

    assert [point.coefficient for point in result.points] == [0.25, 0.75]
    assert [point.clean_loss for point in result.points] == pytest.approx([0.0625, 0.5625])


def test_interpolation_rejects_incompatible_endpoints():
    start = _scalar_model(0.0)
    end = nn.Linear(2, 1, bias=False)

    with pytest.raises(ValueError, match="shapes"):
        evaluate_average_sharpness_interpolation_closure(
            start,
            end,
            lambda: start.weight.square().sum(),
            0.1,
            samples=2,
        )


def test_supervised_interpolation_wrapper_and_plot():
    start = _scalar_model(0.0)
    end = _scalar_model(1.0)
    loader = DataLoader(
        TensorDataset(torch.ones((2, 1)), torch.zeros((2, 1))),
        batch_size=1,
    )

    result = evaluate_average_sharpness_interpolation(
        start,
        end,
        loader,
        nn.MSELoss(),
        0.0,
        interpolation_points=2,
        samples=2,
    )
    figure = plot_sharpness_interpolation(
        result,
        endpoint_labels=("SGD", "S-SAM"),
    )

    assert [point.clean_loss for point in result.points] == pytest.approx([0.0, 1.0])
    assert len(figure.axes) == 2
