import numpy as np

from ssam import plot_training_history


def test_training_history_plot_smooths_noise_and_reports_robust_loss_metrics():
    steps = np.arange(101)
    clean = 50_000.0 * np.exp(-steps / 18.0)
    clean[::10] *= 3.0
    regularized = clean * 1.15
    history = {
        "step": steps.tolist(),
        "loss": clean.tolist(),
        "clean_loss": clean.tolist(),
        "regularized_loss": regularized.tolist(),
        "learning_rate": np.geomspace(0.1, 0.0001, steps.size).tolist(),
        "sharpness_scale": np.linspace(1.0, 0.0, steps.size).tolist(),
        "layer_balance": [[] for _ in steps],
    }

    figure = plot_training_history(history, smoothing_window=9)
    loss_axis = next(
        axis
        for axis in figure.axes
        if axis.get_title(loc="left") == "Objective loss (from step 50)"
    )

    assert len(figure.axes) == 3
    assert loss_axis.get_yscale() == "log"
    assert len(loss_axis.lines) == 4  # raw and trend for both objectives
    assert all(float(line.get_xdata()[0]) == 50.0 for line in loss_axis.lines)
    assert loss_axis.lines[1].get_color() != loss_axis.lines[3].get_color()
    assert loss_axis.lines[1].get_linestyle() != loss_axis.lines[3].get_linestyle()
    assert loss_axis.collections  # shaded gap between the objective trends
    metric_text = "\n".join(text.get_text() for text in loss_axis.texts)
    assert "9-step median" in metric_text
    assert "Improvement" in metric_text
    assert "Final reg. gap" in metric_text


def test_training_history_plot_can_disable_smoothing():
    history = {
        "step": list(range(10)),
        "loss": [float(10 - step) for step in range(10)],
    }

    figure = plot_training_history(history, smoothing_window=1)
    loss_axis = figure.axes[0]

    assert len(figure.axes) == 1
    assert len(loss_axis.lines) == 1
    assert "observed" in loss_axis.texts[0].get_text()


def test_training_history_plot_can_include_the_warmup():
    history = {
        "step": list(range(60)),
        "loss": [float(60 - step) for step in range(60)],
    }

    figure = plot_training_history(
        history,
        smoothing_window=1,
        loss_start_step=None,
    )
    loss_axis = figure.axes[0]

    assert float(loss_axis.lines[0].get_xdata()[0]) == 0.0
    assert loss_axis.get_title(loc="left") == "Objective loss"


def test_automatic_smoothing_window_is_not_too_wide():
    history = {
        "step": list(range(1_000)),
        "loss": np.geomspace(1_000.0, 1.0, 1_000).tolist(),
    }

    figure = plot_training_history(history)
    metric_text = figure.axes[0].texts[0].get_text()

    assert "15-step median" in metric_text
