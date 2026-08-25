from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np


Array = np.ndarray
Gradient = Callable[[Array, float], Array]
Schedule = float | Sequence[float] | Callable[[int], float]


def _schedule_value(schedule: Schedule, k: int) -> float:
    """Return the value of a scalar, sequence, or callable schedule at step k."""
    if callable(schedule):
        return float(schedule(k))
    if np.isscalar(schedule):
        return float(schedule)
    return float(schedule[k])


def explicit_euler_step(
    theta_k: Array,
    alpha_k: float,
    eta_k: float,
    grad_loss: Gradient,
) -> Array:
    """Equation (5.3): theta_(k+1) = theta_k - alpha_k grad L_R."""
    theta_k = np.asarray(theta_k, dtype=float)
    return theta_k - alpha_k * np.asarray(grad_loss(theta_k, eta_k), dtype=float)


def implicit_euler_step(
    theta_k: Array,
    alpha_k: float,
    eta_k: float,
    grad_loss: Gradient,
    *,
    solver_tol: float = 1e-10,
    maxfev: int = 1000,
) -> Array:
    """Equation (5.4), solved for theta_(k+1) with a nonlinear root solver.

    The explicit-Euler prediction is used as the initial guess. A failed solve
    raises RuntimeError rather than silently returning an invalid iterate.
    """
    from scipy.optimize import root

    theta_k = np.asarray(theta_k, dtype=float)

    def residual(theta_next: Array) -> Array:
        return theta_next - theta_k + alpha_k * np.asarray(
            grad_loss(theta_next, eta_k), dtype=float
        )

    initial_guess = explicit_euler_step(theta_k, alpha_k, eta_k, grad_loss)
    solution = root(
        residual,
        initial_guess,
        method="hybr",
        options={"xtol": solver_tol, "maxfev": maxfev},
    )
    if not solution.success:
        residual_norm = np.linalg.norm(residual(solution.x))
        raise RuntimeError(
            "Implicit Euler solve failed at "
            f"theta={theta_k}: {solution.message}; "
            f"residual norm={residual_norm:.3e}"
        )
    return np.asarray(solution.x, dtype=float)


def tamed_gradient_descent_step(
    theta_k: Array,
    alpha_k: float,
    eta_k: float,
    grad_loss: Gradient,
) -> Array:
    """Equation (5.5): gradient descent with denominator 1 + alpha_k ||g_k||."""
    theta_k = np.asarray(theta_k, dtype=float)
    gradient = np.asarray(grad_loss(theta_k, eta_k), dtype=float)
    return theta_k - alpha_k * gradient / (1.0 + alpha_k * np.linalg.norm(gradient))


def hessian_F_eta(theta: Array, eta: float, w_star: float) -> Array:
    """Exact theta-Hessian of the F_eta implemented in the supplied utils.py."""
    w1, w2 = np.asarray(theta, dtype=float)
    return np.array(
        [
            [2.0 * w2**2 + 2.0 * eta**2, 4.0 * w1 * w2 - 2.0 * w_star],
            [4.0 * w1 * w2 - 2.0 * w_star, 2.0 * w1**2 + 2.0 * eta**2],
        ],
        dtype=float,
    )


def tamed_gd_stepsize_bound(
    theta: Array,
    eta: float,
    w_star: float,
    delta: float,
    grad_loss: Gradient,
) -> float:
    """Evaluate the displayed tamed-GD bound at the current iterate.

    alpha <= 2(1-delta) /
             (||H_eta(theta)||_op - 2(1-delta)||grad F_eta(theta)||).

    If the denominator is nonpositive, the preceding inequality is satisfied
    for every positive alpha, so ``np.inf`` is returned.
    """
    if not 0.0 <= delta < 1.0:
        raise ValueError("delta must satisfy 0 <= delta < 1")

    theta = np.asarray(theta, dtype=float)
    gradient = np.asarray(grad_loss(theta, eta), dtype=float)
    hessian = hessian_F_eta(theta, eta, w_star)
    hessian_op_norm = float(np.max(np.abs(np.linalg.eigvalsh(hessian))))

    numerator = 2.0 * (1.0 - delta)
    denominator = hessian_op_norm - numerator * np.linalg.norm(gradient)
    return np.inf if denominator <= 0.0 else numerator / denominator


def run_tamed_gd_with_bound(
    theta0: Array,
    n_steps: int,
    eta_schedule: Schedule,
    grad_loss: Gradient,
    w_star: float,
    delta: float,
    *,
    safety: float = 0.95,
    fallback_alpha: float | None = None,
) -> tuple[Array, Array, Array]:
    """Run tamed GD using its bound as a state-dependent step-size schedule.

    ``fallback_alpha`` is required only if the theoretical bound is infinite;
    a finite practical step must then be chosen because infinity cannot be used
    in the update formula.
    """
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety must satisfy 0 < safety <= 1")
    if fallback_alpha is not None and fallback_alpha <= 0.0:
        raise ValueError("fallback_alpha must be positive")

    trajectory = np.empty((n_steps + 1, np.asarray(theta0).size), dtype=float)
    etas = np.empty(n_steps, dtype=float)
    alphas = np.empty(n_steps, dtype=float)
    trajectory[0] = np.asarray(theta0, dtype=float)

    for k in range(n_steps):
        etas[k] = _schedule_value(eta_schedule, k)
        bound = tamed_gd_stepsize_bound(
            trajectory[k], etas[k], w_star, delta, grad_loss
        )
        if np.isinf(bound):
            if fallback_alpha is None:
                raise ValueError(
                    "The tamed-GD bound is infinite at step "
                    f"{k}; provide a finite fallback_alpha."
                )
            alphas[k] = fallback_alpha
        else:
            alphas[k] = safety * bound

        trajectory[k + 1] = tamed_gradient_descent_step(
            trajectory[k], alphas[k], etas[k], grad_loss
        )

    return trajectory, etas, alphas


def run_discrete_method(
    theta0: Array,
    n_steps: int,
    alpha_schedule: Schedule,
    eta_schedule: Schedule,
    grad_loss: Gradient,
    *,
    method: Literal["explicit_euler", "implicit_euler", "tamed_gd"],
) -> tuple[Array, Array, Array]:
    """Run one of the three methods and return (trajectory, etas, alphas)."""
    steppers = {
        "explicit_euler": explicit_euler_step,
        "implicit_euler": implicit_euler_step,
        "tamed_gd": tamed_gradient_descent_step,
    }
    if method not in steppers:
        raise ValueError(f"Unknown method {method!r}; choose one of {tuple(steppers)}")

    trajectory = np.empty((n_steps + 1, np.asarray(theta0).size), dtype=float)
    etas = np.empty(n_steps, dtype=float)
    alphas = np.empty(n_steps, dtype=float)
    trajectory[0] = np.asarray(theta0, dtype=float)

    step = steppers[method]
    for k in range(n_steps):
        etas[k] = _schedule_value(eta_schedule, k)
        alphas[k] = _schedule_value(alpha_schedule, k)
        trajectory[k + 1] = step(
            trajectory[k], alphas[k], etas[k], grad_loss
        )

    return trajectory, etas, alphas


def explicit_euler_stepsize_bound(
    theta: Array,
    eta: float | Array,
    w_star: float,
    *,
    convention: Literal["utils", "paper"] = "utils",
) -> Array:
    """Evaluate an explicit-Euler bound for one point or a trajectory.

    ``convention="utils"`` (default) uses the largest Hessian eigenvalue of
    the supplied utils.py loss

        F_eta = (w_star - w1*w2)^2 + eta^2 * (w1^2 + w2^2).

    ``convention="paper"`` reproduces the displayed eigenvalue formula in the
    paper. Both versions contain ``2*eta**2``; they differ only in the
    coefficient on ``w_star``. The additive ``eta**4`` in the paper's loss is
    constant with respect to theta and therefore does not enter this Hessian.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.shape[-1] != 2:
        raise ValueError("theta must have final dimension 2 (w1, w2)")

    w1 = theta[..., 0]
    w2 = theta[..., 1]
    eta = np.asarray(eta, dtype=float)
    if convention == "utils":
        denominator = (
            np.sum(theta * theta, axis=-1)
            + 2.0 * eta**2
            + np.sqrt(
                (w1**2 - w2**2) ** 2
                + (4.0 * w1 * w2 - 2.0 * w_star) ** 2
            )
        )
    elif convention == "paper":
        denominator = (
            np.sum(theta * theta, axis=-1)
            + 2.0 * eta**2
            + np.sqrt(
                (w1**2 - w2**2) ** 2
                + (4.0 * w1 * w2 - w_star) ** 2
            )
        )
    else:
        raise ValueError("convention must be 'utils' or 'paper'")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0.0, 2.0 / denominator, np.inf)


def plot_explicit_euler_stepsize_bound(
    times: Array,
    exact_trajectory: Array,
    eta_func: Callable[[float], float],
    w_star: float,
    filename: str | Path,
    *,
    color: str = "#FF6B4A",
    convention: Literal["utils", "paper"] = "utils",
) -> Array:
    """Plot the explicit-Euler bound along an already-computed exact solution."""
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=float)
    exact_trajectory = np.asarray(exact_trajectory, dtype=float)
    if exact_trajectory.shape != (times.size, 2):
        raise ValueError("exact_trajectory must have shape (len(times), 2)")

    eta_values = np.asarray([eta_func(t) for t in times], dtype=float)
    bounds = explicit_euler_stepsize_bound(
        exact_trajectory,
        eta_values,
        w_star,
        convention=convention,
    )

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(times, bounds, color=color, linewidth=2.4)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"explicit Euler bound $\alpha_{\max}(t)$")
    ax.set_title("Explicit Euler step-size bound along the exact solution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return bounds
