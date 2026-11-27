"""Reproducible projected S-SAM examples for the diagonal linear network.

Run ``python projected_examples.py --example all --paint R_min R_max projection``.
The distributions below have finite support, so population moments and radii
are calculated from their probability masses, never estimated from a batch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


@dataclass(frozen=True)
class DiscreteDistribution:
    name: str
    x: np.ndarray  # shape (number of atoms, d)
    y: np.ndarray  # shape (number of atoms,)
    probability: np.ndarray

    def __post_init__(self):
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        p = np.asarray(self.probability, dtype=float)
        if x.ndim != 2 or y.shape != (len(x),) or p.shape != (len(x),):
            raise ValueError("Expected x=(atoms,d), y=(atoms,), probability=(atoms,).")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Data atoms must be finite.")
        if (p < 0).any() or not np.isclose(p.sum(), 1.0):
            raise ValueError("Probabilities must be nonnegative and sum to one.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "probability", p)

    def sample(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        indices = rng.choice(len(self.y), size=n, p=self.probability)
        return self.x[indices], self.y[indices]


@dataclass(frozen=True)
class Example:
    distribution: DiscreteDistribution
    initial: np.ndarray  # shape (L,d)

    @property
    def L(self) -> int:
        return self.initial.shape[0]

    @property
    def d(self) -> int:
        return self.initial.shape[1]


def examples() -> dict[str, Example]:
    # A Rademacher input gives X^2=1 exactly, matching the old scalar contours.
    def scalar(name: str, target: float) -> Example:
        return Example(
            DiscreteDistribution(name, np.array([[-1.0], [1.0]]),
                                 np.array([-target, target]), np.array([0.5, 0.5])),
            np.array([[0.0], [-1.0]]),
        )

    return {
        "simple": scalar("simple", 1.0),
        "legacy": scalar("legacy", 3.14159),
        # Y is not a multiple of X: there is positive irreducible squared loss.
        "general_l2": Example(
            DiscreteDistribution("general_l2", np.array([[-2.0], [-1.0], [1.0], [2.0]]),
                                 np.array([-1.0, 0.0, 1.0, 1.0]), np.full(4, 0.25)),
            np.array([[0.0], [-1.0]]),
        ),
        # Correlated features, nonzero b in both coordinates, and nonteacher labels.
        "general_l3": Example(
            DiscreteDistribution(
                "general_l3",
                np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.],
                          [1., 1.], [-1., -1.]]),
                np.array([2., -1., -1., 0., 2., -0.5]), np.full(6, 1 / 6),
            ),
            np.array([[-0.8, 0.3], [-0.8, 0.3], [0.8, -0.3]]),
        ),
    }


def population_numbers(example: Example) -> dict:
    """Apply the predraft's Section 6 bounds to exact distribution moments.

    P_min names the ratio inside its definition of R_min. R_max is a radius:
    take the square root of the bound on ||theta||^2 in Lemma 6.4.
    """
    data = example.distribution
    sigma = data.x.T @ (data.probability[:, None] * data.x)
    b = data.x.T @ (data.probability * data.y)
    eigenvalues = np.linalg.eigvalsh(sigma)
    lam, sigma_op = float(eigenvalues[0]), float(eigenvalues[-1])
    if lam <= 0:
        raise ValueError("The input second-moment matrix must be positive definite.")
    nonzero = np.abs(b) > 1e-12
    if not nonzero.any():
        raise ValueError("R_min requires E[XY] != 0; choose another example.")
    p_min = float(np.min(np.abs(b[nonzero])) / sigma_op)
    p_crit = float(np.linalg.norm(b) / lam)
    L, d = example.L, example.d
    return {
        "L": L, "d": d,
        "Sigma_X": sigma.tolist(), "b_E_XY": b.tolist(),
        "E_Y_squared": float(data.probability @ data.y**2),
        "lambda_min": lam, "Sigma_operator_norm": sigma_op,
        "P_min": p_min, "P_crit": p_crit,
        "R_min": float(np.sqrt(L) * p_min ** (1 / L)),
        "R_max": float(np.sqrt(L * d) * p_crit ** (1 / L)),
        "least_squares_product": np.linalg.solve(sigma, b).tolist(),
    }


def population_loss(weights: np.ndarray, eta: float, numbers: dict) -> float | np.ndarray:
    """Exact Gaussian-perturbation average, also accepts a grid of weights."""
    w = np.asarray(weights)
    p = np.prod(w, axis=-2)
    sigma = np.asarray(numbers["Sigma_X"])
    b = np.asarray(numbers["b_E_XY"])
    base = numbers["E_Y_squared"] - 2 * np.sum(b * p, axis=-1)
    base = base + np.einsum("...i,ij,...j->...", p, sigma, p)
    sharpness = np.sum(np.diag(sigma) *
                       (np.prod(w**2 + eta**2, axis=-2) - p**2), axis=-1)
    return base + sharpness


def nested_gradient(weights: np.ndarray, eta: float, data: DiscreteDistribution,
                    outer_batch: int, inner_batch: int,
                    rng: np.random.Generator) -> np.ndarray:
    """One g_k: independent Z_a, and independent xi_(a,s) for each Z_a."""
    if outer_batch < 1 or inner_batch < 1:
        raise ValueError("Both batch sizes must be positive.")
    x, y = data.sample(rng, outer_batch)
    perturbed = weights + eta * rng.standard_normal((outer_batch, inner_batch, *weights.shape))
    products = np.prod(perturbed, axis=2)
    residual = np.sum(products * x[:, None, :], axis=-1) - y[:, None]
    gradients = np.empty_like(perturbed)
    for layer in range(len(weights)):
        # Explicit product avoids division by zero at zero-valued weights.
        other = np.prod(np.delete(perturbed, layer, axis=2), axis=2)
        gradients[:, :, layer, :] = 2 * residual[:, :, None] * x[:, None, :] * other
    return gradients.mean(axis=(0, 1))


def project_ball(weights: np.ndarray, radius: float) -> np.ndarray:
    norm = np.linalg.norm(weights)
    return weights * min(1.0, radius / norm) if norm else weights.copy()


def projected_iteration(example: Example, numbers: dict, *, steps: int, outer_batch: int,
                        inner_batch: int, eta0: float, alpha0: float, radius: float,
                        seed: int) -> tuple[np.ndarray, int]:
    if steps < 1 or eta0 < 0 or alpha0 <= 0 or radius <= numbers["R_max"]:
        raise ValueError("Require steps >= 1, eta0 >= 0, alpha0 > 0, r > R_max.")
    if np.linalg.norm(example.initial) > radius:
        raise ValueError("Initial weights must lie in the projection ball.")
    rng = np.random.default_rng(seed)
    path = np.empty((steps + 1, *example.initial.shape))
    path[0] = example.initial
    projected_steps = 0
    for k in range(steps):
        eta_k = eta0 / (k + 1) ** 0.1
        alpha_k = alpha0 / (k + 1) ** 0.65
        g_k = nested_gradient(path[k], eta_k, example.distribution,
                              outer_batch, inner_batch, rng)
        candidate = path[k] - alpha_k * g_k
        projected_steps += int(np.linalg.norm(candidate) > radius)
        path[k + 1] = project_ball(candidate, radius)
    return path, projected_steps


RADII = ("R_min", "R_max", "projection")


def plot_example(example: Example, path: np.ndarray, numbers: dict, radius: float,
                 paint: tuple[str, ...] | list[str], filename: Path, eta: float) -> None:
    unknown = set(paint) - set(RADII)
    if unknown:
        raise ValueError(f"Unknown radii {sorted(unknown)}; choose from {RADII}.")
    # For L>2 or d>1, coordinates (w_1,1, w_2,1) are a projection.
    two_dimensional = example.initial.size == 2
    xy = np.column_stack((path[:, 0, 0], path[:, 1, 0]))
    fig, axes = plt.subplots(1, 1 if two_dimensional else 2,
                             figsize=(7, 6) if two_dimensional else (12, 5))
    ax = axes if two_dimensional else axes[0]
    displayed = [numbers[key] if key != "projection" else radius for key in paint]
    extent = max(1.2, np.abs(xy).max() * 1.2, *(v * 1.08 for v in displayed))
    grid = np.linspace(-extent, extent, 260)
    u, v = np.meshgrid(grid, grid)
    if two_dimensional:
        # Both coordinates vary over the entire parameter space.
        weights = np.stack((u, v), axis=-1)[..., :, None]
        contours = population_loss(weights, eta, numbers)
        ax.contourf(u, v, contours, 35, cmap="bone", alpha=0.9)
        product = numbers["least_squares_product"][0]
        ax.contour(u, v, u * v, levels=[product], colors=["#78d091"], linewidths=1)
    else:
        # A genuine 2D slice: other coordinates fixed at the initial weights.
        weights = np.broadcast_to(example.initial, u.shape + example.initial.shape).copy()
        weights[..., 0, 0] = u
        weights[..., 1, 0] = v
        ax.contourf(u, v, population_loss(weights, eta, numbers), 35, cmap="bone", alpha=0.9)
    colors = {"R_min": "#4287c2", "R_max": "#ffab45", "projection": "#aa64bd"}
    for key in paint:
        value = radius if key == "projection" else numbers[key]
        # All balls are centered at the origin. For d>1 or L>2 their
        # displayed circles are projections, not tests of full norm.
        ax.add_patch(Circle((0, 0), value, fill=False, lw=2.6 if key == "R_min" else 1.6,
                            linestyle="--" if key == "R_max" else (":" if key == "projection" else "-"),
                            ec=colors[key], label=f"{key} = {value:.3f}"))
    ax.plot(xy[:, 0], xy[:, 1], color="#f25842", lw=1.6, label="projected iterates")
    markers = xy[::max(1, len(xy) // 30)]
    ax.scatter(markers[:, 0], markers[:, 1], s=8, color="#f25842", zorder=3)
    ax.scatter(*xy[0], s=60, c="#52c974", edgecolors="black", zorder=4, label="start")
    ax.scatter(*xy[-1], s=60, c="#ffdf62", edgecolors="black", zorder=4, label="end")
    ax.set(xlabel=r"$w_{1,1}$", ylabel=r"$w_{2,1}$", xlim=(-extent, extent),
           ylim=(-extent, extent), title=example.distribution.name +
           (" (coordinate projection)" if not two_dimensional else ""))
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    if not two_dimensional:
        # The theoretical outer bound can be much larger than the trajectory.
        # Keep a detail view so individual moves remain visible at that scale.
        detail = ax.inset_axes((0.06, 0.55, 0.34, 0.28))
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 0.1)
        center = (xy.min(axis=0) + xy.max(axis=0)) / 2
        detail.plot(xy[:, 0], xy[:, 1], color="#f25842", lw=1.6)
        detail.scatter(*xy[0], c="#52c974", s=28, edgecolors="black", zorder=3)
        detail.scatter(*xy[-1], c="#ffdf62", s=28, edgecolors="black", zorder=3)
        detail.set(xlim=(center[0] - span * .8, center[0] + span * .8),
                   ylim=(center[1] - span * .8, center[1] + span * .8),
                   title="trajectory detail")
        detail.set_aspect("equal")
        detail.tick_params(labelsize=7)
        ax_norm = axes[1]
        norms = np.linalg.norm(path.reshape(len(path), -1), axis=1)
        ax_norm.plot(norms, color="#f25842", label=r"$\|\theta_k\|_2$")
        for key in paint:
            value = radius if key == "projection" else numbers[key]
            ax_norm.axhline(value, color=colors[key],
                            linestyle="--" if key == "R_max" else (":" if key == "projection" else "-"),
                            label=f"{key} = {value:.3f}")
        ax_norm.set(xlabel="iteration k", ylabel=r"full $\|\theta_k\|_2$",
                    title="Full parameter norm")
        ax_norm.legend(fontsize=8)
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def run_example(example: Example, output_dir: Path, *, paint: tuple[str, ...] | list[str],
                steps: int = 1200, outer_batch: int = 32, inner_batch: int = 8,
                eta0: float = 0.7, alpha0: float = 0.08, radius: float | None = None,
                seed: int = 42) -> dict:
    numbers = population_numbers(example)
    radius = radius if radius is not None else numbers["R_max"] + 0.5
    path, projected_steps = projected_iteration(
        example, numbers, steps=steps, outer_batch=outer_batch, inner_batch=inner_batch,
        eta0=eta0, alpha0=alpha0, radius=radius, seed=seed)
    name = example.distribution.name
    plot_example(example, path, numbers, radius, paint, output_dir / f"{name}.png",
                 eta=eta0 / (steps + 1) ** 0.1)
    report = {
        "example": name,
        "distribution": {"X_atoms": example.distribution.x.tolist(),
                         "Y_atoms": example.distribution.y.tolist(),
                         "probabilities": example.distribution.probability.tolist()},
        "population": numbers, "projection_radius": radius,
        "paint": list(paint), "steps": steps, "outer_batch": outer_batch,
        "inner_batch": inner_batch, "seed": seed, "eta0": eta0, "alpha0": alpha0,
        "projected_steps": projected_steps, "last_weights": path[-1].tolist(),
        "last_product": np.prod(path[-1], axis=0).tolist(),
        "maximum_iterate_norm": float(np.linalg.norm(path.reshape(len(path), -1), axis=1).max()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["all", *examples()], default="all")
    parser.add_argument("--paint", nargs="*", choices=RADII, default=list(RADII),
                        help="Radii to draw; all values are computed regardless of this selection.")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--outer-batch", type=int, default=32)
    parser.add_argument("--inner-batch", type=int, default=8)
    parser.add_argument("--eta0", type=float, default=0.7)
    parser.add_argument("--alpha0", type=float, default=0.08)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("plots/projected_examples"))
    args = parser.parse_args()
    selected = examples()
    if args.example != "all":
        selected = {args.example: selected[args.example]}
    for index, example in enumerate(selected.values()):
        report = run_example(example, args.output_dir, paint=args.paint, steps=args.steps,
                             outer_batch=args.outer_batch, inner_batch=args.inner_batch,
                             eta0=args.eta0, alpha0=args.alpha0, radius=args.radius,
                             seed=args.seed + index)
        numbers = report["population"]
        print(f"{report['example']}: lambda_min={numbers['lambda_min']:.6g}, "
              f"P_min={numbers['P_min']:.6g}, P_crit={numbers['P_crit']:.6g}, "
              f"R_min={numbers['R_min']:.6g}, R_max={numbers['R_max']:.6g}, "
              f"r={report['projection_radius']:.6g}; "
              f"projection active on {report['projected_steps']} steps")
        print(f"  {args.output_dir / (report['example'] + '.png')}")
        print(f"  {args.output_dir / (report['example'] + '.json')}")


if __name__ == "__main__":
    main()
