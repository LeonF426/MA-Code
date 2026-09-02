"""Step schedules and state-dependent learning-rate policies."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


Schedule = Callable[[int], float]
LearningRateFunction = Callable[[int, float, float | None], float]
_SCHEDULES: dict[str, Callable[[Mapping[str, Any]], Schedule]] = {}


def register_schedule(name: str, factory: Callable[[Mapping[str, Any]], Schedule]) -> None:
    """Register a scalar, step-only schedule."""

    _SCHEDULES[name.lower()] = factory


def _constant(config: Mapping[str, Any]) -> Schedule:
    value = float(config.get("value", config.get("initial", 0.0)))
    return lambda step: value


def _inverse_time(config: Mapping[str, Any]) -> Schedule:
    initial = float(config.get("initial", config.get("value", 1.0)))
    power = float(config.get("power", 1.0))
    offset = float(config.get("offset", 1.0))
    floor = float(config.get("floor", 0.0))
    if offset <= 0.0:
        raise ValueError("inverse_time.offset must be positive")
    return lambda step: floor + initial / (offset + step) ** power


def _linear(config: Mapping[str, Any]) -> Schedule:
    start = float(config.get("start", config.get("initial", 1.0)))
    end = float(config.get("end", 0.0))
    duration = max(1, int(config["duration"]))
    return lambda step: start + (end - start) * min(step, duration) / duration


def _cosine(config: Mapping[str, Any]) -> Schedule:
    initial = float(config.get("initial", 1.0))
    final = float(config.get("final", 0.0))
    duration = max(1, int(config["duration"]))

    def schedule(step: int) -> float:
        progress = min(step, duration) / duration
        return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * progress))

    return schedule


def _piecewise(config: Mapping[str, Any]) -> Schedule:
    points = sorted((int(step), float(value)) for step, value in config["values"])
    if not points:
        raise ValueError("A piecewise schedule requires at least one [step, value] pair")

    def schedule(step: int) -> float:
        value = points[0][1]
        for boundary, candidate in points:
            if step < boundary:
                break
            value = candidate
        return value

    return schedule


register_schedule("constant", _constant)
register_schedule("inverse_time", _inverse_time)
register_schedule("linear", _linear)
register_schedule("cosine", _cosine)
register_schedule("piecewise", _piecewise)


def build_schedule(config: float | int | Mapping[str, Any]) -> Schedule:
    """Create a scalar schedule from a number or dictionary."""

    if isinstance(config, (float, int)):
        return _constant({"value": float(config)})
    name = str(config.get("name", "constant")).lower()
    try:
        return _SCHEDULES[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown schedule {name!r}. Available: {sorted(_SCHEDULES)}") from exc


@dataclass(frozen=True)
class LearningRatePolicy:
    """Uniform interface for static and objective-dependent learning rates."""

    name: str
    requires_regularized_loss: bool
    function: LearningRateFunction

    def __call__(
        self,
        step: int,
        sharpness_scale: float,
        regularized_loss: float | None = None,
    ) -> float:
        if self.requires_regularized_loss and regularized_loss is None:
            raise ValueError(
                f"Learning-rate policy {self.name!r} requires a regularized-loss estimate"
            )
        value = float(self.function(step, sharpness_scale, regularized_loss))
        if not math.isfinite(value) or value < 0.0:
            raise FloatingPointError(f"Learning-rate policy returned invalid value {value}")
        return value


def strong_descent_diag(
    *,
    dimension: int,
    depth: int,
    delta: float,
    safety: float = 1.0,
    max_lr: float | None = None,
    loss_floor: float = 1e-12,
) -> LearningRateFunction:
    """Build the state-dependent upper bound from Theorem 6.1.1.

    For non-diagonal or Monte Carlo-trained models this remains a useful adaptive
    policy, but the theorem's deterministic guarantee does not automatically carry
    over.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if depth < 1:
        raise ValueError("depth must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 < safety <= 1.0:
        raise ValueError("safety must be in (0, 1]")
    if loss_floor <= 0.0:
        raise ValueError("loss_floor must be positive")
    if max_lr is not None and max_lr <= 0.0:
        raise ValueError("max_lr must be positive")

    sqrt_depth = math.sqrt(depth)
    constant = sqrt_depth * (
        2.0 + 2.0 * dimension * sqrt_depth + 5.0 * math.sqrt(depth - 1)
    )

    def policy(
        step: int,
        sharpness_scale: float,
        regularized_loss: float | None,
    ) -> float:
        del step
        eta = float(sharpness_scale)
        if regularized_loss is None:
            raise ValueError("strong_descent_diag requires the current regularized loss")
        loss = float(regularized_loss)
        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError(f"eta_k must be positive and finite, received {eta}")
        if not math.isfinite(loss):
            raise FloatingPointError(f"L_R must be finite, received {loss}")
        if loss < 0.0:
            raise ValueError(f"L_R must be non-negative, received {loss}")

        upper_bound = 2.0 * (1.0 - delta) * eta**2 / (
            constant * max(loss, loss_floor)
        )
        learning_rate = safety * upper_bound
        return min(learning_rate, max_lr) if max_lr is not None else learning_rate

    return policy


def build_learning_rate_policy(
    config: float | int | Mapping[str, Any],
    *,
    default_dimension: int | None = None,
    default_depth: int | None = None,
) -> LearningRatePolicy:
    """Build a learning-rate policy with one uniform three-argument interface."""

    if isinstance(config, Mapping):
        name = str(config.get("name", "constant")).lower()
    else:
        name = "constant"

    if name == "tamed":
         t = config["inserted_lr"].get("name", "constant")


    if name == "strong_descent_diag":
        if not isinstance(config, Mapping):
            raise TypeError("strong_descent_diag requires a configuration dictionary")
        dimension_value = config.get("dimension", default_dimension)
        depth_value = config.get("depth", default_depth)
        if dimension_value is None or depth_value is None:
            raise ValueError(
                "strong_descent_diag requires dimension and depth, either explicitly "
                "or inferable from the configured model"
            )
        function = strong_descent_diag(
            dimension=int(dimension_value),
            depth=int(depth_value),
            delta=float(config["delta"]),
            safety=float(config.get("safety", 1.0)),
            max_lr=(None if config.get("max_lr") is None else float(config["max_lr"])),
            loss_floor=float(config.get("loss_floor", 1e-12)),
        )
        return LearningRatePolicy(name, True, function)

    scalar_schedule = build_schedule(config)
    return LearningRatePolicy(
        name,
        False,
        lambda step, sharpness_scale, regularized_loss: scalar_schedule(step),
    )

