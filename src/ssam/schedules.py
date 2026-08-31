"""Scalar schedules shared by learning rates and S-SAM sharpness scales."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any


Schedule = Callable[[int], float]
_SCHEDULES: dict[str, Callable[[Mapping[str, Any]], Schedule]] = {}


def register_schedule(name: str, factory: Callable[[Mapping[str, Any]], Schedule]) -> None:
    _SCHEDULES[name.lower()] = factory


def _constant(config: Mapping[str, Any]) -> Schedule:
    value = float(config.get("value", config.get("initial", 0.0)))
    return lambda step: value


def _inverse_time(config: Mapping[str, Any]) -> Schedule:
    initial = float(config.get("initial", config.get("value", 1.0)))
    power = float(config.get("power", 1.0))
    offset = float(config.get("offset", 1.0))
    floor = float(config.get("floor", 0.0))
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


def strong_descent_diag(
    *,
    dimension: int,
    depth: int,
    delta: float,
    safety: float = 1.0,
    max_lr: float | None = None,
    loss_floor: float = 1e-12,
):
    """
    Return alpha_k satisfying

        alpha_k <= 2(1-delta) eta_k^2
                   ---------------------
                   C(d,L) L_R(eta_k, theta_k)

    where

        C(d,L) = sqrt(L) * (
            2 + 2*d*sqrt(L) + 5*sqrt(L-1)
        ).
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
        2.0
        + 2.0 * dimension * sqrt_depth
        + 5.0 * math.sqrt(depth - 1)
    )

    def schedule(
        step: int,
        sharpness_scale: float,
        regularized_loss: float,
    ) -> float:
        del step  # Included for compatibility and logging.

        eta = float(sharpness_scale)
        loss = float(regularized_loss)

        if not math.isfinite(eta) or eta <= 0.0:
            raise ValueError(
                f"eta_k must be positive and finite, received {eta}"
            )

        if not math.isfinite(loss):
            raise FloatingPointError(
                f"L_R must be finite, received {loss}"
            )

        if loss < 0.0:
            raise ValueError(
                f"L_R must be non-negative, received {loss}"
            )

        safe_loss = max(loss, loss_floor)

        upper_bound = (
            2.0
            * (1.0 - delta)
            * eta**2
            / (constant * safe_loss)
        )

        learning_rate = safety * upper_bound

        # Taking the minimum preserves the theorem's bound.
        if max_lr is not None:
            learning_rate = min(learning_rate, max_lr)

        if not math.isfinite(learning_rate):
            raise FloatingPointError(
                f"Computed non-finite learning rate: {learning_rate}"
            )

        return learning_rate

    return schedule



register_schedule("constant", _constant)
register_schedule("inverse_time", _inverse_time)
register_schedule("linear", _linear)
register_schedule("cosine", _cosine)
register_schedule("piecewise", _piecewise)


def build_schedule(config: float | int | Mapping[str, Any]) -> Schedule:
    """Create a schedule from a number or a dictionary."""

    if isinstance(config, (float, int)):
        return _constant({"value": float(config)})
    name = str(config.get("name", "constant")).lower()
    try:
        return _SCHEDULES[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown schedule {name!r}. Available: {sorted(_SCHEDULES)}") from exc
