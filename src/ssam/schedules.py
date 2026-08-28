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
