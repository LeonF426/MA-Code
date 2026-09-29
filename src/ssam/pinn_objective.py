"""Selective clean and Gaussian-averaged objectives for PINN components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


PINN_LOSS_COMPONENTS = frozenset({"pde", "boundary"})


def resolve_sharpness_components(config: Mapping[str, Any]) -> frozenset[str]:
    """Return PINN components trained with the Gaussian-averaged objective."""

    configured = config.get("pinn", {}).get(
        "sharpness_components",
        ("pde", "boundary"),
    )
    if isinstance(configured, str):
        configured = (
            ("pde", "boundary")
            if configured.lower() == "all"
            else (configured,)
        )
    try:
        selected = frozenset(str(name).lower() for name in configured)
    except TypeError as exc:
        raise TypeError(
            "pinn.sharpness_components must be a string or a sequence"
        ) from exc
    unknown = selected - PINN_LOSS_COMPONENTS
    if unknown:
        raise ValueError(
            "Unknown pinn.sharpness_components values: "
            f"{sorted(unknown)}; expected 'pde' and/or 'boundary'"
        )
    if not selected:
        raise ValueError("pinn.sharpness_components must not be empty")
    return selected


def configure_selective_pinn_loss(
    components: Mapping[str, torch.Tensor],
    *,
    pde_weight: float,
    boundary_weight: float,
    sharpness_components: frozenset[str],
) -> dict[str, torch.Tensor]:
    """Add objectives for selected perturbed and unselected clean components."""

    result = dict(components)
    if sharpness_components == PINN_LOSS_COMPONENTS:
        return result

    weighted = {
        "pde": pde_weight * result["pde_loss"],
        "boundary": boundary_weight * result["boundary_loss"],
    }
    clean_components = PINN_LOSS_COMPONENTS - sharpness_components
    result["sharpness_loss"] = sum(
        (weighted[name] for name in sharpness_components),
        torch.zeros_like(result["loss"]),
    )
    result["clean_only_loss"] = sum(
        (weighted[name] for name in clean_components),
        torch.zeros_like(result["loss"]),
    )
    return result
