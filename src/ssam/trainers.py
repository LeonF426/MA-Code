# src/lin_sgd/trainers.py
import torch
from typing import Callable, Dict
from .losses import regularized_loss

def gradient_descent_train(
    model,
    data_sampler: Callable[[int], tuple],
    n_steps: int,
    lr_schedule: Callable[[int], float],
    eta_schedule: Callable[[int], float],
    batch_size: int,
    device: torch.device,
) -> Dict:
    model.to(device)
    history = {"step": [], "lr": [], "eta": [], "loss": []}

    for k in range(n_steps):
        lr = lr_schedule(k)
        eta = eta_schedule(k)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        X, Y = data_sampler(batch_size)
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        loss = regularized_loss(model, X, Y, eta)
        loss.backward()
        optimizer.step()

        history["step"].append(k)
        history["lr"].append(lr)
        history["eta"].append(eta)
        history["loss"].append(loss.item())

    return history


def sgd_with_weight_noise(
    model,
    data_sampler,
    n_steps: int,
    lr_schedule,
    eta_schedule,
    batch_size: int,
    device: torch.device,
):
    """
    Noisy SGD: at step k, use learning rate lr_schedule(k) and noise std eta_schedule(k).
    The gradient is computed at a perturbed parameter vector, but the update is
    applied to the clean parameters.

    data_sampler(bs) -> (X, Y) on the correct device or moved below.
    """
    model.to(device)

    # Create optimizer ONCE; actual lr will be overwritten every step.
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    history = {"step": [], "lr": [], "eta": [], "loss": []}

    for k in range(n_steps):
        lr_k = lr_schedule(k)
        eta_k = eta_schedule(k)

        # ensure optimizer actually uses lr_k at this step
        for pg in optimizer.param_groups:
            pg["lr"] = lr_k

        X, Y = data_sampler(batch_size)
        X, Y = X.to(device), Y.to(device)

        # Save clean params
        saved_params = [p.data.clone() for p in model.parameters()]

        # Add Gaussian perturbation: θ̃_k = θ_k + ξ_k,  ξ_k ~ N(0, η_k^2 I)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * eta_k)

        optimizer.zero_grad()
        pred = model(X)
        loss = ((Y - pred) ** 2).mean()
        loss.backward()

        # Restore clean θ_k
        with torch.no_grad():
            for p, saved in zip(model.parameters(), saved_params):
                p.data.copy_(saved)

        # Now apply SGD update using ∇_θ ℓ(θ̃_k)
        optimizer.step()

        history["step"].append(k)
        history["lr"].append(lr_k)
        history["eta"].append(eta_k)
        history["loss"].append(loss.item())

    return history

