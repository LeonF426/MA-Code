# src/lin_sgd/trainers.py
import torch
from typing import Callable, Dict
from .losses import regularized_loss
import torch.nn as nn
from typing import List
from .layers import DiagLinear  # adjust import to your package layout

def get_layer_matrices(model: nn.Module) -> List[torch.Tensor]:
    """
    Return a list [W1, ..., WL] of weight matrices for the model.
    For DiagLinear layers, returns the full diagonal matrix.
    """
    Ws = []
    for layer in model.layers:  # assumes MixedLinearNet with .layers
        if isinstance(layer, DiagLinear):
            W = torch.diag(layer.weight)  # (d,d)
        elif isinstance(layer, nn.Linear):
            W = layer.weight              # (out_dim, in_dim)
        else:
            continue
        Ws.append(W)
    return Ws

def max_balancing_norm(model: nn.Module) -> float:
    """
    Compute max_l || W_{l+1}^2 - W_l^2 ||_F for the *current* model weights.
    Interprets W^2 as matrix product W @ W (for square layers).
    """
    Ws = get_layer_matrices(model)
    if len(Ws) < 2:
        return 0.0

    norms = []
    for l in range(len(Ws) - 1):
        Wl = Ws[l]
        Wlp1 = Ws[l + 1]

        # interpret W^2 as W @ W (assumes square)
        Wl2 = Wl @ Wl
        Wlp12 = Wlp1 @ Wlp1

        diff = Wlp12 - Wl2
        norms.append(torch.norm(diff, p="fro"))

    max_norm = torch.stack(norms).max()
    return float(max_norm.item())


def gradient_descent_train(
    model,
    data_sampler: Callable[[int], tuple],
    n_steps: int,
    lr_schedule: Callable[[int,float], float],
    eta_schedule: Callable[[int], float],
    batch_size: int,
    device: torch.device,
) -> Dict:
    model.to(device)
    history = {"step": [], "lr": [], "eta": [], "loss": [], "balancedness": []}
    

    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    for k in range(n_steps):
        
        X, Y = data_sampler(batch_size)
        X, Y = X.to(device), Y.to(device)


        eta = eta_schedule(k)
        loss = regularized_loss(model, X, Y, eta)
        loss_scalar = loss.item()
        lr = lr_schedule(k,loss_scalar)

        for pg in optimizer.param_groups:
            pg["lr"] = lr


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        balancing_val = max_balancing_norm(model)

        history["step"].append(k)
        history["lr"].append(float(lr))
        history["eta"].append(float(eta))
        history["loss"].append(loss_scalar)
        history["balancedness"].append(balancing_val)

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

