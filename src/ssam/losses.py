# src/lin_sgd/losses.py
import torch
from typing import List
from .models import MixedLinearNet
from .layers import DiagLinear

def extract_weight_matrices(model: MixedLinearNet) -> List[torch.Tensor]:
    Ws = []
    for layer in model.layers:
        if isinstance(layer, DiagLinear):
            W = torch.diag(layer.weight)
        else:
            W = layer.weight
        Ws.append(W)
    return Ws

def regularized_loss(model, X, Y, eta: float) -> torch.Tensor:
    device = X.device
    pred = model(X)
    L_emp = ((Y - pred) ** 2).mean()

    Ws = extract_weight_matrices(model)
    P_no_noise = torch.eye(Ws[0].shape[0], device=device)
    P_noise = torch.eye(Ws[0].shape[0], device=device)

    for W in Ws:
        W2 = W @ W
        P_no_noise = P_no_noise @ W2
        P_noise = P_noise @ (W2 + eta**2 * torch.eye(W2.shape[0], device=device))

    diag_A = torch.diag(P_noise - P_no_noise)
    X_sq = X**2
    R_emp = (X_sq * diag_A.unsqueeze(0)).sum(dim=1).mean()

    return L_emp + R_emp

def exact_L_R_for_diagonal_model(
    model,          # diagonal-only network
    eta: float,
    mu_X: torch.Tensor,          # (d,)
    mu_Y: float,
    Sigma_xx: torch.Tensor,      # (d, d)
    Sigma_xy: torch.Tensor,      # (d,)
    sigma_Y2: float,
) -> torch.Tensor:
    """
    Exact L_R(eta, theta) using known 1st/2nd moments.
    Assumes model is a *purely diagonal* linear network (all layers DiagLinear).
    """
    device = next(model.parameters()).device
    mu_X = mu_X.to(device)
    Sigma_xx = Sigma_xx.to(device)
    Sigma_xy = Sigma_xy.to(device)

    # 1) extract diagonal weights: list w_1,...,w_L, each shape (d,)
    diag_ws = []
    for layer in model.layers:
        # here we assume they are all DiagLinear
        diag_ws.append(layer.weight)

    L = len(diag_ws)
    d = diag_ws[0].shape[0]

    # a_i(theta) = prod_l w_{l,i}
    a = torch.ones(d, device=device)
    for w in diag_ws:
        a = a * w

    # --- data part ---
    # centered version:
    data_loss = (
        sigma_Y2
        - 2.0 * (a @ Sigma_xy)
        + (a.unsqueeze(0) @ Sigma_xx @ a.unsqueeze(1)).squeeze()
    )
    # non-centered correction:
    data_loss = data_loss + (mu_Y - a @ mu_X) ** 2

    # --- regularizer part ---
    b = torch.ones(d, device=device)  # prod w_{l,i}^2
    c = torch.ones(d, device=device)  # prod (w_{l,i}^2 + eta^2)
    for w in diag_ws:
        w2 = w**2
        b = b * w2
        c = c * (w2 + eta**2)

    EX2 = torch.diag(Sigma_xx) + mu_X**2   # E[X_i^2]
    R = ((c - b) * EX2).sum()

    return data_loss + R

def compute_initial_LR(
    model,
    data_sampler,
    eta0: float,
    batch_size: int,
    device: torch.device,
) -> float:
    """
    Approximate L_R(eta0, theta0) once, at initialization.
    """
    model.to(device)
    model.eval()  # no dropout etc., just for consistency

    X0, Y0 = data_sampler(batch_size)
    X0, Y0 = X0.to(device), Y0.to(device)

    with torch.no_grad():
        L_R0 = regularized_loss(model, X0, Y0, eta0)

    return float(L_R0.item())
