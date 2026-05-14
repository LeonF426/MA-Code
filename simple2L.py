import numpy as np
import sys
import os
# Add the project's src/ directory to sys.path
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
import math
import matplotlib.pyplot as plt
from pathlib import Path
from ssam.optim_schedules import inverse_time_eta, step_decay_lr
# ---------------------------------------------------------------------
# 2. Loss and gradient
# ---------------------------------------------------------------------

w_star = 3.14159

def F_eta(w, eta):
    """
    w: array-like shape (2,)
    F_eta(w1, w2) = (w_* - w2*w1)^2 + eta^2 (w1^2 + w2^2)
    """
    w1, w2 = w
    return (w_star - w2 * w1) ** 2 + eta ** 2 * (w1 ** 2 + w2 ** 2)

def grad_F_eta(w, eta):
    """
    Analytic gradient of F_eta.
    """
    w1, w2 = w
    g = w_star - w2 * w1
    d_w1 = -2.0 * g * w2 + 2.0 * eta ** 2 * w1
    d_w2 = -2.0 * g * w1 + 2.0 * eta ** 2 * w2
    return np.array([d_w1, d_w2], dtype=float)

# ---------------------------------------------------------------------
# 3. Descent loop with LR + eta scheduler
# ---------------------------------------------------------------------

def run_descent_with_schedulers(
    w0,
    n_steps: int,
    delta:float,
    C: float,
    eta_schedule,
    method: str = "gd"
):
    """
    Run (noisy) gradient descent:
    w_{k+1} = w_k - lr_k * grad F_{eta_k}(w_k),
    where lr_k = lr_schedule(k), eta_k = eta_schedule(k).

    Returns:
      ws: (n_steps+1, 2)  iterates
      etas: list of eta_k
      lrs:  list of lr_k
    """
    ws = np.zeros((n_steps + 1, 2), dtype=float)
    ws[0] = np.array(w0, dtype=float)
    etas = []
    lrs = []
    losses = []

    for k in range(n_steps):
        w = ws[k]
        eta_k = eta_schedule(k)
        loss = F_eta(w=w,eta=eta_k)
        losses.append(loss)
        lr_k = 2*(1-delta)/C * eta_k**2 /loss
        if method=="gd":
            g = grad_F_eta(w, eta_k)
        elif method=="sgd":
            g = 0

        ws[k + 1] = w - lr_k * g
        etas.append(eta_k)
        lrs.append(lr_k)

    return ws, np.array(etas), np.array(lrs), losses

# ---------------------------------------------------------------------
# 4. Plotting snapshots (using current eta_k at each snapshot)
# ---------------------------------------------------------------------

def make_contour_with_path(
    eta,
    path,
    xlim=(-2.5, 2.5),
    ylim=(-2.5, 2.5),
    num_grid=500,
    levels=100,
    cmap="RdPu_r",
    marker_every=10,
    title=None,
    filename=None,
):
    w1_vals = np.linspace(xlim[0], xlim[1], num_grid)
    w2_vals = np.linspace(ylim[0], ylim[1], num_grid)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    Z = F_eta((W1, W2), eta)

    fig, ax = plt.subplots(figsize=(5, 5))
    
    # filled contours with many levels for smooth gradient
    ax.contourf(W1, W2, Z, levels=levels, cmap=cmap)
    
    # trajectory: line + markers every marker_every points
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="navy",
        linewidth=1.5,
        alpha=0.9,
    )
    # markers at regular intervals
    indices = np.arange(0, len(path), marker_every)
    ax.plot(
        path[indices, 0],
        path[indices, 1],
        marker="o",
        markersize=4,
        color="cyan",
        linestyle="",
    )
    # start/end markers
    ax.scatter(path[0, 0], path[0, 1], color="lime", s=50, zorder=5, edgecolor="black", linewidth=0.5)
    ax.scatter(path[-1, 0], path[-1, 1], color="red", s=50, zorder=5, edgecolor="black", linewidth=0.5)

    ax.set_xlabel(r"$w_1$", fontsize=11)
    ax.set_ylabel(r"$w_2$", fontsize=11)
    
    if title is None:
        title = fr"$\eta = {eta:.2f}$"
    ax.set_title(title, fontsize=12, pad=10)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=250, bbox_inches="tight")
        plt.close(fig)
# 5. Glue everything together
# ---------------------------------------------------------------------

def main():
    n_steps = 200
    snapshot_every = 100
    w0 = [1.0, -2.0]
    L=2
    d=1
    # plug in your schedulers here
    eta_schedule = inverse_time_eta(eta0=5, alpha=1/2)
    delta = 1/10
    C = math.sqrt(L)*(2+2*math.sqrt(L)*d + 2*math.sqrt(L-1)) 
    LR0 = F_eta(w=w0,eta=5)
    print("first loss :",LR0)
    print("constant C:", C)
    lr_schedule =  step_decay_lr(lr0= (C*LR0)/2,delta=1/2, eta_sched=eta_schedule)

    ws, etas, lrs, losses = run_descent_with_schedulers(
        w0=w0,
        n_steps=n_steps,
        delta = delta,
        C=C,
        eta_schedule=eta_schedule,
        method="gd"
    )
    # print(etas)
    print(lrs[-10:])
    # print(losses)

    for k in range(0, n_steps + 1, snapshot_every):
        eta_k = etas[k] if k < len(etas) else etas[-1]
        alpha_k = lrs[k] if k < len(lrs) else lrs[-1]

        sub_path = ws[: k + 1]
        title = fr"$\eta_k = {eta_k:.2f}$ and $\alpha_k = {alpha_k:.2f}$, step {k}"
        filename = f"./plots/ll/eta_sched_step_{k:04d}.png"

        make_contour_with_path(
            eta=eta_k,
            path=sub_path,
            title=title,
            filename=filename,
            marker_every=15,
        )

    # final snapshot
    if n_steps not in range(0, n_steps + 1, snapshot_every):
        eta_k = etas[-1]
        alpha_k = lrs[-1]
        title = fr"$\eta_k = {eta_k:.2f}$ and $\alpha_k = {alpha_k:.2f}$, step {n_steps}"
        filename =f"./plots/ll/eta_sched_step_{n_steps:04d}.png"
        make_contour_with_path(
            eta=eta_k,
            path=ws,
            title=title,
            filename=filename,
            marker_every=15,
        )

if __name__ == "__main__":
    main()
