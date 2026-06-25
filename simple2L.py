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
from ssam.optim_schedules import inverse_time_eta, step_decay_lr, lr_balancing
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
    return (w_star - w2 * w1) ** 2 + eta ** 2 * (w1 ** 2 + w2 ** 2) + eta ** 4

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

def gradient_descent_conv(
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
    Explicitly compare with unregularised loss and ODE trajectories.

    Returns:
      ws: (n_steps+1, 2)  iterates
      etas: list of eta_k
      lrs:  list of lr_k
    """
    ws = np.zeros((n_steps + 1, 2), dtype=float)
    ws_unreg = np.zeros((n_steps + 1, 2), dtype=float)
    ws[0] = np.array(w0, dtype=float)
    ws_unreg[0] = np.array(w0, dtype=float)
    etas = []
    lrs = []
    losses = []
    losses_unreg = []

    for k in range(n_steps):
        w = ws[k]
        w_unreg = ws_unreg[k]

        eta_k = eta_schedule(k)
        loss = F_eta(w=w,eta=eta_k)
        losses.append(loss)

        loss_unreg = F_eta(w=w_unreg,eta=0)
        losses_unreg.append(loss_unreg)

        lr_k = 5*(1-delta)/C * eta_k**2 /loss
        if method=="gd":
            g = grad_F_eta(w, eta_k)
            g_unreg = grad_F_eta(w_unreg, 0)
        elif method=="sgd":
            g = 0
            g_unreg = 0

        ws[k + 1] = w - lr_k * g
        ws_unreg[k+1] = w_unreg - lr_k * g_unreg
        etas.append(eta_k)
        lrs.append(lr_k)

    return ws, ws_unreg, np.array(etas), np.array(lrs), losses, losses_unreg


def gradient_descent_balancing(
    w0,
    n_steps: int,
    L: int,
    eta_schedule,
    lr_schedule,
    method: str = "gd"
):
    """
    Run gradient descent:
    w_{k+1} = w_k - lr_k * grad F_{eta_k}(w_k),
    where lr_k = lr_schedule(k) for balancing condition, min{...}
    eta_k = eta_schedule(k).
    Runs gradient descent for each entry of the min statement to compare 
    and one trajectory for combined min{}.

    Question: Is there implicit balancing via the learning rate?

    Returns:
      ws: (n_steps+1, 2)  iterates
      etas: list of eta_k
      lrs:  list of lr_k
    """

    ws = {"lr": np.zeros((n_steps + 1, 2), dtype=float),
          "lr_1": np.zeros((n_steps+1,2),dtype=float),
          "lr_2": np.zeros((n_steps+1,2),dtype=float),
          "lr_3": np.zeros((n_steps+1,2),dtype=float)}

    for key in ws.keys():
        ws[key][0] = np.array(w0, dtype=float)

    etas = []
    lrs = {"lr": [], "lr_1": [], "lr_2": [], "lr_3": []}
    losses = {"lr": [], "lr_1": [], "lr_2": [], "lr_3": []}

    for k in range(n_steps):

        eta_k = eta_schedule(k)

        for key in ws.keys():
            w = ws[key][k]

            loss = F_eta(w=w,eta=eta_k)
            losses[key].append(loss)


            # lr_k = 5*(1-delta)/C * eta_k**2 /loss
            lr_ks = lr_schedule(k=k,eta_k=eta_k,loss=loss)
            lr_k = lr_ks[key]

            if method=="gd":
                g = grad_F_eta(w, eta_k)
            elif method=="sgd":
                g = 0

            ws[key][k + 1] = w - lr_k * g
            etas.append(eta_k)
            lrs[key].append(lr_k)

    return ws,  np.array(etas), lrs, losses

# 5. Glue everything together
# ---------------------------------------------------------------------

def make_contour_with_two_paths(
    eta,
    path1,
    path2,
    w_star,
    level_color="#00FF00",
    xlim=(-2.5, 2.5),
    ylim=(-2.5, 2.5),
    num_grid=500,
    levels=100,
    cmap="RdPu_r",
    marker_every=10,
    title=None,
    filename=None,
    label1="Path 1",
    label2="Path 2",
    level_label=None,
    level_linewidth=1.2,
):
    w1_vals = np.linspace(xlim[0], xlim[1], num_grid)
    w2_vals = np.linspace(ylim[0], ylim[1], num_grid)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    Z = F_eta((W1, W2), eta)

    fig, ax = plt.subplots(figsize=(5, 5))

    # filled contours of F_eta
    ax.contourf(W1, W2, Z, levels=levels, cmap=cmap)

    # level set: w1 * w2 = w_star
    levelset = ax.contour(
        W1,
        W2,
        W1 * W2,
        levels=[w_star],
        colors=[level_color],
        linewidths=level_linewidth,
        linestyles="solid",
    )

    if level_label is None:
        level_label = fr"$w_1 w_2 = {w_star:.3g}$"

    # dummy line for legend entry of the level set
    if path1 != None:
        ax.plot(
            [],
            [],
            color=level_color,
            linewidth=level_linewidth,
            label=level_label,
        )

        # ----- Path 1 -----
        ax.plot(
            path1[:, 0],
            path1[:, 1],
            color="navy",
            linewidth=1.5,
            alpha=0.9,
            label=label1,
        )

        indices1 = np.arange(0, len(path1), marker_every)

        ax.plot(
            path1[indices1, 0],
            path1[indices1, 1],
            marker="o",
            markersize=4,
            color="cyan",
            linestyle="",
        )

        ax.scatter(
            path1[0, 0],
            path1[0, 1],
            color="lime",
            s=50,
            zorder=5,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.scatter(
            path1[-1, 0],
            path1[-1, 1],
            color="red",
            s=50,
            zorder=5,
            edgecolor="black",
            linewidth=0.5,
        )

    # ----- Path 2 -----
    if path2 != None:
        ax.plot(
            path2[:, 0],
            path2[:, 1],
            color="darkorange",
            linewidth=1.5,
            alpha=0.9,
            label=label2,
        )

        indices2 = np.arange(0, len(path2), marker_every)

        ax.plot(
            path2[indices2, 0],
            path2[indices2, 1],
            marker="s",
            markersize=4,
            color="yellow",
            linestyle="",
        )

        ax.scatter(
            path2[0, 0],
            path2[0, 1],
            color="lime",
            s=50,
            zorder=5,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.scatter(
            path2[-1, 0],
            path2[-1, 1],
            color="red",
            s=50,
            zorder=5,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel(r"$w_1$", fontsize=11)
        ax.set_ylabel(r"$w_2$", fontsize=11)

    if title is None:
        title = fr"$\eta = {eta:.2f}$"

    ax.set_title(title, fontsize=12, pad=10)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    ax.legend(frameon=True, fontsize=9)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=250, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


def main_conv():
    n_steps = 150
    snapshot_every = 10
    w0 = [1.0, -2.0]
    L=2
    d=1
    eta0=2.5
    # plug in your schedulers here
    eta_schedule = inverse_time_eta(eta0=eta0, alpha=1/2)
    delta = 1/2
    C = math.sqrt(L)*(2+2*math.sqrt(L)*d + 5*math.sqrt(L-1)) 
    LR0 = F_eta(w=w0,eta=eta0)
    print("first loss :",LR0)
    print("constant C:", C)

    ws, ws_unreg,  etas, lrs, losses, losses_unreg = gradient_descent_conv(
        w0=w0,
        n_steps=n_steps,
        delta = delta,
        C=C,
        eta_schedule=eta_schedule,
        method="gd"
    )
    # print(etas)
    # print(lrs[-10:])
    # print(losses)
    # print(losses)
    # print(losses_unreg)

    lr_schedule = lr_balancing(L=L,lamb=1.0)

    ws_b, etas_b, lrs_b, losses_b = gradient_descent_balancing(
        w0=w0,
        L=2,
        n_steps=n_steps,
        eta_schedule=eta_schedule,
        lr_schedule=lr_schedule,
        method="gd"
    )
    key = "lr"
    ws_b = ws_b[key]
    lrs_b = lrs_b[key]
    losses_b = losses_b[key]


    for k in range(0, n_steps + 1, snapshot_every):
        eta_k = etas[k] if k < len(etas) else etas[-1]
        alpha_k = lrs[k] if k < len(lrs) else lrs[-1]
        alpha_k_b = lrs_b[k] if k < len(lrs_b) else lrs_b[-1]

        sub_path1 = ws[: k + 1]
        sub_path2 = ws_b[: k + 1]
        title = fr"$\eta_k = {eta_k:.2f}$ and $\alpha_k = {alpha_k:.2f}$,$\alpha_b = {alpha_k_b: .2f} step {k}$"
        filename = f"./plots/balance/eta_sched_step_{k:04d}.png"

        make_contour_with_two_paths(
            eta=eta_k,
            path1=sub_path1,
            path2=sub_path2,
            w_star=w_star,
            level_color="#00FF00",
            title=title,
            filename=filename,
            label1="regularised",
            label2="balancing",
            marker_every=snapshot_every
        )

def main_balance():

    n_steps = 10
    snapshot_every = 10
    w0 = [1.0, -2.0]
    L=2
    d=1
    eta0=2.5
    # plug in your schedulers here
    eta_schedule = inverse_time_eta(eta0=eta0, alpha=1/2)
    LR0 = F_eta(w=w0,eta=eta0)

    lr_schedule = lr_balancing(L=L,lamb=1.0)

    ws, etas, lrs, losses = gradient_descent_balancing(
        w0=w0,
        L=2,
        n_steps=n_steps,
        eta_schedule=eta_schedule,
        lr_schedule=lr_schedule,
        method="gd"
    )
    key = "lr"
    ws = ws[key]
    lrs = lrs[key]
    losses = losses[key]

if __name__ == "__main__":
    main_conv()
