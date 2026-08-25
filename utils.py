import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.integrate import solve_ivp

# Add the project's src/ directory to sys.path
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from ssam.optim_schedules import inverse_time_eta
except ImportError:
    # Fallback if ssam is unavailable
    def inverse_time_eta(eta0, alpha,eta1):
        return lambda k: eta0 / (1.0 + k) ** alpha + eta1

def constant_eta(eta0):
    return lambda k: eta0
# ---------------------------------------------------------------------
# 1. Loss and gradient
# ---------------------------------------------------------------------

w_star = 3.14159


def F_eta(w, eta):
    """
    F_eta(w1, w2) = (w_* - w2*w1)^2 + eta^2 (w1^2 + w2^2)
    """
    w1, w2 = w
    return (w_star - w2 * w1) ** 2 + eta ** 2 * (w1 ** 2 + w2 ** 2) + eta**4


def grad_F_eta(w, eta):
    """
    Analytic gradient of F_eta.
    """
    w1, w2 = w
    g = w_star - w2 * w1

    d_w1 = -2.0 * g * w2 + 2.0 * eta ** 2 * w1
    d_w2 = -2.0 * g * w1 + 2.0 * eta ** 2 * w2

    return np.array([d_w1, d_w2], dtype=float)

def explicit_euler_stepsize_bound(w, eta, w_star, safety=0.95):
    """
    Paper's explicit-Euler stability bound evaluated at the current iterate.

    safety < 1 keeps the step strictly inside the stability region.
    """
    w1, w2 = np.asarray(w, dtype=float)

    denominator = (
        w1**2
        + w2**2
        + 2.0 * eta**2
        + np.sqrt(
            (w1**2 - w2**2) ** 2
            + (4.0 * w1 * w2 - w_star) ** 2
        )
    )

    if denominator <= 0.0:
        return np.inf

    return safety * 2.0 / denominator


# ---------------------------------------------------------------------
# 2. Discrete GD paths
# ---------------------------------------------------------------------

def run_descent_with_schedulers(
    w0,
    n_steps: int,
    delta: float,
    C: float,
    eta_schedule,
    lab: float = 1.0,
    L : int = 2,
    method: str = "sdc",
):
    """
    Computes two GD paths:

    1. Regularized GD:
       w_{k+1} = w_k - alpha_k * grad F_{eta_k}(w_k)

    2. Unregularized GD:
       u_{k+1} = u_k - alpha_k * grad F_0(u_k)

    Important:
    The unregularized GD path uses the same alpha_k sequence as the
    regularized path. This matches your original script.
    """

    ws_reg = np.zeros((n_steps + 1, 2), dtype=float)
    ws_unreg = np.zeros((n_steps + 1, 2), dtype=float)

    ws_reg[0] = np.array(w0, dtype=float)
    ws_unreg[0] = np.array(w0, dtype=float)

    etas = np.zeros(n_steps + 1, dtype=float)
    lrs = np.zeros(n_steps, dtype=float)

    losses_reg = []
    losses_unreg = []

    for k in range(n_steps):
        w_reg = ws_reg[k]
        w_unreg = ws_unreg[k]

        eta_k = eta_schedule(k)
        etas[k] = eta_k

        loss_reg = F_eta(w_reg, eta_k)
        loss_unreg = F_eta(w_unreg, 0.0)

        losses_reg.append(loss_reg)
        losses_unreg.append(loss_unreg)



        # Strong Descent Condition
        alpha_k_sdc = (
            2.0 * (1.0 - delta)
            / C
            * eta_k**2
            / loss_reg
        )

# Balancing Condition
        a_k_b_1 = eta_k**2 / (4.0 * loss_reg)
        a_k_b_2 = 3.0 * lab * eta_k ** (2 * L - 2) * a_k_b_1**2
        a_k_b_3 = (lab * eta_k ** (2 * L - 2)) ** (-1)

        alpha_k_balancing = np.min([
            a_k_b_1,
            a_k_b_2,
            a_k_b_3,
        ])

# Explicit-Euler stability bound at the current regularized iterate
        alpha_k_euler = explicit_euler_stepsize_bound(
            w=w_reg,
            eta=eta_k,
            w_star=w_star,
            safety=0.95,
        )

        if method == "sdc":
            alpha_k = alpha_k_sdc
        elif method == "balancing":
            alpha_k = alpha_k_balancing
        elif method == "explicit_euler_bound":
            alpha_k = alpha_k_euler
        else:
            raise ValueError(
                "method must be 'sdc', 'balancing', or 'explicit_euler_bound'"
            )

    etas[n_steps] = eta_schedule(n_steps)

    return ws_reg, ws_unreg, etas, lrs, np.array(losses_reg), np.array(losses_unreg)


# ---------------------------------------------------------------------
# 3. ODE gradient flow
# ---------------------------------------------------------------------

def ode_rhs(t, theta, eta_func):
    """
    Continuous-time gradient flow:

        d theta / dt = - grad F_{eta(t)}(theta)
    """
    eta_t = eta_func(t)
    return -grad_F_eta(theta, eta_t)


def solve_gradient_flow_ode(
    theta0,
    eta_func,
    t_span,
    t_eval=None,
    rtol=1e-9,
    atol=1e-11,
):
    """
    Solve the gradient flow ODE.
    """
    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, eta_func),
        t_span=t_span,
        y0=np.array(theta0, dtype=float),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        t_eval=t_eval,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    return sol


def make_inverse_time_eta_func(eta0,eta1, alpha):
    """
    Continuous version of eta_k = eta0 / (1 + k)^alpha.
    """
    def eta_func(t):
        return eta0 / (1.0 + t) ** alpha + eta1

    return eta_func

# ---------------------------------------------------------------------
# 4. Plotting function: contour + arbitrary dict of paths
# ---------------------------------------------------------------------

def plot_contour_with_paths(
    eta,
    paths,
    w_star,
    level_color,
    xlim=(-2.0, 2.0),
    ylim=(-2.0, 2.0),
    num_grid=500,
    levels=50,
    cmap="RdPu_r",
    marker_every=25,
    title=None,
    filename=None,
    level_label=None,
    level_linewidth=1.2,
    path_styles=None,
):

    if path_styles is None:
        path_styles = {}

    default_colors = [
        "navy",
        "darkorange",
        "black",
        "purple",
        "teal",
        "crimson",
        "brown",
        "gray",
    ]

    default_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    w1_vals = np.linspace(xlim[0], xlim[1], num_grid)
    w2_vals = np.linspace(ylim[0], ylim[1], num_grid)

    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = F_eta((W1, W2), eta)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Filled contours of F_eta
    ax.contourf(W1, W2, Z, levels=levels, cmap=cmap, alpha=0.8)
    if level_color != None:
        # Level set: w1 * w2 = w_star
        ax.contour(
            W1,
            W2,
            W1 * W2,
            levels=[w_star],
            colors=[level_color],
            linewidths=level_linewidth,
            linestyles="solid",
        )

        if level_label is None:
            level_label = fr"$w_1 w_2 = w_*$"

        # Dummy artist for level set legend
        ax.plot(
            [],
            [],
            color=level_color,
            linewidth=level_linewidth,
            label=level_label,
        )
    # Plot all paths
    for i, (label, path) in enumerate(paths.items()):
        path = np.asarray(path, dtype=float)

        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError(
                f"Path '{label}' must have shape (N, 2), got {path.shape}."
            )

        if len(path) == 0:
            continue

        style = path_styles.get(label, {})

        color = style.get("color", default_colors[i % len(default_colors)])
        marker = style.get("marker", default_markers[i % len(default_markers)])
        linewidth = style.get("linewidth", 1.8)
        alpha = style.get("alpha", 0.9)
        linestyle = style.get("linestyle", "-")

        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=label,
        )

        if isinstance(marker_every, dict):
            me = marker_every.get(label, 25)
        else:
            me = marker_every

        me = max(int(me), 1)
        indices = np.arange(0, len(path), me)

        ax.plot(
            path[indices, 0],
            path[indices, 1],
            marker=marker,
            markersize=4,
            color=color,
            linestyle="",
            alpha=alpha,
        )

        # Start point
        ax.scatter(
            path[0, 0],
            path[0, 1],
            color="#7AB547",
            s=30,
            zorder=6,
            edgecolor="#111111",
            linewidth=0.5,
        )

        # End point
        ax.scatter(
            path[-1, 0],
            path[-1, 1],
            color=color,
            s=30,
            zorder=6,
            edgecolor="#111111",
            linewidth=0.5,
        )

    ax.set_xlabel(r"$w_1$", fontsize=11)
    ax.set_ylabel(r"$w_2$", fontsize=11)

    if title is None:
        title = ""

    ax.set_title(title, fontsize=12, pad=10)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    ax.legend(frameon=True, fontsize=8, loc="best")

    fig.tight_layout()

    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=250, bbox_inches="tight")
        plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------

def main():
    # -----------------------------
    # Shared setup
    # -----------------------------

    n_steps = 1000
    snapshot_every = 1000

    w0 = np.array([1.0, -2.0], dtype=float)

    L = 2
    d = 1

    C = math.sqrt(L) * (
        2 + 2 * math.sqrt(L) * d + 5 * math.sqrt(L - 1)
    )


    eta0 = 2.5
    eta1 = 0
    alpha_eta = 1/2

    delta = 1 / 2

    eta_schedule = inverse_time_eta(eta0=eta0, alpha=alpha_eta,eta1=eta1)


    print("initial regularized loss:", F_eta(w0, eta0))
    print("constant C:", C)
    print("eta0: ", eta0)
    print("alpha: ", alpha_eta)
    print("eta1: ", eta1)

    # -----------------------------
    # 1 + 3: Discrete GD paths
    # -----------------------------

    ws_reg_gd, ws_unreg_gd, etas, lrs, losses_reg, losses_unreg = (
        run_descent_with_schedulers(
            w0=w0,
            n_steps=n_steps,
            delta=delta,
            C=C,
            eta_schedule=eta_schedule,
            method="sdc",
        )
    )

    # -----------------------------
    # 2 + 4: ODE paths
    # -----------------------------

    # We use t in [0, n_steps] so eta(t) matches eta_k at integer times k.
    t_start = 0.0
    t_end = float(n_steps)

    t_eval_dense = np.linspace(t_start, t_end, 4000)

    eta_func_reg = make_inverse_time_eta_func(
        eta0=eta0,
        eta1=eta1,
        alpha=alpha_eta
    )
    eta_func_const = lambda t: eta0

    eta_func_unreg = lambda t: 0.0

    sol_reg_ode = solve_gradient_flow_ode(
        theta0=w0,
        eta_func=eta_func_reg,
        t_span=(t_start, t_end),
        t_eval=t_eval_dense,
        rtol=1e-9,
        atol=1e-11,
    )

    sol_unreg_ode = solve_gradient_flow_ode(
        theta0=w0,
        eta_func=eta_func_unreg,
        t_span=(t_start, t_end),
        t_eval=t_eval_dense,
        rtol=1e-9,
        atol=1e-11,
    )

    traj_reg_ode_full = sol_reg_ode.y.T
    traj_unreg_ode_full = sol_unreg_ode.y.T

    # -----------------------------
    # Combined snapshots
    # -----------------------------

    out_dir = Path("./plots/example2")
    out_dir.mkdir(parents=True, exist_ok=True)

    level_color = None#"#9FD46F"  
    # cmap = "PuRd_r"
    cmap = "bone"

    path_styles = {
        "regularized GD": {
            "color": "#FF6B4A",  # light blue
            "marker": "o",
            "linewidth": 2.4,
            "linestyle": "-",
        },
        "unregularized GD": {
            "color": "#E0A82E",  # warm yellow
            "marker": "s",
            "linewidth": 1.8,
            "linestyle": "-",
        },
        "regularized ODE": {
            "color": "#F4F1DE",  # white
            "marker": "^",
            "linewidth": 1.4,
            "linestyle": "--",
        },
        "unregularized ODE": {
            "color": "#A78BFA",  # soft green
            "marker": "D",
            "linewidth": 1.4,
            "linestyle": "--",
        },
    }
    marker_every = {
        "regularized GD": 20,
        "unregularized GD": 20,
        "regularized ODE": 100,
        "unregularized ODE":100, 
    }

    snapshot_steps = list(range(0, n_steps + 1, snapshot_every))

    for k in snapshot_steps:
        # ODE time corresponding to GD step k
        t_snap = float(k)

        idx_reg = np.searchsorted(sol_reg_ode.t, t_snap, side="right") - 1
        idx_unreg = np.searchsorted(sol_unreg_ode.t, t_snap, side="right") - 1

        idx_reg = max(idx_reg, 0)
        idx_unreg = max(idx_unreg, 0)

        # Include dense ODE trajectory up to the current snapshot time
        path_reg_ode = traj_reg_ode_full[: idx_reg + 1]
        path_unreg_ode = traj_unreg_ode_full[: idx_unreg + 1]

        # Include GD trajectory up to step k
        path_reg_gd = ws_reg_gd[: k + 1]
        path_unreg_gd = ws_unreg_gd[: k + 1]

        eta_k = eta_schedule(k)

        if k < len(lrs):
            alpha_k = lrs[k]
        else:
            alpha_k = lrs[-1]

        paths = {
            "regularized GD": path_reg_gd,
            "unregularized GD": path_unreg_gd,
            "regularized ODE": path_reg_ode,
            "unregularized ODE": path_unreg_ode,
        }

        title = (
            fr"$\eta_k = {eta_k:.3f}$, "
            fr"$\alpha_k = {alpha_k:.3g}$, "
            fr"step/time $= {k}$"
        )

        filename = out_dir / f"alpha_{alpha_eta:.2f}_eta0_{eta0}_baseline_{eta1}_{k:01d}.png"

        plot_contour_with_paths(
            eta=eta_k,
            paths=paths,
            w_star=w_star,
            level_color=level_color,
            cmap=cmap,
            xlim=(-3.0, 1.2),
            ylim=(-3.0, 1.2),
            title=title,
            filename=filename,
            marker_every=marker_every,
            path_styles=path_styles,
        )

    print(f"Created {len(snapshot_steps)} combined snapshots in {out_dir}/")

    print("\nFinal points:")
    print("regularized GD:   ", ws_reg_gd[-1])
    print("unregularized GD: ", ws_unreg_gd[-1])
    print("regularized ODE:  ", traj_reg_ode_full[-1])
    print("unregularized ODE:", traj_unreg_ode_full[-1])

    print("\nFinal products w1*w2:")
    print("regularized GD:   ", np.prod(ws_reg_gd[-1]))
    print("regularised GD final loss:  ", losses_reg[-1])
    print("unregularized GD: ", np.prod(ws_unreg_gd[-1]))
    print("unregularised GD final loss:  ", losses_unreg[-1])
    print("regularized ODE:  ", np.prod(traj_reg_ode_full[-1]))
    print("unregularized ODE:", np.prod(traj_unreg_ode_full[-1]))
    print("target w_star:    ", w_star)


if __name__ == "__main__":
    main()
