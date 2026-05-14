"""
Continuous-time gradient flow ODE with time-varying eta(t):
  d/dt theta_t = - nabla L_R(eta(t), theta_t)

Visualize the trajectory on contour plots of the loss landscape at selected times.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# ======================================================================
# 1. Loss function and gradient (2D example from the paper)
# ======================================================================

w_star = 3.14159

def F_eta(w, eta):
    """
    L_R(eta, w) = (w_* - w2*w1)^2 + eta^2 (w1^2 + w2^2)
    w: array-like (2,)
    """
    w1, w2 = w
    return (w_star - w2 * w1) ** 2 + eta ** 2 * (w1 ** 2 + w2 ** 2)

def grad_F_eta(w, eta):
    """
    Analytic gradient of F_eta with respect to w = (w1, w2).
    """
    w1, w2 = w
    g = w_star - w2 * w1
    d_w1 = -2.0 * g * w2 + 2.0 * eta ** 2 * w1
    d_w2 = -2.0 * g * w1 + 2.0 * eta ** 2 * w2
    return np.array([d_w1, d_w2], dtype=float)

# ======================================================================
# 2. ODE right-hand side
# ======================================================================

def ode_rhs(t, theta, eta_func):
    """
    RHS of the gradient flow ODE:
      d/dt theta = - grad F_{eta(t)}(theta)
    
    theta: (2,) current state
    eta_func: callable, eta_func(t) -> float
    """
    eta_t = eta_func(t)
    return -grad_F_eta(theta, eta_t)

# ======================================================================
# 3. Solve the ODE with high accuracy
# ======================================================================

def solve_gradient_flow_ode(
    theta0,
    eta_func,
    t_span,
    t_eval=None,
    rtol=1e-8,
    atol=1e-10,
):
    """
    Solve the gradient flow ODE from t_span[0] to t_span[1].
    
    theta0: (2,) initial condition
    eta_func: callable, eta(t)
    t_span: (t_start, t_end)
    t_eval: array of times at which to store solution (optional)
    
    Returns a scipy OdeSolution object with .t and .y attributes.
    """
    sol = solve_ivp(
        fun=lambda t, y: ode_rhs(t, y, eta_func),
        t_span=t_span,
        y0=theta0,
        method="DOP853",    # high-order Runge-Kutta for accuracy
        rtol=rtol,
        atol=atol,
        dense_output=True,  # allows continuous interpolation
        t_eval=t_eval,
    )
    return sol

# ======================================================================
# 4. Plotting: contour + trajectory
# ======================================================================

def plot_contour_with_ode_trajectory(
    eta,
    trajectory,       # (N, 2) array of points on the ODE path
    padding=0.5,
    num_grid=500,
    levels=100,
    cmap="RdPu_r",
    marker_every=50,
    title=None,
    filename=None,
):
    """
    Plot contour of F_eta with ODE trajectory overlaid.
    Axis limits adapt to the trajectory bounding box.
    """
    w1_min, w1_max = trajectory[:, 0].min(), trajectory[:, 0].max()
    w2_min, w2_max = trajectory[:, 1].min(), trajectory[:, 1].max()
    
    w1_range = max(w1_max - w1_min, 0.1)
    w2_range = max(w2_max - w2_min, 0.1)
    
    xlim = (-3,3)#(w1_min - padding, w1_max + padding)
    ylim = (-3,3)#(w2_min - padding, w2_max + padding)
    
    w1_vals = np.linspace(xlim[0], xlim[1], num_grid)
    w2_vals = np.linspace(ylim[0], ylim[1], num_grid)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    Z = F_eta((W1, W2), eta)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(W1, W2, Z, levels=levels, cmap=cmap)
    
    # smooth trajectory line
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="navy",
        linewidth=2.0,
        alpha=0.9,
    )
    # markers at intervals
    indices = np.arange(0, len(trajectory), marker_every)
    ax.plot(
        trajectory[indices, 0],
        trajectory[indices, 1],
        marker="o",
        markersize=4,
        color="cyan",
        linestyle="",
    )
    # start/end
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color="lime", s=60, zorder=5, edgecolor="black", linewidth=0.5)
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color="red", s=60, zorder=5, edgecolor="black", linewidth=0.5)

    ax.set_xlabel(r"$w_1$", fontsize=11)
    ax.set_ylabel(r"$w_2$", fontsize=11)
    
    if title is None:
        title = fr"$\eta(t) = {eta:.3f}$"
    ax.set_title(title, fontsize=12, pad=10)
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    fig.tight_layout()

    if filename is not None:
        # Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=250, bbox_inches="tight")
        plt.close(fig)

# ======================================================================
# 5. Main: solve ODE, take snapshots at specified times
# ======================================================================

def main():
    # initial condition
    theta0 = np.array([1.0, -2.0], dtype=float)
    t0 = 5
    # time-varying eta function (you can change this)
    def eta_func(t):
        alpha = 1/2
        if t <= t0:
            return 3
        else:
        # example: inverse time decay
            return 3.0 / (1.0 + t-t0)**(alpha)
    
    # time span for the ODE
    t_start = 0.0
    t_end = 100.0
    
    # times at which to take snapshots
    snapshot_times = np.linspace(t_start, t_end, 12)  # 9 snapshots
    
    # solve ODE with dense output (continuous interpolation)
    # we request solution at snapshot times + many intermediate points for smooth plotting
    t_eval_dense = np.linspace(t_start, t_end, 2000)
    
    sol = solve_gradient_flow_ode(
        theta0=theta0,
        eta_func=eta_func,
        t_span=(t_start, t_end),
        t_eval=t_eval_dense,
        rtol=1e-9,
        atol=1e-11,
    )
    
    # extract full trajectory
    trajectory_full = sol.y.T  # (N, 2)
    
    # out_dir = Path("ode_contour_snapshots")
    
    # create snapshots at each specified time
    for i, t_snap in enumerate(snapshot_times):
        # find closest index in t_eval_dense
        idx = np.argmin(np.abs(sol.t - t_snap))
        t_actual = sol.t[idx]
        
        # trajectory up to this time
        trajectory_up_to_t = trajectory_full[: idx + 1]
        
        # current eta(t)
        eta_t = eta_func(t_actual)
        
        title = fr"$\eta(t={t_actual:.2f}) = {eta_t:.3f}$"
        filename =  f"./plots/ode/ode_snapshot_{i:03d}_t{t_actual:.2f}.png"
        
        plot_contour_with_ode_trajectory(
            eta=eta_t,
            trajectory=trajectory_up_to_t,
            padding=0.5,
            title=title,
            filename=filename,
            marker_every=50,
        )
    
    print(f"Created {len(snapshot_times)} snapshots in ./plots/ode/")

if __name__ == "__main__":
    main()
