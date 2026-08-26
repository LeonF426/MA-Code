
import numpy as np
import math
import utils
import os, sys
from pathlib import Path
from additional_optimization_methods import (
    run_discrete_method,
    plot_explicit_euler_stepsize_bound,
    run_tamed_gd_with_bound
)
# Add the project's src/ directory to sys.path
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


####################################
###### Run Config##################
####################################

w_star = 3.14159

n_steps = 100
snapshot_every = 10

w0 = np.array([0.0, -1.0], dtype=float)

L = 2
d = 1

C = math.sqrt(L) * (
    2 + 2 * math.sqrt(L) * d + 5 * math.sqrt(L - 1)
)


# Change polynomial eta schedule here:

eta0 = 2.5 
eta1 = 0
alpha_eta = 1/2

delta = 1 / 2

# Select either constant or polynomial
# eta_schedule = utils.constant_eta(eta0)
eta_schedule = utils.inverse_time_eta(eta0=eta0,alpha=alpha_eta,eta1=eta1)


## Same for the continuous case:
# eta_func_reg = lambda t: eta0
eta_func_reg = utils.make_inverse_time_eta_func(eta0=eta0,alpha=alpha_eta,eta1=eta1)


print("initial regularized loss:", utils.F_eta(w0, eta0))
print("constant C:", C)
print("eta0: ", eta0)
print("alpha: ", alpha_eta)
print("eta1: ", eta1)

# -----------------------------
# 1 + 3: Discrete GD paths
# -----------------------------

ws_reg_gd, ws_unreg_gd, etas, lrs, losses_reg, losses_unreg = (
    utils.run_descent_with_schedulers(
        w0=w0,
        n_steps=n_steps,
        delta=delta,
        C=C,
        eta_schedule=eta_schedule,
        method="explicit_euler_bound", # change this to "balancing" if necessary
    )
)

lrs_implicit = 0.05

ws_implicit, _, _ = run_discrete_method(
    theta0=w0,
    n_steps=n_steps,
    alpha_schedule=lrs_implicit,
    eta_schedule=eta_schedule,
    grad_loss=utils.grad_F_eta,
    method="implicit_euler",
)

ws_tamed, etas_tamed, lrs_tamed = run_tamed_gd_with_bound(
    theta0=w0,
    n_steps=n_steps,
    eta_schedule=eta_schedule,
    grad_loss=utils.grad_F_eta,
    w_star=w_star,
    delta=delta,
    safety=0.95,
    fallback_alpha=0.1,
)

# -----------------------------
# 2 + 4: ODE paths
# -----------------------------

# We use t in [0, n_steps] so eta(t) matches eta_k at integer times k.
t_start = 0.0
t_end = float(n_steps)

t_eval_dense = np.linspace(t_start, t_end, 4000)


eta_func_unreg = lambda t: 0.0

sol_reg_ode = utils.solve_gradient_flow_ode(
    theta0=w0,
    eta_func=eta_func_reg,
    t_span=(t_start, t_end),
    t_eval=t_eval_dense,
    rtol=1e-9,
    atol=1e-11,
)

sol_unreg_ode = utils.solve_gradient_flow_ode(
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

out_dir = Path("./plots/paper") #change output path if necessary
out_dir.mkdir(parents=True, exist_ok=True)

# color settings:
level_color = "#9FD46F"  
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
    "implicit Euler": {
        "color": "#55C1FF",
        "marker": "P",
        "linewidth": 2.0,
        "linestyle": "-",
    },
    "tamed GD": {
        "color": "#E879F9",
        "marker": "X",
        "linewidth": 2.0,
        "linestyle": "-",
    },
    }
marker_every = {
    "regularized GD": 20,
    "unregularized GD": 20,
    "regularized ODE": 100,
    "unregularized ODE":100,
    "implicit Euler": 20,
    "tamed GD": 20,
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
    path_implicit = ws_implicit[: k + 1]
    path_tamed = ws_tamed[: k + 1]

    eta_k = eta_schedule(k)

    if k < len(lrs):
        alpha_k = lrs[k]
    else:
        alpha_k = lrs[-1]

    # This determines which paths will be plotted
    paths = {
        "explicit Euler": path_reg_gd,
        "implicit Euler": path_implicit,
        "tamed GD": path_tamed,
        # "regularized ODE": path_reg_ode,
        # "unregularized GD": path_unreg_gd,
        # "unregularized ODE": path_unreg_ode,
    }

    title = (
        fr"$\eta = {eta_k:.3f}$, "
        fr"step/time $= {k}$"
    )

    filename = None #out_dir / f"comp5_eta0_{eta0}_{k:01d}.png"

    utils.plot_contour_with_paths(
        eta=eta_k,
        paths=paths,
        w_star=w_star,
        level_color=level_color,
        cmap=cmap,
        xlim=(-3.0, 3.0),
        ylim=(-3.0, 3.0),
        title=title,
        filename=filename,
        marker_every=marker_every,
        path_styles=path_styles,
    )

# print(f"Created {len(snapshot_steps)} combined snapshots in {out_dir}/")
#
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

