# MA-Code

This repository contains the earlier SSAM gradient-descent and ODE plots in
`main.py`, `utils.py`, and the other experiment files. Their existing entry
points are unchanged.

## Projected S-SAM examples

Run from the repository root (with the dependencies in `pyproject.toml`):

```bash
python projected_examples.py
python projected_examples.py --example general_l3 --paint R_min
python projected_examples.py --example legacy --paint R_min R_max projection
python -m unittest -v test_projected_examples
```

`--paint` selects which centered balls are **drawn**; all numerical values
are calculated and written to `plots/projected_examples/<example>.json` in
every run. The trajectory figure is saved alongside the JSON as a PNG. An
empty `--paint` draws no balls. The Python API likewise accepts, for example,
`run_example(examples()["simple"], Path("plots/projected_examples"),
paint=["R_min"])`.

The four examples are:

| Name | Layers and coordinates | Exact distribution |
| --- | --- | --- |
| `simple` | `L=2, d=1` | `X=±1` equiprobably, `Y=X` |
| `legacy` | `L=2, d=1` | `X=±1` equiprobably, `Y=3.14159 X`; same `w_*` and population objective as the original plots |
| `general_l2` | `L=2, d=1` | Four equally likely atoms `(X,Y)=(-2,-1),(-1,0),(1,1),(2,1)`; no exact linear teacher |
| `general_l3` | `L=3, d=2` | Six equally likely atoms specified in `examples()`; correlated input coordinates and no exact linear teacher |

The implementation uses the nested estimator from the predraft: it draws
`N=32` independent data samples and `b=8` independent Gaussian parameter
perturbations **per data sample** by default. Each step applies
`theta[k+1] = project_ball(theta[k] - alpha[k] * g[k], r)`, with
`alpha[k]=0.08/(k+1)^0.65` and `eta[k]=0.7/(k+1)^0.1`. These are adjustable
through command-line flags; the default is 1200 steps and a fixed random seed.
The default projection radius is `r = R_max + 0.5`. The JSON records whether
projection was active on any step and the largest full parameter norm.

Population quantities are obtained from the **distribution**, not from the
training batch: `Sigma_X = E[XX^T]`, `b = E[XY]`,
`lambda_min = lambda_min(Sigma_X)`, `P_crit = ||b||/lambda_min`, and
`P_min = min_{i:b_i != 0}|b_i|/||Sigma_X||_op`. Here `P_min` names the ratio
inside the predraft's definition of `R_min`; it is not an additional assumption.
The plotted radii use
`R_min = sqrt(L) * P_min^(1/L)` and
`R_max = sqrt(L*d) * P_crit^(1/L)` (the square root of the bound on the
**squared** parameter norm). For all three scalar examples these two radii
coincide exactly; selecting both draws two styles on the same circle.

In the `general_l3` figure, the left plot displays the first coordinate of
each of the first two layers and a two-dimensional loss slice with all other
parameters fixed at their initial values. Its circles are the coordinate
projections of the centered balls. The right plot shows the actual norm of
the full six-dimensional iterate against the selected radii; use that plot
to check full-space membership.
