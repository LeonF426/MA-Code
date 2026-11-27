# MA-Code

This repository contains the earlier SSAM gradient-descent and ODE plots in
`main.py`, `utils.py`, and the other experiment files. Their existing entry
points are unchanged.

## Projected S-SAM examples

Edit the `CONFIG` dictionary near the top of `projected_examples.py` to choose
examples, radii to paint, step count, batch sizes, schedules, radius, seed,
and output directory. Run `projected_examples.py` directly in PyCharm, or
run from the repository root (with the dependencies in `pyproject.toml`):

```bash
python projected_examples.py
python projected_examples.py --example general_l3 --paint R_min
python projected_examples.py --example legacy --paint R_min R_max projection
python -m unittest -v test_projected_examples
```

Command-line options override `CONFIG`. `paint=["R_min"]` draws only that
circle. `paint=[]` draws none. All numerical values
are calculated and written to `plots/projected_examples/<example>.json` in
every run. The trajectory figure is saved alongside the JSON as a PNG. An
empty `--paint` draws no balls. The Python API likewise accepts, for example,
`run_example(examples()["simple"], Path("plots/projected_examples"),
paint=["R_min"])`.

The four examples currently available are:

| Name | Layers and coordinates | Exact distribution |
| --- | --- | --- |
| `simple` | `L=2, d=1` | `X=±1` equiprobably, `Y=X` |
| `legacy` | `L=2, d=1` | `X=±1` equiprobably, `Y=3.14159 X`; same `w_*` and population objective as the original plots |
| `general_l2` | `L=2, d=1` | Four equally likely atoms `(X,Y)=(-2,-1),(-1,0),(1,1),(2,1)`; no exact linear teacher |
| `general_l3` | `L=3, d=2` | Six equally likely atoms specified in `examples()`; correlated input coordinates and no exact linear teacher |

Each run creates one `<example>.png` and one `<example>.json` in the configured
output directory. For the three scalar examples, the PNG contains a contour
plot of the exact population loss at the final perturbation scale, the complete
projected stochastic iterate path (with marked start, end, and intermediate
points), and the selected circles centered at zero. The `projection` circle
has radius `r`. The green contour indicates the least-squares product.

For `general_l3`, the left PNG panel instead shows a population-loss *slice*
through the initial parameters in the `(w_1,1, w_2,1)` plane, the projected
iterate path in those two coordinates, an inset zoom of that path, and the
selected circles (projections of the full balls). The right panel shows the
full six-dimensional iterate norm against the selected radii. The JSON
records the exact distribution atoms and probabilities, `Sigma_X`, `E[XY]`,
the eigenvalue and radius bounds, configuration, and the final result.

The implementation uses the nested estimator from the predraft: it draws
`N=32` independent data samples and `b=8` independent Gaussian parameter
perturbations **per data sample** by default. Each step applies
`theta[k+1] = project_ball(theta[k] - alpha[k] * g[k], r)`, with
`alpha[k]=0.08/(k+1)^0.65` and `eta[k]=0.7/(k+1)^0.1`. The initial values can
also be overridden through command-line flags; the decay exponents are in
`CONFIG`. The default is 1200 steps and a fixed random seed.
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
