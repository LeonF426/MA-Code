# Poisson PINN Robustness and Reproducibility: Conversation Summary

This document records the questions, scientific conclusions, implementation
changes, and remaining experimental work from the discussion about comparing
SGD- and S-SAM-trained Poisson PINNs. It is a distilled project record, not a
verbatim chat transcript.

## 1. Original research question

The motivating hypothesis was that a Poisson PINN trained with S-SAM may be
more stable than one trained with SGD when the assumed PDE is wrong or
uncertain. The proposed inexpensive test was to keep each trained model fixed
and evaluate its final parameters under modified right-hand sides (RHSs), for
example

\[
-\Delta u = a f, \qquad a \in \{0.5, 0.75, 1.0, 1.25, 1.5\}.
\]

For the repository's Poisson problem,

\[
f(x,y) = 2\pi^2\sin(\pi x)\sin(\pi y),
\]

with homogeneous boundary conditions. Because the PDE is linear, the exact
solution for RHS scale `a` is `a * u*`.

The important constraint is that this is a post-training evaluation: the SGD
and S-SAM models are not retrained for each alternative PDE.

## 2. What the RHS sweep does and does not mean

The implemented RHS sweep is a deterministic misspecification stress test. It
answers:

> How incompatible is this one frozen learned solution with a specified family
> of alternative PDEs?

It is useful for comparing how quickly each final model's errors deteriorate as
the PDE specification moves away from the nominal problem. It is cheap because
the model prediction and its Laplacian can be reused across RHS scales.

It is not, by itself, a probabilistic model of classical model uncertainty:

- the selected RHS scales have no probabilities;
- there is no probability distribution over PDEs, functions, or parameters;
- the frozen model cannot adapt its solution to a changed RHS;
- it explores only one narrow misspecification direction: global multiplicative
  scaling of `f`.

Calling this *PDE misspecification sensitivity* or an *RHS stress test* is more
precise than calling it model uncertainty. A more complete treatment would
need, for example, a probability law over RHS functions, uncertain PDE
coefficients, a parametric PINN such as `u(x, y, a)`, or a joint Bayesian model.

Adding random uncertainty to the RHS after training can make sense only after
defining what is random and why. Examples include a random scalar `a`, a random
field perturbation to `f`, or uncertain PDE coefficients. Sampling arbitrary
noise without a defensible distribution remains a robustness experiment, not
calibrated uncertainty quantification.

## 3. Implemented RHS sensitivity evaluation

For every configured RHS scale, the final SGD and S-SAM models now report:

- the explicit modified RHS coefficient and equation;
- RHS RMSE and RHS-change RMSE relative to the nominal PDE;
- nominal-normalized L2 solution mismatch;
- PDE-residual RMSE;
- changes in both model metrics from their nominal values.

The aggregate results include mean and worst solution mismatch, mean and worst
PDE-residual mismatch, and the worst change in either metric per unit RHS-scale
change. These values are saved under `forcing_sensitivity` in `metrics.json`.

The console explicitly labels the scale-one equation as `ORIGINAL RHS`, then
prints every alternative RHS with its corresponding model response.

The plot `forcing_sensitivity.png` shows:

1. the spatial field of the original RHS;
2. solution mismatch versus RHS scale for both trained models;
3. PDE-residual mismatch versus RHS scale for both trained models;
4. optional Laplace uncertainty bands described below.

The earlier labels "physics response" and "solution response" were replaced
because they could suggest two differently composed training objectives. Both
curves evaluate the same frozen network; they are simply different observables:

- **Solution mismatch** compares the prediction with the exact solution of the
  alternative PDE.
- **PDE residual mismatch** evaluates how strongly that same prediction
  violates the alternative differential equation.

## 4. Why the main sensitivity metric changed

An ordinary relative L2 error divides by the norm of the exact solution for
each alternative PDE:

\[
\frac{\lVert u_\theta-a u^*\rVert_2}{\lVert a u^*\rVert_2}.
\]

That denominator changes with `a`. It can therefore make equally sized
perturbations above and below the nominal scale look asymmetric, and it can
become poorly behaved as `a` approaches zero.

The sensitivity plot now uses the fixed nominal solution norm:

\[
E_{\mathrm{solution}}(a)
=
\frac{\lVert u_\theta-a u^*\rVert_2}{\lVert u^*\rVert_2}.
\]

This *nominal-normalized L2 mismatch* measures every perturbation on one common
scale. The ordinary relative L2 error is still retained in the nominal final
metrics because it is a familiar PINN accuracy measure; it is no longer the
primary metric for the RHS sensitivity curve.

## 5. S-SAM noise versus post-training evaluation

Only the S-SAM-trained model uses Gaussian parameter perturbations during
training. SGD does not use those samples.

This distinction must remain explicit:

- **S-SAM training:** Gaussian parameter samples are part of its optimized
  smoothed objective.
- **SGD training:** no Gaussian parameter samples are drawn.
- **Deterministic RHS sweep:** neither model is retrained or randomly perturbed.
- **Post-training sharpness evaluation:** either final model may be probed with
  parameter perturbations as a common diagnostic; doing so does not mean SGD
  was trained with noise.
- **Laplace evaluation:** a local posterior approximation may be fitted around
  either final parameter vector after training; this is an evaluation method,
  not part of either optimizer's training procedure.

The code uses independent random generators for S-SAM parameter perturbations
and mini-batch selection. Changing the S-SAM perturbation seed therefore cannot
silently change the batch order.

## 6. Laplace approximation around the final parameters

The implemented Laplace analysis approximates parameter uncertainty locally
around each trained parameter vector. In broad terms:

1. treat the final parameters as the center of a local approximation;
2. approximate the loss curvature with a diagonal generalized Gauss-Newton
   (GGN) matrix;
3. combine this curvature with an explicit prior precision to obtain an
   approximate Gaussian covariance in parameter space;
4. use first-order linearization (the delta method) to propagate that covariance
   to the solution-mismatch and PDE-residual metrics.

Random Rademacher probes estimate the GGN diagonal, avoiding the cost of a full
dense Hessian. This produces confidence bands in `forcing_sensitivity.png` and
full diagnostics under each algorithm's `laplace_sensitivity` metrics entry.

The method is scientifically interesting because local curvature distinguishes
directions in parameter space to which the fitted model is more or less
sensitive. It converts that local geometry, together with explicit probabilistic
assumptions, into an approximate posterior and predictive uncertainty. The
approach follows the scalable and linearized-predictive Laplace methods reviewed
in [Laplace Redux](https://arxiv.org/abs/2106.14806). Interpreting squared PINN
residual terms through a Gaussian likelihood is consistent with the Bayesian
PINN construction in [B-PINNs](https://arxiv.org/abs/2003.06097).

Important limitations remain:

- a strict Laplace posterior is centered at a MAP estimate, while an S-SAM
  endpoint minimizes a Gaussian-smoothed objective and need not be a stationary
  point of the clean loss;
- a diagonal GGN discards parameter correlations;
- the delta method is only a local, first-order approximation;
- band widths depend strongly on the chosen prior and likelihood precisions;
- without calibration, the bands are comparative diagnostics rather than
  calibrated Bayesian credible intervals.

The implementation reports the clean gradient norm to make the first limitation
visible. Prior precision, likelihood precision, number of curvature probes,
confidence level, and evaluation seed are explicit configuration values and are
saved with the run.

## 7. Why apparently large configuration changes had little effect

An audit was started after training curves appeared identical across random
seeds and nearly identical for several batch sizes. The concern was justified:
some accepted configuration values did not affect the computation.

The audit found:

- specialized PINN trainers ignored `training.batch_size` and always used all
  collocation points;
- `model.parameter_init.seed` was present in configuration but not consumed;
- `learning_rate.type` was a vestigial no-op;
- `inverse_time.alpha` appeared in examples but was ignored;
- raw histories were not saved, making exact comparisons difficult;
- plot smoothing and a hidden warm-up interval could obscure differences;
- configuration hashes did not identify the source code used for a run;
- output locations depended on the process working directory.

These were implementation defects, not evidence that the optimization was
intrinsically seed- or batch-insensitive.

## 8. Reproducibility fixes applied across the repository

### PINN mini-batching

All specialized Poisson, advection, polynomial-Poisson, and heat PINN trainers
now use the shared `PINNBatcher`.

`training.batch_size` denotes the number of interior collocation points in an
SGD or S-SAM update. Other point groups receive the same fraction of their full
sets. For example, with 512 interior and 256 boundary points, a batch size of
128 produces 128 interior and 64 boundary points per normal update.

Each point group uses a deterministic shuffled, non-dropping permutation
stream. A batch is selected exactly once per optimizer step, and the clean
evaluation plus all S-SAM perturbation samples in that step use the same batch.
This prevents data-sampling noise from being mixed into the parameter-smoothed
objective. `algorithm: gd` intentionally remains full-batch.

Effective group sizes and actual per-step sizes are recorded in the result and
raw history.

### Independent seed responsibilities

The repository now distinguishes:

| Configuration key | Responsibility |
| --- | --- |
| `model.parameter_init.seed` | Initial model parameters |
| `model.parameter_init.rescaling.seed` | Optional post-initialization rescaling |
| `data.seed` | Dataset, observation, or collocation generation |
| `training.batch_seed` | Batch permutations |
| `training.perturbation.seed` | S-SAM parameter perturbations |
| `training.seed` | Other stochastic training operations |
| evaluation-specific seeds | Post-training Monte Carlo diagnostics only |

Model construction and initialization use local random-number-generator state,
so initializing one model does not unintentionally mutate unrelated global RNG
state. Generic data loaders also use their own generator for batch order.

For fair SGD/S-SAM comparisons, both models receive the same reference initial
state and the same batch seed. Only S-SAM receives a parameter-perturbation
seed.

### Strict configuration validation

Unknown keys now fail loudly for training, schedules, optimizers, S-SAM
perturbations, and parameter initialization. Named configuration alternatives
replace incompatible defaults rather than deep-merging stray fields from a
different policy.

The unused learning-rate `type` setting was removed and is now rejected. The
inverse-time schedule now actually uses

\[
\text{floor}+
\frac{\text{initial}}
     {(\text{offset}+\alpha\,\text{step})^{\text{power}}}.
\]

### Artifact and plot transparency

Examples now save:

- `config.json` with the requested configuration;
- `manifest.json` with Git commit, dirty flag, tracked-diff hash, source-tree
  hash, Python/PyTorch versions, command, and working directory;
- `training_history.json` with raw unsmoothed history and effective training
  configuration;
- `metrics.json` with final metrics and effective training metadata;
- plots whose smoothing and start-step choices are explicit in configuration.

Example configurations use `smoothing_window=1` and `loss_start_step=0` so the
full unsmoothed history is visible by default. Output directories are anchored
to the repository root rather than the caller's working directory.

The detailed contract for future examples is in
[`EXPERIMENT_REPRODUCIBILITY.md`](EXPERIMENT_REPRODUCIBILITY.md).

## 9. Verification performed

The complete test suite passed:

```text
86 passed
```

The only warning was that pytest could not update `.pytest_cache` because of
local access permissions; it did not indicate a test failure.

New or expanded tests cover:

- initialization-seed effectiveness and rejection of unknown init keys;
- PINN batch ratios and update differences across batch sizes;
- identical batch semantics for SGD and S-SAM;
- changed permutations when the batch seed changes;
- use of `inverse_time.alpha`;
- rejection of the removed learning-rate `type` option;
- strict JSON histories and provenance manifests;
- validation of configuration dictionaries in every example.

A controlled one-step Poisson check after the fix produced:

| Requested interior batch | Effective interior/boundary | Clean loss |
| ---: | ---: | ---: |
| 512 | 512 / 256 | 90.051033 |
| 256 | 256 / 128 | 83.825951 |
| 128 | 128 / 64 | 97.785286 |

The maximum absolute parameter differences after that update were approximately
`2.94e-4` for 512 versus 256, `4.52e-4` for 512 versus 128, and `6.24e-4` for
256 versus 128. This verifies that the requested batch size now changes the
actual data used and the resulting optimization update.

Thirteen example configuration dictionaries were also loaded and validated.

## 10. Current interpretation and next experimental step

The code paths and regression tests now support meaningful seed and batch-size
experiments, but existing saved results are not repaired retroactively. Any runs
created before these fixes may have silently used full-batch PINN training and
must not be used to infer batch-size effects.

The full Poisson experiments should be rerun from clean artifact directories.
For a scientifically credible SGD/S-SAM comparison:

1. use multiple independent initialization/data replicates, not one seed;
2. match initial parameters, collocation points, and batch order within each
   SGD/S-SAM pair;
3. vary only the optimizer-specific S-SAM perturbations within a matched pair;
4. retain raw histories and manifests for every run;
5. report distributions or confidence intervals over replicates;
6. keep the deterministic RHS stress test separate from the Laplace parameter-
   uncertainty analysis;
7. state that uncalibrated Laplace bands are local comparative diagnostics.

The decisive empirical question is then whether S-SAM shows consistently lower
nominal-normalized solution mismatch and/or PDE-residual mismatch across RHS
scales, with the difference persisting across independent replicates. A single
curve or a single seed cannot establish that conclusion.

## 11. Main implementation locations

- `examples/poisson_pinn.py`: experiment configuration, evaluation, printing,
  metrics, and artifact writing.
- `src/ssam/pinn.py`: Poisson training, RHS sensitivity, and diagonal-GGN
  Laplace analysis.
- `src/ssam/visualization.py`: RHS and sensitivity visualization with optional
  Laplace bands.
- `src/ssam/pinn_batching.py`: deterministic ratio-preserving PINN batching.
- `src/ssam/models.py`: seeded model construction and strict initialization.
- `src/ssam/config.py`: normalized and validated training configuration.
- `src/ssam/schedules.py`: strict, effective learning-rate schedules.
- `src/ssam/update_rules.py`: optimizer validation and independent S-SAM noise.
- `src/ssam/artifacts.py`: raw history and source-provenance artifacts.
- `EXPERIMENT_REPRODUCIBILITY.md`: rules for creating and interpreting future
  examples.

## 12. Working-tree note

At the time of this summary, these changes are present in the working tree but
are not described here as a Git commit. The pre-existing untracked `.codex/` and
`.idea/` directories were not modified as part of this work.
