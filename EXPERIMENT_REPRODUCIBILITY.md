# Experiment and Example Reproducibility

This document records the repository's experimental contract. Follow it when
adding an example or interpreting an existing run.

## Why this contract exists

An audit of the Poisson PINN found several settings that were accepted without
affecting the computation:

- specialized PINN trainers used every collocation point and ignored
  `training.batch_size`;
- `model.parameter_init.seed` was passed through configuration but not used;
- `learning_rate.type` was stored but had no effect;
- `inverse_time.alpha` was present in examples but ignored;
- plots hid the first 50 updates and emphasized a rolling median, while the raw
  history was not saved;
- run hashes identified only the configuration, not the source code.

Those defects are fixed. They illustrate the central rule: a serialized setting
must either affect the run, be explicitly documented as metadata, or be rejected.

## Randomness must have named responsibilities

Use independent seeds so that one experimental factor can change without
silently changing another:

| Setting | Responsibility |
| --- | --- |
| `model.parameter_init.seed` | Initial parameter values |
| `model.parameter_init.rescaling.seed` | Optional post-initialization rescaling |
| `data.seed` | Dataset split, observations, or fixed collocation set |
| `training.batch_seed` | Mini-batch order |
| `training.perturbation.seed` | S-SAM parameter perturbations |
| `training.seed` | Other stochastic training operations, such as stochastic layers |
| evaluation-specific seeds | Monte Carlo evaluation only; never training |

For an SGD/S-SAM comparison, construct one reference state and load it into both
models. Give both runs the same `batch_seed`. This isolates the update rule while
keeping initialization, data, and batch order matched. S-SAM may then use its own
perturbation seed.

Changing a seed is not expected to change a deterministic component. For
example, `training.perturbation.seed` should not change SGD, because SGD draws no
parameter perturbations.

## PINN batching semantics

For PINNs, `training.batch_size` is the number of interior collocation points in
an ordinary SGD or S-SAM update. Other point groups use the same fraction of
their full sets. With 512 interior and 256 boundary points, a batch size of 128
therefore means 128 interior and 64 boundary points.

Each point group is shuffled by a deterministic, non-dropping permutation
stream. The final batch of a pass may be smaller. `algorithm: gd` deliberately
uses every point, regardless of `batch_size`.

A PINN batch is selected exactly once per optimization step. The clean loss and
all S-SAM perturbation samples in that step reuse that batch. Selecting a new
batch per perturbation would mix parameter uncertainty with data-sampling noise
and would not estimate the intended Gaussian-smoothed objective.

The effective group sizes and actual per-step sizes are stored in the training
result and raw history. A new PINN trainer must use the shared `PINNBatcher`
rather than capturing complete point tensors in a permanent closure.

## Configuration rules

- Do not add decorative settings. Every non-label field needs a consumer and a
  regression test demonstrating its effect.
- Alternative named policies replace their defaults; they are not deep-merged
  with keys belonging to a different policy.
- Unknown training, schedule, optimizer, perturbation, and parameter-
  initialization keys must fail loudly. This catches misspellings such as
  `sead` instead of `seed`.
- The optimizer algorithm and data regime are separate concepts. An SGD
  optimizer operating on the complete dataset is full-batch gradient descent.
- `tamed` is a learning-rate transformation. It has no `type` setting; the
  training algorithm and batch size determine whether its gradient is full-batch
  or stochastic.
- `inverse_time` uses
  `floor + initial / (offset + alpha * step) ** power`. Every listed parameter is
  active.

## What every run must save

Examples must use `create_run_artifacts` and save:

1. `config.json`: the requested experiment configuration;
2. `manifest.json`: Git commit, dirty-state information, source-tree hash,
   Python/PyTorch versions, command, and working directory;
3. `training_history.json`: raw unsmoothed values and the effective normalized
   training configuration;
4. `metrics.json`: final metrics plus effective training metadata;
5. plots with their smoothing and warm-up choices present in `config.json`.

Example output roots are resolved from the repository root, not the caller's
working directory. Running an example from an IDE and from the command line must
therefore write to the same output tree.

The configuration hash alone is not a complete experiment identity. Compare the
source hash in `manifest.json` as well.

## Minimum tests for a new example

- Identical seeds reproduce parameters and raw histories.
- Changing the initialization seed changes a randomly initialized model but not
  the data.
- Changing the data seed changes the sampled data but not model initialization.
- Changing `batch_size` changes the number of examples or collocation points used
  and, for a non-degenerate problem, changes the update.
- Matched algorithms receive the same initial state and mini-batch sequence.
- Every S-SAM perturbation within one step sees the same data batch.
- An unknown or misspelled configuration key raises an error.
- Raw histories and provenance can be parsed from the completed artifact folder.

Do not infer robustness from one seed. Report results over multiple independent
replicates, keeping the seed roles above explicit. Likewise, a frozen-model RHS
stress test measures sensitivity to specified PDE misspecification; it is not a
probabilistic model of model uncertainty unless a probability law over PDEs is
defined.
