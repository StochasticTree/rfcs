# Sampling from the BART Prior

PR link: [#4](https://github.com/StochasticTree/rfcs/pull/4)

Tracking issue: [StochasticTree/stochtree#200](https://github.com/StochasticTree/stochtree/issues/200)

# Overview

Add first-class support for sampling from the BART prior — the distribution over sum-of-trees functions before any data is observed. This is accomplished in two phases:

1. Exposing **observation weights** ("case weights") in the high-level `bart()` and `bcf()` interfaces. Weights are broadly useful independent of prior sampling (survey-weighted regression, importance resampling, subgroup upweighting).
2. A dedicated **direct C++ prior sampler** (`SampleForestFromPrior`) that samples tree structures from the branching process prior and leaf parameters from `N(0, σ²_μ)` — no outcome data, no MCMC, independent draws. Thin wrappers `bart_prior_sample()` (Python) / `sampleBARTPrior()` (R) expose this to users.

# Motivation

Sampling from the BART prior is a standard Bayesian workflow step with at least three concrete use cases:

1. **Prior calibration.** Users want to understand what functions the BART prior places non-negligible probability on before fitting, so they can choose hyperparameters (`num_trees`, $\alpha$, $\beta$, leaf scale $\sigma^2_{\mu}$) that encode their substantive prior beliefs rather than relying on defaults.

2. **Prior predictive checks.** Generating outcome data conditioned on covariates and the model prior lets users verify that the prior is not inadvertently ruling out plausible outcomes (e.g., too-narrow leaf scale causing all predicted values to cluster near zero).

3. **Standalone prior documentation / vignette.** A worked example showing what BART "believes" out of the box is a valuable pedagogical resource.

The current stochtree high-level interface (`bart()` / `bcf()`) has no mechanism for users to tell the sampler to ignore the data. The only path today is through the low-level `ForestDataset` / `TreeSampler` API, which is not a particularly user-friendly interface.

Additionally, observation weights have independent value beyond prior sampling: survey-weighted regression, importance-weighted resampling, and upweighting rare subpopulations all require them. Exposing weights in the high-level interface is a broadly useful enhancement that prior sampling happens to need.

# Proposed Implementation

## Phase 1 — Expose observation weights in the high-level interface

### Background: weight support in the C++ core

Observation weights are already fully supported in the stochtree C++ core. In `include/stochtree/leaf_model.h` the weighted BART likelihood is documented explicitly:

```
y_i | - ~ N(μ(X_i), σ² / w_i)
```

Weighted sufficient statistics (`s_{w,ℓ}`, `s_{wy,ℓ}`, `s_{wyy,ℓ}`) are used in both split evaluation and leaf parameter sampling. The `ForestDataset` class (`include/stochtree/data.h`) stores weights as a `ColumnVector var_weights_` and exposes `SetVarWeightValue()`. The R-level `ForestDataset` wrapper (`R/data.R`) already has `AddVarianceWeights()` and `update_variance_weights()` methods. No new C++ work is required.

### Python: `BARTModel.sample()`

Add an `observation_weights` keyword argument to `BARTModel.sample()` in `stochtree/bart.py`:

```python
def sample(self, X_train, y_train, ..., observation_weights=None, ...):
```

- Type: `np.ndarray` of shape `(n,)`, or `None` (default = all ones).
- Validated to be non-negative with `np.all(observation_weights >= 0)`.
- Passed to the `ForestDataset` constructor / `add_weights()` call that builds `forest_dataset_train` (currently constructed around line 210 of `bart.py`).

The same change applies to `BCFModel.sample()` in `stochtree/bcf.py`. Both the prognostic forest dataset and the treatment effect forest dataset should receive the same weights (observation weights are properties of the outcome, not the forest).

### R: `bart()` and `bcf()`

Add an `observation_weights` argument (default `NULL`) to the `bart()` function in `R/bart.R` and to `bcf()` in `R/bcf.R`:

```r
bart <- function(X_train, y_train, ..., observation_weights = NULL, ...) {
```

- When non-`NULL`: validate as a numeric vector of length `nrow(X_train)` with all values `>= 0`.
- Pass to `forest_dataset$AddVarianceWeights()` after `ForestDataset` construction (currently done around line 120 of `bart.R`).

The `bcf()` function should apply weights to both `mu_forest_data` and `tau_forest_data`.

### Serialization

Observation weights are not part of the model — they are properties of the training data and do not need to be serialized to JSON. No changes to the serialization layer are required.

### Interactions

Most other stochtree modeling features should be compatible with general-purpose observation weights, though we need to be careful to accumulate weighted sufficient statistics for other model terms.

Variance forests and discrete outcome models should at minimum raise a warning and potentially preclude the use of observation weights. See [[#Open Questions]] below.

---

## Phase 2 — Direct C++ prior sampler

### Approach

Phase 2 implements a standalone prior sampler that bypasses the MCMC machinery entirely. Rather than running the sampler on a degenerate dataset, a dedicated C++ function samples tree structures directly from the branching process prior and draws leaf parameters from their prior distribution.

**Why not zero-weight MCMC (originally planned):**

The original Phase 2 design called for running `BARTModel.sample()` / `bart()` with `y_dummy = zeros` and `observation_weights = zeros`. This was abandoned after implementation revealed two blocking problems:

1. **C++ NaN with exact zero weights.** Setting `w_i = 0` produces NaN predictions from the C++ sampler. This is because they are treated as unit-specific variance adjustments that accumulate as `1 / w_i` with no special code path for zero weights.

2. **Large-σ² workaround has MCMC autocorrelation.** An alternative — setting `sigma2_init = 1e6` and `sample_sigma2_global = False` so the leaf posterior collapses to its prior — works mathematically but produces correlated samples (the MCMC chain still mixes between tree topologies, so draws are not independent). It also means eschewing sampling from the $\sigma^2$ prior.

**Chosen design — `MCMCPriorSampleOneIter`:**

Add a new C++ function `MCMCPriorSampleOneIter` (in `src/tree_sampler.cpp` / `include/stochtree/tree_sampler.h`) that:

1. Accepts most of the same arguments as `MCMCSampleOneIter` (i.e. `TreeEnsemble`, `ForestTracker`, `ForestContainer`, etc).
2. For each sample, iterates over all trees and grows each tree by repeatedly sampling grow/prune moves from the branching process prior alone — no data, no sufficient statistics, no likelihood in the MH acceptance ratio.
3. After each tree has been sampled, draws each leaf parameter independently from `N(0, σ²_μ)`.

This produces **independent** samples (no chain, no autocorrelation), requires no outcome data, and is free of any likelihood computation. The covariate matrix `X` is still needed to evaluate the forest at prediction time but plays no role during sampling.

New pybind11 bindings (`py_stochtree.cpp`) and cpp11 wrappers (`R/sampler.R`) will expose `MCMCPriorSampleOneIter` to the high-level interfaces.

### New API

#### Python: `sample_bart_prior()`

```python
def sample_bart_prior(
    X: np.ndarray,
    num_samples: int = 100,
    general_params: dict | None = None,
    mean_forest_params: dict | None = None,
) -> BARTModel:
    """
    Sample from the BART prior over functions f: X -> R.

    Produces independent prior draws by sampling tree structures from
    the branching process prior and leaf parameters from N(0, sigma2_leaf).
    No outcome data is required or used.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Covariate matrix defining the domain.
    num_samples : int
        Number of independent prior draws.
    general_params, mean_forest_params : dict, optional
        Controls num_trees, alpha/beta, leaf scale, etc.

    Returns
    -------
    BARTModel
        Object whose y_hat_train contains num_samples prior draws of f(X).
    """
    # initialize data structures

    # call pybind11 wrapper around MCMCPriorSampleOneIter

    # unpack and return predictions and prior parameter samples
```

Module-level function in `stochtree/bart.py`, exported from `stochtree/__init__.py`.

#### R: `sampleBARTPrior()`

```r
sampleBARTPrior <- function(
  X,
  num_samples = 100,
  general_params = list(),
  mean_forest_params = list()
) {
  # initialize data structures

  # call pybind11 wrapper around MCMCPriorSampleOneIter

  # unpack and return predictions and prior parameter samples
}
```

Lives in `R/bart.R`, exported from `NAMESPACE`.

### Handling `num_gfr`

Not applicable — the direct prior sampler has no GFR warm-start. The convenience wrappers do not expose a `num_gfr` argument.

---

## Phase 3 — Vignette

A new multilingual vignette should demonstrate:

1. Sampling `f ~ BART prior` using `sampleBARTPrior()` (R) / `sample_bart_prior()` (Python)
2. Plotting several prior draws as functions of a 1D covariate to build intuition about how `num_trees`, `alpha`, `beta`, and leaf scale affect the prior
3. Computing prior predictive distributions: `p(y | X, prior)` by adding $\sigma^2$ draws on top of the `f` draws
4. A simple calibration example: choosing `num_trees` so that the prior 95% interval for `f(x)` spans a user-specified range

# Value

- **Prior calibration** becomes accessible to any stochtree user with a one-line call, rather than the low-level API
- **Observation weights** independently enable weighted regression for survey data, importance sampling, and subgroup upweighting — use cases that appear repeatedly in applied work
- **Pedagogical value**: a "what does BART believe before data?" vignette is a commonly requested resource and helps new users build correct intuition about the model

# Risks and Drawbacks

- **New C++ code path.** `MCMCPriorSampleOneIter` is a new function that bypasses the MCMC machinery entirely. It needs careful unit testing to confirm that the marginal variance of `f(x)` matches `num_trees × leaf_scale²` for a flat leaf prior and that tree depth distributions match the branching process prior analytically.
- **`observation_weights` name vs. `variance_weights` in C++.** The C++ layer calls these `var_weights` (short for "variance weights") because they scale the residual variance as `σ²/w_i`. The high-level name `observation_weights` is more familiar to applied users. This naming gap should be documented in the API docstrings to avoid confusion.
- **BCF prior sampling deferred.** The BCF model has additional complexity (adaptive coding, parametric CATE intercept, propensity model) that makes designing a BCF prior sampler non-trivial. This is deferred until the BART prior sampler is complete and validated.

# Alternatives

## Alternative 1: Zero-weight MCMC (originally planned for Phase 2)

Run the MCMC sampler on a dummy dataset with `y_train = zeros` and `observation_weights = zeros`. With all weights zero, the weighted sufficient statistics vanish and the sampler should draw from the prior.

**Why not chosen:** Implemented and then abandoned. Exact zero weights produce NaN predictions from the C++ sampler (likely a `σ²/w_i` evaluation in the residual computation). A large-σ² workaround (`sigma2_init = 1e6, sample_sigma2_global = False`) makes the leaf posterior collapse to its prior and gives numerically correct results, but the chain still mixes between tree topologies so draws are correlated, not independent. The direct C++ prior sampler (chosen design) solves both problems cleanly.

## Alternative 2: Expose only case weights, document the trick manually

Add `observation_weights` to the high-level interface (Phase 1) but skip the convenience wrapper (Phase 2), leaving it to users and documentation to combine the approach manually.

**Why not chosen:** The convenience wrapper is small and substantially lowers the barrier for the prior-sampling use case. Moot now that the zero-weight approach is off the table — the direct sampler has to be implemented regardless.

## Alternative 3: Epsilon-weight hack

Use a tiny positive weight (e.g., `w_i = 1e-10`) instead of exact zero to avoid division-by-zero.

**Why not chosen:** Implemented and tested experimentally. With `w = 1e-10`, predictions are near-zero (the data contribution is effectively silenced) but the sampler appears stuck and the variance behavior is unpredictable. With `w = 0.1`, the chain mixes but is clearly data-influenced. No epsilon value makes the sampler cleanly sample from the prior without either NaN, stuck chains, or data contamination.

# Open Questions

1. **Interactions with observation weights and complex models**: resolved for Phase 1 — variance forests produce a warning (untested), cloglog link produces an error (not compatible), probit is allowed (latent Gaussian, weights well-defined at the latent level).
