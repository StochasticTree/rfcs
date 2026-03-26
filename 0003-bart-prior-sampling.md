# Sampling from the BART Prior

PR link: [TBD]

Tracking issue: [StochasticTree/stochtree#200](https://github.com/StochasticTree/stochtree/issues/200)

# Overview

Add first-class support for sampling from the BART prior — the distribution over sum-of-trees functions before any data is observed. This is accomplished in two phases: 

1. exposing **observation weights** ("case weights") in the high-level `bart()` and `bcf()` interfaces, and 
2. providing a convenience wrapper `bart_prior_sample()` (Python) / `bart_sample_prior()` (R) that hides the mechanics from users who only need prior draws.

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

## Phase 2 — Prior sampling convenience function

### Approach

The cleanest way to sample from the BART prior within the existing MCMC framework is to run the sampler on a dataset where all observation weights are zero. With `w_i = 0` for all `i`:

- **Split evaluation**: all weighted sufficient statistics are zero, so the Metropolis-Hastings acceptance ratio for grow/prune/change steps reduces to a ratio of tree structure prior probabilities only. The data have no effect on which tree topologies are sampled.
- **Leaf parameter sampling**: the posterior `p(μ | data, tree) = p(μ | tree)` because the likelihood term vanishes. Leaves are drawn from their prior $N(0, \sigma^2_{\mu})$.
- **Global variance $\sigma^2$**: with zero total weight, $\sigma^2$ is not informed by the data and samples from its IG prior throughout. This is the desired behavior — the marginal prior over $\sigma^2$ is preserved.

The dummy dataset `y_train = 0` (or any constant) with `X_train` drawn from the user's intended covariate distribution is sufficient.

### New API

#### Python: `bart_prior_sample()`

```python
def bart_prior_sample(
    X: np.ndarray,
    num_samples: int = 100,
    num_burnin: int = 0,
    general_params: dict | None = None,
    mean_forest_params: dict | None = None,
) -> BARTModel:
    """
    Sample from the BART prior over functions f: X -> R.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Covariate matrix defining the domain. Samples will reflect the
        marginal tree prior evaluated at these X values.
    num_samples : int
        Number of prior draws to return.
    num_burnin : int
        Burn-in iterations (usually 0 since there is no posterior to mix
        towards, but provided for interface consistency).
    general_params, mean_forest_params : dict, optional
        Same as in BARTModel.sample(). Controls num_trees, alpha/beta,
        leaf scale, etc.

    Returns
    -------
    BARTModel
        A fitted BARTModel whose y_hat_train samples are draws from
        the BART prior p(f(X)).
    """
    n = X.shape[0]
    y_dummy = np.zeros(n)
    weights = np.zeros(n)
    model = BARTModel()
    model.sample(
        X_train=X,
        y_train=y_dummy,
        observation_weights=weights,
        num_gfr=0,
        num_burnin=num_burnin,
        num_mcmc=num_samples,
        general_params=general_params,
        mean_forest_params=mean_forest_params,
    )
    return model
```

This lives in `stochtree/bart.py` as a module-level function (not a method) and is exported from `stochtree/__init__.py`.

#### R: `bart_sample_prior()`

```r
sampleBARTPrior <- function(
  X,
  num_samples = 100,
  num_burnin = 0,
  general_params = list(),
  mean_forest_params = list()
) {
  n <- nrow(X)
  y_dummy <- rep(0, n)
  observation_weights <- rep(0, n)
  bart(
    X_train = X,
    y_train = y_dummy,
    observation_weights = observation_weights,
    num_gfr = 0,
    num_burnin = num_burnin,
    num_mcmc = num_samples,
    general_params = general_params,
    mean_forest_params = mean_forest_params
  )
}
```

This lives in `R/bart.R` and is exported from `NAMESPACE`.

### Handling `num_gfr`

The GFR (grow-from-root) warm-start uses a regression-tree-style initialization that depends on the data. With zero weights, GFR initialization is ill-defined. The convenience wrappers above therefore hard-code `num_gfr = 0`. The low-level `bart()` / `BARTModel.sample()` do not enforce this restriction — advanced users who pass `observation_weights = zeros` manually with `num_gfr > 0` should receive an error.

---

## Phase 3 — Vignette

A new multilingual vignette should demonstrate:

1. Sampling `f ~ BART prior` using `bart_sample_prior()`
2. Plotting several prior draws as functions of a 1D covariate to build intuition about how `num_trees`, `alpha`, `beta`, and leaf scale affect the prior
3. Computing prior predictive distributions: `p(y | X, prior)` by adding $\sigma^2$ draws on top of the `f` draws
4. A simple calibration example: choosing `num_trees` so that the prior 95% interval for `f(x)` spans a user-specified range

# Value

- **Prior calibration** becomes accessible to any stochtree user with a one-line call, rather than the low-level API
- **Observation weights** independently enable weighted regression for survey data, importance sampling, and subgroup upweighting — use cases that appear repeatedly in applied work
- **Pedagogical value**: a "what does BART believe before data?" vignette is a commonly requested resource and helps new users build correct intuition about the model

# Risks and Drawbacks

- **Zero-weight sampler behavior needs validation.** While the math is straightforward, we should verify empirically (via a unit test) that the sampler with `w_i = 0` actually produces draws consistent with the analytical prior — e.g., that the marginal variance of `f(x)` matches `num_trees * leaf_scale²` for a flat leaf prior.
- **$\sigma^2$ sampling with zero weights.** The IG posterior for `σ²` when all weights are zero reduces to the prior. The existing sampler should handle this gracefully (no data contribution to the shape/rate), but this edge case should be tested explicitly.
- **`observation_weights` name vs. `variance_weights` in C++.** The C++ layer calls these `var_weights` (short for "variance weights") because they scale the residual variance as `σ²/w_i`. The high-level name `observation_weights` is more familiar to applied users. This naming gap should be documented in the API docstrings to avoid confusion.
- **BCF complication.** The BCF model has an internal propensity model (`bart()` fit on `Z ~ X`) that is also affected by the observation weights and many more potential terms (adaptive coding, parametric CATE intercept), and designing its prior sampler will require additional complexity, which is deferred to after completion of this work.

# Alternatives

## Alternative 1: Direct prior sampler (no MCMC)

Rather than running MCMC with zero weights, implement a standalone function that samples tree structures directly from the branching process prior and then draws leaf parameters from `N(0, σ²_μ)`. This is more computationally efficient (no chain mixing concerns) and conceptually cleaner.

**Why not chosen (now):** This requires a new code path in the C++ tree sampler that bypasses all data-dependent logic. The effort is substantially larger than Phase 1+2 above, and the zero-weight MCMC approach produces correct samples for the same cost. The direct sampler could be a future enhancement once the zero-weight approach is validated and user demand is confirmed.

## Alternative 2: Expose only case weights, document the trick manually

Add `observation_weights` to the high-level interface (Phase 1) but skip the convenience wrapper (Phase 2), leaving it to users and documentation to combine zero-outcome + zero-weight inputs.

**Why not chosen:** The convenience wrapper is small (< 20 lines in each language) and substantially lowers the barrier for the prior-sampling use case. The vignette can serve as the primary documentation regardless.

## Alternative 3: Sample a single long chain and thin

Use a tiny positive weight (e.g., `w_i = 1e-10`) instead of exact zero to avoid any potential division-by-zero issues in the sampler.

**Why not chosen:** Exact zero is mathematically cleaner and the C++ code path for zero weights should be safe (sufficient statistics are zero, not `NaN`). This should be verified in testing, but epsilon-weight hacks introduce hyperparameter sensitivity and conceptual murkiness. If numerical issues arise in testing, this can serve as a fallback.

# Open Questions

1. **Should `bart_prior_sample()` / `bart_sample_prior()` also return $\sigma^2$ samples?** The current `BARTModel` does return `global_var_samples` when `sample_sigma2_global = True`. For prior sampling this is the correct behavior ($\sigma^2$ is drawn from its IG prior). The convenience wrapper should probably enable `sample_sigma2_global` by default so users can easily construct the full prior predictive distribution `y ~ N(f(x), σ²)`.

2. **Interactions with complex models**: work out the details of whether observations weights can be used with variance forests and probit / cloglog outcome models and add appropriate warnings / error handling and testing.
