# Refactoring Sampler Logic to C++

PR link: [#4](https://github.com/StochasticTree/rfcs/pull/4)

Tracking issue: [#340](https://github.com/StochasticTree/stochtree/issues/340)

# Overview

`stochtree` currently exposes the low-level C++ sampling constructs as R and Python classes, which are composed in language-specific sampling loops. This duplicates a lot of logic across two interfaces, opening a large bug surface, and also leaves some performance "on the table". 

This RFC proposes a **C++ API** for running BART and BCF samplers. Input and output objects are owned by callers (i.e. they reside in R and Python), while the sampler and its ephemeral state reside in C++.

**Scope:** The end state of this RFC is that both BART (`BARTModel.sample()` in Python / `bart()` in R) and BCF (`BCFModel.sample()` in Python / `bcf()` in R) run the same exact "core" C++ code, with each package handling language specific marshalling of inputs and unpacking / storage of outputs.

This RFC covers the design of the C++ dispatch layer, leaving the specifics of performance improvements unlocked by this redesign (i.e. quantization of features for MCMC, histograms for split evaluations in GFR, single-precision arithmetic) to a later design document.

# Motivation

## The duplication problem

stochtree's MCMC sampling loop is implemented twice — once in Python (`stochtree/bart.py`, `stochtree/bcf.py`) and once in R (`R/bart.R`, `R/bcf.R`):

| File | Lines |
|---|---|
| `stochtree/bart.py` | ~3,800 |
| `stochtree/bcf.py` | ~5,000 |
| `R/bart.R` | ~4,700 |
| `R/bcf.R` | ~6,100 |

The sampling loop itself — GFR warm-start, MCMC burn-in, MCMC sample collection, residual tracking, variance sampling, chain initialization — accounts for roughly 600 lines in `bart.py` and proportionally more in `bcf.py`. Equivalent code exists in both R files. Every new feature must be added in both languages. Bugs found in one are frequently present in the other.

Recent examples of headaches caused by duplication:

- **BCF warm-start propensity reuse (#158):** Had to understand and fix identical logic independently in `bcf.py` and `bcf.R`.
- **Multi-chain MCMC bugs (#328):** Chain initialization from GFR ensembles diverged subtly between R and Python; required multiple rounds of fixes.

## What this RFC does not fix

The R and Python interfaces have other sources of complexity beyond the sampling loop: covariate preprocessing (`CovariatePreprocessor` in Python, its R equivalent), input validation, factor handling, and output marshaling. These stay in their respective languages — they are inherently language-specific and are not the source of the duplication bugs above.

## Model Support Inventory

The C++ dispatch layer must support every model variant currently exposed through the high-level interface. The following sections enumerate them. All rows are included in this RFC, as full C++ dispatch must support the entirety of the stochtree BART and BCF interfaces.

### BART Model Inventory

| Variant | Key mechanism |
|---|---|
| Continuous outcome, identity link | Standard BART; no augmentation |
| Observation weights | Setting weights in a `ForestDataset` |
| Binary outcome, probit link | Albert & Chib (1993) truncated normal augmentation per GFR/MCMC iter |
| Univariate leaf regression | Leaf basis passed to `ForestDataset`; `leaf_model=1` |
| Multivariate leaf regression | `leaf_model=2`; leaf scale sampling disabled |
| Variance forest | Second forest modeling log-variance; residual scaling after each draw |
| Random effects (intercept-only) | `RFXModel`, `RFXDataset`, `RFXTracker`, `LabelMapper`; basis = all-ones column |
| Random effects (custom basis) | Same machinery; user-supplied basis matrix |
| Binary outcome, cloglog link | `OrdinalSampler` Gibbs steps; `update_latent_variables` + `update_gamma_params` + `update_cumulative_exp_sums` per iter |
| Ordinal outcome, cloglog link | Superset of binary cloglog; multi-level cutpoints; `num_categories > 2` |

Constraints on interactions:

- Sampling global error scale not compatible with variance forest or binary / ordinal models
- Observation weights incompatible with cloglog link and raise a warning for variance forests
- Sampling leaf scale not compatible with multivariate leaf regressions

### BCF Model Inventory

| Variant | Key mechanism |
|---|---|
| Standard BCF | Mu/tau forests, residual separation per iter |
| BCF with probit | Mu/tau forests, separated by partial residual defined by latent outcome sampled by probit model |
| BCF with internal propensity | Fits internal BART model when no propensity scores provided |
| BCF with adaptive coding | Gibbs update of coding parameters `b0` and `b1` attached to `Z=0` and `Z=1` via linear regression |
| BCF with `sample_intercept` | Constant parametric term added to treatment effect model (so that forest is an "offset" to this constant) |
| BCF with variance forest | Third forest for log-variance; same mechanism as heteroskedastic BART |
| BCF with random effects | Same RFX machinery as BART RFX, with added `intercept_plus_treatment` model specification |


# Proposed Implementation

This design keeps many of the "core" stochtree C++ data structures and routines, but adds higher-level structure. For BART, the structure will look like

1. A "stateful" `BARTSampler` object which maintains all of the ephemeral state needed to sample a BART model, but owns none of the inputs or outputs, which are supplied by the clients (R / Python packages or other callers of the C++ API)
2. A `BARTData` wrapper object which carries pointers to data created on the client side (i.e. R matrices, numpy array, or C++ vectors)
3. A `BARTConfig` object that sets hyperparameter values and determines the nature of the BART model to be run (i.e. random effects, cloglog, variance forest, etc...)
4. A `BARTResult` object that contains (optional) outputs for each of the model's terms

We envision the BCF structure will look exactly the same, so we focus on the details of this design and how it supports all of stochtree's current functionality with an easier maintenance surface.

## `BARTSampler` Class

The `BARTSampler` object is the workhorse of this new approach. It owns all the "ephemeral" state of a BART / BCF sampler, specifically:

1. `TreeEnsemble` objects that store the "active forest" for both a mean forest and [optional] variance forest
2. `ForestTracker` objects that cache predictions, leaf node indices, and maintain a partitioned record of all elements in a given leaf node
3. `RNG` objects that wrap a `std::mt19937` random number generator
4. [Optional] random effects data structures (i.e. `RandomEffectsModel` storing current parameter state and routines for generating new parameters and `RandomEffectsTracker` mapping observations to groups and vice verse)
5. [Optional] `OrdinalSampler` object controlling the cloglog model's data augmentation and additional parameter set
6. Snapshots of "warm-start" state for forests that are used to initialize MCMC chains but not themselves retained
7. `ForestDataset` wrapper around covariates, bases, and observation weights (used for heteroskedastic models or other weighting use cases)
8. Other model state, such as the global error scale parameter and leaf scale parameter(s)

## `BARTData` Struct

`BARTData` is a simple struct that carries pointers to, and metadata about, all of the input data needed for a given model, in a pattern like

```cpp
const double* X_train = nullptr;
int n_train = 0;
int p = 0;
const double* y_train = nullptr;
const double* X_test = nullptr;
int n_test = 0;
const int* feature_types = nullptr;
const double* basis_train = nullptr;
const double* basis_test  = nullptr;
int basis_dim = 0;
```

For models that don't use a leaf basis (i.e. Gaussian constant leaf BART), the associated data pointer will not be set by the caller and its use by `BARTSampler` is guarded by a null pointer check.

These raw pointers and dimension metadata are used on the C++ side to initialize `ForestDataset` and `Outcome` objects, though eventually this internal copying of data could be refactored out.

## `BARTConfig` Struct

The `BARTConfig` struct is a composition of numerous config structs for different model components.

```cpp
struct TreePriorConfig {
    int    num_trees          = 200;
    double alpha              = 0.95;
    double beta               = 2.0;
    int    min_samples_leaf   = 5;
    int    max_depth          = 10;
    int    cutpoint_grid_size = 100;
    std::vector<double> variable_weights;
};

// ...

struct BARTSamplerConfig {
    // High level
    bool standardize = true;
    int num_threads = -1;
    
    // Mean forest
    std::optional<TreePriorConfig> mean_forest_config;
    std::optional<LeafConfig> leaf_config;
    
    // Global model options
    std::optional<GlobalVarianceConfig> global_variance_config;

    // Variance forest
    std::optional<TreePriorConfig> variance_forest;
    
    // Random effects
    std::optional<RFXConfig> rfx;

    // Link function and relevant link-specific config
    LinkFunction link = LinkFunction::Identity;
    std::optional<CloglogConfig> cloglog;
};
```

## `BARTSamples` Struct

The `BARTSamples` struct wraps all of the sampled outputs of a BART MCMC sampler

Outputs are either stored as a `std::unique_ptr` for forests and random effect sample objects (which can be null if a model term is unincluded) or `std::vector` for parameter samples (which can be empty if a model term is unincluded)

## New C++ Sampling Routines

Several of stochtree's supported model terms are implemented in R / Python, specifically:

1. **Probit data augmentation**: we sample the latent outcome via two truncated normal draws, centered at the mean predictions -- one for `y = 0` and one for `y = 1`.
2. **BCF adaptive coding**: we recode `Z = 1` and `Z = 0` to `b1` and `b0`, respectively, by regressing `y - mu(x)` on `tau(x) * [Z, 1-Z]`
3. **BCF treatment effect intercept**: we decompose BCF's CATE function into `tau(x) = tau_0 + t(x)` where `tau_0` is a parametric term and `t(x)` is a forest, and we sample `tau_0` by regressing `y - mu(x) - (b_1 * Z + b_0 * (1-Z)) * t(x)` on `b_1 * Z + b_0 * (1-Z)`

### Probit

Probit requires a C++ implementation of a truncated normal sampler, which we can adapt from `scipy`'s implementation, with simplifications allowed by the fact that the probit use case means we are always truncated on `(-inf, 0)` or `[0, inf)`, so don't have to worry about as many of the numerical stability edge cases that their implementation handles. With `boost::math` (already a stochtree dependency), we can achieve this relatively straightforwardly via:

```cpp
#include <boost/math/special_functions/erf.hpp>
#include <random>

inline double standard_uniform_draw_53bit(std::mt19937& gen) {
  int32_t a = gen() >> 5;
  int32_t b = gen() >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

double norm_cdf(double x) {
  return 0.5 * boost::math::erfc(x / std::sqrt(2.0));
}
double norm_inv_cdf(double p) {
  return -std::sqrt(2.0) * boost::math::erfc_inv(2.0 * p);
}

static constexpr double Phi_0 = norm_cdf(0);

inline double sample_std_truncnorm_upper(std::mt19937& gen) {
  double uniform_draw = standard_uniform_draw_53bit(gen);
  return norm_inv_cdf(uniform_draw * Phi_0);
}

inline double sample_std_truncnorm_lower(std::mt19937& gen) {
  double uniform_draw = standard_uniform_draw_53bit(gen);
  return norm_inv_cdf(uniform_draw + (1 - uniform_draw) * Phi_0);
}
```

### Linear Regression

With diagonal Gaussian priors, the linear regressions implemented in 2 and 3 are simply a matter of writting helper functions that wrap stochtree's `normal_sampler.h`

## Ownership and Call Semantics

`BARTSampler` is "stateful" -- it creates, owns, and stores all of the ephemeral state needed to run a sampler, and this state is not retained when it goes out of scope. `BARTSampler` accepts `BARTConfig`, `BARTSamples`, and `BARTData` as inputs.

`BARTData` is a thin wrapper around data pointers -- it owns nothing and is essentially a tracker of memory addresses and metadata.

`BARTConfig` and `BARTSamples` are owned by the "caller" of `BARTSampler` (typically an R or Python session, though it can also be run in standalone C++). The R and Python interfaces unpack `BARTSamples` results into the list / class-based format in which they persist in the `BARTModel` objects in both languages. Ownership of pointers is transferred to the R / Python classes and any parameter sample vectors are copied.

# Value

There are two primary benefits of this RFC:

1. Reducing maintenance headaches and shrinking the bug surface. Avoiding duplication of sampler logic in two different languages means that new modeling features can be implemented and tested faster.
2. Opening the door to future performance boosts. The experiments in building out [faststochtree](https://github.com/andrewherren/faststochtree) have revealed several straightforward opportunities for speedups that can be implemented in stochtree (namely, quantization of features for MCMC, histograms for GFR, multi-lane accumulation for scatter / gather routines).

1 is a quality of life improvement for maintainers and 2 is beneficial both to current users and in opening the door to new use cases.

# Risks and Drawbacks

This is a fairly involved overhaul of core sampling logic. It will involve writing a lot of new C++ code as well as "wrapper" code in R and Python.

Rigorous testing, both for correctness and performance regression, must be performed at every step. Initially, this will be done by including a `cpp_loop` argument to the BART and BCF sampler calls in R and Python that determines whether or not to call the C++ dispatch. This gives us a way of comparing the runtime and empirical performance of the two implementations side-by-side.

We will also add many C++ unit tests and modify the standalone C++ program to run this new interface and measure its performance.

# Alternatives

There are many ways to structure a C++ overhaul of the core sampling loop, but the first and obvious alternative is to leave stochtree the way it is. Several experiments have shown that the current interface is fairly low overhead for most models, compared to a C++ rewrite. The obvious exceptions are models like probit or BCF with adaptive coding, in which substantial sampling logic takes place in R or Python. The counterpoint is outlined in the [Value](#value) section: rewriting the sampling logic in C++ alone is not in itself a major performance boost, but it unlocks many well-documented performance boosts.

# Open Questions

There are many concrete implementation details to be hashed out, namely:

1. How is state managed in multi-threaded samplers?
2. How best to refactor the `ForestTracker` to avoid its current replication of GFR and MCMC state?
3. How do we design for and document the introduction of new models?
4. What points of flexibility do we want to "bake in" to this redesign (i.e. the ability to zero-copy data on input and output, the ability to replace Eigen)?
