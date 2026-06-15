# Standardizing Serialization/Deserialization Formats for Models

Tracking issue: [#401](https://github.com/StochasticTree/stochtree/issues/401)

# Overview

Right now, `stochtree` makes it possible to continue sampling an existing model by passing a JSON representation of the old model to `bart()` / `bcf()` in R or `BARTModel.sample()` / `BCFModel.sample()` in Python. This is a rather "heavy" interface, as it requires a serialization / deserialization roundtrip and also prevents reloading an R model in Python or vice versa. This RFC streamlines the interface for continuing a sampler or predicting from a serialized model and makes it possible to do so cross-platform.

# Motivation

`stochtree`'s JSON-based interface for resurrecting old models supports two primary use cases:

1. Continuing to run an existing BART / BCF sampler, perhaps after inspecting traceplots and deciding to run for longer or with different hyperparameters
2. Resurrecting a BART / BCF model saved to disk or just JSON string and sampling it for longer

[RFC 4](https://github.com/StochasticTree/rfcs/blob/main/0004-cpp-dispatch.md) means the R and Python routines have a much larger shared C++ core and there is a documented API for passing structured inputs / outputs to / from C++. This allows us to simplify use case 1 above, so that a `BARTModel` or `BCFModel` object retains an in-memory pointer to a C++ `BARTSamples` / `BCFSamples` object, which can be appended to by continuing a sampler (with different hyperparameters). Similarly, with the model's samples held in a persistent in-memory `BARTSamples` / `BCFSamples`, we can standardize deserialization so that R / Python reconstruct that samples object from a serialized model (reusing the existing C++ `ForestContainer` serde) and hand it to the sampler to resume — no JSON roundtrip for in-memory continuation, and no need for C++ to parse the envelope.

Finally, this RFC will unlock a currently unsupported use case:

* Allow models built in R to be deserialized into Python and vice versa (for models fit on purely numeric covariates; categorical data must be reduced to a numeric matrix before fitting to be cross-platform portable), and

# Proposed Implementation

This RFC requires a few significant changes to the codebase, which we detail below

## Standardized JSON Format for R and Python Models

Currently, the JSON formats written by R and Python overlap heavily but have a few differences. 
We propose standardized "schemas" for both BART and BCF models, so that JSON outputs from R and Python will match after this RFC

### BART JSON Schema

The BART JSON object follows the schema below

| Field | Value |
| --- | --- |
| `"stochtree_version"` | Version number that model was sampled under |
| `"outcome_scale"` | Scale factor used for standardizing continuous outcomes (1 if no standardization) |
| `"outcome_mean"` | Mean shift factor used for standardizing continuous outcomes (0 if no standardization) |
| `"standardize"` | Whether a continuous outcome was standardized during sampling |
| `"sigma2_init"` | Initial value of global error scale used in the model |
| `"sample_sigma2_global"` | Whether global error scale was sampled or not |
| `"sample_sigma2_leaf"` | Whether mean forest leaf scale was sampled or not |
| `"include_mean_forest"` | Whether the model includes a forest for the conditional mean |
| `"include_variance_forest"` | Whether the model includes a forest for the conditional variance (heteroskedasticity) |
| `"has_rfx"` | Whether the model includes an additive random effects term |
| `"has_rfx_basis"` | Whether or not the model's random effects terms has a basis |
| `"num_rfx_basis"` | Dimension of the random effects basis vector |
| `"num_gfr"` | Number of GFR samples drawn |
| `"num_burnin"` | Number of burnin MCMC samples drawn |
| `"num_mcmc"` | Number of retained MCMC samples (per-chain) |
| `"num_chains"` | Number of MCMC chains run |
| `"keep_every"` | How often an MCMC draw should be retained (values > 1 are referred to as "thinning") |
| `"num_samples"` | Total number of samples in the resulting model |
| `"num_basis"` | Dimension of the leaf basis for the mean forest |
| `"requires_basis"` | Whether the model's mean forest requires a regression basis |
| `"probit_outcome_model"` | Whether the model was sampled under probit data augmentation |
| `"outcome"` | Type of outcome (continuous, binary, ordinal) |
| `"link"` | Link function used in the model (identity, probit, cloglog) |
| `"rfx_model_spec"` | Whether the random effects basis is user-provided or constructed according to the "intercept_only" spec |
| `"cloglog_num_categories"` | Number of cloglog categories |
| `"rfx_unique_group_ids"` | Unique group IDs for random effects terms (NULL if no random effects term included in a model); will be dropped from the unified schema (see Reconciliation section below) |
| `"num_forests"` | Number of forest terms included in a model |
| `"num_random_effects"` | Number of random effects terms included in a model |
| `"parameters"` | Any parametric terms sampled by the model (see below) |
| `"covariate_preprocessor"` (`"preprocessor_metadata"` in R) | Details on how to reconstruct the covariate preprocessor (see below) |
| `"forests"` | Details on how to reconstruct the mean and variance forests (see below) |
| `"random_effects"` | Details on how to reconstruct the random effects terms (see below) |

#### Parameters

| Field | Value |
| --- | --- |
| `"sigma2_global_samples"` | Global error scale parameter |
| `"sigma2_leaf_samples"` | Leaf scale parameter for the mean forest |
| `"cloglog_cutpoint_samples_{i}"` | Cutpoint terms for a cloglog model |

#### Covariate Preprocessor

This is the main point of divergence between the R and Python serialization routines, so it behooves us to review each language's current schema

##### R Preprocessor Schema

| Field | Value |
| --- | --- |
| `"num_numeric_vars"` | Number of numeric variables in a dataset |
| `"num_ordered_cat_vars"` | Number of ordered categorical variables in a dataset |
| `"num_unordered_cat_vars"` | Number of unordered categorical variables in a dataset |
| `"feature_types"` | Vector of type indicators for each feature in a dataset |
| `"original_var_indices"` | Numeric indices of the "original" variable in a dataset (before i.e. unordered categoricals are one-hot-encoded) |
| `"numeric_vars"` | Vector of column names of numeric variables |
| `"ordered_cat_vars"` | Vector of column names of ordered categorical variables |
| `"ordered_unique_level_keys"` | JSON object that maps ordered categorical variable names to a unique key of the form `key_i` (see below) |
| `"ordered_unique_levels"` | JSON object that maps ordered categorical variable names to the unique levels observed for that variable in a training set (see below) |
| `"unordered_cat_vars"` | Vector of column names of unordered categorical variables |
| `"unordered_unique_level_keys"` | JSON object that maps unordered categorical variable names to a unique key of the form `key_i` (see below) |
| `"unordered_unique_levels"` | JSON object that maps unordered categorical variable names to the unique levels observed for that variable in a training set (see below) |

**Variable name key remapping schema `*_unique_level_keys` and `*_unique_levels`**

The `*_unique_level_keys` object creates a series of unique keys in standardized format and maps them to a corresponding variable name. For example if `x1`, `x3`, and `x7` are ordered categorical variables, then `ordered_unique_level_keys` will look like

```
{
  "key_1": "x1",
  "key_2": "x3",
  "key_3": "x7"
}
```

The `*_unique_levels` maps original variable names to their unique categorical values. For example, if `x1` has values `1,2,3`, `x3` has values `2,4,6`, and `x7` has values `-1,0,1`, then this object will look like

```
{
  "x1": ["1","2","3"],
  "x3": ["2","4","6"],
  "x7": ["-1","0","1"]
}
```

##### Python Preprocessor Schema

| Field | Value |
| --- | --- |
| `"is_fitted"` | Whether a preprocessor has been fitted (always true if a model's being serialized) |
| `"num_original_features"` | Number of features in the array / data frame passed to the preprocessor |
| `"num_ordinal_features"` | Number of ordered categorical variables in a dataset (called `num_ordered_cat_vars` in R) |
| `"num_onehot_features"` | Number of unordered categorical variables in a dataset, or other variable types that must be one-hot encoded (called `num_unordered_cat_vars` in R) |
| `"ordinal_feature_index"` | (see below) |
| `"onehot_feature_index"` | (see below) |
| `"processed_feature_types"` | (see below) |
| `"original_feature_types"` | (see below) |
| `"original_feature_indices"` | (see below) |
| `"ordinal_dtype_list"` | (see below) |
| `"ordinal_categories_list"` | (see below) |
| `"onehot_dtype_list"` | (see below) |
| `"onehot_categories_list"` | (see below) |

**Feature type lists**

For convenience, we'll say the original covariate set (data frame or numpy array) is of dimension `p` and the potentially expanded covariate set (with categorical columns one-hot-encoded) is of dimension `p_new`.

`original_feature_types` is a length `p` list that stores a string representation of the feature type of a column in the original dataset. Options are `"category"`, `"string"`, `"boolean"`, `"integer"`, `"float"`, and `"unsupported"` (rare).

`processed_feature_types` is a length `p_new` JSON array that stores a binary representation of the transformed feature set: 0 indicates that the feature was passed through as-is and 1 indicates some sort of transformation (either one-hot encoding or converting ordered categorical variables to standardized 0-based integers).

`"original_feature_indices"` is a length `p_new` JSON array that stores the index of the original feature set that corresponds to a column in the transformed feature set. For example, in a simple dataset where feature 0 is numeric and feature 1 is unordered categorical with 3 levels, `"original_feature_indices"` would be a list `[0,1,1,1,1]` (the fourth 1 corresponds to an "other" category that we include by default in one-hot encodings to handle unseen categories).

`"ordinal_feature_index"` is a length `p` JSON array that stores -1 if a column isn't ordinal and, otherwise, a unique index `i` that stores the position of an ordinal feature's unique values in a `"ordinal_categories_list"` list (see below).

`"ordinal_categories_list"` is a JSON object that stores the unique ordinal categories for each ordinal feature. Ordinal features are included in this list in the order their column appears in the original dataset (i.e. for a dataset with 5 features, of which columns 2 and 4 are ordinal, the first element in `"ordinal_categories_list"` (indexed by `cats_0`) would correspond to feature 2 and the second element (indexed by `cats_1`) would correspond to feature 4).

`"ordinal_dtype_list"` is a JSON object that maps each ordinal feature to a string representation of the type of its original categories (options are `int`, `float`, and `str`).

`"onehot_feature_index"` is a length `p` list that stores -1 if a column isn't unordered categorical and, otherwise, a unique index `i` that stores the position of a categorical feature's unique values in a `"onehot_categories_list"` list (see below).

`"onehot_categories_list"` is a list-of-lists that stores the unique categories for each unordered categorical feature. Unordered categorical features are included in this list in the order their column appears in the original dataset (i.e. for a dataset with 5 features, of which columns 1 and 3 are categorical, the first element in `"onehot_categories_list"`  (indexed by `cats_0`) would correspond to feature 1 and the second element (indexed by `cats_1`) would correspond to feature 3).

`"onehot_dtype_list"` is a JSON object that maps each unordered categorical feature to a string representation of the type of its original categories (options are `int`, `float`, and `str`).

#### Forests

Forests are stored under the `"forests"` key. Each forest is serialized to JSON in C++ and added under a self-describing named key rather than a positional index. For BART the keys are `"mean_forest"` and `"variance_forest"`; a model includes whichever of these it sampled. Named keys replace the previous positional `forest_0` / `forest_1` convention, whose meaning depended on which forests were present (e.g. the variance forest was `forest_1` if a mean forest existed and `forest_0` otherwise). With named keys a reader can load a forest directly, without first inspecting `include_mean_forest` / `include_variance_forest`.

#### Random Effects

Random effects terms are organized under the `"random_effects"` key. There are three JSON objects that comprise a random effects term. Each is technically numbered, though BART and BCF only support a single additive random effects term out of the box.

1. Random effects container: stored under the `'random_effect_container_0'` key (0 is a convention to allow for multiple random effects terms in custom models)
2. Random effects "label mapper": stored under the `'random_effect_label_mapper_0'`
3. Unique group IDs are stored under the `'random_effect_groupids_0'` key

#### Reconciliation

We standardize on a new unified schema for both R and Python BART

| New Field | Previous R Field | Previous Python Field | Notes | Action |
| --- | --- | --- | --- | --- |
| `"schema_version"` |  |  | Integer identifying the serialized format (see [Explicit handling of older JSON formats](#explicit-handling-of-older-json-formats)); `1` for this RFC's schema | Add to R and Python |
| `"stochtree_version"` | `"stochtree_version"` | `"stochtree_version"` | Version number that model was sampled under |  |
| `"platform"` |  |  | Whether model was sampled in R or Python | Add to R and Python |
| `"outcome_scale"` | `"outcome_scale"` | `"outcome_scale"` | Scale factor for standardizing continuous outcomes (1 if none) |  |
| `"outcome_mean"` | `"outcome_mean"` | `"outcome_mean"` | Mean shift factor for standardizing continuous outcomes (0 if none) |  |
| `"standardize"` | `"standardize"` | `"standardize"` | Whether a continuous outcome was standardized during sampling |  |
| `"sigma2_init"` | `"sigma2_init"` | `"sigma2_init"` | Initial value of global error scale |  |
| `"sample_sigma2_global"` | `"sample_sigma2_global"` | `"sample_sigma2_global"` | Whether global error scale was sampled |  |
| `"sample_sigma2_leaf"` | `"sample_sigma2_leaf"` | `"sample_sigma2_leaf"` | Whether mean forest leaf scale was sampled |  |
| `"include_mean_forest"` | `"include_mean_forest"` | `"include_mean_forest"` | Whether model includes a forest for the conditional mean |  |
| `"include_variance_forest"` | `"include_variance_forest"` | `"include_variance_forest"` | Whether model includes a forest for the conditional variance (heteroskedasticity) |  |
| `"has_rfx"` | `"has_rfx"` | `"has_rfx"` | Whether model includes an additive random effects term |  |
| `"has_rfx_basis"` | `"has_rfx_basis"` | `"has_rfx_basis"` | Whether the random effects term has a basis |  |
| `"num_rfx_basis"` | `"num_rfx_basis"` | `"num_rfx_basis"` | Dimension of the random effects basis vector |  |
| `"num_gfr"` | `"num_gfr"` | `"num_gfr"` | Number of GFR samples drawn |  |
| `"num_burnin"` | `"num_burnin"` | `"num_burnin"` | Number of burnin MCMC samples drawn |  |
| `"num_mcmc"` | `"num_mcmc"` | `"num_mcmc"` | Number of retained MCMC samples (per-chain) |  |
| `"num_chains"` | `"num_chains"` | `"num_chains"` | Number of MCMC chains run |  |
| `"keep_every"` | `"keep_every"` | `"keep_every"` | How often an MCMC draw is retained (>1 = thinning) |  |
| `"num_samples"` | `"num_samples"` | `"num_samples"` | Total number of samples in the resulting model |  |
| `"num_covariates"` | `"num_covariates"` |  | Dimension of covariates from which the original model was trained | Add to Python |
| `"num_basis"` | `"num_basis"` | `"num_basis"` | Dimension of the leaf basis for the mean forest |  |
| `"requires_basis"` | `"requires_basis"` | `"requires_basis"` | Whether the mean forest requires a regression basis |  |
| `"probit_outcome_model"` | `"probit_outcome_model"` | `"probit_outcome_model"` | Whether sampled under probit data augmentation |  |
| `"outcome"` | `"outcome"` | `"outcome"` | Type of outcome (continuous, binary, ordinal) |  |
| `"link"` | `"link"` | `"link"` | Link function (identity, probit, cloglog) |  |
| `"rfx_model_spec"` | `"rfx_model_spec"` | `"rfx_model_spec"` | Whether rfx basis is user-provided or `"intercept_only"` |  |
| `"cloglog_num_categories"` | `"cloglog_num_categories"` | `"cloglog_num_categories"` | Number of cloglog categories |  |
| `"rfx_unique_group_ids"` | `"rfx_unique_group_ids"` | | Only ever written by R and duplicates a term inside `"random_effects"` | Remove from R and Python |
| `"num_forests"` | `"num_forests"` | `"num_forests"` | Number of forest terms included |  |
| `"num_random_effects"` | `"num_random_effects"` | `"num_random_effects"` | Number of random effects terms included |  |
| `"parameters"` |  |  | Parametric terms sampled by the model |  |
| `"covariate_preprocessor"` | `"preprocessor_metadata"` | `"covariate_preprocessor"` |  | Harmonize naming between R and Python |
| `"forests"` |  |  | Details to reconstruct the mean and variance forests |  |
| `"random_effects"` |  |  | Details to reconstruct the random effects terms |  |

Note that the top-level `"rfx_unique_group_ids"` field is being removed from the R serialization routine (it duplicates `"random_effect_groupids_0"` stored in `"random_effects"` and is only written to from R)

##### Parameters

[*Unchanged*]

##### Covariate Preprocessor

Both covariate preprocessor schemas contain similar information: which features are ordered categorical (codes remapped to integers) or unordered categorical (one-hot encoded), and which categories are observed in each feature. They differ in structure, reflecting the design of the underlying R and Python objects.

For 0.5.0 we deliberately **do not unify** these two native schemas. Each platform's native preprocessor keeps its existing schema (documented above) and is **same-platform only**: an R-written native preprocessor is read back by R, a Python-written one by Python. The only changes here are the high-level field rename (`preprocessor_metadata` -> `covariate_preprocessor`, see the [Reconciliation](#reconciliation) table) and one added field inside the `covariate_preprocessor` object, `cross_platform_portable`, described below.

**Cross-platform portability**

Superficially, the R `data.frame` and pandas `DataFrame` are both containers for storing columns of data with (potentially) disparate types. Dataframes with entirely numeric columns can be treated more or less equivalently between R and Python. However, categorical, string, and other more complex column types have many implementation differences between R and Python that make it difficult to process data equivalently.

Each platform's internal covariate preprocessor stores the information needed to convert an R data frame to an R matrix or a pandas data frame to a numpy array. Rewriting this schema to ensure fully replicable cross-platform processing is beyond the scope of this proposal. Instead, we articulate three common cases and establish clear expectations for how `stochtree` handles each case:

* **Portable: all-numeric training data.** If every covariate is numeric (`numeric` / `integer` / `logical` in R; `float` / `int` / `bool` in pandas; or a bare matrix / `ndarray`), the covariate transform for each column is the identity and the model carries no platform-specific preprocessing complications. Such a model **must** load and predict in either language. 
    * *Note*: this also provides a separate path to cross-platform replicability: users who want a portable model may pre-convert their data to a numeric model matrix / array themselves before sampling. After that, they can run the same model from a different platform as long as the data is arranged and oriented in the same state as the training data.
* **Same-platform only: non-numeric training data.** Any model with non-numeric covariates (categorical, string, etc) that were encoded through an R / Python preprocessor can only be deserialized on the same-platform. Same-platform deserialization is unaffected. Any attempt at a *cross-platform* load is **refused with a clear error** naming the offending column(s), as opposed to being silently mis-transformed.

The contract concerns reproduction of the *covariate transformation* (raw covariates -> numeric model matrix), not reconstruction of a native `data.frame` / `DataFrame`. Because the model only ever consumes the transformed numeric matrix, the all-numeric guarantee is sufficient for both prediction and continued sampling.

**The `cross_platform_portable` flag.** The determination is machine-checkable via a single boolean stored in the `covariate_preprocessor` object, computed at serialization time:

```
cross_platform_portable = (all covariates numeric)
```

When false it is accompanied by the list of offending columns. The cross-platform loader checks it: when the writer's `"platform"` differs from the loading platform and the flag is false, the load errors; same-platform loads ignore the flag entirely. For legacy (v0) models the flag is computed during the v0 -> v1 migration from the model's feature types (true only for all-numeric models).

The user-facing mental model: **a cross-platform load works exactly when the model was fit on purely numeric covariates.** An unsupported cross-platform load is detected and refused, never silently mis-transformed, so a successful cross-platform load can be trusted as a correct one.

> *Future work (out of scope for 0.5.0).* A standardized, language-neutral preprocessor that reduces categorical data to a numeric matrix portably -- letting categorical models also qualify as cross-platform -- is a natural extension and is deliberately deferred. Nothing in this schema forecloses it: such a preprocessor would simply be another way to produce an all-numeric (portable) model, requiring no envelope changes. For 0.5.0 we document only the recommended practice (reduce categoricals to a numeric matrix before fitting); a thin convenience helper may ship, but it carries no portable-serialization guarantee.

##### Forests

The forest keys are renamed to the self-describing names defined in the schema section above (`mean_forest` / `variance_forest`). This rename applies identically to R and Python, so there are no R-vs-Python reconciliation differences to resolve.

##### Random Effects

[*Unchanged*]

### BCF JSON Schema

The BCF JSON object follows the schema below

| Field | Value |
| --- | --- |
| `"stochtree_version"` | Version number that model was sampled under |
| `"outcome_scale"` | Scale factor used for standardizing continuous outcomes (1 if no standardization) |
| `"outcome_mean"` | Mean shift factor used for standardizing continuous outcomes (0 if no standardization) |
| `"standardize"` | Whether a continuous outcome was standardized during sampling |
| `"sigma2_init"` | Initial value of global error scale used in the model |
| `"sample_sigma2_global"` | Whether global error scale was sampled or not |
| `"sample_sigma2_leaf_mu"` | Whether prognostic forest leaf scale was sampled or not |
| `"sample_sigma2_leaf_tau"` | Whether treatment effect forest leaf scale was sampled or not |
| `"include_variance_forest"` | Whether the model includes a forest for the conditional variance (heteroskedasticity) |
| `"propensity_covariate"` | Whether and how the propensity score is included as a covariate |
| `"has_rfx"` | Whether the model includes an additive random effects term |
| `"has_rfx_basis"` | Whether or not the model's random effects terms has a basis |
| `"num_rfx_basis"` | Dimension of the random effects basis vector |
| `"multivariate_treatment"` | Whether treatment is multivariate |
| `"adaptive_coding"` | Whether coding parameters for treated and control groups are sampled |
| `"sample_tau_0"` | Whether a parametric "intercept term" is sampled for the treatment effect function |
| `"internal_propensity_model"` | Whether an internal propensity model is sampled and maintained alongside the BCF model |
| `"num_gfr"` | Number of GFR samples drawn |
| `"num_burnin"` | Number of burnin MCMC samples drawn |
| `"num_mcmc"` | Number of retained MCMC samples (per-chain) |
| `"num_chains"` | Number of MCMC chains run |
| `"keep_every"` | How often an MCMC draw should be retained (values > 1 are referred to as "thinning") |
| `"num_samples"` | Total number of samples in the resulting model |
| `"num_covariates"` | Total number of covariates in the input dataset |
| `"probit_outcome_model"` | Whether the model was sampled under probit data augmentation |
| `"outcome"` | Type of outcome (continuous, binary, ordinal) |
| `"link"` | Link function used in the model (identity, probit, cloglog) |
| `"rfx_model_spec"` | Whether the random effects basis is user-provided or constructed according to the `"intercept_only"` / `"intercept_plus_treatment"` specs |
| `"tau_0_dim"` | Dimension of the treatment effect parametric term, if it is in the model |
| `"num_forests"` | Number of forest terms included in a model |
| `"num_random_effects"` | Number of random effects terms included in a model |
| `"parameters"` | Any parametric terms sampled by the model (see below) |
| `"covariate_preprocessor"` (`"preprocessor_metadata"` in R) | Details on how to reconstruct the covariate preprocessor (see the discussion in the [BART JSON Schema](#bart-json-schema) section) |
| `"forests"` | Details on how to reconstruct the mean and variance forests (see below) |
| `"random_effects"` | Details on how to reconstruct the random effects terms (see below) |
| `"bart_propensity_model"` | JSON representation of the internal BART model used to model `P(Z|X)` |

#### Parameters

| Field | Value |
| --- | --- |
| `"sigma2_global_samples"` | Global error scale parameter |
| `"sigma2_leaf_mu_samples"` | Prognostic forest leaf scale parameter |
| `"sigma2_leaf_tau_samples"` | Treatment effect forest leaf scale parameter |
| `"b1_samples"` | Treated coding parameter in the "adaptive coding" term |
| `"b0_samples"` | Control coding parameter in the "adaptive coding" term |
| `"tau_0_samples"` | Parametric treatment effect samples |

#### Forests

Forests are stored under the `"forests"` key. Each forest is serialized to JSON in C++ and added under a self-describing named key rather than a positional index: `"prognostic_forest"`, `"treatment_forest"`, and (if sampled) `"variance_forest"`. As with BART, named keys let a reader load each forest without inspecting which forests are present, replacing the previous positional `forest_0` / `forest_1` / `forest_2` convention.

#### Random Effects

The components of a random effects term are stored under the `"random_effects"` key and contain three separate entities:

1. Random effects container: stored under the `'random_effect_container_0'` key (0 is a convention to allow for multiple random effects terms in custom models)
2. Random effects "label mapper": stored under the `'random_effect_label_mapper_0'`
3. Unique group IDs are stored under the `'random_effect_groupids_0'` key

#### Reconciliation

We standardize on a new unified schema for both R and Python BCF

| New Field | Previous R Field | Previous Python Field | Notes | Action |
| --- | --- | --- | --- | --- |
| `"schema_version"` |  |  | Integer identifying the serialized format (see [Explicit handling of older JSON formats](#explicit-handling-of-older-json-formats)); `1` for this RFC's schema | Add to R and Python |
| `"stochtree_version"` | `"stochtree_version"` | `"stochtree_version"` | Version number that model was sampled under |  |
| `"platform"` |  |  | Whether model was sampled in R or Python | Add to R and Python |
| `"outcome_scale"` | `"outcome_scale"` | `"outcome_scale"` | Scale factor for standardizing continuous outcomes (1 if none) |  |
| `"outcome_mean"` | `"outcome_mean"` | `"outcome_mean"` | Mean shift factor for standardizing continuous outcomes (0 if none) |  |
| `"standardize"` | `"standardize"` | `"standardize"` | Whether a continuous outcome was standardized during sampling |  |
| `"sigma2_init"` | `"sigma2_init"` | `"sigma2_init"` | Initial value of global error scale |  |
| `"sample_sigma2_global"` | `"sample_sigma2_global"` | `"sample_sigma2_global"` | Whether global error scale was sampled |  |
| `"sample_sigma2_leaf_mu"` | `"sample_sigma2_leaf_mu"` | `"sample_sigma2_leaf_mu"` | Whether prognostic forest leaf scale was sampled |  |
| `"sample_sigma2_leaf_tau"` | `"sample_sigma2_leaf_tau"` | `"sample_sigma2_leaf_tau"` | Whether treatment effect forest leaf scale was sampled |  |
| `"include_variance_forest"` | `"include_variance_forest"` | `"include_variance_forest"` | Whether model includes a forest for the conditional variance (heteroskedasticity) |  |
| `"propensity_covariate"` | `"propensity_covariate"` | `"propensity_covariate"` | Whether and how the propensity score is included as a covariate |  |
| `"has_rfx"` | `"has_rfx"` | `"has_rfx"` | Whether model includes an additive random effects term |  |
| `"has_rfx_basis"` | `"has_rfx_basis"` | `"has_rfx_basis"` | Whether the random effects term has a basis |  |
| `"num_rfx_basis"` | `"num_rfx_basis"` | `"num_rfx_basis"` | Dimension of the random effects basis vector |  |
| `"multivariate_treatment"` | `"multivariate_treatment"` | `"multivariate_treatment"` | Whether treatment is multivariate |  |
| `"adaptive_coding"` | `"adaptive_coding"` | `"adaptive_coding"` | Whether coding parameters for treated and control groups are sampled |  |
| `"sample_tau_0"` | `"sample_tau_0"` | `"sample_tau_0"` | Whether a parametric "intercept term" is sampled for the treatment effect function |  |
| `"internal_propensity_model"` | `"internal_propensity_model"` | `"internal_propensity_model"` | Whether an internal propensity model is sampled alongside the BCF model |  |
| `"num_gfr"` | `"num_gfr"` | `"num_gfr"` | Number of GFR samples drawn |  |
| `"num_burnin"` | `"num_burnin"` | `"num_burnin"` | Number of burnin MCMC samples drawn |  |
| `"num_mcmc"` | `"num_mcmc"` | `"num_mcmc"` | Number of retained MCMC samples (per-chain) |  |
| `"num_chains"` | `"num_chains"` | `"num_chains"` | Number of MCMC chains run |  |
| `"keep_every"` | `"keep_every"` | `"keep_every"` | How often an MCMC draw is retained (>1 = thinning) |  |
| `"num_samples"` | `"num_samples"` | `"num_samples"` | Total number of samples in the resulting model |  |
| `"num_covariates"` | `"num_covariates"` |  | Total number of covariates in the input dataset | Add to Python |
| `"probit_outcome_model"` | `"probit_outcome_model"` | `"probit_outcome_model"` | Whether sampled under probit data augmentation |  |
| `"outcome"` | `"outcome"` | `"outcome"` | Type of outcome (continuous, binary, ordinal) |  |
| `"link"` | `"link"` | `"link"` | Link function (identity, probit, cloglog) |  |
| `"rfx_model_spec"` | `"rfx_model_spec"` | `"rfx_model_spec"` | Whether rfx basis is user-provided or `"intercept_only"` / `"intercept_plus_treatment"` |  |
| `"tau_0_dim"` | `"tau_0_dim"` | `"tau_0_dim"` | Dimension of the treatment effect parametric term, if present |  |
| `"num_forests"` | `"num_forests"` | `"num_forests"` | Number of forest terms included |  |
| `"num_random_effects"` | `"num_random_effects"` | `"num_random_effects"` | Number of random effects terms included |  |
| `"parameters"` |  |  | Parametric terms sampled by the model |  |
| `"covariate_preprocessor"` | `"preprocessor_metadata"` | `"covariate_preprocessor"` |  | Harmonize schema between R and Python |
| `"forests"` | `"forests"` | `"forests"` | Details to reconstruct the prognostic, treatment effect, and variance forests |  |
| `"random_effects"` | `"random_effects"` | `"random_effects"` | Details to reconstruct the random effects terms |  |
| `"bart_propensity_model"` | `"bart_propensity_model"` | `"bart_propensity_model"` | JSON representation of the internal BART model used to model `P(Z\|X)` |  |

Note that the top-level `"rfx_unique_group_ids"` field is being removed from the R serialization routine (it duplicates `"random_effect_groupids_0"` stored in `"random_effects"` and is only written to from R)

##### Parameters

[*Unchanged*]

##### Covariate Preprocessor

This is handled exactly as in the [BART JSON Schema](#bart-json-schema) section

##### Forests

The forest keys are renamed to the self-describing names defined in the schema section above (`prognostic_forest` / `treatment_forest` / `variance_forest`). This rename applies identically to R and Python, so there are no R-vs-Python reconciliation differences to resolve.

##### Random Effects

[*Unchanged*]

## Explicit handling of older JSON formats

### Schema versioning

We introduce a single top-level integer field, `"schema_version"`, that identifies the *structure and meaning* of the serialized envelope. It is deliberately decoupled from the package version:

| Field | Role | Gates behavior? |
| --- | --- | --- |
| `"schema_version"` | Identifies the structure / contract of the serialized format | Yes |
| `"stochtree_version"` | Package version the model was sampled under | No (informational breadcrumb only) |
| `"platform"` | Which native preprocessor object (R vs Python) to reconstruct | No (reconstruction only, not parsing) |

`schema_version` is a monotonic integer rather than a semantic version -- the only operations we ever perform on it are `==`, `<`, and `>`, so an integer avoids any "is 1.2 compatible with 1.1" ambiguity. The unified schema introduced by this RFC is `schema_version = 1`; a missing field denotes a legacy (pre-0.5.0) model, treated as version 0.

Keeping these three fields separate is the core idea: `schema_version` says *what fields exist and what they mean*, `platform` says *which native object to rebuild*, and `stochtree_version` is a human-readable breadcrumb that -- consistent with the current behavior documented in `utils.py` -- never gates deserialization. This is what lets a model serialized under a dev build (where the stamped `stochtree_version` may be `"dev"` or `0.5.0.9000`) parse identically to one serialized under the released `0.5.0`.

**Bump policy.** `schema_version` increments *only* on a breaking change:

* Additive, optional field with a safe default -> **do not bump** (readers tolerate unknown fields and supply defaults for missing ones)
* Rename / remove / re-type a field, change a field's semantics, or change a structural convention (e.g. the positional -> named forest keys) -> **bump**

This keeps the integer meaningful and bumps rare.

**Augmentation vs. a migration rung.** The two halves of the bump policy are two *different* mechanisms, and it helps to keep them distinct:

* **Augmentation** handles additive drift *within* a `schema_version`: an older file simply lacks a field a newer writer adds. There is no version change and therefore no rung -- the **parser** supplies the default. As long as we hold the additive-only discipline, loading an older file is *just* augmentation and the ladder stays empty.
* **A migration rung** (`vN -> vN+1`) handles a *non-additive* change -- a rename, removal, re-type, or structural-convention change. Augmentation cannot express these (it can default a missing `mean_forest`, but it cannot rename `forest_0` into one); the rung performs the actual key rewrite.

So the parser's robust defaulting is what keeps rungs rare: anything we can change additively costs zero rungs, and forward-compatibility rests on the parser defaulting *every* optional field reliably -- not on the ladder. The sole exception at launch is `v0 -> v1` (below), which bundles all three flavors: augmentation of missing legacy fields, several rewrites (`preprocessor_metadata` -> `covariate_preprocessor`, dropping `rfx_unique_group_ids`, positional -> named forest keys), and two synthesized fields (`platform`, `cross_platform_portable`).

**Reader rules.** A reader compares the loaded `schema_version` (call it `v`) against the maximum version it supports (`current`):

| Loaded `v` | Action |
| --- | --- |
| `v == current` | Parse directly |
| `v < current` | Run the migration ladder up to `current`, then parse |
| `v > current` | **Hard error**: the model was written by a newer stochtree; instruct the user to upgrade |
| absent | Legacy model; treat as version 0 and run the field-presence heuristic below |

We fail loudly on `v > current` rather than attempting a best-effort parse: silently mis-loading a statistical model is a correctness hazard.

### Migration ladder

Rather than a parser that branches on version throughout (the pattern the field-presence heuristic below already embodies), we up-convert on read through a chain of pure functions `migrate_v0_to_v1`, `migrate_v1_to_v2`, ..., applied in sequence until the JSON reaches `current`. A single parser then ingests only current-schema JSON:

```
json_in -> [detect schema_version] -> migrate (0 -> 1 -> 2 -> ...) -> parse(current) -> BARTSamples / BCFSamples
```

All legacy knowledge lives in small, append-only, individually testable migration steps (each with a `vN fixture -> vN+1 fixture` unit test); the live parser stays clean and only ever sees the current schema. Bumping the schema is then "append one migration plus one golden fixture," never a rewrite of the parser.

Migrations live in **R and Python, not C++**. At 0.5.0 the ladder has exactly one rung (`v0 -> v1`, since `current = 1`), and that rung closely mirrors the default-filling and field reconciliation each language already performs on load, so a per-language implementation is the lighter path. Divergence between the two is caught by the shared, language-neutral golden fixtures -- the same fixture JSON is asserted in both languages (see [Testing schema migrations](#testing-schema-migrations)). Consolidating the ladder into a single shared C++ `migrate_json(string) -> string` remains a clean option for *later*, warranted only once the ladder grows additional rungs; it is deferred for 0.5.0.

**v0 -> v1 is the platform-aware migration.** Pre-0.5.0 R and Python diverge (`preprocessor_metadata` vs `covariate_preprocessor`, the R-only `rfx_unique_group_ids`), so the v0 -> v1 step branches on platform and *is* the [Reconciliation](#reconciliation) work described above: the high-level field renames and removals, plus computing `cross_platform_portable` from the model's feature types. It does **not** unify the two native preprocessor layouts -- those stay per-platform and same-platform-only -- so the migration leaves the `covariate_preprocessor` sub-object essentially in place (renamed, not restructured). Note the circularity: `platform` is a field this RFC *introduces* in v1, so a legacy model does not carry it -- the v0 -> v1 migration is the *producer* of `platform`, not a consumer. It therefore infers the source platform from structural fingerprints in the legacy JSON (`preprocessor_metadata` present and `covariate_preprocessor` absent => R; `covariate_preprocessor` present => Python; the R-only `rfx_unique_group_ids` corroborates R), with a documented fallback discriminator for models that carry no preprocessor metadata at all. Every migration from v1 onward operates on the single unified format and is platform-agnostic.

**Forest container versioning.** `schema_version` covers the model *envelope* (metadata, parameters, preprocessor, and how forests are keyed). The lower-level C++ `ForestContainer` JSON -- the component most likely to change for performance reasons -- should carry its own independent version internally, so that a forest-layout change does not force re-migration of unrelated envelope components.

**Nested propensity model.** The BCF `bart_propensity_model` is a nested BART envelope written by the same unified writer, so it carries the same `schema_version`; migrations recurse into it.

### Legacy version detection (the v0 sub-classifier)

When `schema_version` is absent, the model predates this RFC and we fall back to the existing field-presence heuristic to determine which legacy migration entry point to use. This is the current inference logic, demoted from a primary mechanism to the bottom rung of the migration ladder.

Note that this heuristic returns a heterogeneous result: a concrete version string (e.g. `"0.4.3"`) when version stamping existed, a bracket (e.g. `"<0.4.1"`) for the pre-stamping era, or `"unknown"`. Because the bulk of serialized models in the wild come from the stamped-but-pre-`schema_version` era (~0.3.0 through 0.4.x), the v0 importer normalizes all of these onto a single internal `legacy_format_id` enum before dispatching, rather than routing on a value that is sometimes a version and sometimes a range. This step also defines an explicit supported floor (models older than the floor, and any `"unknown"` result, are rejected with a clear message rather than parsed best-effort -- the same correctness-hazard argument applied to `v > current` above).

On the R side, the heuristic proceeds largely by checking for the presence of fields:

```r
inferStochtreeJsonVersion <- function(json_object) {
  has_field <- function(name) {
    json_contains_field_cpp(json_object$json_ptr, name)
  }
  has_subfolder_field <- function(subfolder, name) {
    json_contains_field_subfolder_cpp(json_object$json_ptr, subfolder, name)
  }

  if (has_field("stochtree_version")) {
    return(json_object$get_string("stochtree_version"))
  }

  # outcome/link in outcome_model were added in ~0.4.1
  if (
    !has_subfolder_field("outcome_model", "outcome") ||
      !has_subfolder_field("outcome_model", "link")
  ) {
    return("<0.4.1")
  }

  # has_rfx_basis / num_rfx_basis were added in ~0.4.0
  if (!has_field("has_rfx_basis") || !has_field("num_rfx_basis")) {
    return("<0.4.0")
  }

  # internal_propensity_model was added in ~0.3.2 (BCF only)
  if (
    has_field("propensity_covariate") && !has_field("internal_propensity_model")
  ) {
    return("<0.3.2")
  }

  # rfx_model_spec and preprocessor_metadata were added in ~0.3.0
  if (!has_field("rfx_model_spec") || (!has_field("preprocessor_metadata") && !has_field("covariate_preprocessor"))) {
    return("<0.3.0")
  }

  return("unknown")
}
```

This process is similar in python:

```python
def _infer_stochtree_version(json_string: str) -> str:
    try:
        d = json.loads(json_string)
    except Exception:
        return "unknown"

    if "stochtree_version" in d:
        return d["stochtree_version"]

    # outcome/link were added in ~0.4.1
    outcome_model = d.get("outcome_model", {})
    if "outcome" not in outcome_model or "link" not in outcome_model:
        return "<0.4.1"

    # has_rfx_basis / num_rfx_basis were added in ~0.4.0
    if "has_rfx_basis" not in d or "num_rfx_basis" not in d:
        return "<0.4.0"

    # internal_propensity_model was added in ~0.3.2 (BCF only; absent in BART JSON)
    # Only flag this if we can confirm it's a BCF JSON by checking a BCF-only field
    if "propensity_covariate" in d and "internal_propensity_model" not in d:
        return "<0.3.2"

    # rfx_model_spec and covariate_preprocessor were added in ~0.3.0
    if "rfx_model_spec" not in d or ("covariate_preprocessor" not in d and "preprocessor_metadata" not in d):
        return "<0.3.0"

    return "unknown"
```

### Testing schema migrations

Forward- and backward-compatibility are guarded by checked-in golden fixtures: one serialized model per `schema_version` (and per legacy bracket), with a test that loads each and asserts the resulting in-memory state. Older fixtures must continue to load indefinitely. Cross-platform fixtures pair an R-written and a Python-written **all-numeric** model at the same `schema_version` and assert that both load in *both* languages to identical state; a separate fixture pairs a native-categorical model with the assertion that a cross-platform load is *refused* with a clear error (cross-platform loading is only guaranteed for all-numeric models). Adding a new schema version means adding a migration plus a new golden fixture; existing fixtures are never modified.

The coverage is not uniform across rungs. Each `vN -> vN+1` step for `N >= 1` operates on the single unified format and needs only its one `vN fixture -> vN+1 fixture` pair. The `v0 -> v1` rung is the exception and needs the heaviest coverage: its input space is the messy pre-0.5.0 era (multiple legacy brackets, optional/absent fields, R-vs-Python divergence, and the platform inference it produces rather than consumes), so it warrants roughly one golden fixture *per legacy bracket* and per platform, exercising the field-presence sub-classifier and the platform-fingerprint fallback. Treat `v0 -> v1` as the rung most likely to harbor edge cases and fixture it accordingly.

## Persisting pointers to C++ model samples objects in BART and BCF model classes

Right now, the C++ API for BART and BCF models creates several objects in-memory, namely:

1. A `BARTSamples` / `BCFSamples` object which wraps containers for parameter, forest, and random effect samples
2. A `BARTSampler` / `BCFSampler` object which initializes model state and runs the sampler, writing out to `BARTSamples` / `BCFSamples`

The sampler object is destroyed when it goes out of scope (i.e. after the call to C++ concludes) and the samples object has each of its components unpacked (copied) into R and Python containers or moved to managed external pointers and then is also destroyed when it goes out of scope.

This RFC proposes that the BART and BCF objects in R and Python maintain persistent pointers to `BARTSamples` / `BCFSamples` C++ objects, unpacking them only when they are queried by the `extract_parameter` method.

Both the samples object and the `BARTSampler` / `BCFSampler` (which holds the sampler's RNG state and references to the training data) are owned by the R / Python model object and live exactly as long as it does. Tying their lifetime to model-object scope has two consequences: (1) `continue_sampling` resumes the same RNG stream, so a continued run is statistically equivalent to having drawn the additional samples in the original call; and (2) the training data and sampler state remain resident in C++ memory for the model object's lifetime -- a deliberate space-for-functionality tradeoff. For multi-chain models, this implies one persistent sampler state per chain.

### Migration plan

Redesigning the BART and BCF objects to store results in a C++ samples object, rather than internal fields like `global_var_samples` would break any code that directly references these internal fields (as opposed to code that queries them via `extract_parameter`), so we need a plan for graceful deprecation.

For the initial release (0.5.0) we keep these accesses working rather than breaking them outright: when a user reads a moved field, we **auto-load the requested term** from the C++ samples object (equivalent to the corresponding `extract_parameter` call) and emit a `DeprecationWarning`. The shim is removed in a later release, after which the access raises (Python) / returns `NULL` (R). In Python, we do this with a custom `__getattr__`

```python
# Maps each moved attribute to the extract_parameter key that now backs it.
_MOVED_ATTRS = {
  "global_var_samples": "sigma2_global",
  "leaf_scale_samples": "sigma2_leaf",
}

class BARTModel:
    def __getattr__(self, name):
        if name in _MOVED_ATTRS:
            param = _MOVED_ATTRS[name]
            warnings.warn(
                f"'{name}' is no longer a direct attribute of BARTModel as of version 0.5.0; "
                f"access it via model.extract_parameter('{param}'). This compatibility shim "
                f"will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            # __getattr__ only fires on failed lookups, so this does not recurse.
            return self.extract_parameter(param)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

In R with an S3 object, we'd do this by overloading the `$` access operator

```r
`$.bartmodel` <- function(x, name) {
  # Maps each moved field to the extract_parameter key that now backs it.
  moved <- list(
    sigma2_global_samples = "sigma2_global",
    sigma2_leaf_samples = "sigma2_leaf"
  )
  if (name %in% names(moved)) {
    warning(sprintf("'%s' is no longer a direct field of the bartmodel object as of version 0.5.0; use extract_parameter(model, '%s'). This compatibility shim will be removed in a future release.", name, moved[[name]]), call. = FALSE)
    # Auto-load the requested term for one deprecation cycle.
    return(extract_parameter(x, moved[[name]]))
  }
  .subset2(x, name)
}
```

### Continued sampling

The proposed redesign keeps both `BARTSampler` / `BCFSampler` and `BARTSamples` / `BCFSamples` C++ classes alive, so that continued sampling is as straightforward as dispatching the C++ sampler for more iterations. The simplest way to make this happen is a new `continue_sampling` method in both R and Python that appends to the `BARTSamples` / `BCFSamples` objects.

This method should allow users to pass through updated model hyperparameters in the same format as the original BART / BCF sampler dispatch. These configs are supplied by the user at continuation time and may differ from the original run (e.g. different tree split priors). Structural parameters that are fixed by the existing model -- notably `num_trees`, which is determined by the serialized forest -- are recovered from the model rather than re-specified; the runtime warns and ignores attempts to change them. One consequence for models reconstructed from JSON: any hyperparameter the user does not re-specify falls back to the package default rather than the value used in the original fit (those values are not part of the serialized schema), so faithful continuation of a deserialized model requires re-passing the relevant hyperparameters.

#### BART API

The continued sampler API in R BART will look like:

```r
continue_sampling.bartmodel <- function(
  num_gfr = 5,
  num_burnin = 0,
  num_mcmc = 100,
  general_params = list(),
  mean_forest_params = list(),
  variance_forest_params = list(),
  random_effects_params = list()
) 
```

where the `params` lists all match the format of the original `bart()` function.

Similarly, in Python the API will be a new method of the `BARTModel` class:

```python
def continue_sampling(
    self,
    num_gfr: int = 5,
    num_burnin: int = 0,
    num_mcmc: int = 100,
    general_params: Optional[Dict[str, Any]] = None,
    mean_forest_params: Optional[Dict[str, Any]] = None,
    variance_forest_params: Optional[Dict[str, Any]] = None,
    random_effects_params: Optional[Dict[str, Any]] = None,
)
```

In general, the only `params` that can be changed within a model run are hyperparameters (i.e. the tree split prior parameters `alpha` and `beta`), and we will have runtime checks that disregard changes to the model itself (i.e. `num_trees`), along with a warning that a given parameter (e.g. `num_trees`) cannot be changed.

Also note that we allow `num_gfr` even though it is often used to "warm-start" an MCMC chain, there are many use cases for continuing to run a GFR sampler, either because XBART is the desired model or just to let the GFR run under different parameters before handing off to MCMC.

#### BCF API

The continued sampler API in R BCF will look like:

```r
continue_sampling.bcfmodel <- function(
  num_gfr = 5,
  num_burnin = 0,
  num_mcmc = 100,
  general_params = list(),
  prognostic_forest_params = list(),
  treatment_effect_forest_params = list(),
  variance_forest_params = list(),
  random_effects_params = list(),
) 
```

where the `params` lists all match the format of the original `bcf()` function.

Similarly, in Python the API will be a new method of the `BCFModel` class:

```python
def continue_sampling(
    self,
    num_gfr: int = 5,
    num_burnin: int = 0,
    num_mcmc: int = 100,
    general_params: Optional[Dict[str, Any]] = None,
    prognostic_forest_params: Optional[Dict[str, Any]] = None,
    treatment_effect_forest_params: Optional[Dict[str, Any]] = None,
    variance_forest_params: Optional[Dict[str, Any]] = None,
    random_effects_params: Optional[Dict[str, Any]] = None,
)
```

As with BART, the only `params` that can be changed within a model run are hyperparameters (i.e. the tree split prior parameters `alpha` and `beta`), and we will have runtime checks that disregard changes to the model itself (i.e. `num_trees`), along with a warning that a given parameter (e.g. `num_trees`) cannot be changed.

Also note that we allow `num_gfr` even though it is often used to "warm-start" an MCMC chain, there are many use cases for continuing to run a GFR sampler, either because XBCF is the desired model or just to let the GFR run under different parameters before handing off to MCMC.

#### Sample accumulation

Each run of `continue_sampling` appends to a model's existing `BARTSamples` / `BCFSamples` objects and updates the internal counts of `num_gfr`, `num_burnin`, and `num_mcmc`.

## Loading and continuing models from JSON

BART and BCF both support reloading a serialized model to predict or to continue sampling. Consistent with the [migration ladder](#migration-ladder) decision, **envelope serialization and deserialization stay in R and Python.** C++ contributes only the pieces that genuinely live there: the existing `ForestContainer` JSON serde and a resumable in-memory `BARTSamples` / `BCFSamples`. There is no C++-owned envelope reader/writer and no JSON-ingesting sampler constructor.

### Version and platform inference

The existing R and Python helpers that infer a model's version from its JSON are retained (and extended to also report `platform`); they do not move to C++.

```r
inferStochtreeVersion(json)     # R
inferStochtreePlatform(json)
```
```python
_infer_stochtree_version(json)  # Python
_infer_stochtree_platform(json)
```

### Writing the envelope (R / Python)

R and Python assemble the unified envelope themselves. The only forest-specific logic is a call to the existing C++ `ForestContainer` serializer, now keyed by self-describing names. So today's

```r
if (object$model_params$include_mean_forest) jsonobj$add_forest(object$mean_forests)
if (object$model_params$include_variance_forest) jsonobj$add_forest(object$variance_forests)
```

becomes the same calls writing under named keys (`mean_forest` / `variance_forest`), alongside the scalar metadata, the parameter samples read from the held `BARTSamples` / `BCFSamples`, and the language's **native** `covariate_preprocessor` sub-object plus its portability header. The native preprocessor is serialized by the language that owns it (see [Covariate Preprocessor](#covariate-preprocessor)); C++ never sees it.

### Reading the envelope and reconstructing samples (R / Python)

On read, R / Python:

1. run the [migration ladder](#migration-ladder) to bring the JSON up to `current`,
2. enforce the cross-platform gate (`platform` mismatch and `cross_platform_portable == false` => error),
3. load each forest via the existing `loadForestContainerJson`, now addressed by named key rather than position —

```r
# named keys remove the need to branch on which forests are present
mean_forests <- loadForestContainerJson(json_object, "mean_forest")          # if include_mean_forest
variance_forests <- loadForestContainerJson(json_object, "variance_forest")  # if include_variance_forest
```

4. rebuild the native preprocessor (same-platform) or a trivial identity preprocessor (cross-platform numeric, foreign body ignored), and
5. populate an in-memory `BARTSamples` / `BCFSamples` from the loaded forest containers and parameter arrays.

Step 5 is the one new piece of C++ surface: a way to **populate** a samples object from externally-supplied forest containers and parameter vectors (the "some copying / moving of pointers" cost), reusing the structured-input patterns from the R/Python binding design. The forest containers are already C++ objects; the samples object just takes ownership of them and the parameter arrays.

### Continuing from a reconstructed model

Because the sampler operates on an in-memory `BARTSamples` / `BCFSamples`, no JSON-aware sampler is needed and the constructor signature is unchanged:

```cpp
BARTSampler(BARTSamples& samples, BARTConfig& config, BARTData& data);
BCFSampler(BCFSamples& samples, BCFConfig& config, BCFData& data);
```

- **In-memory continuation** operates directly on the model's held samples object (and resumes its RNG stream — see [Continued sampling](#continued-sampling)).
- **From-disk continuation** (`previous_model_json`) is *read → populate samples → resume*: the JSON ingestion happens in R / Python (above), then the sampler runs on the reconstructed in-memory object. A freshly deserialized model carries no RNG state, so its continuation starts a new stream (consistent with the deserialized-continuation caveat noted earlier).

## Predicting from serialized models

**Status: considered but explicitly deprioritized for the initial 0.5.0 delivery** (see [Open Questions](#open-questions)). This is a performance convenience with no blocking design need, so the design below is recorded for completeness and may be implemented in a later release.

Right now, serialized models must be reloaded in R or Python before predicting on new data. This RFC will add an interface for obtaining predictions directly from JSON, avoiding some of the costly object reconstruction (though of course not the JSON parsing cost).

The interface for doing so will follow the current prediction interface, with an overload of the C++ prediction functions

```cpp
// BART
BARTPredictionResult predict_bart_model(BARTData& data, BARTPredictionInput& model_refs);
BARTPredictionResult predict_bart_model(BARTData& data, nlohmann::json& json);

// BCF
BCFPredictionResult predict_bcf_model(BCFData& data, BCFPredictionInput& model_refs);
BCFPredictionResult predict_bcf_model(BCFData& data, nlohmann::json& json);
```

And then R / Python wrappers that accept JSON arguments

```r
predict_bart_json <- function(
  bart_json,
  X,
  leaf_basis = NULL,
  rfx_group_ids = NULL,
  rfx_basis = NULL,
  type = "posterior",
  terms = "all",
  scale = "linear",
  ...
) {
  # Preprocess covariates R-side
  X_preprocessed <- preprocessCovariatesJson(bart_json, X)
  
  # Call cpp11 wrapper around C++ prediction function
  output <- bart_predict_json_cpp(
    json = bart_json,
    X = X_preprocessed,
    leaf_basis = leaf_basis,
    # ...
    obs_weights = NULL,
    rfx_group_ids = rfx_group_ids,
    rfx_basis = rfx_basis,
    posterior = type == "posterior",
    # ...
  )

  # Unpack and return outputs
  y_hat = reshape_cpp_pred_2d(output$y_hat, n, num_samples_output)
  # ...
  return() # ...
}
```

```python
def predict_json(
    X: Union[np.array, pd.DataFrame],
    leaf_basis: np.array = None,
    rfx_group_ids: np.array = None,
    rfx_basis: np.array = None,
    json: JsonCpp = None,
    type: str = "posterior",
    terms: Union[list[str], str] = "all",
    scale: str = "linear",
) -> Union[List[np.ndarray], np.ndarray]:
  # Preprocess covariates python-side
  X_preprocessed = preprocess_covariates_json(bart_json, X)

  # Call pybind11 wrapper around C++ prediction method
  output = bart_predict_json_cpp(
      json=json.json_ptr,
      X=np.asfortranarray(X_preprocessed),
      leaf_basis=np.asfortranarray(leaf_basis) if leaf_basis is not None else None,
      # ...
      rfx_group_ids=rfx_group_ids.astype(np.int32) if rfx_group_ids is not None else None,
      rfx_basis=np.asfortranarray(rfx_basis) if rfx_basis is not None else None,
      posterior=(type == "posterior"),
      # ...
  )

  # Unpack and return outputs
  y_hat_r = reshape_cpp_pred_2d(output["y_hat"])
  # ...
  return # ...
```

# Value

This RFC adds value in two (related) ways:

1. It avoids the costly roundtrip to JSON and back to continue sampling in-memory models
2. It unlocks cross-platform serialization for models fit on purely numeric covariates -- so a numeric model built in R can be loaded in Python for inference or continued sampling -- with non-portable (native-categorical) models detected and refused on cross-platform load rather than silently mis-transformed

# Risks and Drawbacks

This is a large lift which bundles several concerns: JSON parsing, version-tagging, external pointer persistence. 
The main risk of this RFC is timeline -- it requires some non-trivial C++ design and extensive testing. 

Scoping cross-platform portability to all-numeric models deliberately removes the riskiest sub-problem -- proving that two independently maintained native preprocessors encode categoricals identically -- at the cost of requiring users who want a portable categorical model to reduce it to a numeric matrix themselves before fitting. We mitigate the resulting UX cliff by making the cross-platform refusal loud and actionable (naming the offending columns) rather than silent.

# Alternatives

The first and most straightforward alternative is the status quo: all model reloading is mediated through JSON and all JSON inputs are processed on the R / Python side. There is no expectation of cross-platform JSON support and no interface for predicting from JSON objects.

Several "middle" ground approaches include:

1. Leave the JSON interface as-is but include a persistent pointer to a `BARTSamples` / `BCFSamples` object for continued sampling of in-memory models.
2. Support cross-platform serialization via R / Python logic while otherwise keeping the JSON / continued-sampling interface the same

# Open Questions

1. Predicting directly from JSON, unlike many of the other aspects of this RFC, does not have an obvious design need and is primarily a performance convenience. **Resolved: considered but explicitly deprioritized** -- deferred out of the initial 0.5.0 delivery and may be implemented separately in a later release. The design sketch is retained in [Predicting from serialized models](#predicting-from-serialized-models) for reference.
