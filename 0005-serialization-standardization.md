# Standardizing Serialization/Deserialization Formats for Models

Tracking issue: TBD

# Overview

Right now, `stochtree` makes it possible to continue sampling an existing model by passing a JSON representation of the old model to `bart()` / `bcf()` in R or `BARTModel.sample()` / `BCFModel.sample()` in Python. This is a rather "heavy" interface, as it requires a serialization / deserialization roundtrip and also prevents reloading an R model in Python or vice versa. This RFC streamlines the interface for continuing a sampler or predicting from a serialized model and makes it possible to do so cross-platform.

# Motivation

`stochtree`'s JSON-based interface for resurrecting old models supports two primary use cases:

1. Continuing to run an existing BART / BCF sampler, perhaps after inspecting traceplots and deciding to run for longer or with different hyperparameters
2. Resurrecting a BART / BCF model saved to disk or just JSON string and sampling it for longer

[RFC 4](https://github.com/StochasticTree/rfcs/blob/main/0004-cpp-dispatch.md) means the R and Python routines have a much larger shared C++ core and there is a documented API for passing structured inputs / outputs to / from C++. This allows us to simplify use case 1 above, so that a `BARTModel` or `BCFModel` object retains an in-memory pointer to a C++ `BARTSamples` / `BCFSamples` object, which can be appended to by continuing a sampler (with different hyperparameters). Similarly, with more of the sampling logic in C++, we can standardize model / parameter deserialization so that the C++ samplers ingest a(n optional) JSON string and initialize model state from the provided model.

Finally, this RFC will unlock two currently unsupported use cases:

1. Allow models built in R to be deserialized into Python and vice versa, and
2. Predict directly from a serialized model

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
| `"outcome_mean"` | Mean shift factor used for standardizing continuous outcomes (1 if no standardization) |
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

Forests are stored under the `"forests"` key. Forests are serialized to JSON all in C++ and the JSON is added to the model's JSON object under the `forests_{i}` key, where by convention `forest_0` is the mean forest, if one exists. If a variance forest exists, it is indexed at `forest_1` if a mean forest also exists, otherwise `forest_0`.

#### Random Effects

Random effects terms are organized under the `"random_effects"` key. There are three JSON objects that comprise a random effects term. Each is technically numbered, though BART and BCF only support a single additive random effects term out of the box.

1. Random effects container: stored under the `'random_effect_container_0'` key (0 is a convention to allow for multiple random effects terms in custom models)
2. Random effects "label mapper": stored under the `'random_effect_label_mapper_0'`
3. Unique group IDs are stored under the `'random_effect_groupids_0'` key

#### Reconciliation

We standardize on a new unified schema for both R and Python BART

| New Field | Previous R Field | Previous Python Field | Notes | Action |
| --- | --- | --- | --- | --- |
| `"stochtree_version"` | `"stochtree_version"` | `"stochtree_version"` | Version number that model was sampled under |  |
| `"platform"` |  |  | Whether model was sampled in R or Python | Add to R and Python |
| `"outcome_scale"` | `"outcome_scale"` | `"outcome_scale"` | Scale factor for standardizing continuous outcomes (1 if none) |  |
| `"outcome_mean"` | `"outcome_mean"` | `"outcome_mean"` | Mean shift factor for standardizing continuous outcomes (1 if none) |  |
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

Both covariate preprocessor schemas contain similar information: (1) which features need to be treated as ordered categorical (with categories possibly remapped to integers) or unordered categorical (one-hot encoded) and (2) which categories are observed in a feature.

Differences in structure largely reflect the design of the associated R and Python objects. We propose a "superset" schema that includes enough information to load either the R or Python preprocessor

| New Field | Previous R Field | Previous Python Field | Notes | Action |
| --- | --- | --- | --- | --- |
| `"num_original_features"` | | `"num_original_features"` | | Add to R |
| `"num_numeric_features"` | `"num_numeric_vars"` | | | Add to Python |
| `"num_ordinal_features"` | `"num_ordered_cat_vars"` | `"num_ordinal_features"` |  | |
| `"num_onehot_features"` | `"num_unordered_cat_vars"` | `"num_onehot_features"` |  | |
| `"ordinal_feature_index"` |  | `"ordinal_feature_index"` | R's `"ordered_cat_vars"` can be constructed from this array as well as the original feature names | Add to R | 
| `"onehot_feature_index"` |  | `"onehot_feature_index"` | R's `"unordered_cat_vars"` can be constructed from this array as well as the original feature names | Add to R | 
| `"processed_feature_types"` |  | `"processed_feature_types"` | | Add to R | 
| `"original_feature_types"` | `"feature_types"` | `"original_feature_types"` | | Harmonize schema between R and Python |
| `"original_feature_indices"` | `"original_var_indices"` | `"original_feature_indices"` | | |
| `"ordinal_dtype_list"` |  | `"ordinal_dtype_list"` | All categories stored as strings in R | Add to R, all string |
| `"ordinal_categories_list"` |  | `"ordinal_categories_list"` | R's `"ordered_unique_levels"` can be constructed from this list | Add to R |
| `"onehot_dtype_list"` | | `"onehot_dtype_list"` | All categories stored as strings in R | Add to R, all string |
| `"onehot_categories_list"` |  | `"onehot_categories_list"` | R's `"unordered_unique_levels"` can be constructed from this list | Add to R |
| `"original_feature_names"` | | | Not technically stored in either JSON at the moment, but needed for reconstructing R's preprocessor from the elements above | Add to Python and R |

R's `"numeric_vars"` list will be reconstructed as the set difference of the covariate's names and the reconstructed `"ordered_cat_vars"` and `"unordered_cat_vars"` lists

**Original feature types schema**

Python's `"original_feature_types"` encodes features in one of six labels: `"category"`, `"string"`, `"boolean"`, `"integer"`, `"float"`, and `"unsupported"`. Both `"category"` and `"string"` can be either ordered or unordered categorical (this information is encoded by `"ordinal_feature_index"` and `"onehot_feature_index"`). `"boolean"`, `"integer"`, and `"float"` are all converted to numeric features.

R's `"feature_types"` encodes features as 0 for numeric, 1 for ordered categorical and 2 for unordered categorical.

Our schema uses the Python approach, where the data type of an R column determines which of the labels a feature gets. In R, the mapping will be:

* `"numeric"` -> `"float"`
* `"integer"` -> `"integer"`
* `"logical"` -> `"boolean"`
* `"character"` -> `"string"`
* `"factor"` -> `"category"`

**Original feature name schema**

R matrices and numpy arrays do not have "column names," so we handle the "null case" consistently on both platforms: columns are labeled as `"x1"`, `"x2"`, ..., `"xp"`.

**Processed feature type list in R**

We add the `"processed_feature_types"` to R by the following procedure:

* If a column is numeric (i.e. passed through without transformation), the index of its corresponding column in the transformed data matrix is marked with a 0
* If a column is ordinal / categorical (i.e. transformed to integer or one-hot encoded), the indices of its corresponding columns in the transformed data matrix are marked with a 1

##### Forests

[*Unchanged*]

##### Random Effects

[*Unchanged*]

### BCF JSON Schema

The BCF JSON object follows the schema below

| Field | Value |
| --- | --- |
| `"stochtree_version"` | Version number that model was sampled under |
| `"outcome_scale"` | Scale factor used for standardizing continuous outcomes (1 if no standardization) |
| `"outcome_mean"` | Mean shift factor used for standardizing continuous outcomes (1 if no standardization) |
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

Forests are stored under the `"forests"` key. Forests are serialized to JSON all in C++ and the JSON is added to the model's JSON object under the `forests_{i}` key, where by convention `forest_0` is the prognostic forest and `"forest_1"` is the treatment effect forest. If a variance forest exists, it is indexed at `forest_2`.

#### Random Effects

The components of a random effects term are stored under the `"random_effects"` key and contain three separate entities:

1. Random effects container: stored under the `'random_effect_container_0'` key (0 is a convention to allow for multiple random effects terms in custom models)
2. Random effects "label mapper": stored under the `'random_effect_label_mapper_0'`
3. Unique group IDs are stored under the `'random_effect_groupids_0'` key

#### Reconciliation

We standardize on a new unified schema for both R and Python BCF

| New Field | Previous R Field | Previous Python Field | Notes | Action |
| --- | --- | --- | --- | --- |
| `"stochtree_version"` | `"stochtree_version"` | `"stochtree_version"` | Version number that model was sampled under |  |
| `"platform"` |  |  | Whether model was sampled in R or Python | Add to R and Python |
| `"outcome_scale"` | `"outcome_scale"` | `"outcome_scale"` | Scale factor for standardizing continuous outcomes (1 if none) |  |
| `"outcome_mean"` | `"outcome_mean"` | `"outcome_mean"` | Mean shift factor for standardizing continuous outcomes (1 if none) |  |
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

[*Unchanged*]

##### Random Effects

[*Unchanged*]

## Explicit handling of older JSON formats

Current serialization / deserialization routines are designed to (attempt to) gracefully handle JSON objects from earlier stochtree versions **within** a single platform. On the R side, version inference proceeds largely by checking for the presence of fields:

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

## Persisting pointers to C++ model samples objects in BART and BCF model classes

Right now, the C++ API for BART and BCF models creates several objects in-memory, namely:

1. A `BARTSamples` / `BCFSamples` object which wraps containers for parameter, forest, and random effect samples
2. A `BARTSampler` / `BCFSampler` object which initializes model state and runs the sampler, writing out to `BARTSamples` / `BCFSamples`

The sampler object is destroyed when it goes out of scope (i.e. after the call to C++ concludes) and the samples object has each of its components unpacked (copied) into R and Python containers or moved to managed external pointers and then is also destroyed when it goes out of scope.

This RFC proposes that the BART and BCF objects in R and Python maintain persistent pointers to `BARTSamples` / `BCFSamples` C++ objects, unpacking them only when they are queried by the `extract_parameter` method.

### Migration plan

Redesigning the BART and BCF objects to store results in a C++ samples object, rather than internal fields like `global_var_samples` would break any code that directly references these internal fields (as opposed to code that queries them via `extract_parameter`), so we need a plan for graceful deprecation.

We propose to do this via a warning that runs when users try to directly access deprecated fields from a `BARTModel` or `BCFModel` object. In Python, we'd do this with a custom `__getattr__`

```python
_MOVED_ATTRS = {
  "global_var_samples": "Access via `model.extract_parameter('sigma2_global')`.",
  "leaf_scale_samples": "Access via `model.extract_parameter('sigma2_leaf')`.",
}

class BARTModel:
    def __getattr__(self, name):
        if name in _MOVED_ATTRS:
            warnings.warn(
                f"'{name}' is no longer a direct attribute of BARTModel as of version 0.5.0. "
                f"{_MOVED_ATTRS[name]}",
                DeprecationWarning,
                stacklevel=2,
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

In R with an S3 object, we'd do this by overloading the `$` access operator

```r
`$.bartmodel` <- function(x, name) {
  moved <- list(
    sigma2_global_samples = "Use extract_parameter(model, 'sigma2_global').",
    sigma2_leaf_samples = "Use extract_parameter(model, 'sigma2_leaf')."
  )
  if (name %in% names(moved)) {
    warning(sprintf("'%s' is no longer a direct field of the bartmodel object as of version 0.5.0. %s", name, moved[[name]]), call. = FALSE)
    return(NULL)
  }
  .subset2(x, name)
}
```

### Continued sampling

The proposed redesign keeps both `BARTSampler` / `BCFSampler` and `BARTSamples` / `BCFSamples` C++ classes alive, so that continued sampling is as straightforward as dispatching the C++ sampler for more iterations. The simplest way to make this happen is a new `continue_sampling` method in both R and Python that appends to the `BARTSamples` / `BCFSamples` objects.

This method should allow users to pass through updated model hyperparameters in the same format as the original BART / BCF sampler dispatch.

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

## C++ sampler support for JSON inputs

BART and BCF both support sampling models reloaded from JSON in R and Python, so a full C++ overhaul of the interface must include the same logic. We propose to do this in several steps:

### Version and platform inference in C++

We have R and Python helpers to infer the version number of a stochtree model based on its JSON representaiton and we need to extend this to C++ and add platform inference. The proposed API:

```cpp
std::string infer_stochtree_version(std::string& json_string)
std::string infer_stochtree_platform(std::string& json_string)
```

### End-to-end JSON serialization and deserialization in C++

This RFC necessitates several C++-specific serialization changes:

1. Routines that can unpack a JSON file into model state in a `BARTSampler` / `BCFSampler` class, to support the `previous_model_json` interface in BART and BCF
2. Routines that write model components from a `BARTSampler` / `BCFSampler` class to a JSON object in the schema outlined above
3. Routines that read a JSON object in the schema outlined above and construct write an appropriate `BARTSampler` / `BCFSampler` class

#### JSON to sampler deserialization

Right now, we initialize `BARTSampler` / `BCFSampler` classes with a single constructor that calls an `InitializeState` method. We propose to overload this API to allow for initialization from JSON

```cpp
class BARTSampler {
 public:
  BARTSampler(BARTSamples& samples, BARTConfig& config, BARTData& data);
  BARTSampler(nlohmann::json& json, BARTSamples& samples, BARTConfig& config, BARTData& data);
 private:
  void InitializeState(BARTSamples& samples);
  void InitializeState(nlohmann::json& json, BARTSamples& samples);
}

class BCFSampler {
 public:
  BCFSampler(BCFSamples& samples, BCFConfig& config, BCFData& data);
  BCFSampler(nlohmann::json& json, BCFSamples& samples, BCFConfig& config, BCFData& data);
 private:
  void InitializeState(BCFSamples& samples);
  void InitializeState(nlohmann::json& json, BCFSamples& samples);
}
```

#### Writing to JSON from `BARTSamples` / `BCFSamples`

Here, the primary change is that we replace serialization logic in R / Python like

```r
if (object$model_params$include_mean_forest) {
  jsonobj$add_forest(object$mean_forests)
}
if (object$model_params$include_variance_forest) {
  jsonobj$add_forest(object$variance_forests)
}
```

With a single C++ function that writes to a JSON reference based on a `BARTSamples` / `BCFSamples` object

```cpp
static inline void writeJSON(nlohmann::json& json, BARTSamples& samples) {
  int forest_num = 0;
  if (samples.mean_forests != nullptr) {
    std::string forest_label = "forest_" + std::to_string(forest_num);
    nlohmann::json forest_json = samples.mean_forests->to_json();
    json.at("forests").emplace(forest_label, forest_json);
    forest_num++;
  }
  // ...
}
```

Both the R and Python interfaces will create this JSON object and then call a wrapper around this C++ serialization function

#### Creating `BARTSamples` / `BCFSamples` from JSON

As above, we replace R / Python deserialization logic like

```r
if (include_mean_forest) {
  output[["mean_forests"]] <- loadForestContainerJson(
    json_object,
    "forest_0"
  )
  if (include_variance_forest) {
    output[["variance_forests"]] <- loadForestContainerJson(
      json_object,
      "forest_1"
    )
  }
} else {
  output[["variance_forests"]] <- loadForestContainerJson(
    json_object,
    "forest_0"
  )
}
```

With a single C++ function that writes to `BARTSamples` / `BCFSamples` object based on JSON

```cpp
static inline void readJSON(nlohmann::json& json, BARTSamples& samples) {
  bool include_mean_forest = json.at("include_mean_forest");
  bool include_variance_forest = json.at("include_variance_forest");
  if (include_mean_forest) {
    nlohmann::json forest_json = json.at("forests").at("forest_0");
    samples.mean_forests->Reset();
    samples.mean_forests->from_json(forest_json);
    if (include_variance_forest) {
      nlohmann::json forest_json = json.at("forests").at("forest_1");
      samples.variance_forests->Reset();
      samples.variance_forests->from_json(forest_json);
    }
  } else {
    nlohmann::json forest_json = json.at("forests").at("forest_0");
    samples.variance_forests->Reset();
    samples.variance_forests->from_json(forest_json);
  }
  // ...
}
```

Both the R and Python interfaces will create this JSON object and then call a wrapper around this C++ serialization function

## Predicting from serialized models

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
2. It unlocks cross-platform serialization -- models built in R can be loaded in Python for inference or continued sampling

# Risks and Drawbacks

This is a large lift which bundles several concerns: JSON parsing, version-tagging, external pointer persistence. 
The main risk of this RFC is timeline -- it requires some non-trivial C++ design and extensive testing. 

# Alternatives

The first and most straightforward alternative is the status quo: all model reloading is mediated through JSON and all JSON inputs are processed on the R / Python side. There is no expectation of cross-platform JSON support and no interface for predicting from JSON objects.

Several "middle" ground approaches include:

1. Leave the JSON interface as-is but include a persistent pointer to a `BARTSamples` / `BCFSamples` object for continued sampling of in-memory models.
2. Support cross-platform serialization via R / Python logic while otherwise keeping the JSON / continued-sampling interface the same

# Open Questions

1. Predicting directly from JSON, unlike many of the other aspects of this RFC, does not have an obvious design need and is primarily a performance convenience. Should it be deferred and implemented separately?
